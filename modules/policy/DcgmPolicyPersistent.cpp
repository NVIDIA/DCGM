/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * Expected YAML File Format:
 *
 * The on-disk schema consists of two top-level keys:
 *   - schema_version: integer written by SerializeToYaml (currently 1).
 *     LoadPolicies rejects files whose schema_version exceeds CURRENT_SCHEMA_VERSION
 *     and logs a migration notice for older versions.
 *   - policies: sequence of policy entries (see below).
 *
 * schema_version: 1                      # Written by SerializeToYaml; must appear before policies
 * policies:
 *   - fieldId: 150                           # DCGM field ID (DCGM_FI_DEV_GPU_TEMP)
 *     name: "GPU Temperature Warning"        # Human-readable policy name
 *     operator: "GREATER_THAN"               # Comparison operator
 *     threshold: 85.0                        # Threshold value
 *     entities:                              # Empty = all GPUs for now
 *       - { entityGroupId: 1, entityId: 0 }
 *     channels: ["CONSOLE", "FILE"]          # Notification channels
 *     status: "ENABLED"                      # ENABLED or DISABLED
 *     rateLimitSeconds: 60                   # Minimum seconds between notifications
 *
 *   - fieldId: 155                           # DCGM_FI_DEV_POWER_USAGE
 *     name: "Power Limit Exceeded"
 *     operator: "GREATER_THAN"
 *     threshold: 300.0
 *     entities: []                           # Empty = all GPUs for now
 *     channels: ["CONSOLE", "CALLBACK"]
 *     status: "ENABLED"
 *     rateLimitSeconds: 30
 *
 * Supported Operators:
 *   EQUAL, NOT_EQUAL, GREATER_THAN, LESS_THAN, GREATER_THAN_OR_EQUAL, LESS_THAN_OR_EQUAL, CHANGED
 *
 * Supported Channels:
 *   CONSOLE (dcgmi policy watch), FILE (JSON logs), CALLBACK (programmatic notifications)
 */

#include "DcgmPolicyPersistent.h"
#include "DcgmPolicyRegistry.h"
#include "DcgmPolicyUtils.h"

#include <DcgmStringHelpers.h>
#include <Defer.hpp>

#include <DcgmLogging.h>

#include <dcgm_fields.h>
#include <fmt/format.h>
#include <yaml-cpp/yaml.h>

#include <array>
#include <cerrno>
#include <cstdlib>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <string_view>
#include <sys/stat.h>
#include <unistd.h>

// Default config path following Linux FHS (package name: datacenter-gpu-manager-4)
static constexpr std::string_view DEFAULT_CONFIG_PATH = "/etc/datacenter-gpu-manager-4/policies.yaml";

// Allowed directories for policy configuration files
static constexpr std::array<std::string_view, 2> ALLOWED_DIRECTORIES
    = { "/etc/datacenter-gpu-manager-4/", "/var/log/nvidia-dcgm/" };

// Current schema version for YAML files (for upgrade/downgrade support)
static constexpr int CURRENT_SCHEMA_VERSION = 1;

// File permissions for policy files (owner read/write only)
static constexpr mode_t FILE_PERMISSIONS = 0600;

// Environment variable for the policy configuration file path
static constexpr std::string_view CONFIG_PATH_ENV = "DCGM_POLICY_CONFIG_PATH";

// Admin/test override: colon-separated extra allowed directories, on top of the built-in ones.
static constexpr std::string_view ALLOWED_DIRS_ENV = "DCGM_POLICY_ALLOWED_DIRS";

// Maximum size of the policy configuration file (10 MB)
static constexpr std::size_t MAX_CONFIG_FILE_SIZE = 10 * 1024 * 1024; // 10 MB

// YAML key names — shared between ParseYamlFile (read) and SerializeToYaml (write)
static constexpr std::string_view YAML_KEY_SCHEMA_VERSION = "schema_version";
static constexpr std::string_view YAML_KEY_POLICIES       = "policies";
static constexpr std::string_view YAML_KEY_FIELD_ID       = "fieldId";
static constexpr std::string_view YAML_KEY_NAME           = "name";
static constexpr std::string_view YAML_KEY_OPERATOR       = "operator";
static constexpr std::string_view YAML_KEY_THRESHOLD      = "threshold";
static constexpr std::string_view YAML_KEY_ENTITIES       = "entities";
static constexpr std::string_view YAML_KEY_ENTITY_GROUP   = "entityGroupId";
static constexpr std::string_view YAML_KEY_ENTITY_ID      = "entityId";
static constexpr std::string_view YAML_KEY_CHANNELS       = "channels";
static constexpr std::string_view YAML_KEY_STATUS         = "status";
static constexpr std::string_view YAML_KEY_RATE_LIMIT     = "rateLimitSeconds";

/*****************************************************************************/
DcgmPolicyPersistence::DcgmPolicyPersistence(DcgmPolicyRegistry &registry, std::filesystem::path configFilePath)
    : m_registry(&registry)
    , m_configFilePath(std::move(configFilePath))
{}

/*****************************************************************************/
DcgmPolicyPersistence::~DcgmPolicyPersistence() = default;

/*****************************************************************************/
dcgmReturn_t DcgmPolicyPersistence::ValidatePath(std::filesystem::path const &path) const
{
    std::error_code ec;

    // Convert to canonical path (resolves symlinks and normalizes)
    std::filesystem::path canonicalPath = std::filesystem::weakly_canonical(path, ec);
    if (ec)
    {
        log_error("Failed to resolve canonical path: {}", path.native());
        return DCGM_ST_BADPARAM;
    }

    // native() returns const std::string& on Linux — no allocation
    std::string const &absStr = canonicalPath.native();

    // Require path to be under allowed directories
    bool isAllowed = false;
    for (auto const &allowedDir : ALLOWED_DIRECTORIES)
    {
        if (absStr.starts_with(allowedDir))
        {
            isAllowed = true;
            break;
        }
    }

    // Also accept directories from the DCGM_POLICY_ALLOWED_DIRS override (canonicalized prefixes).
    if (!isAllowed)
    {
        char const *allowedDirsEnv = getenv(ALLOWED_DIRS_ENV.data());
        if (allowedDirsEnv != nullptr && allowedDirsEnv[0] != '\0')
        {
            for (std::string_view const entry : DcgmNs::Split(allowedDirsEnv, ':'))
            {
                if (!entry.empty() && DcgmPolicyUtils::IsPathWithinDirectory(absStr, entry))
                {
                    log_warning(
                        "Allowing path under override directory from {} ({}): {}", ALLOWED_DIRS_ENV, entry, absStr);
                    isAllowed = true;
                    break;
                }
            }
        }
    }

    if (!isAllowed)
    {
        log_error("Path outside allowed directories: {}", absStr);
        return DCGM_ST_BADPARAM;
    }

    return DCGM_ST_OK;
}

std::filesystem::path DcgmPolicyPersistence::GetConfigPath() const
{
    if (!m_configFilePath.empty())
    {
        return m_configFilePath;
    }

    char const *envPath = getenv(CONFIG_PATH_ENV.data());
    if (envPath != nullptr && envPath[0] != '\0')
    {
        return std::filesystem::path(envPath);
    }

    return std::filesystem::path(DEFAULT_CONFIG_PATH);
}

/*****************************************************************************/
dcgmReturn_t DcgmPolicyPersistence::SetConfigPath(std::filesystem::path const &path)
{
    // Canonicalize to resolve any relative components or symlinks
    std::error_code ec;
    std::filesystem::path canonical = std::filesystem::weakly_canonical(path, ec);
    if (ec)
    {
        log_error("Failed to canonicalize config path {}: {}", path.native(), ec.message());
        return DCGM_ST_BADPARAM;
    }

    // Reject paths outside allowed directories
    dcgmReturn_t ret = ValidatePath(canonical);
    if (ret != DCGM_ST_OK)
    {
        log_error("Config path rejected by validation: {}", canonical.native());
        return ret;
    }

    // Store after validation; persistence is owned by serialized module command handling.
    log_debug("Policy config path set to: {}", canonical.native());
    m_configFilePath = std::move(canonical);
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmPolicyPersistence::LoadPolicies()
{
    // Resolve and validate the target path
    std::filesystem::path filePath = GetConfigPath();
    dcgmReturn_t ret               = ValidatePath(filePath);
    if (ret != DCGM_ST_OK)
    {
        log_error("Invalid config path: {}", filePath.native());
        return ret;
    }

    log_debug("Loading policies from: {}", filePath.native());

    // Parse YAML into a local vector first — no registry interaction until we have a complete set
    std::vector<DcgmPolicy> parsed;
    ret = ParseYamlFile(filePath, parsed);
    if (ret != DCGM_ST_OK)
    {
        log_error("Failed to parse YAML file: {}", filePath.native());
        return ret;
    }

    // Replace registry contents only after the parsed set has passed batch validation.
    ret = m_registry->ReplaceAllPolicies(parsed.data(), parsed.data() + parsed.size());
    if (ret != DCGM_ST_OK)
    {
        log_error("Failed to replace registry with parsed policies");
        return ret;
    }

    log_debug("Successfully loaded policies from: {}", filePath.native());
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmPolicyPersistence::SavePolicies()
{
    // Resolve and validate the target path
    std::filesystem::path filePath = GetConfigPath();
    dcgmReturn_t ret               = ValidatePath(filePath);
    if (ret != DCGM_ST_OK)
    {
        log_error("Invalid config path: {}", filePath.native());
        return ret;
    }

    // Collect all registered policies to serialize
    std::vector<DcgmPolicy> policies;
    (void)m_registry->GetAllPolicies(std::back_inserter(policies));

    log_debug("Saving {} policies to: {}", policies.size(), filePath.native());

    // Create parent directory if it doesn't exist
    ret = EnsureDirectoryExists(filePath);
    if (ret != DCGM_ST_OK)
    {
        log_error("Failed to create directory for: {}", filePath.native());
        return ret;
    }

    // Serialize policies to YAML string
    std::string yamlContent;
    ret = SerializeToYaml(policies.data(), policies.data() + policies.size(), yamlContent);
    if (ret != DCGM_ST_OK)
    {
        log_error("Failed to serialize policies to YAML");
        return ret;
    }

    // Atomically write via temp file + rename to avoid partial writes
    ret = AtomicWriteFile(filePath, yamlContent);
    if (ret != DCGM_ST_OK)
    {
        log_error("Failed to write policies to file: {}", filePath.native());
        return ret;
    }

    log_debug("Saved {} policies to: {}", policies.size(), filePath.native());
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmPolicyPersistence::ParsePolicyNode(YAML::Node const &node, DcgmPolicy &policy)
{
    int fieldId      = node[YAML_KEY_FIELD_ID.data()].as<int>();
    std::string name = node[YAML_KEY_NAME.data()].as<std::string>();
    double threshold = node[YAML_KEY_THRESHOLD.data()].as<double>();

    if (fieldId < 0 || fieldId > std::numeric_limits<unsigned short>::max())
    {
        log_error("Skipping policy '{}': fieldId {} is outside the valid range", name, fieldId);
        return DCGM_ST_BADPARAM;
    }

    // Validate and map the operator string to enum
    auto opOpt = DcgmPolicyUtils::StringToOperator(node[YAML_KEY_OPERATOR.data()].as<std::string>());
    if (!opOpt.has_value())
    {
        log_error(
            "Skipping policy '{}': unknown operator '{}'", name, node[YAML_KEY_OPERATOR.data()].as<std::string>());
        return DCGM_ST_BADPARAM;
    }

    std::vector<dcgmGroupEntityPair_t> entities;
    YAML::Node const entitiesNode = node[YAML_KEY_ENTITIES.data()];
    if (entitiesNode.IsDefined() && !entitiesNode.IsSequence())
    {
        log_error("Skipping policy '{}': entities must be a sequence", name);
        return DCGM_ST_BADPARAM;
    }

    if (entitiesNode.IsDefined())
    {
        for (auto const &entityNode : entitiesNode)
        {
            auto const entityGroupId = entityNode[YAML_KEY_ENTITY_GROUP.data()].as<unsigned int>();
            auto const entityId      = entityNode[YAML_KEY_ENTITY_ID.data()].as<unsigned int>();
            if (entityGroupId != DCGM_FE_GPU)
            {
                log_error("Unsupported policy entity group {}", entityGroupId);
                return DCGM_ST_BADPARAM;
            }
            entities.push_back({ .entityGroupId = static_cast<dcgm_field_entity_group_t>(entityGroupId),
                                 .entityId      = static_cast<dcgm_field_eid_t>(entityId) });
        }
    }

    // Validate and map channel strings; collapse into a bitmask
    unsigned int channelMask = 0;
    for (auto const &ch : node[YAML_KEY_CHANNELS.data()])
    {
        auto chOpt = DcgmPolicyUtils::StringToChannel(ch.as<std::string>());
        if (!chOpt.has_value())
        {
            log_error("Skipping policy '{}': unknown channel '{}'", name, ch.as<std::string>());
            return DCGM_ST_BADPARAM;
        }
        channelMask |= static_cast<unsigned int>(*chOpt);
    }

    if (channelMask == 0)
    {
        log_error("Skipping policy '{}': at least one notification channel is required", name);
        return DCGM_ST_BADPARAM;
    }

    dcgmPolicyState_t state = (node[YAML_KEY_STATUS.data()].as<std::string>() == "ENABLED")
                                  ? DCGM_POLICY_STATE_ENABLED
                                  : DCGM_POLICY_STATE_DISABLED;
    // Default to 60s when rateLimitSeconds is omitted so older/hand-crafted configs keep loading.
    long long parsedRateLimit = 0;
    try
    {
        parsedRateLimit = node[YAML_KEY_RATE_LIMIT.data()].as<long long>(60LL);
    }
    catch (YAML::Exception const &e)
    {
        log_error("Skipping policy '{}': invalid rate limit: {}", name, e.what());
        return DCGM_ST_BADPARAM;
    }
    if (parsedRateLimit < 0
        || static_cast<unsigned long long>(parsedRateLimit) > std::numeric_limits<unsigned int>::max())
    {
        log_error("Skipping policy '{}': rate limit must fit in an unsigned int", name);
        return DCGM_ST_BADPARAM;
    }
    auto const rateLimit = static_cast<unsigned int>(parsedRateLimit);

    // Persisted YAML carries no live observed value — seed value to 0.0; the cache manager
    // updates it on the first sample.
    policy = DcgmPolicy(
        static_cast<unsigned short>(fieldId), name, 0.0, threshold, *opOpt, entities, channelMask, state, rateLimit);
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmPolicyPersistence::ParseYamlFile(std::filesystem::path const &filePath,
                                                  std::vector<DcgmPolicy> &policies)
{
    std::error_code ec;

    // Guard against missing files, symlinks, and oversized inputs before parsing
    if (!std::filesystem::exists(filePath, ec))
    {
        log_debug("Config file does not exist: {}", filePath.native());
        return DCGM_ST_FILE_IO_ERROR;
    }
    if (std::filesystem::is_symlink(filePath, ec))
    {
        log_error("Config path must not be a symlink: {}", filePath.native());
        return DCGM_ST_FILE_IO_ERROR;
    }
    auto fileSize = std::filesystem::file_size(filePath, ec);
    if (ec)
    {
        log_error("Failed to read config file size {}: {}", filePath.native(), ec.message());
        return DCGM_ST_FILE_IO_ERROR;
    }
    if (fileSize > MAX_CONFIG_FILE_SIZE)
    {
        log_error("Config file too large ({} bytes): {}", fileSize, filePath.native());
        return DCGM_ST_BADPARAM;
    }

    try
    {
        YAML::Node root = YAML::LoadFile(filePath.native());

        // Check schema version for compatibility
        int schemaVersion = root[YAML_KEY_SCHEMA_VERSION.data()].as<int>(1);
        if (schemaVersion > CURRENT_SCHEMA_VERSION)
        {
            log_error("Config file schema version {} is newer than supported version {}. "
                      "Please upgrade DCGM to read this file.",
                      schemaVersion,
                      CURRENT_SCHEMA_VERSION);
            return DCGM_ST_BADPARAM;
        }
        if (schemaVersion < CURRENT_SCHEMA_VERSION)
        {
            log_info("Migrating config file from schema version {} to {}", schemaVersion, CURRENT_SCHEMA_VERSION);
            // Future: Add migration logic here when schema changes
        }

        if (!root[YAML_KEY_POLICIES.data()] || !root[YAML_KEY_POLICIES.data()].IsSequence())
        {
            log_debug("No policies found in file");
            return DCGM_ST_OK;
        }

        // Parse each policy node; skip invalid entries and track counts for summary
        int parsedCount  = 0;
        int skippedCount = 0;
        for (auto const &node : root[YAML_KEY_POLICIES.data()])
        {
            DcgmPolicy policy;
            dcgmReturn_t parseRet = DCGM_ST_BADPARAM;
            try
            {
                parseRet = ParsePolicyNode(node, policy);
            }
            catch (YAML::Exception const &e)
            {
                log_error("Skipping malformed policy entry: {}", e.what());
            }
            catch (std::exception const &e)
            {
                log_error("Skipping malformed policy entry: {}", e.what());
            }

            if (parseRet == DCGM_ST_OK)
            {
                policies.push_back(std::move(policy));
                parsedCount++;
            }
            else
            {
                skippedCount++;
            }
        }

        if (skippedCount > 0)
        {
            log_warning("Policy parse summary: {} parsed, {} skipped", parsedCount, skippedCount);
        }
        else
        {
            log_debug("Policy parse summary: {} parsed, {} skipped", parsedCount, skippedCount);
        }

        // Treat all-skipped as an error — a file where every entry is invalid should not
        // silently wipe the registry via RemoveAllPolicies + AddPolicies({}).
        if (parsedCount == 0 && skippedCount > 0)
        {
            log_error("All {} policies in config file were invalid — refusing to replace registry", skippedCount);
            return DCGM_ST_BADPARAM;
        }

        return DCGM_ST_OK;
    }
    catch (YAML::Exception const &e)
    {
        log_error("YAML parse error: {}", e.what());
        return DCGM_ST_BADPARAM;
    }
    catch (std::exception const &e)
    {
        log_error("Parse error: {}", e.what());
        return DCGM_ST_BADPARAM;
    }
}

/*****************************************************************************/
dcgmReturn_t DcgmPolicyPersistence::SerializeToYaml(DcgmPolicy const *first,
                                                    DcgmPolicy const *last,
                                                    std::string &yamlContent)
{
    try
    {
        YAML::Emitter out;

        // Write top-level map with schema version and policies sequence
        out << YAML::BeginMap;
        out << YAML::Key << YAML_KEY_SCHEMA_VERSION.data() << YAML::Value << CURRENT_SCHEMA_VERSION;

        out << YAML::Key << YAML_KEY_POLICIES.data();
        out << YAML::Value << YAML::BeginSeq;

        // Serialize each policy as a YAML map entry
        for (DcgmPolicy const *it = first; it != last; ++it)
        {
            DcgmPolicy const &policy = *it;

            out << YAML::BeginMap;
            out << YAML::Key << YAML_KEY_FIELD_ID.data() << YAML::Value << policy.GetFieldId();
            out << YAML::Key << YAML_KEY_NAME.data() << YAML::Value << policy.GetName();
            out << YAML::Key << YAML_KEY_OPERATOR.data() << YAML::Value
                << DcgmPolicyUtils::OperatorToString(policy.GetPolicyOperator()).value_or("unknown").data();
            out << YAML::Key << YAML_KEY_THRESHOLD.data() << YAML::Value << policy.GetThreshold();

            out << YAML::Key << YAML_KEY_ENTITIES.data() << YAML::Value << YAML::Flow << YAML::BeginSeq;
            for (auto const &entity : policy.GetEntities())
            {
                out << YAML::BeginMap;
                out << YAML::Key << YAML_KEY_ENTITY_GROUP.data() << YAML::Value
                    << static_cast<unsigned int>(entity.entityGroupId);
                out << YAML::Key << YAML_KEY_ENTITY_ID.data() << YAML::Value
                    << static_cast<unsigned int>(entity.entityId);
                out << YAML::EndMap;
            }
            out << YAML::EndSeq;

            out << YAML::Key << YAML_KEY_CHANNELS.data() << YAML::Value << YAML::Flow << YAML::BeginSeq;
            for (auto const ch :
                 { DCGM_POLICY_CHANNEL_CONSOLE, DCGM_POLICY_CHANNEL_FILE, DCGM_POLICY_CHANNEL_CALLBACK })
            {
                if (policy.GetChannelMask() & static_cast<unsigned int>(ch))
                {
                    out << DcgmPolicyUtils::ChannelToString(ch).value_or("unknown").data();
                }
            }
            out << YAML::EndSeq;

            out << YAML::Key << YAML_KEY_STATUS.data() << YAML::Value << (policy.IsEnabled() ? "ENABLED" : "DISABLED");
            out << YAML::Key << YAML_KEY_RATE_LIMIT.data() << YAML::Value << policy.GetRateLimit().count();
            out << YAML::EndMap;
        }

        out << YAML::EndSeq;
        out << YAML::EndMap;

        if (!out.good())
        {
            log_error("YAML serialization error: {}", out.GetLastError());
            return DCGM_ST_GENERIC_ERROR;
        }

        std::string result = out.c_str();
        if (result.size() > MAX_CONFIG_FILE_SIZE)
        {
            log_error(
                "Serialized YAML exceeds maximum allowed size ({} > {} bytes)", result.size(), MAX_CONFIG_FILE_SIZE);
            return DCGM_ST_BADPARAM;
        }

        yamlContent = std::move(result);
        return DCGM_ST_OK;
    }
    catch (YAML::Exception const &e)
    {
        log_error("YAML serialize error: {}", e.what());
        return DCGM_ST_GENERIC_ERROR;
    }
}

/*****************************************************************************/
dcgmReturn_t DcgmPolicyPersistence::AtomicWriteFile(std::filesystem::path const &filePath, std::string_view content)
{
    std::error_code ec;

    // Write content to a sibling .tmp file first
    std::filesystem::path tempPath = filePath;
    tempPath += ".tmp";

    // O_NOFOLLOW rejects a symlink planted at tempPath (open fails with ELOOP); O_TRUNC replaces a
    // stale regular temp file. This avoids a TOCTOU pre-check while keeping symlink swaps out.
    int fd = open(tempPath.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_NOFOLLOW, FILE_PERMISSIONS);
    if (fd < 0)
    {
        log_error("Failed to create temp file: {} (errno: {})", tempPath.native(), errno);
        return DCGM_ST_FILE_IO_ERROR;
    }

    // RAII guards: close the fd and unlink the temp file on any early return.
    DcgmNs::Defer closeFd([&] {
        if (fd >= 0)
        {
            close(fd);
        }
    });

    DcgmNs::Defer removeTemp([&] {
        std::error_code rmEc;
        std::filesystem::remove(tempPath, rmEc);
    });

    // Write the content to the temp file.
    char const *data      = content.data();
    std::size_t remaining = content.size();
    while (remaining > 0)
    {
        ssize_t const written = write(fd, data, remaining);
        if (written < 0)
        {
            if (errno == EINTR)
            {
                continue;
            }
            log_error("Failed to write temp file: {} (errno: {})", tempPath.native(), errno);
            return DCGM_ST_FILE_IO_ERROR;
        }
        if (written == 0)
        {
            // write() wrote nothing and made no progress; bail out instead of looping forever.
            log_error("Failed to write temp file: {} (no bytes written)", tempPath.native());
            return DCGM_ST_FILE_IO_ERROR;
        }
        data += written;
        remaining -= static_cast<std::size_t>(written);
    }

    if (fsync(fd) != 0)
    {
        log_error("Failed to fsync temp file: {} (errno: {})", tempPath.native(), errno);
        return DCGM_ST_FILE_IO_ERROR;
    }

    // Explicit checked close (can surface deferred write errors); fd = -1 disarms the closeFd guard.
    int const closeRc = close(fd);
    fd                = -1;
    if (closeRc != 0)
    {
        log_error("Failed to close temp file: {} (errno: {})", tempPath.native(), errno);
        return DCGM_ST_FILE_IO_ERROR;
    }

    // Reject symlink targets to prevent symlink-swap attacks
    auto status = std::filesystem::symlink_status(filePath, ec);
    if (!ec && std::filesystem::is_symlink(status))
    {
        log_error("Config path must not be a symlink: {}", filePath.native());
        return DCGM_ST_FILE_IO_ERROR;
    }

    // Atomically replace the target file via rename
    std::filesystem::rename(tempPath, filePath, ec);
    if (ec)
    {
        log_error("Failed to rename {} to {}: {}", tempPath.native(), filePath.native(), ec.message());
        return DCGM_ST_FILE_IO_ERROR;
    }

    // The temp file is now the live config; nothing left to clean up.
    removeTemp.Disarm();

    status = std::filesystem::symlink_status(filePath, ec);
    if (ec || std::filesystem::is_symlink(status))
    {
        log_error("Config path verification failed after rename: {}", filePath.native());
        return DCGM_ST_FILE_IO_ERROR;
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmPolicyPersistence::EnsureDirectoryExists(std::filesystem::path const &filePath)
{
    std::filesystem::path dirPath = filePath.parent_path();

    // Nothing to do if directory already exists or path has no parent
    std::error_code ec;
    if (dirPath.empty() || std::filesystem::exists(dirPath, ec))
    {
        return DCGM_ST_OK;
    }

    // Create all missing intermediate directories (reuse ec — prior call's state is no longer needed)
    ec.clear();
    std::filesystem::create_directories(dirPath, ec);
    if (ec)
    {
        log_error("Failed to create directory {}: {}", dirPath.native(), ec.message());
        return DCGM_ST_FILE_IO_ERROR;
    }

    log_debug("Created directory: {}", dirPath.native());
    return DCGM_ST_OK;
}
