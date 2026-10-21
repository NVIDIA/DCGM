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

#pragma once

#include "DcgmPolicy.h"
#include "DcgmPolicyRegistry.h"
#include "dcgm_policy_structs.h"

#include <dcgm_structs.h>

#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace YAML
{
class Node;
}

/**
 * Policy Persistence - Serialization layer for policy configuration
 *
 * Provides YAML-based persistent storage for monitoring policies. Handles
 * loading policies at startup and saving policies when they are created,
 * modified, or deleted. Uses DcgmPolicyRegistry to create and manage policies,
 * ensuring policies are properly registered when loaded from disk.
 */
class DcgmPolicyPersistence
{
public:
    /**
     * Constructs persistence layer
     *
     * @param[in] registry Reference to policy registry for policy management
     * @param[in] configFilePath Path to YAML config file (empty = use default path resolution)
     */
    explicit DcgmPolicyPersistence(DcgmPolicyRegistry &registry, std::filesystem::path configFilePath = {});

    ~DcgmPolicyPersistence();

    /**
     * Parse YAML file and replace all registry policies with the parsed set (invalid entries skipped).
     *
     * @return DCGM_ST_OK on success
     * @return DCGM_ST_BADPARAM if path is outside allowed directories, file too large,
     *         schema version unsupported, or YAML is malformed
     * @return DCGM_ST_FILE_IO_ERROR if file doesn't exist or is a symlink
     */
    dcgmReturn_t LoadPolicies();

    /**
     * Save registered policies to YAML file (atomic write)
     *
     * @return DCGM_ST_OK on success
     * @return DCGM_ST_BADPARAM if path is outside allowed directories
     * @return DCGM_ST_GENERIC_ERROR if registry retrieval or YAML serialization fails
     * @return DCGM_ST_FILE_IO_ERROR if directory creation, symlink check, temp write, or rename fails
     */
    dcgmReturn_t SavePolicies();

    /**
     * Set the config file path for subsequent operations.
     * Validates the path via ValidatePath before storing it.
     * On failure the existing path is preserved unchanged.
     *
     * @param[in] path New path to use
     *
     * @return DCGM_ST_OK on success
     * @return DCGM_ST_BADPARAM if path is outside allowed directories or cannot be canonicalized
     */
    dcgmReturn_t SetConfigPath(std::filesystem::path const &path);

    /**
     * Get configured/default config file path.
     *
     * @return Path that will be used
     */
    [[nodiscard]] std::filesystem::path GetConfigPath() const;

private:
    // Non-owning; the registry outlives this persistence object.
    DcgmPolicyRegistry *m_registry;
    std::filesystem::path m_configFilePath;

    /**
     * Validate path for security (no traversal, within allowed directories).
     *
     * Accepts paths under the built-in production directories, plus any directories named in the
     * DCGM_POLICY_ALLOWED_DIRS environment variable (colon-separated). The override is intended for
     * admin/test use and never relaxes the built-in directories.
     *
     * @param[in] path Path to validate
     *
     * @return DCGM_ST_OK if valid
     * @return DCGM_ST_BADPARAM if path cannot be canonicalized or is outside allowed directories
     */
    dcgmReturn_t ValidatePath(std::filesystem::path const &path) const;

    /**
     * Parse YAML file into a vector of DcgmPolicy objects (no registry interaction).
     *
     * @param[in] filePath Path to YAML file
     * @param[out] policies Parsed policies appended here
     *
     * @return DCGM_ST_OK on success (partial success if some entries fail validation)
     * @return DCGM_ST_FILE_IO_ERROR if file doesn't exist or is a symlink
     * @return DCGM_ST_BADPARAM if file too large, schema version unsupported, or YAML is malformed
     */
    dcgmReturn_t ParseYamlFile(std::filesystem::path const &filePath, std::vector<DcgmPolicy> &policies);

    /**
     * Parse a single YAML policy node into a DcgmPolicy.
     *
     * @param[in] node YAML map node for one policy entry
     * @param[out] policy Populated on success
     *
     * @return DCGM_ST_OK on success
     * @return DCGM_ST_BADPARAM if operator or channel string is unrecognised
     */
    dcgmReturn_t ParsePolicyNode(YAML::Node const &node, DcgmPolicy &policy);

    /**
     * Serialize policies to YAML string
     *
     * @param[in] first Iterator to the first policy to serialize
     * @param[in] last Iterator one past the last policy to serialize
     * @param[out] yamlContent YAML string content
     *
     * @return DCGM_ST_OK on success
     * @return DCGM_ST_GENERIC_ERROR if YAML emitter throws an exception
     */
    dcgmReturn_t SerializeToYaml(DcgmPolicy const *first, DcgmPolicy const *last, std::string &yamlContent);

    /**
     * Atomic write: write to temp file then rename
     *
     * @param[in] filePath Target file path
     * @param[in] content Content to write
     *
     * @return DCGM_ST_OK on success
     * @return DCGM_ST_FILE_IO_ERROR if target is a symlink, temp file write fails,
     *         or atomic rename fails
     */
    dcgmReturn_t AtomicWriteFile(std::filesystem::path const &filePath, std::string_view content);

    /**
     * Ensure parent directory exists for file path
     *
     * @param[in] filePath File path to check
     *
     * @return DCGM_ST_OK on success or if directory already exists
     * @return DCGM_ST_FILE_IO_ERROR if directory creation fails
     */
    dcgmReturn_t EnsureDirectoryExists(std::filesystem::path const &filePath);
};
