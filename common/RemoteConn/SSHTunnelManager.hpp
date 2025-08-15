/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

#include <ChildProcess/ChildProcess.hpp>
#include <DcgmUtilities.h>
#include <dcgm_structs_internal.h>

#include <boost/filesystem/exception.hpp>
#include <boost/process.hpp>
#include <fmt/format.h>

#include <expected>
#include <string_view>
#include <unordered_map>

#include <pwd.h>
#include <sys/types.h>

#define MIN_PORT_RANGE_ENV     "__DCGM_SSH_PORT_RANGE_MIN__"
#define MAX_PORT_RANGE_ENV     "__DCGM_SSH_PORT_RANGE_MAX__"
#define CONNECTION_SUCCESS_MSG "ntering interactive session"
#define ADDRESS_IN_USE_MSG     "already in use"

namespace DcgmNs::Common::RemoteConn
{

namespace detail
{
    enum class TunnelState
    {
        AddressInUse,
        GenericFailure,
        Active,
    };

    struct ChildProcessFuncs
    {
        std::function<dcgmReturn_t(dcgmChildProcessParams_t const &params, ChildProcessHandle_t &handle, int &pid)>
            Spawn;
        std::function<dcgmReturn_t(ChildProcessHandle_t handle, dcgmChildProcessStatus_t &status)> GetStatus;
        std::function<dcgmReturn_t(ChildProcessHandle_t handle, int &fd)> GetStdErrHandle;
        std::function<dcgmReturn_t(ChildProcessHandle_t handle, int &fd)> GetStdOutHandle;
        std::function<dcgmReturn_t(ChildProcessHandle_t handle, int &fd)> GetDataChannelHandle;
        std::function<dcgmReturn_t(ChildProcessHandle_t handle, bool force)> Stop;
        std::function<dcgmReturn_t(ChildProcessHandle_t handle, int timeoutSec)> Wait;
        std::function<dcgmReturn_t(ChildProcessHandle_t handle, int sigTermTimeoutSec)> Destroy;
    };

    /**
     * @brief Base class for ssh tunnel managers.
     * The class definition is inline to help with template instantiation in tests.
     * @tparam AddressType The type of address to use.
     */
    template <typename AddressType>
    class SSHTunnelManagerBase
    {
    public:
        /**
         * @brief Launches a remote ssh session to remoteHostname and initiates address forwarding
         * from localhost localAddress to remoteHostname remoteAddress.
         * If a uid is provided, the session will be created with the given uid. If no uid is provided,
         * the session will be created with the current process uid.
         * If a session to remoteHostname:remoteAddress is already active, the existing localAddress
         * will be returned and a new session will not be created. The sessions that reuse an existing
         * address are reference counted and will not be ended in EndSession until the reference
         * count drops to 0.
         */
        TunnelState StartSession(std::string_view remoteHostname,
                                 AddressType const &remoteAddress,
                                 AddressType &localAddress,
                                 std::optional<uid_t> uid = std::nullopt)
        {
            {
                std::lock_guard<std::mutex> lg(m_mutex);
                if (m_childProcessFuncs == nullptr)
                {
                    log_error("Child process functions not set");
                    return TunnelState::GenericFailure;
                }
            }

            uid_t sessionUid = uid.value_or(m_currentUid);

            // Increment session refcount if session is already active
            auto mapKey        = GetSessionInfoMapKey(remoteHostname, remoteAddress, sessionUid);
            bool sessionExists = false;
            {
                std::lock_guard<std::mutex> lg(m_mutex);
                auto const it = m_sessionInfo.find(mapKey);
                if (it != m_sessionInfo.end())
                {
                    it->second.refCount++;
                    localAddress  = it->second.localAddress;
                    sessionExists = true;
                }
            }
            if (sessionExists)
            {
                log_debug("Ssh session to {} already active.", mapKey);
                return TunnelState::Active;
            }

            auto nextAddress = GetNextAddress(sessionUid);
            if (!nextAddress)
            {
                log_error("Failed to get address: {}", nextAddress.error());
                return TunnelState::GenericFailure;
            }
            localAddress            = *nextAddress;
            auto startAddress       = localAddress;
            TunnelState tunnelState = TunnelState::GenericFailure;

            do
            {
                tunnelState = StartSessionImpl(remoteHostname, remoteAddress, localAddress, sessionUid);
                if (tunnelState != TunnelState::AddressInUse)
                {
                    break;
                }
                nextAddress = GetNextAddress(sessionUid);
                if (!nextAddress)
                {
                    log_error("Failed to get address: {}", nextAddress.error());
                    tunnelState = TunnelState::GenericFailure;
                    break;
                }
                localAddress = *nextAddress;
            } while (localAddress != startAddress);

            switch (tunnelState)
            {
                case TunnelState::AddressInUse:
                    log_error("SSH session could not be created, no available addresses.");
                    break;
                case TunnelState::GenericFailure:
                    log_error("SSH session could not be created.");
                    break;
                case TunnelState::Active:
                    log_info("SSH port forwarding session to \"{}\" established with local address: {}.",
                             mapKey,
                             localAddress);
                    break;
            }

            return tunnelState;
        }

        /**
         * @brief Ends the remote port forwarding session to remoteHostname:remoteAddress for the
         * given uid when the reference count of the existing session is 1, otherwise only decrements
         * the reference count.
         * @param forceEnd If true, the session will be ended even if it is in use by another process
         * and the reference count is greater than 1. Default is false.
         */
        void EndSession(std::string_view remoteHostname,
                        AddressType const &remoteAddress,
                        std::optional<uid_t> uid     = std::nullopt,
                        std::optional<bool> forceEnd = std::nullopt)
        {
            bool sessionExists = false;
            bool lastSession   = false;
            auto const mapKey  = GetSessionInfoMapKey(remoteHostname, remoteAddress, uid.value_or(m_currentUid));
            auto const node    = [&] {
                std::lock_guard<std::mutex> lg(m_mutex);
                auto const it = m_sessionInfo.find(mapKey);
                if (it != m_sessionInfo.end())
                {
                    sessionExists = true;
                    if (it->second.refCount == 1 || forceEnd.value_or(false))
                    {
                        lastSession = true;
                        return m_sessionInfo.extract(mapKey);
                    }
                    it->second.refCount--;
                }
                return typename SSHSessionInfoType::node_type();
            }();
            if (!sessionExists)
            {
                log_debug("End session called on {}, session does not exist.", mapKey);
                return;
            }
            if (!lastSession)
            {
                log_info("Not ending SSH forwarding session to \"{}\" because it is in use.", mapKey);
            }
            else
            {
                m_childProcessFuncs->Destroy(node.mapped().procHandle, 5);
                CleanUpPostSessionEnd(node.mapped().localAddress);
                log_info("Ending SSH port forwarding session to \"{}\"", mapKey);
            }
        }

        /**
         * @brief Sets the ssh binary executable path (absolute path). This will do nothing if ssh
         * sessions are already active.
         * @returns True on success.
         */
        bool SetSshBinaryPath(std::string_view binaryPath)
        {
            boost::filesystem::path localPath = binaryPath;
            std::lock_guard<std::mutex> lg(m_mutex);
            if (m_sessionInfo.empty())
            {
                std::swap(m_binaryPath, localPath);
                return true;
            }
            return false;
        }

        /**
         * @brief Sets the child process functions for the tunnel manager if not already set. This needs
         * to be set before calling StartSession.
         * @param childProcessFuncs The child process functions to set.
         * @returns True if the child process functions were updated, false otherwise.
         */
        bool SetChildProcessFuncs(ChildProcessFuncs const *childProcessFuncs)
        {
            bool retValue   = false;
            bool isNullPtr  = false;
            bool alreadySet = false;
            {
                std::lock_guard<std::mutex> lg(m_mutex);
                if (childProcessFuncs != nullptr && m_childProcessFuncs == nullptr)
                {
                    m_childProcessFuncs = std::make_unique<ChildProcessFuncs>(*childProcessFuncs);
                    retValue            = true;
                }
                else if (childProcessFuncs == nullptr)
                {
                    isNullPtr = true;
                }
                else
                {
                    alreadySet = true;
                }
            }
            if (isNullPtr)
            {
                log_error("Child process function argument is nullptr. Cannot set child process functions.");
            }
            else if (alreadySet)
            {
                log_info("Child process functions already set. Cannot set child process functions.");
            }
            return retValue;
        }

    protected:
        /**
         * @brief Returns the ssh address forwarding string required for -L argument of ssh command.
         */
        virtual std::string GetSSHAddressForwardingString(AddressType const &local, AddressType const &remote) const
            = 0;

        /**
         * @brief Returns the next address to use of type AddressType for the session if the current address is in use.
         * This is a thread safe function.
         * @param uid The uid for the session.
         */
        virtual std::expected<AddressType, std::string> GetNextAddress(uid_t uid) = 0;

        SSHTunnelManagerBase()
        {
            std::vector<boost::filesystem::path> const binarySearchDirs = { "/usr/bin", "/usr/sbin" };
            m_binaryPath = boost::process::search_path("ssh", binarySearchDirs);
            m_currentUid = getuid();
        }

        /*
         * @brief Non-blocking reads from the given file descriptor into the given string
         * @param fd The file descriptor to read from.
         * @param buf The buffer to read into.
         */
        std::expected<void, std::string> ReadNonBlocking(int fd, std::string &buf)
        {
            char buffer[4096];
            if (fcntl(fd, F_SETFL, fcntl(fd, F_GETFL) | O_NONBLOCK) == -1)
            {
                return std::unexpected(fmt::format("Failed to set non-blocking mode for fd: {}", fd));
            }
            int numBytesRead = 0;
            while (true)
            {
                numBytesRead = read(fd, buffer, sizeof(buffer));
                if (numBytesRead == -1)
                {
                    if (errno == EINTR)
                    {
                        continue;
                    }
                    else if (errno == EAGAIN)
                    {
                        break;
                    }
                    else
                    {
                        return std::unexpected(fmt::format("Failed to read from fd: {}, errno: {}", fd, errno));
                    }
                }
                else if (numBytesRead == 0) // EOF reached
                {
                    break;
                }
                else
                {
                    buf.append(buffer, numBytesRead);
                }
            }
            return {};
        }

        /*
         * @brief Blocking reads from the given file descriptor into the given string
         * @param fd The file descriptor to read from.
         * @param buf The buffer to read into.
         */
        std::expected<void, std::string> ReadBlocking(int fd, std::string &buf)
        {
            char buffer[4096];
            while (true)
            {
                int numBytesRead = read(fd, buffer, sizeof(buffer));
                if (numBytesRead > 0)
                {
                    buf.append(buffer, numBytesRead);
                }
                else if (numBytesRead == -1)
                {
                    if (errno == EINTR || errno == EAGAIN)
                    {
                        continue;
                    }
                    else
                    {
                        return std::unexpected(fmt::format("Failed to read from fd: {}, errno: {}", fd, errno));
                    }
                }
                else if (numBytesRead == 0) // EOF reached
                {
                    break;
                }
            }
            return {};
        }

        TunnelState StartSessionImpl(std::string_view remoteHostname,
                                     AddressType const &remoteAddress,
                                     AddressType &localAddress,
                                     uid_t sessionUid)
        {
            auto fwdAddress = GetSSHAddressForwardingString(localAddress, remoteAddress);
            std::vector<char const *> args
                = { "-o", "ExitOnForwardFailure=yes", "-v", "-N", "-L", fwdAddress.data(), remoteHostname.data() };
            char const **env = nullptr;

            // Provide username for ChildProcess creation only if session uid is not the same as
            // current uid
            std::string username;
            if (sessionUid != m_currentUid)
            {
                username = GetUsernameForUid(sessionUid);
                if (username.empty())
                {
                    log_error("Failed to get username for given uid: {}. Cannot start ssh session.", sessionUid);
                    return TunnelState::GenericFailure;
                }
            }
            dcgmChildProcessParams_t const params = {
                .version       = dcgmChildProcessParams_version1,
                .executable    = m_binaryPath.string().c_str(),
                .args          = args.data(),
                .numArgs       = args.size(),
                .env           = env,
                .numEnv        = 0,
                .userName      = username.empty() ? nullptr : username.c_str(),
                .dataChannelFd = -1,
            };
            ChildProcessHandle_t procHandle;
            int pid = -1;
            if (auto ret = m_childProcessFuncs->Spawn(params, procHandle, pid); ret != DCGM_ST_OK)
            {
                log_error("Failed to spawn ssh child process for {}:{}: {}", remoteHostname, remoteAddress, ret);
                return TunnelState::GenericFailure;
            }

            fmt::memory_buffer stderrBuf;
            std::string debugString;
            // Wait for session activation message. This can take more than 1 second
            // sometimes.
            dcgmChildProcessStatus_t status = {};
            status.version                  = dcgmChildProcessStatus_version1;
            int errFd                       = -1;
            if (auto ret = m_childProcessFuncs->GetStdErrHandle(procHandle, errFd); ret != DCGM_ST_OK)
            {
                log_error("Failed to get stderr handle: {}", ret);
                return TunnelState::GenericFailure;
            }
            dcgmReturn_t statusRet = DCGM_ST_OK;
            while ((statusRet = m_childProcessFuncs->GetStatus(procHandle, status)) == DCGM_ST_OK && status.running)
            {
                if (auto ret = ReadNonBlocking(errFd, debugString); ret != std::unexpected(std::string()))
                {
                    if (debugString.find(CONNECTION_SUCCESS_MSG) != std::string::npos)
                    {
                        break;
                    }
                }
                else
                {
                    log_error("Failed to read from stderr: {}", ret.error());
                    return TunnelState::GenericFailure;
                }
            }

            status.version = dcgmChildProcessStatus_version1;
            if ((statusRet = m_childProcessFuncs->GetStatus(procHandle, status)) == DCGM_ST_OK && !status.running)
            {
                // Block and ensure all stderr is read until pipe is closed by the child process
                if (auto ret = ReadBlocking(errFd, debugString); ret != std::unexpected(std::string()))
                {
                    auto retCode = TunnelState::GenericFailure;
                    if (debugString.find(ADDRESS_IN_USE_MSG) != std::string::npos)
                    {
                        debugString = fmt::format("Address {} already in use. {}", localAddress, debugString);
                        retCode     = TunnelState::AddressInUse;
                    }
                    log_debug("Ssh error: {}", debugString);
                    close(errFd);
                    return retCode;
                }
                else
                {
                    log_error("Failed to read from stderr: {}", ret.error());
                    return TunnelState::GenericFailure;
                }
            }
            else if (statusRet != DCGM_ST_OK)
            {
                log_error("Failed to get child process status: {}", statusRet);
                return TunnelState::GenericFailure;
            }

            auto mapKey = GetSessionInfoMapKey(remoteHostname, remoteAddress, sessionUid);
            SSHSessionInfo sessionInfo;
            sessionInfo.localAddress = localAddress;
            sessionInfo.procHandle   = procHandle;
            sessionInfo.refCount     = 1;

            // The following creates a temporary map to allocate and extract a node that is used
            // to insert into m_sessionInfo. This ensures that the mutex is not held during map
            // node allocation.
            auto newNode = [&] {
                SSHSessionInfoType temp;
                auto [it, success] = temp.emplace(std::move(mapKey), std::move(sessionInfo));
                static_cast<void>(success); // We know it was successful. Suppress unused variable warnings
                return temp.extract(it);
            }();
            bool insertSuccess = false;
            {
                std::lock_guard<std::mutex> lg(m_mutex);

                auto [it, wasInserted, _] = m_sessionInfo.insert(std::move(newNode));
                insertSuccess             = wasInserted;
                if (!wasInserted)
                {
                    localAddress = it->second.localAddress;
                    // Overflow is not expected here because this class is intended to be used by
                    // DcgmConnect that can support at the maximum UINT16_MAX connections, or by
                    // mnubergemm that requires atmost one connection at a time.
                    it->second.refCount++;
                }
            }
            if (!insertSuccess)
            {
                log_debug("SSH session to {}:{} already active with local address {}.",
                          remoteHostname,
                          remoteAddress,
                          localAddress);
                // Destroy the newly spawned child process since it is no longer needed
                m_childProcessFuncs->Destroy(procHandle, 1);
            }

            return TunnelState::Active;
        }

        std::optional<AddressType> LocalAddressFor(std::string_view remoteHostnameAddress) const
        {
            std::lock_guard<std::mutex> lg(m_mutex);
            auto const it = m_sessionInfo.find(remoteHostnameAddress.data());
            if (it != m_sessionInfo.end())
            {
                return it->second.localAddress;
            }
            return std::nullopt;
        }

        virtual void CleanUpPostSessionEnd(AddressType const &)
        {}

        virtual ~SSHTunnelManagerBase()
        {
            for (auto &[key, sessionInfo] : m_sessionInfo)
            {
                log_info("Ending SSH forwarding session to \"{}\"", key);
                m_childProcessFuncs->Destroy(sessionInfo.procHandle, 5);
            }
            m_sessionInfo.clear();
        }

        virtual std::string GetUsernameForUid(uid_t uid) const
        {
            auto pw = getpwuid(uid);
            if (pw == nullptr)
            {
                return std::string();
            }
            return pw->pw_name;
        }

        std::string GetSessionInfoMapKey(std::string_view remoteHostname,
                                         AddressType const &remoteAddress,
                                         uid_t uid) const
        {
            return fmt::format("{}:{}:{}", remoteHostname, remoteAddress, uid);
        }

        uid_t m_currentUid;

        SSHTunnelManagerBase(SSHTunnelManagerBase const &)            = delete;
        SSHTunnelManagerBase &operator=(SSHTunnelManagerBase const &) = delete;
        SSHTunnelManagerBase(SSHTunnelManagerBase &&)                 = delete;
        SSHTunnelManagerBase &operator=(SSHTunnelManagerBase &&)      = delete;

    private:
        struct SSHSessionInfo
        {
            AddressType localAddress;
            ChildProcessHandle_t procHandle;
            uint32_t refCount;
        };

        using SSHSessionInfoType = std::unordered_map<std::string, SSHSessionInfo>;
        SSHSessionInfoType m_sessionInfo;
        boost::filesystem::path m_binaryPath;
        mutable std::mutex m_mutex;
        std::unique_ptr<ChildProcessFuncs> m_childProcessFuncs;
    };

    /**
     * @brief TcpSSHTunnelManager is a child class of SSHTunnelManagerBase that manages ssh tunnel sessions for tcp
     * connections with port forwarding and AddressType as uint16_t. The class definition is inline to help with
     * template instantiation in tests.
     */
    class TcpSSHTunnelManager : public SSHTunnelManagerBase<uint16_t>
    {
    public:
        static constexpr uint16_t DEFAULT_START_PORT_RANGE = 42000;
        static constexpr uint16_t DEFAULT_END_PORT_RANGE   = 65535;

        TcpSSHTunnelManager()
        {
            try
            {
                auto parsePortEnv = [](std::string_view envVarName, uint16_t &port, uint16_t defaultValue) {
                    char *portEnvValue = getenv(envVarName.data());
                    if (!portEnvValue)
                    {
                        log_debug("{} unset. Using default value {}", envVarName.data(), defaultValue);
                        port = defaultValue;
                    }
                    else
                    {
                        log_debug("Reading port value {}, set from {}", portEnvValue, envVarName.data());
                        unsigned long localPort = std::stoul(portEnvValue);
                        port                    = static_cast<uint16_t>(localPort);
                        if (localPort > UINT16_MAX)
                        {
                            log_debug("Port value {} set from {} out of bounds; using value {} instead.",
                                      localPort,
                                      envVarName.data(),
                                      port);
                        }
                    }
                };

                parsePortEnv(MIN_PORT_RANGE_ENV, m_minPort, DEFAULT_START_PORT_RANGE);
                parsePortEnv(MAX_PORT_RANGE_ENV, m_maxPort, DEFAULT_END_PORT_RANGE);

                if (m_maxPort < m_minPort)
                {
                    log_debug("Ssh port range max greater than port range min.");
                    throw std::invalid_argument("Ssh port range max greater than port range min.");
                }
            }
            catch (...)
            {
                log_error("Invalid __DCGM_SSH_PORT_RANGE_*__ values. Using default values.");
                m_minPort = DEFAULT_START_PORT_RANGE;
                m_maxPort = DEFAULT_END_PORT_RANGE;
            }
            m_nextLocalPort = m_minPort;
        }

#ifndef DCGM_SSH_TUNNEL_MANAGER_TEST // Allow tests to peek in
    private:
#endif
        inline std::string GetSSHAddressForwardingString(uint16_t const &localPort,
                                                         uint16_t const &remotePort) const override
        {
            constexpr std::string_view loopback = "127.0.0.1";
            return fmt::format("{}:{}:{}:{}", loopback, localPort, loopback, remotePort);
        }

        std::expected<uint16_t, std::string> GetNextAddress(uid_t) override
        {
            std::lock_guard<std::mutex> lg(m_mutex);
            auto nextPort   = m_nextLocalPort;
            m_nextLocalPort = (m_nextLocalPort == m_maxPort) ? m_minPort : m_nextLocalPort + 1;
            return nextPort;
        }

        uint16_t m_minPort;
        uint16_t m_maxPort;
        uint16_t m_nextLocalPort;
        mutable std::mutex m_mutex;
    };

    /**
     * @brief UdsSSHTunnelManager is a child class of SSHTunnelManagerBase that manages ssh tunnel sessions for unix
     * domain socket forwarding with AddressType as std::string. The class definition is inline to help with template
     * instantiation in tests.
     */
    class UdsSSHTunnelManager : public SSHTunnelManagerBase<std::string>
    {
    public:
        ~UdsSSHTunnelManager()
        {
            try
            {
                // Remove /tmp/dcgm_<uid> and all its contents if it exists
                for (auto const &path : m_tmpDirsCreated)
                {
                    if (boost::filesystem::exists(path))
                    {
                        log_debug("Removing {} directory and all its contents.", path);
                        boost::filesystem::remove_all(path);
                    }
                }
            }
            catch (const boost::filesystem::filesystem_error &e)
            {
                log_debug("Error during directory cleanup: {}", e.what());
            }
        }

    protected:
        virtual inline std::string_view GetPrimaryPath() const
        {
            return RUN_PATH;
        }
        virtual inline std::string_view GetSecondaryPath() const
        {
            return TMP_PATH;
        }
        virtual inline bool IsRunningAsRoot() const
        {
            return this->m_currentUid == ROOT_UID;
        }

        /**
         * @brief Checks if the given user ID is the root user.
         *
         * @param uid User ID to check
         * @return True if the user ID is the root user, false otherwise
         */
        virtual inline bool IsRootUser(uid_t uid) const
        {
            return (uid == ROOT_UID);
        }

        /**
         * @brief Verifies that a path is owned by the given user and has no write permission for others.
         *
         * @param path Path to verify
         * @param expectedOwner Expected owner
         * @param expectedPerms Expected permissions
         * @return Void on success, error message on failure
         */
        virtual std::expected<void, std::string> VerifyPathOwnershipAndPermissions(std::string_view path,
                                                                                   uid_t expectedOwner,
                                                                                   mode_t expectedPerms) const
        {
            struct stat st = {};

            if (stat(path.data(), &st) != 0)
            {
                return std::unexpected(fmt::format("Failed to get status for '{}': {}", path, strerror(errno)));
            }
            if (st.st_uid != expectedOwner)
            {
                return std::unexpected(fmt::format(
                    "Invalid ownership for '{}': owner uid is {} but expected {}", path, st.st_uid, expectedOwner));
            }
            if ((st.st_mode & expectedPerms) != expectedPerms)
            {
                return std::unexpected(fmt::format("Invalid permissions for '{}': current {:o}, expected {:o}",
                                                   path,
                                                   st.st_mode & 0777,
                                                   expectedPerms));
            }

            return {};
        }

    private:
        inline std::string GetSSHAddressForwardingString(std::string const &localUnixPath,
                                                         std::string const &remoteUnixPath) const override
        {
            return fmt::format("{}:{}", localUnixPath, remoteUnixPath);
        }

        std::expected<std::string, std::string> GetNextAddress(uid_t uid) override
        {
            auto basePath = GetUdsFileBasePath(uid);
            if (!basePath)
            {
                return std::unexpected(basePath.error());
            }

            return GetUdsPath(basePath.value());
        }

#define CREATE_DIR_WITH_OWNER_AND_PERMS_IF_NOT_EXISTS(path, owner, perms) \
    if (!boost::filesystem::exists(path))                                 \
    {                                                                     \
        boost::filesystem::create_directory(path);                        \
        auto result = SetPathOwnershipAndPermissions(path, owner, perms); \
        if (!result)                                                      \
        {                                                                 \
            return std::unexpected(result.error());                       \
        }                                                                 \
        if (path.starts_with(TMP_PATH))                                   \
        {                                                                 \
            m_tmpDirsCreated.emplace_back(path);                          \
        }                                                                 \
    }

#define CREATE_OR_VERIFY_DIR_WITH_OWNER_AND_PERMS(path, owner, perms)        \
    CREATE_DIR_WITH_OWNER_AND_PERMS_IF_NOT_EXISTS(path, owner, perms)        \
    else                                                                     \
    {                                                                        \
        auto result = VerifyPathOwnershipAndPermissions(path, owner, perms); \
        if (!result)                                                         \
        {                                                                    \
            return std::unexpected(result.error());                          \
        }                                                                    \
    }

        /**
         * @brief Get the base path for Uds file.
         *
         * Primary path is /run and falls back to /tmp if it is not available.
         * If running as root and primary path is available, it checks if the user (m_userInfo.uid) is root or not.
         *      For root user base path is /run/dcgm
         *      For non-root user base path is /run/user/<uid>/dcgm
         * If primary path isn't available, base path is /tmp/dcgm/<uid>.
         *
         * @return Base path on success, error message on failure.
         */
        std::expected<std::string, std::string> GetUdsFileBasePath(uid_t sessionUid)
        {
            try
            {
                std::string basePath;
                auto primaryPath   = GetPrimaryPath();
                auto secondaryPath = GetSecondaryPath();

                // Notes on directory handling:
                // - nv-hostengine runs as root.
                // - We do not create or modify /run or /tmp directories.
                // - We do not modify ownership or permissions of existing directories.
                // - We only verify ownership and permissions of existing directories under /tmp and not under /run.
                // - If we create directories, we set appropriate ownership and permissions.
                // - We only clean up Uds files at the end of the ssh session and /tmp/dcgm directory at the end of the
                // process. We do not clean up /run directory.
                bool usePrimaryPath = IsRunningAsRoot() && boost::filesystem::exists(primaryPath);
                if (usePrimaryPath && IsRootUser(sessionUid))
                {
                    log_debug("Running as root and using primary path {} as Uds base path prefix for /dcgm.",
                              primaryPath);
                    basePath = fmt::format("{}/{}", primaryPath, DCGM_DIR);
                    CREATE_DIR_WITH_OWNER_AND_PERMS_IF_NOT_EXISTS(basePath, sessionUid, 0700 /* rwx------ */);
                }
                else if (basePath = fmt::format("{}/{}/{}", primaryPath, USER_DIR, sessionUid);
                         usePrimaryPath && boost::filesystem::exists(basePath))
                {
                    log_debug("Running as root and using primary path {} as Uds base path prefix for /user/<uid>/dcgm.",
                              primaryPath);
                    basePath = fmt::format("{}/{}", basePath, DCGM_DIR);
                    CREATE_DIR_WITH_OWNER_AND_PERMS_IF_NOT_EXISTS(basePath, sessionUid, 0700 /* rwx------ */);
                }
                else if (boost::filesystem::exists(secondaryPath))
                {
                    log_info("Using secondary path {} as Uds base path prefix for /dcgm_<uid>.", secondaryPath);
                    basePath = fmt::format("{}/{}_{}", secondaryPath, DCGM_DIR, sessionUid);
                    CREATE_OR_VERIFY_DIR_WITH_OWNER_AND_PERMS(basePath, sessionUid, 0700 /* rwx------ */);
                }
                else
                {
                    return std::unexpected(
                        fmt::format("Failed to establish base path for unix domain socket files. "
                                    "Details: use_primary_path={}, running_as_root={}, is_root_user={}",
                                    usePrimaryPath,
                                    IsRunningAsRoot(),
                                    IsRootUser(sessionUid)));
                }

                log_debug("Uds base path: {}", basePath);
                return basePath;
            }
            catch (const boost::filesystem::filesystem_error &e)
            {
                return std::unexpected(fmt::format("Filesystem error: {}", e.what()));
            }
        }

        /**
         * @brief Get the Uds path.
         *
         * @param dir Base path.
         * @return Uds file path.
         */
        inline std::string GetUdsPath(std::string_view dir)
        {
            return fmt::format("{}/ssh_{}.sock", dir, m_udsFileIndex++);
        }

        /**
         * @brief Sets ownership and permissions for the given path.
         *
         * @param path Path to set ownership and permissions for
         * @param owner User ID to set ownership to
         * @param perms Permissions to set
         * @return Void on success, error message on failure
         */
        std::expected<void, std::string> SetPathOwnershipAndPermissions(std::string_view path,
                                                                        uid_t owner,
                                                                        mode_t perms) const
        {
            if (chown(path.data(), owner, -1) != 0)
            {
                return std::unexpected(fmt::format("Failed to set ownership for {}: {}", path, strerror(errno)));
            }
            if (chmod(path.data(), perms) != 0)
            {
                return std::unexpected(fmt::format("Failed to set permissions for {}: {}", path, strerror(errno)));
            }

            return {};
        }

        void CleanUpPostSessionEnd(std::string const &localAddress) override
        {
            // Remove the local unix domain socket file if it exists.
            if (boost::filesystem::exists(localAddress))
            {
                log_debug("Removing local unix domain socket file {}", localAddress);
                try
                {
                    boost::filesystem::remove(localAddress);
                }
                catch (const boost::filesystem::filesystem_error &e)
                {
                    log_error("Failed to remove Uds file {}: {}", localAddress, e.what());
                }
            }
        }

        std::atomic<uint64_t> m_udsFileIndex { 0 };
        std::vector<std::string> m_tmpDirsCreated;

        static constexpr uid_t ROOT_UID            = 0;
        static constexpr std::string_view RUN_PATH = "/run";
        static constexpr std::string_view TMP_PATH = "/tmp";
        static constexpr std::string_view USER_DIR = "user";
        static constexpr std::string_view DCGM_DIR = "dcgm";

        friend class TestUdsSSHTunnelManager; // Allow tests to access private members.
    };
} //namespace detail

class TcpSSHTunnelManager : public detail::TcpSSHTunnelManager
{
public:
    static TcpSSHTunnelManager &GetInstance()
    {
        static TcpSSHTunnelManager instance;
        return instance;
    }

private:
    TcpSSHTunnelManager()  = default;
    ~TcpSSHTunnelManager() = default;
};

class UdsSSHTunnelManager : public detail::UdsSSHTunnelManager
{
public:
    static UdsSSHTunnelManager &GetInstance()
    {
        static UdsSSHTunnelManager instance;
        return instance;
    }

private:
    UdsSSHTunnelManager()  = default;
    ~UdsSSHTunnelManager() = default;
};

} // namespace DcgmNs::Common::RemoteConn
