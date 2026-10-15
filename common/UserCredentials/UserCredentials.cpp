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

#include "UserCredentials.hpp"

#include <fmt/core.h>
#include <fmt/format.h>

#include <algorithm>
#include <bit>
#include <cerrno>
#include <cstddef>
#include <cstdio>
#include <grp.h>
#include <pwd.h>
#include <system_error>
#include <type_traits>
#include <unistd.h>

std::optional<UserCredentials> GetUserCredentials(char const *userName) noexcept(false)
{
    constexpr size_t INITIAL_BUFFER_SIZE = 1024;
    constexpr size_t MAX_BUFFER_SIZE     = 1024 * 1024; // 1MB
    passwd pwInfo {};
    passwd *result = nullptr;
    std::vector<std::byte> buffer;
    int err = 0;

    if (userName == nullptr)
    {
        fmt::print(stderr, "No username provided\n");
        fflush(stderr);
        return std::nullopt;
    }

    auto pwRecSizeMax = sysconf(_SC_GETPW_R_SIZE_MAX);
    if (pwRecSizeMax > 0 && pwRecSizeMax > static_cast<long>(MAX_BUFFER_SIZE))
    {
        // try with the max size, and hope for the best
        pwRecSizeMax = static_cast<long>(MAX_BUFFER_SIZE);
        fmt::print(stderr, "Limiting buffer size for getpwnam_r to {}\n", MAX_BUFFER_SIZE);
        fflush(stderr);
    }

    size_t bufferSize
        = std::min((pwRecSizeMax > 0 ? static_cast<size_t>(pwRecSizeMax) : INITIAL_BUFFER_SIZE), MAX_BUFFER_SIZE);

    do
    {
        if (bufferSize > MAX_BUFFER_SIZE)
        {
            throw std::system_error(std::error_code(ERANGE, std::generic_category()),
                                    "Buffer size for getpwnam_r exceeds maximum allowed");
        }

        buffer.resize(bufferSize);
        err = getpwnam_r(userName, &pwInfo, std::bit_cast<char *>(buffer.data()), buffer.size(), &result);

        if (err == ERANGE && bufferSize <= MAX_BUFFER_SIZE)
        {
            size_t const newSize = bufferSize * 2;
            if (newSize <= bufferSize)
            {
                throw std::system_error(std::error_code(ERANGE, std::generic_category()),
                                        "Buffer size for getpwnam_r exceeds maximum allowed");
            }
            bufferSize = newSize;
        }
    } while ((err == ERANGE && bufferSize <= MAX_BUFFER_SIZE) || err == EINTR);

    if (err != 0)
    {
        auto const msg = fmt::format("Unable to get gid/uid for user {}", userName);
        fmt::print(stderr, "{}. Error = {}\n", msg, err);
        fflush(stderr);
        throw std::system_error(std::error_code(err, std::generic_category()), msg);
    }

    if (result != nullptr)
    {
        return UserCredentials { result->pw_gid, result->pw_uid, {} };
    }

    return std::nullopt;
}

namespace
{
/**
 * Changes current uid/euid, gid/egid to the provided \a newCredentials .uid/.gid respectively.
 * @param[in] permanent         If permanent is required, the uid/gid will be changed irreversibly. Otherwise only
 *                              euid/egid will be changed that will allow regain permissions back later.
 * @param[in] newCredentials    The uid/gid of a target user to switch to.
 * @return \c std::nullopt                    If the permanent change is requested.
 * @return \c std::optional<UserCredentials>  UserCredentials of the user that called this method if \a permanent was
 *                                            false.
 * @see \c RestorePrivs()
 * @note If the \a permanent is set to true, the function will not return previous credentials (credentials of the user
 *       who called this method, and it will double check that the old permissions cannot be reacquired. This has an
 *       implication that \a newCredentials should not belong to user with root privileges, as the check will fail as
 *       root users can change their uid/gid to whatever they want.
 * @note If the \a permanent is set to false, the credentials of the user that called this function will be returned.
 *       One can reacquire the permissions back later calling \c RestorePrivs() function.
 * @throws std::system_error            if underlying uid/gid functions return error codes.
 * @throws ChangeUserException          if there is some logic error that does not depend on underlying system call. For
 *                                      example, if it's possible to regain privileges back if the \a permanent is set
 *                                      to true.
 */
std::optional<UserCredentials> DropPrivs(bool permanent, UserCredentials const &newCredentials) noexcept(false)
{
    /*
     * Please read this article before doing any changes to this function:
     * https://www.oreilly.com/library/view/secure-programming-cookbook/0596003943/ch01s03.html
     */
    UserCredentials oldCredentials { getegid(), geteuid(), {} };
    auto const ngroupsMax = sysconf(_SC_NGROUPS_MAX);
    if (ngroupsMax < 0)
    {
        auto const err = errno;
        fmt::print(stderr, "Unable to query max groups. errno = {}\n", err);
        fflush(stderr);
        throw std::system_error(std::error_code(err, std::generic_category()), "Unable to query max groups");
    }
    oldCredentials.groups.resize(static_cast<size_t>(ngroupsMax));
    auto const newGroupsSize = getgroups(static_cast<int>(ngroupsMax), oldCredentials.groups.data());
    if (newGroupsSize < 0)
    {
        auto const err = errno;
        fmt::print(stderr, "Unable to get groups. errno = {}\n", err);
        fflush(stderr);
        throw std::system_error(std::error_code(err, std::generic_category()), "Unable to get groups");
    }
    oldCredentials.groups.resize(static_cast<size_t>(newGroupsSize));

    if (oldCredentials.uid == 0 && setgroups(1, &newCredentials.gid) != 0)
    {
        auto const err = errno;
        fmt::print(stderr, "Unable to change groups. errno = {}\n", err);
        fflush(stderr);
        throw std::system_error(std::error_code(err, std::generic_category()), "Unable to change groups");
    }

    if (setegid(newCredentials.gid) != 0)
    {
        auto const err = errno;
        fmt::print(stderr, "Unable to drop root privileges. setegid returned errno = {}\n", err);
        fflush(stderr);
        throw std::system_error(std::error_code(err, std::generic_category()), "Dropping root privileges");
    }

    if (permanent && setgid(newCredentials.gid) != 0)
    {
        auto const err = errno;
        fmt::print(stderr, "Unable to permanently drop root privileges. setgid returned errno = {}\n", err);
        fflush(stderr);
        throw std::system_error(std::error_code(err, std::generic_category()), "Permanently dropping root privileges");
    }

    if (newCredentials.uid != oldCredentials.uid)
    {
        if (permanent)
        {
            if (setuid(newCredentials.uid) != 0)
            {
                auto const err = errno;
                fmt::print(stderr, "Unable to permanently drop root privileges. setuid returned errno = {}\n", err);
                fflush(stderr);
                throw std::system_error(std::error_code(err, std::generic_category()),
                                        "Permanently dropping root privileges");
            }
        }
        else if (seteuid(newCredentials.uid) != 0)
        {
            auto const err = errno;
            fmt::print(stderr, "Unable to drop root privileges. seteuid returned errno = {}\n", err);
            fflush(stderr);
            throw std::system_error(std::error_code(err, std::generic_category()), "Dropping root privileges");
        }
    }

    if (permanent)
    {
        /*
         * Trying to reacquire root privileges. This must fail if we are permanently dropping the root privileges.
         */
        if (newCredentials.gid != oldCredentials.gid
            && (setegid(oldCredentials.gid) == 0 || getegid() != newCredentials.gid))
        {
            throw ChangeUserException("Failed to drop root privileges. Managed to reacquire root egid.");
        }

        if (newCredentials.uid != oldCredentials.uid
            && (seteuid(oldCredentials.uid) == 0 || geteuid() != newCredentials.uid))
        {
            throw ChangeUserException("Failed to drop root privileges. Managed to reacquire root euid.");
        }

        return std::nullopt;
    }

    return oldCredentials;
}

void RestorePrivs(UserCredentials const &oldCredentials) noexcept(false)
{
    if (geteuid() != oldCredentials.uid)
    {
        if (seteuid(oldCredentials.uid) != 0)
        {
            auto const err = errno;
            auto const msg = fmt::format("Unable to restore euid to {}", oldCredentials.uid);
            fmt::print(stderr, "{}\n", msg);
            fflush(stderr);
            throw std::system_error(std::error_code(err, std::generic_category()), msg);
        }
        if (geteuid() != oldCredentials.uid)
        {
            auto const msg
                = fmt::format("Restoring euid to {} failed. Actual euid remains {}", oldCredentials.uid, geteuid());
            fmt::print(stderr, "{}\n", msg);
            fflush(stderr);
            throw ChangeUserException(msg);
        }
    }

    if (getegid() != oldCredentials.gid)
    {
        if (setegid(oldCredentials.gid) != 0)
        {
            auto const err = errno;
            auto const msg = fmt::format("Unable to restore egid to {}", oldCredentials.gid);
            fmt::print(stderr, "{}\n", msg);
            fflush(stderr);
            throw std::system_error(std::error_code(err, std::generic_category()), msg);
        }
        if (getegid() != oldCredentials.gid)
        {
            auto const msg
                = fmt::format("Restoring egid to {} failed. Actual egid remains {}", oldCredentials.gid, getegid());
            fmt::print(stderr, "{}\n", msg);
            fflush(stderr);
            throw ChangeUserException(msg);
        }
    }

    if (oldCredentials.uid == 0 && !oldCredentials.groups.empty()
        && setgroups(oldCredentials.groups.size(), oldCredentials.groups.data()) != 0)
    {
        auto const err = errno;
        fmt::print(stderr, "Unable to restore groups. errno = {}\n", err);
        fflush(stderr);
        throw std::system_error(std::error_code(err, std::generic_category()), "Unable to restore groups");
    }
}

} // namespace

std::optional<UserCredentials> ChangeUser(ChangeUserPolicy policy,
                                          UserCredentials const &newCredentials) noexcept(false)
{
    switch (policy)
    {
        case ChangeUserPolicy::Permanently:
            return DropPrivs(true, newCredentials);
        case ChangeUserPolicy::Temporarily:
            return DropPrivs(false, newCredentials);
        case ChangeUserPolicy::Rollback:
            RestorePrivs(newCredentials);
            break;
        default:
            fmt::print(stderr,
                       "Unexpected privilege policy: {}\n",
                       static_cast<std::underlying_type_t<ChangeUserPolicy>>(policy));
            fflush(stderr);
    }
    return std::nullopt;
}
