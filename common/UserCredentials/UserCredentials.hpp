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

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <sys/types.h>
#include <vector>

/**
 * Defines the behavior of the \c ChangeUser() function.
 */
enum class ChangeUserPolicy : std::uint8_t
{
    Permanently, /*!< Request to change the current process user permanently. No rollback possible after that */
    Temporarily, /*!< Request to change the current process user temporarily. Later this change can be rolled back */
    Rollback,    /*!< Request to rollback the current user to the previously preserved one */
};

/**
 * Represents information about user account (uid/gid) and groups it belongs to.
 */
struct UserCredentials
{
    gid_t gid;                 /*!< User group id */
    uid_t uid;                 /*!< User id */
    std::vector<gid_t> groups; /*!< List of user's group ids. This is filled and used by \c ChangeUser(). */
};

/**
 * Exception happens if some logic error unrelated to the underlying syscalls happens in the ChangeUser().
 */
struct ChangeUserException : std::runtime_error
{
    using std::runtime_error::runtime_error;
};

/**
 * Changes the effective user/group of the current process.
 *
 * This function allows to change the user of the current process either temporarily or permanently.
 * The main purpose of this function is to drop/reacquire the root privileges of the calling process.
 * If \a policy is \c ChangeUserPolicy::Temporarily, then the function returns old credentials of the
 * user that called this function, and later those credentials can be used to reacquire permissions
 * back with \c ChangeUserPolicy::Rollback. In case of permanent policy, the old credentials cannot be
 * reacquired, thus nothing is returned from this function.
 *
 * @param[in] policy            Defines the behavior of the function - whether the user should be
 * changed permanently/temporarily or previous user should be restored.
 * @param[in] newCredentials    UserCredentials for the target user.
 *
 * @return \c std::nullopt      If the \a policy was either \c ChangeUserPolicy::Permanently or \c
 *                              ChangeUserPolicy::Rollback.
 * @return \c UserCredentials   UserCredentials of the user that called this function. That value can
 * be used later with \c ChangeUserPolicy::Rollback \a policy.
 *
 * @throws std::system_error            If underlying syscalls return errors.
 * @throws ChangeUserException          If there is some logic error that is not directly related to
 * the syscalls. For example, if \a policy is \c ChangeUserPolicy::Permanently and it's possible to
 *                                      reacquire the old privileges back.
 * @note This function should not get credentials of a user with root privileges if the \a policy is
 *       ChangeUserPolicy::Permanently. In such a case, the function will throw an exception as it
 * will be possible to reacquire the uid/gid of the user who is originally called the function.
 * @example
 * @code
 *      //
 *      // Dropping the root privileges
 *      //
 *      try {
 *          ChangeUser(ChangeUserPolicy::Permanently, GetUserCredentials("nobody"));
 *      } except (std::exception const& ex) {
 *          // it's impossible to drop the root permissions
 *      }
 *
 *      //
 *      // Temporarily dropping the root privileges
 *      //
 *      UserCredentials oldCred;
 *      try {
 *          oldCred = ChangeUser(ChangeUserPolicy::Temporarily,
 * GetUserCredentials("nobody"))->value(); } catch (std::exception const& ex) {
 *          //it's impossible to drop the root privileges
 *      }
 *      // do some non-root activity
 *      try {
 *          ChangeUser(ChangeUserPolicy::Rollback, oldCred);
 *      } except (std::exception const& ex) {
 *          //it's impossible to reacquire previous user privileges
 *      }
 * @endcode
 */
std::optional<UserCredentials> ChangeUser(ChangeUserPolicy policy,
                                          UserCredentials const &newCredentials) noexcept(false);

/**
 * Returns credentials for the user with the given \a userName.
 * @param[in] userName  User name, the gid/uid are got
 * @return \c std::nullopt if there is not such login in the system
 * @return \c UserCredentials  gid/uid for the given \a userName
 * @throws \c std::system_error if the underlying OS API returns errors
 * @note The \c UserCredentials::groups is not filled in this function. It's only used/filled by \c ChangeUser()
 */
std::optional<UserCredentials> GetUserCredentials(const char *userName) noexcept(false);

/**
 * User-credential lookup, exposed as an injectable dependency.
 */
class UserCredentialsOps
{
public:
    /**
     * Calls the global ::GetUserCredentials.
     *
     * @param[in] userName  User name to resolve.
     * @return std::nullopt if no such user exists; otherwise the user's UserCredentials.
     */
    std::optional<UserCredentials> GetUserCredentials(char const *userName) const
    {
        return ::GetUserCredentials(userName);
    }

    bool operator==(UserCredentialsOps const &) const = default;
};
