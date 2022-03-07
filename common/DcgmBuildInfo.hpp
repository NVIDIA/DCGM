/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

#ifndef DCGM_DCGMBUILDINFO_H_IN
#define DCGM_DCGMBUILDINFO_H_IN

#include <memory>
#include <ostream>
#include <string_view>
#include <unordered_map>


#ifdef __cplusplus
extern "C" {
#endif

extern char const c_dcgmBuildInfo[];

#define BUILDINFO_MAXSIZE 512

#ifdef __cplusplus
}
#endif

namespace DcgmNs
{
/**
 * Struct for a convenient work with the DCGM Build Info
 */
struct DcgmBuildInfo
{
    using BuildInfoMap = std::unordered_map<std::string_view, std::string_view>;

    /**
     * Construct the DcgmBuildInfo object
     * \param[in] rawBuildInfo  A raw build info string. If none is provided, `c_dcgmBuildInfo` of the current built
     *                          module will be used.
     * \sa `struct dcgmVersionInfo_v2`
     */
    explicit DcgmBuildInfo(const char *rawBuildInfo = c_dcgmBuildInfo) noexcept;
    std::string_view GetArchitecture() const;  //!< Target architecture of the DCGM build
    std::string_view GetBranchName() const;    //!< Build branch
    std::string_view GetBuildDate() const;     //!< Build Date
    std::string_view GetBuildId() const;       //!< Build ID
    std::string_view GetBuildPlatform() const; //!< Platform of the build machine (not the target)
    std::string_view GetBuildType() const;     //!< Build Type (Release, Debug, etc.)
    std::string_view GetGitCommit() const;     //!< Latest git commit hash
    std::string_view GetVersion() const;       //!< DCGM Version
    std::string_view GetCRC() const;           //!< CRC of build info
    std::string GetBuildInfoStr() const;       //!< All build parameters as single string

private:
    std::shared_ptr<BuildInfoMap> m_map; //!< KV-map of the build info.
};

} // namespace DcgmNs

std::ostream &operator<<(std::ostream &, DcgmNs::DcgmBuildInfo const &);
std::string GetBuildInfo();

#endif // DCGM_DCGMBUILDINFO_H_IN
