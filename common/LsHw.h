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

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "DcgmUtilities.h"

class LsHw
{
public:
    LsHw();

    void SetChecker(std::unique_ptr<DcgmNs::Utils::RunningUserChecker> checker);
    void SetRunCmdHelper(std::unique_ptr<DcgmNs::Utils::RunCmdHelper> runCmdHelper);

    virtual std::optional<std::vector<std::string>> GetCpuSerials() const;

protected:
    std::optional<std::vector<std::string>> ParseCpuSerials(std::string const &stdout) const;

private:
    std::optional<std::string> Exec() const;

    std::unique_ptr<DcgmNs::Utils::RunningUserChecker> m_runningUserChecker;
    std::unique_ptr<DcgmNs::Utils::RunCmdHelper> m_runCmdHelper;
};