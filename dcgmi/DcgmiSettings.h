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

#include "Command.h"
#include "DcgmiOutput.h"
#include "dcgm_structs.h"

class DcgmiSettingsSetLoggingSeverity : public Command
{
private:
    std::string mTargetLogger;
    std::string mTargetSeverity;

public:
    DcgmiSettingsSetLoggingSeverity(const std::string &hostname,
                                    const std::string &targetLogger,
                                    const std::string &targetSeverity,
                                    bool outputAsJson);

protected:
    dcgmReturn_t DoExecuteConnected() override;
};
