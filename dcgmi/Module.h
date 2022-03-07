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
#ifndef MODULE_H_
#define MODULE_H_

#include "Command.h"
#include "DcgmiOutput.h"
#include "dcgm_structs.h"

class Module
{
public:
    Module();
    virtual ~Module();
    dcgmReturn_t RunBlacklistModule(dcgmHandle_t dcgmHandle, dcgmModuleId_t moduleId, DcgmiOutput &out);
    dcgmReturn_t RunListModule(dcgmHandle_t dcgmHandle, DcgmiOutput &out);
    dcgmReturn_t statusToStr(dcgmModuleStatus_t status, std::string &str);
    static dcgmReturn_t moduleIdToName(dcgmModuleId_t moduleId, std::string &str);

private:
};

class BlacklistModule : public Command
{
public:
    BlacklistModule(const std::string &hostname, const std::string &moduleName, bool json);

protected:
    dcgmReturn_t DoExecuteConnected() override;
    dcgmReturn_t DoExecuteConnectionFailure(dcgmReturn_t connectionStatus) override;

private:
    Module mModuleObj;
    const std::string mModuleName;
};

class ListModule : public Command
{
public:
    ListModule(const std::string &hostname, bool json);

protected:
    dcgmReturn_t DoExecuteConnected() override;
    dcgmReturn_t DoExecuteConnectionFailure(dcgmReturn_t connectionStatus) override;

private:
    Module mModuleObj;
};

#endif /* MODULE_H_ */
