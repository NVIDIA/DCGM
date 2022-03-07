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
#include <sstream>
#include <stdexcept>

#include "DcgmModule.h"

/*****************************************************************************/
DcgmModule::~DcgmModule() = default;

/*****************************************************************************/
dcgmReturn_t DcgmModule::CheckVersion(dcgm_module_command_header_t const *moduleCommand, unsigned int compareVersion)
{
    if (moduleCommand == nullptr)
    {
        DCGM_LOG_ERROR << "Bad parameter.";
        return DCGM_ST_BADPARAM;
    }

    if (moduleCommand->version != compareVersion)
    {
        DCGM_LOG_ERROR << "Version mismatch " << std::hex << moduleCommand->version << " != " << compareVersion
                       << " for module " << std::dec << moduleCommand->moduleId << " subCommand "
                       << moduleCommand->subCommand;
        return DCGM_ST_VER_MISMATCH;
    }

    return DCGM_ST_OK;
}

dcgmReturn_t PassMessageToModule(DcgmModule *module, dcgm_module_command_header_t *moduleCommand)
{
    if (nullptr == module)
    {
        DCGM_LOG_ERROR << "Module is null";
        return DCGM_ST_BADPARAM;
    }

    if (moduleCommand == nullptr)
    {
        DCGM_LOG_ERROR << "Module command is null";
        return DCGM_ST_BADPARAM;
    }

    dcgmReturn_t ret = DCGM_ST_OK;

    try
    {
        ret = module->ProcessMessage(moduleCommand);
    }
    catch (std::runtime_error const &ex)
    {
        DCGM_LOG_ERROR << "An exception occurred in DcgmModule::ProcessMessage. Ex: " << ex.what();
        ret = DCGM_ST_BADPARAM;
    }
    catch (std::exception const &ex)
    {
        DCGM_LOG_ERROR << "A generic exception occurred in DcgmModule::ProcessMessage. Ex: " << ex.what();
        ret = DCGM_ST_BADPARAM;
    }
    catch (...)
    {
        DCGM_LOG_ERROR << "An unknown exception occurred in DcgmModule::ProcessMessage";
        ret = DCGM_ST_BADPARAM;
    }

    return ret;
}
/*****************************************************************************/
