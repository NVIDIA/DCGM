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
#include "Module.h"
#include "DcgmiOutput.h"
#include "dcgm_agent.h"
#include "dcgm_fields.h"
#include "dcgm_structs.h"

#include "algorithm"
#include "cctype"
#include "iomanip"
#include "iostream"
#include "sstream"
#include <tclap/XorHandler.h>

// Module names
#define MODULE_CORE_NAME       "Core"
#define MODULE_NVSWITCH_NAME   "NvSwitch"
#define MODULE_VGPU_NAME       "VGPU"
#define MODULE_INTROSPECT_NAME "Introspection"
#define MODULE_HEALTH_NAME     "Health"
#define MODULE_POLICY_NAME     "Policy"
#define MODULE_CONFIG_NAME     "Config"
#define MODULE_DIAG_NAME       "Diag"
#define MODULE_PROFILING_NAME  "Profiling"

// Commands
#define BLACKLIST_MODULE "Blacklist Module"
#define LIST_MODULES     "List Modules"

// Misc. strings
#define FAILURE     "Failure"
#define MODULE_ID   "Module ID"
#define MODULES     "Modules"
#define NAME        "Name"
#define RETURN      "Return"
#define STATE       "State"
#define STATUS      "Status"
#define STATUS_CODE "Status Code"
#define SUCCESS     "Success"

std::string toLower(std::string str)
{
    // Note ::tolower does not support UTF-8 strings.
    std::transform(str.begin(), str.end(), str.begin(), ::tolower);
    return str;
}

std::string usage()
{
    std::ostringstream ss;
    std::string moduleList, buf;
    bool firstModule = true;

    for (unsigned int i = 0; (dcgmModuleId_t)i < DcgmModuleIdCount; i = (dcgmModuleId_t)(i + 1))
    {
        Module::moduleIdToName((dcgmModuleId_t)i, buf);

        ss.clear();
        ss.str("");
        ss << '(' << i << ')';

        moduleList += (firstModule ? "" : ", ") + buf + ' ' + ss.str();
        firstModule = false;
    }

    return "Module has to be one of: " + moduleList;
}

dcgmReturn_t Module::moduleIdToName(dcgmModuleId_t moduleId, std::string &str)
{
    // If adding a case here, you very likely also need to add one in the
    // BlacklistModule::Execute method below
    switch (moduleId)
    {
        case DcgmModuleIdCore:
            str = MODULE_CORE_NAME;
            return DCGM_ST_OK;
        case DcgmModuleIdNvSwitch:
            str = MODULE_NVSWITCH_NAME;
            return DCGM_ST_OK;
        case DcgmModuleIdVGPU:
            str = MODULE_VGPU_NAME;
            return DCGM_ST_OK;
        case DcgmModuleIdIntrospect:
            str = MODULE_INTROSPECT_NAME;
            return DCGM_ST_OK;
        case DcgmModuleIdHealth:
            str = MODULE_HEALTH_NAME;
            return DCGM_ST_OK;
        case DcgmModuleIdPolicy:
            str = MODULE_POLICY_NAME;
            return DCGM_ST_OK;
        case DcgmModuleIdConfig:
            str = MODULE_CONFIG_NAME;
            return DCGM_ST_OK;
        case DcgmModuleIdDiag:
            str = MODULE_DIAG_NAME;
            return DCGM_ST_OK;
        case DcgmModuleIdProfiling:
            str = MODULE_PROFILING_NAME;
            return DCGM_ST_OK;
        case DcgmModuleIdCount:
            return DCGM_ST_BADPARAM;

            // No default case in the hopes the compiler will complain about missing cases
    }
    return DCGM_ST_BADPARAM;
}

dcgmReturn_t Module::statusToStr(dcgmModuleStatus_t status, std::string &str)
{
    switch (status)
    {
        case DcgmModuleStatusNotLoaded:
            str = "Not loaded";
            return DCGM_ST_OK;
        case DcgmModuleStatusBlacklisted:
            str = "Blacklisted";
            return DCGM_ST_OK;
        case DcgmModuleStatusFailed:
            str = "Failed to load";
            return DCGM_ST_OK;
        case DcgmModuleStatusLoaded:
            str = "Loaded";
            return DCGM_ST_OK;
        case DcgmModuleStatusUnloaded:
            str = "Unloaded";
            return DCGM_ST_OK;

            // No default case in the hopes the compiler will complain about missing cases
    }
    return DCGM_ST_BADPARAM;
}

Module::Module()
{}


Module::~Module()
{}

dcgmReturn_t Module::RunBlacklistModule(dcgmHandle_t dcgmHandle, dcgmModuleId_t moduleId, DcgmiOutput &out)
{
    dcgmReturn_t st;
    std::string moduleStr;

    st = moduleIdToName(moduleId, moduleStr);
    if (st)
    {
        out.addHeader(STATUS ": " FAILURE);
        out.addHeader("Could not convert module id to name. Logic error");
        return st;
    }

    st = dcgmModuleBlacklist(dcgmHandle, moduleId);
    if (st)
    {
        out.addHeader(STATUS ": " FAILURE);
        out.addHeader("Could not blacklist module " + moduleStr);
        out[RETURN] = errorString(st);
        return st;
    }

    out.addHeader(STATUS ": " SUCCESS);
    out.addHeader("Successfully blacklisted module " + moduleStr);

    return DCGM_ST_OK;
}

dcgmReturn_t Module::RunListModule(dcgmHandle_t dcgmHandle, DcgmiOutput &out)
{
    dcgmReturn_t st;
    std::string moduleStr;
    std::string statusStr;
    dcgmModuleGetStatuses_t statuses;

    memset(&statuses, 0, sizeof(statuses));
    statuses.version = dcgmModuleGetStatuses_version;
    st               = dcgmModuleGetStatuses(dcgmHandle, &statuses);

    if (st)
    {
        return DCGM_ST_GENERIC_ERROR;
    }

    for (unsigned int i = 0; i < statuses.numStatuses; i++)
    {
        if (moduleIdToName(statuses.statuses[i].id, moduleStr))
        {
            out.addHeader(STATUS ": " FAILURE);
            out.addHeader("Could not find module name. This is probably a bug");
            return DCGM_ST_GENERIC_ERROR;
        }
        if (statusToStr(statuses.statuses[i].status, statusStr))
        {
            out.addHeader(STATUS ": " FAILURE);
            out.addHeader("Could not find status string. This is probably a bug");
            return DCGM_ST_GENERIC_ERROR;
        }
        out[moduleStr][STATE]     = statusStr;
        out[moduleStr][MODULE_ID] = i;
        out[moduleStr][NAME]      = moduleStr;
    }

    out.addHeader(STATUS ": " SUCCESS);
    return DCGM_ST_OK;
}

/*****************************************************************************/
BlacklistModule::BlacklistModule(const std::string &hostname, const std::string &moduleName, bool json)
    : mModuleName(moduleName)
{
    m_hostName = hostname;
    m_json     = json;
}


/*****************************************************************************/
dcgmReturn_t BlacklistModule::DoExecuteConnected()
{
    unsigned int moduleId;
    std::istringstream ss(mModuleName);
    DcgmiOutputTree outTree(30, 50);
    DcgmiOutputJson outJson;
    DcgmiOutput &out = m_json ? (DcgmiOutput &)outJson : (DcgmiOutput &)outTree;

    ss >> moduleId;

    // If we can't turn moduleName into an unsigned int, try to match it to one
    // of the module names
    if (ss.rdstate() != ss.goodbit && ss.rdstate() != ss.eofbit)
    {
        std::string lowerModule = toLower(mModuleName);
        if (lowerModule == toLower(MODULE_CORE_NAME))
        {
            moduleId = DcgmModuleIdCore;
        }
        else if (lowerModule == toLower(MODULE_NVSWITCH_NAME))
        {
            moduleId = DcgmModuleIdNvSwitch;
        }
        else if (lowerModule == toLower(MODULE_VGPU_NAME))
        {
            moduleId = DcgmModuleIdVGPU;
        }
        else if (lowerModule == toLower(MODULE_INTROSPECT_NAME))
        {
            moduleId = DcgmModuleIdIntrospect;
        }
        else if (lowerModule == toLower(MODULE_HEALTH_NAME))
        {
            moduleId = DcgmModuleIdHealth;
        }
        else if (lowerModule == toLower(MODULE_POLICY_NAME))
        {
            moduleId = DcgmModuleIdPolicy;
        }
        else if (lowerModule == toLower(MODULE_CONFIG_NAME))
        {
            moduleId = DcgmModuleIdConfig;
        }
        else if (lowerModule == toLower(MODULE_DIAG_NAME))
        {
            moduleId = DcgmModuleIdDiag;
        }
        else
        {
            throw TCLAP::CmdLineParseException(usage());
        }
    }

    if (moduleId >= DcgmModuleIdCount)
    {
        throw TCLAP::CmdLineParseException(usage());
    }

    out.addHeader(BLACKLIST_MODULE);

    auto const st = mModuleObj.RunBlacklistModule(m_dcgmHandle, (dcgmModuleId_t)moduleId, out);
    std::cout << out.str();
    return st;
}

dcgmReturn_t BlacklistModule::DoExecuteConnectionFailure(dcgmReturn_t connectionStatus)
{
    DcgmiOutputTree outTree(30, 50);
    DcgmiOutputJson outJson;
    DcgmiOutput &out = m_json ? (DcgmiOutput &)outJson : (DcgmiOutput &)outTree;

    out.addHeader(BLACKLIST_MODULE);

    out.addHeader(STATUS ": " FAILURE);
    out.addHeader("Error: Unable to connect to host engine.");
    out.addHeader(errorString(connectionStatus));
    std::cerr << "Error: Unable to connect to host engine." << std::endl;
    return connectionStatus;
}

/*****************************************************************************/
ListModule::ListModule(const std::string &hostname, bool json)
{
    m_hostName = hostname;
    m_json     = json;
}

/*****************************************************************************/
dcgmReturn_t ListModule::DoExecuteConnected()
{
    DcgmiOutputColumns outColumns;
    DcgmiOutputJson outJson;
    DcgmiOutput &out                            = m_json ? (DcgmiOutput &)outJson : (DcgmiOutput &)outColumns;
    DcgmiOutputFieldSelector moduleIdSelector   = DcgmiOutputFieldSelector().child(MODULE_ID);
    DcgmiOutputFieldSelector moduleNameSelector = DcgmiOutputFieldSelector().child(NAME);
    DcgmiOutputFieldSelector stateSelector      = DcgmiOutputFieldSelector().child(STATE);

    out.addColumn(11, MODULE_ID, moduleIdSelector);
    out.addColumn(20, NAME, moduleNameSelector);
    out.addColumn(50, STATE, stateSelector);

    out.addHeader(LIST_MODULES);

    auto const st = mModuleObj.RunListModule(m_dcgmHandle, out);
    std::cout << out.str();
    return st;
}

dcgmReturn_t ListModule::DoExecuteConnectionFailure(dcgmReturn_t connectionStatus)
{
    DcgmiOutputColumns outColumns;
    DcgmiOutputJson outJson;
    DcgmiOutput &out = m_json ? (DcgmiOutput &)outJson : (DcgmiOutput &)outColumns;

    out.addHeader(LIST_MODULES);

    out.addHeader(STATUS ": " FAILURE);
    out.addHeader("Error: Unable to connect to host engine.");
    out.addHeader(errorString(connectionStatus));
    std::cerr << "Error: Unable to connect to host engine." << std::endl;
    return connectionStatus;
}
