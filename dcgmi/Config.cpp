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
/*
 * File:   Config.cpp
 */

#include "Config.h"
#include "CommandOutputController.h"
#include "DcgmiOutput.h"
#include "dcgm_agent.h"
#include "dcgm_structs.h"
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string.h>
#include <utility>

/**************************************************************************/

#define FIELD          "Field"
#define CURRENT        "Current"
#define TARGET         "Target"
#define NOT_APPLICABLE std::string("****")

#define CONFIG_SYNC_BOOST_TAG   "Sync Boost"
#define CONFIG_SM_APP_CLK_TAG   "SM Application Clock"
#define CONFIG_MEM_APP_CLK_TAG  "Memory Application Clock"
#define CONFIG_ECC_MODE_TAG     "ECC Mode"
#define CONFIG_PWR_LIM_TAG      "Power Limit"
#define CONFIG_COMPUTE_MODE_TAG "Compute Mode"


/*****************************************************************************/
dcgmReturn_t Config::RunGetConfig(dcgmHandle_t pNvcmHandle, bool verbose, bool json)
{
    dcgmGroupInfo_t stNvcmGroupInfo;
    dcgmStatus_t stHandle            = 0;
    dcgmConfig_t *pNvcmCurrentConfig = NULL;
    dcgmConfig_t *pNvcmTargetConfig  = NULL;
    dcgmReturn_t ret                 = DCGM_ST_OK;
    dcgmReturn_t result;
    dcgmReturn_t targetResult;
    dcgmDeviceAttributes_t stDeviceAttributes;
    GPUErrorOutputController gpuErrView;
    DcgmiOutputFieldSelector fieldSelector   = DcgmiOutputFieldSelector().child(FIELD);
    DcgmiOutputFieldSelector currentSelector = DcgmiOutputFieldSelector().child(CURRENT);
    DcgmiOutputFieldSelector targetSelector  = DcgmiOutputFieldSelector().child(TARGET);
    unsigned int i;
    std::stringstream ss;

    stDeviceAttributes.version = dcgmDeviceAttributes_version;

    /* Add config watches for the newly created group */
    result = dcgmUpdateAllFields(pNvcmHandle, 1);
    if (DCGM_ST_OK != result)
    {
        std::cout << "Error: Unable to update fields. Return: " << errorString(result) << std::endl;
        PRINT_ERROR("%d", "Error: UpdateAllFields. Return: %d", result);
        return result;
    }

    stNvcmGroupInfo.version = dcgmGroupInfo_version;
    result                  = dcgmGroupGetInfo(pNvcmHandle, mGroupId, &stNvcmGroupInfo);
    if (DCGM_ST_OK != result)
    {
        std::string error = (result == DCGM_ST_NOT_CONFIGURED) ? "The Group is not found" : errorString(result);
        std::cout << "Error: Unable to get group information. Return: " << error << std::endl;
        PRINT_ERROR(
            "%u,%d", "Error: GroupGetInfo for GroupId: %u. Return: %d", (unsigned int)(uintptr_t)mGroupId, result);
        return DCGM_ST_GENERIC_ERROR;
    }

    /* Create Status handler */
    result = dcgmStatusCreate(&stHandle);
    if (DCGM_ST_OK != result)
    {
        std::cout << "Error: Unable to create status handler. Return:" << errorString(result) << std::endl;
        ret = DCGM_ST_GENERIC_ERROR;
        goto cleanup_local;
    }

    pNvcmCurrentConfig = new dcgmConfig_t[stNvcmGroupInfo.count];
    for (i = 0; i < stNvcmGroupInfo.count; i++)
    {
        pNvcmCurrentConfig[i].version = dcgmConfig_version;
    }

    pNvcmTargetConfig = new dcgmConfig_t[stNvcmGroupInfo.count];
    for (i = 0; i < stNvcmGroupInfo.count; i++)
    {
        pNvcmTargetConfig[i].version = dcgmConfig_version;
    }

    result = dcgmConfigGet(
        pNvcmHandle, mGroupId, DCGM_CONFIG_CURRENT_STATE, stNvcmGroupInfo.count, pNvcmCurrentConfig, stHandle);

    targetResult = dcgmConfigGet(
        pNvcmHandle, mGroupId, DCGM_CONFIG_TARGET_STATE, stNvcmGroupInfo.count, pNvcmTargetConfig, stHandle);

    // Populate information in displayInfo for each GPU and print

    for (i = 0; i < stNvcmGroupInfo.count; i++)
    {
        DcgmiOutputColumns outColumns;
        DcgmiOutputJson outJson;
        DcgmiOutput &out = json ? (DcgmiOutput &)outJson : (DcgmiOutput &)outColumns;

        out.addColumn(30, FIELD, fieldSelector);
        out.addColumn(30, TARGET, targetSelector);
        out.addColumn(30, CURRENT, currentSelector);

        out[CONFIG_COMPUTE_MODE_TAG][FIELD] = CONFIG_COMPUTE_MODE_TAG;
        out[CONFIG_ECC_MODE_TAG][FIELD]     = CONFIG_ECC_MODE_TAG;
        out[CONFIG_SYNC_BOOST_TAG][FIELD]   = CONFIG_SYNC_BOOST_TAG;
        out[CONFIG_MEM_APP_CLK_TAG][FIELD]  = CONFIG_MEM_APP_CLK_TAG;
        out[CONFIG_SM_APP_CLK_TAG][FIELD]   = CONFIG_SM_APP_CLK_TAG;
        out[CONFIG_PWR_LIM_TAG][FIELD]      = CONFIG_PWR_LIM_TAG;

        ss.str("");
        if (verbose)
        {
            ss << "GPU ID: " << pNvcmCurrentConfig[i].gpuId;
            out.addHeader(ss.str());
            // Get device name
            dcgmGetDeviceAttributes(pNvcmHandle, pNvcmCurrentConfig[i].gpuId, &stDeviceAttributes);
            out.addHeader(stDeviceAttributes.identifiers.deviceName);
        }
        else
        {
            out.addHeader(stNvcmGroupInfo.groupName);
            ss << "Group of " << stNvcmGroupInfo.count << " GPUs";
            out.addHeader(ss.str());
        }

        // Current Configurations
        if (!verbose
            && !HelperCheckIfAllTheSameMode(pNvcmCurrentConfig, &dcgmConfig_t::computeMode, stNvcmGroupInfo.count))
        {
            out[CONFIG_COMPUTE_MODE_TAG][CURRENT] = NOT_APPLICABLE;
        }
        else
        {
            out[CONFIG_COMPUTE_MODE_TAG][CURRENT] = HelperDisplayComputeMode(pNvcmCurrentConfig[i].computeMode);
        }

        if (!verbose && !HelperCheckIfAllTheSameMode(pNvcmCurrentConfig, &dcgmConfig_t::eccMode, stNvcmGroupInfo.count))
        {
            out[CONFIG_ECC_MODE_TAG][CURRENT] = NOT_APPLICABLE;
        }
        else
        {
            out[CONFIG_ECC_MODE_TAG][CURRENT] = HelperDisplayBool(pNvcmCurrentConfig[i].eccMode);
        }

        if (!verbose
            && !HelperCheckIfAllTheSameBoost(
                pNvcmCurrentConfig, &dcgmConfigPerfStateSettings_t::syncBoost, stNvcmGroupInfo.count))
        {
            out[CONFIG_SYNC_BOOST_TAG][CURRENT] = NOT_APPLICABLE;
        }
        else
        {
            out[CONFIG_SYNC_BOOST_TAG][CURRENT] = HelperDisplayBool(pNvcmCurrentConfig->perfState.syncBoost);
        }

        if (!verbose
            && !HelperCheckIfAllTheSameClock(pNvcmCurrentConfig, &dcgmClockSet_t::memClock, stNvcmGroupInfo.count))
        {
            out[CONFIG_MEM_APP_CLK_TAG][CURRENT] = NOT_APPLICABLE;
        }
        else
        {
            out[CONFIG_MEM_APP_CLK_TAG][CURRENT] = pNvcmCurrentConfig[i].perfState.targetClocks.memClock;
        }

        if (!verbose
            && !HelperCheckIfAllTheSameClock(pNvcmCurrentConfig, &dcgmClockSet_t::smClock, stNvcmGroupInfo.count))
        {
            out[CONFIG_SM_APP_CLK_TAG][CURRENT] = NOT_APPLICABLE;
        }
        else
        {
            out[CONFIG_SM_APP_CLK_TAG][CURRENT] = pNvcmCurrentConfig[i].perfState.targetClocks.smClock;
        }

        if (!verbose && !HelperCheckIfAllTheSamePowerLim(pNvcmCurrentConfig, stNvcmGroupInfo.count))
        {
            out[CONFIG_PWR_LIM_TAG][CURRENT] = NOT_APPLICABLE;
        }
        else
        {
            out[CONFIG_PWR_LIM_TAG][CURRENT] = pNvcmCurrentConfig[i].powerLimit.val;
        }


        // Target Configurations
        if (targetResult != DCGM_ST_OK)
        {
            out[CONFIG_COMPUTE_MODE_TAG][TARGET] = HelperDisplayComputeMode(targetResult);
        }
        else if (!verbose
                 && !HelperCheckIfAllTheSameMode(pNvcmTargetConfig, &dcgmConfig_t::computeMode, stNvcmGroupInfo.count))
        {
            out[CONFIG_COMPUTE_MODE_TAG][TARGET] = NOT_APPLICABLE;
        }
        else
        {
            out[CONFIG_COMPUTE_MODE_TAG][TARGET] = HelperDisplayComputeMode(pNvcmTargetConfig[i].computeMode);
        }

        if (targetResult != DCGM_ST_OK)
        {
            out[CONFIG_ECC_MODE_TAG][TARGET] = HelperDisplayComputeMode(targetResult);
        }
        else if (!verbose
                 && !HelperCheckIfAllTheSameMode(pNvcmTargetConfig, &dcgmConfig_t::eccMode, stNvcmGroupInfo.count))
        {
            out[CONFIG_ECC_MODE_TAG][TARGET] = NOT_APPLICABLE;
        }
        else
        {
            out[CONFIG_ECC_MODE_TAG][TARGET] = HelperDisplayBool(pNvcmTargetConfig[i].eccMode);
        }

        if (targetResult != DCGM_ST_OK)
        {
            out[CONFIG_SYNC_BOOST_TAG][TARGET] = HelperDisplayComputeMode(targetResult);
        }
        else if (!verbose
                 && !HelperCheckIfAllTheSameBoost(
                     pNvcmTargetConfig, &dcgmConfigPerfStateSettings_t::syncBoost, stNvcmGroupInfo.count))
        {
            out[CONFIG_SYNC_BOOST_TAG][TARGET] = NOT_APPLICABLE;
        }
        else
        {
            out[CONFIG_SYNC_BOOST_TAG][TARGET] = HelperDisplayBool(pNvcmTargetConfig->perfState.syncBoost);
        }

        if (targetResult != DCGM_ST_OK)
        {
            out[CONFIG_MEM_APP_CLK_TAG][TARGET] = HelperDisplayComputeMode(targetResult);
        }
        else if (!verbose
                 && !HelperCheckIfAllTheSameClock(pNvcmTargetConfig, &dcgmClockSet_t::memClock, stNvcmGroupInfo.count))
        {
            out[CONFIG_MEM_APP_CLK_TAG][TARGET] = NOT_APPLICABLE;
        }
        else
        {
            out[CONFIG_MEM_APP_CLK_TAG][TARGET] = pNvcmTargetConfig[i].perfState.targetClocks.memClock;
        }

        if (targetResult != DCGM_ST_OK)
        {
            out[CONFIG_SM_APP_CLK_TAG][TARGET] = HelperDisplayComputeMode(targetResult);
        }
        else if (!verbose
                 && !HelperCheckIfAllTheSameClock(pNvcmTargetConfig, &dcgmClockSet_t::smClock, stNvcmGroupInfo.count))
        {
            out[CONFIG_SM_APP_CLK_TAG][TARGET] = NOT_APPLICABLE;
        }
        else
        {
            out[CONFIG_SM_APP_CLK_TAG][TARGET] = pNvcmTargetConfig[i].perfState.targetClocks.smClock;
        }

        if (targetResult != DCGM_ST_OK)
        {
            out[CONFIG_PWR_LIM_TAG][TARGET] = HelperDisplayComputeMode(targetResult);
        }
        else if (!verbose && !HelperCheckIfAllTheSamePowerLim(pNvcmTargetConfig, stNvcmGroupInfo.count))
        {
            out[CONFIG_PWR_LIM_TAG][TARGET] = NOT_APPLICABLE;
        }
        else
        {
            out[CONFIG_PWR_LIM_TAG][TARGET] = pNvcmTargetConfig[i].powerLimit.val;
        }

        std::cout << out.str();

        if (!verbose)
            break; // only need one output in this case
    }

    if (!verbose)
    {
        std::cout << "**** Non-homogenous settings across group. Use with –v flag to see details.\n";
    }

    /**
     * Check for errors (if any)
     */
    if (DCGM_ST_OK != result)
    {
        std::cout << "\nUnable to get some of the configuration properties. Return: " << errorString(result)
                  << std::endl;
        /* Look at status to get individual errors */
        gpuErrView.addError(stHandle);
        gpuErrView.display();
        ret = DCGM_ST_GENERIC_ERROR;
        goto cleanup_local;
    }

cleanup_local:
    /* Destroy Status message */
    if (stHandle)
    {
        result = dcgmStatusDestroy(stHandle);
        if (DCGM_ST_OK != result)
        {
            std::cout << "Unable to destroy status handler. Return: " << result << std::endl;
        }
    }

    if (pNvcmCurrentConfig)
    {
        delete[] pNvcmCurrentConfig;
    }

    if (pNvcmTargetConfig)
    {
        delete[] pNvcmTargetConfig;
    }

    return ret;
}

/*****************************************************************************/
dcgmReturn_t Config::RunSetConfig(dcgmHandle_t pNvcmHandle)
{
    dcgmReturn_t ret = DCGM_ST_OK;
    dcgmReturn_t result;
    dcgmStatus_t stHandle = 0;
    GPUErrorOutputController gpuErrView;

    /* Add config watches for the newly created group */
    result = dcgmUpdateAllFields(pNvcmHandle, 1);
    if (DCGM_ST_OK != result)
    {
        std::cout << "Error: Unable to update fields. Return: " << errorString(result) << std::endl;
        PRINT_ERROR("%d", "Error: UpdateAllFields. Return: %d", result);
        return result;
    }

    /* Create Status handler */
    result = dcgmStatusCreate(&stHandle);
    if (DCGM_ST_OK != result)
    {
        std::cout << "Error: Unable to create status handler. Return:" << errorString(result) << std::endl;
        ret = DCGM_ST_GENERIC_ERROR;
        goto cleanup_local;
    }

    mConfigVal.version = dcgmConfig_version;

    result = dcgmConfigSet(pNvcmHandle, mGroupId, &mConfigVal, stHandle);
    if (DCGM_ST_OK != result)
    {
        std::string error = (result == DCGM_ST_NOT_CONFIGURED) ? "The Group is not found" : errorString(result);
        std::cout << "Error: Unable to set some of the configuration properties. Return: " << error << std::endl;

        if (mConfigVal.perfState.syncBoost == 1)
        {
            gpuErrView.addErrorStringOverride(
                DCGM_FI_SYNC_BOOST, DCGM_ST_BADPARAM, "Syncboost - A GPU is invalid or in another sync boost group");
        }
        else
        {
            gpuErrView.addErrorStringOverride(
                DCGM_FI_SYNC_BOOST, DCGM_ST_BADPARAM, "Syncboost - Already disabled on GPU(s) in group");
        }

        gpuErrView.addError(stHandle);
        gpuErrView.display();

        PRINT_ERROR("%u, %d",
                    "Error: Unable to set configuration on group %u. Return: %d",
                    (unsigned int)(uintptr_t)mGroupId,
                    result);

        ret = result;
        goto cleanup_local;
    }
    else
    {
        std::cout << "Configuration successfully set.\n";
    }

cleanup_local:
    /* Destroy Status message */
    if (stHandle)
    {
        result = dcgmStatusDestroy(stHandle);
        if (DCGM_ST_OK != result)
        {
            std::cout << "Error: Unable to destroy status handler. Return: " << errorString(result) << std::endl;
        }
    }

    return ret;
}

/*****************************************************************************/
dcgmReturn_t Config::RunEnforceConfig(dcgmHandle_t pNvcmHandle)
{
    dcgmGroupInfo_t stNvcmGroupInfo;
    dcgmStatus_t stHandle = 0;
    dcgmReturn_t ret      = DCGM_ST_OK;
    dcgmReturn_t result;
    GPUErrorOutputController gpuErrView;
    /* Add config watches for the newly created group */
    result = dcgmUpdateAllFields(pNvcmHandle, 1);
    if (DCGM_ST_OK != result)
    {
        std::cout << "Error: Unable to update fields. Return: " << errorString(result) << std::endl;
        PRINT_ERROR("%d", "Error: UpdateAllFields. Return: %d", result);
        return result;
    }

    stNvcmGroupInfo.version = dcgmGroupInfo_version;
    result                  = dcgmGroupGetInfo(pNvcmHandle, mGroupId, &stNvcmGroupInfo);
    if (DCGM_ST_OK != result)
    {
        std::string error = (result == DCGM_ST_NOT_CONFIGURED) ? "The Group is not found" : errorString(result);
        std::cout << "Error: Unable to get group information. Return: " << error << std::endl;
        PRINT_ERROR(
            "%u,%d", "Error: GroupGetInfo for GroupId: %u. Return: %d", (unsigned int)(uintptr_t)mGroupId, result);
        return DCGM_ST_GENERIC_ERROR;
    }

    /* Create Status handler */
    result = dcgmStatusCreate(&stHandle);
    if (DCGM_ST_OK != result)
    {
        std::cout << "Error: Unable to create status handler. Return:" << errorString(result) << std::endl;
        ret = DCGM_ST_GENERIC_ERROR;
        goto cleanup_local;
    }

    result = dcgmConfigEnforce(pNvcmHandle, mGroupId, stHandle);

    /**
     * Check for errors (if any)
     */
    if (DCGM_ST_OK != result)
    {
        std::cout << " Error: Unable to enforce some of the configuration properties. Return: " << errorString(result)
                  << std::endl;

        // Add this to override not very informative error messages within the status handle. BUG ->
        gpuErrView.addErrorStringOverride(
            DCGM_FI_UNKNOWN, DCGM_ST_NOT_CONFIGURED, "Unknown - Target configuration not specified.");

        gpuErrView.addError(stHandle);
        gpuErrView.display();

        ret = DCGM_ST_GENERIC_ERROR;
        goto cleanup_local;
    }
    else
    {
        std::cout << "Configuration successfully enforced.\n";
    }

cleanup_local:
    /* Destroy Status message */
    if (stHandle)
    {
        result = dcgmStatusDestroy(stHandle);
        if (DCGM_ST_OK != result)
        {
            std::cout << "Error: Unable to destroy status handler. Return: " << errorString(result) << std::endl;
        }
    }

    return ret;
}

/*****************************************************************************/
template <typename TMember>
bool Config::HelperCheckIfAllTheSameMode(dcgmConfig_t *configs, TMember member, unsigned int numGpus)
{
    for (unsigned int i = 1; i < numGpus; i++)
    {
        if (configs[0].*member != configs[i].*member)
        {
            return false;
        }
    }
    return true;
}

/*****************************************************************************/
template <typename TMember>
bool Config::HelperCheckIfAllTheSameBoost(dcgmConfig_t *configs, TMember member, unsigned int numGpus)
{
    for (unsigned int i = 1; i < numGpus; i++)
    {
        if (configs[0].perfState.*member != configs[i].perfState.*member)
        {
            return false;
        }
    }
    return true;
}

/*****************************************************************************/
template <typename TMember>
bool Config::HelperCheckIfAllTheSameClock(dcgmConfig_t *configs, TMember member, unsigned int numGpus)
{
    for (unsigned int i = 1; i < numGpus; i++)
    {
        if (configs[0].perfState.targetClocks.*member != configs[i].perfState.targetClocks.*member)
        {
            return false;
        }
    }
    return true;
}

/*****************************************************************************/
bool Config::HelperCheckIfAllTheSamePowerLim(dcgmConfig_t *configs, unsigned int numGpus)
{
    for (unsigned int i = 1; i < numGpus; i++)
    {
        if (configs[0].powerLimit.val != configs[i].powerLimit.val)
        {
            return false;
        }
    }
    return true;
}

/*****************************************************************************/
int Config::SetArgs(unsigned int groupId, dcgmConfig_t *pConfigVal)
{
    mGroupId = (dcgmGpuGrp_t)(long long)groupId;

    if (NULL != pConfigVal)
    {
        mConfigVal = *pConfigVal;
    }

    return 0;
}

/*****************************************************************************/
std::string Config::HelperDisplayComputeMode(unsigned int val)
{
    std::stringstream ss;

    if (DCGM_INT32_IS_BLANK(val))
    {
        switch (val)
        {
            case DCGM_INT32_BLANK:
                ss << "Not Specified";
                break;

            case DCGM_INT32_NOT_FOUND:
                ss << "Not Found";
                break;

            case DCGM_INT32_NOT_SUPPORTED:
                ss << "Not Supported";
                break;

            case DCGM_INT32_NOT_PERMISSIONED:
                ss << "Insf. Permission";
                break;

            case (unsigned int)DCGM_ST_NOT_CONFIGURED:
                ss << "Not Configured";
                break;

            default:
                ss << "Unknown";
                break;
        }
    }
    else
    {
        if (DCGM_CONFIG_COMPUTEMODE_DEFAULT == val)
            ss << "Unrestricted";
        else if (DCGM_CONFIG_COMPUTEMODE_PROHIBITED == val)
            ss << "Prohibited";
        else if (DCGM_CONFIG_COMPUTEMODE_EXCLUSIVE_PROCESS == val)
            ss << "E. Process";
        else
            ss << "Unknown"; /* This should never happen */
    }

    return ss.str();
}

/*****************************************************************************/
std::string Config::HelperDisplayCurrentSyncBoost(unsigned int val)
{
    std::stringstream ss;

    if (DCGM_INT32_IS_BLANK(val))
    {
        switch (val)
        {
            case DCGM_INT32_BLANK:
                ss << "Not Specified";
                break;

            case DCGM_INT32_NOT_FOUND:
                ss << "Disabled"; // Not found implies sync-boost is disabled
                break;

            case DCGM_INT32_NOT_SUPPORTED:
                ss << "Not Supported";
                break;

            case DCGM_INT32_NOT_PERMISSIONED:
                ss << "Insf. Permission";
                break;

            default:
                ss << "Unknown";
                break;
        }
    }
    else
    {
        ss << "Enabled [id=";
        ss << val;
        ss << "]";
    }

    return ss.str();
}


/****************************************************************************/
std::string Config::HelperDisplayBool(unsigned int val)
{
    std::stringstream ss;

    if (DCGM_INT32_IS_BLANK(val))
    {
        switch (val)
        {
            case DCGM_INT32_BLANK:
                ss << "Not Specified";
                break;

            case DCGM_INT32_NOT_FOUND:
                ss << "Not Found";
                break;

            case DCGM_INT32_NOT_SUPPORTED:
                ss << "Not Supported";
                break;

            case DCGM_INT32_NOT_PERMISSIONED:
                ss << "Insf. Permission";
                break;

            default:
                ss << "Unknown";
                break;
        }
    }
    else
    {
        if (0 == val)
        {
            ss << "Disabled";
        }
        else if (1 == val)
        {
            ss << "Enabled";
        }
        else
        {
            ss << "Error";
        }
    }

    return ss.str();
}


/*****************************************************************************
 *****************************************************************************
 * Set Configuration Invoker
 *****************************************************************************
 *****************************************************************************/

/*****************************************************************************/
SetConfig::SetConfig(std::string hostname, Config obj)
    : Command()
    , configObj(std::move(obj))
{
    m_hostName = std::move(hostname);

    /* We want group actions to persist once this DCGMI instance exits */
    SetPersistAfterDisconnect(1);
}

/*****************************************************************************/
dcgmReturn_t SetConfig::DoExecuteConnected()
{
    return configObj.RunSetConfig(m_dcgmHandle);
}

/*****************************************************************************
 *****************************************************************************
 * Get Configuration Invoker
 *****************************************************************************
 *****************************************************************************/

/*****************************************************************************/
GetConfig::GetConfig(std::string hostname, Config obj, bool verbose, bool json)
    : Command()
    , configObj(std::move(obj))
    , verbose(verbose)
{
    m_hostName = std::move(hostname);
    m_json     = json;

    /* We want group actions to persist once this DCGMI instance exits */
    SetPersistAfterDisconnect(1);
}

/*****************************************************************************/
dcgmReturn_t GetConfig::DoExecuteConnected()
{
    return configObj.RunGetConfig(m_dcgmHandle, verbose, m_json);
}

/*****************************************************************************
 *****************************************************************************
 * Enforce Configuration Invoker
 *****************************************************************************
 *****************************************************************************/

EnforceConfig::EnforceConfig(std::string hostname, Config obj)
    : Command()
    , configObj(std::move(obj))
{
    m_hostName = std::move(hostname);

    /* We want group actions to persist once this DCGMI instance exits */
    SetPersistAfterDisconnect(1);
}

dcgmReturn_t EnforceConfig::DoExecuteConnected()
{
    return configObj.RunEnforceConfig(m_dcgmHandle);
}
