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

#include "NvvsDeviceList.h"
#include "NvvsCommon.h"
#include "timelib.h"
#include <DcgmGroup.h>
#include <DcgmHandle.h>
#include <DcgmSystem.h>

/*****************************************************************************/

extern DcgmHandle dcgmHandle;
extern DcgmSystem dcgmSystem;

/*****************************************************************************/
NvvsDevice::NvvsDevice(Plugin *plugin)
{
    /* Initialize to bad values */
    m_gpuId  = 0xFFFFFFFF;
    m_plugin = plugin;

    memset(&m_savedState, 0, sizeof(m_savedState));
}

/*****************************************************************************/
NvvsDevice::~NvvsDevice()
{}

/*****************************************************************************/
unsigned int NvvsDevice::GetGpuId()
{
    return m_gpuId;
}

/*****************************************************************************/
int NvvsDevice::SaveState(nvvs_device_state_t *savedState)
{
    unsigned int flags               = DCGM_FV_FLAG_LIVE_DATA; // Set the flag to get data without watching first
    dcgmFieldValue_v2 computeModeVal = {};

    if (savedState->populated)
    {
        PRINT_ERROR("", "Cannot save state - this object is already populated");
        return -1; /* This object is already populated */
    }

    memset(savedState, 0, sizeof(*savedState));
    savedState->populated = 1;

    /* Save compute mode */
    dcgmReturn_t ret = dcgmSystem.GetGpuLatestValue(m_gpuId, DCGM_FI_DEV_COMPUTE_MODE, flags, computeModeVal);

    if (ret != DCGM_ST_OK)
    {
        DcgmError d { m_gpuId };
        DCGM_ERROR_FORMAT_MESSAGE_DCGM(DCGM_FR_DCGM_API, d, ret, "dcgmEntitiesGetLatestValues");
        RecordWarning(d, 0);
    }
    else if (computeModeVal.status != DCGM_ST_OK)
    {
        savedState->computeModeSupported = 0;
        savedState->computeMode          = DCGM_CONFIG_COMPUTEMODE_DEFAULT;
    }
    else
    {
        savedState->computeMode          = computeModeVal.value.i64;
        savedState->computeModeSupported = 1;
    }

    return 0;
}

/*****************************************************************************/
int NvvsDevice::RestoreState(void)
{
    nvvs_device_state_t afterState;
    int NstatesRestored = 0; /* How many device states did we restore? */

    /* Record the current state of the device for comparison */
    afterState.populated = 0;
    SaveState(&afterState);

    /* Do the easy check to see if anything has changed */
    if (!memcmp(&m_savedState, &afterState, sizeof(m_savedState)))
    {
        /* Nothing to restore */
        return 0;
    }

    // Create a group
    dcgmConfig_v1 config;
    config.version                         = dcgmConfig_version1;
    config.gpuId                           = m_gpuId;
    config.eccMode                         = DCGM_INT32_BLANK;
    config.computeMode                     = m_savedState.computeMode;
    config.perfState.targetClocks.memClock = DCGM_INT32_BLANK;
    config.perfState.targetClocks.smClock  = DCGM_INT32_BLANK;
    config.powerLimit.val                  = DCGM_INT32_BLANK;

    /* Auto boost clocks */
    if (m_savedState.computeModeSupported && m_savedState.computeMode != afterState.computeMode)
    {
        DcgmGroup dg;
        std::stringstream groupName;
        groupName << "restore" << m_gpuId;
        dg.Init(dcgmHandle.GetHandle(), groupName.str());
        dcgmReturn_t ret = dg.AddGpu(m_gpuId);
        if (ret == DCGM_ST_OK)
        {
            ret = dcgmConfigSet(dcgmHandle.GetHandle(), dg.GetGroupId(), &config, 0);
        }

        if (ret != DCGM_ST_OK)
        {
            DcgmError d { m_gpuId };
            DCGM_ERROR_FORMAT_MESSAGE_DCGM(DCGM_FR_DCGM_API, d, ret, "dcgmConfigSet");
            RecordWarning(d, 0);
        }
        else
        {
            NstatesRestored++;
        }
    }

    if (NstatesRestored > 0)
        return 1;
    else
    {
        PRINT_ERROR("", "Unexpected no states restored. Should have been shortcut");
        return -1;
    }
}

/*****************************************************************************/
int NvvsDevice::Init(unsigned int gpuId)
{
    m_gpuId = gpuId;

    /* Save initial state */
    m_savedState.populated = 0;
    SaveState(&m_savedState);

    return 0;
}

/*****************************************************************************/
void NvvsDevice::RecordInfo(const char *logText)
{
    if (m_plugin)
    {
        m_plugin->AddInfo(logText);
    }
    else
    {
        PRINT_INFO("%s", "%s", logText);
    }
}

/*****************************************************************************/
void NvvsDevice::RecordWarning(const DcgmError &d, int failPlugin)
{
    if (m_plugin)
    {
        m_plugin->AddErrorForGpu(m_gpuId, d);
        if (failPlugin)
        {
            m_plugin->SetResultForGpu(m_gpuId, NVVS_RESULT_FAIL);
        }
    }
    else
    {
        PRINT_WARNING("%s", "%s", d.GetMessage().c_str());
        if (failPlugin)
        {
            throw std::runtime_error(d.GetMessage().c_str());
        }
    }
}

/*****************************************************************************/
/*****************************************************************************/
/*****************************************************************************/
NvvsDeviceList::NvvsDeviceList(Plugin *plugin)
{
    m_plugin = plugin;
}

/*****************************************************************************/
NvvsDeviceList::~NvvsDeviceList(void)
{
    std::vector<NvvsDevice *>::iterator it;
    NvvsDevice *device;

    /* Free all of the devices */
    for (it = m_devices.begin(); it != m_devices.end(); it++)
    {
        device = *it;

        delete (device);
    }

    m_devices.clear();
}

/*****************************************************************************/
int NvvsDeviceList::Init(std::vector<unsigned int> gpuIds)
{
    NvvsDevice *nvvsDevice;
    char buf[256];

    if (m_devices.size() > 0)
    {
        PRINT_ERROR("", "m_devices already initialized.");
        return -1;
    }

    for (size_t i = 0; i < gpuIds.size(); i++)
    {
        unsigned int gpuId = gpuIds[i];

        nvvsDevice = new NvvsDevice(m_plugin);
        int st     = nvvsDevice->Init(gpuId);
        if (st)
        {
            DcgmError d { gpuId };
            snprintf(buf, sizeof(buf), "Got error %d while initializing NvvsDevice index %u", st, gpuId);
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_INTERNAL, d, buf);
            RecordWarning(d, 1);
            delete (nvvsDevice);
            nvvsDevice = 0;
        }

        m_devices.push_back(nvvsDevice);
    }

    return 0;
}

/*****************************************************************************/
void NvvsDeviceList::RecordInfo(const char *logText)
{
    if (m_plugin)
        m_plugin->AddInfo(logText);
    else
        PRINT_INFO("%s", "%s", logText);
}

/*****************************************************************************/
void NvvsDeviceList::RecordWarning(const DcgmError &d, int failPlugin)
{
    if (m_plugin)
    {
        m_plugin->AddError(d);
        if (failPlugin)
        {
            m_plugin->SetResult(NVVS_RESULT_FAIL);
        }
    }
    else
    {
        PRINT_WARNING("%s", "%s", d.GetMessage().c_str());
        if (failPlugin)
        {
            throw std::runtime_error(d.GetMessage().c_str());
        }
    }
}

/*****************************************************************************/
int NvvsDeviceList::RestoreState(int failOnRestore)
{
    int i, st;
    NvvsDevice *nvvsDevice;
    std::vector<unsigned int> changedIndexes;

    for (i = 0; i < (int)m_devices.size(); i++)
    {
        nvvsDevice = m_devices[i];
        st         = nvvsDevice->RestoreState();
        if (!st)
            continue; /* Nothing changed. Great */
        else if (failOnRestore)
        {
            /* Save for later complaint. Restore rest of devices for now */
            changedIndexes.push_back(nvvsDevice->GetGpuId());
        }
    }

    if (changedIndexes.size() > 0 && failOnRestore)
    {
        std::stringstream ss;
        DcgmError d { DcgmError::GpuIdTag::Unknown };

        for (i = 0; i < (int)changedIndexes.size(); i++)
        {
            if (i > 0)
                ss << ",";
            ss << " " << changedIndexes[i];
        }

        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_HAD_TO_RESTORE_STATE, d, ss.str().c_str());

        RecordWarning(d, 1);
        return 1;
    }

    return 0;
}

/*****************************************************************************/
