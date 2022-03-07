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
 * An object storing basic information about a GPU discovered in the system.  This
 * includes both general matching information as well as information specific to
 * that device (e.g. UUID).  Gpu is a child of Device which is a rudementary
 * class meant to potentially describe other types of test endpoints
 * (like a Mellanox IB adapter) in the future.
 */
#ifndef _NVVS_NVVS_GPU_H
#define _NVVS_NVVS_GPU_H

#include "Device.h"
#include "dcgm_structs.h"
#include "dcgm_structs_internal.h"
#include <string>

typedef struct
{
    bool migEnabled              = false;
    bool migInvalidConfiguration = false;
} dcgmMigValidity_t;

// define logging mechanisms, meta data, etc.

class Gpu : public Device
{
    /***************************PUBLIC***********************************/
public:
    Gpu(unsigned int gpuId);
    ~Gpu();

    enum gpuEnumMethod_enum
    {
        NVVS_GPUENUM_NVML,
        NVVS_GPUENUM_LAST
    };

    dcgmReturn_t Init();

    // public getters
    std::string getDeviceBrandAsString()
    {
        return m_attributes.identifiers.brandName;
    }
    std::string getDeviceGpuUuid()
    {
        return m_attributes.identifiers.uuid;
    }
    std::string getDevicePciBusId()
    {
        return m_attributes.identifiers.pciBusId;
    }
    std::string getDevicePciDeviceId()
    {
        return m_pciDeviceId;
    }
    std::string getDevicePciSubsystemId()
    {
        return m_pciSubSystemId;
    }
    std::string getDeviceName()
    {
        return m_attributes.identifiers.deviceName;
    }

    std::string getDeviceId()
    {
        if (!m_useSsid)
        {
            return m_pciDeviceId;
        }

        return m_pciDeviceId + m_pciSubSystemId;
    }

    unsigned int getDeviceIndex(gpuEnumMethod_enum method = NVVS_GPUENUM_NVML) const;
    bool getDeviceIsSupported()
    {
        return m_isSupported;
    }
    uint64_t getDeviceArchitecture()
    {
        return m_gpuArch;
    }
    DcgmEntityStatus_t GetDeviceEntityStatus() const
    {
        return m_status;
    }
    unsigned int getMaxMemoryClock()
    {
        return m_maxMemoryClock;
    }

    void setDeviceIsSupported(bool value)
    {
        m_isSupported = value;
    }

    void setUseSsid(bool value)
    {
        m_useSsid = value;
    }

    void setDeviceEntityStatus(DcgmEntityStatus_t status)
    {
        m_status = status;
    }
    unsigned int GetEnforcedPowerLimit()
    {
        return m_attributes.powerLimits.enforcedPowerLimit;
    }

    dcgmDeviceThermals_t GetDeviceThermals() const
    {
        return m_attributes.thermalSettings;
    }
    uint64_t GetMaxOperatingTemperature() const
    {
        return m_maxGpuOpTemp;
    }
    unsigned int GetGpuId() const
    {
        return m_index;
    }

    bool PersistenceModeEnabled() const
    {
        return m_attributes.settings.persistenceModeEnabled;
    }

    dcgmDeviceAttributes_t GetAttributes() const
    {
        return m_attributes;
    }

    /*
     * Returns true if the MIG mode is compatible with running the diagnostic.
     * The MIG mode is compatible if it is disabled or if there is only 1
     * GPU instance configured.
     */
    dcgmMigValidity_t IsMigModeDiagCompatible() const;

    /***************************PRIVATE**********************************/
private:
    dcgmDeviceAttributes_t m_attributes;
    unsigned int m_index;
    bool m_isSupported;
    std::string m_pciDeviceId;
    std::string m_pciSubSystemId;
    uint64_t m_gpuArch;
    unsigned int m_maxMemoryClock; /* Maximum memory clock supported for the GPU. DCGM_FI_DEV_MAX_MEM_CLOCK */
    uint64_t m_maxGpuOpTemp;
    DcgmEntityStatus_t m_status;
    bool m_useSsid = false;

    void PopulateMaxMemoryClock(void);

    /***************************PROTECTED********************************/
protected:
};

#endif //_NVVS_NVVS_GPU_H
