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

#ifndef __PLUGINDEVICE_H__
#define __PLUGINDEVICE_H__

#include "DcgmError.h"
#include "NvvsDeviceList.h"
#include "cuda.h"
#include "cuda_runtime_api.h"

class PluginDevice
{
public:
    int cudaDeviceIdx;
    unsigned int gpuId;
    cudaDeviceProp cudaDevProp;
    NvvsDevice *nvvsDevice;
    std::string warning;
    std::string m_pciBusId;

    PluginDevice(unsigned int ndi, const char *pciBusId, Plugin *p)
        : cudaDeviceIdx(0)
        , gpuId(ndi)
        , nvvsDevice(0)
        , warning()
    {
        int st;
        char buf[256] = { 0 };
        cudaError_t cuSt;

        memset(&this->cudaDevProp, 0, sizeof(this->cudaDevProp));

        this->nvvsDevice = new NvvsDevice(p);
        st               = this->nvvsDevice->Init(this->gpuId);
        if (st)
        {
            snprintf(buf, sizeof(buf), "Couldn't initialize NvvsDevice for GPU %u", this->gpuId);
            throw(std::runtime_error(buf));
        }

        /* Resolve cuda device index from PCI bus id */
        if (pciBusId == nullptr)
        {
            DcgmError d { gpuId };
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_BAD_PARAMETER, d, "cudaDeviceGetByPCIBusId");
            warning = d.GetMessage();
            throw(d);
        }

        m_pciBusId = pciBusId;
        cuSt       = cudaDeviceGetByPCIBusId(&this->cudaDeviceIdx, pciBusId);
        if (cuSt != cudaSuccess)
        {
            DcgmError d { gpuId };
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_CUDA_API, d, "cudaDeviceGetByPCIBusId");
            snprintf(
                buf, sizeof(buf) - 1, "'%s' for GPU %u, bus ID = %s", cudaGetErrorString(cuSt), this->gpuId, pciBusId);
            d.AddDetail(buf);
            warning = d.GetMessage();
            throw(d);
        }

#if 0 /* Don't lock clocks. Otherwise, we couldn't call this for non-root and would have different behavior \
         between root and nonroot */
		/* Turn off auto boost if it is present on the card and enabled */
		st = this->nvvsDevice->DisableAutoBoostedClocks();
		if(st)
			return -1;

		/* Try to maximize application clocks */
		unsigned int maxMemoryClock = (unsigned int)bgGlobals->testParameters->GetDouble(PCIE_STR_MAX_MEMORY_CLOCK);
		unsigned int maxGraphicsClock = (unsigned int)bgGlobals->testParameters->GetDouble(PCIE_STR_MAX_GRAPHICS_CLOCK);

		st = this->nvvsDevice->SetMaxApplicationClocks(maxMemoryClock, maxGraphicsClock);
		if(st)
			return -1;
#endif
    }

    ~PluginDevice()
    {
        if (this->nvvsDevice)
        {
            try
            {
                this->nvvsDevice->RestoreState();
            }
            catch (std::exception &e)
            {
                DCGM_LOG_ERROR << "Caught exception in destructor. Swallowing " << e.what();
            }
            delete nvvsDevice;
            this->nvvsDevice = 0;
        }
    }
};

#endif // __PLUGINDEVICE_H__
