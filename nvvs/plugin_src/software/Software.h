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
#ifndef _NVVS_NVVS_Software_H_
#define _NVVS_NVVS_Software_H_

#include "Gpu.h"
#include "Plugin.h"
#include "PluginStrings.h"
#include "TestParameters.h"
#include <DcgmRecorder.h>
#include <NvvsStructs.h>
#include <iostream>
#include <string>
#include <vector>

class Software : public Plugin
{
public:
    Software(dcgmHandle_t handle, dcgmDiagPluginGpuList_t *gpuInfo);
    ~Software()
    {
        delete tp;
    }

    void Go(unsigned int numParameters, const dcgmDiagPluginTestParameter_t *testParameters);
    void setArgs(std::string args)
    {
        myArgs = args;
    }

    /*
     * Returns true if we should count the specified entry in /dev as an nvidia device.
     *
     * @param entryName - the name of the entry we're evaluating.
     * @return true if the entry follows the pattern nvidia[0-9]+, false otherwise
     */
    bool CountDevEntry(const std::string &entryName);

private:
    enum libraryCheck_t
    {
        CHECK_NVML,   // NVML library
        CHECK_CUDA,   // CUDA library (installed as part of driver)
        CHECK_CUDATK, // CUDA toolkit libraries (blas, fft, etc.)
    };

    // variables
    std::string myArgs;
    TestParameters *tp;
    DcgmRecorder m_dcgmRecorder;
    DcgmSystem m_dcgmSystem;
    dcgmHandle_t m_handle;
    dcgmDiagPluginGpuList_t m_gpuInfo;

    // methods
    bool checkPermissions();
    bool checkLibraries(libraryCheck_t libs);
    bool checkBlacklist();
    bool findLib(std::string, std::string &error);
    int checkDriverPathBlacklist(std::string, std::vector<std::string> const &);
    int retrieveDeviceCount(unsigned int *count);
    int checkPersistenceMode();
    int checkForGraphicsProcesses();
    int checkForBadEnvVaribles();
    int checkPageRetirement();
    int checkRowRemapping();
    int checkInforom();
};


#endif // _NVVS_NVVS_Software_H_
