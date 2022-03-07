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
#include "Software.h"
#include "DcgmError.h"
#include "DcgmGPUHardwareLimits.h"
#include "DcgmLogging.h"
#include "dcgm_errors.h"
#include <assert.h>
#include <dirent.h>
#include <dlfcn.h>
#include <errno.h>
#include <iostream>
#include <stdexcept>
#include <string.h>
#include <unistd.h>

#ifdef __x86_64__
__asm__(".symver memcpy,memcpy@GLIBC_2.2.5");
#endif

Software::Software(dcgmHandle_t handle, dcgmDiagPluginGpuList_t *gpuInfo)
    : m_dcgmRecorder(handle)
    , m_dcgmSystem()
    , m_handle(handle)
    , m_gpuInfo()
{
    m_infoStruct.testIndex        = DCGM_SOFTWARE_INDEX;
    m_infoStruct.shortDescription = "Software deployment checks plugin.";
    m_infoStruct.testGroups       = "Software";
    m_infoStruct.selfParallel     = true;
    m_infoStruct.logFileTag       = SW_PLUGIN_NAME;

    tp = new TestParameters();
    tp->AddString(PS_RUN_IF_GOM_ENABLED, "True");
    tp->AddString(SW_STR_DO_TEST, "None");
    tp->AddString(SW_STR_REQUIRE_PERSISTENCE, "True");
    m_infoStruct.defaultTestParameters = tp;

    if (gpuInfo == nullptr)
    {
        DcgmError d { DcgmError::GpuIdTag::Unknown };
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_INTERNAL, d, "No GPU information specified");
        AddError(d);
    }
    else
    {
        m_gpuInfo = *gpuInfo;
        InitializeForGpuList(*gpuInfo);
    }

    m_dcgmSystem.Init(handle);
}

void Software::Go(unsigned int numParameters, const dcgmDiagPluginTestParameter_t *tpStruct)
{
    if (UsingFakeGpus())
    {
        PRINT_ERROR("%s", "Plugin is using fake gpus");
        SetResult(NVVS_RESULT_PASS);
        checkPageRetirement();
        return;
    }

    TestParameters testParameters(*(m_infoStruct.defaultTestParameters));
    testParameters.SetFromStruct(numParameters, tpStruct);

    if (testParameters.GetString(SW_STR_DO_TEST) == "blacklist")
        checkBlacklist();
    else if (testParameters.GetString(SW_STR_DO_TEST) == "permissions")
        checkPermissions();
    else if (testParameters.GetString(SW_STR_DO_TEST) == "libraries_nvml")
        checkLibraries(CHECK_NVML);
    else if (testParameters.GetString(SW_STR_DO_TEST) == "libraries_cuda")
        checkLibraries(CHECK_CUDA);
    else if (testParameters.GetString(SW_STR_DO_TEST) == "libraries_cudatk")
        checkLibraries(CHECK_CUDATK);
    else if (testParameters.GetString(SW_STR_DO_TEST) == "persistence_mode")
    {
        int shouldCheckPersistence = testParameters.GetBoolFromString(SW_STR_REQUIRE_PERSISTENCE);

        if (!shouldCheckPersistence)
        {
            PRINT_INFO("", "Skipping persistence check");
            SetResult(NVVS_RESULT_SKIP);
        }
        else
        {
            checkPersistenceMode();
        }
    }
    else if (testParameters.GetString(SW_STR_DO_TEST) == "env_variables")
        checkForBadEnvVaribles();
    else if (testParameters.GetString(SW_STR_DO_TEST) == "graphics_processes")
        checkForGraphicsProcesses();
    else if (testParameters.GetString(SW_STR_DO_TEST) == "page_retirement")
    {
        checkPageRetirement();
        checkRowRemapping();
    }
    else if (testParameters.GetString(SW_STR_DO_TEST) == "inforom")
        checkInforom();
}

bool Software::CountDevEntry(const std::string &entryName)
{
    if (entryName.compare(0, 6, "nvidia") == 0)
    {
        for (size_t i = 6; i < entryName.size(); i++)
        {
            if (!isdigit(entryName.at(i)))
            {
                return false;
            }
        }
        return true;
    }

    return false;
}

bool Software::checkPermissions()
{
    // check how many devices we see reporting and compare to
    // the number of devices listed in /dev
    unsigned int gpuCount    = 0;
    unsigned int deviceCount = 0;

    DIR *dir;
    struct dirent *ent;
    std::string dirName = "/dev";

    // Count the number of GPUs
    std::vector<unsigned int> gpuIds;
    dcgmReturn_t ret = m_dcgmSystem.GetAllDevices(gpuIds);
    if (DCGM_ST_OK != ret)
    {
        return false;
    }
    gpuCount = gpuIds.size();

    // everything below here is not necessarily a failure
    SetResult(NVVS_RESULT_PASS);
    dir = opendir(dirName.c_str());

    if (NULL == dir)
        return false;

    ent = readdir(dir);
    while (NULL != ent)
    {
        std::string entryName = ent->d_name;
        if (CountDevEntry(entryName))
        {
            deviceCount++;
            std::stringstream ss;
            ss << dirName << "/" << entryName;
            if (access(ss.str().c_str(), R_OK) != 0)
            {
                DcgmError d { DcgmError::GpuIdTag::Unknown };
                DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_NO_ACCESS_TO_FILE, d, ss.str().c_str(), strerror(errno));
                AddError(d);
                SetResult(NVVS_RESULT_WARN);
            }
        }

        ent = readdir(dir);
    }
    closedir(dir);

    if (deviceCount != gpuCount)
    {
        DcgmError d { DcgmError::GpuIdTag::Unknown };
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_DEVICE_COUNT_MISMATCH, d);
        AddError(d);
        SetResult(NVVS_RESULT_WARN);
    }
    return false;
}

bool Software::checkLibraries(libraryCheck_t checkLib)
{
    // check whether the NVML, CUDA, and CUDA toolkit libraries can be found
    // via default paths
    bool fail = false;
    std::vector<std::string> libs;

    switch (checkLib)
    {
        case CHECK_NVML:
            libs.push_back("libnvidia-ml.so.1");
            break;
        case CHECK_CUDA:
            libs.push_back("libcuda.so");
            break;
        case CHECK_CUDATK:
            libs.push_back("libcudart.so");
            libs.push_back("libcublas.so");
            break;
        default:
        {
            // should never get here
            DcgmError d { DcgmError::GpuIdTag::Unknown };
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_BAD_PARAMETER, d, __func__);
            AddError(d);
            SetResult(NVVS_RESULT_FAIL);
        }
    }

    for (std::vector<std::string>::iterator it = libs.begin(); it != libs.end(); it++)
    {
        std::string error;
        if (!findLib(*it, error))
        {
            DcgmError d { DcgmError::GpuIdTag::Unknown };
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_CANNOT_OPEN_LIB, d, it->c_str(), error.c_str());
            AddError(d);
            if (checkLib != CHECK_CUDATK)
            {
                SetResult(NVVS_RESULT_FAIL);
            }
            else
            {
                SetResult(NVVS_RESULT_WARN);
            }

            fail = true;
        }
    }

    // The statements that follow are all classified as info statements because only messages directly tied
    // to failures are errors.
    if (checkLib == CHECK_CUDATK && fail == true)
    {
        AddInfo("The CUDA Toolkit libraries could not be found.");
        AddInfo("Is LD_LIBRARY_PATH set to the 64-bit library path? (usually /usr/local/cuda/lib64)");
        AddInfo("Some tests will not run.");
    }
    if (checkLib == CHECK_CUDA && fail == true)
    {
        AddInfo("The CUDA main library could not be found.");
        AddInfo("Skipping remainder of tests.");
    }
    if (checkLib == CHECK_NVML && fail == true)
    {
        AddInfo("The NVML main library could not be found in the default search paths.");
        AddInfo(
            "Please check to see if it is installed or that LD_LIBRARY_PATH contains the path to libnvidia-ml.so.1.");
        AddInfo("Skipping remainder of tests.");
    }
    return fail;
}

bool Software::checkBlacklist()
{
    // check whether the nouveau driver is installed and if so, fail this test
    bool status = false;

    const std::string searchPaths[] = { "/sys/bus/pci/devices", "/sys/bus/pci_express/devices" };
    const std::string driverDirs[]  = { "driver", "subsystem/drivers" };

    const std::vector<std::string> blackList = { "nouveau" };

    for (int i = 0; i < sizeof(searchPaths) / sizeof(searchPaths[0]); i++)
    {
        DIR *dir;
        struct dirent *ent;

        dir = opendir(searchPaths[i].c_str());

        if (NULL == dir)
            continue;

        ent = readdir(dir);
        while (NULL != ent)
        {
            if ((strcmp(ent->d_name, ".") == 0) || (strcmp(ent->d_name, "..") == 0))
            {
                ent = readdir(dir);
                continue;
            }
            for (int j = 0; j < sizeof(driverDirs) / sizeof(driverDirs[0]); j++)
            {
                std::string baseDir = searchPaths[i];
                std::stringstream testPath;
                testPath << baseDir << "/" << ent->d_name << "/" << driverDirs[j];
                if (checkDriverPathBlacklist(testPath.str(), blackList))
                {
                    SetResult(NVVS_RESULT_FAIL);
                    status = true;
                }
            }
            ent = readdir(dir);
        }
        closedir(dir);
    }
    if (!status)
        SetResult(NVVS_RESULT_PASS);
    return status;
}

int Software::checkDriverPathBlacklist(std::string driverPath, std::vector<std::string> const &blackList)
{
    int ret;
    char symlinkTarget[1024];
    ret = readlink(driverPath.c_str(), symlinkTarget, sizeof(symlinkTarget));
    if (ret >= (signed int)sizeof(symlinkTarget))
    {
        assert(0);
        return ENAMETOOLONG;
    }
    else if (ret < 0)
    {
        int errorno = errno;

        switch (errorno)
        {
            case ENOENT:
                // driverPath does not exist, ignore it
                // this driver doesn't use this path format
                return 0;
            case EINVAL: // not a symlink
                return 0;

            case EACCES:
            case ENOTDIR:
            case ELOOP:
            case ENAMETOOLONG:
            case EIO:
            default:
                // Something bad happened
                return errorno;
        }
    }
    else
    {
        symlinkTarget[ret] = '\0'; // readlink doesn't null terminate
        for (auto const &item : blackList)
        {
            if (strcmp(item.c_str(), basename(symlinkTarget)) == 0)
            {
                DcgmError d { DcgmError::GpuIdTag::Unknown };
                DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_BLACKLISTED_DRIVER, d, item.c_str());
                AddError(d);
                return 1;
            }
        }
    }

    return 0;
}

bool Software::findLib(std::string library, std::string &error)
{
    void *handle;
    handle = dlopen(library.c_str(), RTLD_NOW);
    if (!handle)
    {
        error = dlerror();
        return false;
    }
    dlclose(handle);
    return true;
}

int Software::checkForGraphicsProcesses()
{
    std::vector<unsigned int>::const_iterator gpuIt;
    unsigned int gpuId;
    unsigned int flags = DCGM_FV_FLAG_LIVE_DATA;
    dcgmFieldValue_v2 graphicsPidsVal;

    for (gpuIt = m_gpuList.begin(); gpuIt != m_gpuList.end(); gpuIt++)
    {
        gpuId = *gpuIt;

        dcgmReturn_t ret
            = m_dcgmRecorder.GetCurrentFieldValue(gpuId, DCGM_FI_DEV_GRAPHICS_PIDS, graphicsPidsVal, flags);

        if (ret != DCGM_ST_OK)
        {
            DcgmError d { gpuId };
            DCGM_ERROR_FORMAT_MESSAGE_DCGM(DCGM_FR_FIELD_QUERY, d, ret, "graphics_pids", gpuId);
            AddError(d);
            SetResult(NVVS_RESULT_FAIL);
            continue;
        }

        if (graphicsPidsVal.status != DCGM_ST_OK)
        {
            std::stringstream buf;
            buf << "Error getting the graphics pids for GPU " << gpuId << ". Status = " << graphicsPidsVal.status
                << " skipping check.";
            std::string info(buf.str());
            DCGM_LOG_WARNING << info;
            AddInfo(info);
            continue;
        }
        else if (graphicsPidsVal.value.blob[0] != '\0')
        {
            // If there's any information here, it means a process is running
            DcgmError d { gpuId };
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_GRAPHICS_PROCESSES, d);
            AddError(d);
            SetResult(NVVS_RESULT_WARN);
        }
    }

    return 0;
}

int Software::checkPersistenceMode()
{
    unsigned int gpuId;
    std::vector<unsigned int>::const_iterator gpuIt;

    for (gpuIt = m_gpuList.begin(); gpuIt != m_gpuList.end(); gpuIt++)
    {
        gpuId = *gpuIt;

        for (unsigned int sindex = 0; sindex < m_gpuInfo.numGpus; sindex++)
        {
            if (m_gpuInfo.gpus[sindex].gpuId != gpuId)
            {
                continue;
            }

            if (m_gpuInfo.gpus[sindex].attributes.settings.persistenceModeEnabled == false)
            {
                DcgmError d { gpuId };
                DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_PERSISTENCE_MODE, d, gpuId);
                AddError(d);
                SetResult(NVVS_RESULT_FAIL);
                break;
            }
        }
    }

    return 0;
}

int Software::checkPageRetirement()
{
    unsigned int gpuId;
    std::vector<unsigned int>::const_iterator gpuIt;
    dcgmFieldValue_v2 pendingRetirementsFieldValue;
    dcgmFieldValue_v2 dbeFieldValue;
    dcgmFieldValue_v2 sbeFieldValue;
    dcgmReturn_t ret;
    int64_t retiredPagesTotal;

    /* Flags to pass to dcgmRecorder.GetCurrentFieldValue. Get live data since we're not watching the fields ahead of
     * time */
    unsigned int flags = DCGM_FV_FLAG_LIVE_DATA;

    if (UsingFakeGpus())
    {
        /* fake gpus don't support live data */
        flags = 0;
    }

    for (gpuIt = m_gpuList.begin(); gpuIt != m_gpuList.end(); gpuIt++)
    {
        gpuId = *gpuIt;
        // Check for pending page retirements
        ret = m_dcgmRecorder.GetCurrentFieldValue(
            gpuId, DCGM_FI_DEV_RETIRED_PENDING, pendingRetirementsFieldValue, flags);
        if (ret != DCGM_ST_OK)
        {
            DcgmError d { gpuId };
            DCGM_ERROR_FORMAT_MESSAGE_DCGM(DCGM_FR_FIELD_QUERY, d, ret, "retired_pages_pending", gpuId);
            AddError(d);
            SetResult(NVVS_RESULT_FAIL);
            continue;
        }

        if (pendingRetirementsFieldValue.status != DCGM_ST_OK
            || DCGM_INT64_IS_BLANK(pendingRetirementsFieldValue.value.i64))
        {
            DCGM_LOG_WARNING << "gpuId " << gpuId << " returned status " << pendingRetirementsFieldValue.status
                             << ", value " << pendingRetirementsFieldValue.value.i64
                             << "for DCGM_FI_DEV_RETIRED_PENDING. Skipping this check.";
        }
        else if (pendingRetirementsFieldValue.value.i64 > 0)
        {
            dcgmFieldValue_v2 volDbeVal = {};
            ret = m_dcgmRecorder.GetCurrentFieldValue(gpuId, DCGM_FI_DEV_ECC_DBE_VOL_TOTAL, volDbeVal, flags);
            if (ret == DCGM_ST_OK && (volDbeVal.value.i64 > 0 && !DCGM_INT64_IS_BLANK(volDbeVal.value.i64)))
            {
                DcgmError d { gpuId };
                DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_DBE_PENDING_PAGE_RETIREMENTS, d, gpuId);
                AddError(d);
                SetResult(NVVS_RESULT_FAIL);
            }
            else
            {
                DcgmError d { gpuId };
                DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_PENDING_PAGE_RETIREMENTS, d, gpuId);
                AddError(d);
                SetResult(NVVS_RESULT_FAIL);
            }
            continue;
        }

        // Check total page retirement count
        retiredPagesTotal = 0;

        // DBE retired pages
        ret = m_dcgmRecorder.GetCurrentFieldValue(gpuId, DCGM_FI_DEV_RETIRED_DBE, dbeFieldValue, flags);
        if (ret != DCGM_ST_OK)
        {
            DcgmError d { gpuId };
            DCGM_ERROR_FORMAT_MESSAGE_DCGM(DCGM_FR_FIELD_QUERY, d, ret, "retired_pages_dbe", gpuId);
            AddError(d);
            SetResult(NVVS_RESULT_FAIL);
            continue;
        }

        if (dbeFieldValue.status != DCGM_ST_OK || DCGM_INT64_IS_BLANK(dbeFieldValue.value.i64))
        {
            DCGM_LOG_WARNING << "gpuId " << gpuId << " returned status " << dbeFieldValue.status << ", value "
                             << dbeFieldValue.value.i64 << "for DCGM_FI_DEV_RETIRED_DBE. Skipping this check.";
        }
        else
        {
            retiredPagesTotal += dbeFieldValue.value.i64;
        }

        // SBE retired pages
        ret = m_dcgmRecorder.GetCurrentFieldValue(gpuId, DCGM_FI_DEV_RETIRED_SBE, sbeFieldValue, flags);
        if (ret != DCGM_ST_OK)
        {
            DcgmError d { gpuId };
            DCGM_ERROR_FORMAT_MESSAGE_DCGM(DCGM_FR_FIELD_QUERY, d, ret, "retired_pages_sbe", gpuId);
            AddError(d);
            SetResult(NVVS_RESULT_FAIL);
            continue;
        }

        if (sbeFieldValue.status != DCGM_ST_OK || DCGM_INT64_IS_BLANK(sbeFieldValue.value.i64))
        {
            DCGM_LOG_WARNING << "gpuId " << gpuId << " returned status " << sbeFieldValue.status << ", value "
                             << sbeFieldValue.value.i64 << "for DCGM_FI_DEV_RETIRED_SBE. Skipping this check.";
        }
        else
        {
            retiredPagesTotal += sbeFieldValue.value.i64;
        }

        if (retiredPagesTotal >= DCGM_LIMIT_MAX_RETIRED_PAGES)
        {
            DcgmError d { gpuId };
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_RETIRED_PAGES_LIMIT, d, DCGM_LIMIT_MAX_RETIRED_PAGES, gpuId);
            AddError(d);
            SetResult(NVVS_RESULT_FAIL);
            continue;
        }
    }

    return 0;
}

int Software::checkRowRemapping()
{
    unsigned int gpuId;
    std::vector<unsigned int>::const_iterator gpuIt;
    dcgmFieldValue_v2 pendingRowRemap;
    dcgmFieldValue_v2 rowRemapFailure;
    dcgmReturn_t ret;

    /* Flags to pass to dcgmRecorder.GetCurrentFieldValue. Get live data since we're not watching the fields ahead of
     * time */
    unsigned int flags = DCGM_FV_FLAG_LIVE_DATA;

    if (UsingFakeGpus())
    {
        /* fake gpus don't support live data */
        flags = 0;
    }

    for (gpuIt = m_gpuList.begin(); gpuIt != m_gpuList.end(); gpuIt++)
    {
        gpuId = *gpuIt;

        memset(&rowRemapFailure, 0, sizeof(rowRemapFailure));

        // Row remap failure
        ret = m_dcgmRecorder.GetCurrentFieldValue(gpuId, DCGM_FI_DEV_ROW_REMAP_FAILURE, rowRemapFailure, flags);
        if (ret != DCGM_ST_OK)
        {
            DcgmError d { gpuId };
            DCGM_ERROR_FORMAT_MESSAGE_DCGM(DCGM_FR_FIELD_QUERY, d, ret, "row_remap_failure", gpuId);
            AddError(d);
            SetResultForGpu(gpuId, NVVS_RESULT_FAIL);
            continue;
        }

        if (rowRemapFailure.status != DCGM_ST_OK || DCGM_INT64_IS_BLANK(rowRemapFailure.value.i64))
        {
            DCGM_LOG_INFO << "gpuId " << gpuId << " returned status " << rowRemapFailure.status << ", value "
                          << rowRemapFailure.value.i64 << "for DCGM_FI_DEV_ROW_REMAP_FAILURE. Skipping this check.";
        }
        else if (rowRemapFailure.value.i64 > 0)
        {
            DcgmError d { gpuId };
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_ROW_REMAP_FAILURE, d, gpuId);
            AddError(d);
            SetResultForGpu(gpuId, NVVS_RESULT_FAIL);

            continue;
        }

        memset(&pendingRowRemap, 0, sizeof(pendingRowRemap));

        // Check for pending row remappings
        ret = m_dcgmRecorder.GetCurrentFieldValue(gpuId, DCGM_FI_DEV_ROW_REMAP_PENDING, pendingRowRemap, flags);
        if (ret != DCGM_ST_OK)
        {
            DcgmError d { gpuId };
            DCGM_ERROR_FORMAT_MESSAGE_DCGM(DCGM_FR_FIELD_QUERY, d, ret, "row_remap_pending", gpuId);
            AddError(d);
            SetResultForGpu(gpuId, NVVS_RESULT_FAIL);
            continue;
        }

        if (pendingRowRemap.status != DCGM_ST_OK || DCGM_INT64_IS_BLANK(pendingRowRemap.value.i64))
        {
            DCGM_LOG_INFO << "gpuId " << gpuId << " returned status " << pendingRowRemap.status << ", value "
                          << pendingRowRemap.value.i64 << "for DCGM_FI_DEV_ROW_REMAP_PENDING. Skipping this check.";
        }
        else if (pendingRowRemap.value.i64 > 0)
        {
            dcgmFieldValue_v2 uncRemap = {};
            ret = m_dcgmRecorder.GetCurrentFieldValue(gpuId, DCGM_FI_DEV_UNCORRECTABLE_REMAPPED_ROWS, uncRemap, flags);
            if (ret == DCGM_ST_OK && (uncRemap.value.i64 > 0 && !DCGM_INT64_IS_BLANK(uncRemap.value.i64)))
            {
                DcgmError d { gpuId };
                DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_UNCORRECTABLE_ROW_REMAP, d, gpuId);
                AddError(d);
                SetResultForGpu(gpuId, NVVS_RESULT_FAIL);
            }
            else
            {
                DcgmError d { gpuId };
                DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_PENDING_ROW_REMAP, d, gpuId);
                AddError(d);
                SetResultForGpu(gpuId, NVVS_RESULT_FAIL);
            }
        }
    }

    return 0;
}

int Software::checkInforom()
{
    std::vector<unsigned int>::const_iterator gpuIt;
    unsigned int flags = DCGM_FV_FLAG_LIVE_DATA;
    dcgmFieldValue_v2 inforomValidVal;

    for (gpuIt = m_gpuList.begin(); gpuIt != m_gpuList.end(); gpuIt++)
    {
        unsigned int gpuId = *gpuIt;

        dcgmReturn_t ret
            = m_dcgmRecorder.GetCurrentFieldValue(gpuId, DCGM_FI_DEV_INFOROM_CONFIG_VALID, inforomValidVal, flags);

        if (ret != DCGM_ST_OK)
        {
            DcgmError d { gpuId };
            DCGM_ERROR_FORMAT_MESSAGE_DCGM(DCGM_FR_FIELD_QUERY, d, ret, "inforom_config_valid", gpuId);
            AddError(d);
            SetResult(NVVS_RESULT_FAIL);
            continue;
        }

        if ((inforomValidVal.status == DCGM_ST_NOT_SUPPORTED)
            || (inforomValidVal.status == DCGM_ST_OK && DCGM_INT64_IS_BLANK(inforomValidVal.value.i64)))
        {
            std::stringstream buf;
            buf << "DCGM returned status " << inforomValidVal.status << " for GPU " << gpuId
                << " when checking the validity of the inforom. Skipping this check.";
            std::string info(buf.str());
            DCGM_LOG_WARNING << info;
            AddInfo(info);
            SetResult(NVVS_RESULT_SKIP);
            continue;
        }
        else if (inforomValidVal.status != DCGM_ST_OK)
        {
            std::stringstream buf;
            buf << "DCGM returned status " << inforomValidVal.status << " for GPU " << gpuId
                << " when checking the validity of the inforom. Skipping this check.";
            std::string info(buf.str());
            DCGM_LOG_WARNING << info;
            AddInfo(info);
            continue;
        }
        else if (inforomValidVal.value.i64 == 0)
        {
            // Inforom is not valid
            DcgmError d { gpuId };
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_CORRUPT_INFOROM, d, gpuId);
            AddError(d);
            SetResult(NVVS_RESULT_FAIL);
            continue;
        }
    }

    return 0;
}

int Software::checkForBadEnvVaribles()
{
    std::vector<std::string> checkKeys;
    std::vector<std::string>::iterator checkKeysIt;
    std::string checkKey;

    /* Env variables to look for */
    checkKeys.push_back(std::string("NSIGHT_CUDA_DEBUGGER"));
    checkKeys.push_back(std::string("CUDA_INJECTION32_PATH"));
    checkKeys.push_back(std::string("CUDA_INJECTION64_PATH"));
    checkKeys.push_back(std::string("CUDA_AUTO_BOOST"));
    checkKeys.push_back(std::string("CUDA_ENABLE_COREDUMP_ON_EXCEPTION"));
    checkKeys.push_back(std::string("CUDA_COREDUMP_FILE"));
    checkKeys.push_back(std::string("CUDA_DEVICE_WAITS_ON_EXCEPTION"));
    checkKeys.push_back(std::string("CUDA_PROFILE"));
    checkKeys.push_back(std::string("COMPUTE_PROFILE"));
    checkKeys.push_back(std::string("OPENCL_PROFILE"));

    for (checkKeysIt = checkKeys.begin(); checkKeysIt != checkKeys.end(); checkKeysIt++)
    {
        checkKey = *checkKeysIt;

        /* Does the variable exist in the environment? */
        if (getenv(checkKey.c_str()) == nullptr)
        {
            PRINT_DEBUG("%s", "Env Variable %s not found (GOOD)", checkKey.c_str());
            continue;
        }

        /* Variable found. Warn */
        DcgmError d { DcgmError::GpuIdTag::Unknown };
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_BAD_CUDA_ENV, d, checkKey.c_str());
        AddError(d);
        SetResult(NVVS_RESULT_WARN);
    }

    return 0;
}
