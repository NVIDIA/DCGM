/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
#include "NvmlHelpers.h"
#include "PluginInterface.h"
#include "dcgm_errors.h"
#include "dcgm_fields.h"
#include "dcgm_fields_internal.hpp"
#include <DcgmStringHelpers.h>
#include <assert.h>
#include <dirent.h>
#include <dlfcn.h>
#include <errno.h>
#include <fmt/format.h>
#include <iostream>
#include <stdexcept>
#include <string.h>
#include <unistd.h>

#ifdef __x86_64__
__asm__(".symver memcpy,memcpy@GLIBC_2.2.5");
#endif

#define DCGM_STRINGIFY_IMPL(x) #x
#define DCGM_STRINGIFY(x)      DCGM_STRINGIFY_IMPL(x)

#ifndef DCGM_NVML_SONAME
#ifndef DCGM_NVML_SOVERSION
#define DCGM_NVML_SOVERSION 1
#endif
#define DCGM_NVML_SONAME "libnvidia-ml.so." DCGM_STRINGIFY(DCGM_NVML_SOVERSION)
#else
#ifdef DCGM_NVML_SOVERSION
#pragma message DCGM_NVML_SONAME set explicitly; DCGM_NVML_SOVERSION ignored !
#endif
#endif

#ifndef DCGM_CUDA_SONAME
#ifndef DCGM_CUDA_SOVERSION
#define DCGM_CUDA_SOVERSION 1
#endif
#define DCGM_CUDA_SONAME "libcuda.so." DCGM_STRINGIFY(DCGM_CUDA_SOVERSION)
#else
#ifdef DCGM_CUDA_SOVERSION
#pragma message DCGM_CUDA_SONAME set explicitly; DCGM_CUDA_SOVERSION ignored !
#endif
#endif

#ifndef DCGM_CUDART_SONAME
#ifndef DCGM_CUDART_SOVERSION
#define DCGM_CUDART_SOVERSION 1
#endif
#define DCGM_CUDART_SONAME "libcudart.so." DCGM_STRINGIFY(DCGM_CUDART_SOVERSION)
#else
#ifdef DCGM_CUDART_SOVERSION
#pragma message DCGM_CUDART_SONAME set explicitly; DCGM_CUDART_SOVERSION ignored !
#endif
#endif

#ifndef DCGM_CUBLAS_SONAME
#ifndef DCGM_CUBLAS_SOVERSION
#define DCGM_CUBLAS_SOVERSION 1
#endif
#define DCGM_CUBLAS_SONAME "libcublas.so." DCGM_STRINGIFY(DCGM_CUBLAS_SOVERSION)
#else
#ifdef DCGM_CUBLAS_SOVERSION
#pragma message DCGM_CUBLAS_SONAME set explicitly; DCGM_CUBLAS_SOVERSION ignored !
#endif
#endif

const std::string SW_SUBTEST_DENYLIST("Denylist");
const std::string SW_SUBTEST_LIBNVML("NVML Library");
const std::string SW_SUBTEST_LIBCUDA("CUDA Main Library");
const std::string SW_SUBTEST_CUDATK("CUDA Toolkit Library");
const std::string SW_SUBTEST_PERMS("Permissions and OS Blocks");
const std::string SW_SUBTEST_PERSISTENCE("Persistence Mode");
const std::string SW_SUBTEST_ENV("Environment Variables");
const std::string SW_SUBTEST_PAGE_RETIREMENT("Page Retirement/Row Remap");
const std::string SW_SUBTEST_SRAM_THRESHOLD("SRAM Threshold Count");
const std::string SW_SUBTEST_GRAPHICS("Graphics Processes");
const std::string SW_SUBTEST_INFOROM("Inforom");
const std::string SW_SUBTEST_FABRIC_MANAGER("Fabric Manager");

Software::Software(dcgmHandle_t handle)
    : m_dcgmRecorder(handle)
    , m_dcgmSystem()
    , m_handle(handle)
    , m_entityInfo(std::make_unique<dcgmDiagPluginEntityList_v1>())
{
    m_infoStruct.testIndex        = DCGM_SOFTWARE_INDEX;
    m_infoStruct.shortDescription = "Software deployment checks plugin.";
    m_infoStruct.testCategories   = SW_PLUGIN_CATEGORY;
    m_infoStruct.selfParallel     = true;
    m_infoStruct.logFileTag       = SW_PLUGIN_NAME;

    tp.AddString(PS_RUN_IF_GOM_ENABLED, "True");
    tp.AddString(SW_STR_DO_TEST, "None");
    tp.AddString(SW_STR_REQUIRE_PERSISTENCE, "True");
    tp.AddString(SW_STR_SKIP_DEVICE_TEST, "False");
    tp.AddString(PS_IGNORE_ERROR_CODES, "");
    m_infoStruct.defaultTestParameters = &tp;

    m_dcgmSystem.Init(handle);
}

void Software::Go(std::string const &testName,
                  dcgmDiagPluginEntityList_v1 const *entityInfo,
                  unsigned int numParameters,
                  dcgmDiagPluginTestParameter_t const *tpStruct)
{
    if (testName != GetSoftwareTestName())
    {
        log_error("failed to test due to unknown test name [{}].", testName);
        return;
    }

    if (!entityInfo)
    {
        log_error("failed to test due to entityInfo is nullptr.");
        return;
    }

    InitializeForEntityList(testName, *entityInfo);

    if (UsingFakeGpus(testName))
    {
        log_error("Plugin is using fake gpus");
        SetResult(GetSoftwareTestName(), NVVS_RESULT_PASS);
        checkPageRetirement();
        checkRowRemapping();
        return;
    }
    TestParameters testParameters(*(m_infoStruct.defaultTestParameters));
    testParameters.SetFromStruct(numParameters, tpStruct);

    ParseIgnoreErrorCodesParam(testName, testParameters.GetString(PS_IGNORE_ERROR_CODES));
    m_dcgmRecorder.SetIgnoreErrorCodes(GetIgnoreErrorCodes(testName));

    if (testParameters.GetString(SW_STR_DO_TEST) == "denylist")
        checkDenylist();
    else if (testParameters.GetString(SW_STR_DO_TEST) == "permissions")
    {
        checkPermissions(testParameters.GetBoolFromString(SW_STR_CHECK_FILE_CREATION),
                         testParameters.GetBoolFromString(SW_STR_SKIP_DEVICE_TEST));
    }
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
            log_info("Skipping persistence check");
            SetResult(GetSoftwareTestName(), NVVS_RESULT_SKIP);
        }
        else
        {
            checkPersistenceMode(*entityInfo);
        }
    }
    else if (testParameters.GetString(SW_STR_DO_TEST) == "env_variables")
        checkForBadEnvVaribles();
    else if (testParameters.GetString(SW_STR_DO_TEST) == "sram_threshold")
        checkForSramThreshold();
    else if (testParameters.GetString(SW_STR_DO_TEST) == "graphics_processes")
        checkForGraphicsProcesses();
    else if (testParameters.GetString(SW_STR_DO_TEST) == "page_retirement")
    {
        checkPageRetirement();
        checkRowRemapping();
    }
    else if (testParameters.GetString(SW_STR_DO_TEST) == "inforom")
        checkInforom();
    else if (testParameters.GetString(SW_STR_DO_TEST) == "fabric_manager")
        checkFabricManager();
}

bool Software::CountDevEntry(std::string const &entryName)
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

bool Software::checkPermissions(bool checkFileCreation, bool skipDeviceTest)
{
    // check how many devices we see reporting and compare to
    // the number of devices listed in /dev
    unsigned int gpuCount    = 0;
    unsigned int deviceCount = 0;

    DIR *dir;
    struct dirent *ent;
    std::string dirName = "/dev";

    setSubtestName(SW_SUBTEST_PERMS);

    // Count the number of GPUs
    std::vector<unsigned int> gpuIds;
    dcgmReturn_t ret = m_dcgmSystem.GetAllDevices(gpuIds);
    if (DCGM_ST_OK != ret)
    {
        return false;
    }
    gpuCount = gpuIds.size();

    // everything below here is not necessarily a failure
    SetResult(GetSoftwareTestName(), NVVS_RESULT_PASS);
    if (skipDeviceTest == false)
    {
        dir = opendir(dirName.c_str());

        if (NULL == dir)
            return false;

        ent = readdir(dir);

        std::vector<DcgmError> accessWarnings;

        while (NULL != ent)
        {
            std::string entryName = ent->d_name;
            if (CountDevEntry(entryName))
            {
                std::stringstream ss;
                ss << dirName << "/" << entryName;
                if (access(ss.str().c_str(), R_OK) != 0)
                {
                    DcgmError d { DcgmError::GpuIdTag::Unknown };
                    DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_NO_ACCESS_TO_FILE, d, ss.str().c_str(), strerror(errno));
                    accessWarnings.emplace_back(d);
                }
                else
                {
                    deviceCount++;
                }
            }

            ent = readdir(dir);
        }
        closedir(dir);

        if (deviceCount < gpuCount)
        {
            DcgmError d { DcgmError::GpuIdTag::Unknown };
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_DEVICE_COUNT_MISMATCH, d);
            addError(d);
            for (auto &warning : accessWarnings)
            {
                addError(warning);
            }
            SetResult(GetSoftwareTestName(), NVVS_RESULT_WARN);
        }
    }

    if (checkFileCreation)
    {
        // Make sure we have the ability to create files in this directory
        if (euidaccess(".", W_OK))
        {
            char cwd[1024];
            char const *working_dir = getcwd(cwd, sizeof(cwd));

            DcgmError d { DcgmError::GpuIdTag::Unknown };
            d.SetCode(DCGM_FR_FILE_CREATE_PERMISSIONS);
            d.SetMessage(fmt::format("No permission to create a file in directory '{}'", working_dir));
            d.SetNextSteps(DCGM_FR_FILE_CREATE_PERMISSIONS_NEXT);
            addError(d);
            return false;
        }
    }

    return false;
}

// This function suggests time-of-check/time-of-use (TOCTOU) race conditions
bool Software::checkLibraries(libraryCheck_t checkLib)
{
    // Check whether the NVML, CUDA, or CUDA toolkit libraries can be found with sufficient permissions
    using span_t   = std::span<char const *const>;
    using result_t = nvvsPluginResult_t;

    auto const check = [this](span_t const libraries, span_t const diagnostics, result_t const failureCode) {
        bool failure = false;
        std::string error;

        for (char const *const library : libraries)
        {
            if (!findLib(library, error))
            {
                DcgmError d { DcgmError::GpuIdTag::Unknown };
                DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_CANNOT_OPEN_LIB, d, library, error.c_str());
                addError(d);
                SetResult(GetSoftwareTestName(), failureCode);
                failure = true;
            }
        }

        if (failure)
        {
            for (char const *const diagnostic : diagnostics)
            {
                AddInfo(GetSoftwareTestName(), diagnostic);
            }
        }

        return failure;
    };

    switch (checkLib)
    {
        case CHECK_NVML:
        {
            static constexpr char const *libraries[]   = { DCGM_NVML_SONAME };
            static constexpr char const *diagnostics[] = {
                "The NVML main library could not be found in the default search paths.",
                "Please check to see if it is installed or that LD_LIBRARY_PATH contains the path to " DCGM_NVML_SONAME,
                "Skipping remainder of tests."
            };

            setSubtestName(SW_SUBTEST_LIBNVML);
            return check(libraries, diagnostics, NVVS_RESULT_FAIL);
        }
        case CHECK_CUDA:
        {
            static constexpr char const *libraries[]   = { DCGM_CUDA_SONAME };
            static constexpr char const *diagnostics[] = { "The CUDA main library could not be found."
                                                           "Skipping remainder of tests." };

            setSubtestName(SW_SUBTEST_LIBCUDA);
            return check(libraries, diagnostics, NVVS_RESULT_WARN);
        }
        case CHECK_CUDATK:
        {
            static constexpr char const *libraries[] = { DCGM_CUDART_SONAME, DCGM_CUBLAS_SONAME };
            static constexpr char const *diagnostics[]
                = { "The CUDA Toolkit libraries could not be found.",
                    "Is LD_LIBRARY_PATH set to the 64-bit library path? (usually /usr/local/cuda/lib64)",
                    "Some tests will not run." };

            setSubtestName(SW_SUBTEST_CUDATK);
            return check(libraries, diagnostics, NVVS_RESULT_WARN);
        }
        default:
        {
            // should never get here
            DcgmError d { DcgmError::GpuIdTag::Unknown };
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_BAD_PARAMETER, d, __func__);
            addError(d);
            SetResult(GetSoftwareTestName(), NVVS_RESULT_FAIL);
            return true;
        }
    }
}

bool Software::checkDenylist()
{
    // check whether the nouveau driver is installed and if so, fail this test
    bool status = false;

    std::string const searchPaths[] = { "/sys/bus/pci/devices", "/sys/bus/pci_express/devices" };
    std::string const driverDirs[]  = { "driver", "subsystem/drivers" };

    std::vector<std::string> const denyList = { "nouveau" };

    setSubtestName(SW_SUBTEST_DENYLIST);

    for (unsigned long i = 0; i < sizeof(searchPaths) / sizeof(searchPaths[0]); ++i)
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
            for (unsigned long j = 0; j < sizeof(driverDirs) / sizeof(driverDirs[0]); ++j)
            {
                std::string baseDir = searchPaths[i];
                std::stringstream testPath;
                testPath << baseDir << "/" << ent->d_name << "/" << driverDirs[j];
                if (checkDriverPathDenylist(testPath.str(), denyList))
                {
                    SetResult(GetSoftwareTestName(), NVVS_RESULT_FAIL);
                    status = true;
                }
            }
            ent = readdir(dir);
        }
        closedir(dir);
    }
    if (!status)
        SetResult(GetSoftwareTestName(), NVVS_RESULT_PASS);
    return status;
}

int Software::checkDriverPathDenylist(std::string driverPath, std::vector<std::string> const &denyList)
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
        for (auto const &item : denyList)
        {
            if (strcmp(item.c_str(), basename(symlinkTarget)) == 0)
            {
                DcgmError d { DcgmError::GpuIdTag::Unknown };
                DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_DENYLISTED_DRIVER, d, item.c_str());
                addError(d);
                return 1;
            }
        }
    }

    return 0;
}

bool Software::findLib(std::string library, std::string &error)
{
    // On Linux, the search procedure considers
    // 1. (ELF binaries) the directories described by the binary RPATH (if the RUNPATH tag is absent)
    // 2. the directories described by the LD_LIBRARY_PATH environment variable
    // 3. (ELF binaries) the directories described by the binary RUNPATH (if the RUNPATH tag is present)
    // 4. the /etc/ld.so.cache
    // 5. the /lib directory
    // 6. the /usr/lib directory
    void *const handle = dlopen(library.c_str(), RTLD_NOW);
    if (!handle)
    {
        error = dlerror();
        return false;
    }
    dlclose(handle);
    return true;
}

int Software::checkForSramThreshold()
{
    unsigned int flags = DCGM_FV_FLAG_LIVE_DATA;
    dcgmFieldValue_v2 thresholdExceeded;

    setSubtestName(SW_SUBTEST_SRAM_THRESHOLD);

    auto const &gpuList = m_tests.at(GetSoftwareTestName()).GetGpuList();

    for (auto const gpuId : gpuList)
    {
        dcgmReturn_t ret
            = m_dcgmRecorder.GetCurrentFieldValue(gpuId, DCGM_FI_DEV_THRESHOLD_SRM, thresholdExceeded, flags);

        if (ret != DCGM_ST_OK)
        {
            DcgmError d { gpuId };
            DCGM_ERROR_FORMAT_MESSAGE_DCGM(DCGM_FR_FIELD_QUERY, d, ret, "sram_threshold", gpuId);
            addError(d);
            SetResult(GetSoftwareTestName(), NVVS_RESULT_FAIL);
            continue;
        }

        if (thresholdExceeded.status != DCGM_ST_OK || DCGM_INT64_IS_BLANK(thresholdExceeded.value.i64))
        {
            std::stringstream buf;
            buf << "Error getting the SRAM Threshold Count for GPU " << gpuId
                << ". Status = " << thresholdExceeded.status << ".";

            if (thresholdExceeded.status == DCGM_ST_OK)
            {
                buf << " Value = " << thresholdExceeded.value.i64 << ".";
            }

            buf << " SRAM Threshold checking was skipped.";

            std::string info(buf.str());
            DCGM_LOG_WARNING << info;
            AddInfoVerbose(GetSoftwareTestName(), info);
            continue;
        }
        else if (thresholdExceeded.value.i64 != 0)
        {
            DcgmError d { gpuId };
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_SRAM_THRESHOLD, d, gpuId, thresholdExceeded.value.i64);
            addError(d);
            SetResult(GetSoftwareTestName(), NVVS_RESULT_WARN);
        }
    }

    return 0;
}

int Software::checkForGraphicsProcesses()
{
    unsigned int flags = DCGM_FV_FLAG_LIVE_DATA;
    dcgmFieldValue_v2 graphicsPidsVal;

    setSubtestName(SW_SUBTEST_GRAPHICS);

    auto const &gpuList = m_tests.at(GetSoftwareTestName()).GetGpuList();

    for (auto const gpuId : gpuList)
    {
        dcgmReturn_t ret
            = m_dcgmRecorder.GetCurrentFieldValue(gpuId, DCGM_FI_DEV_GRAPHICS_PIDS, graphicsPidsVal, flags);

        if (ret != DCGM_ST_OK)
        {
            DcgmError d { gpuId };
            DCGM_ERROR_FORMAT_MESSAGE_DCGM(DCGM_FR_FIELD_QUERY, d, ret, "graphics_pids", gpuId);
            addError(d);
            SetResult(GetSoftwareTestName(), NVVS_RESULT_FAIL);
            continue;
        }

        if (graphicsPidsVal.status != DCGM_ST_OK)
        {
            std::stringstream buf;
            buf << "Error getting the graphics pids for GPU " << gpuId << ". Status = " << graphicsPidsVal.status
                << " skipping check.";
            std::string info(buf.str());
            DCGM_LOG_WARNING << info;
            AddInfo(GetSoftwareTestName(), info);
            continue;
        }
        else if (graphicsPidsVal.value.blob[0] != '\0')
        {
            // If there's any information here, it means a process is running
            DcgmError d { gpuId };
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_GRAPHICS_PROCESSES, d);
            addError(d);
            SetResult(GetSoftwareTestName(), NVVS_RESULT_WARN);
        }
    }

    return 0;
}

int Software::checkPersistenceMode(dcgmDiagPluginEntityList_v1 const &entityList)
{
    unsigned const numEntities = std::min(
        entityList.numEntities, static_cast<unsigned>(sizeof(entityList.entities) / sizeof(entityList.entities[0])));

    setSubtestName(SW_SUBTEST_PERSISTENCE);
    auto const &gpuList = m_tests.at(GetSoftwareTestName()).GetGpuList();

    for (auto const gpuId : gpuList)
    {
        for (unsigned int sindex = 0; sindex < numEntities; sindex++)
        {
            if (entityList.entities[sindex].entity.entityGroupId != DCGM_FE_GPU)
            {
                continue;
            }
            if (entityList.entities[sindex].entity.entityId != gpuId)
            {
                continue;
            }

            if (entityList.entities[sindex].auxField.gpu.attributes.settings.persistenceModeEnabled == false)
            {
                DcgmError d { gpuId };
                DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_PERSISTENCE_MODE, d, gpuId);
                addError(d);
                SetResult(GetSoftwareTestName(), NVVS_RESULT_WARN);
                break;
            }
        }
    }

    return 0;
}

int Software::checkPageRetirement()
{
    dcgmFieldValue_v2 pendingRetirementsFieldValue;
    dcgmFieldValue_v2 dbeFieldValue;
    dcgmFieldValue_v2 sbeFieldValue;
    dcgmReturn_t ret;
    int64_t retiredPagesTotal;

    /* Flags to pass to dcgmRecorder.GetCurrentFieldValue. Get live data since we're not watching the fields ahead of
     * time */
    unsigned int flags = DCGM_FV_FLAG_LIVE_DATA;

    setSubtestName(SW_SUBTEST_PAGE_RETIREMENT);

    if (UsingFakeGpus(GetSoftwareTestName()))
    {
        /* fake gpus don't support live data */
        flags = 0;
    }

    auto const &gpuList = m_tests.at(GetSoftwareTestName()).GetGpuList();
    for (auto const gpuId : gpuList)
    {
        // Check for pending page retirements
        ret = m_dcgmRecorder.GetCurrentFieldValue(
            gpuId, DCGM_FI_DEV_RETIRED_PENDING, pendingRetirementsFieldValue, flags);
        if (ret != DCGM_ST_OK)
        {
            DcgmError d { gpuId };
            DCGM_ERROR_FORMAT_MESSAGE_DCGM(DCGM_FR_FIELD_QUERY, d, ret, "retired_pages_pending", gpuId);
            addError(d);
            SetResult(GetSoftwareTestName(), NVVS_RESULT_FAIL);
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
                addError(d);
                SetResult(GetSoftwareTestName(), NVVS_RESULT_FAIL);
            }
            else
            {
                DcgmError d { gpuId };
                DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_PENDING_PAGE_RETIREMENTS, d, gpuId);
                addError(d);
                SetResult(GetSoftwareTestName(), NVVS_RESULT_FAIL);
            }
            /*
             * Halt nvvs for failures related to 'pending page retirements' or 'RETIRED_DBE/SBE'. Please be aware that
             * we will not stop for internal DCGM failures, such as issues with retrieving the current field value.
             */
            main_should_stop.store(1);
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
            addError(d);
            SetResult(GetSoftwareTestName(), NVVS_RESULT_FAIL);
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
            addError(d);
            SetResult(GetSoftwareTestName(), NVVS_RESULT_FAIL);
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
            addError(d);
            SetResult(GetSoftwareTestName(), NVVS_RESULT_FAIL);
            main_should_stop.store(1);
            continue;
        }
    }

    return 0;
}

int Software::checkRowRemapping()
{
    dcgmFieldValue_v2 pendingRowRemap;
    dcgmFieldValue_v2 rowRemapFailure;
    dcgmReturn_t ret;

    /* Flags to pass to dcgmRecorder.GetCurrentFieldValue. Get live data since we're not watching the fields ahead of
     * time */
    unsigned int flags = DCGM_FV_FLAG_LIVE_DATA;

    setSubtestName(SW_SUBTEST_PAGE_RETIREMENT);

    if (UsingFakeGpus(GetSoftwareTestName()))
    {
        /* fake gpus don't support live data */
        flags = 0;
    }

    auto const &gpuList = m_tests.at(GetSoftwareTestName()).GetGpuList();
    for (auto const gpuId : gpuList)
    {
        memset(&rowRemapFailure, 0, sizeof(rowRemapFailure));

        // Row remap failure
        ret = m_dcgmRecorder.GetCurrentFieldValue(gpuId, DCGM_FI_DEV_ROW_REMAP_FAILURE, rowRemapFailure, flags);
        if (ret != DCGM_ST_OK)
        {
            DcgmError d { gpuId };
            DCGM_ERROR_FORMAT_MESSAGE_DCGM(DCGM_FR_FIELD_QUERY, d, ret, "row_remap_failure", gpuId);
            addError(d);
            SetResultForGpu(GetSoftwareTestName(), gpuId, NVVS_RESULT_FAIL);
            continue;
        }

        dcgmGroupEntityPair_t entity = { DCGM_FE_GPU, gpuId };
        if (rowRemapFailure.status != DCGM_ST_OK || DCGM_INT64_IS_BLANK(rowRemapFailure.value.i64))
        {
            DCGM_LOG_INFO << "gpuId " << gpuId << " returned status " << rowRemapFailure.status << ", value "
                          << rowRemapFailure.value.i64 << "for DCGM_FI_DEV_ROW_REMAP_FAILURE. Skipping this check.";
        }
        else if (rowRemapFailure.value.i64 > 0)
        {
            DcgmError d { gpuId };
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_ROW_REMAP_FAILURE, d, gpuId);
            bool ignoreRowRemapError = ShouldIgnoreError(GetSoftwareTestName(), entity, DCGM_FR_ROW_REMAP_FAILURE);
            if (ignoreRowRemapError)
            {
                std::string infoStr
                    = fmt::format("{} {}: {}", SUPPRESSED_ERROR_STR, SW_SUBTEST_PAGE_RETIREMENT, d.GetMessage());
                AddInfoVerboseForEntity(GetSoftwareTestName(), entity, infoStr);
            }
            else
            {
                addError(d);
                SetResultForGpu(GetSoftwareTestName(), gpuId, NVVS_RESULT_FAIL);

                /*
                 * Halt nvvs for failures related to 'row remap/pending' or 'uncorrectable remapped row'. Please be
                 * aware that we will not stop for internal DCGM failures, such as issues with retrieving the current
                 * field value.
                 */
                main_should_stop.store(1);
                continue;
            }
        }

        memset(&pendingRowRemap, 0, sizeof(pendingRowRemap));

        // Check for pending row remappings
        ret = m_dcgmRecorder.GetCurrentFieldValue(gpuId, DCGM_FI_DEV_ROW_REMAP_PENDING, pendingRowRemap, flags);
        if (ret != DCGM_ST_OK)
        {
            DcgmError d { gpuId };
            DCGM_ERROR_FORMAT_MESSAGE_DCGM(DCGM_FR_FIELD_QUERY, d, ret, "row_remap_pending", gpuId);
            addError(d);
            SetResultForGpu(GetSoftwareTestName(), gpuId, NVVS_RESULT_FAIL);
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
                addError(d);
                SetResultForGpu(GetSoftwareTestName(), gpuId, NVVS_RESULT_FAIL);
            }
            else
            {
                DcgmError d { gpuId };
                DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_PENDING_ROW_REMAP, d, gpuId);
                addError(d);
                SetResultForGpu(GetSoftwareTestName(), gpuId, NVVS_RESULT_FAIL);
            }
            main_should_stop.store(1);
        }
    }

    return 0;
}

int Software::checkInforom()
{
    unsigned int flags = DCGM_FV_FLAG_LIVE_DATA;
    dcgmFieldValue_v2 inforomValidVal;

    setSubtestName(SW_SUBTEST_INFOROM);

    auto const &gpuList = m_tests.at(GetSoftwareTestName()).GetGpuList();

    for (auto const gpuId : gpuList)
    {
        dcgmReturn_t ret
            = m_dcgmRecorder.GetCurrentFieldValue(gpuId, DCGM_FI_DEV_INFOROM_CONFIG_VALID, inforomValidVal, flags);

        if (ret != DCGM_ST_OK)
        {
            DcgmError d { gpuId };
            DCGM_ERROR_FORMAT_MESSAGE_DCGM(DCGM_FR_FIELD_QUERY, d, ret, "inforom_config_valid", gpuId);
            addError(d);
            SetResult(GetSoftwareTestName(), NVVS_RESULT_FAIL);
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
            AddInfo(GetSoftwareTestName(), info);
            SetResult(GetSoftwareTestName(), NVVS_RESULT_SKIP);
            continue;
        }
        else if (inforomValidVal.status != DCGM_ST_OK)
        {
            std::stringstream buf;
            buf << "DCGM returned status " << inforomValidVal.status << " for GPU " << gpuId
                << " when checking the validity of the inforom. Skipping this check.";
            std::string info(buf.str());
            DCGM_LOG_WARNING << info;
            AddInfo(GetSoftwareTestName(), info);
            continue;
        }
        else if (inforomValidVal.value.i64 == 0)
        {
            // Inforom is not valid
            DcgmError d { gpuId };
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_CORRUPT_INFOROM, d, gpuId);
            addError(d);
            SetResult(GetSoftwareTestName(), NVVS_RESULT_FAIL);
            continue;
        }
    }

    return 0;
}

int Software::checkForBadEnvVaribles()
{
    std::vector<std::string> const checkKeys = { "NSIGHT_CUDA_DEBUGGER",
                                                 "CUDA_INJECTION32_PATH",
                                                 "CUDA_INJECTION64_PATH",
                                                 "CUDA_AUTO_BOOST",
                                                 "CUDA_ENABLE_COREDUMP_ON_EXCEPTION",
                                                 "CUDA_COREDUMP_FILE",
                                                 "CUDA_DEVICE_WAITS_ON_EXCEPTION",
                                                 "CUDA_PROFILE",
                                                 "COMPUTE_PROFILE",
                                                 "OPENCL_PROFILE" };

    setSubtestName(SW_SUBTEST_ENV);

    for (auto const &checkKey : checkKeys)
    {
        /* Does the variable exist in the environment? */
        if (getenv(checkKey.c_str()) == nullptr)
        {
            log_debug("Env Variable {} not found (GOOD)", checkKey.c_str());
            continue;
        }

        /* Variable found. Warn */
        DcgmError d { DcgmError::GpuIdTag::Unknown };
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_BAD_CUDA_ENV, d, checkKey.c_str());
        addError(d);
        SetResult(GetSoftwareTestName(), NVVS_RESULT_WARN);
    }

    return 0;
}

void Software::checkFabricManager()
{
    dcgmFieldValue_v2 fmStatusVal;
    unsigned int flags = DCGM_FV_FLAG_LIVE_DATA;

    setSubtestName(SW_SUBTEST_FABRIC_MANAGER);

    auto const &gpuList = m_tests.at(GetSoftwareTestName()).GetGpuList();

    for (auto gpuIt = gpuList.begin(); gpuIt != gpuList.end(); gpuIt++)
    {
        unsigned int gpuId = *gpuIt;

        dcgmReturn_t ret
            = m_dcgmRecorder.GetCurrentFieldValue(gpuId, DCGM_FI_DEV_FABRIC_MANAGER_STATUS, fmStatusVal, flags);

        if (ret != DCGM_ST_OK)
        {
            std::string couldntReadMsg
                = fmt::format("Couldn't read fabric manager status for GPU {} - attempting to run the tests.", gpuId);
            log_warning(couldntReadMsg);
            AddInfo(GetSoftwareTestName(), couldntReadMsg);
            continue;
        }

        if (fmStatusVal.value.i64 == DcgmFMStatusNotSupported)
        {
            log_debug("Fabric manager isn't supported for GPU {}", gpuId);
            SetResultForGpu(GetSoftwareTestName(), gpuId, NVVS_RESULT_PASS);
        }
        else if (fmStatusVal.value.i64 == DcgmFMStatusNvmlTooOld)
        {
            log_debug("Nvml is too old in this system. Skip FabricManager checking.");
            SetResult(GetSoftwareTestName(), NVVS_RESULT_SKIP);
            break;
        }
        else if (fmStatusVal.value.i64 == DcgmFMStatusSuccess)
        {
            log_debug("Fabric manager successfully started for GPU {}", gpuId);
            SetResultForGpu(GetSoftwareTestName(), gpuId, NVVS_RESULT_PASS);
        }
        else
        {
            LogAndSetFMError(gpuId, (dcgmFabricManagerStatus_t)fmStatusVal.value.i64);
        }
    }
}

void Software::LogAndSetFMError(unsigned int gpuId, dcgmFabricManagerStatus_t status)
{
    static unsigned int const INVALID_CLIQUE_ID = UINT_MAX;
    std::string errorMessage;
    dcgmFieldValue_v2 fmUuidVal;
    dcgmFieldValue_v2 fmCliqueIdVal;
    dcgmReturn_t ret;
    unsigned int flags    = DCGM_FV_FLAG_LIVE_DATA;
    unsigned int cliqueId = INVALID_CLIQUE_ID;

    switch (status)
    {
        case DcgmFMStatusNotStarted:
            errorMessage = "training not started";
            break;

        case DcgmFMStatusInProgress:
            errorMessage = "training in progress";
            break;

        case DcgmFMStatusFailure:
        {
            dcgmFieldValue_v2 fmErrorVal;
            ret = m_dcgmRecorder.GetCurrentFieldValue(gpuId, DCGM_FI_DEV_FABRIC_MANAGER_ERROR_CODE, fmErrorVal, flags);
            if (ret == DCGM_ST_OK)
            {
                errorMessage = fmt::format("Training completed with an error: {}",
                                           errorString((dcgmReturn_t)fmErrorVal.value.i64));
            }
            else
            {
                errorMessage = fmt::format("Training completed with an error, but failed to read the error code: {}",
                                           errorString(ret));
            }
            break;
        }

        default:
        {
            errorMessage = "Receiving an unknown fabric manager state from NVML";
        }
    }

    ret = m_dcgmRecorder.GetCurrentFieldValue(gpuId, DCGM_FI_DEV_FABRIC_CLUSTER_UUID, fmUuidVal, flags);
    if (ret != DCGM_ST_OK)
    {
        log_debug("Couldn't read FM cluster uuid: {}", errorString(ret));
        SafeCopyTo(fmUuidVal.value.str, "unknown uuid");
    }

    ret = m_dcgmRecorder.GetCurrentFieldValue(gpuId, DCGM_FI_DEV_FABRIC_CLIQUE_ID, fmCliqueIdVal, flags);
    if (ret != DCGM_ST_OK)
    {
        log_debug("Couldn't read FM clique id: {}", errorString(ret));
    }
    else
    {
        cliqueId = fmCliqueIdVal.value.i64;
    }

    DcgmError d { DcgmError::GpuIdTag::Unknown };
    DCGM_ERROR_FORMAT_MESSAGE(
        DCGM_FR_FABRIC_MANAGER_TRAINING_ERROR, d, fmUuidVal.value.str, cliqueId, errorMessage.c_str());
    addError(d);
    SetResult(GetSoftwareTestName(), NVVS_RESULT_FAIL);
}

std::string Software::GetSoftwareTestName() const
{
    return SW_PLUGIN_NAME;
}

std::string Software::GetSoftwareTestCategory() const
{
    return m_infoStruct.testCategories;
}

/**
 * Sets the current subtest name to specified `name`.
 */
void Software::setSubtestName(std::string const &name)
{
    m_subtestName = name;
}

/**
 * Adds the specified DcgmError `d` with any previously set `subtestName` as a prefix.
 */
void Software::addError(DcgmError &d)
{
    if (m_subtestName.empty())
    {
        AddError(GetSoftwareTestName(), d);
    }
    else
    {
        std::string msg = fmt::format("{}: {}", m_subtestName, d.GetMessage());

        /* Clear NextSteps and Detail, which are included from GetMessage()
           to prevent repeating them when FullMessage is reconstructed below. */
        d.SetNextSteps("");
        d.AddDetail("");

        d.SetMessage(msg);
        AddError(GetSoftwareTestName(), d);
    }
}

std::vector<dcgmDiagPluginParameterInfo_t> GetSwParameterInfo()
{
    std::vector<dcgmDiagPluginParameterInfo_t> ret;
    std::array<std::string, 2> parameterNames { SW_STR_REQUIRE_PERSISTENCE, SW_STR_SKIP_DEVICE_TEST };

    for (std::string const &name : parameterNames)
    {
        dcgmDiagPluginParameterInfo_t param {};
        SafeCopyTo(param.parameterName, name.c_str());
        param.parameterType = DcgmPluginParamBool;
        ret.emplace_back(param);
    }

    return ret;
}
