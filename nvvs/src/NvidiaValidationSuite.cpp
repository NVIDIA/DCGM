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

#include "NvidiaValidationSuite.h"
#include "DcgmHandle.h"
#include "DcgmLogging.h"
#include "DcgmSystem.h"
#include "EntitySet.h"
#include "IgnoreErrorCodesHelper.h"
#include "NvvsJsonStrings.h"
#include "ParameterValidator.h"
#include "ParsingUtility.h"
#include "PluginStrings.h"
#include "dcgm_fields.h"
#include "dcgm_structs_internal.h"
#include <DcgmBuildInfo.hpp>
#include <DcgmStringHelpers.h>
#include <DcgmUtilities.h>
#include <NvvsException.hpp>
#include <PluginInterface.h>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <setjmp.h>
#include <signal.h>
#include <sstream>
#include <stdexcept>
#include <tclap/CmdLine.h>
#include <time.h>
#include <vector>

using namespace DcgmNs::Nvvs;

DcgmHandle dcgmHandle;
DcgmSystem dcgmSystem;
NvvsCommon nvvsCommon __attribute__((visibility("default")));
bool initTimedOut = false;
jmp_buf exitInitialization;

/*****************************************************************************/
NvidiaValidationSuite::NvidiaValidationSuite()
    : logInit(false)
    , m_gpuVect()
    , testVect()
    , tpVect()
    , m_allowlist(nullptr)
    , fwcfg()
    , parser(nullptr)
    , m_tf(nullptr)
    , configFile()
    , debugFile(NVVS_LOGGING_DEFAULT_NVVS_LOGFILE)
    , debugLogLevel("")
    , hwDiagLogFile()
    , listTests(false)
    , listGpus(false)
    , initTimer(0)
    , restoreSigAction {}
    , initWaitTime(120)
    , m_pv()
{
    // init globals
    nvvsCommon.Init();
}

/*****************************************************************************/
NvidiaValidationSuite::~NvidiaValidationSuite()
{
    for (std::vector<Gpu *>::iterator it = m_gpuVect.begin(); it != m_gpuVect.end(); ++it)
    {
        delete (*it);
    }

    for (std::vector<TestParameters *>::iterator it = tpVect.begin(); it != tpVect.end(); ++it)
    {
        delete (*it);
    }
    delete m_tf;
    delete parser;

    dcgmShutdown();
}

/*****************************************************************************/
void NvidiaValidationSuite::CheckDriverVersion()
{
    if (!dcgmSystem.IsInitialized())
    {
        log_error("DCGM is not initialized");
        throw std::runtime_error("DCGM is not initialized");
    }

    dcgmDeviceAttributes_t deviceAttr;

    memset(&deviceAttr, 0, sizeof(deviceAttr));
    deviceAttr.version = dcgmDeviceAttributes_version3;
    dcgmReturn_t ret   = dcgmSystem.GetDeviceAttributes(0, deviceAttr);
    unsigned int count = 0;
    std::stringstream additionalMsg;

    // Attempt re-connecting if we have trouble with our initial interaction with the hostengine.
    for (count = 0; count < 3 && ret == DCGM_ST_CONNECTION_NOT_VALID; ++count)
    {
        dcgmReturn_t connectionRet = dcgmHandle.ConnectToDcgm(nvvsCommon.dcgmHostname);
        if (connectionRet == DCGM_ST_OK)
        {
            ret = dcgmSystem.GetDeviceAttributes(0, deviceAttr);
        }
        else
        {
            additionalMsg.str("");
            additionalMsg << "Couldn't re-connect to hostengine after establishing an invalid connection: "
                          << dcgmHandle.RetToString(connectionRet);
        }
    }

    std::stringstream buf;
    std::string s_version;
    unsigned int i_version = 0;
    std::stringstream ss;

    if (ret != DCGM_ST_OK)
    {
        buf << "Unable to get the driver version: " << dcgmHandle.RetToString(ret);
        if (additionalMsg.rdbuf()->in_avail() == 0)
        {
            buf << ". Couldn't succeed despite " << count << " retries.";
        }
        else
        {
            buf << ". " << additionalMsg.str();
        }
        throw std::runtime_error(buf.str());
    }

    s_version = deviceAttr.identifiers.driverVersion;
    s_version = s_version.substr(0, s_version.find("."));
    ss << s_version;

    ss >> i_version;

    if (i_version < MIN_MAJOR_VERSION || i_version > MAX_MAJOR_VERSION)
    {
        std::stringstream exceptionSs;
        exceptionSs << "Detected driver major version " << i_version << " is not between the required versions "
                    << MIN_MAJOR_VERSION << " and " << MAX_MAJOR_VERSION << ".";
        log_error("Cannot run on incompatible driver version '{}'", deviceAttr.identifiers.driverVersion);
        throw std::runtime_error(exceptionSs.str());
    }
}

// Handler for SIGALRM as part of ensuring we can't stall indefinitely on DCGM init
void alarm_handler(int /* sig */)
{
    initTimedOut = true;
    longjmp(exitInitialization, 1);
}

/*
 * startTimer - Starts a timer to make sure we don't get stuck in the DCGM initialization and or loading
 * the CUDA library
 */
void NvidiaValidationSuite::startTimer()
{
    struct itimerspec value;
    struct sigaction act;

    memset(&value, 0, sizeof(value));
    memset(&act, 0, sizeof(act));

    // Set SIGALRM to use alarm_handler
    act.sa_handler = alarm_handler;
    sigemptyset(&act.sa_mask);
    sigaction(SIGALRM, &act, &this->restoreSigAction);

    // Set the timer for 20 seconds
    value.it_value.tv_sec = this->initWaitTime;
    timer_create(CLOCK_REALTIME, NULL, &this->initTimer);
    timer_settime(this->initTimer, 0, &value, NULL);
}

/*
 * stopTimer - disables the timer previously set to make sure we don't get stuck in the DCGM initialization
 * and or loading the CUDA library
 */
void NvidiaValidationSuite::stopTimer()
{
    struct itimerspec value;
    memset(&value, 0, sizeof(value));
    timer_settime(this->initTimer, 0, &value, NULL);

    // Restore any previous signal handler
    sigaction(SIGALRM, &this->restoreSigAction, NULL);
}

bool NvidiaValidationSuite::IsGpuIncluded(unsigned int gpuIndex, std::vector<unsigned int> &gpuIndices) const
{
    if (gpuIndices.size() == 0)
    {
        return true;
    }

    for (size_t i = 0; i < gpuIndices.size(); i++)
    {
        if (gpuIndex == gpuIndices[i])
        {
            return true;
        }
    }

    return false;
}

std::string NvidiaValidationSuite::BuildCommonGpusList(std::vector<unsigned int> &gpuIndices,
                                                       const std::vector<Gpu *> &visibleGpus)
{
    std::string errorStr;
    bool isMigModeEnabled   = false;
    bool anyMigModeDisabled = false;
    unsigned int numGpus    = gpuIndices.size();
    std::vector<unsigned int> originalList(gpuIndices);
    if (numGpus == 0)
    {
        numGpus = visibleGpus.size();
    }

    for (size_t i = 0; i < visibleGpus.size(); i++)
    {
        dcgmMigValidity_t mv = visibleGpus[i]->IsMigModeDiagCompatible();
        if (mv.migInvalidConfiguration == true)
        {
            std::stringstream buf;
            buf << "GPU " << visibleGpus[i]->GetGpuId()
                << "'s MIG configuration is incompatible with the diagnostic because it prevents access to the entire GPU.";
            DCGM_LOG_ERROR << buf.str();
            errorStr = buf.str();
            return errorStr;
        }

        anyMigModeDisabled = !mv.migEnabled || anyMigModeDisabled;

        isMigModeEnabled = isMigModeEnabled || mv.migEnabled;
        if (isMigModeEnabled && numGpus > 1)
        {
            // CUDA does not support enumerating GPUs with MIG mode enabled
            std::stringstream buf;
            buf << "Cannot run diagnostic: CUDA does not support enumerating GPUs with MIG mode enabled, yet GPU "
                << visibleGpus[i]->GetGpuId() << " is MIG enabled and the diagnostic is attempting to run on "
                << numGpus << " GPUs.";
            DCGM_LOG_ERROR << buf.str();
            errorStr = buf.str();
            return errorStr;
        }

        if (IsGpuIncluded(visibleGpus[i]->GetGpuId(), originalList) == false)
        {
            continue;
        }

        if (mv.migEnabled && mv.computeInstanceCount == 0)
        {
            return "Cannot run diagnostic: MIG is enabled, but no compute instances are configured."
                   " CUDA needs to execute on a compute instance if MIG is enabled.";
        }

        if (originalList.empty())
        {
            // If our originally specified list was empty, record the GPUs being used
            gpuIndices.push_back(visibleGpus[i]->getDeviceIndex());
        }
    }

    if (anyMigModeDisabled && isMigModeEnabled)
    {
        std::stringstream buf;
        buf << "Cannot run diagnostic: CUDA does not support enumerating GPUs when one more GPUs has MIG mode enabled "
            << "and one or more GPUs has MIG mode disabled.";
        errorStr = buf.str();
        DCGM_LOG_ERROR << errorStr;
        return errorStr;
    }

    if (isMigModeEnabled && !gpuIndices.empty())
    {
        // We know there's at most 1 GPU because we fail above if there's more than one
        std::stringstream val;
        val << *gpuIndices.begin();
        errno = 0;
        if (setenv("CUDA_VISIBLE_DEVICES", val.str().c_str(), 1))
        {
            std::stringstream buf;
            buf << "Cannot run diagnostic: failed to set the environment variable CUDA_VISIBLE_DEVICES: "
                << strerror(errno);
            DCGM_LOG_ERROR << buf.str();
            errorStr = buf.str();
            return errorStr;
        }
    }

    return errorStr;
}

/*****************************************************************************/
std::string NvidiaValidationSuite::Go(int argc, char *argv[])
{
    std::vector<unsigned int> gpuIndices;

    processCommandLine(argc, argv);
    if (nvvsCommon.quietMode)
    {
        std::cout.setstate(std::ios::failbit);
    }
    banner();

    DcgmLoggingInit(debugFile.c_str(),
                    LoggingSeverityFromString(debugLogLevel.c_str(), DcgmLoggingSeverityDebug),
                    DcgmLoggingSeverityNone);
    RouteLogToBaseLogger(SYSLOG_LOGGER);
    log_info("Initialized NVVS logger, version: {}", DcgmNs::DcgmBuildInfo().GetVersion());
    log_debug("Build info: {}", DcgmNs::DcgmBuildInfo().GetBuildInfoStr());

    logInit = true;
    {
        std::ostringstream out;
        for (int i = 0; i < argc; i++)
        {
            out << argv[i] << " ";
        }
        DCGM_LOG_DEBUG << "argc: " << argc << ". argv: " << out.str();
    }


    parser = new ConfigFileParser_v2("", fwcfg);

    /*
    startTimer();

    // Mark this as the point to return too if DCGM's init takes too long
    setjmp(exitInitialization);

    // If initTimedOut is set, then we've timed out while trying to initialize DCGM and load the cuda library
    if (initTimedOut == true)
    {
        std::stringstream buf;
        buf << "We reached the " << this->initWaitTime << " second timeout while attempting to initialize DCGM";
        buf << " and load the CUDA library.";
        buf << "\nPlease check why these systems are unresponsive.\n";

        return buf.str();
    }
    */

    dcgmReturn_t ret = dcgmHandle.ConnectToDcgm(nvvsCommon.dcgmHostname);
    if (ret != DCGM_ST_OK)
    {
        std::stringstream buf;
        buf << "Unable to connect to DCGM: " << dcgmHandle.GetLastError();
        return buf.str();
    }
    dcgmSystem.Init(dcgmHandle.GetHandle());

    /*
    stopTimer();
    */

    if (configFile.size() > 0)
        parser->setConfigFile(configFile);
    if (!parser->Init() && !nvvsCommon.configless)
    {
        std::ostringstream out;
        out << std::endl << "Unable to open config file ";
        if (configFile.size() > 0)
        {
            out << configFile << ", please check path and try again." << std::endl;
            return out.str();
        }
        else
        {
            // If they didn't specify a config file, just warn
            out << "/etc/datacenter-gpu-manager-4/nvvs.conf. " << std::endl;
            out << "Please check the path or specify a config file via the \"-c\" command line option." << std::endl;
            out << "Add --configless to suppress this warning." << std::endl;
            log_warning(out.str());

            // Force to true if we couldn't open a config file
            nvvsCommon.configless = true;
        }
    }

    parser->PrepareEntitySets(dcgmHandle.GetHandle());
    parser->legacyGlobalStructHelper();

    m_allowlist                                         = std::make_unique<Allowlist>(*parser);
    std::vector<std::unique_ptr<EntitySet>> &entitySets = parser->GetEntitySets();
    bool hasGpuEntity                                   = false;

    for (size_t i = 0; i < entitySets.size(); i++)
    {
        if (entitySets[i]->GetEntityGroup() != DCGM_FE_GPU)
        {
            continue;
        }

        if (!hasGpuEntity)
        {
            CheckDriverVersion();
            EnumerateAllVisibleGpus();
        }
        hasGpuEntity   = true;
        GpuSet *gpuSet = ToGpuSet(entitySets[i].get());
        if (gpuSet == nullptr)
        {
            log_error("failed to covert entity set to gpu set.");
            continue;
        }
        for (size_t j = 0; j < gpuSet->GetProperties().index.size(); j++)
        {
            gpuIndices.push_back(gpuSet->GetProperties().index[j]);
        }
        InitializeAndCheckGpuObjs(gpuSet);
    }

    if (listGpus)
    {
        std::cout << "Supported GPUs available:" << std::endl;

        for (std::vector<Gpu *>::iterator it = m_gpuVect.begin(); it != m_gpuVect.end(); ++it)
        {
            std::cout << "\t" << "[" << (*it)->getDevicePciBusId() << "] -- " << (*it)->getDeviceName() << std::endl;
        }
        std::cout << std::endl;
        return "";
    }

    if (entitySets.empty())
    {
        log_error("No available testing entities.");
        throw std::runtime_error("Error: No available testing entities.");
    }

    std::string errorString = BuildCommonGpusList(gpuIndices, m_gpuVect);
    if (errorString.empty() == false)
    {
        return errorString;
    }

    errorString
        = ParseIgnoreErrorCodesString(nvvsCommon.ignoreErrorCodesString, nvvsCommon.parsedIgnoreErrorCodes, gpuIndices);
    if (!errorString.empty())
    {
        return errorString;
    }

    m_tf = new TestFramework(entitySets);
    m_tf->loadPlugins();
    if (auto ret = m_tf->SetDiagResponseVersion(nvvsCommon.diagResponseVersion); ret != DCGM_ST_OK)
    {
        std::string const errMsg
            = fmt::format("failed to set version [{}], err: [{}].", nvvsCommon.diagResponseVersion, ret);
        log_error(errMsg);
        std::cerr << errMsg << std::endl;
        return errMsg;
    }

    ValidateSubtestParameters();

    enumerateAllVisibleTests();
    if (listTests)
    {
        std::cout << "Tests available:" << std::endl;

        for (std::vector<Test *>::iterator it = testVect.begin(); it != testVect.end(); ++it)
        {
            if (it + 1 != testVect.end()) // last object is a "skip" object that does not need to be displayed
                std::cout << "\t" << (*it)->GetTestName() << " -- " << (*it)->getTestDesc() << std::endl;
        }
        std::cout << std::endl;
        return "";
    }

    DistributeTests(entitySets);

    // Execute the tests... let the TF catch all exceptions and decide
    // whether to throw them higher.
    m_tf->Go(entitySets);

    return "";
}

/*****************************************************************************/
void NvidiaValidationSuite::banner()
{
    std::cout << std::endl
              << NVVS_NAME << " (version " << std::string(DcgmNs::DcgmBuildInfo().GetVersion()) << ")\n"
              << std::endl;
}

/*****************************************************************************/
void NvidiaValidationSuite::ValidateSubtestParameters()
{
    auto parms = m_tf->GetSubtestParameters();

    m_pv.Initialize(parms);

    InitializeParameters(nvvsCommon.parmsString, m_pv);
}

/*****************************************************************************/
void NvidiaValidationSuite::enumerateAllVisibleTests()
{
    // for now just use the testVec stored in the Framework
    // but eventually obfuscate this some

    testVect = m_tf->getTests();
}

bool NvidiaValidationSuite::HasGenericSupport(const std::string &gpuBrand, uint64_t gpuArch)
{
    static const unsigned int MAJOR_MAXWELL_COMPAT = 5;
    static const unsigned int MAJOR_KEPLER_COMPAT  = 3;

    if ((DCGM_CUDA_COMPUTE_CAPABILITY_MAJOR(gpuArch) >= MAJOR_MAXWELL_COMPAT)
        || (gpuBrand == "Tesla" && DCGM_CUDA_COMPUTE_CAPABILITY_MAJOR(gpuArch) >= MAJOR_KEPLER_COMPAT))
        return true;

    return false;
}

/*****************************************************************************/
void NvidiaValidationSuite::EnumerateAllVisibleGpus()
{
    if (!dcgmSystem.IsInitialized())
    {
        log_error("DCGM is not initialized");
        throw std::runtime_error("DCGM is not initialized");
    }

    bool isAllowlisted;
    std::vector<unsigned int> gpuIds;
    dcgmReturn_t ret = dcgmSystem.GetAllSupportedDevices(gpuIds);
    std::stringstream buf;

    if (ret != DCGM_ST_OK)
    {
        buf << "Unable to retrieve device count: " << dcgmHandle.RetToString(ret);
        throw std::runtime_error(buf.str());
    }

    for (size_t i = 0; i < gpuIds.size(); i++)
    {
        std::unique_ptr<Gpu> gpu { new Gpu(gpuIds[i]) };
        if ((ret = gpu->Init()) != DCGM_ST_OK)
        {
            buf << "Unable to initialize GPU " << gpuIds[i] << ": " << dcgmHandle.RetToString(ret);
            throw std::runtime_error(buf.str());
        }

        /* Find out if this device is supported, which is any of the following:
           1. On the NVVS allowlist explicitly
             1.a Search for specific "ID + SSID" first
             1.b Search for generic "ID" second
           2. A Kepler or newer Tesla part
           3. a Maxwell or newer part of any other brand (Quadro, GeForce, Titan, Grid)
        */

        if (m_allowlist->IsAllowlisted(gpu->getDevicePciDeviceId(), gpu->getDevicePciSubsystemId()))
        {
            isAllowlisted = true;
            gpu->setUseSsid(true);
        }
        else if (m_allowlist->IsAllowlisted(gpu->getDevicePciDeviceId()))
        {
            isAllowlisted = true;
        }
        else
        {
            isAllowlisted = false;
        }
        std::string gpuBrand = gpu->getDeviceBrandAsString();
        uint64_t gpuArch     = gpu->getDeviceArchitecture();

        if (!nvvsCommon.fakegpusString.empty())
        {
            log_debug("attempting to use fake gpus: {}", nvvsCommon.fakegpusString.c_str());
            DcgmEntityStatus_t status;
            dcgmSystem.GetGpuStatus(gpuIds[i], &status);
            gpu->setDeviceEntityStatus(status);

            log_debug("status of gpu {} is {}", gpuIds[i], status);
            /* TODO: check return */
            /* How to determine if gpu is fake? */
            if (status == DcgmEntityStatusFake)
            {
                log_debug("dcgmIndex {}, brand {}, arch {} is supported (only supporting fake gpus)",
                          gpuIds[i],
                          gpuBrand.c_str(),
                          static_cast<unsigned int>(gpuArch));
                gpu->setDeviceIsSupported(true);
            }
#ifdef INJECTION_LIBRARY_AVAILABLE
            // Fake NVML injection GPUs need this else if block.
            else if (status == DcgmEntityStatusOk)
            {
                DCGM_LOG_DEBUG << "dcgmIndex " << gpuIds[i] << ", brand " << gpuBrand << " arch " << gpuArch
                               << " is supported.";
                gpu->setDeviceIsSupported(true);
            }
#endif
            else
            {
                log_debug("dcgmIndex {}, brand {}, arch {} is not supported (only supporting fake gpus)",
                          gpuIds[i],
                          gpuBrand.c_str(),
                          static_cast<unsigned int>(gpuArch));
            }
        }
        else if (isAllowlisted)
        {
            log_debug("dcgmIndex {} is directly on the allowlist.", gpuIds[i]);
            gpu->setDeviceIsSupported(true);
        }
        else if (HasGenericSupport(gpuBrand, gpuArch))
        {
            log_debug("dcgmIndex {}, brand {}, arch {} is supported",
                      gpuIds[i],
                      gpuBrand.c_str(),
                      static_cast<unsigned int>(gpuArch));
            gpu->setDeviceIsSupported(true);
        }
        else
        {
            log_debug("dcgmIndex {}, brand {}, arch {} is NOT supported",
                      gpuIds[i],
                      gpuBrand.c_str(),
                      static_cast<unsigned int>(gpuArch));
            gpu->setDeviceIsSupported(false);
        }

        if (gpu->getDeviceIsSupported())
        {
            DCGM_LOG_INFO << "Device " << gpuIds[i] << ", serial " << gpu->getDeviceSerial()
                          << ", added to supported list";
            m_gpuVect.push_back(gpu.release());
        }
        else
        {
            log_info("\t[{}] -- {} -- Not Supported", gpu->getDevicePciBusId(), gpu->getDeviceName());
        }
    }

    /* Allow the allowlist to adjust itself now that GPUs have been read in */
    m_allowlist->PostProcessAllowlist(m_gpuVect);
}

/*****************************************************************************/
void NvidiaValidationSuite::overrideParameters(TestParameters *tp, const std::string &lowerCaseTestName)
{
    OverwriteTestParamtersIfAny(tp, lowerCaseTestName, nvvsCommon.parms);
}

void NvidiaValidationSuite::InitializeAndCheckGpuObjs(GpuSet *gpuSet)
{
    if (!gpuSet->GetProperties().present)
    {
        gpuSet->SetGpuObjs(m_gpuVect);
    }
    else
    {
        gpuSet->SetGpuObjs(decipherProperties(gpuSet));
    }

    if (gpuSet->GetGpuObjs().empty())
    { // nothing matched
        std::ostringstream ss;
        ss << "Unable to match GPU set '" << gpuSet->GetName() << "' to any GPU(s) on the system.";
        log_error(ss.str());
        throw std::runtime_error(ss.str());
    }

    // ensure homogeneity
    std::string firstName = gpuSet->GetGpuObjs()[0]->getDeviceName();
    for (auto gpuIt = gpuSet->GetGpuObjs().cbegin(); gpuIt != gpuSet->GetGpuObjs().cend(); gpuIt++)
    {
        // no need to check the first but...
        if (firstName != (*gpuIt)->getDeviceName())
        {
            std::ostringstream ss;
            ss << "NVVS does not support running on non-homogeneous GPUs during a single run: " << firstName
               << " != " << (*gpuIt)->getDeviceName() << ".";
            ss << "Please use the -i option to specify a list of identical GPUs. ";
            ss << "Run nvvs -g to list the GPUs on the system. Run nvvs --help for additional usage info. ";
            log_error(ss.str());
            throw std::runtime_error(ss.str());
        }
    }
}

void NvidiaValidationSuite::ThrowTestNotFoundExecption() const
{
    std::vector<std::string> notFoundTests;
    std::vector<std::string> noSupportedEntityTests;
    std::vector<std::string> eudTests;

    for (auto const &requestedTestName : nvvsCommon.desiredTest)
    {
        bool foundInTestVect = false;

        for (auto const *test : testVect)
        {
            std::string compareTestName = test->GetTestName();
            std::transform(compareTestName.begin(), compareTestName.end(), compareTestName.begin(), ::tolower);
            std::string const compareRequestedName = m_tf->GetCompareName(requestedTestName);

            if (compareTestName == compareRequestedName)
            {
                foundInTestVect = true;
                break;
            }
        }

        if (foundInTestVect)
        {
            noSupportedEntityTests.push_back(requestedTestName);
        }
        else
        {
            // If the required packages are not available, EUD-related tests will not be exported.
            // To inform users about installing packages, we'd like to display a distinct notification.
            if (requestedTestName == EUD_PLUGIN_NAME || requestedTestName == CPU_EUD_TEST_NAME)
            {
                eudTests.push_back(requestedTestName);
            }
            else
            {
                notFoundTests.push_back(requestedTestName);
            }
        }
    }
    std::string noSupportedEntityTestsMsg;
    if (!noSupportedEntityTests.empty())
    {
        noSupportedEntityTestsMsg
            = fmt::format("[{}] cannot find supported entity to test. ", fmt::join(noSupportedEntityTests, ", "));
    }
    std::string notFoundTestsMsg;
    if (!notFoundTests.empty())
    {
        notFoundTestsMsg
            = fmt::format("[{}] were not found among possible test choices. ", fmt::join(notFoundTests, ", "));
    }
    std::string eudTestsMsg;
    if (!eudTests.empty())
    {
        std::string flatEudTests = fmt::to_string(fmt::join(eudTests, ", "));
        eudTestsMsg              = fmt::format(
            "[{}] cannot find dependent package(s). Please ensure that both the [{}] dependent package and the DCGM proprietary package are installed.",
            flatEudTests,
            flatEudTests);
    }
    std::string errMsg
        = fmt::format("Error: requested test(s): {}{}{}", noSupportedEntityTestsMsg, notFoundTestsMsg, eudTestsMsg);
    log_error(errMsg);
    throw NvvsException(errMsg, NVVS_ST_TEST_NOT_FOUND);
}

/*****************************************************************************/
// take our entity sets vector and fill in the appropriate GPU objects that match that set
void NvidiaValidationSuite::DistributeTests(std::vector<std::unique_ptr<EntitySet>> &entitySets)
{
    std::unordered_set<std::string> foundSuites;
    bool anyTestAssigned = false;

    // The rules are:
    // a) the "properties" struct is exclusionary. If properties is empty (properties.present == false)
    //    then all available GPU objects are included in the set
    // b) the "tests" vector is also exclusionary. If tests.size() == 0 then all available test
    //    objects are included in the set
    for (auto &&entitySet : entitySets)
    {
        bool first_pass = true;

        // go through the vector of tests requested and try to match them with an actual test.
        // push a warning if no match found
        for (auto requestedTestNameIt = nvvsCommon.desiredTest.begin();
             requestedTestNameIt != nvvsCommon.desiredTest.end();)
        {
            bool found                           = false;
            bool foundSuite                      = false;
            std::string const &requestedTestName = *requestedTestNameIt;
            std::string compareTestName          = requestedTestName;
            std::transform(compareTestName.begin(), compareTestName.end(), compareTestName.begin(), ::tolower);

            // first check the test suite names
            suiteNames_enum suite;
            if (compareTestName == "quick" || compareTestName == "short")
            {
                foundSuite = true;
                suite      = NVVS_SUITE_QUICK;
            }
            else if (compareTestName == "medium")
            {
                foundSuite = true;
                suite      = NVVS_SUITE_MEDIUM;
            }
            else if (compareTestName == "long")
            {
                foundSuite = true;
                suite      = NVVS_SUITE_LONG;
            }
            else if (compareTestName == "xlong")
            {
                foundSuite = true;
                suite      = NVVS_SUITE_XLONG;
            }
            else if (compareTestName == "production_testing")
            {
                foundSuite = true;
                suite      = NVVS_SUITE_PRODUCTION_TESTING;
            }

            if (foundSuite)
            {
                // It at least contains software plugin.
                found = (entitySet->GetEntityGroup() == DCGM_FE_GPU);
                if (FillTestVectors(suite, Test::NVVS_CLASS_HARDWARE, entitySet.get()))
                {
                    found = true;
                }
                if (FillTestVectors(suite, Test::NVVS_CLASS_INTEGRATION, entitySet.get()))
                {
                    found = true;
                }
                if (FillTestVectors(suite, Test::NVVS_CLASS_PERFORMANCE, entitySet.get()))
                {
                    found = true;
                }
            }
            // then check the test categories
            else
            {
                /*
                 * When diag module runs EUD with enabled service-account, the EUD is the only test in the command line.
                 */
                if (first_pass == true && compareTestName != "eud")
                {
                    found      = FillTestVectors(NVVS_SUITE_CUSTOM, Test::NVVS_CLASS_SOFTWARE, entitySet.get());
                    first_pass = false;
                }
                std::map<std::string, std::vector<Test *>> testCategories = m_tf->GetTestCategories();
                std::map<std::string, std::vector<Test *>>::iterator it   = testCategories.find(requestedTestName);

                if (it != testCategories.end())
                {
                    // Add each test from the list
                    for (size_t i = 0; i < testCategories[requestedTestName].size(); i++)
                    {
                        if (testCategories[requestedTestName][i]->GetTargetEntityGroup() != entitySet->GetEntityGroup())
                        {
                            continue;
                        }
                        found = true;
                        entitySet->AddTestObject(CUSTOM_TEST_OBJS, testCategories[requestedTestName][i]);
                    }
                }
                else // now check individual tests
                {
                    for (std::vector<Test *>::iterator testIt = testVect.begin(); testIt != testVect.end(); ++testIt)
                    {
                        // convert everything to lower case for comparison
                        std::string compareTestName = (*testIt)->GetTestName();
                        std::transform(
                            compareTestName.begin(), compareTestName.end(), compareTestName.begin(), ::tolower);
                        std::string compareRequestedName = m_tf->GetCompareName(requestedTestName);

                        if (compareTestName == compareRequestedName
                            && (*testIt)->GetTargetEntityGroup() == entitySet->GetEntityGroup())
                        {
                            found = true;
                            // Make a full copy of the test parameters
                            TestParameters *tp = new TestParameters();
                            tpVect.push_back(tp); // purely for accounting when we go to cleanup

                            if (entitySet->GetEntityGroup() == DCGM_FE_GPU)
                            {
                                GpuSet *gpuSet = ToGpuSet(entitySet.get());
                                m_allowlist->GetDefaultsByDeviceId(
                                    compareRequestedName, gpuSet->GetGpuObjs()[0]->getDeviceId(), tp);
                            }

                            if (nvvsCommon.parms.size() > 0)
                            {
                                overrideParameters(tp, compareRequestedName);
                            }

                            tp->AddString(PS_PLUGIN_NAME, (*testIt)->GetPluginName());
                            tp->AddString(PS_TEST_NAME, (*testIt)->GetTestName());
                            tp->AddDouble(PS_LOGFILE_TYPE, (double)nvvsCommon.logFileType);

                            (*testIt)->pushArgVectorElement(Test::NVVS_CLASS_CUSTOM, tp);
                            entitySet->AddTestObject(CUSTOM_TEST_OBJS, (*testIt));
                            break;
                        }
                    }
                }
            }

            if (found)
            {
                anyTestAssigned = true;
                if (foundSuite)
                {
                    // A suite can consist of tests that span multiple entity sets.
                    // To ensure all tests are well distributed into all entity sets.
                    // We keep suite here and remove the suite from desiredTest in the end.
                    foundSuites.insert(*requestedTestNameIt);
                    ++requestedTestNameIt;
                }
                else
                {
                    requestedTestNameIt = nvvsCommon.desiredTest.erase(requestedTestNameIt);
                }
            }
            else
            {
                ++requestedTestNameIt;
            }
        }
    }

    for (auto const &suite : foundSuites)
    {
        nvvsCommon.desiredTest.erase(suite);
    }

    if (!nvvsCommon.desiredTest.empty())
    {
        if (!anyTestAssigned || !nvvsCommon.rerunAsRoot)
        {
            ThrowTestNotFoundExecption();
        }
    }
}

/*****************************************************************************/
bool NvidiaValidationSuite::FillTestVectors(suiteNames_enum suite, Test::testClasses_enum testClass, EntitySet *set)
{
    int type;
    std::vector<std::string> testNames;

    switch (testClass)
    {
        case Test::NVVS_CLASS_SOFTWARE:
            testNames.push_back("Denylist");
            testNames.push_back("NVML Library");
            testNames.push_back("CUDA Main Library");
            /* Now that we link statically against cuda from the plugins, there's no need for this check. Furthermore,
             * having it makes it so nv-hostengine has to have the cuda toolkit in its LD_LIBRARY_PATH
             */
            testNames.push_back("Permissions and OS-related Blocks");
            testNames.push_back("Persistence Mode");
            testNames.push_back("Environmental Variables");
            testNames.push_back("Page Retirement/Row Remap");
            testNames.push_back("Graphics Processes");
            testNames.push_back("Inforom");
            testNames.push_back("Fabric Manager");
            type = SOFTWARE_TEST_OBJS;
            break;
        case Test::NVVS_CLASS_HARDWARE:
            if (suite >= NVVS_SUITE_MEDIUM)
                testNames.push_back(MEMORY_PLUGIN_NAME);
            if (suite >= NVVS_SUITE_LONG)
            {
                testNames.push_back(DIAGNOSTIC_PLUGIN_NAME);
                if (DcgmNs::Utils::IsRunningAsRoot())
                {
                    testNames.push_back(EUD_PLUGIN_NAME);
                    testNames.push_back(CPU_EUD_TEST_NAME);
                }
                testNames.push_back(NVBANDWIDTH_PLUGIN_NAME);
            }
            if (suite >= NVVS_SUITE_XLONG)
            {
                testNames.push_back(MEMTEST_PLUGIN_NAME);
                testNames.push_back(PULSE_TEST_PLUGIN_NAME);
            }
            type = HARDWARE_TEST_OBJS;
            break;
        case Test::NVVS_CLASS_INTEGRATION:
            if (suite >= NVVS_SUITE_MEDIUM)
                testNames.push_back(PCIE_PLUGIN_NAME);
            type = INTEGRATION_TEST_OBJS;
            break;
        case Test::NVVS_CLASS_PERFORMANCE:
            if (suite >= NVVS_SUITE_LONG)
            {
                testNames.push_back(MEMBW_PLUGIN_NAME);
                testNames.push_back(TS_PLUGIN_NAME);
                testNames.push_back(TP_PLUGIN_NAME);
            }
            type = PERFORMANCE_TEST_OBJS;
            break;
        default:
        {
            throw std::runtime_error(fmt::format("Received test class '{}' that is not valid.", testClass));
        }
    }

    bool testAdded = false;
    for (std::vector<std::string>::iterator it = testNames.begin(); it != testNames.end(); it++)
    {
        std::vector<Test *>::iterator testIt;
        testIt = FindTestName(*it);

        if (testIt != testVect.end())
        {
            TestParameters *tp = new TestParameters();
            tpVect.push_back(tp); // purely for accounting when we go to cleanup

            if ((*testIt)->GetTargetEntityGroup() != set->GetEntityGroup())
            {
                continue;
            }

            if (testClass != Test::NVVS_CLASS_SOFTWARE)
            {
                // for uniformity downstream
                /*
                std::string lowerCaseTestName = *it;
                std::transform(
                    lowerCaseTestName.begin(), lowerCaseTestName.end(), lowerCaseTestName.begin(), ::tolower);
                */

                // pull just the first GPU device ID since they are all meant to be the same at this point
                if (set->GetEntityGroup() == DCGM_FE_GPU)
                {
                    GpuSet *gpuSet = ToGpuSet(set);
                    m_allowlist->GetDefaultsByDeviceId(*it, gpuSet->GetGpuObjs()[0]->getDeviceId(), tp);
                }

                if (nvvsCommon.parms.size() > 0)
                    overrideParameters(tp, *it);
            }

            tp->AddString(PS_PLUGIN_NAME, (*testIt)->GetPluginName());
            tp->AddString(PS_TEST_NAME, (*it));
            tp->AddDouble(PS_LOGFILE_TYPE, (double)nvvsCommon.logFileType);
            tp->AddDouble(PS_SUITE_LEVEL, (double)suite);

            (*testIt)->pushArgVectorElement(testClass, tp);
            set->AddTestObject(type, *testIt);
            testAdded = true;
        }
    }

    return testAdded;
}

/*****************************************************************************/
std::vector<Test *>::iterator NvidiaValidationSuite::FindTestName(std::string testName)
{
    std::string compareName = m_tf->GetCompareName(testName);

    for (auto it = testVect.begin(); it != testVect.end(); ++it)
    {
        if ((*it)->GetTestName() == compareName)
            return it;
    }

    return testVect.end();
}

/*****************************************************************************/
std::vector<Gpu *> NvidiaValidationSuite::decipherProperties(GpuSet *set)
{
    // exclusionary rules:
    // 1) If pci busID exists, ignore everything else
    // 2) If UUID exists, ignore everything else
    // 3) If index exists check for brand/name exclusions
    // 4) Otherwise match only on brand/name combos
    std::vector<Gpu *> tempGpuVec;

    auto it = m_gpuVect.begin();
    while (it != m_gpuVect.end())
    {
        bool brand = false, name = false;
        if (set->GetProperties().brand.length() > 0)
            brand = true;
        if (set->GetProperties().name.length() > 0)
        {
            name = true;
            // kludge to handle special naming of K10
            if (set->GetProperties().name == "Tesla K10")
                set->GetProperties().name = "Tesla K10.G1.8GB";
        }

        if (set->GetProperties().uuid.length() > 0)
        {
            if (set->GetProperties().uuid == (*it)->getDeviceGpuUuid())
                tempGpuVec.push_back(*it);
            ++it;
            continue; // skip everything else
        }
        else if (set->GetProperties().busid.length() > 0)
        {
            if (set->GetProperties().busid == (*it)->getDevicePciBusId())
                tempGpuVec.push_back(*it);
            ++it;
            continue; // skip everything else
        }
        else if (set->GetProperties().index.size() > 0)
        {
            for (unsigned int i = 0; i < set->GetProperties().index.size(); i++)
            {
                if (set->GetProperties().index[i] == (*it)->getDeviceIndex())
                {
                    if (!brand && !name)
                        tempGpuVec.push_back(*it);
                    if (brand && !name && set->GetProperties().brand == (*it)->getDeviceBrandAsString())
                        tempGpuVec.push_back(*it);
                    if (name && !brand && set->GetProperties().name == (*it)->getDeviceName())
                        tempGpuVec.push_back(*it);
                    if (brand && name && set->GetProperties().brand == (*it)->getDeviceBrandAsString()
                        && set->GetProperties().name == (*it)->getDeviceName())
                        tempGpuVec.push_back(*it);
                }
            }
        }
        else if (brand || name)
        {
            if (brand && !name && set->GetProperties().brand == (*it)->getDeviceBrandAsString())
                tempGpuVec.push_back(*it);
            if (name && !brand && set->GetProperties().name == (*it)->getDeviceName())
                tempGpuVec.push_back(*it);
            if (brand && name && set->GetProperties().brand == (*it)->getDeviceBrandAsString()
                && set->GetProperties().name == (*it)->getDeviceName())
                tempGpuVec.push_back(*it);
        }
        ++it;
    }

    return tempGpuVec;
}

/*****************************************************************************/
void initializeDesiredTests(const std::string &specifiedTests)
{
    if (specifiedTests.size() > 0)
    {
        std::vector<std::string> testNames;
        dcgmTokenizeString(specifiedTests, ",", testNames);
        for (size_t i = 0; i < testNames.size(); i++)
        {
            nvvsCommon.desiredTest.insert(testNames[i]);
        }
    }
}

/*****************************************************************************/
void NvidiaValidationSuite::InitializeParameters(const std::string &parms, const ParameterValidator &pv)
{
    std::stringstream buf;
    buf << "Invalid Parameter String: ";

    if (parms.size() > 0)
    {
        std::vector<std::string> parmsVec;
        dcgmTokenizeString(parms, ";", parmsVec);

        for (size_t i = 0; i < parmsVec.size(); i++)
        {
            std::string testName;
            std::string parmName;
            std::string parmValue;

            size_t dot    = parmsVec[i].find('.');
            size_t equals = parmsVec[i].find('=');

            if (dot != std::string::npos && equals != std::string::npos)
            {
                testName  = parmsVec[i].substr(0, dot);
                parmName  = parmsVec[i].substr(dot + 1, equals - (dot + 1));
                parmValue = parmsVec[i].substr(equals + 1);

                if (pv.IsValidTestName(testName) == false)
                {
                    buf << "test '" << testName << "' does not match any loaded tests. ";
                    auto transformedTestName = ParameterValidator::TransformTestName(testName);
                    if (transformedTestName == EUD_PLUGIN_NAME || transformedTestName == CPU_EUD_TEST_NAME)
                    {
                        buf << "Please ensure that both the [" << transformedTestName
                            << "] dependent package and the DCGM proprietary package are installed.";
                    }
                    else
                    {
                        buf << "Check logs for plugin failures.";
                    }
                    throw NvvsException(buf.str(), NVVS_ST_TEST_NOT_FOUND);
                }

                size_t subtestDot = parmName.find('.');
                if (subtestDot != std::string::npos)
                {
                    // Found a subtest
                    std::string subtest(parmName.substr(0, subtestDot));
                    std::string subtestParm(parmName.substr(subtestDot + 1));

                    if (pv.IsValidSubtest(testName, subtest) == false)
                    {
                        buf << "test '" << testName << "' has no subtest '" << subtest << "'.";
                        throw std::runtime_error(buf.str());
                    }

                    if (pv.IsValidSubtestParameter(testName, subtest, subtestParm) == false)
                    {
                        buf << "test '" << testName << "' subtest '" << subtest << "' has no parameter '" << subtestParm
                            << "'.";
                        throw std::runtime_error(buf.str());
                    }
                }
                else if (pv.IsValidParameter(testName, parmName) == false)
                {
                    buf << "test '" << testName << "' has no parameter '" << parmName << "'.";
                    throw std::runtime_error(buf.str());
                }

                std::string requestedName                 = m_tf->GetCompareName(testName);
                nvvsCommon.parms[requestedName][parmName] = std::move(parmValue);
            }
            else
            {
                buf << "unable to parse test, parameter name, and value from string '" << parmsVec[i]
                    << "'. Format should be <testname>[.<subtest>].<parameter name>=<parameter value>";
                throw std::runtime_error(buf.str());
            }
        }
    }
}

/*****************************************************************************/
/* Special class to handle custom output for CL
 */

class NVVSOutput : public TCLAP::StdOutput
{
public:
    void usage(TCLAP::CmdLineInterface &_cmd) override
    {
        TCLAP::StdOutput::usage(_cmd);

        std::cout << "Please email cudatools@nvidia.com with any questions, bug reports, etc.\n" << std::endl;
    }
};

/*****************************************************************************/
/* Process command line arguments and use those arguments to override anything specified
 * by the config file.
 */
void NvidiaValidationSuite::processCommandLine(int argc, char *argv[])
{
    std::string configFileArg;

    try
    {
        TCLAP::CmdLine cmd(NVVS_NAME, ' ', std::string(DcgmNs::DcgmBuildInfo().GetVersion()));
        NVVSOutput nvout;
        cmd.setOutput(&nvout);
        // add this so it displays as part of help but it is effectively ignored
        TCLAP::SwitchArg verboseArg("v", "verbose", "Enable verbose reporting for some plugins.", cmd, false);
        TCLAP::SwitchArg listTestsArg(
            "t", "listTests", "List the tests available to be executed through NVVS.", cmd, false);
        TCLAP::SwitchArg statsOnFailArg(
            "", "statsonfail", "Output statistic logs only if a test failure is encountered.", cmd, false);
        TCLAP::ValueArg<std::string> hwdiaglogfileArg(
            "",
            "hwdiaglogfile",
            "Encrypted HW diagnostics log file. Append this to save the HW diagnostics logs at any specified location.\
            If path is not specified then \"nvidia-diagnostic.log\" is the default logfile .",
            false,
            "",
            "HW Diagnostics log file",
            cmd);
        TCLAP::ValueArg<std::string> specificTestArg(
            "",
            "specifiedtest",
            "Run a specific test in a configless mode. Multiple word tests should be in quotes.",
            false,
            "",
            "specific test to run",
            cmd);
        TCLAP::SwitchArg parseArg(
            "s", "scriptable", "Give output in colon-separated, more script-friendly format.", cmd, false);
        TCLAP::SwitchArg quietModeArg(
            "", "quiet", "No console output given.  See logs and return code for errors.", cmd, false);
        TCLAP::ValueArg<std::string> pluginPathArg(
            "p", "pluginpath", "Specify a custom path for the NVVS plugins.", false, "", "path to plugins", cmd);
        TCLAP::ValueArg<std::string> debugFileArg("l",
                                                  "debugLogFile",
                                                  "Encrypted logfile for debug information. If a debug level \
            has been specified then \"" NVVS_LOGGING_DEFAULT_NVVS_LOGFILE "\" is the default logfile.",
                                                  false,
                                                  "",
                                                  "debug file",
                                                  cmd);
        TCLAP::ValueArg<std::string> indexArg(
            "i", "indexes", "Comma separated list of indexes to run NVVS on.", false, "", "indexes", cmd);
        TCLAP::SwitchArg listGpusArg("g", "listGpus", "List the GPUS available.", cmd, false);
        TCLAP::ValueArg<std::string> debugLevelArg("d",
                                                   "debugLevel",
                                                   "Debug level (One of " DCGM_LOGGING_SEVERITY_OPTIONS
                                                   "). Default:" DCGM_LOGGING_DEFAULT_NVVS_SEVERITY,
                                                   false,
                                                   "",
                                                   "debug level",
                                                   cmd);
        TCLAP::SwitchArg configLessArg(
            "",
            "configless",
            "Run NVVS in a configless mode.  Executes a \"long\" test on all supported GPUs.",
            cmd,
            false);
        TCLAP::ValueArg<std::string> configArg(
            "c", "config", "Specify a path to the configuration file.", false, "", "path to config file", cmd);
        TCLAP::ValueArg<std::string> fakeGpusArg(
            "f",
            "",
            "", // Comma separated list of fake gpu indexes to run NVVS on. For internal/testing use only.
            false,
            "",
            "",
            cmd);
        TCLAP::ValueArg<std::string> statsPathArg(
            "",
            "statspath",
            "Write the plugin statistics to a given path rather than the current directory.",
            false,
            "",
            "plugin statistics path",
            cmd);
        TCLAP::ValueArg<std::string> parms("",
                                           "parameters",
                                           "Specify test parameters in a configless mode.",
                                           false,
                                           "",
                                           "parameters to set for tests",
                                           cmd);
        TCLAP::ValueArg<unsigned int> initializationWaitTime(
            "w",
            "initwaittime",
            "Number of seconds to wait before aborting DCGM initialization",
            false,
            120,
            "initialization wait time",
            cmd);
        TCLAP::ValueArg<std::string> dcgmHost(
            "", "dcgmHostname", "Specify the hostname where DCGM is running.", false, "", "DCGM hostname", cmd);
        TCLAP::ValueArg<std::string> throttleMask("",
                                                  "throttle-mask",
                                                  "Deprecated: please use clocksevent-mask instead.",
                                                  false,
                                                  "",
                                                  "deprecated: throttle reasons to ignore",
                                                  cmd);

        TCLAP::ValueArg<std::string> clocksEventMask(
            "",
            "clocksevent-mask",
            "Specify which clocks event reasons should be ignored. You can provide a comma separated list of reasons. "
            "For example, specifying 'HW_SLOWDOWN,SW_THERMAL' would ignore the HW_SLOWDOWN and SW_THERMAL clocks event "
            "reasons. Alternatively, you can specify the integer value of the ignore bitmask. For the bitmask, "
            "multiple reasons may be specified by the sum of their bit masks. For "
            "example, specifying '40' would ignore the HW_SLOWDOWN and SW_THERMAL clocks event reasons.\n"
            "Valid clocks event reasons and their corresponding bitmasks (given in parentheses) are:\n"
            "HW_SLOWDOWN (8)\nSW_THERMAL (32)\nHW_THERMAL (64)\nHW_POWER_BRAKE (128)",
            false,
            "",
            "clocks event reasons to ignore",
            cmd);

        TCLAP::SwitchArg failEarly(
            "",
            "fail-early",
            "Enable early failure checks for the Targeted Power, Targeted Stress, SM Stress, and Diagnostic tests. "
            "When enabled, these tests check for a failure once every 5 seconds (can be modified by the "
            "--check-interval parameter) while the test is running instead of a single check performed after the "
            "test is complete. Disabled by default.",
            cmd,
            false);

        TCLAP::ValueArg<unsigned int> failCheckInterval(
            "",
            "check-interval",
            "Specify the interval (in seconds) at which the early failure checks should occur for the "
            "Targeted Power, Targeted Stress, SM Stress, and Diagnostic tests when early failure checks are enabled. "
            "Default is once every 5 seconds. Interval must be between 1 and 300",
            false,
            5,
            "failure check interval",
            cmd);

        TCLAP::ValueArg<unsigned int> currentIteration(
            "",
            "current-iteration",
            "Specify which iteration of the diagnostic is currently running.",
            false,
            0,
            "current iteration",
            cmd);

        TCLAP::ValueArg<unsigned int> totalIterations("",
                                                      "total-iterations",
                                                      "Specify how many iterations of the diagnostic will be run.",
                                                      false,
                                                      1,
                                                      "total iterations",
                                                      cmd);
        TCLAP::ValueArg<std::string> entityIds(
            "", "entity-id", " Comma-separated list of entities to run the diag on.", false, "", "entityId", cmd);
        TCLAP::ValueArg<int> channelFd(
            "", "channel-fd", "A file description used to send back response to caller.", false, -1, "channel fd", cmd);
        TCLAP::ValueArg<unsigned int> responseVersion("",
                                                      "response-version",
                                                      "The version of diag response to be returned via channel-fd.",
                                                      false,
                                                      dcgmDiagResponse_version12,
                                                      "responseVersion",
                                                      cmd);
        TCLAP::SwitchArg rerunAsRoot(
            "",
            "rerun-as-root",
            "Flag to indicate we are running all tests with root for the current term, attempting to execute as many tests as possible "
            "without failing due to 'test not found' errors unless all specified tests are genuinely unavailable.",
            cmd,
            false);

        unsigned int constexpr DEFAULT_WATCH_FREQUENCY_IN_MICROSECONDS { 5000000 };
        TCLAP::ValueArg<unsigned int> watchFrequency(
            "",
            "watch-frequency",
            "Specify the watch frequency in microseconds for the fields being watched.",
            false,
            DEFAULT_WATCH_FREQUENCY_IN_MICROSECONDS,
            "watch frquency",
            cmd);

        TCLAP::ValueArg<std::string> ignoreErrorCodes(
            "",
            "ignoreErrorCodes",
            "Specify error codes to be ignored on specific entities."
            "Format: --ignoreErrorCodes=28,140 (ignore error codes 28 and 140 on all entities) \
                    \n--ignoreErrorCodes=gpu0:28;gpu1:140 (ignore error 28 on GPU 0 and error 140 on GPU 1) \
                    \n--ignoreErrorCodes=*:* (ignore all errors that can be ignored on all entities)",
            false,
            "",
            "ignore error codes",
            cmd);

        cmd.parse(argc, argv);

        configFileArg = configArg.getValue();
        if (configFileArg.size() > 0)
        {
            configFile = std::move(configFileArg);
        }

        listGpus                          = listGpusArg.getValue();
        listTests                         = listTestsArg.getValue();
        nvvsCommon.verbose                = verboseArg.getValue();
        nvvsCommon.pluginPath             = pluginPathArg.getValue();
        nvvsCommon.parse                  = parseArg.getValue();
        nvvsCommon.quietMode              = quietModeArg.getValue();
        nvvsCommon.configless             = configLessArg.getValue();
        nvvsCommon.fakegpusString         = fakeGpusArg.getValue();
        nvvsCommon.statsOnlyOnFail        = statsOnFailArg.getValue();
        nvvsCommon.indexString            = indexArg.getValue();
        nvvsCommon.parmsString            = parms.getValue();
        nvvsCommon.dcgmHostname           = dcgmHost.getValue();
        nvvsCommon.currentIteration       = currentIteration.getValue();
        nvvsCommon.totalIterations        = totalIterations.getValue();
        nvvsCommon.entityIds              = entityIds.getValue();
        nvvsCommon.channelFd              = channelFd.getValue();
        nvvsCommon.diagResponseVersion    = responseVersion.getValue();
        nvvsCommon.rerunAsRoot            = rerunAsRoot.isSet();
        nvvsCommon.ignoreErrorCodesString = ignoreErrorCodes.getValue();
        nvvsCommon.SetStatsPath(statsPathArg.getValue());

        this->initWaitTime = initializationWaitTime.getValue();

        if (listGpus || listTests)
            nvvsCommon.configless = true;

        if (nvvsCommon.desiredTest.size() > 0)
            nvvsCommon.configless = true;

        initializeDesiredTests(specificTestArg.getValue());

        debugFile = GetLogFilenameFromArgAndEnv(
            debugFileArg.getValue(), NVVS_LOGGING_DEFAULT_NVVS_LOGFILE, NVVS_ENV_LOG_PREFIX);

        debugLogLevel = GetLogSeverityFromArgAndEnv(
            debugLevelArg.getValue(), DCGM_LOGGING_DEFAULT_NVVS_SEVERITY, NVVS_ENV_LOG_PREFIX);

        if (hwdiaglogfileArg.isSet())
            hwDiagLogFile = hwdiaglogfileArg.getValue();


        if (clocksEventMask.isSet() && throttleMask.isSet())
        {
            throw TCLAP::CmdLineParseException("Must specify no more than one of: clocksevent-mask, throttle-mask");
        }

        // Set bitmask for ignoring user specified clocks reasons
        if (clocksEventMask.isSet())
        {
            std::string reasonStr = clocksEventMask.getValue();
            // Make reasonStr lower case for parsing
            std::transform(reasonStr.begin(), reasonStr.end(), reasonStr.begin(), ::tolower);
            nvvsCommon.clocksEventIgnoreMask = GetClocksEventIgnoreReasonMaskFromString(std::move(reasonStr));
        }
        // Set bitmask for ignoring user specified clocks reasons (deprecated)
        else if (throttleMask.isSet())
        {
            std::string reasonStr = throttleMask.getValue();
            // Make reasonStr lower case for parsing
            std::transform(reasonStr.begin(), reasonStr.end(), reasonStr.begin(), ::tolower);
            nvvsCommon.clocksEventIgnoreMask = GetClocksEventIgnoreReasonMaskFromString(std::move(reasonStr));
        }

        // Enable early failure checks if requested
        if (failEarly.isSet())
        {
            nvvsCommon.failEarly = true;
            if (failCheckInterval.isSet())
            {
                nvvsCommon.failCheckInterval = failCheckInterval.getValue();
            }
        }

        if (watchFrequency.isSet())
        {
            unsigned int watchFrequencyVal = watchFrequency.getValue();
            // Safe guard the lower bound to avoid excessive memory usage
            // Revert to default values when users input too fast (100 ms) /slow (60 s) watch frequency
            if (watchFrequencyVal < 100000 || watchFrequencyVal > 60000000)
            {
                nvvsCommon.watchFrequency = DEFAULT_WATCH_FREQUENCY_IN_MICROSECONDS;
            }
            else
            {
                nvvsCommon.watchFrequency = watchFrequencyVal;
            }
        }
    }
    catch (TCLAP::ArgException &e)
    {
        std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl;
        throw std::runtime_error("An error occurred trying to parse the command line.");
    }
}
