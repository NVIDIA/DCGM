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

#include "NvidiaValidationSuite.h"
#include "DcgmHandle.h"
#include "DcgmLogging.h"
#include "DcgmSystem.h"
#include "NvvsJsonStrings.h"
#include "ParameterValidator.h"
#include "ParsingUtility.h"
#include "PluginStrings.h"
#include "dcgm_structs_internal.h"
#include <DcgmStringHelpers.h>
#include <PluginInterface.h>
#include <cstdlib>
#include <iostream>
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
    , whitelist(0)
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
    , m_sysCheck()
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
    delete whitelist;
    delete parser;

    dcgmShutdown();
}

/*****************************************************************************/
void NvidiaValidationSuite::CheckDriverVersion()
{
    dcgmSystem.Init(dcgmHandle.GetHandle());
    dcgmDeviceAttributes_t deviceAttr;

    memset(&deviceAttr, 0, sizeof(deviceAttr));
    deviceAttr.version = dcgmDeviceAttributes_version2;
    dcgmReturn_t ret   = dcgmSystem.GetDeviceAttributes(0, deviceAttr);
    unsigned int count = 0;
    std::stringstream additionalMsg;

    // Attempt re-connecting if we have trouble with our initial interaction with the hostengine.
    for (count = 0; count < 3 && ret == DCGM_ST_CONNECTION_NOT_VALID; ++count)
    {
        dcgmReturn_t connectionRet = dcgmHandle.ConnectToDcgm(nvvsCommon.dcgmHostname);
        if (connectionRet == DCGM_ST_OK)
        {
            dcgmSystem.Init(dcgmHandle.GetHandle());
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
        PRINT_ERROR("%s", "Cannot run on incompatible driver version '%s'", deviceAttr.identifiers.driverVersion);
        throw std::runtime_error(exceptionSs.str());
    }
}

// Handler for SIGALRM as part of ensuring we can't stall indefinitely on DCGM init
void alarm_handler(int sig)
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

bool NvidiaValidationSuite::IsGpuIncluded(unsigned int gpuIndex, std::vector<unsigned int> &gpuIndices)
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
    unsigned int mgpusIndex = 0;
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

        nvvsCommon.m_gpus[mgpusIndex] = visibleGpus[i];
        mgpusIndex++;

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
        val << gpuIndices[0];
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
    std::string info;

    processCommandLine(argc, argv);
    if (nvvsCommon.quietMode)
    {
        std::cout.setstate(std::ios::failbit);
    }
    banner();

    DcgmLogging::init(debugFile.c_str(),
                      DcgmLogging::severityFromString(debugLogLevel.c_str(), DcgmLoggingSeverityDebug));
    DCGM_LOG_INFO << "Initialized NVVS logger";
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

    /*
    stopTimer();
    */

    CheckDriverVersion();

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
            out << "/etc/nvidia-validation-suite/nvvs.conf. " << std::endl;
            out << "Please check the path or specify a config file via the \"-c\" command line option." << std::endl;
            out << "Add --configless to suppress this warning." << std::endl;
            info = out.str();

            // Force to true if we couldn't open a config file
            nvvsCommon.configless = true;
        }
    }

    parser->legacyGlobalStructHelper();

    std::vector<std::unique_ptr<GpuSet>> &gpuSets = parser->getGpuSetVec();

    EnumerateAllVisibleGpus();

    if (listGpus)
    {
        std::cout << "Supported GPUs available:" << std::endl;

        for (std::vector<Gpu *>::iterator it = m_gpuVect.begin(); it != m_gpuVect.end(); ++it)
        {
            std::cout << "\t"
                      << "[" << (*it)->getDevicePciBusId() << "] -- " << (*it)->getDeviceName() << std::endl;
        }
        std::cout << std::endl;
        return "";
    }

    for (size_t i = 0; i < gpuSets.size(); i++)
    {
        for (size_t j = 0; j < gpuSets[i]->properties.index.size(); j++)
        {
            gpuIndices.push_back(gpuSets[i]->properties.index[j]);
        }
    }

    InitializeAndCheckGpuObjs(gpuSets);

    std::string errorString = BuildCommonGpusList(gpuIndices, m_gpuVect);
    if (errorString.empty() == false)
    {
        return errorString;
    }

    if (nvvsCommon.training && !nvvsCommon.forceTraining)
    {
        std::string sysError = m_sysCheck.CheckSystemInterference();

        if (sysError.size())
        {
            PRINT_WARNING("%s", "%s", sysError.c_str());
        }
    }

    // construct the test framework now that we have the GPU and test objects
    // Only pass gpuSets[0] because there is always only 1 group of GPUs. This will
    // be refactored in a separate check-in.
    m_tf = new TestFramework(nvvsCommon.jsonOutput, gpuSets[0].get());
    m_tf->loadPlugins();
    if (info.size() > 0)
    {
        m_tf->addInfoStatement(info);
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

    if (nvvsCommon.training)
    {
        unsigned int iterations = nvvsCommon.trainingIterations;
        for (unsigned int i = 0; i < iterations; i++)
        {
            for (size_t setIndex = 0; setIndex < gpuSets.size(); setIndex++)
            {
                if (i == 0)
                    fillTestVectors(NVVS_SUITE_LONG, Test::NVVS_CLASS_SOFTWARE, gpuSets[setIndex].get());

                fillTestVectors(NVVS_SUITE_LONG, Test::NVVS_CLASS_HARDWARE, gpuSets[setIndex].get());
                fillTestVectors(NVVS_SUITE_LONG, Test::NVVS_CLASS_INTEGRATION, gpuSets[setIndex].get());
                fillTestVectors(NVVS_SUITE_LONG, Test::NVVS_CLASS_PERFORMANCE, gpuSets[setIndex].get());
            }

            m_tf->go(gpuSets);

            float pcnt = static_cast<float>(i + 1) / static_cast<float>(iterations);
            if (nvvsCommon.jsonOutput == false)
            {
                std::cout << "Completed iteration " << i + 1 << " of " << iterations << " : training is "
                          << static_cast<int>(pcnt * 100) << "% complete." << std::endl;
            }
        }

        m_tf->CalculateAndSaveGoldenValues();
    }
    else
    {
        CheckGpuSetTests(gpuSets);

        // Execute the tests... let the TF catch all exceptions and decide
        // whether to throw them higher.
        m_tf->go(gpuSets);
    }

    return "";
}

/*****************************************************************************/
void NvidiaValidationSuite::banner()
{
    if (nvvsCommon.jsonOutput == false)
    {
        std::cout << std::endl << NVVS_NAME << " (version " << DRIVER_MAJOR_VERSION << ")" << std::endl << std::endl;
    }
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
    bool isWhitelisted;
    std::vector<unsigned int> gpuIds;
    dcgmReturn_t ret = dcgmSystem.GetAllSupportedDevices(gpuIds);
    std::stringstream buf;

    if (ret != DCGM_ST_OK)
    {
        buf << "Unable to retrieve device count: " << dcgmHandle.RetToString(ret);
        throw std::runtime_error(buf.str());
    }

    whitelist = new Whitelist(*parser);

    for (size_t i = 0; i < gpuIds.size(); i++)
    {
        std::unique_ptr<Gpu> gpu { new Gpu(gpuIds[i]) };
        if ((ret = gpu->Init()) != DCGM_ST_OK)
        {
            buf << "Unable to initialize GPU " << gpuIds[i] << ": " << dcgmHandle.RetToString(ret);
            throw std::runtime_error(buf.str());
        }

        /* Find out if this device is supported, which is any of the following:
           1. On the NVVS whitelist explicitly
           2. A Kepler or newer Tesla part
           3. a Maxwell or newer part of any other brand (Quadro, GeForce, Titan, Grid)
        */

        if (whitelist->isWhitelisted(gpu->getDevicePciDeviceId()))
        {
            isWhitelisted = true;
        }
        else if (whitelist->isWhitelisted(gpu->getDevicePciDeviceId(), gpu->getDevicePciSubsystemId()))
        {
            isWhitelisted = true;
            gpu->setUseSsid(true);
        }
        else
        {
            isWhitelisted = false;
        }
        std::string gpuBrand = gpu->getDeviceBrandAsString();
        uint64_t gpuArch     = gpu->getDeviceArchitecture();

        if (!nvvsCommon.fakegpusString.empty())
        {
            PRINT_DEBUG("%s", "attempting to use fake gpus: %s", nvvsCommon.fakegpusString.c_str());
            DcgmEntityStatus_t status;
            dcgmSystem.GetGpuStatus(gpuIds[i], &status);
            gpu->setDeviceEntityStatus(status);

            PRINT_DEBUG("%d %d", "status of gpu %d is %d", gpuIds[i], status);
            /* TODO: check return */
            /* How to determine if gpu is fake? */
            if (status == DcgmEntityStatusFake)
            {
                PRINT_DEBUG("%u %s %u",
                            "dcgmIndex %u, brand %s, arch %u is supported (only supporting fake gpus)",
                            gpuIds[i],
                            gpuBrand.c_str(),
                            static_cast<unsigned int>(gpuArch));
                gpu->setDeviceIsSupported(true);
            }
            else
            {
                PRINT_DEBUG("%u %s %u",
                            "dcgmIndex %u, brand %s, arch %u is not supported (only supporting fake gpus)",
                            gpuIds[i],
                            gpuBrand.c_str(),
                            static_cast<unsigned int>(gpuArch));
            }
        }
        else if (isWhitelisted)
        {
            PRINT_DEBUG("%u", "dcgmIndex %u is directly whitelisted.", gpuIds[i]);
            gpu->setDeviceIsSupported(true);
        }
        else if (HasGenericSupport(gpuBrand, gpuArch))
        {
            PRINT_DEBUG("%u %s %u",
                        "dcgmIndex %u, brand %s, arch %u is supported",
                        gpuIds[i],
                        gpuBrand.c_str(),
                        static_cast<unsigned int>(gpuArch));
            gpu->setDeviceIsSupported(true);
        }
        else
        {
            PRINT_DEBUG("%u %s %u",
                        "dcgmIndex %u, brand %s, arch %u is NOT supported",
                        gpuIds[i],
                        gpuBrand.c_str(),
                        static_cast<unsigned int>(gpuArch));
            gpu->setDeviceIsSupported(false);
        }

        if (gpu->getDeviceIsSupported())
        {
            m_gpuVect.push_back(gpu.release());
        }
        else
        {
            std::stringstream ss;
            ss << "\t"
               << "[" << gpu->getDevicePciBusId() << "] -- " << gpu->getDeviceName() << " -- Not Supported";
            PRINT_INFO("%s", "%s", ss.str().c_str());
        }
    }

    /* Allow the whitelist to adjust itself now that GPUs have been read in */
    whitelist->postProcessWhitelist(m_gpuVect);
}

/*****************************************************************************/
void NvidiaValidationSuite::overrideParameters(TestParameters *tp, const std::string &lowerCaseTestName)
{
    if (nvvsCommon.parms.find(lowerCaseTestName) != nvvsCommon.parms.end())
    {
        for (std::map<std::string, std::string>::iterator it = nvvsCommon.parms[lowerCaseTestName].begin();
             it != nvvsCommon.parms[lowerCaseTestName].end();
             it++)
        {
            tp->OverrideFromString(it->first, it->second);
        }
    }
}

/*****************************************************************************/
void NvidiaValidationSuite::InitializeAndCheckGpuObjs(std::vector<std::unique_ptr<GpuSet>> &gpuSets)
{
    for (unsigned int i = 0; i < gpuSets.size(); i++)
    {
        if (!gpuSets[i]->properties.present)
        {
            gpuSets[i]->gpuObjs = m_gpuVect;
        }
        else
        {
            gpuSets[i]->gpuObjs = decipherProperties(gpuSets[i].get());
        }

        if (gpuSets[i]->gpuObjs.size() == 0)
        { // nothing matched
            std::ostringstream ss;
            ss << "Unable to match GPU set '" << gpuSets[i]->name << "' to any GPU(s) on the system.";
            PRINT_ERROR("%s", "%s", ss.str().c_str());
            throw std::runtime_error(ss.str());
        }

        // ensure homogeneity
        std::string firstName = gpuSets[i]->gpuObjs[0]->getDeviceName();
        for (std::vector<Gpu *>::iterator gpuIt = gpuSets[i]->gpuObjs.begin(); gpuIt != gpuSets[i]->gpuObjs.end();
             gpuIt++)
        {
            // no need to check the first but...
            if (firstName != (*gpuIt)->getDeviceName())
            {
                std::ostringstream ss;
                ss << "NVVS does not support running on non-homogeneous GPUs during a single run. ";
                ss << "Please use the -i option to specify a list of identical GPUs. ";
                ss << "Run nvvs -g to list the GPUs on the system. Run nvvs --help for additional usage info. ";
                PRINT_ERROR("%s", "%s", ss.str().c_str());
                throw std::runtime_error(ss.str());
            }
        }
    }

    if (gpuSets.empty())
    {
        std::string err("No GPUs were found on which to run the tests.");
        throw std::runtime_error(err);
    }
}

/*****************************************************************************/
// take our GPU sets vector and fill in the appropriate GPU objects that match that set
void NvidiaValidationSuite::CheckGpuSetTests(std::vector<std::unique_ptr<GpuSet>> &gpuSets)
{
    // The rules are:
    // a) the "properties" struct is exclusionary. If properties is empty (properties.present == false)
    //    then all available GPU objects are included in the set
    // b) the "tests" vector is also exclusionary. If tests.size() == 0 then all available test
    //    objects are included in the set
    for (unsigned int i = 0; i < gpuSets.size(); i++)
    {
        bool first_pass = true;

        // go through the vector of tests requested and try to match them with an actual test.
        // push a warning if no match found
        for (std::vector<std::map<std::string, std::string>>::iterator reqIt = gpuSets[i]->testsRequested.begin();
             reqIt != gpuSets[i]->testsRequested.end();
             ++reqIt)
        {
            bool found                    = false;
            std::string requestedTestName = (*reqIt)["name"];
            std::string compareTestName   = requestedTestName;
            std::transform(compareTestName.begin(), compareTestName.end(), compareTestName.begin(), ::tolower);

            // first check the test suite names
            suiteNames_enum suite;
            if (compareTestName == "quick" || compareTestName == "short")
            {
                found = true;
                suite = NVVS_SUITE_QUICK;
            }
            else if (compareTestName == "medium")
            {
                found = true;
                suite = NVVS_SUITE_MEDIUM;
            }
            else if (compareTestName == "long")
            {
                found = true;
                suite = NVVS_SUITE_LONG;
            }

            if (found)
            {
                fillTestVectors(suite, Test::NVVS_CLASS_SOFTWARE, gpuSets[i].get());
                fillTestVectors(suite, Test::NVVS_CLASS_HARDWARE, gpuSets[i].get());
                fillTestVectors(suite, Test::NVVS_CLASS_INTEGRATION, gpuSets[i].get());
                fillTestVectors(suite, Test::NVVS_CLASS_PERFORMANCE, gpuSets[i].get());
            }
            // then check the test groups
            else
            {
                if (first_pass == true)
                {
                    fillTestVectors(NVVS_SUITE_CUSTOM, Test::NVVS_CLASS_SOFTWARE, gpuSets[i].get());
                    first_pass = false;
                }
                std::map<std::string, std::vector<Test *>> groups       = m_tf->getTestGroups();
                std::map<std::string, std::vector<Test *>>::iterator it = groups.find(requestedTestName);

                if (it != groups.end())
                {
                    found = true;

                    // Add each test from the list
                    for (size_t i = 0; i < groups[requestedTestName].size(); i++)
                        gpuSets[i]->AddTestObject(CUSTOM_TEST_OBJS, groups[requestedTestName][i]);
                }
                else // now check individual tests
                {
                    for (std::vector<Test *>::iterator testIt = testVect.begin(); testIt != testVect.end(); ++testIt)
                    {
                        // convert everything to lower case for comparison
                        std::string compareTestName = (*testIt)->GetTestName();
                        std::transform(
                            compareTestName.begin(), compareTestName.end(), compareTestName.begin(), ::tolower);
                        std::string compareRequestedName
                            = m_tf->GetCompareName(Test::NVVS_CLASS_CUSTOM, requestedTestName);

                        if (compareTestName == compareRequestedName)
                        {
                            found = true;
                            // Make a full copy of the test parameters
                            TestParameters *tp = new TestParameters();
                            tpVect.push_back(tp); // purely for accounting when we go to cleanup


                            whitelist->getDefaultsByDeviceId(
                                compareRequestedName, gpuSets[i]->gpuObjs[0]->getDeviceId(), tp);

                            if (nvvsCommon.parms.size() > 0)
                            {
                                overrideParameters(tp, compareRequestedName);
                            }

                            tp->AddString(PS_PLUGIN_NAME, (*testIt)->GetTestName());
                            tp->AddDouble(PS_LOGFILE_TYPE, (double)nvvsCommon.logFileType);

                            (*testIt)->pushArgVectorElement(Test::NVVS_CLASS_CUSTOM, tp);
                            gpuSets[i]->AddTestObject(CUSTOM_TEST_OBJS, (*testIt));
                            break;
                        }
                    }
                }
            }

            if (!found)
            {
                std::stringstream ss;
                ss << "Error: requested test \"" << requestedTestName << "\" was not found among possible test choices."
                   << std::endl;
                PRINT_ERROR("%s", "%s", ss.str().c_str());
                throw std::runtime_error(ss.str());
            }
        }
    }
}

/*****************************************************************************/
void NvidiaValidationSuite::fillTestVectors(suiteNames_enum suite, Test::testClasses_enum testClass, GpuSet *set)
{
    int type;
    std::vector<std::string> testNames;

    switch (testClass)
    {
        case Test::NVVS_CLASS_SOFTWARE:
            testNames.push_back("Blacklist");
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
            type = SOFTWARE_TEST_OBJS;
            break;
        case Test::NVVS_CLASS_HARDWARE:
            if (suite == NVVS_SUITE_MEDIUM || suite == NVVS_SUITE_LONG)
                testNames.push_back(MEMORY_PLUGIN_NAME);
            if (suite == NVVS_SUITE_LONG)
                testNames.push_back(DIAGNOSTIC_PLUGIN_NAME);
            type = HARDWARE_TEST_OBJS;
            break;
        case Test::NVVS_CLASS_INTEGRATION:
            if (suite == NVVS_SUITE_MEDIUM || suite == NVVS_SUITE_LONG)
                testNames.push_back(PCIE_PLUGIN_NAME);
            type = INTEGRATION_TEST_OBJS;
            break;
        case Test::NVVS_CLASS_PERFORMANCE:
            if (suite == NVVS_SUITE_LONG)
            {
                testNames.push_back(MEMBW_PLUGIN_NAME);
                testNames.push_back(SMSTRESS_PLUGIN_NAME);
                testNames.push_back(TS_PLUGIN_NAME);
                testNames.push_back(TP_PLUGIN_NAME);
            }
            type = PERFORMANCE_TEST_OBJS;
            break;
        default:
        {
            std::stringstream buf;
            buf << "Received test class '" << testClass << "' that is not valid.";
            throw std::runtime_error(buf.str());
            break;
        }
    }

    for (std::vector<std::string>::iterator it = testNames.begin(); it != testNames.end(); it++)
    {
        std::vector<Test *>::iterator testIt;
        if (testClass != Test::NVVS_CLASS_SOFTWARE)
            testIt = FindTestName(testClass, *it);
        else
        {
            testIt = FindTestName(testClass, SW_PLUGIN_NAME);
            if (testIt == testVect.end())
            {
                throw std::runtime_error(
                    "The software deployment program was not properly loaded. Please check the plugin path and that the plugins are valid.");
            }
        }

        if (testIt != testVect.end())
        {
            TestParameters *tp = new TestParameters();
            tpVect.push_back(tp); // purely for accounting when we go to cleanup

            if (testClass != Test::NVVS_CLASS_SOFTWARE)
            {
                // for uniformity downstream
                /*
                std::string lowerCaseTestName = *it;
                std::transform(
                    lowerCaseTestName.begin(), lowerCaseTestName.end(), lowerCaseTestName.begin(), ::tolower);
                */

                // pull just the first GPU device ID since they are all meant to be the same at this point
                whitelist->getDefaultsByDeviceId(*it, set->gpuObjs[0]->getDeviceId(), tp);

                if (nvvsCommon.parms.size() > 0)
                    overrideParameters(tp, *it);
            }

            tp->AddString(PS_PLUGIN_NAME, (*it));
            tp->AddDouble(PS_LOGFILE_TYPE, (double)nvvsCommon.logFileType);


            (*testIt)->pushArgVectorElement(testClass, tp);
            set->AddTestObject(type, *testIt);
        }
    }
}

/*****************************************************************************/
std::vector<Test *>::iterator NvidiaValidationSuite::FindTestName(Test::testClasses_enum testClass,
                                                                  std::string testName)
{
    std::string compareName = m_tf->GetCompareName(testClass, testName);

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
        if (set->properties.brand.length() > 0)
            brand = true;
        if (set->properties.name.length() > 0)
        {
            name = true;
            // kludge to handle special naming of K10
            if (set->properties.name == "Tesla K10")
                set->properties.name = "Tesla K10.G1.8GB";
        }

        if (set->properties.uuid.length() > 0)
        {
            if (set->properties.uuid == (*it)->getDeviceGpuUuid())
                tempGpuVec.push_back(*it);
            ++it;
            continue; // skip everything else
        }
        else if (set->properties.busid.length() > 0)
        {
            if (set->properties.busid == (*it)->getDevicePciBusId())
                tempGpuVec.push_back(*it);
            ++it;
            continue; // skip everything else
        }
        else if (set->properties.index.size() > 0)
        {
            for (unsigned int i = 0; i < set->properties.index.size(); i++)
            {
                if (set->properties.index[i] == (*it)->getDeviceIndex())
                {
                    if (!brand && !name)
                        tempGpuVec.push_back(*it);
                    if (brand && !name && set->properties.brand == (*it)->getDeviceBrandAsString())
                        tempGpuVec.push_back(*it);
                    if (name && !brand && set->properties.name == (*it)->getDeviceName())
                        tempGpuVec.push_back(*it);
                    if (brand && name && set->properties.brand == (*it)->getDeviceBrandAsString()
                        && set->properties.name == (*it)->getDeviceName())
                        tempGpuVec.push_back(*it);
                }
            }
        }
        else if (brand || name)
        {
            if (brand && !name && set->properties.brand == (*it)->getDeviceBrandAsString())
                tempGpuVec.push_back(*it);
            if (name && !brand && set->properties.name == (*it)->getDeviceName())
                tempGpuVec.push_back(*it);
            if (brand && name && set->properties.brand == (*it)->getDeviceBrandAsString()
                && set->properties.name == (*it)->getDeviceName())
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
                    buf << "test '" << testName << "' does not exist.";
                    throw std::runtime_error(buf.str());
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

                std::string requestedName                 = m_tf->GetCompareName(Test::NVVS_CLASS_CUSTOM, testName);
                nvvsCommon.parms[requestedName][parmName] = parmValue;
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
    virtual void usage(TCLAP::CmdLineInterface &_cmd)
    {
        TCLAP::StdOutput::usage(_cmd);

        std::cout << "Please email cudatools@nvidia.com with any questions, bug reports, etc." << std::endl
                  << std::endl;
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
        TCLAP::CmdLine cmd(NVVS_NAME, ' ', DRIVER_VERSION);
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
        TCLAP::SwitchArg jsonArg(
            "j", "jsonOutput", "Format output as json. Note: prevents progress updates.", cmd, false);
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
        TCLAP::SwitchArg fromDcgmArg("z",
                                     "from-dcgm",
                                     "Specify that this was launched by dcgmi diag and not from invoking nvvs directly",
                                     cmd,
                                     false);
        TCLAP::SwitchArg trainArg(
            "", "train", "Train NVVS to generate golden values for this system's configuration.", cmd, false);
        TCLAP::SwitchArg forceTrainArg(
            "", "force", "Train NVVS for golden values despite warnings to the contrary", cmd, false);
        TCLAP::ValueArg<unsigned int> trainingIterations("",
                                                         "training-iterations",
                                                         "The number of iterations to "
                                                         "use while training the diagnostic. The default is "
                                                         "4.",
                                                         false,
                                                         4,
                                                         "training iterations",
                                                         cmd);
        TCLAP::ValueArg<unsigned int> trainingVariance("",
                                                       "training-variance",
                                                       "The amount of variance - after "
                                                       "normalizing the data - required to trust the data. "
                                                       "The default is 5",
                                                       false,
                                                       5,
                                                       "training variance",
                                                       cmd);
        TCLAP::ValueArg<unsigned int> trainingTolerance("",
                                                        "training-tolerance",
                                                        "The percentage the golden "
                                                        "value should be scaled to allow some tolerance when "
                                                        "running the diagnostic later. For example, if the "
                                                        "calculated golden value for a minimum bandwidth were "
                                                        "9000 and the tolerance were set to 5, then the "
                                                        "minimum bandwidth written to the configuration file "
                                                        "would be 8550, 95% of 9000. The default value is 5.",
                                                        false,
                                                        5,
                                                        "training tolerance",
                                                        cmd);
        TCLAP::ValueArg<std::string> goldenValuesFile("",
                                                      "golden-values-filename",
                                                      "Specify the path where the "
                                                      "DCGM GPU diagnostic should save the golden values file "
                                                      "produced in training mode.",
                                                      false,
                                                      "/tmp/golden_values.yml",
                                                      "path to golden values file",
                                                      cmd);

        TCLAP::ValueArg<std::string> throttleMask(
            "",
            "throttle-mask",
            "Specify which throttling reasons should be ignored. You can provide a comma separated list of reasons. "
            "For example, specifying 'HW_SLOWDOWN,SW_THERMAL' would ignore the HW_SLOWDOWN and SW_THERMAL throttling "
            "reasons. Alternatively, you can specify the integer value of the ignore bitmask. For the bitmask, "
            "multiple reasons may be specified by the sum of their bit masks. For "
            "example, specifying '40' would ignore the HW_SLOWDOWN and SW_THERMAL throttling reasons.\n"
            "Valid throttling reasons and their corresponding bitmasks (given in parentheses) are:\n"
            "HW_SLOWDOWN (8)\nSW_THERMAL (32)\nHW_THERMAL (64)\nHW_POWER_BRAKE (128)",
            false,
            "",
            "throttle reasons to ignore",
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


        cmd.parse(argc, argv);

        configFileArg = configArg.getValue();
        if (configFileArg.size() > 0)
        {
            configFile = configFileArg;
        }

        listGpus                         = listGpusArg.getValue();
        listTests                        = listTestsArg.getValue();
        nvvsCommon.verbose               = verboseArg.getValue();
        nvvsCommon.pluginPath            = pluginPathArg.getValue();
        nvvsCommon.parse                 = parseArg.getValue();
        nvvsCommon.quietMode             = quietModeArg.getValue();
        nvvsCommon.configless            = configLessArg.getValue();
        nvvsCommon.fakegpusString        = fakeGpusArg.getValue();
        nvvsCommon.statsOnlyOnFail       = statsOnFailArg.getValue();
        nvvsCommon.indexString           = indexArg.getValue();
        nvvsCommon.parmsString           = parms.getValue();
        nvvsCommon.jsonOutput            = jsonArg.getValue();
        nvvsCommon.dcgmHostname          = dcgmHost.getValue();
        nvvsCommon.fromDcgm              = fromDcgmArg.getValue();
        nvvsCommon.training              = trainArg.getValue();
        nvvsCommon.forceTraining         = forceTrainArg.getValue();
        nvvsCommon.trainingIterations    = trainingIterations.getValue();
        nvvsCommon.trainingVariancePcnt  = trainingVariance.getValue() / 100.0;
        nvvsCommon.trainingTolerancePcnt = trainingTolerance.getValue() / 100.0;
        nvvsCommon.goldenValuesFile      = goldenValuesFile.getValue();
        nvvsCommon.SetStatsPath(statsPathArg.getValue());

        this->initWaitTime = initializationWaitTime.getValue();

        if (listGpus || listTests)
            nvvsCommon.configless = true;

        if (nvvsCommon.desiredTest.size() > 0)
            nvvsCommon.configless = true;

        initializeDesiredTests(specificTestArg.getValue());

        debugFile = DcgmLogging::getLogFilenameFromArgAndEnv(
            debugFileArg.getValue(), NVVS_LOGGING_DEFAULT_NVVS_LOGFILE, NVVS_ENV_LOG_PREFIX);

        debugLogLevel = DcgmLogging::getLogSeverityFromArgAndEnv(
            debugLevelArg.getValue(), DCGM_LOGGING_DEFAULT_NVVS_SEVERITY, NVVS_ENV_LOG_PREFIX);

        if (hwdiaglogfileArg.isSet())
            hwDiagLogFile = hwdiaglogfileArg.getValue();

        // Set bitmask for ignoring user specified throttling reasons
        if (throttleMask.isSet())
        {
            std::string reasonStr = throttleMask.getValue();
            // Make reasonStr lower case for parsing
            std::transform(reasonStr.begin(), reasonStr.end(), reasonStr.begin(), ::tolower);
            nvvsCommon.throttleIgnoreMask = GetThrottleIgnoreReasonMaskFromString(reasonStr);
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
    }
    catch (TCLAP::ArgException &e)
    {
        std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl;
        throw std::runtime_error("An error occurred trying to parse the command line.");
    }
}
