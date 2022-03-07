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
 * Defines the class structure for the highest level
 * object in the Nvidia Validation Suite.
 *
 * Add other docs in here as methods get defined
 *
 */

#pragma once

#include "ConfigFileParser_v2.h"
#include "Gpu.h"
#include "NvvsCommon.h"
#include "NvvsSystemChecker.h"
#include "ParameterValidator.h"
#include "Test.h"
#include "TestFramework.h"
#include "TestParameters.h"
#include "Whitelist.h"
#include <algorithm>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#define MIN_MAJOR_VERSION 346
#define MAX_MAJOR_VERSION 600

namespace DcgmNs::Nvvs
{
class NvidiaValidationSuite
{
public:
    // ctor/dtor
    NvidiaValidationSuite();
    ~NvidiaValidationSuite();

    /*
     * This method does all of the work: processing the command line, loading the config file,
     * loading the plugins, and running the plugins
     *
     * @return         "" on success
     *                 an error string on failure
     */
    std::string Go(int argc, char *argv[]);

    /*
     * Add the list of GPUs for the test to NvvsCommon
     * @param gpuIndices[in/out] - a list of GPU indicies to use for this run, this is populated if passed in empty
     * @param visibleGpus[in]    - the GPUs visible to NVVS
     * @return an error string if an error occurred, or an empty string on success
     **/
    std::string BuildCommonGpusList(std::vector<unsigned int> &gpuIndices, const std::vector<Gpu *> &visibleGpus);

    /**
     * Determines whether or not a GPU is included in this run of the diagnostic.
     *
     * @param gpuIndex[in]   - the index of the GPU in question
     * @param gpuIndices[in] - a list of the user-specified indices for this run. If this is empty, then the GPU is
     *                         always included.
     * @return true if the GPU is to be included, false otherwise.
     **/
    bool IsGpuIncluded(unsigned int gpuIndex, std::vector<unsigned int> &gpuIndices);

    // not private so they can be called in tests
protected:
    // methods
    void processCommandLine(int argc, char *argv[]);
    void EnumerateAllVisibleGpus();
    void enumerateAllVisibleTests();
    void ValidateSubtestParameters();
    void CheckGpuSetTests(std::vector<std::unique_ptr<GpuSet>> &gpuSets);
    void banner();
    void CheckDriverVersion();
    std::vector<Gpu *> decipherProperties(GpuSet *set);
    std::vector<Test *>::iterator FindTestName(Test::testClasses_enum testClass, std::string testName);
    void fillTestVectors(suiteNames_enum suite, Test::testClasses_enum testClass, GpuSet *set);
    void overrideParameters(TestParameters *tp, const std::string &lowerCaseTestName);
    void startTimer();
    void stopTimer();
    bool HasGenericSupport(const std::string &gpuBrand, uint64_t gpuArch);
    void InitializeAndCheckGpuObjs(std::vector<std::unique_ptr<GpuSet>> &gpuSets);
    void InitializeParameters(const std::string &parms, const ParameterValidator &pv);

    // vars
    bool logInit;
    std::vector<Gpu *> m_gpuVect;
    std::vector<Test *> testVect;
    std::vector<TestParameters *> tpVect;
    Whitelist *whitelist;
    FrameworkConfig fwcfg;

    // classes
    ConfigFileParser_v2 *parser;
    TestFramework *m_tf;

    // parsing variables
    std::string configFile;
    std::string debugFile;
    std::string debugLogLevel;
    std::string hwDiagLogFile;

    bool listTests;
    bool listGpus;
    timer_t initTimer;
    struct sigaction restoreSigAction;
    unsigned int initWaitTime;
    NvvsSystemChecker m_sysCheck;
    ParameterValidator m_pv;

    /***************************PROTECTED********************************/
protected:
};
} // namespace DcgmNs::Nvvs
