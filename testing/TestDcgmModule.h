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
#ifndef TESTNVCMMODULE_H
#define TESTNVCMMODULE_H

#include "dcgm_structs.h"
#include <string>
#include <vector>

/*****************************************************************************/
typedef struct test_nvcm_gpu_t
{
    unsigned int gpuId;     /* NVCM GPU id. use DcgmSettings to convert to nvmlIndex */
    unsigned int nvmlIndex; /* NVML index */
} test_nvcm_gpu_t, *test_nvcm_gpu_p;

/*****************************************************************************/
/*
 * Base class for a Dcgm test module
 *
 * Note: if you add an instance of this class, make sure it gets loaded in
 *       TestDcgmUnitTests::LoadModules() so that it runs
 */

class TestDcgmModule
{
public:
    /*
     * Placeholder constructor and destructor. You do not need to call these
     *
     */
    TestDcgmModule()
        : m_dcgmHandle((dcgmHandle_t) nullptr)
    {}
    virtual ~TestDcgmModule()
    {} /* Virtual to satisfy ancient GCC */

    /*************************************************************************/
    /*
     * Method child classes should implement to initialize an instance of a TestDcgmModule
     * class. When this method is done, Run() should be able to be called
     *
     * All initialization code of your module that depends on NVML or NVCM being
     * at least globally initialized should be in here, as those are not guaranteed
     * to be loaded when your class is instantiated
     *
     * argv represents all of the arguments to the test framework that weren't
     * already parsed globally and are thus assumed to be module parameters
     *
     * gpus represents which GPUs to try to run on as discovered/parsed from
     * the command line
     *
     * Returns 0 on success
     *        !0 on error
     */
    virtual int Init(std::vector<std::string> argv, std::vector<test_nvcm_gpu_t> gpus) = 0;

    /*************************************************************************/
    /*
     * Run the tests of this module
     *
     * Returns  0 on success
     *         <0 on fatal failure. Will abort entire framework
     *         >0 on test failure. Will keep running other tests.
     *
     */
    virtual int Run() = 0;

    /*************************************************************************/
    /*
     * Tell this module to clean up after itself. Your destructor should
     * also call this method
     *
     */
    virtual int Cleanup() = 0;

    /*************************************************************************/
    /*
     * Get the tag for this test. This tag should be how the module is referenced
     * in both command line and logging
     */
    virtual std::string GetTag() = 0;

    /*************************************************************************/
    /*
     * Should this module be included in the default list of modules run?
     *
     * Users can run all modules by passing -a to testdcgmunittests
     */
    virtual bool IncludeInDefaultList(void)
    {
        return true;
    }

    /*****************************************************************************/
    /**
     * Assign copy of DCGM Handle so that it can be used by the derived modules
     */
    void SetDcgmHandle(dcgmHandle_t dcgmHandle)
    {
        m_dcgmHandle = dcgmHandle;
    }

    /*****************************************************************************/
    /*
     * Get a boolean value as to whether this is a debug build of DCGM or not
     *
     * Returns: 0 if not a debug build
     *          1 if is a debug build
     *
     */
    int IsDebugBuild(void)
    {
#ifdef _DEBUG
        return 1;
#else
        return 0;
#endif
    }

    /*****************************************************************************/

protected:
    dcgmHandle_t m_dcgmHandle;
};


/*****************************************************************************/

#define RUN_MODULE_SUBTEST(class, test_name, count)                       \
    do                                                                    \
    {                                                                     \
        int st = test_name();                                             \
        switch (st)                                                       \
        {                                                                 \
            case -1:                                                      \
                fprintf(stderr, "%s::%s FAILED\n", #class, #test_name);   \
                return -1;                                                \
            case 1:                                                       \
                fprintf(stderr, "%s::%s WAIVED\n", #class, #test_name);   \
                count += 1;                                               \
                break;                                                    \
            case 0:                                                       \
            default:                                                      \
                fprintf(stderr, "%s::%s PASSED\n\n", #class, #test_name); \
        }                                                                 \
    } while (false)

#endif // TESTNVCMMODULE_H
