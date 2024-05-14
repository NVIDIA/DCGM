/*
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
#include "TestAllocator.h"
#include "TestCacheManager.h"
#include "TestDcgmConnections.h"
#include "TestDcgmModule.h"
#include "TestDcgmValue.h"
#include "TestDiagManager.h"
#include "TestDiagResponseWrapper.h"
#include "TestFieldGroups.h"
#include "TestGroupManager.h"
#include "TestHealthMonitor.h"
#include "TestKeyedVector.h"
#include "TestPolicyManager.h"
#include "TestTopology.h"
#include "TestVersioning.h"
#include "dcgm_test_apis.h"

#include "DcgmLogging.h"
#include "DcgmSettings.h"
#include "dcgm_agent.h"
#include "dcgm_fields.h"
#include "dcgm_structs.h"
#include <dcgm_nvml.h>
#include <map>
#include <string>
#include <vector>

/*****************************************************************************/
class TestDcgmUnitTests
{
public:
    /*************************************************************************/
    TestDcgmUnitTests()
    {
        m_embeddedStarted  = false;
        m_runAllModules    = false;
        m_moduleInitParams = {};
        m_onlyModulesToRun.clear();
        m_dcgmHandle        = (dcgmHandle_t) nullptr;
        m_startRemoteServer = false;
    }

    /*************************************************************************/
    ~TestDcgmUnitTests()
    {
        Cleanup();
    }

    /*************************************************************************/
    int LoadModules()
    {
        TestDcgmModule *module;
        std::string moduleTag;

        module = (TestDcgmModule *)(new TestGroupManager());
        module->SetDcgmHandle(m_dcgmHandle);
        moduleTag            = module->GetTag();
        m_modules[moduleTag] = module;

        module = (TestDcgmModule *)(new TestKeyedVector());
        module->SetDcgmHandle(m_dcgmHandle);
        moduleTag            = module->GetTag();
        m_modules[moduleTag] = module;

        module = (TestDcgmModule *)(new TestCacheManager());
        module->SetDcgmHandle(m_dcgmHandle);
        moduleTag            = module->GetTag();
        m_modules[moduleTag] = module;

        module = (TestDcgmModule *)(new TestFieldGroups());
        module->SetDcgmHandle(m_dcgmHandle);
        moduleTag            = module->GetTag();
        m_modules[moduleTag] = module;

        module = (TestDcgmModule *)(new TestVersioning());
        module->SetDcgmHandle(m_dcgmHandle);
        moduleTag            = module->GetTag();
        m_modules[moduleTag] = module;

        module = (TestDcgmModule *)(new TestDiagManager());
        module->SetDcgmHandle(m_dcgmHandle);
        moduleTag            = module->GetTag();
        m_modules[moduleTag] = module;

        module = (TestDcgmModule *)(new TestDcgmValue());
        module->SetDcgmHandle(m_dcgmHandle);
        moduleTag            = module->GetTag();
        m_modules[moduleTag] = module;

        module = (TestDcgmModule *)(new TestPolicyManager());
        module->SetDcgmHandle(m_dcgmHandle);
        moduleTag            = module->GetTag();
        m_modules[moduleTag] = module;

        module = (TestDcgmModule *)(new TestHealthMonitor());
        module->SetDcgmHandle(m_dcgmHandle);
        moduleTag            = module->GetTag();
        m_modules[moduleTag] = module;

        module = (TestDcgmModule *)(new TestTopology());
        module->SetDcgmHandle(m_dcgmHandle);
        moduleTag            = module->GetTag();
        m_modules[moduleTag] = module;
        module->SetDcgmHandle(m_dcgmHandle);
        moduleTag            = module->GetTag();
        m_modules[moduleTag] = module;

        module = (TestDcgmModule *)(new TestDcgmConnections());
        module->SetDcgmHandle(m_dcgmHandle);
        moduleTag            = module->GetTag();
        m_modules[moduleTag] = module;

        module = new TestDiagResponseWrapper();
        module->SetDcgmHandle(m_dcgmHandle);
        moduleTag            = module->GetTag();
        m_modules[moduleTag] = module;

        /* Copy above 4 lines to add your own module */
        return 0;
    }

    /*************************************************************************/
    int UnloadModules()
    {
        std::map<std::string, TestDcgmModule *>::iterator moduleIt;
        std::string moduleTag;
        TestDcgmModule *module;

        log_debug("Unloading {} modules", (int)m_modules.size());

        for (moduleIt = m_modules.begin(); moduleIt != m_modules.end(); moduleIt++)
        {
            moduleTag = (*moduleIt).first;
            module    = (*moduleIt).second;

            delete (module);
        }

        /* All of the contained pointers are invalid. Clear the contents */
        m_modules.clear();

        return 0;
    }

    /*************************************************************************/
    void Cleanup()
    {
        UnloadModules();
        dcgmShutdown();
    }

    /*************************************************************************/
    int StartAndConnectToRemoteDcgm(void)
    {
        int ret;

        ret = dcgmEngineRun(5555, (char *)"127.0.0.1", 1);
        if (ret)
        {
            fprintf(stderr, "dcgmServerRun returned %d\n", ret);
            return -1;
        }

        ret = dcgmConnect((char *)"127.0.0.1:5555", &m_dcgmHandle);
        if (ret)
        {
            fprintf(stderr, "dcgmConnect returned %d\n", ret);
            return -1;
        }

        return 0;
    }

    /*************************************************************************/
    int Init()
    {
        int st;

        /* Initialize DCGM */
        if (DCGM_ST_OK != dcgmInit())
        {
            fprintf(stderr, "DCGM could not initialize");
            return -1;
        }

        st = RestartHostEngine();
        if (st)
        {
            fprintf(stderr, "RestartHostEngine() returned %d\n", st);
            return -1;
        }

        /* Load all of the modules */
        st = LoadModules();
        if (st)
        {
            fprintf(stderr, "LoadModules() returned %d\n", st);
            return -1;
        }

        return 0;
    }

    /*************************************************************************/
    int RestartHostEngine()
    {
        dcgmReturn_t dcgmReturn;

        /* Stop the embedded HE if it's already running */
        if (m_embeddedStarted)
        {
            dcgmReturn = dcgmStopEmbedded(m_dcgmHandle);
            if (dcgmReturn != DCGM_ST_OK)
            {
                fprintf(stderr, "Got %d from dcgmStopEmbedded(). Continuing.", dcgmReturn);
            }

            m_embeddedStarted = false;
        }

        /* Embedded Mode */
        dcgmReturn = dcgmStartEmbedded(DCGM_OPERATION_MODE_MANUAL, &m_dcgmHandle);
        if (dcgmReturn != DCGM_ST_OK)
        {
            fprintf(stderr, "dcgmStartEmbedded() returned %d", dcgmReturn);
            return -1;
        }

        m_embeddedStarted = true;

        /* Discover all devices and recreate injected GPUs now that we've
           (re)started the host engine */
        int st = FindAllGpus();
        if (st != 0)
        {
            fprintf(stderr, "FindAllGpus() returned %d", st);
            return -1;
        }

        return 0;
    }

    /*************************************************************************/
    int ParseCommandLine(int argc, char *argv[])
    {
        int i;

        // argv[0] = program name. We don't care about it
        for (i = 1; i < argc; i++)
        {
            /* Parse any arguments. If they are processed here, continue
                     and don't put them into m_moduleArgs */
            if (!strcmp(argv[i], "-m"))
            {
                if (argc - i < 2)
                {
                    fprintf(stderr, "-m requires a 2nd parameter\n");
                    return -1;
                }
                i++; /* Move to the actual argument */

                printf("Adding %s to the list of modules to run\n", argv[i]);
                m_onlyModulesToRun[argv[i]] = 0;
                continue;
            }
            else if (!strcmp(argv[i], "-r"))
            {
                printf("Enabling a local remote DCGM.\n");
                m_startRemoteServer = true;
            }
            else if (!strcmp(argv[i], "-a"))
            {
                printf("Running all modules, including non-L0 ones.\n");
                m_runAllModules = true;
            }

            m_moduleInitParams.moduleArgs.push_back(std::string(argv[i]));
        }

        return 0;
    }

    /*************************************************************************/
    int FindAllGpus()
    {
        std::vector<unsigned int> gpuIds;
        dcgmReturn_t dcgmReturn;

        m_moduleInitParams.fakeGpuIds.clear();
        m_moduleInitParams.liveGpuIds.clear();

        int numDevices = DCGM_MAX_NUM_DEVICES;
        gpuIds.resize(numDevices);
        dcgmReturn = dcgmGetAllDevices(m_dcgmHandle, gpuIds.data(), &numDevices);
        if (dcgmReturn != DCGM_ST_OK)
        {
            fprintf(stderr, "Got unexpected dcgmReturn %d from dcgmGetAllDevices()", dcgmReturn);
            return -1;
        }
        gpuIds.resize(numDevices);

        if (gpuIds.size() < 1)
        {
            fprintf(stderr,
                    "No GPUs found. If you are testing on GPUs not on the allowlist, "
                    "set %s=1 in your environment",
                    DCGM_ENV_WL_BYPASS);
            return -1;
        }

        for (auto &gpuId : gpuIds)
        {
            /* Success. Record device */
            printf("Using GpuId %u\n", gpuId);
            m_moduleInitParams.liveGpuIds.push_back(gpuId);
        }

        /* Create two fake GPUs to test with as well */
        dcgmCreateFakeEntities_t cfe {};
        cfe.version     = dcgmCreateFakeEntities_version;
        cfe.numToCreate = 2;
        for (int i = 0; i < cfe.numToCreate; i++)
        {
            cfe.entityList[i].entity.entityGroupId = DCGM_FE_GPU;
        }

        dcgmReturn = dcgmCreateFakeEntities(m_dcgmHandle, &cfe);
        if (dcgmReturn != DCGM_ST_OK)
        {
            fprintf(stderr, "dcgmCreateFakeEntities() returned unexpected dcgmReturn %d", dcgmReturn);
            return -1;
        }

        for (int i = 0; i < cfe.numToCreate; i++)
        {
            unsigned int gpuId = cfe.entityList[i].entity.entityId;
            printf("Using FAKE GpuId %u\n", gpuId);
            m_moduleInitParams.fakeGpuIds.push_back(gpuId);
        }

        return 0;
    }

    /*************************************************************************/
    int RunOneModule(TestDcgmModule *module)
    {
        int st, runSt;
        TestDcgmModuleConfig config;

        module->GetConfig(config);

        if (config.restartEngineBefore)
        {
            RestartHostEngine();
        }

        st = module->Init(m_moduleInitParams);
        if (st)
        {
            fprintf(stderr, "Module init for %s failed with %d\n", module->GetTag().c_str(), st);
            return -1;
        }

        /* Actually run the module */
        runSt = module->Run();

        /* Clean-up unconditionally before dealing with the run status */
        module->Cleanup();

        if (config.restartEngineAfter)
        {
            RestartHostEngine();
        }

        if (runSt > 0)
        {
            fprintf(stderr, "Module %s had non-fatal failure.\n", module->GetTag().c_str());
            return 1;
        }
        else if (runSt < 0)
        {
            fprintf(stderr, "Module %s had FATAL failure st %d.\n", module->GetTag().c_str(), runSt);
            return -1;
        }

        return 0;
    }

    /*************************************************************************/
    int RunModules(void)
    {
        int st;
        int Nfailed = 0;

        /* Run all modules */
        TestDcgmModule *module;
        std::string moduleTag;

        /* Should we only run certain modules? */
        if (m_onlyModulesToRun.size() > 0)
        {
            std::map<std::string, int>::iterator onlyModuleIt;
            std::map<std::string, TestDcgmModule *>::iterator moduleIt;

            printf("Running %d modules\n", (int)m_onlyModulesToRun.size());

            for (onlyModuleIt = m_onlyModulesToRun.begin(); onlyModuleIt != m_onlyModulesToRun.end(); onlyModuleIt++)
            {
                moduleTag = onlyModuleIt->first;

                moduleIt = m_modules.find(moduleTag);
                if (moduleIt == m_modules.end())
                {
                    fprintf(stderr, "%s is not a valid module name", moduleTag.c_str());
                    return 1;
                }

                module = (*moduleIt).second;

                st = RunOneModule(module);
                if (st)
                    Nfailed++;
                if (st < 0)
                    break; /* Fatal error */
            }
        }
        else /* Run all modules */
        {
            std::map<std::string, TestDcgmModule *>::iterator moduleIt;

            printf("Running %d modules\n", (int)m_modules.size());

            for (moduleIt = m_modules.begin(); moduleIt != m_modules.end(); moduleIt++)
            {
                moduleTag = (*moduleIt).first;
                module    = (*moduleIt).second;

                if (!m_runAllModules && !module->IncludeInDefaultList())
                {
                    printf("Skipping module \"%s\" not included in default list. Pass -a to include all modules.\n",
                           moduleTag.c_str());
                    continue;
                }

                st = RunOneModule(module);
                if (st)
                    Nfailed++;
                if (st < 0)
                    break; /* Fatal error */
            }
        }


        if (Nfailed > 0)
        {
            fprintf(stderr, "%d modules had test failures\n", Nfailed);
            return 1;
        }

        printf("All modules PASSED\n");
        return 0;
    }

    /*************************************************************************/
    /*
     * Main entry point for this class
     */
    int Run(int argc, char *argv[])
    {
        int st;

        /* Parse command line before discovering GPUs in case we specify gpus on
         * the command line
         */
        st = ParseCommandLine(argc, argv);
        if (st)
            return -1;

        /* Do we want to be remote to our DCGM? */
        if (m_startRemoteServer)
        {
            st = StartAndConnectToRemoteDcgm();
            if (st)
                return -1;
        }

        st = RunModules();
        if (st)
            return -1;

        return 0;
    }

    /*************************************************************************/

private:
    dcgmHandle_t m_dcgmHandle; /* Handle to our host engine. Only valid if m_embeddedStarted == 1 */
    bool m_embeddedStarted;    /* Has an embedded host engine been started? 1=yes. 0=no */
    bool m_startRemoteServer;  /* Has a TCP/IP serverbeen started? 1=yes. 0=no (pass -r to the program) */
    bool m_runAllModules;      /* Should we run all modules discovered, even non-default modules? */

    TestDcgmModuleInitParams m_moduleInitParams;       /* Parameters passed to each module's Init() method */
    std::map<std::string, TestDcgmModule *> m_modules; /* Test modules to run, indexed by each
                                                         module's GetTag() */
    std::map<std::string, int> m_onlyModulesToRun;     /* Map of 'moduletag'=>0 of modules we are supposed to run.
                                                    Empty = run all modules */
};


/*****************************************************************************/
int main(int argc, char *argv[])
{
    int st    = 0;
    int retSt = 0;

    TestDcgmUnitTests *dcgmUnitTests = new TestDcgmUnitTests();

    /* Use fprintf(stderr) or printf() in this function because we're not sure if logging is initialized */

    st = dcgmUnitTests->Init();
    if (st)
    {
        fprintf(stderr, "dcgmUnitTests->Init() returned %d\n", st);
        delete dcgmUnitTests;
        return 1;
    }

    /* Actually run tests */
    retSt = dcgmUnitTests->Run(argc, argv);

    delete (dcgmUnitTests);
    dcgmUnitTests = 0;
    return retSt;
}
