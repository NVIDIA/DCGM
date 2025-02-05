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
#include "Memory_wrapper.h"
#include "PluginStrings.h"
#include "memory_plugin.h"

/*****************************************************************************/
Memory::Memory(dcgmHandle_t handle)
    : m_handle(handle)
    , m_entityInfo(std::make_unique<dcgmDiagPluginEntityList_v1>())
{
    m_infoStruct.testIndex        = DCGM_MEMORY_INDEX;
    m_infoStruct.shortDescription = "This plugin will test the memory of a given GPU.";
    m_infoStruct.testCategories   = "";
    m_infoStruct.selfParallel     = false;
    m_infoStruct.logFileTag       = MEMORY_PLUGIN_NAME;

    TestParameters *tp = new TestParameters();
    tp->AddString(PS_RUN_IF_GOM_ENABLED, "True");
    tp->AddString(MEMORY_STR_IS_ALLOWED, "False");
    tp->AddString(MEMORY_STR_MIN_ALLOCATION_PERCENTAGE, "75.0");
    tp->AddString(MEMORY_L1TAG_STR_IS_ALLOWED, "False");
    tp->AddDouble(MEMORY_L1TAG_STR_TEST_DURATION, 1.0);
    tp->AddDouble(MEMORY_L1TAG_STR_TEST_LOOPS, 0);
    tp->AddDouble(MEMORY_L1TAG_STR_INNER_ITERATIONS, 1024);
    tp->AddDouble(MEMORY_L1TAG_STR_ERROR_LOG_LEN, 8192);
    tp->AddString(MEMORY_L1TAG_STR_DUMP_MISCOMPARES, "True");
    tp->AddDouble(MEMORY_L1TAG_STR_L1_CACHE_SIZE_KB_PER_SM, 0.0);
    tp->AddString(PS_LOGFILE, "stats_memory.json");
    tp->AddDouble(PS_LOGFILE_TYPE, 0.0);
    tp->AddString(PS_IGNORE_ERROR_CODES, "");
    m_infoStruct.defaultTestParameters = tp;
}

/*****************************************************************************/
void Memory::Go(std::string const &testName,
                dcgmDiagPluginEntityList_v1 const *entityInfo,
                unsigned int numParameters,
                dcgmDiagPluginTestParameter_t const *tpStruct)
{
    if (testName != GetMemoryTestName())
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
        DCGM_LOG_WARNING << "Plugin is using fake gpus";
        sleep(1);
        SetResult(testName, NVVS_RESULT_PASS);
        return;
    }

    TestParameters testParameters(*(m_infoStruct.defaultTestParameters));
    testParameters.SetFromStruct(numParameters, tpStruct);

    if (testParameters.GetBoolFromString(MEMORY_STR_IS_ALLOWED) == false)
    {
        DcgmError d { DcgmError::GpuIdTag::Unknown };
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_TEST_DISABLED, d, "Memory");
        AddInfo(testName, d.GetMessage());
        SetResult(testName, NVVS_RESULT_SKIP);
        return;
    }

    ParseIgnoreErrorCodesParam(testName, testParameters.GetString(PS_IGNORE_ERROR_CODES));

    unsigned const numEntities = std::min(
        entityInfo->numEntities, static_cast<unsigned>(sizeof(entityInfo->entities) / sizeof(entityInfo->entities[0])));
    for (unsigned int i = 0; i < numEntities; i++)
    {
        if (entityInfo->entities[i].entity.entityGroupId != DCGM_FE_GPU)
        {
            continue;
        }
        main_entry(entityInfo->entities[i], this, &testParameters);
    }
}

/*****************************************************************************/
dcgmHandle_t Memory::GetHandle()
{
    return m_handle;
}

std::string Memory::GetMemoryTestName() const
{
    return MEMORY_PLUGIN_NAME;
}