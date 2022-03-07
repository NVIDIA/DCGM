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
#include "GpuSet.h"
#include "PluginStrings.h"
#include "Test.h"

const int CUSTOM_TEST_OBJS      = 0;
const int SOFTWARE_TEST_OBJS    = 1;
const int HARDWARE_TEST_OBJS    = 2;
const int INTEGRATION_TEST_OBJS = 3;
const int PERFORMANCE_TEST_OBJS = 4;

GpuSet::GpuSet()
    : name()
    , properties()
    , testsRequested()
    , gpuObjs()
    , m_customTestObjs()
    , m_softwareTestObjs()
    , m_hardwareTestObjs()
    , m_integrationTestObjs()
    , m_performanceTestObjs()
{
    properties.present = false;
}

int GpuSet::AddTestObject(int testClass, Test *test)
{
    if (!test)
        return -1;

    switch (testClass)
    {
        case CUSTOM_TEST_OBJS:
            m_customTestObjs.push_back(test);
            break;

        case SOFTWARE_TEST_OBJS:
            m_softwareTestObjs.push_back(test);
            break;

        case HARDWARE_TEST_OBJS:
            m_hardwareTestObjs.push_back(test);
            break;

        case INTEGRATION_TEST_OBJS:
            m_integrationTestObjs.push_back(test);
            break;

        case PERFORMANCE_TEST_OBJS:
            m_performanceTestObjs.push_back(test);
            break;

        default:
            return -1;
            break;
    }

    return 0;
}
