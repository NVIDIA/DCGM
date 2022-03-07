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
#ifndef _NVVS_NVVS_GpuSet_H_
#define _NVVS_NVVS_GpuSet_H_

#include "Gpu.h"
#include "Test.h"
#include <string>
#include <vector>

extern const int CUSTOM_TEST_OBJS;
extern const int SOFTWARE_TEST_OBJS;
extern const int HARDWARE_TEST_OBJS;
extern const int INTEGRATION_TEST_OBJS;
extern const int PERFORMANCE_TEST_OBJS;

class GpuSet
{
    /***************************PUBLIC***********************************/
public:
    // Default initializers?
    GpuSet();
    ~GpuSet() {};

    std::string name;
    struct Props
    {
        bool present;
        std::string brand;
        std::vector<unsigned int> index;
        std::string name;
        std::string busid;
        std::string uuid;
    };

    Props properties;

    std::vector<std::map<std::string, std::string>> testsRequested;
    std::vector<Gpu *> gpuObjs; // corresponding GPU objects

    std::vector<Test *> m_customTestObjs;      // user-specified test objects
    std::vector<Test *> m_softwareTestObjs;    // software-class test objects
    std::vector<Test *> m_hardwareTestObjs;    // hardware-class test objects
    std::vector<Test *> m_integrationTestObjs; // integration-class test objects
    std::vector<Test *> m_performanceTestObjs; // performance-class test objects

    int AddTestObject(int testClass, Test *test);

    /***************************PRIVATE**********************************/
private:
    /***************************PROTECTED********************************/
protected:
};

#endif //_NVVS_NVVS_GpuSet_H_
