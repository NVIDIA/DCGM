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
 * File:   TestTopology.h
 */

#ifndef TESTTOPOLOGY_H
#define TESTTOPOLOGY_H

#include "TestDcgmModule.h"
#include "dcgm_agent.h"
#include "dcgm_structs.h"
#include "dcgm_structs_internal.h"

class TestTopology : public TestDcgmModule
{
public:
    TestTopology();
    virtual ~TestTopology();

    /*************************************************************************/
    /* Inherited methods from TestDcgmModule */
    int Init(std::vector<std::string> argv, std::vector<test_nvcm_gpu_t> gpus);
    int Run();
    int Cleanup();
    std::string GetTag();

private:
    int TestTopologyDevice();
    int TestTopologyGroup();

    std::vector<test_nvcm_gpu_t> m_gpus; /* List of GPUs to run on, copied in Init() */
};

#endif /* TESTVERSIONING_H */
