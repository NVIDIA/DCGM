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
 * File:   TestActionManager.h
 */

#ifndef TESTDIAG_H
#define TESTDIAG_H

#include "TestDcgmiModule.h"
#include "dcgm_structs.h"
#include "json/json.h"

class TestDiag : public TestDcgmiModule
{
public:
    TestDiag();
    virtual ~TestDiag();

    /*************************************************************************/
    /* Inherited methods from TestNvcmModule */
    int Run();
    std::string GetTag();

private:
    /*************************************************************************/
    /*
     * Actual test cases. These should return a status like below
     *
     * Returns 0 on success
     *        <0 on fatal error. Will abort entire framework
     *        >0 on non-fatal error
     *
     **/
    int TestPopulateGpuList();
    int TestHelperGetPluginName();
    int TestHelperJsonAddResult();
    int TestHelperJsonAddBasicTests();
    int TestHelperJsonBuildOutput();
    int TestHelperJsonTestEntry(Json::Value &testEntry,
                                int gpuIndex,
                                const std::string &status,
                                const std::string &warning);
    int TestGetFailureResult();
};

#endif /* TESTDIAG_H */
