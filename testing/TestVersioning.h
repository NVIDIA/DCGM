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
 * File:   TestVersioning.h
 */

#ifndef TESTVERSIONING_H
#define TESTVERSIONING_H

#include "TestDcgmModule.h"
#include "dcgm_structs.h"
#include "dcgm_structs_internal.h"

class TestVersioning : public TestDcgmModule
{
public:
    TestVersioning();
    virtual ~TestVersioning();

    /*************************************************************************/
    /* Inherited methods from TestDcgmModule */
    int Init(std::vector<std::string> argv, std::vector<test_nvcm_gpu_t> gpus);
    int Run();
    int Cleanup();
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
    /*****************************************************************************
     * This method is used to test basic recognition of API versions
     *****************************************************************************/
    int TestBasicAPIVersions();

#if 0
    int AddInt32FieldToCommand(command_t *pcmd, int fieldType, int32_t val);
    int AddInt64FieldToCommand(command_t *pcmd, int fieldType, int64_t val);
    int AddDoubleFieldToCommand(command_t *pcmd, int fieldType, double val);
    int AddStrFieldToCommand(command_t *pcmd, int fieldType, char *pChar);
    int TestFieldEncodeDecode(int fieldType, int valType);

    int UpdateCommandToEncode(command_t *pcmd, unsigned int fieldType, unsigned int valType);
    int VerifyDecodedCommand(command_t *pcmd, int fieldType, int valType);
#endif
};

#endif /* TESTVERSIONING_H */
