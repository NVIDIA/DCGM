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
 * File:   TestProtobuf.h
 */

#ifndef TESTPROTOBUF_H
#define TESTPROTOBUF_H

#include "DcgmProtobuf.h"
#include "TestDcgmModule.h"

class TestProtobuf : public TestDcgmModule
{
public:
    TestProtobuf();
    virtual ~TestProtobuf();

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
     * This method is used to test encoding/decoding of basic types int, double
     *****************************************************************************/
    int TestExchangeBasicTypes();

    /*****************************************************************************
     * This method is used to test encoding/decoding of structs
     *****************************************************************************/
    int TestExchangeStructs();

    /*****************************************************************************
     * This method is used to test encoding/decoding of batch of commands
     *****************************************************************************/
    int TestExchangeBatchCommands();

    /*************************************************************************/

    int TestFieldEncodeDecode(int fieldType, int valType);
    int UpdateFieldValueToTx(dcgm::FieldValue *pFieldValue, unsigned int fieldType, unsigned int valType);
    int VerifyDecodedCommand(dcgm::Command *pcmd, int fieldType, int valType);
};

#endif /* TESTPROTOBUF_H */
