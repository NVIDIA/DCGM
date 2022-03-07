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
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/*
 * File:   TestAllocator.h
 * Author: ankjain
 *
 * Created on July 31, 2017, 4:06 PM
 */
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "DcgmCacheManager.h"
#include "TestDcgmModule.h"
#include "dcgm_fields.h"
#include "timelib.h"


#ifndef TESTALLOCATOR_H
#define TESTALLOCATOR_H

class TestAllocator : public TestDcgmModule
{
public:
    TestAllocator();
    virtual ~TestAllocator();

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
    int testAllocateVarSize_FreeInorder();
    int testAllocateVarSize_FreeReverse();
    int testAllocateFixedSize_FreeInorder();
    int testAllocateFixedSize_FreeReverse();
    int testAllocateVarSize_FreeSerial();
    int testAllocateFixedSize_FreeSerial();
    int testInternalFragFixedSize();
};

#endif /* TESTALLOCATOR_H */
