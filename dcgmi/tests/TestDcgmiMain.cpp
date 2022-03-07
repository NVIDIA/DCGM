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
#include <iostream>
#include <set>

#include "TestCommandOutputController.h"
#include "TestDcgmiModule.h"
#include "TestDiag.h"

// The main test file for dcgmi
int main(int argc, char *argv[])
{
    int totalFailures = 0;

    std::set<TestDcgmiModule *> testSet;

    // Add each test here
    testSet.insert(new TestDiag());
    testSet.insert(new TestCommandOutputController());

    std::cout << "******* Running Dcgmi Unit Tests *******" << std::endl;

    for (std::set<TestDcgmiModule *>::iterator it = testSet.begin(); it != testSet.end(); it++)
    {
        TestDcgmiModule *tdm = *it;
        std::cout << std::endl << "\t*** Testing " << tdm->GetTag() << " ***" << std::endl;
        int failed = tdm->Run();
        if (failed == 0)
        {
            std::cout << "PASSED all tests for " << tdm->GetTag() << std::endl;
        }
        else
        {
            std::cout << "\tFinished testing " << tdm->GetTag() << " with " << failed << " failures." << std::endl;
            totalFailures += failed;
        }

        delete tdm;
    }

    testSet.clear();

    std::cout << std::endl;

    if (totalFailures == 0)
    {
        std::cout << "******* PASSED All Dcgmi Unit Tests" << std::endl;
    }
    else
    {
        std::cout << "******* Finished Dcgmi Unit Tests With " << totalFailures << " Total Failures" << std::endl;
        std::cout << "******* All Tests Must Pass Before Submiting CL" << std::endl;
    }
}
