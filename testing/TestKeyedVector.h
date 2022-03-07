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
#ifndef TESTKEYEDVECTOR_H
#define TESTKEYEDVECTOR_H

#include "TestDcgmModule.h"
#include "keyedvector.h"

/*****************************************************************************/
class TestKeyedVector : public TestDcgmModule
{
public:
    TestKeyedVector();
    ~TestKeyedVector();

    /*************************************************************************/
    /* Inherited methods from TestDcgmModule */
    int Init(std::vector<std::string> argv, std::vector<test_nvcm_gpu_t> gpus);
    int Run();
    int Cleanup();
    std::string GetTag();

    /*************************************************************************/
private:
    /*************************************************************************/
    /*
     * Sub function of TestFindByKey
     */
    int TestFindByKeyAtSize(int testNelems);

    /*************************************************************************/
    /*
     * Actual test cases. These should return a status like below
     *
     * Returns 0 on success
     *        <0 on fatal error. Will abort entire framework
     *        >0 on non-fatal error
     *
     **/
    int TestTimingLinearInsert();
    int TestTimingRandomInsert();
    int TestLinearRemoveForward();
    int TestLinearRemoveByRangeForward();
    int TestLinearInsertBackward();
    int TestRemoveRangeByCursor();
    int TestFindByKey();
    int TestFindByIndex();
    int TestLinearInsertForward();

    /*************************************************************************/
    /*
     * Helper for verifying the elements of the keyed vector are indeed in order
     *
     * Returns 0 if OK
     *        !0 if keyed vector is corrupt
     *
     */
    int HelperVerifyOrder(keyedvector_p kv);

    /*************************************************************************/
    /*
     * Helper for verifying the cached size of the keyed vector against the
     * computed size of the keyed vector
     *
     * Returns 0 if OK
     *        !0 if keyed vector is corrupt
     *
     */
    int HelperVerifySize(keyedvector_p kv);

    /*************************************************************************/
    /* Virtual method inherited from TestDcgmModule */
    bool IncludeInDefaultList(void)
    {
        return false;
    }

    /*************************************************************************/
};


#endif // TESTKEYEDVECTOR_H
