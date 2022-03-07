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
 * File:   TestGroupManager.h
 */

#ifndef TESTGROUPMANAGER_H
#define TESTGROUPMANAGER_H

#include "DcgmGroupManager.h"
#include "TestDcgmModule.h"
#include "dcgm_fields.h"
#include "timelib.h"


class TestGroupManager : public TestDcgmModule
{
public:
    TestGroupManager();
    virtual ~TestGroupManager();

    int Init(std::vector<std::string> argv, std::vector<test_nvcm_gpu_t> gpus);
    int Run();
    int Cleanup();
    std::string GetTag();

private:
    int TestGroupCreation();
    int TestGroupGetAllGrpIds();
    int TestGroupManageGpus();
    int TestGroupReportErrOnDuplicate();
    int TestDefaultGpusAreDynamic();

    int HelperOperationsOnGroup(DcgmGroupManager *pDcgmGrpManager,
                                unsigned int groupId,
                                std::string groupName,
                                DcgmCacheManager *cacheManager);

    std::vector<test_nvcm_gpu_t> m_gpus; /* List of GPUs to run on, copied in Init() */
};

#endif /* TESTGROUPMANAGER_H */
