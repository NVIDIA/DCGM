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

#include "TestFieldGroups.h"
#include "DcgmWatcher.h"
#include "dcgm_fields.h"
#include "dcgm_structs.h"
#include "dcgm_structs_internal.h"
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>


/*****************************************************************************/
TestFieldGroups::TestFieldGroups()
{}

/*****************************************************************************/
TestFieldGroups::~TestFieldGroups()
{}

/*************************************************************************/
std::string TestFieldGroups::GetTag()
{
    return std::string("fieldgroups");
}

/*****************************************************************************/
int TestFieldGroups::Init(std::vector<std::string> argv, std::vector<test_nvcm_gpu_t> gpus)
{
    m_gpus = gpus;
    return 0;
}

/*****************************************************************************/
int TestFieldGroups::Cleanup()
{
    return 0;
}

/*****************************************************************************/
int TestFieldGroups::TestFieldGroupObject(void)
{
    int numErrors = 0;

    std::vector<unsigned short> fieldIdsBefore;
    std::vector<unsigned short> fieldIdsAfter;

    fieldIdsBefore.push_back(DCGM_FI_DEV_OEM_INFOROM_VER);
    fieldIdsBefore.push_back(DCGM_FI_DEV_SM_CLOCK);
    fieldIdsBefore.push_back(DCGM_FI_DEV_MEM_CLOCK);
    fieldIdsBefore.push_back(DCGM_FI_DRIVER_VERSION);
    fieldIdsBefore.push_back(DCGM_FI_DEV_COUNT);
    fieldIdsBefore.push_back(DCGM_FI_DEV_NAME);

    DcgmWatcher watcher(DcgmWatcherTypeClient);
    DcgmWatcher watcher2(DcgmWatcherTypeHostEngine);
    DcgmWatcher watcher3(DcgmWatcherTypeHostEngine);

    if (watcher == watcher2)
    {
        std::cerr << "TestFieldGroupObject watchers 1 and 2 should be different." << std::endl;
        numErrors++;
    }
    else
    {
        std::cerr << "TestFieldGroupObject 1 PASS" << std::endl;
    }

    if (watcher2 != watcher3)
    {
        std::cerr << "TestFieldGroupObject watchers 2 and 3 should be same." << std::endl;
        numErrors++;
    }
    else
    {
        std::cerr << "TestFieldGroupObject 2 PASS" << std::endl;
    }

    std::string groupNameBefore = "bestgroupever";

    DcgmFieldGroup *fieldGroup = new DcgmFieldGroup(1, fieldIdsBefore, groupNameBefore, watcher);

    std::string groupNameAfter = fieldGroup->GetName();
    if (groupNameBefore != groupNameAfter)
    {
        std::cerr << "TestFieldGroupObject groupNameBefore " << groupNameBefore << " != groupNameAfter "
                  << groupNameAfter << std::endl;
        numErrors++;
    }

    fieldGroup->GetFieldIds(fieldIdsAfter);

    if (fieldIdsAfter.size() != fieldIdsBefore.size())
    {
        std::cerr << "TestFieldGroupObject fieldIdsAfter.size() " << fieldIdsAfter.size()
                  << " != fieldIdsBefore.size() " << fieldIdsBefore.size() << std::endl;
        numErrors++;
    }

    if (fieldIdsAfter != fieldIdsBefore)
    {
        std::cerr << "fieldIdsAfter != fieldIdsBefore " << std::endl;
        numErrors++;
    }

    delete (fieldGroup);
    fieldGroup = 0;

    return numErrors;
}

/*****************************************************************************/
int TestFieldGroups::TestFieldGroupManager(void)
{
    DcgmFieldGroupManager *fieldGroupManager;
    dcgmReturn_t dcgmReturn;
    dcgmFieldGrp_t fieldGrpHandle = 0, fieldGrpHandleTemp = 0;
    int numErrors = 0;
    /* Use a fake connection ID to exercise the client paths */
    DcgmWatcher watcher(DcgmWatcherTypeClient, (dcgm_connection_id_t)1);

    std::vector<unsigned short> fieldIdsBefore;
    std::vector<unsigned short> fieldIdsAfter;

    std::string groupNameBefore = "bestgroupever";

    fieldIdsBefore.push_back(DCGM_FI_DEV_OEM_INFOROM_VER);
    fieldIdsBefore.push_back(DCGM_FI_DEV_SM_CLOCK);
    fieldIdsBefore.push_back(DCGM_FI_DEV_MEM_CLOCK);
    fieldIdsBefore.push_back(DCGM_FI_DRIVER_VERSION);
    fieldIdsBefore.push_back(DCGM_FI_DEV_COUNT);
    fieldIdsBefore.push_back(DCGM_FI_DEV_NAME);
    fieldIdsBefore.push_back(DCGM_FI_DEV_INFOROM_IMAGE_VER);

    fieldGroupManager = new DcgmFieldGroupManager();

    dcgmReturn = fieldGroupManager->AddFieldGroup(groupNameBefore, fieldIdsBefore, &fieldGrpHandle, watcher);
    if (dcgmReturn != DCGM_ST_OK)
    {
        std::cerr << "TestFieldGroupManager AddFieldGroup() returned " << (int)dcgmReturn << std::endl;
        delete (fieldGroupManager);
        return 1; /* Can't do much else without a group */
    }

    /* Inserting a duplicate group should fail */
    dcgmReturn = fieldGroupManager->AddFieldGroup(groupNameBefore, fieldIdsBefore, &fieldGrpHandleTemp, watcher);
    if (dcgmReturn != DCGM_ST_DUPLICATE_KEY)
    {
        std::cerr << "TestFieldGroupManager AddFieldGroup() returned " << (int)dcgmReturn << std::endl;
        numErrors++;
    }

    fieldIdsAfter.clear();
    dcgmReturn = fieldGroupManager->GetFieldGroupFields(fieldGrpHandle, fieldIdsAfter);
    if (dcgmReturn != DCGM_ST_OK)
    {
        std::cerr << "TestFieldGroupManager GetFieldGroupFields() returned " << (int)dcgmReturn << std::endl;
        numErrors++;
    }

    if (fieldIdsAfter.size() != fieldIdsBefore.size())
    {
        std::cerr << "TestFieldGroupObject fieldIdsAfter.size() " << fieldIdsAfter.size()
                  << " != fieldIdsBefore.size() " << fieldIdsBefore.size() << std::endl;
        numErrors++;
    }

    dcgmReturn = fieldGroupManager->RemoveFieldGroup(fieldGrpHandle, watcher);
    if (dcgmReturn != DCGM_ST_OK)
    {
        std::cerr << "TestFieldGroupManager RemoveFieldGroup() returned " << (int)dcgmReturn << std::endl;
        numErrors++;
    }

    /* Getting the group after we've deleted it should be an error */
    fieldIdsAfter.clear();
    dcgmReturn = fieldGroupManager->GetFieldGroupFields(fieldGrpHandle, fieldIdsAfter);
    if (dcgmReturn != DCGM_ST_NO_DATA)
    {
        std::cerr << "TestFieldGroupManager GetFieldGroupFields() returned " << (int)dcgmReturn << std::endl;
        numErrors++;
    }

    /* Removing the field group again should fail */
    dcgmReturn = fieldGroupManager->RemoveFieldGroup(fieldGrpHandle, watcher);
    if (dcgmReturn != DCGM_ST_NO_DATA)
    {
        std::cerr << "TestFieldGroupManager RemoveFieldGroup() returned " << (int)dcgmReturn << std::endl;
        numErrors++;
    }

    delete (fieldGroupManager);
    return numErrors;
}

/*****************************************************************************/
int TestFieldGroups::TestGetAll(void)
{
    int numErrors = 0;
    int i, j;
    dcgmAllFieldGroup_t beforeFieldGroup;
    dcgmAllFieldGroup_t afterFieldGroup;
    dcgmFieldGroupInfo_t *fgi;
    dcgmFieldGrp_t fieldGroupHandles[DCGM_MAX_NUM_FIELD_GROUPS] = { 0 };
    dcgmReturn_t dcgmReturn;
    DcgmFieldGroupManager *fieldGroupManager;
    DcgmWatcher watcher(DcgmWatcherTypeClient, (dcgm_connection_id_t)1);

    memset(&beforeFieldGroup, 0, sizeof(beforeFieldGroup));
    memset(&afterFieldGroup, 0, sizeof(afterFieldGroup));

    /* Create an allFieldGroup message, add its contents to a FieldGroupManager,
     * and verify that attempting to return the same object actually does
     */

    fieldGroupManager = new DcgmFieldGroupManager();

    beforeFieldGroup.version = dcgmAllFieldGroup_version;

    std::vector<unsigned short> fieldIds;

    for (i = 0; i < DCGM_MAX_NUM_FIELD_GROUPS; i++)
    {
        fgi = &beforeFieldGroup.fieldGroups[i];
        snprintf(fgi->fieldGroupName, sizeof(fgi->fieldGroupName) - 1, "bestgroupever%02d", i);

        fieldIds.clear();

        for (j = 0; j < DCGM_MAX_FIELD_IDS_PER_FIELD_GROUP; j++)
        {
            fgi->fieldIds[j] = i + j; /* Offset each list of fieldIds by i so they are different */
            fieldIds.push_back(fgi->fieldIds[j]);
            fgi->numFieldIds++;
        }

        dcgmReturn = fieldGroupManager->AddFieldGroup(fgi->fieldGroupName, fieldIds, &fieldGroupHandles[i], watcher);
        if (dcgmReturn != DCGM_ST_OK)
        {
            std::cerr << "TestFieldGroupManager AddFieldGroup() returned " << (int)dcgmReturn << std::endl;
            numErrors++;
        }

        fgi->fieldGroupId = fieldGroupHandles[i];

        beforeFieldGroup.numFieldGroups++;
    }

    /* Now, retrieve the full list of groups and validate it */
    dcgmReturn = fieldGroupManager->PopulateFieldGroupGetAll(&afterFieldGroup);
    if (dcgmReturn != DCGM_ST_OK)
    {
        std::cerr << "TestFieldGroupManager PopulateFieldGroupGetAll() returned " << (int)dcgmReturn << std::endl;
        numErrors++;
    }

    /* Short path: See if the structures are identical */
    if (memcmp(&beforeFieldGroup, &afterFieldGroup, sizeof(beforeFieldGroup)))
    {
        std::cerr << "TestFieldGroupManager before and after structures don't match" << std::endl;
        numErrors++;
    }

    delete (fieldGroupManager);
    return numErrors;
}

/*****************************************************************************/
void TestFieldGroups::CompleteTest(std::string testName, int testReturn, int &Nfailed)
{
    if (testReturn)
    {
        Nfailed++;
        std::cerr << "TestFieldGroups::" << testName << " FAILED with " << testReturn << std::endl;

        // fatal test failure
        if (testReturn < 0)
            throw std::runtime_error("fatal test failure");
    }
    else
    {
        std::cout << "TestFieldGroups::" << testName << " PASSED" << std::endl;
    }
}

/*****************************************************************************/
int TestFieldGroups::Run()
{
    int Nfailed = 0;

    try
    {
        CompleteTest("TestFieldGroupObject", TestFieldGroupObject(), Nfailed);
        CompleteTest("TestFieldGroupManager", TestFieldGroupManager(), Nfailed);
        CompleteTest("TestGetAll", TestGetAll(), Nfailed);
    }
    // fatal test return ocurred
    catch (const std::runtime_error &e)
    {
        return -1;
    }

    if (Nfailed > 0)
    {
        fprintf(stderr, "%d tests FAILED\n", Nfailed);
        return 1;
    }

    printf("All tests passed\n");

    return 0;
}

/*****************************************************************************/
