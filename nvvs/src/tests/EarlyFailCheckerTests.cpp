/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

#include <catch2/catch_all.hpp>

#include <EarlyFailChecker.h>
#include <MockDcgmOps.h>
#include <NvvsCommon.h>

#include <climits>
#include <stdexcept>


namespace
{
struct EarlyFailCheckerTestFixture
{
    DcgmNs::MockDcgmOps mockDcgmOps;
    DcgmNs::DcgmOpsAnyPtr dcgmOps;
    NvvsCommon nvvsCommon;

    EarlyFailCheckerTestFixture()
        : dcgmOps(&mockDcgmOps)
    {}
};
} // namespace

namespace
{
class TestEarlyFailChecker : public EarlyFailChecker
{
public:
    using EarlyFailChecker::EarlyFailChecker;

    unsigned long LastCheckTime() const
    {
        return m_lastCheckTime;
    }

    std::vector<dcgmDiagPluginEntityInfo_v1> const &EntityInfos() const
    {
        return m_entityInfos;
    }
};

dcgmDiagPluginEntityList_v1 MakeEntityList(std::initializer_list<unsigned int> entityIds)
{
    dcgmDiagPluginEntityList_v1 entityList {};
    if (entityIds.size() > DCGM_GROUP_MAX_ENTITIES_V2)
    {
        throw std::runtime_error("Too many entities requested in MakeEntityList()");
    }
    entityList.numEntities = entityIds.size();

    unsigned int index = 0;
    for (auto entityId : entityIds)
    {
        entityList.entities[index].entity.entityGroupId = DCGM_FE_GPU;
        entityList.entities[index].entity.entityId      = entityId;
        ++index;
    }

    return entityList;
}
} // namespace

TEST_CASE_METHOD(EarlyFailCheckerTestFixture, "EarlyFailChecker copies entity info at construction")
{
    TestParameters tp;
    auto entityList = MakeEntityList({ 3, 7, 11 });

    TestEarlyFailChecker checker(&tp, true, 10, entityList);

    REQUIRE(checker.EntityInfos().size() == 3);
    CHECK(checker.EntityInfos()[0].entity.entityGroupId == DCGM_FE_GPU);
    for (unsigned int i = 0; i < entityList.numEntities; i++)
    {
        CHECK(checker.EntityInfos()[i].entity.entityId == entityList.entities[i].entity.entityId);
    }
    CHECK(checker.LastCheckTime() == ULONG_MAX);
    CHECK(checker.GetErrors().empty());
}

TEST_CASE_METHOD(EarlyFailCheckerTestFixture,
                 "EarlyFailChecker returns pass and preserves state when fail-early is disabled")
{
    TestParameters tp;
    auto entityList = MakeEntityList({ 5 });
    TestEarlyFailChecker checker(&tp, false, 1, entityList);
    DcgmRecorder recorder(mockDcgmOps.m_handleId, dcgmOps, &nvvsCommon);

    auto result = checker.CheckCommonErrors(100, 50, recorder);

    CHECK(result == NVVS_RESULT_PASS);
    CHECK(checker.LastCheckTime() == ULONG_MAX);
    CHECK(checker.GetErrors().empty());
}

TEST_CASE_METHOD(EarlyFailCheckerTestFixture,
                 "EarlyFailChecker does not poll recorder before the first interval elapses")
{
    TestParameters tp;
    auto entityList = MakeEntityList({ 1, 2 });
    TestEarlyFailChecker checker(&tp, true, 10, entityList);
    DcgmRecorder recorder(mockDcgmOps.m_handleId, dcgmOps, &nvvsCommon);

    auto result = checker.CheckCommonErrors(0, 0, recorder);

    CHECK(result == NVVS_RESULT_PASS);
    CHECK(checker.LastCheckTime() == ULONG_MAX);
    CHECK(checker.GetErrors().empty());
}
