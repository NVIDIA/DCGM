/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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
#include <ScopedEnvVars.hpp>
#include <ScopedWorkingDirectory.hpp>
#include <TempDir.hpp>
#include <catch2/catch_all.hpp>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <memory>
#include <optional>
#include <sstream>
#include <stdlib.h>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <DcgmModuleSysmon.h>
#include <DcgmMutex.h>

#include <DcgmCoreCommunication.h>

#include <tests/DcgmSysmonTestUtils.h>

using namespace DcgmNs;
using namespace DcgmNs::Cpu;

namespace
{
static constexpr char SupportNonNvidiaCpuEnv[] = "DCGM_SUPPORT_NON_NVIDIA_CPU";
static constexpr char HangDetectDisableEnv[]   = "DCGM_HANGDETECT_DISABLE";
static constexpr unsigned int MaxCpuCreateIds  = DCGM_MAX_CPU_CREATE_IDS;
} //namespace

struct CapturedAppendSamples
{
    DcgmMutex mutex { 0 };
    std::vector<char> buffer;
    unsigned int count = 0;

    void Capture(dcgmCoreAppendSamples_t const &appendSamples)
    {
        DcgmLockGuard lock(&mutex);
        buffer.assign(appendSamples.request.buffer, appendSamples.request.buffer + appendSamples.request.bufferSize);
        count++;
    }
};

std::unique_ptr<DcgmFvBuffer> GetCapturedAppendSamples(CapturedAppendSamples &capture)
{
    DcgmLockGuard lock(&capture.mutex);
    if (capture.buffer.empty())
    {
        return nullptr;
    }

    auto buffer = std::make_unique<DcgmFvBuffer>();
    if (buffer->SetFromBuffer(capture.buffer.data(), capture.buffer.size()) != DCGM_ST_OK)
    {
        return nullptr;
    }
    return buffer;
}

// The implementation depends structures based on these defines and indexes structures
// assuming the size of the arrays are a power of 2.
DCGM_CASSERT((DCGM_MAX_NUM_CPUS & (DCGM_MAX_NUM_CPUS - 1)) == 0, 1);
DCGM_CASSERT((DCGM_MAX_NUM_CPU_CORES & (DCGM_MAX_NUM_CPU_CORES - 1)) == 0, 1);

//noop
dcgmReturn_t postRequestToCore(dcgm_module_command_header_t *req, void *poster)
{
    if (req->moduleId == DcgmModuleIdCore && req->subCommand == DcgmCoreReqIdCMAppendSamples)
    {
        auto *appendSamples = reinterpret_cast<dcgmCoreAppendSamples_t *>(req);
        if (poster != nullptr)
        {
            static_cast<CapturedAppendSamples *>(poster)->Capture(*appendSamples);
        }
        appendSamples->ret = DCGM_ST_OK;
    }
    return DCGM_ST_OK;
}

dcgmCoreCallbacks_t g_coreCallbacks(dcgmCoreCallbacks_version, postRequestToCore, nullptr, nullptr);

/**
 * Test adapter that exposes protected sysmon helpers and sample state.
 */
class TestDcgmModuleSysmon : public DcgmModuleSysmon
{
public:
    explicit TestDcgmModuleSysmon(dcgmCoreCallbacks_t &dcc)
        : DcgmModuleSysmon(dcc, TaskRunnerStartMode::Defer)
    {
        StartTaskRunner();
    }

    TestDcgmModuleSysmon(dcgmCoreCallbacks_t &dcc, SysmonSystemMonitorAnyPtr systemMonitor)
        : DcgmModuleSysmon(dcc, systemMonitor, TaskRunnerStartMode::Defer)
    {
        StartTaskRunner();
    }

    TestDcgmModuleSysmon(dcgmCoreCallbacks_t &dcc,
                         SysmonSystemMonitorAnyPtr systemMonitor,
                         SysmonWatchTableAnyPtr watchTable)
        : DcgmModuleSysmon(dcc, systemMonitor, watchTable, TaskRunnerStartMode::Defer)
    {
        StartTaskRunner();
    }

    ~TestDcgmModuleSysmon() override
    {
        StopTaskRunner();
    }

    using DcgmModuleSysmon::AddUtilizationSample;
    using DcgmModuleSysmon::CalculateCoreUtilization;
    using DcgmModuleSysmon::GetSocketFromThermalZoneFileContents;
    using DcgmModuleSysmon::GetSubscribed;
    using DcgmModuleSysmon::GetUtilizationSampleSize;
    using DcgmModuleSysmon::m_coreSpeedBaseDir;
    using DcgmModuleSysmon::m_paused;
    using DcgmModuleSysmon::m_socketTemperatureCritFileMap;
    using DcgmModuleSysmon::m_socketTemperatureFileMap;
    using DcgmModuleSysmon::m_socketTemperatureWarnFileMap;
    using DcgmModuleSysmon::m_tzBaseDir;
    using DcgmModuleSysmon::ParseProcStatCpuLine;
    using DcgmModuleSysmon::PopulateTemperatureFileMap;
    using DcgmModuleSysmon::PruneSamples;
    using DcgmModuleSysmon::ReadCoreSpeed;
    using DcgmModuleSysmon::ReadTemperature;
    using DcgmModuleSysmon::RunOnce;
};

struct FakeSysmonFileSystem
{
    std::unordered_map<std::string, std::string> files;

    std::optional<std::string> ReadFirstToken(std::string const &path) const
    {
        auto it = files.find(path);
        if (it == files.end())
        {
            return std::nullopt;
        }
        std::istringstream stream(it->second);
        std::string token;
        stream >> token;
        return token;
    }
};

struct FakeSysmonSystemMonitor
{
    double cpuPower          = 123.5;
    double sysIoPower        = 45.25;
    double modulePower       = 67.75;
    double powerCap          = 250.0;
    dcgmReturn_t powerReturn = DCGM_ST_OK;
    std::string vendor       = "InjectedVendor";
    std::string model        = "InjectedModel";

    void Init()
    {}

    bool AreNvidiaCpusPresent() const
    {
        return true;
    }

    dcgmReturn_t GetCurrentCPUPowerUsage(unsigned int, double &usage)
    {
        if (powerReturn == DCGM_ST_OK)
        {
            usage = cpuPower;
        }
        return powerReturn;
    }

    dcgmReturn_t GetCurrentSysIOPowerUsage(unsigned int, double &usage)
    {
        if (powerReturn == DCGM_ST_OK)
        {
            usage = sysIoPower;
        }
        return powerReturn;
    }

    dcgmReturn_t GetCurrentModulePowerUsage(unsigned int, double &usage)
    {
        if (powerReturn == DCGM_ST_OK)
        {
            usage = modulePower;
        }
        return powerReturn;
    }

    dcgmReturn_t GetCurrentPowerCap(unsigned int, double &cap)
    {
        if (powerReturn == DCGM_ST_OK)
        {
            cap = powerCap;
        }
        return powerReturn;
    }

    std::string GetCpuVendor() const
    {
        return vendor;
    }

    std::string GetCpuModel() const
    {
        return model;
    }
};

struct FakeSysmonWatchTable
{
    struct Watch
    {
        dcgm_field_entity_group_t entityGroupId;
        dcgm_field_eid_t entityId;
        unsigned short fieldId;
        DcgmWatcher watcher;
        timelib64_t updateIntervalUsec;
        timelib64_t maxAgeUsec;
        bool isSubscribed;
    };

    DcgmMutex mutex { 0 };
    std::vector<Watch> watches;
    std::unordered_map<unsigned short, dcgm_field_meta_t> fieldMetas;
    std::unordered_set<unsigned int> ignoredFields;
    dcgmReturn_t getFieldsToUpdateReturn = DCGM_ST_OK;
    std::atomic<unsigned int> getFieldsToUpdateCalls { 0 };

    void IgnoreField(unsigned int fieldId)
    {
        DcgmLockGuard lock(&mutex);
        ignoredFields.insert(fieldId);
    }

    void SetGetFieldsToUpdateReturn(dcgmReturn_t ret)
    {
        DcgmLockGuard lock(&mutex);
        getFieldsToUpdateReturn = ret;
    }

    bool IsEmpty()
    {
        DcgmLockGuard lock(&mutex);
        return watches.empty();
    }

    size_t GetWatchCount()
    {
        DcgmLockGuard lock(&mutex);
        return watches.size();
    }

    Watch GetWatch(size_t index)
    {
        DcgmLockGuard lock(&mutex);
        return watches.at(index);
    }

    bool IsFieldIgnored(unsigned int fieldId)
    {
        DcgmLockGuard lock(&mutex);
        return ignoredFields.contains(fieldId);
    }

    bool AddWatcher(dcgm_field_entity_group_t entityGroupId,
                    dcgm_field_eid_t entityId,
                    unsigned short fieldId,
                    DcgmWatcher watcher,
                    timelib64_t updateIntervalUsec,
                    timelib64_t maxAgeUsec,
                    bool isSubscribed)
    {
        DcgmLockGuard lock(&mutex);
        watches.push_back({ entityGroupId, entityId, fieldId, watcher, updateIntervalUsec, maxAgeUsec, isSubscribed });
        dcgm_field_meta_t &meta = fieldMetas[fieldId];
        meta.fieldId            = fieldId;
        meta.fieldType
            = (fieldId == DCGM_FI_DEV_CPU_VENDOR || fieldId == DCGM_FI_DEV_CPU_MODEL) ? DCGM_FT_STRING : DCGM_FT_DOUBLE;
        return true;
    }

    void GetMaxAgeUsecAllWatches(timelib64_t &minAge, timelib64_t &maxAge)
    {
        DcgmLockGuard lock(&mutex);
        minAge = 0;
        maxAge = 0;
        for (auto const &watch : watches)
        {
            minAge = minAge == 0 ? watch.maxAgeUsec : std::min(minAge, watch.maxAgeUsec);
            maxAge = std::max(maxAge, watch.maxAgeUsec);
        }
    }

    dcgmReturn_t RemoveWatches(DcgmWatcher watcher, dcgmPostWatchInfo_t *)
    {
        DcgmLockGuard lock(&mutex);
        auto const toRemove = std::remove_if(
            watches.begin(), watches.end(), [&watcher](Watch const &watch) { return watch.watcher == watcher; });
        watches.erase(toRemove, watches.end());
        return DCGM_ST_OK;
    }

    dcgmReturn_t GetFieldsToUpdate(timelib64_t now,
                                   std::vector<dcgm_field_update_info_t> &toUpdate,
                                   timelib64_t &earliestNextUpdate)
    {
        DcgmLockGuard lock(&mutex);
        getFieldsToUpdateCalls++;
        earliestNextUpdate = 0;
        if (getFieldsToUpdateReturn != DCGM_ST_OK)
        {
            return getFieldsToUpdateReturn;
        }

        for (auto const &watch : watches)
        {
            if (!watch.isSubscribed)
            {
                continue;
            }

            dcgm_field_update_info_t updateInfo {};
            updateInfo.entityGroupId      = watch.entityGroupId;
            updateInfo.entityId           = watch.entityId;
            updateInfo.fieldMeta          = &fieldMetas.at(watch.fieldId);
            updateInfo.updateIntervalUsec = watch.updateIntervalUsec;
            toUpdate.push_back(updateInfo);

            timelib64_t nextUpdate = now + watch.updateIntervalUsec;
            earliestNextUpdate     = earliestNextUpdate == 0 ? nextUpdate : std::min(earliestNextUpdate, nextUpdate);
        }
        return DCGM_ST_OK;
    }

    void GetMinAndMaxUpdateInterval(timelib64_t &minUpdateInterval, timelib64_t &maxUpdateInterval)
    {
        DcgmLockGuard lock(&mutex);
        minUpdateInterval = 0;
        maxUpdateInterval = 0;
        for (auto const &watch : watches)
        {
            minUpdateInterval = minUpdateInterval == 0 ? watch.updateIntervalUsec
                                                       : std::min(minUpdateInterval, watch.updateIntervalUsec);
            maxUpdateInterval = std::max(maxUpdateInterval, watch.updateIntervalUsec);
        }
    }

    bool GetIsSubscribed(dcgm_field_entity_group_t entityGroupId, dcgm_field_eid_t entityId, unsigned short fieldId)
    {
        DcgmLockGuard lock(&mutex);
        for (auto const &watch : watches)
        {
            if (watch.entityGroupId == entityGroupId && watch.entityId == entityId && watch.fieldId == fieldId
                && watch.isSubscribed)
            {
                return true;
            }
        }
        return false;
    }
};

dcgm_sysmon_msg_watch_fields_t MakeWatchFieldsMessage(DcgmWatcher watcher,
                                                      dcgm_field_entity_group_t entityGroupId,
                                                      dcgm_field_eid_t entityId,
                                                      unsigned short fieldId)
{
    dcgm_sysmon_msg_watch_fields_t watchMsg {};
    watchMsg.header.moduleId              = DcgmModuleIdSysmon;
    watchMsg.header.subCommand            = DCGM_SYSMON_SR_WATCH_FIELDS;
    watchMsg.header.version               = dcgm_sysmon_msg_watch_fields_version;
    watchMsg.updateIntervalUsec           = 1;
    watchMsg.maxKeepAge                   = 2;
    watchMsg.maxKeepSamples               = 3;
    watchMsg.watcher                      = watcher;
    watchMsg.numEntities                  = 1;
    watchMsg.entityPairs[0].entityGroupId = entityGroupId;
    watchMsg.entityPairs[0].entityId      = entityId;
    watchMsg.numFieldIds                  = 1;
    watchMsg.fieldIds[0]                  = fieldId;
    return watchMsg;
}

dcgm_sysmon_msg_create_fake_entities_t MakeCreateFakeEntitiesMessage(dcgm_field_entity_group_t groupToCreate,
                                                                     unsigned int numToCreate)
{
    dcgm_sysmon_msg_create_fake_entities_t createMsg {};
    createMsg.header.moduleId   = DcgmModuleIdSysmon;
    createMsg.header.subCommand = DCGM_SYSMON_SR_CREATE_FAKE_ENTITIES;
    createMsg.header.version    = dcgm_sysmon_msg_create_fake_entities_version;
    createMsg.header.length     = sizeof(createMsg);
    createMsg.groupToCreate     = groupToCreate;
    createMsg.numToCreate       = numToCreate;
    return createMsg;
}

dcgm_sysmon_msg_get_cpus_t GetCpuSnapshot(DcgmModuleSysmon &sysmon)
{
    dcgm_sysmon_msg_get_cpus_t getCpusMsg {};
    getCpusMsg.header.moduleId   = DcgmModuleIdSysmon;
    getCpusMsg.header.subCommand = DCGM_SYSMON_SR_GET_CPUS;
    getCpusMsg.header.version    = dcgm_sysmon_msg_get_cpus_version;
    getCpusMsg.header.length     = sizeof(getCpusMsg);

    REQUIRE(sysmon.ProcessMessage(&getCpusMsg.header) == DCGM_ST_OK);
    return getCpusMsg;
}

unsigned int GetCpuCount(DcgmModuleSysmon &sysmon)
{
    return GetCpuSnapshot(sysmon).cpuCount;
}

unsigned int GetTotalCoreCount(DcgmModuleSysmon &sysmon)
{
    auto const cpus        = GetCpuSnapshot(sysmon);
    unsigned int coreCount = 0;
    for (unsigned int i = 0; i < cpus.cpuCount; ++i)
    {
        coreCount += cpus.cpus[i].coreCount;
    }
    return coreCount;
}

bool IsValidEntityId(DcgmModuleSysmon &sysmon, dcgm_field_entity_group_t entityGroupId, dcgm_field_eid_t entityId)
{
    dcgm_sysmon_msg_get_entity_status_t statusMsg {};
    statusMsg.header.moduleId   = DcgmModuleIdSysmon;
    statusMsg.header.subCommand = DCGM_SYSMON_SR_GET_ENTITY_STATUS;
    statusMsg.header.version    = dcgm_sysmon_msg_get_entity_status_version;
    statusMsg.header.length     = sizeof(statusMsg);
    statusMsg.entityGroupId     = entityGroupId;
    statusMsg.entityId          = entityId;

    REQUIRE(sysmon.ProcessMessage(&statusMsg.header) == DCGM_ST_OK);
    return statusMsg.entityStatus == DcgmEntityStatusOk;
}

unsigned int GetRemainingCapacityForTest(unsigned int current, unsigned int limit)
{
    if (current >= limit)
    {
        return 0;
    }

    return limit - current;
}

std::vector<dcgm_field_eid_t> CreateFakeEntities(DcgmModuleSysmon &sysmon,
                                                 dcgm_field_entity_group_t groupToCreate,
                                                 unsigned int numToCreate,
                                                 dcgmGroupEntityPair_t parent = {})
{
    std::vector<dcgm_field_eid_t> createdIds;
    createdIds.reserve(numToCreate);

    while (createdIds.size() < numToCreate)
    {
        auto const remaining = numToCreate - static_cast<unsigned int>(createdIds.size());
        auto createMsg       = MakeCreateFakeEntitiesMessage(groupToCreate, std::min(remaining, MaxCpuCreateIds));
        createMsg.parent     = parent;

        REQUIRE(sysmon.ProcessMessage(&createMsg.header) == DCGM_ST_OK);
        REQUIRE(createMsg.numCreated == createMsg.numToCreate);

        for (unsigned int i = 0; i < createMsg.numCreated; ++i)
        {
            createdIds.push_back(createMsg.ids[i]);
        }
    }

    return createdIds;
}

std::optional<dcgm_field_eid_t> TryCreateFakeCpu(DcgmModuleSysmon &sysmon)
{
    auto createMsg = MakeCreateFakeEntitiesMessage(DCGM_FE_CPU, 1);
    if (sysmon.ProcessMessage(&createMsg.header) != DCGM_ST_OK || createMsg.numCreated != 1
        || createMsg.ids[0] == DCGM_MAX_NUM_CPUS)
    {
        return std::nullopt;
    }

    return createMsg.ids[0];
}

dcgm_field_eid_t CreateFakeCpu(DcgmModuleSysmon &sysmon)
{
    auto cpuId = TryCreateFakeCpu(sysmon);
    REQUIRE(cpuId.has_value());
    return *cpuId;
}

std::vector<dcgm_field_eid_t> CreateFakeCores(DcgmModuleSysmon &sysmon,
                                              dcgm_field_eid_t parentCpuId,
                                              unsigned int numToCreate)
{
    dcgmGroupEntityPair_t parent {};
    parent.entityGroupId = DCGM_FE_CPU;
    parent.entityId      = parentCpuId;
    return CreateFakeEntities(sysmon, DCGM_FE_CPU_CORE, numToCreate, parent);
}

void FillCpusTo(DcgmModuleSysmon &sysmon, unsigned int target)
{
    while (GetCpuCount(sysmon) < target)
    {
        auto const current     = GetCpuCount(sysmon);
        auto const numToCreate = std::min(target - current, MaxCpuCreateIds);
        auto const createdIds  = CreateFakeEntities(sysmon, DCGM_FE_CPU, numToCreate);
        REQUIRE_FALSE(createdIds.empty());
    }
}

void FillCoresTo(DcgmModuleSysmon &sysmon, dcgm_field_eid_t parentCpuId, unsigned int target)
{
    while (GetTotalCoreCount(sysmon) < target)
    {
        auto const current     = GetTotalCoreCount(sysmon);
        auto const numToCreate = std::min(target - current, MaxCpuCreateIds);
        auto const createdIds  = CreateFakeCores(sysmon, parentCpuId, numToCreate);
        REQUIRE_FALSE(createdIds.empty());
    }
}

static bool HasDoubleFieldValue(DcgmFvBuffer &buffer, unsigned short fieldId, double value)
{
    dcgmBufferedFvCursor_t cursor = 0;
    for (dcgmBufferedFv_t *fv = buffer.GetNextFv(&cursor); fv; fv = buffer.GetNextFv(&cursor))
    {
        if (fv->fieldId == fieldId && fv->status == DCGM_ST_OK && fv->value.dbl == value)
        {
            return true;
        }
    }
    return false;
}

static bool HasStringFieldValue(DcgmFvBuffer &buffer, unsigned short fieldId, std::string const &value)
{
    dcgmBufferedFvCursor_t cursor = 0;
    for (dcgmBufferedFv_t *fv = buffer.GetNextFv(&cursor); fv; fv = buffer.GetNextFv(&cursor))
    {
        if (fv->fieldId == fieldId && fv->status == DCGM_ST_OK && value == fv->value.str)
        {
            return true;
        }
    }
    return false;
}

static bool HasBlankDoubleFieldValue(DcgmFvBuffer &buffer, unsigned short fieldId, dcgmReturn_t status)
{
    dcgmBufferedFvCursor_t cursor = 0;
    for (dcgmBufferedFv_t *fv = buffer.GetNextFv(&cursor); fv; fv = buffer.GetNextFv(&cursor))
    {
        if (fv->fieldId == fieldId && fv->status == status && DCGM_FP64_IS_BLANK(fv->value.dbl))
        {
            return true;
        }
    }
    return false;
}

static bool CapturedFieldsContainExpectedValues(CapturedAppendSamples &capture,
                                                FakeSysmonSystemMonitor const &fakeSysmon)
{
    std::unique_ptr<DcgmFvBuffer> buffer = GetCapturedAppendSamples(capture);
    return buffer != nullptr && HasDoubleFieldValue(*buffer, DCGM_FI_DEV_CPU_POWER_WATTS, fakeSysmon.cpuPower)
           && HasDoubleFieldValue(*buffer, DCGM_FI_DEV_SYSIO_POWER_WATTS, fakeSysmon.sysIoPower)
           && HasDoubleFieldValue(*buffer, DCGM_FI_DEV_MODULE_POWER_WATTS, fakeSysmon.modulePower)
           && HasDoubleFieldValue(*buffer, DCGM_FI_DEV_CPU_POWER_LIMIT_WATTS, fakeSysmon.powerCap)
           && HasStringFieldValue(*buffer, DCGM_FI_DEV_CPU_VENDOR, fakeSysmon.vendor)
           && HasStringFieldValue(*buffer, DCGM_FI_DEV_CPU_MODEL, fakeSysmon.model);
}

static bool CapturedPowerFieldsAreBlank(CapturedAppendSamples &capture, dcgmReturn_t status)
{
    std::unique_ptr<DcgmFvBuffer> buffer = GetCapturedAppendSamples(capture);
    return buffer != nullptr && HasBlankDoubleFieldValue(*buffer, DCGM_FI_DEV_CPU_POWER_WATTS, status)
           && HasBlankDoubleFieldValue(*buffer, DCGM_FI_DEV_SYSIO_POWER_WATTS, status)
           && HasBlankDoubleFieldValue(*buffer, DCGM_FI_DEV_MODULE_POWER_WATTS, status)
           && HasBlankDoubleFieldValue(*buffer, DCGM_FI_DEV_CPU_POWER_LIMIT_WATTS, status);
}

static std::string CapturedFieldsSummary(CapturedAppendSamples &capture)
{
    std::ostringstream out;
    {
        DcgmLockGuard lock(&capture.mutex);
        out << "appendSamplesCount=" << capture.count << " bufferBytes=" << capture.buffer.size();
    }

    std::unique_ptr<DcgmFvBuffer> buffer = GetCapturedAppendSamples(capture);
    if (buffer == nullptr)
    {
        return out.str();
    }

    dcgmBufferedFvCursor_t cursor = 0;
    for (dcgmBufferedFv_t *fv = buffer->GetNextFv(&cursor); fv; fv = buffer->GetNextFv(&cursor))
    {
        out << " fieldId=" << fv->fieldId << " status=" << static_cast<int>(fv->status);
        if (fv->fieldType == DCGM_FT_DOUBLE)
        {
            out << " dbl=" << fv->value.dbl;
        }
        else if (fv->fieldType == DCGM_FT_STRING)
        {
            out << " str=" << fv->value.str;
        }
    }
    return out.str();
}

TEST_CASE("Sysmon: initialize module")
{
    // Verify that instructing the constructor to skip the hardware check allows us to initialize correctly.
    bool differentHardware = false;

    try
    {
        TestDcgmModuleSysmon sysmon(g_coreCallbacks);
    }
    catch (std::runtime_error &e)
    {
        differentHardware = true;
    }

    if (differentHardware)
    {
        ScopedEnvVars envVars(SupportNonNvidiaCpuEnv, "1");
        try
        {
            TestDcgmModuleSysmon sysmon(g_coreCallbacks);
        }
        catch (std::runtime_error &e)
        {
            // We should never reach this point
            CHECK(false);
        }
    }
}

TEST_CASE("DcgmModuleSysmon::PopulateOwnedCoresBitmaskFromRangeString")
{
    dcgm_sysmon_cpu_t cpu0;
    memset(&cpu0, 0, sizeof(cpu0));

    std::string const cpu0Range = "0-5,12-17";
    CHECK(DcgmModuleSysmon::PopulateOwnedCoresBitmaskFromRangeString(cpu0, cpu0Range) == DCGM_ST_OK);
    CHECK(cpu0.coreCount == 12);
    for (unsigned int i = 0; i < 64; i++)
    {
        if (i < 6)
        {
            CHECK((cpu0.ownedCores.bitmask[0] & (1ULL << i)) != 0);
        }
        else if (i < 12)
        {
            CHECK((cpu0.ownedCores.bitmask[0] & (1ULL << i)) == 0);
        }
        else if (i < 18)
        {
            CHECK((cpu0.ownedCores.bitmask[0] & (1ULL << i)) != 0);
        }
        else
        {
            CHECK((cpu0.ownedCores.bitmask[0] & (1ULL << i)) == 0);
        }
    }

    // Check some failure cases
    std::string const malformedRange = "2-1-2";
    CHECK(DcgmModuleSysmon::PopulateOwnedCoresBitmaskFromRangeString(cpu0, malformedRange) == DCGM_ST_BADPARAM);
    std::string const negativeRange = "-1";
    CHECK(DcgmModuleSysmon::PopulateOwnedCoresBitmaskFromRangeString(cpu0, negativeRange) == DCGM_ST_BADPARAM);

    dcgm_sysmon_cpu_t cpu1 {};
    memset(&cpu1, 0, sizeof(cpu1));
    std::string const cpu1Range = "32-63,96-127";
    CHECK(DcgmModuleSysmon::PopulateOwnedCoresBitmaskFromRangeString(cpu1, cpu1Range) == DCGM_ST_OK);
    CHECK(cpu1.coreCount == 64);
    unsigned int memberSize = CHAR_BIT * sizeof(cpu1.ownedCores.bitmask[0]);
    for (unsigned int i = 0; i < 128; i++)
    {
        unsigned int index  = i / memberSize;
        unsigned int offset = i % memberSize;
        if (offset >= 32)
        {
            CHECK((cpu1.ownedCores.bitmask[index] & (1ULL << offset)) != 0);
        }
        else
        {
            CHECK((cpu1.ownedCores.bitmask[index] & (1ULL << offset)) == 0);
        }
    }
}

SCENARIO("DcgmModuleSysmon::PopulateOwnedCoresBitmaskFromRangeString handles public boundary inputs")
{
    GIVEN("an empty CPU hierarchy entry")
    {
        dcgm_sysmon_cpu_t cpu {};

        WHEN("a single core is specified")
        {
            dcgmReturn_t ret = DcgmModuleSysmon::PopulateOwnedCoresBitmaskFromRangeString(cpu, "7");

            THEN("only that core is marked as owned")
            {
                REQUIRE(ret == DCGM_ST_OK);
                CHECK(cpu.coreCount == 1);
                CHECK((cpu.ownedCores.bitmask[0] & (1ULL << 7)) != 0);
                CHECK((cpu.ownedCores.bitmask[0] & (1ULL << 6)) == 0);
                CHECK((cpu.ownedCores.bitmask[0] & (1ULL << 8)) == 0);
            }
        }
    }

    GIVEN("an empty CPU hierarchy entry")
    {
        dcgm_sysmon_cpu_t cpu {};

        WHEN("the range contains an empty token")
        {
            dcgmReturn_t ret = DcgmModuleSysmon::PopulateOwnedCoresBitmaskFromRangeString(cpu, "");

            THEN("the range is rejected without recording owned cores")
            {
                REQUIRE(ret == DCGM_ST_BADPARAM);
                CHECK(cpu.coreCount == 0);
            }
        }
    }

    GIVEN("an empty CPU hierarchy entry")
    {
        dcgm_sysmon_cpu_t cpu {};
        unsigned int const ownedCoresSize = CHAR_BIT * sizeof(cpu.ownedCores.bitmask);

        WHEN("the range references a core outside the owned-core bitmask")
        {
            dcgmReturn_t ret
                = DcgmModuleSysmon::PopulateOwnedCoresBitmaskFromRangeString(cpu, std::to_string(ownedCoresSize));

            THEN("the range reports insufficient output capacity")
            {
                REQUIRE(ret == DCGM_ST_INSUFFICIENT_SIZE);
                CHECK(cpu.coreCount == 0);
            }
        }
    }
}

SCENARIO("DcgmModuleSysmon::PopulateOwnedCoresBitmask reads owned cores through injected filesystem")
{
    GIVEN("a CPU node with a cpulist file")
    {
        FakeSysmonFileSystem fakeFs;
        fakeFs.files["/sys/devices/system/node/node2/cpulist"] = "4-6\n";
        SysmonFileSystemAnyPtr fs(&fakeFs);
        dcgm_sysmon_cpu_t cpu {};
        cpu.cpuId = 2;

        WHEN("the owned-core bitmask is populated")
        {
            DcgmModuleSysmon::PopulateOwnedCoresBitmask(cpu, fs);

            THEN("the cpulist range determines owned cores")
            {
                CHECK(cpu.coreCount == 3);
                CHECK((cpu.ownedCores.bitmask[0] & (1ULL << 4)) != 0);
                CHECK((cpu.ownedCores.bitmask[0] & (1ULL << 5)) != 0);
                CHECK((cpu.ownedCores.bitmask[0] & (1ULL << 6)) != 0);
                CHECK((cpu.ownedCores.bitmask[0] & (1ULL << 3)) == 0);
                CHECK((cpu.ownedCores.bitmask[0] & (1ULL << 7)) == 0);
            }
        }
    }

    GIVEN("a CPU node with malformed cpulist contents")
    {
        FakeSysmonFileSystem fakeFs;
        fakeFs.files["/sys/devices/system/node/node3/cpulist"] = "1-2-3\n";
        SysmonFileSystemAnyPtr fs(&fakeFs);
        dcgm_sysmon_cpu_t cpu {};
        cpu.cpuId = 3;

        WHEN("the owned-core bitmask is populated")
        {
            THEN("the malformed cpulist is rejected")
            {
                REQUIRE_THROWS_AS(DcgmModuleSysmon::PopulateOwnedCoresBitmask(cpu, fs), std::runtime_error);
                CHECK(cpu.coreCount == 0);
            }
        }
    }

    GIVEN("a null filesystem dependency")
    {
        SysmonFileSystemAnyPtr fs(static_cast<FakeSysmonFileSystem *>(nullptr));
        dcgm_sysmon_cpu_t cpu {};
        cpu.cpuId = 4;

        WHEN("the owned-core bitmask is populated")
        {
            THEN("the null dependency is rejected before file access")
            {
                REQUIRE_THROWS_AS(DcgmModuleSysmon::PopulateOwnedCoresBitmask(cpu, fs), std::invalid_argument);
                CHECK(cpu.coreCount == 0);
            }
        }
    }
}

struct WatchProcessingResult
{
    dcgmReturn_t watchRet;
    dcgmReturn_t barrierRet;
};

static WatchProcessingResult ProcessWatchAndDrain(DcgmModuleSysmon &sysmon, dcgm_sysmon_msg_watch_fields_t &watchMsg)
{
    dcgmReturn_t const watchRet = sysmon.ProcessMessage(&watchMsg.header);

    dcgm_sysmon_msg_get_cpus_t barrierMsg {};
    barrierMsg.header.moduleId   = DcgmModuleIdSysmon;
    barrierMsg.header.subCommand = DCGM_SYSMON_SR_GET_CPUS;
    barrierMsg.header.version    = dcgm_sysmon_msg_get_cpus_version;

    return { watchRet, sysmon.ProcessMessage(&barrierMsg.header) };
}

SCENARIO("DcgmModuleSysmon records injected system monitor values through public watches")
{
    GIVEN("a sysmon module with an injected system monitor dependency")
    {
        CapturedAppendSamples capturedSamples;
        dcgmCoreCallbacks_t capturingCallbacks(dcgmCoreCallbacks_version, postRequestToCore, &capturedSamples, nullptr);
        FakeSysmonSystemMonitor fakeSysmon;
        FakeSysmonWatchTable fakeWatchTable;
        DcgmModuleSysmon sysmon(capturingCallbacks, &fakeSysmon, &fakeWatchTable);
        dcgm_sysmon_msg_create_fake_entities_t createCpuMsg {};
        createCpuMsg.header.moduleId   = DcgmModuleIdSysmon;
        createCpuMsg.header.subCommand = DCGM_SYSMON_SR_CREATE_FAKE_ENTITIES;
        createCpuMsg.header.version    = dcgm_sysmon_msg_create_fake_entities_version;
        createCpuMsg.groupToCreate     = DCGM_FE_CPU;
        createCpuMsg.numToCreate       = 1;
        REQUIRE(sysmon.ProcessMessage(&createCpuMsg.header) == DCGM_ST_OK);
        REQUIRE(createCpuMsg.numCreated == 1);

        DcgmWatcher watcher(DcgmWatcherTypeClient, 9);
        dcgm_sysmon_msg_watch_fields_t watchMsg {};
        watchMsg.header.moduleId              = DcgmModuleIdSysmon;
        watchMsg.header.subCommand            = DCGM_SYSMON_SR_WATCH_FIELDS;
        watchMsg.header.version               = dcgm_sysmon_msg_watch_fields_version;
        watchMsg.updateIntervalUsec           = 1;
        watchMsg.maxKeepAge                   = 2;
        watchMsg.maxKeepSamples               = 3;
        watchMsg.watcher                      = watcher;
        watchMsg.numEntities                  = 1;
        watchMsg.entityPairs[0].entityGroupId = DCGM_FE_CPU;
        watchMsg.entityPairs[0].entityId      = createCpuMsg.ids[0];
        watchMsg.numFieldIds                  = 6;
        watchMsg.fieldIds[0]                  = DCGM_FI_DEV_CPU_POWER_WATTS;
        watchMsg.fieldIds[1]                  = DCGM_FI_DEV_SYSIO_POWER_WATTS;
        watchMsg.fieldIds[2]                  = DCGM_FI_DEV_MODULE_POWER_WATTS;
        watchMsg.fieldIds[3]                  = DCGM_FI_DEV_CPU_POWER_LIMIT_WATTS;
        watchMsg.fieldIds[4]                  = DCGM_FI_DEV_CPU_VENDOR;
        watchMsg.fieldIds[5]                  = DCGM_FI_DEV_CPU_MODEL;

        WHEN("the watched fields are processed")
        {
            auto const processing = ProcessWatchAndDrain(sysmon, watchMsg);

            THEN("the values posted to core come from the injected dependency")
            {
                REQUIRE(processing.watchRet == DCGM_ST_OK);
                REQUIRE(processing.barrierRet == DCGM_ST_OK);
                INFO(CapturedFieldsSummary(capturedSamples));
                CHECK(CapturedFieldsContainExpectedValues(capturedSamples, fakeSysmon));
            }
        }

        WHEN("the power getters fail without setting their output values")
        {
            fakeSysmon.powerReturn = DCGM_ST_BADPARAM;

            auto const processing = ProcessWatchAndDrain(sysmon, watchMsg);

            THEN("blank power values with the getter error are posted to core")
            {
                REQUIRE(processing.watchRet == DCGM_ST_OK);
                REQUIRE(processing.barrierRet == DCGM_ST_OK);
                INFO(CapturedFieldsSummary(capturedSamples));
                CHECK(CapturedPowerFieldsAreBlank(capturedSamples, fakeSysmon.powerReturn));
            }
        }
    }
}

TEST_CASE("DcgmModuleSysmon validates public watch requests through injected watch table")
{
    FakeSysmonSystemMonitor fakeSysmon;
    FakeSysmonWatchTable fakeWatchTable;
    DcgmModuleSysmon sysmon(g_coreCallbacks, &fakeSysmon, &fakeWatchTable);
    DcgmWatcher watcher(DcgmWatcherTypeClient, 10);
    dcgm_field_eid_t cpuId = CreateFakeCpu(sysmon);
    auto watchMsg          = MakeWatchFieldsMessage(watcher, DCGM_FE_CPU, cpuId, DCGM_FI_DEV_CPU_POWER_WATTS);

    SECTION("zero fields")
    {
        watchMsg.numFieldIds = 0;
        CHECK(sysmon.ProcessMessage(&watchMsg.header) == DCGM_ST_BADPARAM);
        CHECK(fakeWatchTable.IsEmpty());
    }

    SECTION("too many fields")
    {
        watchMsg.numFieldIds = SYSMON_MSG_WATCH_FIELDS_MAX_NUM_FIELDS + 1;
        CHECK(sysmon.ProcessMessage(&watchMsg.header) == DCGM_ST_BADPARAM);
        CHECK(fakeWatchTable.IsEmpty());
    }

    SECTION("zero entities")
    {
        watchMsg.numEntities = 0;
        CHECK(sysmon.ProcessMessage(&watchMsg.header) == DCGM_ST_BADPARAM);
        CHECK(fakeWatchTable.IsEmpty());
    }

    SECTION("too many entities")
    {
        watchMsg.numEntities = SYSMON_MSG_WATCH_FIELDS_MAX_NUM_ENTITIES + 1;
        CHECK(sysmon.ProcessMessage(&watchMsg.header) == DCGM_ST_BADPARAM);
        CHECK(fakeWatchTable.IsEmpty());
    }
}

SCENARIO("DcgmModuleSysmon: rejects fake entity creation above response id capacity")
{
    ScopedEnvVars envVars(HangDetectDisableEnv, "1");
    FakeSysmonSystemMonitor fakeSysmon;
    FakeSysmonWatchTable fakeWatchTable;
    TestDcgmModuleSysmon sysmon(g_coreCallbacks, &fakeSysmon, &fakeWatchTable);
    auto const cpuId = TryCreateFakeCpu(sysmon);
    if (!cpuId.has_value())
    {
        SKIP("No CPU slots remain for a fake CPU parent.");
    }
    unsigned int const coresBefore = GetTotalCoreCount(sysmon);

    auto createMsg                 = MakeCreateFakeEntitiesMessage(DCGM_FE_CPU_CORE, DCGM_MAX_CPU_CREATE_IDS + 1);
    createMsg.numCreated           = 7;
    createMsg.parent.entityGroupId = DCGM_FE_CPU;
    createMsg.parent.entityId      = *cpuId;

    CHECK(sysmon.ProcessMessage(&createMsg.header) == DCGM_ST_BADPARAM);
    CHECK(createMsg.numCreated == 0);
    CHECK(GetTotalCoreCount(sysmon) == coresBefore);
}

SCENARIO("DcgmModuleSysmon: accepts fake core creation at response id capacity")
{
    ScopedEnvVars envVars(HangDetectDisableEnv, "1");
    FakeSysmonSystemMonitor fakeSysmon;
    FakeSysmonWatchTable fakeWatchTable;
    TestDcgmModuleSysmon sysmon(g_coreCallbacks, &fakeSysmon, &fakeWatchTable);
    auto const cpuId = TryCreateFakeCpu(sysmon);
    if (!cpuId.has_value())
    {
        SKIP("No CPU slots remain for a fake CPU parent.");
    }
    unsigned int const coresBefore = GetTotalCoreCount(sysmon);
    if (GetRemainingCapacityForTest(coresBefore, DCGM_MAX_NUM_CPU_CORES) < DCGM_MAX_CPU_CREATE_IDS)
    {
        SKIP("Not enough CPU core capacity remains for the response-capacity boundary test.");
    }

    auto createMsg                 = MakeCreateFakeEntitiesMessage(DCGM_FE_CPU_CORE, DCGM_MAX_CPU_CREATE_IDS);
    createMsg.parent.entityGroupId = DCGM_FE_CPU;
    createMsg.parent.entityId      = *cpuId;

    CHECK(sysmon.ProcessMessage(&createMsg.header) == DCGM_ST_OK);
    CHECK(createMsg.numCreated == DCGM_MAX_CPU_CREATE_IDS);
    CHECK(GetTotalCoreCount(sysmon) == coresBefore + DCGM_MAX_CPU_CREATE_IDS);

    SECTION("Verify that created IDs are valid and unique")
    {
        std::unordered_set<unsigned int> createdIds;

        for (unsigned int i = 0; i < DCGM_MAX_CPU_CREATE_IDS; ++i)
        {
            CHECK(createMsg.ids[i] != DCGM_MAX_NUM_CPU_CORES);
            CHECK(IsValidEntityId(sysmon, DCGM_FE_CPU_CORE, createMsg.ids[i]));
            CHECK(createdIds.insert(createMsg.ids[i]).second == true);
        }
    }
}

SCENARIO("DcgmModuleSysmon: rejects fake CPU creation above remaining capacity")
{
    ScopedEnvVars envVars(HangDetectDisableEnv, "1");
    FakeSysmonSystemMonitor fakeSysmon;
    FakeSysmonWatchTable fakeWatchTable;
    TestDcgmModuleSysmon sysmon(g_coreCallbacks, &fakeSysmon, &fakeWatchTable);
    FillCpusTo(sysmon, DCGM_MAX_NUM_CPUS - 1);

    unsigned int const cpusBefore        = GetCpuCount(sysmon);
    unsigned int const remainingCapacity = GetRemainingCapacityForTest(cpusBefore, DCGM_MAX_NUM_CPUS);
    auto createMsg                       = MakeCreateFakeEntitiesMessage(DCGM_FE_CPU, remainingCapacity + 1);
    createMsg.numCreated                 = 3;

    CHECK(sysmon.ProcessMessage(&createMsg.header) == DCGM_ST_INSUFFICIENT_RESOURCES);
    CHECK(createMsg.numCreated == 0);
    CHECK(GetCpuCount(sysmon) == cpusBefore);
}

SCENARIO("DcgmModuleSysmon: rejects fake core creation above remaining capacity")
{
    ScopedEnvVars envVars(HangDetectDisableEnv, "1");
    FakeSysmonSystemMonitor fakeSysmon;
    FakeSysmonWatchTable fakeWatchTable;
    TestDcgmModuleSysmon sysmon(g_coreCallbacks, &fakeSysmon, &fakeWatchTable);
    auto const parentCpuId = TryCreateFakeCpu(sysmon);
    if (!parentCpuId.has_value())
    {
        SKIP("No CPU slots remain for a fake CPU parent.");
    }
    FillCoresTo(sysmon, *parentCpuId, DCGM_MAX_NUM_CPU_CORES - 1);

    unsigned int const coresBefore       = GetTotalCoreCount(sysmon);
    unsigned int const remainingCapacity = GetRemainingCapacityForTest(coresBefore, DCGM_MAX_NUM_CPU_CORES);
    auto createMsg                       = MakeCreateFakeEntitiesMessage(DCGM_FE_CPU_CORE, remainingCapacity + 1);
    createMsg.numCreated                 = 3;
    createMsg.parent.entityGroupId       = DCGM_FE_CPU;
    createMsg.parent.entityId            = *parentCpuId;

    CHECK(sysmon.ProcessMessage(&createMsg.header) == DCGM_ST_INSUFFICIENT_RESOURCES);
    CHECK(createMsg.numCreated == 0);
    CHECK(GetTotalCoreCount(sysmon) == coresBefore);
}

TEST_CASE("DcgmModuleSysmon skips invalid watch entities without subscribing")
{
    FakeSysmonSystemMonitor fakeSysmon;
    FakeSysmonWatchTable fakeWatchTable;
    DcgmModuleSysmon sysmon(g_coreCallbacks, &fakeSysmon, &fakeWatchTable);
    DcgmWatcher watcher(DcgmWatcherTypeClient, 11);

    SECTION("invalid CPU")
    {
        auto watchMsg = MakeWatchFieldsMessage(watcher, DCGM_FE_CPU, 999, DCGM_FI_DEV_CPU_POWER_WATTS);
        CHECK(sysmon.ProcessMessage(&watchMsg.header) == DCGM_ST_OK);
        CHECK(fakeWatchTable.IsEmpty());
    }

    SECTION("invalid core")
    {
        auto watchMsg = MakeWatchFieldsMessage(watcher, DCGM_FE_CPU_CORE, 999, DCGM_FI_DEV_CPU_UTIL_USER_RATIO);
        CHECK(sysmon.ProcessMessage(&watchMsg.header) == DCGM_ST_OK);
        CHECK(fakeWatchTable.IsEmpty());
    }

    SECTION("unsupported entity group")
    {
        auto watchMsg = MakeWatchFieldsMessage(watcher, DCGM_FE_GPU, 0, DCGM_FI_DEV_CPU_POWER_WATTS);
        CHECK(sysmon.ProcessMessage(&watchMsg.header) == DCGM_ST_OK);
        CHECK(fakeWatchTable.IsEmpty());
    }
}

TEST_CASE("DcgmModuleSysmon rejects ignored fields before subscribing")
{
    FakeSysmonSystemMonitor fakeSysmon;
    FakeSysmonWatchTable fakeWatchTable;
    fakeWatchTable.IgnoreField(DCGM_FI_DEV_CPU_POWER_WATTS);
    DcgmModuleSysmon sysmon(g_coreCallbacks, &fakeSysmon, &fakeWatchTable);
    DcgmWatcher watcher(DcgmWatcherTypeClient, 12);
    dcgm_field_eid_t cpuId = CreateFakeCpu(sysmon);

    auto watchMsg = MakeWatchFieldsMessage(watcher, DCGM_FE_CPU, cpuId, DCGM_FI_DEV_CPU_POWER_WATTS);
    CHECK(sysmon.ProcessMessage(&watchMsg.header) == DCGM_ST_BADPARAM);
    CHECK(fakeWatchTable.IsEmpty());
}

TEST_CASE("DcgmModuleSysmon unwatches through injected watch table")
{
    FakeSysmonSystemMonitor fakeSysmon;
    FakeSysmonWatchTable fakeWatchTable;
    DcgmModuleSysmon sysmon(g_coreCallbacks, &fakeSysmon, &fakeWatchTable);
    DcgmWatcher watcher(DcgmWatcherTypeClient, 13);
    DcgmWatcher otherWatcher(DcgmWatcherTypeClient, 14);
    dcgm_field_eid_t cpuId = CreateFakeCpu(sysmon);
    auto watchMsg          = MakeWatchFieldsMessage(watcher, DCGM_FE_CPU, cpuId, DCGM_FI_DEV_CPU_POWER_WATTS);
    auto otherWatchMsg     = MakeWatchFieldsMessage(otherWatcher, DCGM_FE_CPU, cpuId, DCGM_FI_DEV_CPU_POWER_WATTS);

    REQUIRE(sysmon.ProcessMessage(&watchMsg.header) == DCGM_ST_OK);
    REQUIRE(sysmon.ProcessMessage(&otherWatchMsg.header) == DCGM_ST_OK);
    REQUIRE(fakeWatchTable.GetWatchCount() == 2);

    dcgm_sysmon_msg_unwatch_fields_t unwatchMsg {};
    unwatchMsg.header.moduleId   = DcgmModuleIdSysmon;
    unwatchMsg.header.subCommand = DCGM_SYSMON_SR_UNWATCH_FIELDS;
    unwatchMsg.header.version    = dcgm_sysmon_msg_unwatch_fields_version;
    unwatchMsg.watcher           = watcher;

    CHECK(sysmon.ProcessMessage(&unwatchMsg.header) == DCGM_ST_OK);
    REQUIRE(fakeWatchTable.GetWatchCount() == 1);
    CHECK(fakeWatchTable.GetWatch(0).watcher == otherWatcher);
}

TEST_CASE("DcgmModuleSysmon stops field update when watch table reports an error")
{
    CapturedAppendSamples capturedSamples;
    dcgmCoreCallbacks_t capturingCallbacks(dcgmCoreCallbacks_version, postRequestToCore, &capturedSamples, nullptr);
    FakeSysmonSystemMonitor fakeSysmon;
    FakeSysmonWatchTable fakeWatchTable;
    DcgmModuleSysmon sysmon(capturingCallbacks, &fakeSysmon, &fakeWatchTable);
    DcgmWatcher watcher(DcgmWatcherTypeClient, 14);

    dcgm_sysmon_msg_create_fake_entities_t createCpuMsg {};
    createCpuMsg.header.moduleId   = DcgmModuleIdSysmon;
    createCpuMsg.header.subCommand = DCGM_SYSMON_SR_CREATE_FAKE_ENTITIES;
    createCpuMsg.header.version    = dcgm_sysmon_msg_create_fake_entities_version;
    createCpuMsg.groupToCreate     = DCGM_FE_CPU;
    createCpuMsg.numToCreate       = 1;

    REQUIRE(sysmon.ProcessMessage(&createCpuMsg.header) == DCGM_ST_OK);
    dcgm_field_eid_t cpuId = createCpuMsg.ids[0];

    fakeWatchTable.SetGetFieldsToUpdateReturn(DCGM_ST_BADPARAM);
    unsigned int callsBeforeWatch   = fakeWatchTable.getFieldsToUpdateCalls.load();
    unsigned int samplesBeforeWatch = 0;

    {
        DcgmLockGuard lock(&capturedSamples.mutex);
        samplesBeforeWatch = capturedSamples.count;
    }

    auto watchMsg = MakeWatchFieldsMessage(watcher, DCGM_FE_CPU, cpuId, DCGM_FI_DEV_CPU_POWER_WATTS);
    REQUIRE(sysmon.ProcessMessage(&watchMsg.header) == DCGM_ST_OK);

    dcgm_sysmon_msg_get_cpus_t getCpusMsg {};
    getCpusMsg.header.moduleId   = DcgmModuleIdSysmon;
    getCpusMsg.header.subCommand = DCGM_SYSMON_SR_GET_CPUS;
    getCpusMsg.header.version    = dcgm_sysmon_msg_get_cpus_version;
    REQUIRE(sysmon.ProcessMessage(&getCpusMsg.header) == DCGM_ST_OK);

    CHECK(fakeWatchTable.getFieldsToUpdateCalls.load() > callsBeforeWatch);
    {
        DcgmLockGuard lock(&capturedSamples.mutex);
        CHECK(capturedSamples.count == samplesBeforeWatch);
    }
}

TEST_CASE("DcgmModuleSysmon rejects null injected dependencies")
{
    SECTION("system monitor")
    {
        SysmonSystemMonitorAnyPtr nullSystemMonitor(static_cast<FakeSysmonSystemMonitor *>(nullptr));
        CHECK_THROWS_AS(DcgmModuleSysmon(g_coreCallbacks, nullSystemMonitor), std::invalid_argument);
    }

    SECTION("watch table")
    {
        FakeSysmonSystemMonitor fakeSysmon;
        SysmonWatchTableAnyPtr nullWatchTable(static_cast<FakeSysmonWatchTable *>(nullptr));
        CHECK_THROWS_AS(DcgmModuleSysmon(g_coreCallbacks, &fakeSysmon, nullWatchTable), std::invalid_argument);
    }
}

TEST_CASE("DcgmModuleSysmon::ParseProcStatCpuLine")
{
    ScopedEnvVars envVars({ { SupportNonNvidiaCpuEnv, "1" }, { HangDetectDisableEnv, "1" } });

    TestDcgmModuleSysmon sysmon(g_coreCallbacks);

    SysmonUtilizationSample sample = {};
    // Not enough room
    std::string const notEnoughRoom = "cpu0 75 5 6 25";
    CHECK(sysmon.ParseProcStatCpuLine(notEnoughRoom, sample) == DCGM_ST_BADPARAM);

    sample.m_cores.resize(12);

    std::vector<std::string> const badCpuLines {
        "cpu0 bad line",
        "cpu0 bad line with enough tokens for checking",
        "cpu0 75 5 6 bad",
        "cpu0 badusertime 5 6 76",
    };
    for (auto const &line : badCpuLines)
    {
        CHECK(sysmon.ParseProcStatCpuLine(line, sample) == DCGM_ST_BADPARAM);
    }

    std::string const validCpuLine = "cpu0 75 5 6 25 0 8";
    CHECK(sysmon.ParseProcStatCpuLine(validCpuLine, sample) == DCGM_ST_OK);
    CHECK(sample.m_cores[0].m_user == 75);
    CHECK(sample.m_cores[0].m_nice == 5);
    CHECK(sample.m_cores[0].m_system == 6);
    CHECK(sample.m_cores[0].m_idle == 25);
    CHECK(sample.m_cores[0].m_irq == 8);

    // Some lines from my system should pass
    std::vector<std::string> const validCpuLines {
        "cpu0 9733046 9991 2371910 659811695 177306 0 8925 0 0 0",
        "cpu1 9772560 8585 2085158 660512731 161268 0 2882 0 0 0",
        "cpu2 11007090 10098 2076304 659320474 148554 0 1324 0 0 0",
        "cpu3 11113132 10638 2004824 659242757 134281 0 797 0 0 0",
        "cpu4 8031834 7531 2880138 660798436 113679 0 583 0 0 0",
        "cpu5 9078527 7740 2196453 660871019 132242 0 460 0 0 0",
        "cpu6 9460076 7303 2144416 660593573 108288 0 469 0 0 0",
        "cpu7 8886171 5701 2491062 660719039 107810 0 540 0 0 0",
        "cpu8 9981141 8746 2026270 660452649 104743 0 514 0 0 0",
        "cpu9 9197158 8551 2146646 661044381 102656 0 375 0 0 0",
        "cpu10 8317451 9040 1924201 661743503 425237 0 23118 0 0 0",
        "cpu11 8233975 7082 2741281 660626374 273765 0 78044 0 0 0",
    };
    for (auto const &line : validCpuLines)
    {
        CHECK(sysmon.ParseProcStatCpuLine(line, sample) == DCGM_ST_OK);
    }
}

TEST_CASE("DcgmModuleSysmon::ParseThermalFileContentsAndStore")
{
    ScopedEnvVars envVars(SupportNonNvidiaCpuEnv, "1");
    TestDcgmModuleSysmon dms(g_coreCallbacks);

    // Format for matches should be 'Thermal Zone Skt# TJMax'
    struct ThermalZoneTestCase
    {
        std::string contents;
        unsigned int expectedSocketId;
    };
    std::vector<ThermalZoneTestCase> const thermalZoneCases {
        { "too many tokens in line", SYSMON_INVALID_SOCKET_ID },
        { "toofew tokens", SYSMON_INVALID_SOCKET_ID },
        { "Thermal Zone Sktbob TJMax", SYSMON_INVALID_SOCKET_ID },
        { "Thermal Zone Skt0 TJMax", 0 },
        { "Thermal Zone Skt1 TJMax", 1 },
        { "Thermal Zone Skt2 TJMax", 2 },
        { "Thermal Zone Skt3 TJMax", 3 },
    };
    std::string const path;
    for (auto const &testCase : thermalZoneCases)
    {
        CHECK(dms.GetSocketFromThermalZoneFileContents(path, testCase.contents) == testCase.expectedSocketId);
    }
}

TEST_CASE("DcgmModuleSysmon Watches")
{
    ScopedEnvVars envVars(SupportNonNvidiaCpuEnv, "1");

    TestDcgmModuleSysmon sysmon(g_coreCallbacks);
    DcgmWatcher watcher(DcgmWatcherTypeClient, 1);

    const dcgm_field_entity_group_t entityGroupId = DCGM_FE_CPU;
    const unsigned int entityId                   = 0;
    const unsigned int fieldId                    = DCGM_FI_DEV_CPU_UTIL_USER_RATIO;

    dcgm_sysmon_msg_watch_fields_t watchMsg {};
    watchMsg.header.moduleId              = DcgmModuleIdSysmon;
    watchMsg.header.subCommand            = DCGM_SYSMON_SR_WATCH_FIELDS;
    watchMsg.header.version               = dcgm_sysmon_msg_watch_fields_version;
    watchMsg.updateIntervalUsec           = 1;
    watchMsg.maxKeepAge                   = 2;
    watchMsg.maxKeepSamples               = 3;
    watchMsg.watcher                      = watcher;
    watchMsg.numEntities                  = 1;
    watchMsg.entityPairs[0].entityId      = entityId;
    watchMsg.entityPairs[0].entityGroupId = entityGroupId;
    watchMsg.numFieldIds                  = 1;
    watchMsg.fieldIds[0]                  = fieldId;

    dcgm_sysmon_msg_unwatch_fields_t unwatchMsg {};
    unwatchMsg.header.moduleId   = DcgmModuleIdSysmon;
    unwatchMsg.header.subCommand = DCGM_SYSMON_SR_UNWATCH_FIELDS;
    unwatchMsg.header.version    = dcgm_sysmon_msg_unwatch_fields_version;
    unwatchMsg.watcher           = watcher;

    dcgmReturn_t ret = sysmon.ProcessMessage(&watchMsg.header);
    CHECK(ret == DCGM_ST_OK);

    bool isSubscribed { false };
    ret = sysmon.GetSubscribed(watchMsg.entityPairs[0], fieldId, isSubscribed);
    CHECK(ret == DCGM_ST_OK);
    CHECK(isSubscribed == true);

    ret = sysmon.ProcessMessage(&unwatchMsg.header);
    CHECK(ret == DCGM_ST_OK);

    isSubscribed = false;
    ret          = sysmon.GetSubscribed(watchMsg.entityPairs[0], fieldId, isSubscribed);
    CHECK(ret == DCGM_ST_OK);
    CHECK(isSubscribed != true);
}

// Returns whether the specified fieldId and entityPair are subscribed. Helper for maxSampleAge testcase.
static bool IsSubscribed(TestDcgmModuleSysmon &sysmon, dcgmGroupEntityPair_t &pair, unsigned short fieldId)
{
    bool isSubscribed { false };
    dcgmReturn_t ret = sysmon.GetSubscribed(pair, fieldId, isSubscribed);
    CHECK(ret == DCGM_ST_OK);
    return isSubscribed;
}

// Returns current size of the utilization samples map. Helper for maxSampleAge test case.
static size_t GetUtilizationSampleSize(TestDcgmModuleSysmon &sysmon)
{
    size_t uss       = 0;
    dcgmReturn_t ret = sysmon.GetUtilizationSampleSize(uss);
    CHECK(ret == DCGM_ST_OK);
    return uss;
}


TEST_CASE("DcgmModuleSysmon maxSampleAge")
{
    using namespace DcgmNs::Timelib;
    ScopedEnvVars envVars({ { SupportNonNvidiaCpuEnv, "1" }, { HangDetectDisableEnv, "1" } });
    TestDcgmModuleSysmon sysmon(g_coreCallbacks);

    DcgmWatcher watcher(DcgmWatcherTypeClient, 1);

    const dcgm_field_entity_group_t entityGroupId = DCGM_FE_CPU;
    const unsigned int entityId                   = CreateFakeCpu(sysmon);
    const unsigned int fieldId                    = DCGM_FI_DEV_CPU_UTIL_USER_RATIO;

    timelib64_t updateIntervalUsec = 200000;
    double maxKeepAgeSec           = 30.0; // seconds
    size_t maxKeepSamples          = 0;    // handled by maxKeepAge

    dcgm_sysmon_msg_watch_fields_t watchMsg {};
    watchMsg.header.moduleId              = DcgmModuleIdSysmon; // process in task runner
    watchMsg.header.subCommand            = DCGM_SYSMON_SR_WATCH_FIELDS;
    watchMsg.header.version               = dcgm_sysmon_msg_watch_fields_version;
    watchMsg.updateIntervalUsec           = updateIntervalUsec;
    watchMsg.maxKeepAge                   = maxKeepAgeSec;
    watchMsg.maxKeepSamples               = maxKeepSamples;
    watchMsg.watcher                      = watcher;
    watchMsg.numEntities                  = 1;
    watchMsg.entityPairs[0].entityId      = entityId;
    watchMsg.entityPairs[0].entityGroupId = entityGroupId;
    watchMsg.numFieldIds                  = 1;
    watchMsg.fieldIds[0]                  = fieldId;

    /* Verify that there is no watch prior to processing
       the message */

    CHECK(!IsSubscribed(sysmon, watchMsg.entityPairs[0], fieldId));

    dcgmReturn_t ret = DCGM_ST_OK;
    ret              = sysmon.ProcessMessage(&watchMsg.header);
    CHECK(ret == DCGM_ST_OK);

    /* Verify the watch is subscribed after processing the message */
    CHECK(IsSubscribed(sysmon, watchMsg.entityPairs[0], fieldId));

    /* Verify a newly added sample isn't prematurely pruned
       until enough time has passed */
    TimePoint now = Now();

    ret = sysmon.RunOnce(true);
    CHECK(ret == DCGM_ST_OK);

    CHECK(GetUtilizationSampleSize(sysmon) == 0);

    SysmonUtilizationSample sample {};
    sample.m_cores.resize(1);
    sample.m_timestamp       = now;
    sample.m_cores[0].m_user = 0.125;

    ret = sysmon.AddUtilizationSample(sample);
    CHECK(ret == DCGM_ST_OK);

    ret = sysmon.RunOnce(true);
    CHECK(ret == DCGM_ST_OK);
    CHECK(GetUtilizationSampleSize(sysmon) == 1);

    // "Advance" the clock and demonstrate that the recorded sample gets pruned
    ret = sysmon.PruneSamples(TimePoint(now + std::chrono::seconds(30)));
    CHECK(ret == DCGM_ST_OK);

    CHECK(GetUtilizationSampleSize(sysmon) == 0);

    /* Ensure many samples are not pruned until enough
       time has passed */
    now = Now();

    // 200_000 us = 200 ms = 0.2s or 5 Hz
    // For 30s, ~150 samples, not accounting slack

    static const size_t NUM_SAMPLES = 150;
    std::array<SysmonUtilizationSample, NUM_SAMPLES> samples {};

    for (size_t i = 0; i < NUM_SAMPLES; i++)
    {
        auto &sample = samples[i];

        sample.m_cores.resize(1);
        sample.m_timestamp       = (now - std::chrono::seconds(29)) + (i * std::chrono::milliseconds(200));
        sample.m_cores[0].m_user = 1.0 / (1 + i);

        ret = sysmon.AddUtilizationSample(sample);
        CHECK(ret == DCGM_ST_OK);
    }

    CHECK(GetUtilizationSampleSize(sysmon) == 150);
    ret = sysmon.RunOnce(true);
    CHECK(ret == DCGM_ST_OK);
    CHECK(GetUtilizationSampleSize(sysmon) == 150);

    ret = sysmon.PruneSamples(now + std::chrono::seconds(1));
    CHECK(ret == DCGM_ST_OK);
    CHECK(GetUtilizationSampleSize(sysmon) == 149);

    ret = sysmon.PruneSamples(now + std::chrono::seconds(31));
    CHECK(ret == DCGM_ST_OK);
    CHECK(GetUtilizationSampleSize(sysmon) == 0);
}

TEST_CASE("DcgmModuleSysmon::CalculateCoreUtilization")
{
    ScopedEnvVars envVars(SupportNonNvidiaCpuEnv, "1");
    TestDcgmModuleSysmon sysmon(g_coreCallbacks);

    unsigned int numCores = 2;
    CoreId core           = CoreId { 0 };

    double user = 1, nice = 2, system = 3, idle = 4, irq = 5, other = 6;

    SysmonUtilizationSample baselineSample;
    baselineSample.m_cores.resize(numCores);
    baselineSample.m_cores[0].m_user   = user;
    baselineSample.m_cores[0].m_nice   = nice;
    baselineSample.m_cores[0].m_system = system;
    baselineSample.m_cores[0].m_idle   = idle;
    baselineSample.m_cores[0].m_irq    = irq;
    baselineSample.m_cores[0].m_other  = other;

    SysmonUtilizationSample currentSample;
    currentSample.m_cores.resize(numCores);
    currentSample.m_cores[0].m_user   = baselineSample.m_cores[0].m_user + user;
    currentSample.m_cores[0].m_nice   = baselineSample.m_cores[0].m_nice + nice;
    currentSample.m_cores[0].m_system = baselineSample.m_cores[0].m_system + system;
    currentSample.m_cores[0].m_idle   = baselineSample.m_cores[0].m_idle + idle;
    currentSample.m_cores[0].m_irq    = baselineSample.m_cores[0].m_irq + irq;
    currentSample.m_cores[0].m_other  = baselineSample.m_cores[0].m_other + other;

    double activeCycles = user + nice + system + irq + other;
    double totalCycles  = activeCycles + idle;
    CHECK(totalCycles == baselineSample.m_cores[0].GetTotal());
    CHECK(activeCycles == baselineSample.m_cores[0].GetActive());

    // No change in denominator -> error
    CHECK(-1 == sysmon.CalculateCoreUtilization(core, DCGM_FI_DEV_CPU_UTIL_USER_RATIO, baselineSample, baselineSample));

    CHECK(activeCycles / totalCycles
          == sysmon.CalculateCoreUtilization(core, DCGM_FI_DEV_CPU_UTIL_RATIO, baselineSample, currentSample));
    CHECK(user / totalCycles
          == sysmon.CalculateCoreUtilization(core, DCGM_FI_DEV_CPU_UTIL_USER_RATIO, baselineSample, currentSample));
    CHECK(nice / totalCycles
          == sysmon.CalculateCoreUtilization(core, DCGM_FI_DEV_CPU_UTIL_NICE_RATIO, baselineSample, currentSample));
    CHECK(system / totalCycles
          == sysmon.CalculateCoreUtilization(core, DCGM_FI_DEV_CPU_UTIL_SYS_RATIO, baselineSample, currentSample));
    CHECK(irq / totalCycles
          == sysmon.CalculateCoreUtilization(core, DCGM_FI_DEV_CPU_UTIL_IRQ_RATIO, baselineSample, currentSample));
}

static int makeCpuFreqEntry(const std::string &baseDir, const int cpuIndex, const int value)
{
    std::string cpuDir = fmt::format("{}/sys/devices/system/cpu/cpu{}", baseDir, cpuIndex);
    MKDIR_CHECKED_RC(cpuDir);
    std::string subDir = cpuDir + "/cpufreq";
    MKDIR_CHECKED_RC(subDir);
    std::string filename = fmt::format("{}/sys/devices/system/cpu/cpu{}/cpufreq/scaling_cur_freq", baseDir, cpuIndex);

    int ret = remove(filename.c_str());
    if (ret != 0)
    {
        if (errno == ENOENT)
        {
            ret = 0;
        }
        else
        {
            return ret;
        }
    }
    std::ofstream out(filename);
    out << fmt::format("{}", value);
    out.close();

    return 0;
}


int setupTzDirs(const std::string &baseDir)
{
    std::string thermRoot = fmt::format("{}/sys/class/thermal", baseDir);
    MKDIR_CHECKED_RC(baseDir.c_str());
    MKDIR_CHECKED_RC(fmt::format("{}/sys", baseDir.c_str()));
    MKDIR_CHECKED_RC(fmt::format("{}/sys/class", baseDir.c_str()));
    MKDIR_CHECKED_RC(thermRoot.c_str());

    // setup thermal zone 0-3 as socket 0-3
    for (int i = 0; i < 4; i++)
    {
        auto tzPath = fmt::format("{}/thermal_zone{}/device/description", thermRoot, i);
        MKDIR_CHECKED_RC(fmt::format("{}/thermal_zone{}", thermRoot, i).c_str());
        MKDIR_CHECKED_RC(fmt::format("{}/thermal_zone{}/device", thermRoot, i).c_str());
        WRITE_VALUE_TO_FILE_CHECKED_RC(tzPath, fmt::format("Thermal Zone Skt{} TJMax", i).c_str());
    }

    for (int i = 0; i < 3; i++)
    {
        auto tzPath = fmt::format("{}/thermal_zone{}/trip_point_0_type", thermRoot, i);
        WRITE_VALUE_TO_FILE_CHECKED_RC(tzPath, "critical");
        tzPath = fmt::format("{}/thermal_zone{}/trip_point_0_temp", thermRoot, i);
        WRITE_VALUE_TO_FILE_CHECKED_RC(tzPath, "104500");
        tzPath = fmt::format("{}/thermal_zone{}/trip_point_1_type", thermRoot, i);
        WRITE_VALUE_TO_FILE_CHECKED_RC(tzPath, "passive");
        tzPath = fmt::format("{}/thermal_zone{}/trip_point_1_temp", thermRoot, i);
        WRITE_VALUE_TO_FILE_CHECKED_RC(tzPath, "100000");
        tzPath = fmt::format("{}/thermal_zone{}/temp", thermRoot, i);
        WRITE_VALUE_TO_FILE_CHECKED_RC(tzPath, "9001");
    }

    // Force the type assignment to the unexpected case
    auto tzPath = fmt::format("{}/thermal_zone3/trip_point_0_type", thermRoot);
    WRITE_VALUE_TO_FILE_CHECKED_RC(tzPath, "passive");
    tzPath = fmt::format("{}/thermal_zone3/trip_point_0_temp", thermRoot);
    WRITE_VALUE_TO_FILE_CHECKED_RC(tzPath, "100000");
    tzPath = fmt::format("{}/thermal_zone3/trip_point_1_type", thermRoot);
    WRITE_VALUE_TO_FILE_CHECKED_RC(tzPath, "critical");
    tzPath = fmt::format("{}/thermal_zone3/trip_point_1_temp", thermRoot);
    WRITE_VALUE_TO_FILE_CHECKED_RC(tzPath, "104500");
    tzPath = fmt::format("{}/thermal_zone3/temp", thermRoot);
    WRITE_VALUE_TO_FILE_CHECKED_RC(tzPath, "9001");
    return 0;
}

TEST_CASE("DcgmModuleSysmon::PopulateTemperatureFileMap")
{
    ScopedEnvVars envVars(SupportNonNvidiaCpuEnv, "1");
    auto workingRoot = TempDir::Random("dcgm-sysmon-tz-");
    ScopedWorkingDirectory workingDirectory(workingRoot.Path());

    // Setup the directories
    auto tempTzDir = TempDir::Fixed("tz");
    std::string baseDir("tz");

    if (setupTzDirs(baseDir) != 0)
    {
        return;
    }

    TestDcgmModuleSysmon sysmon(g_coreCallbacks);
    // PopulateTemperatureFileMap is called implicitly, but we
    // need to set the base directory parameter and call again
    sysmon.m_tzBaseDir = std::move(baseDir);
    sysmon.PopulateTemperatureFileMap();

    CHECK(sysmon.m_socketTemperatureFileMap.size() == 4);
    CHECK(sysmon.m_socketTemperatureWarnFileMap.size() == 4);
    CHECK(sysmon.m_socketTemperatureCritFileMap.size() == 4);

    for (int i = 0; i < 3; i++)
    {
        auto tzPath = fmt::format("tz/sys/class/thermal/thermal_zone{}/trip_point_0_temp", i);
        CHECK(sysmon.m_socketTemperatureCritFileMap[i] == tzPath);
        tzPath = fmt::format("tz/sys/class/thermal/thermal_zone{}/trip_point_1_temp", i);
        CHECK(sysmon.m_socketTemperatureWarnFileMap[i] == tzPath);
        tzPath = fmt::format("tz/sys/class/thermal/thermal_zone{}/temp", i);
        CHECK(sysmon.m_socketTemperatureFileMap[i] == tzPath);
    }
    auto tzPath = fmt::format("tz/sys/class/thermal/thermal_zone3/trip_point_1_temp");
    CHECK(sysmon.m_socketTemperatureCritFileMap[3] == tzPath);
    tzPath = fmt::format("tz/sys/class/thermal/thermal_zone3/trip_point_0_temp");
    CHECK(sysmon.m_socketTemperatureWarnFileMap[3] == tzPath);
    tzPath = fmt::format("tz/sys/class/thermal/thermal_zone3/temp");
    CHECK(sysmon.m_socketTemperatureFileMap[3] == tzPath);
}

TEST_CASE("DcgmModuleSysmon::ReadTemperature")
{
    ScopedEnvVars envVars(SupportNonNvidiaCpuEnv, "1");
    auto workingRoot = TempDir::Random("dcgm-sysmon-tz-");
    ScopedWorkingDirectory workingDirectory(workingRoot.Path());

    // Setup the directories
    auto tempTzDir = TempDir::Fixed("tz");
    std::string baseDir("tz");

    if (setupTzDirs(baseDir) != 0)
    {
        return;
    }

    TestDcgmModuleSysmon sysmon(g_coreCallbacks);
    sysmon.m_tzBaseDir = std::move(baseDir);
    sysmon.PopulateTemperatureFileMap();

    for (int i = 0; i < 4; i++)
    {
        CHECK(sysmon.ReadTemperature(i, DCGM_FI_DEV_CPU_TEMP_CELSIUS) == 9.001);
        CHECK(sysmon.ReadTemperature(i, DCGM_FI_DEV_CPU_TEMP_WARNING_CELSIUS) == 100.0);
        CHECK(sysmon.ReadTemperature(i, DCGM_FI_DEV_CPU_TEMP_CRITICAL_CELSIUS) == 104.5);
    }

    // TZ that doesn't exist should return 0
    CHECK(sysmon.ReadTemperature(5, DCGM_FI_DEV_CPU_TEMP_CELSIUS) == 0.0);
    CHECK(sysmon.ReadTemperature(5, DCGM_FI_DEV_CPU_TEMP_WARNING_CELSIUS) == 0.0);
    CHECK(sysmon.ReadTemperature(5, DCGM_FI_DEV_CPU_TEMP_CRITICAL_CELSIUS) == 0.0);
}

TEST_CASE("DcgmModuleSysmon::ReadCoreSpeed")
{
    ScopedEnvVars envVars(SupportNonNvidiaCpuEnv, "1");
    auto workingRoot = TempDir::Random("dcgm-sysmon-core-speed-");
    ScopedWorkingDirectory workingDirectory(workingRoot.Path());

    // Number of skinnyjoe cores
    const int NUM_FAKE_CPUS = 288;
    TestDcgmModuleSysmon sysmon(g_coreCallbacks);

    auto tempSysfsDir = TempDir::Fixed("mocksysfs");
    std::string baseDir("mocksysfs");
    sysmon.m_coreSpeedBaseDir = baseDir;
    MKDIR_CHECKED(baseDir.c_str());
    std::string sysDir = baseDir + "/sys";
    MKDIR_CHECKED(sysDir.c_str());
    std::string devDir = sysDir + "/devices";
    MKDIR_CHECKED(devDir);
    std::string systemDir = devDir + "/system";
    MKDIR_CHECKED(systemDir);
    std::string cpuDir = systemDir + "/cpu";
    MKDIR_CHECKED(cpuDir);

    int ret;

    // drop the fake CPUs
    for (int i = 0; i < NUM_FAKE_CPUS; i++)
    {
        if ((ret = makeCpuFreqEntry(baseDir, i, i)) != 0)
        {
            break;
        }
    }

    if (ret != 0)
    {
        return;
    }

    // make sure we read back the correct values
    for (int i = 0; i < NUM_FAKE_CPUS; i++)
    {
        CHECK(sysmon.ReadCoreSpeed(DCGM_FE_CPU_CORE, i) == static_cast<uint64_t>(i));
    }

    // and make sure we return 0 for non-Core entities
    for (int i = 0; i < DCGM_FE_COUNT; i++)
    {
        if (i == DCGM_FE_CPU_CORE)
            continue;
        CHECK(sysmon.ReadCoreSpeed(i, 0) == DCGM_INT64_BLANK);
    }
}

struct PauseResumeTestFixture
{
    dcgm_core_msg_pause_resume_v1 msg {};
    ScopedEnvVars envVars { SupportNonNvidiaCpuEnv, "1" };

    PauseResumeTestFixture()
    {
        msg.header.moduleId   = DcgmModuleIdCore;
        msg.header.version    = dcgm_core_msg_pause_resume_version1;
        msg.header.subCommand = DCGM_CORE_SR_PAUSE_RESUME;
    }

    static TestDcgmModuleSysmon &getSysmon()
    {
        static TestDcgmModuleSysmon sysmonInst(g_coreCallbacks);
        return sysmonInst;
    }
};

TEST_CASE_METHOD(PauseResumeTestFixture,
                 "DcgmModuleSysmon::PauseResume Module resumed after initialization",
                 "[create]")
{
    TestDcgmModuleSysmon &sysmon = getSysmon();
    CHECK_FALSE(sysmon.m_paused);
}

TEST_CASE_METHOD(PauseResumeTestFixture, "DcgmModuleSysmon PauseResume Module rejects invalid messages", "[create]")
{
    SECTION("Module rejects message to incorrect moduleId")
    {
        TestDcgmModuleSysmon &sysmon = getSysmon();
        msg.header.moduleId          = DcgmModuleIdNvSwitch;
        msg.pause                    = true;
        dcgmReturn_t ret             = sysmon.ProcessMessage(&msg.header);
        CHECK(ret == DCGM_ST_BADPARAM);
    }

    SECTION("Module rejects message where subCommand and moduleId mismatch")
    {
        TestDcgmModuleSysmon &sysmon = getSysmon();
        msg.header.moduleId          = DcgmModuleIdSysmon;
        msg.pause                    = true;
        dcgmReturn_t ret             = sysmon.ProcessMessage(&msg.header);
        CHECK(ret == DCGM_ST_FUNCTION_NOT_FOUND);
        CHECK_FALSE(sysmon.m_paused);
    }

    SECTION("Module rejects message with incorrect version info")
    {
        TestDcgmModuleSysmon &sysmon = getSysmon();
        msg.header.version           = ~0;
        msg.header.moduleId          = DcgmModuleIdCore;
        msg.pause                    = true;
        CHECK_THROWS_AS(sysmon.ProcessMessage(&msg.header), VersionMismatchException);
        CHECK_FALSE(sysmon.m_paused);
    }
}

TEST_CASE_METHOD(PauseResumeTestFixture, "DcgmModuleSysmon PauseResume Module accepts valid messages", "[create]")
{
    SECTION("Module accepts valid pause message")
    {
        TestDcgmModuleSysmon &sysmon = getSysmon();
        msg.pause                    = true;
        dcgmReturn_t ret             = sysmon.ProcessMessage(&msg.header);
        CHECK(ret == DCGM_ST_OK);
        CHECK(sysmon.m_paused);
    }

    SECTION("Module accepts valid resume message")
    {
        TestDcgmModuleSysmon &sysmon = getSysmon();
        msg.pause                    = false;
        dcgmReturn_t ret             = sysmon.ProcessMessage(&msg.header);
        CHECK(ret == DCGM_ST_OK);
        CHECK_FALSE(sysmon.m_paused);
    }
}

TEST_CASE("DcgmModuleSysmon::ProcessMessage rejects null moduleCommand")
{
    ScopedEnvVars envVars(SupportNonNvidiaCpuEnv, "1");
    DcgmModuleSysmon sysmon(g_coreCallbacks);
    CHECK(sysmon.ProcessMessage(nullptr) == DCGM_ST_BADPARAM);
}
