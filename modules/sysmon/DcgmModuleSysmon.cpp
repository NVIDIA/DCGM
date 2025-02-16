/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
#include "DcgmModuleSysmon.h"
#include "DcgmSystemMonitor.h"

#include <DcgmEntityTypes.hpp>
#include <DcgmLogging.h>
#include <DcgmStringHelpers.h>
#include <dcgm_api_export.h>
#include <dcgm_helpers.h>
#include <dcgm_structs.h>

#include <algorithm>
#include <charconv>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <stdio.h>
namespace DcgmNs
{
/*****************************************************************************/
/* Adding this macro to validate that we're indeed in the correct thread */
#define ASSERT_IS_SYSMON_THREAD assert(std::this_thread::get_id() == m_sysmonThreadId)

using namespace DcgmNs::Cpu;

unsigned long long SysmonUtilizationSampleCore::GetTotal() const
{
    return m_user + m_nice + m_system + m_idle + m_irq + m_other;
}

unsigned long long SysmonUtilizationSampleCore::GetActive() const
{
    return GetTotal() - m_idle;
}

DcgmModuleSysmon::DcgmModuleSysmon(dcgmCoreCallbacks_t &dcc)
    : DcgmModuleWithCoreProxy(dcc)
    , m_procStat("/proc/stat")
    , m_paused(true)
    , m_sysmonThreadId(0)
{
    DCGM_LOG_DEBUG << "Constructing Sysmon Module";

    PopulateCpusIfNeeded();
    m_sysmon.Init();

    int st = Start();
    if (st)
    {
        DCGM_LOG_ERROR << "Got error " << st << " when trying to start the task runner";
        throw std::runtime_error("Unable to start a DcgmTaskRunner");
    }

    if (getenv(__DCGM_SYSMON_SKIP_HARDWARE_CHECK__) == nullptr)
    {
        if (m_sysmon.AreNvidiaCpusPresent() == false)
        {
            log_debug("Not starting sysmon because Nvidia CPUs aren't present. CPU Vendor is {}.",
                      m_sysmon.GetCpuVendor());
            throw std::runtime_error("Incompatible hardware vendor for sysmon.");
        }
    }

    m_paused = false;

    PopulateTemperatureFileMap();
}

DcgmModuleSysmon::~DcgmModuleSysmon()
{
    try
    {
        if (StopAndWait(60000))
        {
            DCGM_LOG_WARNING << "Not all threads for the Sysmon module exited correctly; exiting anyway";
            Kill();
        }
    }
    catch (std::exception const &ex)
    {
        DCGM_LOG_ERROR << "Exception caught while stopping the Sysmon module: " << ex.what();
    }
    catch (...)
    {
        DCGM_LOG_ERROR << "Unknown exception caught while stopping the Sysmon module";
    }
}

/*************************************************************************/
dcgmReturn_t DcgmModuleSysmon::ProcessMessage(dcgm_module_command_header_t *moduleCommand)
{
    dcgmReturn_t retSt       = DCGM_ST_OK;
    bool processInTaskRunner = false;

    if (moduleCommand->moduleId == DcgmModuleIdCore)
    {
        processInTaskRunner = true;
    }
    else if (moduleCommand->moduleId != DcgmModuleIdSysmon)
    {
        DCGM_LOG_ERROR << "Unexpected module command for module " << moduleCommand->moduleId;
        return DCGM_ST_BADPARAM;
    }
    else /* Sysmon module request */
    {
        switch (moduleCommand->subCommand)
        {
                /* Messages to process on the task runner */
            case DCGM_SYSMON_SR_GET_CPUS:
            case DCGM_SYSMON_SR_WATCH_FIELDS:
            case DCGM_SYSMON_SR_UNWATCH_FIELDS:
            case DCGM_SYSMON_SR_GET_ENTITY_STATUS:
            case DCGM_SYSMON_SR_CREATE_FAKE_ENTITIES:
            {
                processInTaskRunner = true;
                break;
            }

            default:
                DCGM_LOG_DEBUG << "Unknown subcommand: " << static_cast<int>(moduleCommand->subCommand);
                return DCGM_ST_FUNCTION_NOT_FOUND;
                break;
        }
    }

    if (processInTaskRunner)
    {
        using namespace DcgmNs;
        auto task = Enqueue(make_task("Process message in TaskRunner",
                                      [this, moduleCommand] { return ProcessMessageFromTaskRunner(moduleCommand); }));

        if (!task.has_value())
        {
            DCGM_LOG_ERROR << "Unable to enqueue Sysmon Module task";
            retSt = DCGM_ST_GENERIC_ERROR;
        }
        else
        {
            retSt = (*task).get();
        }
    }

    return retSt;
}

/*****************************************************************************/
dcgmReturn_t DcgmModuleSysmon::ProcessMessageFromTaskRunner(dcgm_module_command_header_t *moduleCommand)
{
    dcgmReturn_t retSt = DCGM_ST_OK;

    if (moduleCommand->moduleId == DcgmModuleIdCore)
    {
        retSt = ProcessCoreMessage(moduleCommand);
    }
    else if (moduleCommand->moduleId != DcgmModuleIdSysmon)
    {
        log_error("Unexpected module command for module {}", moduleCommand->moduleId);
        return DCGM_ST_BADPARAM;
    }
    else /* Sysmon module request */
    {
        switch (moduleCommand->subCommand)
        {
            case DCGM_SYSMON_SR_GET_CPUS:
            {
                retSt = ProcessGetCpus(moduleCommand);
                break;
            }
            case DCGM_SYSMON_SR_WATCH_FIELDS:
            {
                ProcessWatchFields(moduleCommand);
                break;
            }
            case DCGM_SYSMON_SR_UNWATCH_FIELDS:
            {
                ProcessUnwatchFields(moduleCommand);
                break;
            }
            case DCGM_SYSMON_SR_GET_ENTITY_STATUS:
            {
                ProcessGetEntityStatus(moduleCommand);
                break;
            }
            case DCGM_SYSMON_SR_CREATE_FAKE_ENTITIES:
            {
                ProcessCreateFakeEntities(moduleCommand);
                break;
            }

            default:
                DCGM_LOG_DEBUG << "Unknown subcommand: " << static_cast<int>(moduleCommand->subCommand);
                return DCGM_ST_FUNCTION_NOT_FOUND;
        }

        if (retSt == DCGM_ST_OK)
        {
            //
            // Reset the RunOnce last run time, so it will be called once the TaskRunner handles all events
            //
            [[maybe_unused]] auto _ = Enqueue(DcgmNs::make_task("Call RunOnce after all events are processed",
                                                                [this]() { ProcessTryRunOnce(true); }));
        }
    }

    return retSt;
}

/*****************************************************************************/
dcgmReturn_t DcgmModuleSysmon::ProcessGetCpus(GetCpusMessage msg)
{
    ASSERT_IS_SYSMON_THREAD;
    PopulateCpusIfNeeded();
    m_cpus.GetCpus(*msg.get());
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmModuleSysmon::ProcessWatchFields(WatchFieldsMessage msg)
{
    ASSERT_IS_SYSMON_THREAD;
    if (msg->numFieldIds >= SYSMON_MSG_WATCH_FIELDS_MAX_NUM_FIELDS || msg->numFieldIds == 0)
    {
        DCGM_LOG_ERROR << "Invalid numFields " << msg->numFieldIds;
        return DCGM_ST_BADPARAM;
    }

    for (unsigned int entityIndex = 0; entityIndex < msg->numEntities; entityIndex++)
    {
        dcgmGroupEntityPair_t const &entity = msg->entityPairs[entityIndex];
        switch (entity.entityGroupId)
        {
            case DCGM_FE_CPU:
            {
                if (!m_cpus.IsValidCpuId(CpuId { entity.entityId }))
                {
                    log_error("Invalid CPU {}", entity.entityId);
                    continue;
                }
                break;
            }
            case DCGM_FE_CPU_CORE:
            {
                if (!m_cpus.IsValidCoreId(CoreId { entity.entityId }))
                {
                    log_error("Invalid core {}", entity.entityId);
                    continue;
                }
                break;
            }
            default:
                log_error("Invalid eg {} for eid {}", entity.entityGroupId, entity.entityId);
                continue;
        }


        for (unsigned int fieldIndex = 0; fieldIndex < msg->numFieldIds; fieldIndex++)
        {
            if (DcgmWatchTable::IsFieldIgnored(msg->fieldIds[fieldIndex], DcgmModuleIdSysmon))
            {
                log_error("Module sysmon cannot watch field ID {}", msg->fieldIds[fieldIndex]);
                return DCGM_ST_BADPARAM;
            }
            else
            {
                EnableMonitoring(SYSMON_MONITORING_SWITCH_UTILIZATION);
                log_debug("Adding watcher eg {} eid {} field {} updateInterval {} maxAge {} maxKeepSamples {}",
                          entity.entityGroupId,
                          entity.entityId,
                          msg->fieldIds[fieldIndex],
                          msg->updateIntervalUsec,
                          msg->maxKeepAge,
                          msg->maxKeepSamples);
            }

            using namespace DcgmNs::Timelib;
            using DcgmNs::Utils::GetMaxAge;
            using std::chrono::milliseconds, std::chrono::microseconds, std::chrono::duration;
            const double slackMultiplier = 2; // 200% slack
            const timelib64_t maxKeepAge = ToLegacyTimestamp(
                GetMaxAge(duration_cast<milliseconds>(FromLegacyTimestamp<microseconds>(msg->updateIntervalUsec)),
                          duration_cast<milliseconds>(duration<double>(msg->maxKeepAge)),
                          msg->maxKeepSamples,
                          slackMultiplier));

            m_watchTable.AddWatcher(entity.entityGroupId,
                                    entity.entityId,
                                    msg->fieldIds[fieldIndex],
                                    msg->watcher,
                                    msg->updateIntervalUsec,
                                    maxKeepAge,
                                    true);

            timelib64_t minSampleAgeUsec, maxSampleAgeUsec;
            m_watchTable.GetMaxAgeUsecAllWatches(minSampleAgeUsec, maxSampleAgeUsec);
            m_maxSampleAge = TimePoint(FromLegacyTimestamp<microseconds>(maxSampleAgeUsec));
        }
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmModuleSysmon::EnableMonitoring(unsigned int monitoringSwitch)
{
    ASSERT_IS_SYSMON_THREAD;
    switch (monitoringSwitch)
    {
        case SYSMON_MONITORING_SWITCH_UTILIZATION:
            m_enabledMonitoring[monitoringSwitch] = true;
            break;
        default:
            return DCGM_ST_BADPARAM;
    }
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmModuleSysmon::ProcessUnwatchFields(UnwatchFieldsMessage msg)
{
    ASSERT_IS_SYSMON_THREAD;
    // Not needed for sysmon watches
    dcgmPostWatchInfo_t *postWatchInfo = nullptr;
    timelib64_t minSampleAgeUsec, maxSampleAgeUsec;
    using namespace DcgmNs::Timelib;
    using std::chrono::microseconds;

    m_watchTable.RemoveWatches(msg->watcher, postWatchInfo);
    m_watchTable.GetMaxAgeUsecAllWatches(minSampleAgeUsec, maxSampleAgeUsec);
    m_maxSampleAge = TimePoint(FromLegacyTimestamp<microseconds>(maxSampleAgeUsec));
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmModuleSysmon::ProcessGetEntityStatus(GetEntityStatusMessage msg)
{
    ASSERT_IS_SYSMON_THREAD;
    msg->entityStatus = DcgmEntityStatusUnknown;

    switch (msg->entityGroupId)
    {
        case DCGM_FE_CPU:
        {
            if (!m_cpus.IsValidCpuId(CpuId { msg->entityId }))
            {
                log_error("Invlaid CPU node eid {}", msg->entityId);
            }
            else
            {
                msg->entityStatus = DcgmEntityStatusOk;
            }
        }
        break;
        case DCGM_FE_CPU_CORE:
        {
            if (!m_cpus.IsValidCoreId(CoreId { msg->entityId }))
            {
                log_error("Invlaid CPU eid {}", msg->entityId);
            }
            else
            {
                msg->entityStatus = DcgmEntityStatusOk;
            }
        }
        break;
        default:
            log_error("Received eg {} eid {}", msg->entityGroupId, msg->entityId);
            return DCGM_ST_BADPARAM;
    }
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmModuleSysmon::ProcessCreateFakeEntities(CreateFakeEntitiesMessage msg)
{
    ASSERT_IS_SYSMON_THREAD;
    for (unsigned int i = 0; i < msg->numToCreate; i++)
    {
        switch (msg->groupToCreate)
        {
            case DCGM_FE_CPU:
            {
                CpuId cpuId = m_cpus.AddFakeCpu();
                if (cpuId.id == DCGM_MAX_NUM_CPUS)
                {
                    log_error("Cannot create a fake CPU because we've already reached the maximum number of CPUs.");
                    return DCGM_ST_INSUFFICIENT_RESOURCES;
                }

                // SUCCESS
                msg->numCreated++;
                msg->ids[i] = cpuId.id;
                break;
            }
            case DCGM_FE_CPU_CORE:
            {
                if (msg->parent.entityGroupId != DCGM_FE_CPU)
                {
                    log_error("Cores can only have CPUs as parents, but {} was requested.", msg->parent.entityGroupId);
                    return DCGM_ST_BADPARAM;
                }

                CoreId coreId = m_cpus.AddFakeCore(CpuId { msg->parent.entityId });
                if (coreId.id == DCGM_MAX_NUM_CPU_CORES)
                {
                    log_error("Cannot create a fake core because we've already reached the maximum number of CPUs.");
                    return DCGM_ST_INSUFFICIENT_RESOURCES;
                }

                // SUCCESS
                msg->numCreated++;
                msg->ids[i] = coreId.id;
                break;
            }
            default:
            {
                return DCGM_ST_NOT_SUPPORTED;
                break;
            }
        }
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmModuleSysmon::ProcessClientDisconnect(dcgm_core_msg_client_disconnect_t *msg)
{
    ASSERT_IS_SYSMON_THREAD;
    DCGM_LOG_INFO << "Unwatching fields watched by connection " << msg->connectionId;
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmModuleSysmon::ProcessCoreMessage(dcgm_module_command_header_t *moduleCommand)
{
    ASSERT_IS_SYSMON_THREAD;
    dcgmReturn_t retSt = DCGM_ST_OK;

    switch (moduleCommand->subCommand)
    {
        case DCGM_CORE_SR_CLIENT_DISCONNECT:
            retSt = ProcessClientDisconnect((dcgm_core_msg_client_disconnect_t *)moduleCommand);
            break;

        case DCGM_CORE_SR_LOGGING_CHANGED:
            OnLoggingSeverityChange((dcgm_core_msg_logging_changed_t *)moduleCommand);
            break;

        case DCGM_CORE_SR_PAUSE_RESUME:
            retSt = ProcessPauseResumeMessage(moduleCommand);
            break;

        default:
            DCGM_LOG_DEBUG << "Unknown subcommand: " << static_cast<int>(moduleCommand->subCommand);
            return DCGM_ST_FUNCTION_NOT_FOUND;
    }

    return retSt;
}

/*****************************************************************************/
static std::tuple<unsigned int, unsigned int> getFirstLastSystemNodes()
{
    std::ifstream file;
    file.open("/sys/devices/system/node/has_cpu");

    // Node range expected to be in the form "x-y" or "x"
    std::string nodeRange;
    file >> nodeRange;

    auto firstLastNode = DcgmNs::Split(nodeRange, '-');

    if (firstLastNode.size() == 2)
    {
        unsigned int firstNode = strtoul(std::string(firstLastNode[0]).c_str(), nullptr, 10);
        unsigned int lastNode  = strtoul(std::string(firstLastNode[1]).c_str(), nullptr, 10);
        return { firstNode, lastNode };
    }
    else if (firstLastNode.size() == 1)
    {
        unsigned int node = strtoul(nodeRange.c_str(), nullptr, 10);
        return { node, node };
    }
    else
    {
        log_error("Could not enumerate NODEs");
        throw std::runtime_error("Coud not enumerate NODEs");
    }
}

/*****************************************************************************/
dcgmReturn_t DcgmModuleSysmon::PopulateOwnedCoresBitmaskFromRangeString(dcgm_sysmon_cpu_t &cpu,
                                                                        const std::string &rangeStr)
{
    cpu.ownedCores.version = dcgmCpuHierarchyOwnedCores_version1;
    auto cpuRanges         = DcgmNs::Split(rangeStr, ',');

    std::memset(cpu.ownedCores.bitmask, 0, sizeof(cpu.ownedCores.bitmask));
    for (auto range : cpuRanges)
    {
        auto firstLastCpu = DcgmNs::Split(range, '-');

        unsigned int firstCpu = 0;
        unsigned int lastCpu  = 0;

        if (firstLastCpu.size() == 2)
        {
            if (firstLastCpu[0].empty() || firstLastCpu[1].empty())
            {
                log_error("Section '{}' of range string '{}' is malformed", range, rangeStr);
                return DCGM_ST_BADPARAM;
            }
            firstCpu = strtoul(std::string(firstLastCpu[0]).c_str(), nullptr, 10);
            lastCpu  = strtoul(std::string(firstLastCpu[1]).c_str(), nullptr, 10);
        }
        else if (firstLastCpu.size() == 1)
        {
            if (firstLastCpu[0].empty())
            {
                log_error("Section '{}' of range string '{}' is malformed", range, rangeStr);
                return DCGM_ST_BADPARAM;
            }
            firstCpu = lastCpu = strtoul(std::string(firstLastCpu[0]).c_str(), nullptr, 10);
        }
        else
        {
            log_error("Section '{}' of range string '{}' is malformed", range, rangeStr);
            return DCGM_ST_BADPARAM;
        }

        const unsigned int ownedCoresSize = CHAR_BIT * sizeof(cpu.ownedCores.bitmask);
        if (firstCpu > ownedCoresSize || lastCpu >= ownedCoresSize)
        {
            return DCGM_ST_INSUFFICIENT_SIZE;
        }

        const unsigned int ownedCoresMemberSize = CHAR_BIT * sizeof(cpu.ownedCores.bitmask[0]);
        for (unsigned int cpuId = firstCpu; cpuId <= lastCpu; cpuId++)
        {
            unsigned int index  = cpuId / ownedCoresMemberSize;
            unsigned int offset = cpuId % ownedCoresMemberSize;

            cpu.ownedCores.bitmask[index] |= (1ULL << offset);
            cpu.coreCount++;
        }
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
void DcgmModuleSysmon::PopulateOwnedCoresBitmask(dcgm_sysmon_cpu_t &cpu)
{
    std::ifstream file;
    std::string filename = fmt::format("/sys/devices/system/node/node{}/cpulist", cpu.cpuId);
    file.open(filename);

    std::string cpuRange;
    file >> cpuRange;

    dcgmReturn_t ret = PopulateOwnedCoresBitmaskFromRangeString(cpu, cpuRange);
    if (ret == DCGM_ST_BADPARAM)
    {
        throw std::runtime_error(fmt::format("Could not enumerate cpus in range: '{}'", cpuRange));
    }
    else if (ret == DCGM_ST_INSUFFICIENT_SIZE)
    {
        throw std::runtime_error(fmt::format("Core index in range '{}' too large for ownedCores struct", cpuRange));
    }
    else if (ret != DCGM_ST_OK)
    {
        throw std::runtime_error(fmt::format("Unable to parse CPU range '{}'", cpuRange));
    }
}

/*****************************************************************************/
dcgmReturn_t DcgmModuleSysmon::PopulateCpusIfNeeded()
{
    // ASSERT_IS_SYSMON_THREAD; - Called before thread is started.

    if (!m_cpus.IsEmpty())
    {
        return DCGM_ST_ALREADY_INITIALIZED;
    }
    auto [firstNode, lastNode] = getFirstLastSystemNodes();
    CpuHelpers cpuHelpers;
    auto cpuSerialNumbers      = cpuHelpers.GetCpuSerials();
    bool getSerialNumberFailed = false;
    if (!cpuSerialNumbers.has_value() || cpuSerialNumbers->size() != (lastNode - firstNode + 1))
    {
        log_warning("Could not retrieve serial numbers for CPUs");
        getSerialNumberFailed = true;
    }

    for (unsigned int cpuId = firstNode; cpuId <= lastNode; cpuId++)
    {
        dcgm_sysmon_cpu_t cpu {};
        cpu.cpuId = cpuId;
        PopulateOwnedCoresBitmask(cpu);
        if (!getSerialNumberFailed)
        {
            SafeCopyTo(cpu.serial, cpuSerialNumbers->at(cpuId).c_str());
        }

        m_cpus.AddCpu(cpu);
    }

    m_cpuTopology.Initialize(m_cpus.GetCpus());

    return DCGM_ST_OK;
}

/*****************************************************************************/
static std::string readEntireFile(std::ifstream &file)
{
    file.clear();
    file.seekg(0);
    std::string contents(std::istreambuf_iterator<char>(file), {});
    return contents;
}

/*****************************************************************************/
dcgmReturn_t DcgmModuleSysmon::ParseProcStatCpuLine(const std::string &line, SysmonUtilizationSample &sample)
{
    /*
     * Looking for lines in the format (some systems might not have info after softirq):
     * cpu0 9718962 9988 2368503 659591203 177159 0 8903 0 0 0
     * cpu<index> user nice system idle iowait irq softirq steal guest guest_nice
     */
    if (line.compare(0, 3, "cpu") != 0)
    {
        return DCGM_ST_OK;
    }

    std::string cpuUtilSubstring = line.substr(3);

    unsigned int coreIndex;
    unsigned long long user   = 0;
    unsigned long long nice   = 0;
    unsigned long long system = 0;
    unsigned long long idle   = 0;
    unsigned long long iowait = 0;
    unsigned long long irq    = 0;
    unsigned long long other  = 0;
    unsigned long long temp   = 0;

    std::istringstream cpuUtilStream(cpuUtilSubstring);
    cpuUtilStream >> coreIndex >> user >> nice >> system >> idle >> iowait >> irq;

    // other sums up the counters we don't expose as metrics
    other = iowait;
    while (cpuUtilStream >> temp)
    {
        other += temp;
    }

    // We should stop reading because the line ended
    if (!cpuUtilStream.eof())
    {
        log_error("Could not parse stat line: {}", line);
        return DCGM_ST_BADPARAM;
    }

    if (coreIndex >= sample.m_cores.size())
    {
        log_error("Core index {} >= sample core count {}", coreIndex, sample.m_cores.size());
        return DCGM_ST_BADPARAM;
    }

    auto &coreObj    = sample.m_cores[coreIndex];
    coreObj.m_user   = user;
    coreObj.m_nice   = nice;
    coreObj.m_idle   = idle;
    coreObj.m_system = system;
    coreObj.m_irq    = irq;
    coreObj.m_other  = other;

    return DCGM_ST_OK;
}

/*****************************************************************************/
const SysmonUtilizationSample &DcgmModuleSysmon::ReadUtilizationSample(DcgmNs::Timelib::TimePoint now)
{
    ASSERT_IS_SYSMON_THREAD;

    // Check if we have already read the sample
    auto currentSampleIt = m_utilizationSamples.find(now);
    if (currentSampleIt != m_utilizationSamples.end())
    {
        return currentSampleIt->second;
    }

    std::string statContents = readEntireFile(m_procStat);
    std::istringstream statStream(statContents);
    std::string line;
    SysmonUtilizationSample sample;
    sample.m_timestamp = now;
    // Allocate space in the sample for all the cores
    sample.m_cores.resize(m_cpus.GetTotalCoreCount());

    std::getline(statStream, line); // Skip first line which lists aggregate stats for the system

    while (std::getline(statStream, line))
    {
        dcgmReturn_t ret = ParseProcStatCpuLine(line, sample);
        if (ret != DCGM_ST_OK)
        {
            log_error("Couldn't parse proc stat line: '{}': {}", line, errorString(ret));
        }
    }

    m_utilizationSamples.emplace(now, sample);
    return m_utilizationSamples[now];
}

dcgmReturn_t DcgmModuleSysmon::UpdateField(DcgmNs::Timelib::TimePoint now, const dcgm_field_update_info_t &updateInfo)
{
    ASSERT_IS_SYSMON_THREAD;

    using namespace DcgmNs::Timelib;
    using namespace std::chrono;

    TimePoint baselineTime = now - FromLegacyTimestamp<milliseconds>(updateInfo.updateIntervalUsec);

    switch (updateInfo.fieldMeta->fieldId)
    {
        case DCGM_FI_DEV_CPU_UTIL_TOTAL:
        case DCGM_FI_DEV_CPU_UTIL_USER:
        case DCGM_FI_DEV_CPU_UTIL_NICE:
        case DCGM_FI_DEV_CPU_UTIL_SYS:
        case DCGM_FI_DEV_CPU_UTIL_IRQ:
        {
            const SysmonUtilizationSample &currentSample = ReadUtilizationSample(now);

            // upper_bound returns the first element with a key above upper_bound
            auto baselineSamplePlusOneIt = m_utilizationSamples.upper_bound(baselineTime);
            if (baselineSamplePlusOneIt == m_utilizationSamples.begin())
            {
                return DCGM_ST_NO_DATA;
            }
            auto baselineSampleIt                         = std::prev(baselineSamplePlusOneIt);
            const SysmonUtilizationSample &baselineSample = baselineSampleIt->second;

            double value = CalculateUtilizationForEntity(updateInfo.entityGroupId,
                                                         updateInfo.entityId,
                                                         updateInfo.fieldMeta->fieldId,
                                                         baselineSample,
                                                         currentSample);

            if (value == -1)
            {
                log_error("Could not calculate utlization eg {} eid {} fieldId {}",
                          updateInfo.entityGroupId,
                          updateInfo.entityId,
                          updateInfo.fieldMeta->fieldId);
                return DCGM_ST_NO_DATA;
            }

            // Now send value to the cache
            DcgmFvBuffer buf;
            buf.AddDoubleValue(updateInfo.entityGroupId,
                               updateInfo.entityId,
                               updateInfo.fieldMeta->fieldId,
                               value,
                               ToLegacyTimestamp(now),
                               DCGM_ST_OK);

            dcgmReturn_t ret = m_coreProxy.AppendSamples(&buf);
            if (ret != DCGM_ST_OK)
            {
                log_warning("Failed to append samples to the cache: {}", errorString(ret));
            }
            break;
        }
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
void DcgmModuleSysmon::ProcessPruneSamples(DcgmNs::Timelib::TimePoint now)
{
    ASSERT_IS_SYSMON_THREAD;
    log_debug("Pruning samples");

    using namespace std::ranges;
    using namespace DcgmNs::Timelib;

    TimePoint cutOffMinimumExclusive = TimePoint(now - m_maxSampleAge);

    auto upperBound = m_utilizationSamples.upper_bound(cutOffMinimumExclusive);

    if (upperBound == m_utilizationSamples.end())
    {
        log_debug("Pruning all samples (upper_bound > now)");
    }

    m_utilizationSamples.erase(m_utilizationSamples.begin(), upperBound);
    log_debug("Pruned old samples. utilizationSamples.size = {}", m_utilizationSamples.size());
}

/*****************************************************************************/
void DcgmModuleSysmon::UpdateFields(timelib64_t &nextUpdateTimeUsec)
{
    ASSERT_IS_SYSMON_THREAD;

    using namespace DcgmNs::Timelib;
    using namespace std::chrono;

    log_debug("Updating fields");

    if (m_paused)
    {
        log_warning("Module paused. Skipping updating fields");
        return;
    }

    std::vector<dcgm_field_update_info_t> toUpdate;
    TimePoint now = Now();
    dcgmReturn_t ret
        = m_watchTable.GetFieldsToUpdate(DcgmModuleIdSysmon, ToLegacyTimestamp(now), toUpdate, nextUpdateTimeUsec);

    if (ret != DCGM_ST_OK)
    {
        log_warning("Received {}", errorString(ret));
        return;
    }

    for (const dcgm_field_update_info_t &updateInfo : toUpdate)
    {
        milliseconds updateInterval = FromLegacyTimestamp<milliseconds>(updateInfo.updateIntervalUsec);
        UpdateField(now, updateInfo);
    }

    timelib64_t minUpdateIntervalUsec, maxUpdateIntervalUsec;
    m_watchTable.GetMinAndMaxUpdateInterval(minUpdateIntervalUsec, maxUpdateIntervalUsec);

    // Convert us->ms
    milliseconds minUpdateInterval = duration_cast<milliseconds>(microseconds(minUpdateIntervalUsec));

    // if minUpdateInterval == 0, we have no watches -> sleep for 5s
    // otherwise, set a floor of 10ms on update interval
    if (minUpdateInterval == milliseconds(0))
    {
        m_runInterval = milliseconds(5000);
    }
    else
    {
        m_runInterval = std::max(minUpdateInterval, milliseconds(10));
    }

    RecordMetrics(ToLegacyTimestamp(now), toUpdate);
    ProcessPruneSamples(now);
}

/*****************************************************************************/
double DcgmModuleSysmon::CalculateCoreUtilization(CoreId core,
                                                  unsigned int fieldId,
                                                  const SysmonUtilizationSample &baselineSample,
                                                  const SysmonUtilizationSample &currentSample)
{
    if (currentSample.m_cores.size() <= core.id || baselineSample.m_cores.size() <= core.id)
    {
        log_error("Invalid core {}", core.id);
        return -1;
    }

    double value = 0;

    auto &baselineCore = baselineSample.m_cores[core.id];
    auto &currentCore  = currentSample.m_cores[core.id];

    switch (fieldId)
    {
        case DCGM_FI_DEV_CPU_UTIL_TOTAL:
            value = currentCore.GetActive() - baselineCore.GetActive();
            break;
        case DCGM_FI_DEV_CPU_UTIL_USER:
            value = currentCore.m_user - baselineCore.m_user;
            break;
        case DCGM_FI_DEV_CPU_UTIL_NICE:
            value = currentCore.m_nice - baselineCore.m_nice;
            break;
        case DCGM_FI_DEV_CPU_UTIL_SYS:
            value = currentCore.m_system - baselineCore.m_system;
            break;
        case DCGM_FI_DEV_CPU_UTIL_IRQ:
            value = currentCore.m_irq - baselineCore.m_irq;
            break;
        default:
            log_error("Invalid field {}", fieldId);
            return -1;
    }

    double totalCyclesCurrent  = currentCore.GetTotal();
    double totalCyclesBaseline = baselineCore.GetTotal();
    double denominator         = totalCyclesCurrent - totalCyclesBaseline;
    if (denominator == 0)
    {
        log_warning("Denominator == 0");
        return -1;
    }

    value /= denominator;

    return value;
}

/*****************************************************************************/
double DcgmModuleSysmon::CalculateCpuUtilization(CpuId cpu,
                                                 unsigned int fieldId,
                                                 const SysmonUtilizationSample &baselineSample,
                                                 const SysmonUtilizationSample &currentSample)
{
    ASSERT_IS_SYSMON_THREAD;

    double value                      = 0;
    std::vector<unsigned int> coreIds = m_cpus.GetCoreIdList(cpu.id);
    for (auto &coreId : coreIds)
    {
        value += CalculateCoreUtilization(CoreId { coreId }, fieldId, baselineSample, currentSample);
    }
    return value / coreIds.size();
}

/*****************************************************************************/
double DcgmModuleSysmon::CalculateUtilizationForEntity(unsigned int entityGroupId,
                                                       unsigned int entityId,
                                                       unsigned int fieldId,
                                                       const SysmonUtilizationSample &baselineSample,
                                                       const SysmonUtilizationSample &currentSample)
{
    switch (fieldId)
    {
        case DCGM_FI_DEV_CPU_UTIL_TOTAL:
        case DCGM_FI_DEV_CPU_UTIL_USER:
        case DCGM_FI_DEV_CPU_UTIL_NICE:
        case DCGM_FI_DEV_CPU_UTIL_SYS:
        case DCGM_FI_DEV_CPU_UTIL_IRQ:
            break;
        default:
            log_error("Invalid field {}", fieldId);
            return -1;
    }

    if (entityGroupId == DCGM_FE_CPU_CORE)
    {
        return CalculateCoreUtilization(CoreId { entityId }, fieldId, baselineSample, currentSample);
    }
    else if (entityGroupId == DCGM_FE_CPU)
    {
        return CalculateCpuUtilization(CpuId { entityId }, fieldId, baselineSample, currentSample);
    }

    log_error("Invalid eg {}", entityGroupId);
    return -1;
}

/*****************************************************************************/
double DcgmModuleSysmon::ReadTemperature(unsigned int cpuId, short fieldId)
{
    std::string path;
    switch (fieldId)
    {
        case DCGM_FI_DEV_CPU_TEMP_CURRENT:
            path = m_socketTemperatureFileMap[cpuId];
            break;
        case DCGM_FI_DEV_CPU_TEMP_WARNING:
            path = m_socketTemperatureWarnFileMap[cpuId];
            break;
        case DCGM_FI_DEV_CPU_TEMP_CRITICAL:
            path = m_socketTemperatureCritFileMap[cpuId];
            break;
        default:
            log_error("Unknown CPU temperature field id {}", fieldId);
            return 0.0;
    }
    std::ifstream file(path);
    if (file)
    {
        std::string contents = readEntireFile(file);
        // Format is 43900 for 43.9 degrees, or 104500 for 104.5 degrees
        unsigned int tempAdjusted = std::stoi(contents);

        return static_cast<double>(tempAdjusted) / 1000.0;
    }
    else
    {
        SYSMON_LOG_IFSTREAM_ERROR("CPU temperature", path);
        return 0.0;
    }
}

/*****************************************************************************/
unsigned int DcgmModuleSysmon::GetSocketFromThermalZoneFileContents(const std::string &path,
                                                                    const std::string &contents)
{
    static const std::string TEMP_MATCH_WORD   = "TJMax";
    static const std::string SOCKET_MATCH_WORD = "Skt";

    // Format for matches should be 'Thermal Zone Skt# TJMax'
    auto tokens = DcgmNs::Split(contents, ' ');
    if (tokens.size() != 4)
    {
        log_debug("Thermal zone file contents ('{}') didn't match correct format: '\\w \\w \\w \\w'", contents);
        return SYSMON_INVALID_SOCKET_ID;
    }

    // Sometimes there's a trailing newline, so just make sure we match for the length of TEMP_MATCH_WORD
    if (!strncmp(tokens[3].data(), TEMP_MATCH_WORD.c_str(), TEMP_MATCH_WORD.size())
        && tokens[2].starts_with(SOCKET_MATCH_WORD))
    {
        unsigned int socketId;
        std::string_view sv(tokens[2].substr(SOCKET_MATCH_WORD.size()));
        auto [ptr, ec] = std::from_chars(sv.data(), sv.data() + sv.size(), socketId);
        if (ec != std::errc())
        {
            log_debug("Couldn't parse a number from: '{}', ignoring file {}", sv, path);
        }
        else if (m_socketTemperatureFileMap.count(socketId) > 0)
        {
            log_debug("Ignoring duplicate socket temperature path '{}' with contents '{}'", path, contents);
        }
        else
        {
            return socketId;
        }
    }
    else
    {
        log_debug("Ignoring non-matching file contents: '{}'", contents);
    }

    return SYSMON_INVALID_SOCKET_ID;
}

/*****************************************************************************/
unsigned int DcgmModuleSysmon::GetSocketIdFromThermalFile(const std::string &path)
{
    std::ifstream file(path);

    if (file)
    {
        std::string contents = readEntireFile(file);

        return GetSocketFromThermalZoneFileContents(path, contents);
    }
    else
    {
        SYSMON_LOG_IFSTREAM_DEBUG("thermal zone", path);
    }

    return SYSMON_INVALID_SOCKET_ID;
}

/*****************************************************************************/
void DcgmModuleSysmon::PopulateTemperatureFileMap()
{
    // ASSERT_IS_SYSMON_THREAD; -- Not invoked from thread.

    std::string THERMAL_BASE_PATH(fmt::format("{}/sys/class/thermal", m_tzBaseDir));
    static const std::string THERMAL_PATH_EXTENSION("device/description");
    static const std::string THERMAL_DIR_NAME_START("thermal_zone");
    static const std::string THERMAL_TEMPERATURE_FILENAME("temp");
    static const std::string THERMAL_TEMPERATURE_TRIPTYPE0_FILENAME("trip_point_0_type");
    static const std::string THERMAL_TEMPERATURE_TRIPTEMP0_FILENAME("trip_point_0_temp");
    static const std::string THERMAL_TEMPERATURE_TRIPTYPE1_FILENAME("trip_point_1_type");
    static const std::string THERMAL_TEMPERATURE_TRIPTEMP1_FILENAME("trip_point_1_temp");

    if (m_socketTemperatureFileMap.empty() == false)
    {
        // Already populated
        return;
    }

    // Make opendir() obey RAII
    auto dirDeleter = [](DIR *dir) {
        if (dir != nullptr)
        {
            closedir(dir);
            dir = nullptr;
        }
    };

    auto dir = std::unique_ptr<DIR, decltype(dirDeleter)>(opendir(THERMAL_BASE_PATH.c_str()), dirDeleter);
    if (!dir)
    {
        auto syserr = std::system_error(errno, std::generic_category());
        log_info("Could not open directory '{}'", THERMAL_BASE_PATH);
        log_debug("Got opendir error: ({}) {}", syserr.code().value(), syserr.what());
        return;
    }

    struct dirent *entry = nullptr;

    while ((entry = readdir(dir.get())) != nullptr)
    {
        if (entry->d_type != DT_DIR || !std::string_view { entry->d_name }.starts_with(THERMAL_DIR_NAME_START))
        {
            // Not a socket temperature match candidate
            log_verbose(
                "Ignoring directory '{}' among thermal zone candidates in path {}", entry->d_type, THERMAL_BASE_PATH);
            continue;
        }

        auto pathPrefix = fmt::format("{}/{}", THERMAL_BASE_PATH, entry->d_name);

        auto path = fmt::format("{}/{}", pathPrefix, THERMAL_PATH_EXTENSION);

        unsigned int socketId = GetSocketIdFromThermalFile(path);
        if (socketId != SYSMON_INVALID_SOCKET_ID)
        {
            auto tempFilePath = fmt::format("{}/{}", pathPrefix, THERMAL_TEMPERATURE_FILENAME);
            log_debug("Recording temperature path '{}' for Socket {}", tempFilePath, socketId);
            m_socketTemperatureFileMap[socketId] = std::move(tempFilePath);

            auto trip0TypePath = fmt::format("{}/{}", pathPrefix, THERMAL_TEMPERATURE_TRIPTYPE0_FILENAME);
            auto trip1TypePath = fmt::format("{}/{}", pathPrefix, THERMAL_TEMPERATURE_TRIPTYPE1_FILENAME);
            auto trip0TempPath = fmt::format("{}/{}", pathPrefix, THERMAL_TEMPERATURE_TRIPTEMP0_FILENAME);
            auto trip1TempPath = fmt::format("{}/{}", pathPrefix, THERMAL_TEMPERATURE_TRIPTEMP1_FILENAME);
            std::ifstream type0File(trip0TypePath);
            std::ifstream type1File(trip1TypePath);
            auto type0Type = readEntireFile(type0File);
            auto type1Type = readEntireFile(type1File);

            if (!strncmp(type0Type.c_str(), "critical", 8))
            {
                m_socketTemperatureCritFileMap[socketId] = std::move(trip0TempPath);
            }
            else if (!strncmp(type0Type.c_str(), "passive", 7))
            {
                m_socketTemperatureWarnFileMap[socketId] = std::move(trip0TempPath);
            }

            if (!strncmp(type1Type.c_str(), "critical", 8))
            {
                m_socketTemperatureCritFileMap[socketId] = std::move(trip1TempPath);
            }
            else if (!strncmp(type1Type.c_str(), "passive", 7))
            {
                m_socketTemperatureWarnFileMap[socketId] = std::move(trip1TempPath);
            }

            if (!m_socketTemperatureWarnFileMap.contains(socketId))
            {
                log_debug("Warning temperature file path not found for socket id {}.", socketId);
            }
            if (!m_socketTemperatureCritFileMap.contains(socketId))
            {
                log_debug("Critical temperature file path not found for socket id {}.", socketId);
            }
        }
    }
}

uint64_t DcgmModuleSysmon::ReadCoreSpeed(unsigned int entityGroupId, unsigned int entityId)
{
    if (entityGroupId == DCGM_FE_CPU_CORE)
    {
        auto path
            = fmt::format("{}/sys/devices/system/cpu/cpu{}/cpufreq/scaling_cur_freq", m_coreSpeedBaseDir, entityId);
        std::ifstream file(path);
        if (file)
        {
            std::string contents;
            file >> contents;
            uint64_t coreSpeed = std::stoi(contents);
            return coreSpeed;
        }
        SYSMON_LOG_IFSTREAM_ERROR("cpu frequency", path);
        return DCGM_INT64_BLANK;
    }
    // CPU speeds currently require dmidecode calls to retrieve, and that is
    // too heavy weight to do regularly. This may be addressed in future revisions.
    log_verbose("Unsupported entity type for CPU/Core Speed: {}.", entityGroupId);
    return DCGM_INT64_BLANK;
}

/*****************************************************************************/
unsigned int DcgmModuleSysmon::GetSocketIdForEntity(unsigned char entityGroupId, dcgm_field_eid_t entityId)
{
    ASSERT_IS_SYSMON_THREAD;

    DcgmNs::Cpu::CpuId cpuId { entityId };
    if (entityGroupId == DCGM_FE_CPU_CORE)
    {
        cpuId = m_cpus.GetCpuIdForCore(DcgmNs::Cpu::CoreId { entityId });
    }
    return m_cpuTopology.GetCpusSocketId(cpuId);
}

/*****************************************************************************/
void DcgmModuleSysmon::RecordMetrics(timelib64_t now, std::vector<dcgm_field_update_info_t> &toUpdate)
{
    ASSERT_IS_SYSMON_THREAD;

    DcgmFvBuffer buf;
    dcgmReturn_t ret = DCGM_ST_OK;

    if (toUpdate.empty())
    {
        log_debug("Nothing to record");
        return;
    }

    for (auto const &fieldToUpdate : toUpdate)
    {
        switch (fieldToUpdate.entityGroupId)
        {
            case DCGM_FE_CPU_CORE:
                if (!m_cpus.IsValidCoreId(CoreId { fieldToUpdate.entityId }))
                {
                    log_error("Invalid core {}", fieldToUpdate.entityId);
                    continue;
                }
                break;
            case DCGM_FE_CPU:
                if (!m_cpus.IsValidCpuId(CpuId { fieldToUpdate.entityId }))
                {
                    log_error("Invalid CPU {}", fieldToUpdate.entityId);
                    continue;
                }
                break;
            default:
                log_error("Unexpected entityGroupId {}", fieldToUpdate.entityGroupId);
                continue;
        }

        switch (fieldToUpdate.fieldMeta->fieldId)
        {
            // Utilizaiton moved to UpdateFields.
            case DCGM_FI_DEV_CPU_UTIL_TOTAL:
            case DCGM_FI_DEV_CPU_UTIL_USER:
            case DCGM_FI_DEV_CPU_UTIL_NICE:
            case DCGM_FI_DEV_CPU_UTIL_SYS:
            case DCGM_FI_DEV_CPU_UTIL_IRQ:
                break;
            case DCGM_FI_DEV_CPU_TEMP_CURRENT:
            case DCGM_FI_DEV_CPU_TEMP_WARNING:
            case DCGM_FI_DEV_CPU_TEMP_CRITICAL:
            {
                unsigned int socketId = GetSocketIdForEntity(fieldToUpdate.entityGroupId, fieldToUpdate.entityId);
                double temperature    = ReadTemperature(socketId, fieldToUpdate.fieldMeta->fieldId);

                if (temperature == 0.0)
                {
                    buf.AddDoubleValue(fieldToUpdate.entityGroupId,
                                       fieldToUpdate.entityId,
                                       fieldToUpdate.fieldMeta->fieldId,
                                       DCGM_FP64_BLANK,
                                       now,
                                       DCGM_ST_OK);
                }
                else
                {
                    buf.AddDoubleValue(fieldToUpdate.entityGroupId,
                                       fieldToUpdate.entityId,
                                       fieldToUpdate.fieldMeta->fieldId,
                                       temperature,
                                       now,
                                       DCGM_ST_OK);
                }

                break;
            }
            case DCGM_FI_DEV_CPU_POWER_UTIL_CURRENT:
            {
                double usage          = 0.0;
                unsigned int socketId = GetSocketIdForEntity(fieldToUpdate.entityGroupId, fieldToUpdate.entityId);

                dcgmReturn_t ret = m_sysmon.GetCurrentCPUPowerUsage(socketId, usage);
                buf.AddDoubleValue(fieldToUpdate.entityGroupId,
                                   fieldToUpdate.entityId,
                                   fieldToUpdate.fieldMeta->fieldId,
                                   usage,
                                   now,
                                   ret);
                break;
            }
            case DCGM_FI_DEV_SYSIO_POWER_UTIL_CURRENT:
            {
                double usage          = 0.0;
                unsigned int socketId = GetSocketIdForEntity(fieldToUpdate.entityGroupId, fieldToUpdate.entityId);

                dcgmReturn_t ret = m_sysmon.GetCurrentSysIOPowerUsage(socketId, usage);
                buf.AddDoubleValue(fieldToUpdate.entityGroupId,
                                   fieldToUpdate.entityId,
                                   fieldToUpdate.fieldMeta->fieldId,
                                   usage,
                                   now,
                                   ret);
                break;
            }
            case DCGM_FI_DEV_MODULE_POWER_UTIL_CURRENT:
            {
                double usage          = 0.0;
                unsigned int socketId = GetSocketIdForEntity(fieldToUpdate.entityGroupId, fieldToUpdate.entityId);

                dcgmReturn_t ret = m_sysmon.GetCurrentModulePowerUsage(socketId, usage);
                buf.AddDoubleValue(fieldToUpdate.entityGroupId,
                                   fieldToUpdate.entityId,
                                   fieldToUpdate.fieldMeta->fieldId,
                                   usage,
                                   now,
                                   ret);
                break;
            }

            case DCGM_FI_DEV_CPU_POWER_LIMIT:
            {
                double cap            = 0.0;
                unsigned int socketId = GetSocketIdForEntity(fieldToUpdate.entityGroupId, fieldToUpdate.entityId);

                dcgmReturn_t ret = m_sysmon.GetCurrentPowerCap(socketId, cap);
                buf.AddDoubleValue(fieldToUpdate.entityGroupId,
                                   fieldToUpdate.entityId,
                                   fieldToUpdate.fieldMeta->fieldId,
                                   cap,
                                   now,
                                   ret);

                break;
            }

            case DCGM_FI_DEV_CPU_CLOCK_CURRENT:
            {
                uint64_t value = ReadCoreSpeed(fieldToUpdate.entityGroupId, fieldToUpdate.entityId);
                buf.AddInt64Value(fieldToUpdate.entityGroupId,
                                  fieldToUpdate.entityId,
                                  fieldToUpdate.fieldMeta->fieldId,
                                  value,
                                  now,
                                  DCGM_ST_OK);
                break;
            }

            case DCGM_FI_DEV_CPU_VENDOR:
            {
                std::string cpuVendor = m_sysmon.GetCpuVendor();
                buf.AddStringValue(fieldToUpdate.entityGroupId,
                                   fieldToUpdate.entityId,
                                   fieldToUpdate.fieldMeta->fieldId,
                                   cpuVendor.c_str(),
                                   now,
                                   DCGM_ST_OK);
                break;
            }

            case DCGM_FI_DEV_CPU_MODEL:
            {
                std::string cpuModel = m_sysmon.GetCpuModel();
                buf.AddStringValue(fieldToUpdate.entityGroupId,
                                   fieldToUpdate.entityId,
                                   fieldToUpdate.fieldMeta->fieldId,
                                   cpuModel.c_str(),
                                   now,
                                   DCGM_ST_OK);
                break;
            }

            default:
            {
                log_error("Invalid field for CPU {}", fieldToUpdate.fieldMeta->fieldId);
                break;
            }
        }
    }

    ret = m_coreProxy.AppendSamples(&buf);
    if (ret != DCGM_ST_OK)
    {
        log_warning("Failed to append samples to the cache: {}", errorString(ret));
    }
}

/*****************************************************************************/
std::chrono::milliseconds DcgmModuleSysmon::ProcessRunOnce()
{
    ASSERT_IS_SYSMON_THREAD;

    timelib64_t nextUpdateTimeUsec = 0;
    UpdateFields(nextUpdateTimeUsec);

    DCGM_LOG_VERBOSE << "Next update " << nextUpdateTimeUsec;

    using namespace std::chrono;
    return duration_cast<milliseconds>(microseconds(nextUpdateTimeUsec));
}

/*****************************************************************************/
std::chrono::system_clock::time_point DcgmModuleSysmon::ProcessTryRunOnce(bool forceRun)
{
    ASSERT_IS_SYSMON_THREAD;

    using namespace std::chrono;
    system_clock::time_point now = system_clock::now();
    if (forceRun || (now > m_nextWakeup))
    {
        log_verbose("Running at {}", now.time_since_epoch().count());
        milliseconds sleepDuration = ProcessRunOnce();
        // sleepDuration / nextUpdateTime is not playing nice with the
        // std::chrono constructs. Disabling for now and running every
        // m_runInterval, which is derived from the subscribed watches
        //
        // TODO(DCGM-XXXX)
        //m_nextWakeup = std::chrono::system_clock::now() + sleepDuration;

        // Updated by UpdateFields/RecordMetrics
        SetRunInterval(m_runInterval);
    }
    return m_nextWakeup;
}

/*****************************************************************************/
void DcgmModuleSysmon::run()
{
    using DcgmNs::TaskRunner;
    m_sysmonThreadId = std::this_thread::get_id();
    ASSERT_IS_SYSMON_THREAD;

    ProcessTryRunOnce(true);
    while (ShouldStop() == 0)
    {
        if (TaskRunner::Run(true) != TaskRunner::RunResult::Ok)
        {
            break;
        }

        ProcessTryRunOnce(false);
    }
}

dcgmReturn_t DcgmModuleSysmon::ProcessPauseResumeMessage(PauseResumeMessage msg)
{
    // Called functions will ASSERT_IS_SYSMON_THREAD to avoid accidental bypass if check were placed here
    return msg->pause ? Pause() : Resume();
}

dcgmReturn_t DcgmModuleSysmon::Pause()
{
    ASSERT_IS_SYSMON_THREAD;
    log_debug("Pausing");
    m_paused = true;
    return DCGM_ST_OK;
}

dcgmReturn_t DcgmModuleSysmon::Resume()
{
    ASSERT_IS_SYSMON_THREAD;
    log_debug("Resuming");
    m_paused = false;
    return DCGM_ST_OK;
}

/*****************************************************************************/
/* Enqueues a request to determine whether there is a subscription for the specified entityPair and fieldId. */
dcgmReturn_t DcgmModuleSysmon::GetSubscribed(dcgmGroupEntityPair_t entityPair,
                                             unsigned short fieldId,
                                             bool &isSubscribed)
{
    auto task = Enqueue(make_task("GetSubscribed request in TaskRunner", [this, entityPair, fieldId]() {
        ASSERT_IS_SYSMON_THREAD;
        return m_watchTable.GetIsSubscribed(entityPair.entityGroupId, entityPair.entityId, fieldId);
    }));

    if (!task.has_value())
    {
        DCGM_LOG_ERROR << "Unable to enqueue Sysmon ProcessGetSubscribed task";
        return DCGM_ST_GENERIC_ERROR;
    }

    bool result  = (*task).get();
    isSubscribed = result;
    return DCGM_ST_OK;
}

/* Enqueues a request to run the main processing loop. Intended for testing. */
dcgmReturn_t DcgmModuleSysmon::RunOnce(bool force)
{
    auto task = Enqueue(DcgmNs::make_task("TryRunOnce", [this, force]() {
        ASSERT_IS_SYSMON_THREAD;
        ProcessTryRunOnce(force);
    }));

    if (!task.has_value())
    {
        DCGM_LOG_ERROR << "Unable to enqueue Sysmon TryRunOnce task";
        return DCGM_ST_GENERIC_ERROR;
    }

    (*task).get();
    return DCGM_ST_OK;
}

/* Enqueues a request to prune samples treating the specified time as current. Intended for testing. */
dcgmReturn_t DcgmModuleSysmon::PruneSamples(DcgmNs::Timelib::TimePoint now)
{
    auto task = Enqueue(DcgmNs::make_task("Prune Utilization Samples", [this, now]() {
        ASSERT_IS_SYSMON_THREAD;
        ProcessPruneSamples(now);
    }));

    if (!task.has_value())
    {
        DCGM_LOG_ERROR << "Unable to enqueue Sysmon PruneSamples task";
        return DCGM_ST_GENERIC_ERROR;
    }

    (*task).get();
    return DCGM_ST_OK;
}

/* Enqueues a request to determine the current number of utilization samples. Intended for testing. */
dcgmReturn_t DcgmModuleSysmon::GetUtilizationSampleSize(size_t &numSamples)
{
    auto task = Enqueue(DcgmNs::make_task("GetUtilizationSampleSize", [this]() {
        ASSERT_IS_SYSMON_THREAD;
        return m_utilizationSamples.size();
    }));

    if (!task.has_value())
    {
        DCGM_LOG_ERROR << "Unable to enqueue Sysmon GetUtilizationSampleSize task";
        return DCGM_ST_GENERIC_ERROR;
    }
    numSamples = (*task).get();
    return DCGM_ST_OK;
}

/* Enqueues a request to add a utilization sample. Intended for testing. */
dcgmReturn_t DcgmModuleSysmon::AddUtilizationSample(SysmonUtilizationSample &item)
{
    auto task = Enqueue(DcgmNs::make_task("AddUtilizationSample", [this, &item]() {
        ASSERT_IS_SYSMON_THREAD;
        m_utilizationSamples.emplace(item.m_timestamp, item);
    }));

    if (!task.has_value())
    {
        DCGM_LOG_ERROR << "Unable to enqueue Sysmon AddUtilizationSample task";
        return DCGM_ST_GENERIC_ERROR;
    }

    (*task).get();
    return DCGM_ST_OK;
}

extern "C" {
/*************************************************************************/
DCGM_PUBLIC_API DcgmModule *dcgm_alloc_module_instance(dcgmCoreCallbacks_t *dcc)
{
    return SafeWrapper([=] { return new DcgmModuleSysmon(*dcc); });
}

/*************************************************************************/
DCGM_PUBLIC_API void dcgm_free_module_instance(DcgmModule *freeMe)
{
    delete (freeMe);
}

DCGM_PUBLIC_API dcgmReturn_t dcgm_module_process_message(DcgmModule *module,
                                                         dcgm_module_command_header_t *moduleCommand)
{
    return PassMessageToModule(module, moduleCommand);
}
} // extern "C"

} // namespace DcgmNs
