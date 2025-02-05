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
#pragma once

#include "DcgmCpuManager.h"
#include "DcgmCpuTopology.h"
#include "DcgmSystemMonitor.h"
#include "MessageGuard.hpp"
#include "dcgm_sysmon_structs.h"

#include <DcgmModule.h>
#include <DcgmTaskRunner.h>
#include <TimeLib.hpp>
#include <dcgm_core_structs.h>

#include <fstream>
#include <vector>

#define __DCGM_SYSMON_SKIP_HARDWARE_CHECK__ "DCGM_SKIP_SYSMON_HARDWARE_CHECK"

namespace DcgmNs
{

typedef enum
{
    SYSMON_MONITORING_SWITCH_UTILIZATION = 0,
    SYSMON_MONITORING_SWITCH_COUNT,
} sysmonMonitoringSwitch_t;

class SysmonUtilizationSampleCore
{
public:
    unsigned long long GetTotal() const;
    unsigned long long GetActive() const;

    unsigned long long m_user;
    unsigned long long m_nice;
    unsigned long long m_system;
    unsigned long long m_idle;
    unsigned long long m_irq;
    unsigned long long m_other;
};

class SysmonUtilizationSample
{
public:
    Timelib::TimePoint m_timestamp;
    std::vector<SysmonUtilizationSampleCore> m_cores;
};

typedef std::map<Timelib::TimePoint, SysmonUtilizationSample> sysmonUtilSampleMap_t;

class DcgmModuleSysmon

    : public DcgmModuleWithCoreProxy<DcgmModuleIdSysmon>
    , DcgmTaskRunner

{
public:
    explicit DcgmModuleSysmon(dcgmCoreCallbacks_t &dcc);

    ~DcgmModuleSysmon();
    /*************************************************************************/
    /**
     * Process a DCGM module message that was sent to this module
     * (inherited from DcgmModule)
     */
    dcgmReturn_t ProcessMessage(dcgm_module_command_header_t *moduleCommand) override;

    /*************************************************************************/
    /*
     * Process a DCGM module message from our taskrunner thread.
     */
    dcgmReturn_t ProcessMessageFromTaskRunner(dcgm_module_command_header_t *moduleCommand);

    /*************************************************************************/
    /*
     * This is the main background worker function of the module.
     *
     * Returns: Minimum ms before we should call this function again. This will
     *          be how long we block on QueueTask() being called again.
     *          Returning 0 = Don't care when we get called back.
     */
    std::chrono::milliseconds ProcessRunOnce();

    /*
     * Populates the specified CPU according to the range string specified
     *
     * @param cpu - the CPU whose bitmask we're populating
     * @param rangeStr - a string representing ranges in the format \d[,-\d][,\d[-\d]]...
     */
    static dcgmReturn_t PopulateOwnedCoresBitmaskFromRangeString(dcgm_sysmon_cpu_t &cpu, const std::string &rangeStr);
    static void PopulateOwnedCoresBitmask(dcgm_sysmon_cpu_t &cpu);

#ifndef DCGM_SYSMON_TEST // Allow sysmon tests to peek in
private:
#endif

    using GetCpusMessage = DcgmNs::MessageGuard<dcgm_sysmon_msg_get_cpus_t, dcgm_sysmon_msg_get_cpus_version>;
    using WatchFieldsMessage
        = DcgmNs::MessageGuard<dcgm_sysmon_msg_watch_fields_t, dcgm_sysmon_msg_watch_fields_version>;
    using UnwatchFieldsMessage
        = DcgmNs::MessageGuard<dcgm_sysmon_msg_unwatch_fields_t, dcgm_sysmon_msg_unwatch_fields_version>;
    using GetEntityStatusMessage
        = DcgmNs::MessageGuard<dcgm_sysmon_msg_get_entity_status_t, dcgm_sysmon_msg_get_entity_status_version>;
    using CreateFakeEntitiesMessage
        = DcgmNs::MessageGuard<dcgm_sysmon_msg_create_fake_entities_t, dcgm_sysmon_msg_create_fake_entities_version>;
    using PauseResumeMessage = DcgmNs::MessageGuard<dcgm_core_msg_pause_resume_v1, dcgm_core_msg_pause_resume_version1>;
    std::chrono::system_clock::time_point m_nextWakeup
        = std::chrono::system_clock::time_point::min(); /*!< Next time when RunOnce should be called. */
    std::chrono::milliseconds m_runInterval {}; /*!< Last result of the latest successful RunOnce function call made in
                                                     the TryRunOnce method.  */
    DcgmCpuManager m_cpus;
    DcgmCpuTopology m_cpuTopology;
    std::ifstream m_procStat;
    sysmonUtilSampleMap_t m_utilizationSamples;
    DcgmWatchTable m_watchTable; /* Table of watchers */
    DcgmSystemMonitor m_sysmon;
    std::string m_coreSpeedBaseDir;
    std::atomic_bool m_paused;
    DcgmNs::Timelib::TimePoint m_maxSampleAge;
    std::thread::id m_sysmonThreadId;

    bool m_enabledMonitoring[SYSMON_MONITORING_SWITCH_COUNT];
    std::vector<double> m_cpuUtilization;
    std::string m_tzBaseDir;
    std::unordered_map<unsigned int, std::string> m_socketTemperatureFileMap;
    std::unordered_map<unsigned int, std::string> m_socketTemperatureWarnFileMap;
    std::unordered_map<unsigned int, std::string> m_socketTemperatureCritFileMap;

    /*************************************************************************/
    dcgmReturn_t ProcessGetCpus(GetCpusMessage msg);
    dcgmReturn_t ProcessGetEntityStatus(GetEntityStatusMessage msg);
    dcgmReturn_t ProcessWatchFields(WatchFieldsMessage msg);
    dcgmReturn_t ProcessUnwatchFields(UnwatchFieldsMessage msg);
    dcgmReturn_t ProcessCreateFakeEntities(CreateFakeEntitiesMessage msg);
    dcgmReturn_t ProcessPauseResumeMessage(PauseResumeMessage msg);
    dcgmReturn_t ProcessClientDisconnect(dcgm_core_msg_client_disconnect_t *msg);
    dcgmReturn_t ProcessCoreMessage(dcgm_module_command_header_t *moduleCommand);
    std::chrono::system_clock::time_point ProcessTryRunOnce(bool forceRun);
    const SysmonUtilizationSample &ReadUtilizationSample(DcgmNs::Timelib::TimePoint now);
    dcgmReturn_t UpdateField(DcgmNs::Timelib::TimePoint now, const dcgm_field_update_info_t &updateInfo);
    void UpdateFields(timelib64_t &nextUpdateTimeUsec);
    double CalculateCoreUtilization(DcgmNs::Cpu::CoreId core,
                                    unsigned int fieldId,
                                    const SysmonUtilizationSample &baselineSample,
                                    const SysmonUtilizationSample &currentSample);
    double CalculateCpuUtilization(DcgmNs::Cpu::CpuId cpu,
                                   unsigned int fieldId,
                                   const SysmonUtilizationSample &baselineSample,
                                   const SysmonUtilizationSample &currentSample);
    double CalculateUtilizationForEntity(unsigned int entityGroupId,
                                         unsigned int entityId,
                                         unsigned int fieldId,
                                         const SysmonUtilizationSample &baselineSample,
                                         const SysmonUtilizationSample &currentSample);
    unsigned int GetSocketFromThermalZoneFileContents(const std::string &path, const std::string &contents);
    unsigned int GetSocketIdFromThermalFile(const std::string &path);
    void PopulateTemperatureFileMap();
    double ReadTemperature(unsigned int cpuId, short fieldId);
    void RecordMetrics(timelib64_t now, std::vector<dcgm_field_update_info_t> &toUpdate);
    uint64_t ReadCoreSpeed(unsigned int entityGroupId, unsigned int entityId);
    dcgmReturn_t PopulateCpusIfNeeded();
    // Probably going to be replaced by a mechanism that relies on the watch table
    dcgmReturn_t EnableMonitoring(unsigned int monitoringSwitch);
    void ProcessPruneSamples(DcgmNs::Timelib::TimePoint now);
    dcgmReturn_t ParseProcStatCpuLine(const std::string &line, SysmonUtilizationSample &sample);

    /*
     * Returns the socket that the specified entity belongs to
     */
    unsigned int GetSocketIdForEntity(unsigned char entityGroupId, dcgm_field_eid_t entityId);
    dcgmReturn_t Pause();
    dcgmReturn_t Resume();

    dcgmReturn_t ProcessGetSubscribed(dcgmGroupEntityPair_t entityPair, unsigned short fieldId, bool &isSubscribed);
    dcgmReturn_t GetSubscribed(dcgmGroupEntityPair_t entityPair, unsigned short fieldId, bool &isSubscribed);
    dcgmReturn_t RunOnce(bool force = false);
    dcgmReturn_t PruneSamples(DcgmNs::Timelib::TimePoint now);
    dcgmReturn_t GetUtilizationSampleSize(size_t &numSamples);
    dcgmReturn_t AddUtilizationSample(SysmonUtilizationSample &item);

    void run() override;
};

} // namespace DcgmNs
