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
#ifndef _DCGM_HEALTH_WATCH_H
#define _DCGM_HEALTH_WATCH_H

#include "DcgmCoreProxy.h"
#include "DcgmError.h"
#include "DcgmGPUHardwareLimits.h"
#include "DcgmHealthResponse.h"
#include "dcgm_core_communication.h"
#include "dcgm_test_apis.h"
#include <unordered_set>

// Number of nvlink error counter types
#define DCGM_HEALTH_WATCH_NVLINK_ERROR_NUM_FIELDS NVML_NVLINK_ERROR_COUNT

/* This class is implements the background health check methods
 * within the hostengine
 * It is intended to set watches, monitor them on demand, and
 * inform a user of any specific problems for watches that have
 * been requested.
 */
class DcgmHealthWatch
{
public:
    DcgmHealthWatch(dcgmCoreCallbacks_t &pDcc);
    ~DcgmHealthWatch();

    /*
     * This method is used to set the watches based on which bits are enabled in the
     * systems variable
     */
    dcgmReturn_t SetWatches(unsigned int groupId,
                            dcgmHealthSystems_t systems,
                            dcgm_connection_id_t connectionId,
                            long long updateInterval,
                            double maxKeepAge);

    /*
     * This method is used to get the watches
     */
    dcgmReturn_t GetWatches(unsigned int groupId, dcgmHealthSystems_t *systems);

    /*
     * This method is used to check an individual gpu's health watches
     */
    dcgmReturn_t MonitorWatchesForGpu(unsigned int gpuId,
                                      long long startTime,
                                      long long endTime,
                                      dcgmHealthSystems_t healthResponse,
                                      DcgmHealthResponse &response);

    /*
     * This method is used to trigger a monitoring of the configured watches for a group
     */
    dcgmReturn_t MonitorWatches(unsigned int groupId,
                                long long startTime,
                                long long endTime,
                                DcgmHealthResponse &response);

    /*
     Notify this module that a group was removed from the group manager
     */
    void OnGroupRemove(unsigned int groupId);

    /*
    Notify this module that a watched field value updated
    */
    void OnFieldValuesUpdate(DcgmFvBuffer *fvBuffer);

    /*
    Process a buffered fv for an XID. Called by OnFieldValuesUpdate()
    */
    void ProcessXidFv(dcgmBufferedFv_t *fv);

private:
    DcgmCoreProxy mpCoreProxy;
    /* Map of groupId -> dcgmHealthSystems_t of the watched health systems of a given groupId */
    typedef std::map<unsigned int, dcgmHealthSystems_t> groupWatchTable_t;
    groupWatchTable_t mGroupWatchState;

    DcgmMutex *m_mutex;

    std::unordered_set<dcgm_field_eid_t>
        m_gpuHadUncontainedErrorXid; /* If a GPU has had an XID 95, its value is set here.
                                       This data structure is protected by m_mutex. */

    /* Prepopulated lists of fields used by various internal methods */
    std::vector<unsigned int> m_nvSwitchNonFatalFieldIds; /* NvSwitch non-fatal errors */
    std::vector<unsigned int> m_nvSwitchFatalFieldIds;    /* NvSwitch fatal errors */

    // miscellaneous helper methods
    std::string MemFieldToString(unsigned short fieldId);

    /* Build internal lists of fieldIds to be used by other methods */
    void BuildFieldLists(void);

    void SetResponse(dcgm_field_entity_group_t entityGroupId,
                     dcgm_field_eid_t entityId,
                     dcgmHealthWatchResults_t status,
                     dcgmHealthSystems_t system,
                     DcgmError &d,
                     DcgmHealthResponse &response);

    // methods to handle setting watches
    dcgmReturn_t SetPcie(dcgm_field_entity_group_t entityGroupId,
                         unsigned int entityId,
                         bool enable,
                         DcgmWatcher watcher,
                         long long updateInterval,
                         double maxKeepAge);
    dcgmReturn_t SetMem(dcgm_field_entity_group_t entityGroupId,
                        unsigned int entityId,
                        bool enable,
                        DcgmWatcher watcher,
                        long long updateInterval,
                        double maxKeepAge);
    dcgmReturn_t SetInforom(dcgm_field_entity_group_t entityGroupId,
                            unsigned int entityId,
                            bool enable,
                            DcgmWatcher watcher,
                            long long updateInterval,
                            double maxKeepAge);
    dcgmReturn_t SetThermal(dcgm_field_entity_group_t entityGroupId,
                            unsigned int entityId,
                            bool enable,
                            DcgmWatcher watcher,
                            long long updateInterval,
                            double maxKeepAge);
    dcgmReturn_t SetPower(dcgm_field_entity_group_t entityGroupId,
                          unsigned int entityId,
                          bool enable,
                          DcgmWatcher watcher,
                          long long updateInterval,
                          double maxKeepAge);
    dcgmReturn_t SetNVLink(dcgm_field_entity_group_t entityGroupId,
                           unsigned int entityId,
                           bool enable,
                           DcgmWatcher watcher,
                           long long updateInterval,
                           double maxKeepAge);
    dcgmReturn_t SetNvSwitchWatches(std::vector<unsigned int> &groupSwitchIds,
                                    dcgmHealthSystems_t systems,
                                    DcgmWatcher watcher,
                                    long long updateInterval,
                                    double maxKeepAge);

    dcgmReturn_t MonitorPcie(dcgm_field_entity_group_t entityGroupId,
                             dcgm_field_eid_t entityId,
                             long long startTime,
                             long long endTime,
                             DcgmHealthResponse &response);
    dcgmReturn_t MonitorMem(dcgm_field_entity_group_t entityGroupId,
                            dcgm_field_eid_t entityId,
                            long long startTime,
                            long long endTime,
                            DcgmHealthResponse &response);
    dcgmReturn_t MonitorInforom(dcgm_field_entity_group_t entityGroupId,
                                dcgm_field_eid_t entityId,
                                long long startTime,
                                long long endTime,
                                DcgmHealthResponse &response);
    dcgmReturn_t MonitorThermal(dcgm_field_entity_group_t entityGroupId,
                                dcgm_field_eid_t entityId,
                                long long startTime,
                                long long endTime,
                                DcgmHealthResponse &response);
    dcgmReturn_t MonitorPower(dcgm_field_entity_group_t entityGroupId,
                              dcgm_field_eid_t entityId,
                              long long startTime,
                              long long endTime,
                              DcgmHealthResponse &response);
    dcgmReturn_t MonitorNVLink(dcgm_field_entity_group_t entityGroupId,
                               dcgm_field_eid_t entityId,
                               long long startTime,
                               long long endTime,
                               DcgmHealthResponse &response);
    dcgmReturn_t MonitorNvSwitchErrorCounts(bool fatal,
                                            dcgm_field_entity_group_t entityGroupId,
                                            dcgm_field_eid_t entityId,
                                            long long startTime,
                                            long long endTime,
                                            DcgmHealthResponse &response);

    /* Helpers called by MonitorMem() */
    dcgmReturn_t MonitorMemVolatileDbes(dcgm_field_entity_group_t entityGroupId,
                                        dcgm_field_eid_t entityId,
                                        long long startTime,
                                        long long endTime,
                                        DcgmHealthResponse &response);
    dcgmReturn_t MonitorMemRetiredPending(dcgm_field_entity_group_t entityGroupId,
                                          dcgm_field_eid_t entityId,
                                          long long startTime,
                                          long long endTime,
                                          DcgmHealthResponse &response);
    dcgmReturn_t MonitorMemSbeDbeRetiredPages(dcgm_field_entity_group_t entityGroupId,
                                              dcgm_field_eid_t entityId,
                                              long long startTime,
                                              long long endTime,
                                              DcgmHealthResponse &response);
    dcgmReturn_t MonitorMemRowRemapFailures(dcgm_field_entity_group_t entityGroupId,
                                            dcgm_field_eid_t entityId,
                                            long long startTime,
                                            long long endTime,
                                            DcgmHealthResponse &response);
    dcgmReturn_t MonitorUncontainedErrors(dcgm_field_entity_group_t entityGroupId,
                                          dcgm_field_eid_t entityId,
                                          long long startTime,
                                          long long endTime,
                                          DcgmHealthResponse &response);

    bool FitsGpuHardwareCheck(dcgm_field_entity_group_t entityGroupId);
};

#endif //_DCGM_HEALTH_WATCH_H
