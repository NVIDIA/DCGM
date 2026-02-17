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
/*
 * File:   DcgmHostEngineHandler.cpp
 */

#include "DcgmHostEngineHandler.h"

#include "DcgmCoreCommunication.h"
#include "DcgmGroupManager.h"

#include <DcgmLogging.h>
#include <DcgmMetadataMgr.h>
#include <DcgmModule.h>
#include <DcgmModuleHealth.h>
#include <DcgmModuleIntrospect.h>
#include <DcgmModulePolicy.h>
#include <DcgmSettings.h>
#include <DcgmStatus.h>
#include <Defer.hpp>
#include <TaskContextManager.hpp>
#include <dcgm_health_structs.h>
#include <dcgm_helpers.h>
#include <dcgm_nvswitch_structs.h>
#include <dcgm_profiling_structs.h>
#include <dcgm_structs.h>
#include <dcgm_structs_internal.h>
#include <dcgm_sysmon_structs.h>
#include <dcgm_util.h>

#include <dcgm_nvml.h>
#include <nvcmvalue.h>

#include <algorithm>
#include <dlfcn.h> //dlopen, dlsym..etc
#include <iostream>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>


#ifdef INJECTION_LIBRARY_AVAILABLE
#include <nvml_injection.h>
#include <ranges>
#endif

namespace
{
auto const DCGM_HANGDETECT_DISABLE    = "DCGM_HANGDETECT_DISABLE";
auto const DCGM_HANGDETECT_TERMINATE  = "DCGM_HANGDETECT_TERMINATE";
auto const DCGM_HANGDETECT_EXPIRY_SEC = "DCGM_HANGDETECT_EXPIRY_SEC";
} // namespace

DcgmHostEngineHandler *DcgmHostEngineHandler::mpHostEngineHandlerInstance = nullptr;
DcgmModuleCore DcgmHostEngineHandler::mModuleCoreObj;

/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::TranslateBitmapToGpuVector(uint64_t gpuBitmap, std::vector<unsigned int> &gpuIds)

{
    dcgmReturn_t ret = DCGM_ST_OK;

    if (gpuBitmap == 0)
    {
        unsigned int gId = DCGM_GROUP_ALL_GPUS;
        ret              = mpGroupManager->verifyAndUpdateGroupId(&gId);

        if (ret == DCGM_ST_OK)
        {
            ret = mpGroupManager->GetGroupGpuIds(0, gId, DcgmGroupOption::ActiveOnly, gpuIds);
        }
    }
    else
    {
        unsigned int gpuId = 0;
        for (uint64_t i = 0x1; gpuBitmap != 0; i <<= 1, gpuId++)
        {
            if ((gpuBitmap & i) != 0)
            {
                // Bit is set, record this gpu
                gpuIds.push_back(gpuId);
            }

            // Clear that bit
            gpuBitmap &= ~i;
        }
    }

    return ret;
}

/*****************************************************************************/
void DcgmHostEngineHandler::RemoveUnhealthyGpus(std::vector<unsigned int> &gpuIds)
{
    std::vector<unsigned int> healthyGpus;
    std::set<unsigned int> unhealthyGpus;
    dcgmReturn_t dcgmReturn;
    std::unique_ptr<dcgm_health_msg_check_gpus_t> msg = std::make_unique<dcgm_health_msg_check_gpus_t>();

    /* Prepare a health check RPC to the health module */
    memset(msg.get(), 0, sizeof(*msg));

    if (gpuIds.size() > DCGM_MAX_NUM_DEVICES)
    {
        log_error("Too many GPU ids: {}. Truncating.", (int)gpuIds.size());
    }

    msg->header.length     = sizeof(*msg);
    msg->header.moduleId   = DcgmModuleIdHealth;
    msg->header.subCommand = DCGM_HEALTH_SR_CHECK_GPUS;
    msg->header.version    = dcgm_health_msg_check_gpus_version;

    msg->systems          = DCGM_HEALTH_WATCH_ALL;
    msg->startTime        = 0;
    msg->endTime          = 0;
    msg->response.version = dcgmHealthResponse_version5;
    msg->numGpuIds        = std::min(gpuIds.size(), (size_t)DCGM_MAX_NUM_DEVICES);


    for (size_t i = 0; i < msg->numGpuIds; i++)
    {
        msg->gpuIds[i] = gpuIds[i];
    }

    dcgmReturn = ProcessModuleCommand(&msg->header);
    if (dcgmReturn == DCGM_ST_MODULE_NOT_LOADED)
    {
        log_debug("RemoveUnhealthyGpus not filtering due to health module not being loaded.");
        return;
    }
    if (dcgmReturn != DCGM_ST_OK)
    {
        log_error("ProcessModuleCommand failed with {}", dcgmReturn);
        return;
    }

    for (unsigned int i = 0; i < msg->response.incidentCount; i++)
    {
        if (msg->response.incidents[i].entityInfo.entityGroupId == DCGM_FE_GPU
            && msg->response.incidents[i].health == DCGM_HEALTH_RESULT_FAIL)
        {
            unhealthyGpus.insert(msg->response.incidents[i].entityInfo.entityId);
        }
    }

    // If there are no unhealthy GPUs then we are done
    if (unhealthyGpus.empty())
    {
        return;
    }

    for (unsigned int gpuId : gpuIds)
    {
        if (unhealthyGpus.find(gpuId) == unhealthyGpus.end())
        {
            healthyGpus.push_back(gpuId);
        }
    }

    gpuIds.clear();
    gpuIds = std::move(healthyGpus);
}

/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::HelperSelectGpusByTopology(uint32_t numGpus,
                                                               uint64_t inputGpus,
                                                               uint64_t hints,
                                                               uint64_t &outputGpus)
{
    std::vector<unsigned int> gpuIds;

    if (!m_nvmlLoaded.load(std::memory_order_acquire))
    {
        log_debug("Cannot select GPUs by topology: NVML is not loaded");
        return DCGM_ST_NVML_NOT_LOADED;
    }

    dcgmReturn_t ret = TranslateBitmapToGpuVector(inputGpus, gpuIds);

    if (ret == DCGM_ST_OK)
    {
        if ((hints & DCGM_TOPO_HINT_F_IGNOREHEALTH) == 0)
        {
            RemoveUnhealthyGpus(gpuIds);
        }

        ret = mpCacheManager->SelectGpusByTopology(gpuIds, numGpus, outputGpus);
    }

    return ret;
}

/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::GetAllEntitiesOfEntityGroup(int activeOnly,
                                                                dcgm_field_entity_group_t entityGroupId,
                                                                std::vector<dcgmGroupEntityPair_t> &entities)
{
    switch (entityGroupId)
    {
        case DCGM_FE_SWITCH:
        {
            dcgm_nvswitch_msg_get_switches_t nvsMsg {};
            nvsMsg.header.length     = sizeof(nvsMsg);
            nvsMsg.header.version    = dcgm_nvswitch_msg_get_switches_version;
            nvsMsg.header.moduleId   = DcgmModuleIdNvSwitch;
            nvsMsg.header.subCommand = DCGM_NVSWITCH_SR_GET_SWITCH_IDS;

            dcgmReturn_t dcgmReturn = ProcessModuleCommand(&nvsMsg.header);
            if (dcgmReturn != DCGM_ST_OK)
            {
                DCGM_LOG_ERROR << "ProcessModuleCommand of DCGM_NVSWITCH_SR_GET_SWITCH_IDS returned "
                               << errorString(dcgmReturn);
                return dcgmReturn;
            }

            dcgmGroupEntityPair_t entityPair;
            entityPair.entityGroupId = DCGM_FE_SWITCH;

            for (unsigned int i = 0; i < nvsMsg.switchCount; i++)
            {
                entityPair.entityId = nvsMsg.switchIds[i];
                entities.push_back(entityPair);
            }
            return DCGM_ST_OK;
        }
        case DCGM_FE_LINK:
        {
            dcgm_nvswitch_msg_get_links_t nvsMsg {};
            nvsMsg.header.length     = sizeof(nvsMsg);
            nvsMsg.header.version    = dcgm_nvswitch_msg_get_links_version;
            nvsMsg.header.moduleId   = DcgmModuleIdNvSwitch;
            nvsMsg.header.subCommand = DCGM_NVSWITCH_SR_GET_LINK_IDS;

            dcgmReturn_t dcgmReturn = ProcessModuleCommand(&nvsMsg.header);
            if (dcgmReturn != DCGM_ST_OK)
            {
                DCGM_LOG_ERROR << "ProcessModuleCommand of DCGM_NVSWITCH_SR_GET_LINK_IDS returned "
                               << errorString(dcgmReturn);
                return dcgmReturn;
            }

            dcgmGroupEntityPair_t entityPair;
            entityPair.entityGroupId = DCGM_FE_LINK;

            for (unsigned int i = 0; i < nvsMsg.linkCount; i++)
            {
                entityPair.entityId = nvsMsg.linkIds[i];
                entities.push_back(entityPair);
            }
            return DCGM_ST_OK;
        }
        case DCGM_FE_CONNECTX:
        {
            dcgm_nvswitch_msg_get_entities_ids_v1 nvsMsg {};
            nvsMsg.header.length     = sizeof(nvsMsg);
            nvsMsg.header.version    = dcgm_nvswitch_msg_get_entities_ids_version;
            nvsMsg.header.moduleId   = DcgmModuleIdNvSwitch;
            nvsMsg.header.subCommand = DCGM_NVSWITCH_SR_GET_ENTITIES_IDS;
            nvsMsg.entityGroup       = entityGroupId;

            dcgmReturn_t dcgmReturn = ProcessModuleCommand(&nvsMsg.header);
            if (dcgmReturn != DCGM_ST_OK)
            {
                log_error("ProcessModuleCommand of DCGM_NVSWITCH_SR_GET_ENTITIES_IDS returned: [{}]",
                          errorString(dcgmReturn));
                return dcgmReturn;
            }

            dcgmGroupEntityPair_t entityPair;
            entityPair.entityGroupId = entityGroupId;

            for (unsigned int i = 0;
                 i < std::min(static_cast<size_t>(nvsMsg.entitiesCount), std::size(nvsMsg.entities));
                 i++)
            {
                entityPair.entityId = nvsMsg.entities[i];
                entities.push_back(entityPair);
            }
            return DCGM_ST_OK;
        }
        case DCGM_FE_CPU:
        {
            dcgm_sysmon_msg_get_cpus_t sysmonMsg {};
            sysmonMsg.header.length     = sizeof(sysmonMsg);
            sysmonMsg.header.version    = dcgm_sysmon_msg_get_cpus_version;
            sysmonMsg.header.moduleId   = DcgmModuleIdSysmon;
            sysmonMsg.header.subCommand = DCGM_SYSMON_SR_GET_CPUS;

            dcgmReturn_t dcgmReturn = ProcessModuleCommand(&sysmonMsg.header);
            if (dcgmReturn != DCGM_ST_OK)
            {
                log_error("Received {}", errorString(dcgmReturn));
                return dcgmReturn;
            }

            dcgmGroupEntityPair_t entityPair;
            entityPair.entityGroupId = DCGM_FE_CPU;

            for (unsigned int cpu = 0; cpu < sysmonMsg.cpuCount; cpu++)
            {
                const auto &cpuObject = sysmonMsg.cpus[cpu];
                entityPair.entityId   = cpuObject.cpuId;
                entities.push_back(entityPair);
            }
            return DCGM_ST_OK;
        }
        case DCGM_FE_CPU_CORE:
        {
            dcgm_sysmon_msg_get_cpus_t sysmonMsg {};
            sysmonMsg.header.length     = sizeof(sysmonMsg);
            sysmonMsg.header.version    = dcgm_sysmon_msg_get_cpus_version;
            sysmonMsg.header.moduleId   = DcgmModuleIdSysmon;
            sysmonMsg.header.subCommand = DCGM_SYSMON_SR_GET_CPUS;

            dcgmReturn_t dcgmReturn = ProcessModuleCommand(&sysmonMsg.header);
            if (dcgmReturn != DCGM_ST_OK)
            {
                log_error("Received {}", errorString(dcgmReturn));
                return dcgmReturn;
            }

            dcgmGroupEntityPair_t entityPair;
            entityPair.entityGroupId = DCGM_FE_CPU_CORE;

            for (unsigned int cpu = 0; cpu < sysmonMsg.cpuCount; cpu++)
            {
                const auto &cpuObject = sysmonMsg.cpus[cpu];
                for (unsigned int core = 0; core < DCGM_MAX_NUM_CPU_CORES; core++)
                {
                    if (dcgmCpuHierarchyCpuOwnsCore(core, &cpuObject.ownedCores))
                    {
                        entityPair.entityId = core;
                        entities.push_back(entityPair);
                    }
                }
            }
            return DCGM_ST_OK;
        }
        default:

            dcgmReturn_t dcgmReturn = mpCacheManager->GetAllEntitiesOfEntityGroup(activeOnly, entityGroupId, entities);
            if (dcgmReturn != DCGM_ST_OK)
            {
                DCGM_LOG_ERROR << "GetAllEntitiesOfEntityGroup(ao " << activeOnly << ", eg " << entityGroupId
                               << ") returned " << dcgmReturn;
            }

            return dcgmReturn;
    }
}

/*****************************************************************************/
bool DcgmHostEngineHandler::GetIsValidEntityId(dcgm_field_entity_group_t entityGroupId, dcgm_field_eid_t entityId)
{
    switch (entityGroupId)
    {
        case DCGM_FE_NONE:
            return true;

        case DCGM_FE_VGPU:
        case DCGM_FE_GPU:
        case DCGM_FE_GPU_I:
        case DCGM_FE_GPU_CI:
        case DCGM_FE_CPU:
        case DCGM_FE_CPU_CORE:
            return mpCacheManager->GetIsValidEntityId(entityGroupId, entityId);

        case DCGM_FE_LINK:
        {
            dcgm_link_t link {};

            link.parsed.switchId = 0;
            link.raw             = entityId;

            switch (link.parsed.type)
            {
                case DCGM_FE_GPU:
                    return mpCacheManager->GetIsValidEntityId(entityGroupId, entityId);

                case DCGM_FE_SWITCH:
                    if (link.parsed.index >= DCGM_NVLINK_MAX_LINKS_PER_NVSWITCH)
                    {
                        return false;
                    }

                    entityGroupId = DCGM_FE_SWITCH;
                    entityId      = link.parsed.switchId;

                    break;

                default:
                    return false;
            }

            break;
        }
        case DCGM_FE_SWITCH:
        case DCGM_FE_CONNECTX:
            break; /* Handle below */

        case DCGM_FE_COUNT:
            return false;

            /* Purposely omitting a default in case we add new entity types */
    }

    std::vector<dcgmGroupEntityPair_t> entities;
    dcgmReturn_t dcgmReturn = GetAllEntitiesOfEntityGroup(0, entityGroupId, entities);
    if (dcgmReturn != DCGM_ST_OK)
    {
        DCGM_LOG_ERROR << "GetAllEntitiesOfEntityGroup failed with " << dcgmReturn << " for eg " << entityGroupId
                       << ", eid " << entityId;
        return false;
    }

    for (auto entity : entities)
    {
        if (entity.entityId == entityId && entity.entityGroupId == entityGroupId)
        {
            return true;
        }
    }

    return false;
}

/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::HelperWatchPredefined(dcgmWatchPredefined_t *watchPredef, DcgmWatcher &dcgmWatcher)
{
    dcgmReturn_t ret;
    dcgmFieldGrp_t fieldGroupId;
    unsigned int groupId;

    if (!watchPredef)
    {
        return DCGM_ST_BADPARAM;
    }

    if (watchPredef->version != dcgmWatchPredefined_version)
    {
        DCGM_LOG_ERROR << "Version mismatch";
        return DCGM_ST_VER_MISMATCH;
    }

    groupId = (unsigned int)(intptr_t)watchPredef->groupId;
    ret     = mpGroupManager->verifyAndUpdateGroupId(&groupId);
    if (ret != DCGM_ST_OK)
    {
        return ret;
    }

    switch (watchPredef->watchPredefType)
    {
        case DCGM_WATCH_PREDEF_PID: /* Intentional fall-through */
        case DCGM_WATCH_PREDEF_JOB:
            fieldGroupId = mFieldGroupPidAndJobStats;
            break;

        case DCGM_WATCH_PREDEF_INVALID:
        default:
            DCGM_LOG_ERROR << "Invalid watchPredefType " << watchPredef->watchPredefType;
            return DCGM_ST_BADPARAM;
    }

    ret = WatchFieldGroup(groupId,
                          fieldGroupId,
                          watchPredef->updateFreq,
                          watchPredef->maxKeepAge,
                          watchPredef->maxKeepSamples,
                          dcgmWatcher);

    return ret;
}

/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::HelperGetTopologyAffinity(unsigned int groupId, dcgmAffinity_t &gpuAffinity)
{
    dcgmReturn_t dcgmReturn;
    dcgmAffinity_t *affinity_p;
    std::vector<dcgmGroupEntityPair_t> entities;
    std::vector<unsigned int> dcgmGpuIds;

    if (!m_nvmlLoaded.load(std::memory_order_acquire))
    {
        log_debug("Cannot get topology: NVML is not loaded");
        return DCGM_ST_NVML_NOT_LOADED;
    }

    if (gpuAffinity.numGpus > DCGM_MAX_NUM_DEVICES)
    {
        log_error("Invalid gpuAffinity.numGpus: {}", gpuAffinity.numGpus);
        return DCGM_ST_BADPARAM;
    }

    /* Verify group id is valid */
    dcgmReturn = mpGroupManager->verifyAndUpdateGroupId(&groupId);
    if (DCGM_ST_OK != dcgmReturn)
    {
        log_error("Error: Bad group id parameter: {}", groupId);
        return dcgmReturn;
    }

    dcgmReturn = mpGroupManager->GetGroupEntities(groupId, DcgmGroupOption::All, entities);
    if (dcgmReturn != DCGM_ST_OK)
    {
        log_error("Error {} from GetGroupEntities()", (int)dcgmReturn);
        return dcgmReturn;
    }

    for (auto &entitie : entities)
    {
        /* Only consider GPUs */
        if (entitie.entityGroupId != DCGM_FE_GPU)
        {
            continue;
        }

        dcgmGpuIds.push_back(entitie.entityId);
    }

    if (dcgmGpuIds.empty())
    {
        log_debug("No GPUs in group {}", groupId);
        return DCGM_ST_NO_DATA;
    }

    // retrieve the latest sample of PCI topology information
    DcgmFvBuffer affFv;
    dcgmReturn = GetCachedOrLiveValueForEntity({ DCGM_FE_GPU, dcgmGpuIds[0] }, DCGM_FI_GPU_TOPOLOGY_AFFINITY, affFv);
    if (DCGM_ST_OK != dcgmReturn)
    {
        DCGM_LOG_ERROR << "Unable to retrieve affinity information" << errorString(dcgmReturn);
        return dcgmReturn;
    }

    dcgmBufferedFvCursor_t fvCursor = 0;
    affinity_p                      = (dcgmAffinity_t *)affFv.GetNextFv(&fvCursor)->value.blob;

    // now run through the topology list comparing it to the group GPU list and copy over
    // applicable elements
    for (unsigned int elNum = 0; elNum < affinity_p->numGpus; elNum++)
    {
        if (std::find(dcgmGpuIds.begin(), dcgmGpuIds.end(), affinity_p->affinityMasks[elNum].dcgmGpuId)
            != dcgmGpuIds.end())
        {
            DCGM_CASSERT(DCGM_MAX_NUM_DEVICES <= DCGM_ARRAY_CAPACITY(gpuAffinity.affinityMasks), 1);
            if (gpuAffinity.numGpus < DCGM_MAX_NUM_DEVICES)
            {
                memcpy(gpuAffinity.affinityMasks[gpuAffinity.numGpus].bitmask,
                       affinity_p->affinityMasks[elNum].bitmask,
                       sizeof(unsigned long) * DCGM_AFFINITY_BITMASK_ARRAY_SIZE);
                gpuAffinity.affinityMasks[gpuAffinity.numGpus].dcgmGpuId = affinity_p->affinityMasks[elNum].dcgmGpuId;
                gpuAffinity.numGpus++;
            }
            else
            {
                log_error("Maximum number of gpuAffinity masks {} reached", DCGM_MAX_NUM_DEVICES);
                return DCGM_ST_MAX_LIMIT;
            }
        }
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::GetCachedOrLiveValueForEntity(dcgmGroupEntityPair_t entity,
                                                                  unsigned short fieldId,
                                                                  DcgmFvBuffer &fvBuffer)
{
    dcgmReturn_t dcgmReturn;

    /* Try to get a cached version. If not available, get a live version */
    dcgmReturn = mpCacheManager->GetLatestSample(entity.entityGroupId, entity.entityId, fieldId, nullptr, &fvBuffer);
    if (dcgmReturn == DCGM_ST_NOT_WATCHED || dcgmReturn == DCGM_ST_NO_DATA)
    {
        std::vector<dcgmGroupEntityPair_t> entities;
        entities.push_back(entity);

        std::vector<unsigned short> fieldIds;
        fieldIds.push_back(fieldId);

        fvBuffer.Clear(); /* GetLatestSample() writes an error entry */
        dcgmReturn = mpCacheManager->GetMultipleLatestLiveSamples(entities, fieldIds, &fvBuffer);
    }

    return dcgmReturn;
}

/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::HelperGetTopologyIO(unsigned int groupId, dcgmTopology_t &gpuTopology)
{
    dcgmReturn_t dcgmReturn;
    dcgmTopology_t *topologyPci_p;
    dcgmTopology_t *topologyNvLink_p;
    std::vector<dcgmGroupEntityPair_t> entities;
    std::vector<unsigned int> dcgmGpuIds;

    /* Verify group id is valid */
    dcgmReturn = mpGroupManager->verifyAndUpdateGroupId(&groupId);
    if (DCGM_ST_OK != dcgmReturn)
    {
        log_error("Error: Bad group id parameter");
        return dcgmReturn;
    }

    dcgmReturn = mpGroupManager->GetGroupEntities(groupId, DcgmGroupOption::All, entities);
    if (dcgmReturn != DCGM_ST_OK)
    {
        log_error("Error {} from GetGroupEntities()", (int)dcgmReturn);
        return dcgmReturn;
    }

    for (auto &entitie : entities)
    {
        /* Only consider GPUs */
        if (entitie.entityGroupId != DCGM_FE_GPU)
        {
            continue;
        }

        dcgmGpuIds.push_back(entitie.entityId);
    }

    if (dcgmGpuIds.empty())
    {
        log_debug("No GPUs in group {}", groupId);
        return DCGM_ST_NO_DATA;
    }

    // retrieve the latest sample of PCI topology information
    DcgmFvBuffer pciFvBuffer;
    dcgmReturn = GetCachedOrLiveValueForEntity({ DCGM_FE_GPU, dcgmGpuIds[0] }, DCGM_FI_GPU_TOPOLOGY_PCI, pciFvBuffer);
    if (DCGM_ST_OK != dcgmReturn)
    {
        DCGM_LOG_ERROR << "Error: unable to retrieve topology information: " << errorString(dcgmReturn);
        return dcgmReturn;
    }

    dcgmBufferedFvCursor_t fvCursor = 0;
    dcgmBufferedFv_t *bufval        = pciFvBuffer.GetNextFv(&fvCursor);
    if (bufval == nullptr)
    {
        DCGM_LOG_ERROR << "Error: unable to retrieve PCIe topology information: " << errorString(dcgmReturn);
        return DCGM_ST_NOT_SUPPORTED;
    }
    topologyPci_p = (dcgmTopology_t *)bufval->value.blob;

    /*  retrieve the latest sample of NVLINK topology information */
    DcgmFvBuffer nvLinkFvBuffer;
    dcgmReturn
        = GetCachedOrLiveValueForEntity({ DCGM_FE_GPU, dcgmGpuIds[0] }, DCGM_FI_GPU_TOPOLOGY_NVLINK, nvLinkFvBuffer);
    if (DCGM_ST_OK != dcgmReturn)
    {
        DCGM_LOG_ERROR << "Error: unable to retrieve NVLink topology information: " << errorString(dcgmReturn);
        return DCGM_ST_NOT_SUPPORTED;
    }

    fvCursor = 0;
    bufval   = nvLinkFvBuffer.GetNextFv(&fvCursor);
    if (bufval == nullptr)
    {
        DCGM_LOG_ERROR << "Error: unable to retrieve topology information: " << errorString(dcgmReturn);
        return DCGM_ST_NOT_SUPPORTED;
    }
    topologyNvLink_p = (dcgmTopology_t *)bufval->value.blob;

    // now run through the topology list comparing it to the group GPU list and copy over
    // applicable elements
    for (unsigned int elNum = 0; elNum < topologyPci_p->numElements; elNum++)
    {
        // PCI is the leader here as all GPUs will have *some* PCI relationship
        // only peek info the NVLINK topology info if we've found a match for PCI
        if (std::find(dcgmGpuIds.begin(), dcgmGpuIds.end(), topologyPci_p->element[elNum].dcgmGpuA) != dcgmGpuIds.end()
            && std::find(dcgmGpuIds.begin(), dcgmGpuIds.end(), topologyPci_p->element[elNum].dcgmGpuB)
                   != dcgmGpuIds.end()) // both gpus are in our list
        {
            gpuTopology.element[gpuTopology.numElements].dcgmGpuA = topologyPci_p->element[elNum].dcgmGpuA;
            gpuTopology.element[gpuTopology.numElements].dcgmGpuB = topologyPci_p->element[elNum].dcgmGpuB;
            gpuTopology.element[gpuTopology.numElements].path     = topologyPci_p->element[elNum].path;
            gpuTopology.element[gpuTopology.numElements].AtoBNvLinkIds
                = 0; // set to zero just in case there is no NVLINK
            gpuTopology.element[gpuTopology.numElements].BtoANvLinkIds
                = 0; // set to zero just in case there is no NVLINK
            for (unsigned int nvLinkElNum = 0; nvLinkElNum < topologyNvLink_p->numElements; nvLinkElNum++)
            {
                if (topologyNvLink_p->element[nvLinkElNum].dcgmGpuA == topologyPci_p->element[elNum].dcgmGpuA
                    && topologyNvLink_p->element[nvLinkElNum].dcgmGpuB == topologyPci_p->element[elNum].dcgmGpuB)
                {
                    gpuTopology.element[gpuTopology.numElements].path
                        = (dcgmGpuTopologyLevel_t)((int)gpuTopology.element[gpuTopology.numElements].path
                                                   | (int)topologyNvLink_p->element[nvLinkElNum].path);
                    gpuTopology.element[gpuTopology.numElements].AtoBNvLinkIds
                        = topologyNvLink_p->element[nvLinkElNum].AtoBNvLinkIds;
                    gpuTopology.element[gpuTopology.numElements].BtoANvLinkIds
                        = topologyNvLink_p->element[nvLinkElNum].BtoANvLinkIds;
                }
            }
            gpuTopology.numElements++;
        }
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::HelperCreateFakeEntities(dcgmCreateFakeEntities_t *createFakeEntities)
{
    if (!createFakeEntities)
    {
        return DCGM_ST_BADPARAM;
    }

    for (unsigned int i = 0; i < createFakeEntities->numToCreate; i++)
    {
        switch (createFakeEntities->entityList[i].entity.entityGroupId)
        {
            case DCGM_FE_GPU:
                createFakeEntities->entityList[i].entity.entityId = mpCacheManager->AddFakeGpu();
                if (createFakeEntities->entityList[i].entity.entityId == DCGM_GPU_ID_BAD)
                {
                    DCGM_LOG_ERROR << "Got bad fake gpuId DCGM_GPU_ID_BAD from cache manager";
                    return DCGM_ST_GENERIC_ERROR;
                }
                break;

            case DCGM_FE_SWITCH:
            {
                dcgm_nvswitch_msg_create_fake_switch_t nvsMsg {};

                nvsMsg.header.length     = sizeof(nvsMsg);
                nvsMsg.header.moduleId   = DcgmModuleIdNvSwitch;
                nvsMsg.header.subCommand = DCGM_NVSWITCH_SR_CREATE_FAKE_SWITCH;
                nvsMsg.header.version    = dcgm_nvswitch_msg_create_fake_switch_version;
                nvsMsg.numToCreate       = 1;

                dcgmReturn_t dcgmReturn = ProcessModuleCommand(&nvsMsg.header);
                if (dcgmReturn == DCGM_ST_OK && nvsMsg.numCreated == nvsMsg.numToCreate)
                {
                    createFakeEntities->entityList[i].entity.entityId = nvsMsg.switchIds[0];
                }
                else
                {
                    DCGM_LOG_ERROR << "DCGM_NVSWITCH_SR_CREATE_FAKE_SWITCH returned " << dcgmReturn << " numCreated "
                                   << nvsMsg.numCreated;
                    /* Use the return unless it was OK. Else return generic error */
                    return (dcgmReturn == DCGM_ST_OK ? DCGM_ST_GENERIC_ERROR : dcgmReturn);
                }
                break;
            }

            case DCGM_FE_GPU_I:
            {
                createFakeEntities->entityList[i].entity.entityId
                    = mpCacheManager->AddFakeInstance(createFakeEntities->entityList[i].parent.entityId);
                if (createFakeEntities->entityList[i].entity.entityId == DCGM_GPU_ID_BAD)
                {
                    DCGM_LOG_ERROR << "Got bad fake GPU instance ID DCGM_GPU_ID_BAD from cache manager";
                    return DCGM_ST_GENERIC_ERROR;
                }
                break;
            }

            case DCGM_FE_GPU_CI:
            {
                createFakeEntities->entityList[i].entity.entityId
                    = mpCacheManager->AddFakeComputeInstance(createFakeEntities->entityList[i].parent.entityId);
                if (createFakeEntities->entityList[i].entity.entityId == DCGM_GPU_ID_BAD)
                {
                    DCGM_LOG_ERROR << "Got bad fake compute instance ID DCGM_GPU_ID_BAD from cache manager";
                    return DCGM_ST_GENERIC_ERROR;
                }
                break;
            }

            case DCGM_FE_CPU: // fall-through
            case DCGM_FE_CPU_CORE:
            {
                dcgm_sysmon_msg_create_fake_entities_t smonMsg {};

                smonMsg.header.length     = sizeof(smonMsg);
                smonMsg.header.moduleId   = DcgmModuleIdSysmon;
                smonMsg.header.subCommand = DCGM_SYSMON_SR_CREATE_FAKE_ENTITIES;
                smonMsg.header.version    = dcgm_sysmon_msg_create_fake_entities_version;
                smonMsg.groupToCreate     = createFakeEntities->entityList[i].entity.entityGroupId;
                smonMsg.numToCreate       = 1;
                smonMsg.parent            = createFakeEntities->entityList[i].parent;

                dcgmReturn_t dcgmReturn = ProcessModuleCommand(&smonMsg.header);
                if (dcgmReturn == DCGM_ST_OK && smonMsg.numCreated == smonMsg.numToCreate)
                {
                    createFakeEntities->entityList[i].entity.entityId = smonMsg.ids[0];
                }
                else
                {
                    log_error("DCGM_SMON_SR_CREATE_FAKE_CPU returned {} numCreated {}", dcgmReturn, smonMsg.numCreated);
                    /* Use the return unless it was OK. Else return generic error */
                    return (dcgmReturn == DCGM_ST_OK ? DCGM_ST_GENERIC_ERROR : dcgmReturn);
                }

                break;
            }

            case DCGM_FE_LINK: /* RSH -- fake link not supported */

            default:
                log_error("CREATE_FAKE_ENTITIES got unhandled eg {}",
                          createFakeEntities->entityList[i].entity.entityGroupId);
                return DCGM_ST_NOT_SUPPORTED;
        }
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
unsigned int DcgmHostEngineHandler::GetHostEngineHealth() const
{
    return m_hostengineHealth;
}

void DcgmHostEngineHandler::UnloadModules(bool notLoad, uint32_t *loadingMask, uint32_t *unloadedMask)
{
    uint32_t infoMask = loadingMask ? *loadingMask : 0; // mask to unload

    uint32_t runningMask = 1; // running mask to check to mark loaded
    uint32_t loadedMask  = 0; // return mask of modules loaded
    uint32_t unloadMask  = 0; // return mask of modules unloaded

    /**
     * We never set the core module to mark not loaded unless we are actually
     * unloading as part of shutdown (in which case this mask is ignored.
     */

    infoMask &= ~(1 << DcgmModuleIdCore);

    bool resumeCore         = false;
    bool markCoreReloadable = false;

    /* Free sub-modules */
    for (auto &m_module : m_modules)
    {
        /**
         * If we are resetting to reloadable, and not unloaded, we must leave
         * the core module alone. Unloaded is used as a state part of shutdown,
         * from which we can't recover. Reloadable returns the module to a
         * state of never having been loaded, with the exception of actually
         * being loaded and possibly having running threads. This is used to
         * support the test framework module loading and denylisting tests.
         *
         * Reloadable modules can be loaded (as if they were never loaded --
         * this is a noop), and denylisted (as well as actually unloaded at
         * termination). When marked reloadable, paused modules are resumed,
         * and, if necessary, the core module is resumed. This is because
         * when a module is first loaded, it is not paused.
         *
         * If a reloadable module is denylisted, it can NOT be loaded again,
         * though it may be running background threads. It CAN be unloadad at
         * shutdown.
         *
         * This may result in some modules paused, and others not, if
         * reloadability is being used to support the test framework. Usually,
         * either all loaded modules are running or paused, and not a mix.
         */
        if (m_module.status == DcgmModuleStatusLoaded)
        {
            loadedMask |= runningMask;
        }

        /**
         * If the mask bit is clear, we don't mark the module not loaded.
         */
        if (!(runningMask & infoMask))
        {
            if (notLoad)
            {
                runningMask <<= 1;

                continue;
            }
        }

        if (!notLoad) /* we actually unload */
        {
            /**
             * At this point we are free to mark the module unloaded, and
             * possibly actually unload it.
             */

            if ((m_module.ptr != nullptr) && (m_module.freeCB != nullptr))
            {
                m_module.freeCB(m_module.ptr);
            }

            m_module.ptr     = nullptr;
            m_module.allocCB = nullptr;
            m_module.freeCB  = nullptr;
            m_module.msgCB   = nullptr;
        }

        if (!notLoad)
        {
            m_module.status = DcgmModuleStatusUnloaded;
        }
        else if (m_module.ptr == nullptr)
        {
            m_module.status = DcgmModuleStatusNotLoaded;
        }
        else
        {
            switch (m_module.status)
            {
                case DcgmModuleStatusLoaded:
                case DcgmModuleStatusDenylisted:
                    m_module.status = DcgmModuleStatusReloadable;
                    break;

                case DcgmModuleStatusPaused:
                    if (m_module.id == DcgmModuleIdCore)
                    {
                        markCoreReloadable = true;
                    }
                    else
                    {
                        ResumeModule(m_module.id);
                        resumeCore      = true;
                        m_module.status = DcgmModuleStatusReloadable;
                    }
                    break;

                case DcgmModuleStatusNotLoaded:
                case DcgmModuleStatusFailed:
                case DcgmModuleStatusUnloaded:
                case DcgmModuleStatusReloadable:
                    break;
            }
        }

        unloadMask |= runningMask;
        runningMask <<= 1;
    }

    /**
     * If we resumed a module to make it reloadable, we must also ensure the
     * Core module is not paused.
     */
    if (resumeCore && (m_modules[DcgmModuleIdCore].status == DcgmModuleStatusPaused))
    {
        ResumeModule(DcgmModuleIdCore);

        if (markCoreReloadable)
        {
            m_modules[DcgmModuleIdCore].status = DcgmModuleStatusReloadable;
        }
    }

    if (loadingMask)
    {
        *loadingMask = loadedMask;
    }

    if (unloadedMask)
    {
        *unloadedMask = unloadMask;
    }
}

/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::HelperModulesReloadable(dcgmMsgModulesReloadable_v1 &info)
{
    uint32_t unloaded;
    uint32_t infoMask = info.moduleMask;

    {
        auto lock = Lock();

        UnloadModules(true, &info.moduleMask, &unloaded);
    }

    uint32_t runningMask = 1;

    for (auto &m_module : m_modules)
    {
        if (unloaded & runningMask)
        {
            DCGM_LOG_DEBUG << "Marking hostengine module " << m_module.id << " Reloadable";
        }

        runningMask <<= 1;
    }

    DCGM_LOG_DEBUG << "Returning loaded mask " << info.moduleMask << " for info mask" << infoMask;

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::HelperModuleDenylist(dcgmModuleId_t moduleId)
{
    if (moduleId <= DcgmModuleIdCore || moduleId >= DcgmModuleIdCount)
    {
        DCGM_LOG_ERROR << "Invalid moduleId " << moduleId;
        return DCGM_ST_BADPARAM;
    }

    /* Lock the host engine so states don't change under us */
    auto lock = Lock();

    /* React to this based on the current module status */
    switch (m_modules[moduleId].status)
    {
        case DcgmModuleStatusNotLoaded:
            break; /* Will be added to the denylist below */

        case DcgmModuleStatusReloadable:
            /*
             * When actually unloading modules, state is not checked, so the
             * module instance WILL be destructed.
             */
            DCGM_LOG_DEBUG << "Module ID " << moduleId
                           << " is reloadable so denylisting will not stop running threads.";
            break;

        case DcgmModuleStatusDenylisted:
            DCGM_LOG_DEBUG << "Module ID " << moduleId << " is already on the denylist.";
            return DCGM_ST_OK;

        case DcgmModuleStatusFailed:
            DCGM_LOG_DEBUG << "Module ID " << moduleId << " already failed to load. Adding to the denylist.";
            break;

        case DcgmModuleStatusLoaded:
        case DcgmModuleStatusPaused:
            DCGM_LOG_WARNING << "Could not add module " << moduleId << " to the denylist as it was already loaded.";
            return DCGM_ST_IN_USE;

        case DcgmModuleStatusUnloaded:
            DCGM_LOG_DEBUG << "Module ID " << moduleId << " has been unloaded. Adding to the denylist.";
            break;

            /* Not adding a default case here so adding future states will cause a compiler error */
    }

    DCGM_LOG_INFO << "Module " << moduleId << " added to the denylist";
    m_modules[moduleId].status = DcgmModuleStatusDenylisted;

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::HelperModuleStatus(dcgmModuleGetStatuses_v1 &msg)
{
    if (msg.version != dcgmModuleGetStatuses_version)
    {
        DCGM_LOG_ERROR << "Version mismatch";
        return DCGM_ST_VER_MISMATCH;
    }

    /* Note: not locking here because we're not looking at anything sensitive */

    msg.numStatuses = 0;
    for (unsigned int moduleId = DcgmModuleIdCore;
         moduleId < DcgmModuleIdCount && msg.numStatuses < DCGM_MODULE_STATUSES_CAPACITY;
         moduleId++)
    {
        msg.statuses[msg.numStatuses].id     = m_modules[moduleId].id;
        msg.statuses[msg.numStatuses].status = m_modules[moduleId].status;
        msg.numStatuses++;
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::HelperGetFieldSummary(dcgmFieldSummaryRequest_t &fieldSummary)
{
    dcgmReturn_t ret;
    dcgm_field_meta_p fm = DcgmFieldGetById(fieldSummary.fieldId);

    if (fm == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }

    int numSummaryTypes   = 0;
    timelib64_t startTime = fieldSummary.startTime;
    timelib64_t endTime   = fieldSummary.endTime;
    auto entityGroupId    = static_cast<dcgm_field_entity_group_t>(fieldSummary.entityGroupId);

    DcgmcmSummaryType_t summaryTypes[DcgmcmSummaryTypeSize];
    memset(&summaryTypes, 0, sizeof(summaryTypes));

    for (int i = 0; i < static_cast<int>(DcgmcmSummaryTypeSize); i++)
    {
        if ((fieldSummary.summaryTypeMask & 0x1 << i) != 0)
        {
            summaryTypes[numSummaryTypes] = static_cast<DcgmcmSummaryType_t>(i);
            numSummaryTypes++;
        }
    }

    fieldSummary.response.fieldType    = fm->fieldType;
    fieldSummary.response.summaryCount = numSummaryTypes;

    switch (fm->fieldType)
    {
        case DCGM_FT_DOUBLE:
        {
            double dSummaryValues[DcgmcmSummaryTypeSize];

            ret = mpCacheManager->GetFp64SummaryData(entityGroupId,
                                                     fieldSummary.entityId,
                                                     fieldSummary.fieldId,
                                                     numSummaryTypes,
                                                     summaryTypes,
                                                     dSummaryValues,
                                                     startTime,
                                                     endTime,
                                                     nullptr,
                                                     nullptr);
            if (ret == DCGM_ST_OK)
            {
                // Copy the values back into the response
                for (int i = 0; i < numSummaryTypes; i++)
                {
                    fieldSummary.response.values[i].fp64 = dSummaryValues[i];
                }
            }

            break;
        }
        case DCGM_FT_INT64:
        {
            long long iSummaryValues[DcgmcmSummaryTypeSize];

            ret = mpCacheManager->GetInt64SummaryData(entityGroupId,
                                                      fieldSummary.entityId,
                                                      fieldSummary.fieldId,
                                                      numSummaryTypes,
                                                      summaryTypes,
                                                      iSummaryValues,
                                                      startTime,
                                                      endTime,
                                                      nullptr,
                                                      nullptr);

            if (ret == DCGM_ST_OK)
            {
                // Copy the values back into the response
                for (int i = 0; i < numSummaryTypes; i++)
                {
                    fieldSummary.response.values[i].i64 = iSummaryValues[i];
                }
            }

            break;
        }
        default:
        {
            // We only support this call for int64 and doubles
            ret = DCGM_ST_FIELD_UNSUPPORTED_BY_API;
            break;
        }
    }

    return ret;
}

/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::SendRawMessageToEmbeddedClient(unsigned int msgType,
                                                                   dcgm_request_id_t requestId,
                                                                   void *msgData,
                                                                   int msgLength,
                                                                   dcgmReturn_t status)
{
    std::shared_ptr<DcgmRequest> request;

    /* Embedded client */
    if (requestId == DCGM_REQUEST_ID_NONE)
    {
        log_error("Can't SendRawMessageToEmbeddedClient() with 0 requestId");
        return DCGM_ST_GENERIC_ERROR;
    }

    {
        std::lock_guard<std::mutex> lock(m_watchedRequestsMutex);

        auto requestIt = m_watchedRequests.find(requestId);
        if (requestIt == m_watchedRequests.end())
        {
            log_error("SendRawMessageToEmbeddedClient unable to find requestId {}", requestId);
            return DCGM_ST_BADPARAM;
        }

        /* Get a shared_ptr copy to keep the request alive after releasing the lock */
        request = requestIt->second;
    }
    /* m_watchedRequestsMutex released here. The shared_ptr keeps the request alive. */

    /* ProcessMessage is expecting an allocated message */
    std::unique_ptr<DcgmMessage> msg = std::make_unique<DcgmMessage>();
    msg->UpdateMsgHdr(msgType, requestId, status, msgLength);

    /* Make a copy of the incoming buffer, as this could be stack-allocated or heap allocated */
    auto msgBytes = msg->GetMsgBytesPtr();
    msgBytes->resize(msgLength);
    memcpy(msgBytes->data(), msgData, msgLength);

    request->ProcessMessage(std::move(msg));
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::SendRawMessageToClient(dcgm_connection_id_t connectionId,
                                                           unsigned int msgType,
                                                           dcgm_request_id_t requestId,
                                                           void *msgData,
                                                           int msgLength,
                                                           dcgmReturn_t status)
{
    if (connectionId == DCGM_CONNECTION_ID_NONE)
    {
        return SendRawMessageToEmbeddedClient(msgType, requestId, msgData, msgLength, status);
    }

    /* Copy the raw message to a message object that we will move to DcgmIpc */
    std::unique_ptr<DcgmMessage> dcgmMessage = std::make_unique<DcgmMessage>();

    dcgmMessage->UpdateMsgHdr(msgType, requestId, status, msgLength);

    auto msgBytes = dcgmMessage->GetMsgBytesPtr();
    msgBytes->resize(msgLength);
    memcpy(msgBytes->data(), msgData, msgLength);

    dcgmReturn_t retSt = m_dcgmIpc->SendMessage(connectionId, std::move(dcgmMessage), false);

    DCGM_LOG_DEBUG << "Sent raw message length " << msgLength << ", requestId " << requestId << ", msgType 0x"
                   << std::hex << msgType << " to connectionId " << std::dec << connectionId << " retSt " << (int)retSt;
    return retSt;
}

/*****************************************************************************/
void DcgmHostEngineHandler::NotifyLoggingSeverityChange()
{
    dcgm_core_msg_logging_changed_t msg;
    memset(&msg, 0, sizeof(msg));
    msg.header.length     = sizeof(msg);
    msg.header.version    = dcgm_core_msg_logging_changed_version;
    msg.header.subCommand = DCGM_CORE_SR_LOGGING_CHANGED;

    /* Notify each module about the group removal */
    for (unsigned int id = 0; id < DcgmModuleIdCount; id++)
    {
        if (m_modules[id].ptr != nullptr)
        {
            SendModuleMessage((dcgmModuleId_t)id, (dcgm_module_command_header_t *)&msg);
        }
    }
}

/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::ProcessModuleCommand(dcgm_module_command_header_t *moduleCommand)
{
    dcgmReturn_t dcgmReturn;

    if (static_cast<std::underlying_type_t<dcgmModuleId_t>>(moduleCommand->moduleId)
        >= static_cast<std::underlying_type_t<dcgmModuleId_t>>(DcgmModuleIdCount))
    {
        log_error("Invalid module id: {}", moduleCommand->moduleId);
        return DCGM_ST_BADPARAM;
    }

    /**
     * Is the module denylisted?
     *
     * This is possible even with a module ptr if it was loaded, then marked
     * reloadable and finally denylisted. It is still loaded and possibly
     * running an existing thread, but can't be reloaded. However, because it
     * has a module ptr field the following tests will result in acting as if
     * it still loaded. So, we have to check for the denylisted case.
     */

    if (m_modules[moduleCommand->moduleId].status == DcgmModuleStatusDenylisted)
    {
        log_error("Module id: {} is denylisted, acting not loaded.", moduleCommand->moduleId);

        return DCGM_ST_MODULE_NOT_LOADED;
    }

    /* Is the module loaded? */
    if (m_modules[moduleCommand->moduleId].ptr == nullptr)
    {
        dcgmReturn = LoadModule(moduleCommand->moduleId);
        if (dcgmReturn != DCGM_ST_OK)
        {
            return dcgmReturn;
        }
    }

    if (m_modules[moduleCommand->moduleId].status == DcgmModuleStatusReloadable)
    {
        dcgmReturn = LoadModule(moduleCommand->moduleId);
        if (dcgmReturn != DCGM_ST_OK)
        {
            return dcgmReturn;
        }
    }

    if (moduleCommand->moduleId == DcgmModuleIdCore && moduleCommand->subCommand == DCGM_CORE_SR_PAUSE_RESUME)
    {
        /* Pause and resume command are dispatched to all modules in specific order */
        return ProcessPauseResume(reinterpret_cast<dcgm_core_msg_pause_resume_v1 *>(moduleCommand));
    }

    /* Dispatch the message */
    return SendModuleMessage(moduleCommand->moduleId, moduleCommand);
}

/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::GetCacheManagerFieldInfo(dcgmCacheManagerFieldInfo_v4_t *fieldInfo)
{
    return mpCacheManager->GetCacheManagerFieldInfo(fieldInfo);
}

/*****************************************************************************/
void DcgmHostEngineHandler::OnConnectionRemove(dcgm_connection_id_t connectionId)
{
    if (mpGroupManager != nullptr)
    {
        mpGroupManager->OnConnectionRemove(connectionId);
    }
    if (mpFieldGroupManager != nullptr)
    {
        mpFieldGroupManager->OnConnectionRemove(connectionId);
    }
    /* Call the cache manager last since the rest of the modules refer to it */
    if (mpCacheManager != nullptr)
    {
        mpCacheManager->OnConnectionRemove(connectionId);
    }

    /* Notify each module about the client disconnect */
    dcgm_core_msg_client_disconnect_t msg;
    memset(&msg, 0, sizeof(msg));
    msg.header.length       = sizeof(msg);
    msg.header.version      = dcgm_core_msg_client_disconnect_version;
    msg.header.subCommand   = DCGM_CORE_SR_CLIENT_DISCONNECT;
    msg.header.connectionId = connectionId;
    msg.connectionId        = connectionId;

    for (unsigned int id = 0; id < DcgmModuleIdCount; id++)
    {
        if (m_modules[id].ptr != nullptr)
        {
            SendModuleMessage((dcgmModuleId_t)id, (dcgm_module_command_header_t *)&msg);
        }
    }

    ClearPersistAfterDisconnect(connectionId);
}

void DcgmHostEngineHandler::DetachGpusFromModules()
{
    dcgm_core_msg_detach_gpus_v1 msg {};
    msg.header.length     = sizeof(msg);
    msg.header.version    = dcgm_core_msg_detach_gpus_version;
    msg.header.subCommand = DCGM_CORE_SR_DETACH_GPUS;

    auto locked    = Lock();
    m_gpusDetached = true;
    for (auto &&[idx, module] : m_modules | std::views::enumerate)
    {
        if (module.ptr != nullptr)
        {
            SendModuleMessage((dcgmModuleId_t)idx, (dcgm_module_command_header_t *)&msg);
        }
    }
}

void DcgmHostEngineHandler::AttachGpusToModules()
{
    dcgm_core_msg_attach_gpus_v1 msg {};
    msg.header.length     = sizeof(msg);
    msg.header.version    = dcgm_core_msg_attach_gpus_version;
    msg.header.subCommand = DCGM_CORE_SR_ATTACH_GPUS;

    auto locked    = Lock();
    m_gpusDetached = false;
    for (auto &&[idx, module] : m_modules | std::views::enumerate)
    {
        if (module.ptr != nullptr)
        {
            SendModuleMessage((dcgmModuleId_t)idx, (dcgm_module_command_header_t *)&msg);
        }
    }
}

/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::ProcessModuleCommandMsg(dcgm_connection_id_t connectionId,
                                                            std::unique_ptr<DcgmMessage> message)
{
    dcgmReturn_t retSt = DCGM_ST_OK;

    auto msgBytes  = message->GetMsgBytesPtr();
    auto msgHeader = message->GetMessageHdr();

/* Resize our buffer to be the maximum size of a DCGM message. This is so
   the module command response can be larger than the request

   Note: We aren't doing this for now since we don't have any assymetric requests
   where module commands have different response size from request size.

   This avoids the performance penalty of allocating and zeroing 4 MB of memory
   on every user request */
#if 0
    msgBytes->resize(DCGM_PROTO_MAX_MESSAGE_SIZE);
#endif

    auto moduleCommand = (dcgm_module_command_header_t *)msgBytes->data();

    /* Verify that we didn't get a malicious moduleCommand->length. This also implicitly
       checks that our message isn't larger than DCGM_PROTO_MAX_MESSAGE_SIZE
       since DcgmIpc checks that when it assembles messages from the socket stream. */
    if (moduleCommand->length != message->GetLength())
    {
        DCGM_LOG_ERROR << "Module command has bad length " << moduleCommand->length << " != " << message->GetLength();
        return DCGM_ST_BADPARAM;
    }

    /* Resize buffer for certain commands that may have large response payloads */
    if (moduleCommand->moduleId == DcgmModuleIdCore)
    {
        switch (moduleCommand->subCommand)
        {
            case DCGM_CORE_SR_ENTITIES_GET_LATEST_VALUES_V3:
                msgBytes->resize(sizeof(dcgm_core_msg_entities_get_latest_values_v3));
                moduleCommand         = (dcgm_module_command_header_t *)msgBytes->data();
                moduleCommand->length = sizeof(dcgm_core_msg_entities_get_latest_values_v3);
                break;
            case DCGM_CORE_SR_ENTITIES_GET_LATEST_VALUES_V2:
                msgBytes->resize(sizeof(dcgm_core_msg_entities_get_latest_values_v2));
                moduleCommand         = (dcgm_module_command_header_t *)msgBytes->data();
                moduleCommand->length = sizeof(dcgm_core_msg_entities_get_latest_values_v2);
                break;
            case DCGM_CORE_SR_ENTITIES_GET_LATEST_VALUES_V1:
                msgBytes->resize(sizeof(dcgm_core_msg_entities_get_latest_values_v1));
                moduleCommand         = (dcgm_module_command_header_t *)msgBytes->data();
                moduleCommand->length = sizeof(dcgm_core_msg_entities_get_latest_values_v1);
                break;
            case DCGM_CORE_SR_GET_MULTIPLE_VALUES_FOR_FIELD_V1:
                msgBytes->resize(sizeof(dcgm_core_msg_get_multiple_values_for_field_v1));
                moduleCommand         = (dcgm_module_command_header_t *)msgBytes->data();
                moduleCommand->length = sizeof(dcgm_core_msg_get_multiple_values_for_field_v1);
                break;
            case DCGM_CORE_SR_GET_MULTIPLE_VALUES_FOR_FIELD_V2:
                msgBytes->resize(sizeof(dcgm_core_msg_get_multiple_values_for_field_v2));
                moduleCommand         = (dcgm_module_command_header_t *)msgBytes->data();
                moduleCommand->length = sizeof(dcgm_core_msg_get_multiple_values_for_field_v2);
                break;
            default:
                /* No need to resize */
                break;
        }
    }

    if (moduleCommand->requestId == DCGM_REQUEST_ID_NONE)
        moduleCommand->requestId = msgHeader->requestId;

    moduleCommand->connectionId = connectionId;

    dcgmReturn_t requestStatus = ProcessModuleCommand(moduleCommand);

    /* Resize msgBytes to whatever moduleCommand's updated size is */
    msgBytes->resize(moduleCommand->length);

    message->UpdateMsgHdr(DCGM_MSG_MODULE_COMMAND, moduleCommand->requestId, requestStatus, moduleCommand->length);

    m_dcgmIpc->SendMessage(connectionId, std::move(message), false);
    return retSt;
}

/*****************************************************************************/
void DcgmHostEngineHandler::ProcessMessage(dcgm_connection_id_t connectionId, std::unique_ptr<DcgmMessage> message)
{
    switch (message->GetMsgType())
    {
        case DCGM_MSG_PROTO_REQUEST:
        case DCGM_MSG_PROTO_RESPONSE:
            DCGM_LOG_ERROR << "Got unhandled protobuf message from connectionId " << connectionId;
            break;

        case DCGM_MSG_MODULE_COMMAND:
            ProcessModuleCommandMsg(connectionId, std::move(message));
            break;

        default:
            DCGM_LOG_ERROR << "Unable to process msgType 0x" << std::hex << message->GetMsgType();
            break;
    }
}

/*****************************************************************************/
void DcgmHostEngineHandler::StaticProcessMessage(dcgm_connection_id_t connectionId,
                                                 std::unique_ptr<DcgmMessage> message,
                                                 void *userData)
{
    DcgmHostEngineHandler *he = (DcgmHostEngineHandler *)userData;
    he->ProcessMessage(connectionId, std::move(message));
}

/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::WatchHostEngineFields()
{
    std::vector<unsigned short> fieldIds;
    dcgmReturn_t dcgmReturn;
    DcgmWatcher watcher(DcgmWatcherTypeHostEngine, DCGM_CONNECTION_ID_NONE);

    if (!m_nvmlLoaded.load(std::memory_order_acquire))
    {
        log_debug("Not watching host engine fields because NVML is not loaded.");
        return DCGM_ST_OK;
    }

    fieldIds.clear();
    fieldIds.push_back(DCGM_FI_DEV_ECC_CURRENT); /* Can really only change once per driver reload. NVML caches this so
                                                    it's virtually a no-op */

    /* Don't bother with vGPU fields unless we're in Host vGPU mode per DCGM-513 */
    if (mpCacheManager->AreAnyGpusInHostVGPUMode())
    {
        fieldIds.push_back(DCGM_FI_DEV_CREATABLE_VGPU_TYPE_IDS); /* Used by dcgmVgpuDeviceAttributes_t */
        fieldIds.push_back(DCGM_FI_DEV_VGPU_INSTANCE_IDS);       /* Used by dcgmVgpuDeviceAttributes_t */
    }

    dcgmReturn = mpFieldGroupManager->AddFieldGroup("DCGM_INTERNAL_30SEC", fieldIds, &mFieldGroup30Sec, watcher);
    if (dcgmReturn != DCGM_ST_OK)
    {
        log_error("AddFieldGroup returned {}", (int)dcgmReturn);
        return dcgmReturn;
    }

    // Max number of entries 14400/30 entries
    dcgmReturn = WatchFieldGroup(mpGroupManager->GetAllGpusGroup(), mFieldGroup30Sec, 30000000, 14400.0, 480, watcher);
    if (dcgmReturn != DCGM_ST_OK)
    {
        log_error("WatchFieldGroup returned {}", (int)dcgmReturn);
        return dcgmReturn;
    }

    fieldIds.clear();
    /* Needed as it is the static info related to GPU attribute associated with vGPU */
    fieldIds.push_back(DCGM_FI_DEV_SUPPORTED_TYPE_INFO);
    fieldIds.push_back(DCGM_FI_DEV_SUPPORTED_VGPU_TYPE_IDS);
    fieldIds.push_back(DCGM_FI_DEV_VGPU_TYPE_INFO);
    fieldIds.push_back(DCGM_FI_DEV_VGPU_TYPE_NAME);
    fieldIds.push_back(DCGM_FI_DEV_VGPU_TYPE_CLASS);
    fieldIds.push_back(DCGM_FI_DEV_VGPU_TYPE_LICENSE);

    dcgmReturn = mpFieldGroupManager->AddFieldGroup("DCGM_INTERNAL_HOURLY", fieldIds, &mFieldGroupHourly, watcher);
    if (dcgmReturn != DCGM_ST_OK)
    {
        log_error("AddFieldGroup returned {}", (int)dcgmReturn);
        return dcgmReturn;
    }

    // Max number of entries 14400/3600 entries. Include non-DCGM GPUs
    dcgmReturn = WatchFieldGroupAllGpus(mFieldGroupHourly, 3600000000, 14400.0, 4, 0, watcher);
    if (dcgmReturn != DCGM_ST_OK)
    {
        log_error("WatchFieldGroupAllGpus returned {}", (int)dcgmReturn);
        return dcgmReturn;
    }

    /* Process / job stats fields. Just add the group. The user will watch the fields */
    fieldIds.clear();
    fieldIds.push_back(DCGM_FI_DEV_ACCOUNTING_DATA);
    fieldIds.push_back(DCGM_FI_DEV_POWER_USAGE);
    fieldIds.push_back(DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION);
    fieldIds.push_back(DCGM_FI_DEV_PCIE_TX_THROUGHPUT);
    fieldIds.push_back(DCGM_FI_DEV_PCIE_RX_THROUGHPUT);
    fieldIds.push_back(DCGM_FI_DEV_PCIE_REPLAY_COUNTER);
    fieldIds.push_back(DCGM_FI_DEV_PCIE_COUNT_CORRECTABLE_ERRORS);
    fieldIds.push_back(DCGM_FI_DEV_GPU_UTIL);
    fieldIds.push_back(DCGM_FI_DEV_MEM_COPY_UTIL);
    fieldIds.push_back(DCGM_FI_DEV_ECC_DBE_VOL_TOTAL);
    fieldIds.push_back(DCGM_FI_DEV_SM_CLOCK);
    fieldIds.push_back(DCGM_FI_DEV_MEM_CLOCK);
    fieldIds.push_back(DCGM_FI_DEV_XID_ERRORS);
    fieldIds.push_back(DCGM_FI_DEV_COMPUTE_PIDS);
    fieldIds.push_back(DCGM_FI_DEV_GRAPHICS_PIDS);
    fieldIds.push_back(DCGM_FI_DEV_POWER_VIOLATION);
    fieldIds.push_back(DCGM_FI_DEV_THERMAL_VIOLATION);
    fieldIds.push_back(DCGM_FI_DEV_SYNC_BOOST_VIOLATION);
    fieldIds.push_back(DCGM_FI_DEV_MEM_COPY_UTIL_SAMPLES);
    fieldIds.push_back(DCGM_FI_DEV_GPU_UTIL_SAMPLES);
    fieldIds.push_back(DCGM_FI_DEV_RETIRED_SBE);
    fieldIds.push_back(DCGM_FI_DEV_RETIRED_DBE);
    fieldIds.push_back(DCGM_FI_DEV_RETIRED_PENDING);
    fieldIds.push_back(DCGM_FI_DEV_INFOROM_CONFIG_VALID);

    fieldIds.push_back(DCGM_FI_DEV_THERMAL_VIOLATION);
    fieldIds.push_back(DCGM_FI_DEV_POWER_VIOLATION);

    /* Add Watch for NVLINK flow control CRC Error Counter for all the lanes */
    fieldIds.push_back(DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_TOTAL);
    /* Add Watch for NVLINK data CRC Error Counter for all the lanes */
    fieldIds.push_back(DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_TOTAL);
    /* Add Watch for NVLINK Replay Error Counter for all the lanes */
    fieldIds.push_back(DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_TOTAL);
    /* Add Watch for NVLINK Recovery Error Counter for all the lanes*/
    fieldIds.push_back(DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_TOTAL);


    // reliability violation time
    // board violation time
    // low utilization time

    dcgmReturn = mpFieldGroupManager->AddFieldGroup("DCGM_INTERNAL_JOB", fieldIds, &mFieldGroupPidAndJobStats, watcher);
    if (dcgmReturn != DCGM_ST_OK)
    {
        log_error("AddFieldGroup returned {}", (int)dcgmReturn);
        return dcgmReturn;
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
static void HostEngineOnGroupEventCB(unsigned int groupId, void *userData)
{
    auto *hostEngineHandler = (DcgmHostEngineHandler *)userData;

    hostEngineHandler->OnGroupRemove(groupId);
}

/*****************************************************************************/
void DcgmHostEngineHandler::OnGroupRemove(unsigned int groupId)
{
    /* Notify each module about the client disconnect */
    dcgm_core_msg_group_removed_t msg;
    memset(&msg, 0, sizeof(msg));
    msg.header.length     = sizeof(msg);
    msg.header.version    = dcgm_core_msg_group_removed_version;
    msg.header.subCommand = DCGM_CORE_SR_GROUP_REMOVED;
    msg.groupId           = groupId;

    /* Notify each module about the group removal */
    for (unsigned int id = 0; id < DcgmModuleIdCount; id++)
    {
        if (m_modules[id].ptr != nullptr)
        {
            SendModuleMessage((dcgmModuleId_t)id, (dcgm_module_command_header_t *)&msg);
        }
    }
}

/*****************************************************************************/
void DcgmHostEngineHandler::OnFvUpdates(DcgmFvBuffer *fvBuffer,
                                        DcgmWatcherType_t *watcherTypes,
                                        int numWatcherTypes,
                                        void * /*userData*/)
{
    static dcgmModuleId_t watcherToModuleMap[DcgmWatcherTypeCount]
        = { DcgmModuleIdCore, DcgmModuleIdCore, DcgmModuleIdHealth,  DcgmModuleIdPolicy,
            DcgmModuleIdCore, DcgmModuleIdCore, DcgmModuleIdNvSwitch };

    /* prepare the message for sending to modules */
    dcgm_core_msg_field_values_updated_t msg;
    size_t elementCount = 0;
    memset(&msg, 0, sizeof(msg));
    msg.header.length      = sizeof(msg);
    msg.header.version     = dcgm_core_msg_field_values_updated_version;
    msg.header.subCommand  = DCGM_CORE_SR_FIELD_VALUES_UPDATED;
    msg.fieldValues.buffer = fvBuffer->GetBuffer();
    fvBuffer->GetSize(&msg.fieldValues.bufferSize, &elementCount);

    /* Dispatch each watcher to the corresponding module */
    dcgmModuleId_t destinationModuleId;
    int i;

    for (i = 0; i < numWatcherTypes; i++)
    {
        destinationModuleId = watcherToModuleMap[watcherTypes[i]];
        if (destinationModuleId == DcgmModuleIdCore)
        {
            log_error("Unhandled watcherType {} can't be dispatched to a module.", watcherTypes[i]);
            continue;
        }

        if (m_modules[destinationModuleId].ptr == nullptr)
        {
            log_debug("Skipping FV update for moduleId {} that is not loaded.", destinationModuleId);
            continue;
        }

        SendModuleMessage((dcgmModuleId_t)destinationModuleId, (dcgm_module_command_header_t *)&msg);
    }
}

/*****************************************************************************/
void DcgmHostEngineHandler::OnMigUpdates(unsigned int gpuId)
{
    dcgm_core_msg_mig_updated_t msg;
    memset(&msg, 0, sizeof(msg));
    msg.header.length     = sizeof(msg);
    msg.header.moduleId   = DcgmModuleIdCore;
    msg.header.subCommand = DCGM_CORE_SR_MIG_UPDATED;
    msg.header.version    = dcgm_core_msg_mig_updated_version;
    msg.gpuId             = gpuId;

    for (auto &m_module : m_modules)
    {
        if (m_module.ptr == nullptr)
        {
            /* Module not loaded */
            continue;
        }

        if (m_module.id == DcgmModuleIdCore)
        {
            /* Don't dispatch to ourself */
            continue;
        }

        SendModuleMessage((dcgmModuleId_t)m_module.id, (dcgm_module_command_header_t *)&msg);
    }
}

/*****************************************************************************/
static void nvHostEngineFvCallback(DcgmFvBuffer *fvBuffer,
                                   DcgmWatcherType_t *watcherTypes,
                                   int numWatcherTypes,
                                   void *userData)
{
    auto *hostEngineHandler = (DcgmHostEngineHandler *)userData;

    hostEngineHandler->OnFvUpdates(fvBuffer, watcherTypes, numWatcherTypes, userData);
}

static void nvHostEngineMigCallback(unsigned int gpuId, void *userData)
{
    auto *hostEngineHandler = (DcgmHostEngineHandler *)userData;
    hostEngineHandler->OnMigUpdates(gpuId);
}

void DcgmHostEngineHandler::ShutdownNvml() noexcept
{
    try
    {
        if (!m_nvmlLoaded.load(std::memory_order_acquire))
        {
            return;
        }

        if (m_usingInjectionNvml)
        {
#ifdef INJECTION_LIBRARY_AVAILABLE
            if (NVML_SUCCESS != nvmlShutdown())
            {
                log_error("Error: Failed to shutdown injection NVML");
            }
            else
            {
                m_nvmlLoaded.store(false, std::memory_order_release);
            }
#endif
        }
        else
        {
            if (NVML_SUCCESS != nvmlShutdown())
            {
                /* we used to throw an exception here, which would crash the host engine on shutdown.
                Just log an error and continue our shutdown. */
                log_error("Error: Failed to ShutDown NVML");
            }
            else
            {
                m_nvmlLoaded.store(false, std::memory_order_release);
            }
        }
    }
    catch (const std::logic_error &e)
    {
        /* The application is hereafter in an undefined state. As above, we could throw an exception or
           abruptly terminate, but we'll just log an error and continue to shutdown. */
        log_error("Failed to shutdown NVML: failed to acquire exclusive lock for NVML: {}", e.what());
    }
}

void DcgmHostEngineHandler::LoadNvml()
{
#ifdef INJECTION_LIBRARY_AVAILABLE
    char *injectionMode = getenv(INJECTION_MODE_ENV_VAR);

    if (injectionMode != nullptr)
    {
        m_usingInjectionNvml = true;
        if (NVML_SUCCESS != nvmlInit_v2())
        {
            throw std::runtime_error("Error: Failed to initialize injected NVML");
        }
        else
        {
            m_nvmlLoaded.store(true, std::memory_order_release);
        }
        log_info("Using injected NVML");
    }
#endif

    if (m_usingInjectionNvml == false)
    {
        auto ret = nvmlInitWithFlags(NVML_INIT_FLAG_NO_GPUS | NVML_INIT_FLAG_FORCE_INIT);
        if (NVML_SUCCESS != ret)
        {
            log_error("Cannot load NVML; DCGM will proceed without managing GPUs. ret: {}", ret);
            m_nvmlLoaded.store(false, std::memory_order_release);
        }
        else
        {
            m_nvmlLoaded.store(true, std::memory_order_release);
        }
    }
}

DcgmNvmlSystemEventThread::DcgmNvmlSystemEventThread(DcgmHostEngineHandler &hostEngineHandler)
    : m_hostEngineHandler(hostEngineHandler)
{
    if (m_hostEngineHandler.IsNvmlLoaded())
    {
        throw std::runtime_error("NVML is already loaded, nvmlInitWithFlags should be the first call to init nvml.");
    }
    SetThreadName("sysev_thread");

    if (Init() != DCGM_ST_OK)
    {
        log_info("Failed to initialize NVML system event thread");
        return;
    }
}

dcgmReturn_t DcgmNvmlSystemEventThread::Init()
{
    if (m_threadReadyToGo)
    {
        return DCGM_ST_OK;
    }

    if (!m_nvmlInitialized)
    {
        log_info("Initializing NVML with NVML_INIT_FLAG_NO_GPUS and NVML_INIT_FLAG_NO_ATTACH flags.");
        auto ret = nvmlInitWithFlags(NVML_INIT_FLAG_NO_GPUS | NVML_INIT_FLAG_NO_ATTACH);
        if (ret != NVML_SUCCESS)
        {
            log_error("Failed to initialize NVML: {}", ret);
            return DcgmNs::Utils::NvmlReturnToDcgmReturn(ret);
        }
        // On driver versions less than 590.37, we need to initialize NVML only with NVML_INIT_FLAG_NO_GPUS. Otherwise,
        // the nvmlDeviceGetCount API will return 0 count for devices and no NVML_INIT_FLAG_FORCE_INIT flag can be
        // used.
        char driverVersionBuf[NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE] = {};
        ret = nvmlSystemGetDriverVersion(driverVersionBuf, NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE);
        if (ret != NVML_SUCCESS)
        {
            log_error("Failed to get driver version: {}", ret);
            return DcgmNs::Utils::NvmlReturnToDcgmReturn(ret);
        }
        log_debug("Driver version read from NVML: {}", driverVersionBuf);
        std::string_view driverVersionStr(driverVersionBuf);
        auto dotPos = driverVersionStr.find(".");
        if (dotPos == std::string_view::npos)
        {
            log_error("Driver version {} is not in the expected format.", driverVersionStr);
            return DCGM_ST_GENERIC_ERROR;
        }
        std::string_view majorDriverVersionStr = driverVersionStr.substr(0, dotPos);
        std::string_view minorDriverVersionStr = driverVersionStr.substr(dotPos + 1);
        auto majorDriverVersion                = DcgmNs::StrToUint32(majorDriverVersionStr);
        auto minorDriverVersion                = DcgmNs::StrToUint32(minorDriverVersionStr);

        if (majorDriverVersion.is_error() || minorDriverVersion.is_error())
        {
            log_error("Failed to parse driver version: {}", driverVersionStr);
            return DCGM_ST_GENERIC_ERROR;
        }

        std::uint32_t constexpr majorDriverVersion590 = 590;
        std::uint32_t constexpr minorDriverVersion37  = 37;
        log_debug("Driver version: {}", driverVersionStr);
        if (*majorDriverVersion < majorDriverVersion590
            || (*majorDriverVersion == majorDriverVersion590 && *minorDriverVersion < minorDriverVersion37))
        {
            log_debug("Driver version {} is less than 590.37, initializing NVML with NVML_INIT_FLAG_NO_GPUS only.",
                      driverVersionStr);
            log_debug("Re-initializing NVML with NVML_INIT_FLAG_NO_GPUS only");
            nvmlShutdown();
            ret = nvmlInitWithFlags(NVML_INIT_FLAG_NO_GPUS);
            if (ret != NVML_SUCCESS)
            {
                log_error("Failed to initialize NVML: {}", ret);
                return DcgmNs::Utils::NvmlReturnToDcgmReturn(ret);
            }
        }
        log_info("NVML initialized successfully");
        m_nvmlInitialized = true;
    }
    DcgmNs::Defer rollBack([this] {
        if (!Shutdown())
        {
            log_error("Failed to shutdown NVML system event thread");
        }
    });

    if (!m_eventSetCreated)
    {
        nvmlSystemEventSetCreateRequest_t createRequest = {};
        createRequest.version                           = nvmlSystemEventSetCreateRequest_v1;
        auto ret                                        = nvmlSystemEventSetCreate(&createRequest);
        if (ret != NVML_SUCCESS)
        {
            if (ret != NVML_ERROR_FUNCTION_NOT_FOUND)
            {
                log_error("Failed to create NVML system event set: {}", ret);
                return DcgmNs::Utils::NvmlReturnToDcgmReturn(ret);
            }
            else
            {
                log_info("NVML system event set is not supported, skipping");
                return DCGM_ST_NOT_SUPPORTED;
            }
        }
        m_eventSetCreated = true;
        m_eventSet        = createRequest.set;
    }

    nvmlSystemRegisterEventRequest_t registerEvent;
    registerEvent.version    = nvmlSystemRegisterEventRequest_v1;
    registerEvent.eventTypes = nvmlSystemEventTypeGpuDriverBind | nvmlSystemEventTypeGpuDriverUnbind;
    registerEvent.set        = m_eventSet;
    auto ret                 = nvmlSystemRegisterEvents(&registerEvent);
    if (ret != NVML_SUCCESS)
    {
        log_error("Failed to register NVML system event: {}", ret);
        return DcgmNs::Utils::NvmlReturnToDcgmReturn(ret);
    }
    m_threadReadyToGo = true;
    rollBack.Disarm();
    return DCGM_ST_OK;
}

DcgmNvmlSystemEventThread::~DcgmNvmlSystemEventThread()
{
    if (!Shutdown())
    {
        log_error("Failed to shutdown NVML system event thread");
    }
}

bool DcgmNvmlSystemEventThread::Shutdown()
{
    bool success = true;

    if (m_eventSetCreated)
    {
        nvmlSystemEventSetFreeRequest_t freeRequest = {};
        freeRequest.version                         = nvmlSystemEventSetFreeRequest_v1;
        freeRequest.set                             = m_eventSet;
        auto ret                                    = nvmlSystemEventSetFree(&freeRequest);
        if (ret != NVML_SUCCESS)
        {
            log_error("Failed to free NVML system event set: {}", ret);
            success = false;
        }
        m_eventSetCreated = false;
    }

    if (m_nvmlInitialized)
    {
        auto ret = nvmlShutdown();
        if (ret != NVML_SUCCESS)
        {
            log_error("Failed to shutdown NVML: {}", ret);
            success = false;
        }
        m_nvmlInitialized = false;
    }

    m_threadReadyToGo = false;
    m_eventSet        = nvmlSystemEventSet_t {};
    return success;
}

dcgmReturn_t DcgmNvmlSystemEventThread::WaitSystemEvent()
{
    if (!m_threadReadyToGo || (m_cacheManager && m_cacheManager->ShouldSkipDriverCalls()))
    {
        return DCGM_ST_PENDING;
    }
    nvmlSystemEventSetWaitRequest_t request = {};
    request.version                         = nvmlSystemEventSetWaitRequest_v1;
    request.timeoutms                       = 1000;
    request.set                             = m_eventSet;
    std::array<nvmlSystemEventData_v1_t, 2> dataArr;
    request.dataSize = dataArr.size();
    request.data     = dataArr.data();
    auto ret         = nvmlSystemEventSetWait(&request);
    if (ret == NVML_ERROR_TIMEOUT)
    {
        return DCGM_ST_OK;
    }

    if (ret != NVML_SUCCESS && ret != NVML_ERROR_INSUFFICIENT_SIZE)
    {
        log_error("Failed to wait for NVML system event: {}", ret);
        return DCGM_ST_NVML_ERROR;
    }

    if (ret == NVML_ERROR_INSUFFICIENT_SIZE)
    {
        // Try our best effort to purge the queue to avoid "ghost" events.
        const unsigned int maxAttempts = 32;
        for (unsigned int i = 0; i < maxAttempts; i++)
        {
            std::vector<nvmlSystemEventData_v1_t> eventBuf(request.numEvent);
            request.dataSize = eventBuf.size();
            request.data     = eventBuf.data();
            ret              = nvmlSystemEventSetWait(&request);
            if (ret == NVML_ERROR_TIMEOUT || ret == NVML_SUCCESS)
            {
                break;
            }
            if (ret == NVML_ERROR_INSUFFICIENT_SIZE)
            {
                continue;
            }
            if (ret != NVML_SUCCESS)
            {
                // even we failed to purge the queue, we should continue to process the event.
                // and hope that the next nvmlSystemEventSetWait will be processed correctly.
                log_error("Failed to purge NVML system event queue: {}", ret);
                break;
            }
        }
    }

    // We need to use the following steps to handle the bind / unbind event.
    //
    // 1. DcgmCacheManager::UninitializeNvmlEventSet (this includes nvmlEventSetFree)
    // 2. DcgmHostEngineHandler::ShutdownNvml (this includes nvmlShutdown)
    // 3. DcgmNvmlSystemEventThread::Shutdown (this includes nvmlShutdown)
    // 4. DcgmNvmlSystemEventThread::Init (this includes nvmlSystemRegisterEvents)
    // 5. DcgmHostEngineHandler::LoadNvml (this includes nvmlInit_v2)
    //
    // After step 1 to 3, the driver is detached from the system. These 3 steps are implemented in DetachDriverImpl.
    // After step 4 to 5, the driver is attached to the system. These 2 steps are implemented in AttachDriverImpl.
    //
    // Note that, we must do DcgmNvmlSystemEventThread::Init before DcgmHostEngineHandler::LoadNvml. This is because we
    // must ensure that nvmlSystemRegisterEvents is called before the normal nvml init. This way, we can avoid a
    // potential race conditions when events occur during nvml re-initialization.
    log_info("bind / unbind event received.");
    if (auto ret = DetachDriverImpl(); ret != DCGM_ST_OK)
    {
        log_error("Failed to detach driver: {}", ret);
        return ret;
    }
    if (auto ret = AttachDriverImpl(); ret != DCGM_ST_OK)
    {
        log_error("Failed to attach driver: {}", ret);
        return ret;
    }
    return DCGM_ST_OK;
}

std::optional<std::shared_future<dcgmReturn_t>> DcgmNvmlSystemEventThread::DetachDriver()
{
    return Enqueue(DcgmNs::make_task([this] { return DetachDriverImpl(); }));
}

dcgmReturn_t DcgmNvmlSystemEventThread::DetachDriverImpl()
{
    if (m_cacheManager)
    {
        m_cacheManager->RecordBindUnbindEvent(DcgmBUEventStateSystemReinitializing);
    }
    if (m_hostEngineHandler.IsNvmlLoaded())
    {
        m_hostEngineHandler.DetachGpusFromModules();
        if (m_cacheManager)
        {
            auto task = m_cacheManager->DetachDriver();
            if (!task.has_value())
            {
                log_error("Failed to create task for detach driver");
                return DCGM_ST_GENERIC_ERROR;
            }
            if (auto ret = task->get(); ret != DCGM_ST_OK)
            {
                log_error("Failed to process detach driver: {}", ret);
                return ret;
            }
        }
    }

    if (m_hostEngineHandler.IsNvmlLoaded())
    {
        m_hostEngineHandler.ShutdownNvml();
    }
    // We must shut down the NVML system event thread regardless of whether NVML is loaded, as it is initialized even if
    // `nvmlInit_v2` fails (e.g., due to no GPU).
    if (!Shutdown())
    {
        log_error("Failed to shutdown NVML system event thread");
        return DCGM_ST_GENERIC_ERROR;
    }
    return DCGM_ST_OK;
}

std::optional<std::shared_future<dcgmReturn_t>> DcgmNvmlSystemEventThread::AttachDriver()
{
    return Enqueue(DcgmNs::make_task([this] { return AttachDriverImpl(); }));
}

dcgmReturn_t DcgmNvmlSystemEventThread::AttachDriverImpl()
{
    auto ret = Init();
    if (ret != DCGM_ST_OK)
    {
        if (ret != DCGM_ST_NOT_SUPPORTED)
        {
            log_error("Failed to initialize NVML system event thread: {}", ret);
        }
        else
        {
            log_info("Skipping listening to NVML system events since it's not supported.");
        }
        m_threadReadyToGo = false;
    }

    if (m_hostEngineHandler.IsNvmlLoaded())
    {
        log_info("NVML is already loaded, skipping attach driver");
        return DCGM_ST_OK;
    }

    m_hostEngineHandler.LoadNvml();
    if (m_cacheManager)
    {
        auto task = m_cacheManager->AttachDriver();
        if (!task.has_value())
        {
            log_error("Failed to create task for attach driver");
            return DCGM_ST_GENERIC_ERROR;
        }
        if (auto ret = task->get(); ret != DCGM_ST_OK)
        {
            log_error("Failed to process attach driver: {}", ret);
            return ret;
        }
    }
    if (m_groupManager)
    {
        if (auto ret = m_groupManager->AttachGpus(); ret != DCGM_ST_OK)
        {
            log_error("Failed to attach GPUs to group manager, ret: {}", ret);
            return ret;
        }
    }
    m_hostEngineHandler.AttachGpusToModules();
    if (m_cacheManager)
    {
        m_cacheManager->RecordBindUnbindEvent(DcgmBUEventStateSystemReinitializationCompleted);
    }
    return DCGM_ST_OK;
}

void DcgmNvmlSystemEventThread::run(void)
{
    log_info("DcgmNvmlSystemEventThread started");

    if (m_monitor)
    {
        m_monitor->AddMonitoredTask(getpid(), m_tid);
    }

    // go back to wait system event immediately if there is no task to run.
    SetRunInterval(std::chrono::milliseconds(0));
    while (!ShouldStop())
    {
        if (auto ret = WaitSystemEvent(); ret != DCGM_ST_OK)
        {
            if (ret != DCGM_ST_PENDING)
            {
                log_error("Failed to wait for NVML system event: {}", ret);
            }
            else
            {
                log_debug("Failed to wait for NVML system event: pending");
            }
            Sleep(1000000);
        }
        if (TaskRunner::Run(true) != TaskRunner::RunResult::Ok)
        {
            break;
        }
    }

    if (m_monitor)
    {
        m_monitor->RemoveMonitoredTask(getpid(), m_tid);
    }
    log_info("DcgmNvmlSystemEventThread ended");
}

void DcgmNvmlSystemEventThread::SetCacheManager(DcgmCacheManager *cacheManager)
{
    m_cacheManager = cacheManager;
}

void DcgmNvmlSystemEventThread::SetGroupManager(DcgmGroupManager *groupManager)
{
    m_groupManager = groupManager;
}

/*****************************************************************************/

void EngineHangDetectHandler::SetTaskContextManager(class TaskContextManager const *taskCtxMgr)
{
    m_taskCtxMgr = taskCtxMgr;
}

void EngineHangDetectHandler::HandleHangDetectedEvent(HangDetectHandler::HangDetectedEvent const &hangEvent)
{
    auto const [pid, tid, duration] = hangEvent;
    auto const taskType             = tid.has_value() ? "thread" : "process";
    auto const taskId               = tid.value_or(pid);
    auto const idType               = tid.has_value() ? "tid" : "pid";

    if (!m_taskCtxMgr)
    {
        log_verbose("Task context manager is not set, skipping hang detected event for {}: {}", idType, taskId);
        return;
    }

    // Only check for task context membership when checking a thread.
    if (tid.has_value() && !m_taskCtxMgr->isTaskIncluded(taskId))
    {
        log_verbose("Task is not registered, skipping hang detected event for {}: {}", idType, taskId);
        return;
    }

    auto const msg
        = fmt::format("A {} ({}: {}) has been unresponsive for {} seconds. "
                      "If the process does not exit, it may need to be killed or the system may need to be restarted. "
                      "Collect problem report logs before restarting.",
                      taskType,
                      idType,
                      taskId,
                      duration.count());

    log_error(msg);

    if (m_terminateOnHang)
    {
        throw std::runtime_error(msg);
    }
}

/*****************************************************************************
 Constructor for DCGM Host Engine Handler
 *****************************************************************************/
DcgmHostEngineHandler::DcgmHostEngineHandler(dcgmStartEmbeddedV2Params_v1 params)
    : m_communicator()
    , m_hostengineHealth(0)
    , m_usingInjectionNvml(false)
{
    int ret;
    dcgmReturn_t dcgmRet;

    InitializeHangDetection();

    m_dcgmIpc = std::make_unique<DcgmIpc>(DCGM_HE_NUM_WORKERS, m_monitor.get());

    mpCacheManager = nullptr;

    /* Set this in case a child class calls our Instance() method during this constructor.
       We really need to move away from the singleton model */
    mpHostEngineHandlerInstance = this;

    m_nextWatchedRequestId = 1;

    memset(&m_modules, 0, sizeof(m_modules));
    /* Do explicit initialization of the modules */
    for (unsigned int i = 0; i < DcgmModuleIdCount; i++)
    {
        m_modules[i].id     = (dcgmModuleId_t)i;
        m_modules[i].status = DcgmModuleStatusNotLoaded;
    }
    /* Core module is always loaded */
    m_modules[DcgmModuleIdCore].status = DcgmModuleStatusLoaded;
    m_modules[DcgmModuleIdCore].ptr    = &mModuleCoreObj;
    m_modules[DcgmModuleIdCore].msgCB  = mModuleCoreObj.GetMessageProcessingCallback();
    /* Set module filenames */
    m_modules[DcgmModuleIdNvSwitch].filename   = "libdcgmmodulenvswitch.so.4";
    m_modules[DcgmModuleIdVGPU].filename       = "libdcgmmodulevgpu.so.4";
    m_modules[DcgmModuleIdIntrospect].filename = "libdcgmmoduleintrospect.so.4";
    m_modules[DcgmModuleIdHealth].filename     = "libdcgmmodulehealth.so.4";
    m_modules[DcgmModuleIdPolicy].filename     = "libdcgmmodulepolicy.so.4";
    m_modules[DcgmModuleIdConfig].filename     = "libdcgmmoduleconfig.so.4";
    m_modules[DcgmModuleIdDiag].filename       = "libdcgmmodulediag.so.4";
    m_modules[DcgmModuleIdProfiling].filename  = "libdcgmmoduleprofiling.so.4";
    m_modules[DcgmModuleIdSysmon].filename     = "libdcgmmodulesysmon.so.4";
    m_modules[DcgmModuleIdMnDiag].filename     = "libdcgmmodulemndiag.so.4";

    /* Apply the denylist that was requested before we possibly load any modules */
    dcgmReturn_t result = ApplyModuleDenylist(params.denyList, params.denyListCount);
    if (result != DCGM_ST_OK)
    {
        throw std::runtime_error("Failed to apply module denylist.");
    }

    m_nvmlSystemEventThread = std::make_unique<DcgmNvmlSystemEventThread>(*this);

    LoadNvml();

    if (m_nvmlLoaded.load(std::memory_order_acquire))
    {
        char driverVersion[80];
        nvmlSystemGetDriverVersion(driverVersion, 80);
        if (strcmp(driverVersion, DCGM_MIN_DRIVER_VERSION) < 0)
        {
            throw std::runtime_error("Driver " + std::string(driverVersion) + " is unsupported. Must be at least "
                                     + std::string(DCGM_MIN_DRIVER_VERSION) + ".");
        }
    }

    ret = DcgmFieldsInit();
    if (ret != 0)
    {
        std::stringstream ss;
        ss << "DCGM Fields Init Failed. Error: " << ret;
        throw std::runtime_error(ss.str());
    }

    unsigned int nvmlDeviceCount = 0;
    if (m_nvmlLoaded.load(std::memory_order_acquire))
    {
        nvmlReturn_t nvmlSt = nvmlDeviceGetCount_v2(&nvmlDeviceCount);
        if (nvmlSt != NVML_SUCCESS)
        {
            throw std::runtime_error(fmt::format("Unable to get the NVML device count. NVML Error: {}", nvmlSt));
        }

        if (nvmlDeviceCount > DCGM_MAX_NUM_DEVICES)
        {
            throw std::runtime_error(fmt::format("DCGM only supports up to {} GPUs. {} GPUs were found in the system.",
                                                 DCGM_MAX_NUM_DEVICES,
                                                 nvmlDeviceCount));
        }
    }

    mpCacheManager = new DcgmCacheManager();
    m_nvmlSystemEventThread->SetCacheManager(mpCacheManager);
    m_nvmlSystemEventThread->SetHangDetectMonitor(m_monitor.get());
    mModuleCoreObj.Initialize(mpCacheManager);

    mpCacheManager->SetHangDetectMonitor(m_monitor.get());

    /* Don't do anything before you call mpCacheManager->Init(). Note: the specified maxSampleAge is ignored. */

    if (params.opMode == DCGM_OPERATION_MODE_AUTO)
    {
        ret = mpCacheManager->Init(0, 86400.0, m_nvmlLoaded.load(std::memory_order_acquire));
        if (ret != 0)
        {
            std::stringstream ss;
            ss << "CacheManager Init Failed. Error: " << ret;
            throw std::runtime_error(ss.str());
        }
    }
    else
    {
        ret = mpCacheManager->Init(1, 14400.0, m_nvmlLoaded.load(std::memory_order_acquire));
        if (ret != 0)
        {
            std::stringstream ss;
            ss << "CacheManager Init Failed. Error: " << ret;
            throw std::runtime_error(ss.str());
        }
    }

    dcgmcmEventSubscription_t fv  = {};
    dcgmcmEventSubscription_t mig = {};
    fv.type                       = DcgmcmEventTypeFvUpdate;
    fv.fn.fvCb                    = nvHostEngineFvCallback;
    fv.userData                   = this;

    dcgmRet = mpCacheManager->SubscribeForEvent(fv);
    if (dcgmRet != DCGM_ST_OK)
    {
        throw std::runtime_error("DCGM was unable to subscribe for cache manager updates.");
    }

    mig.type     = DcgmcmEventTypeMigReconfigure;
    mig.fn.migCb = nvHostEngineMigCallback;
    mig.userData = this;

    dcgmRet = mpCacheManager->SubscribeForEvent(mig);
    if (dcgmRet != DCGM_ST_OK)
    {
        throw std::runtime_error("DCGM was unable to subscribe for mig reconfiguration updates.");
    }

    /* Initialize the group manager before we add our default watches */
    mpGroupManager = new DcgmGroupManager(mpCacheManager, false);
    mpGroupManager->SubscribeForGroupEvents(HostEngineOnGroupEventCB, this);
    m_nvmlSystemEventThread->SetGroupManager(mpGroupManager);
    mModuleCoreObj.SetGroupManager(mpGroupManager);

    mpFieldGroupManager = new DcgmFieldGroupManager();

    m_childProcessManager.SetHangDetectMonitor(m_monitor.get());

    m_communicator.Init(mpCacheManager, mpGroupManager);
    m_coreCallbacks.postfunc   = PostRequestToCore;
    m_coreCallbacks.poster     = &m_communicator;
    m_coreCallbacks.version    = dcgmCoreCallbacks_version;
    m_coreCallbacks.loggerfunc = (dcgmLoggerCallback_f)DcgmLoggingGetCallback();

    /* Create default groups after we've set up core callbacks. This is because creating
       default groups causes the NvSwitch module to load, which in turn tries to ask m_coreCallbacks
       for its logging severity. Hence we are putting this code after the initialization
       of m_coreCallbacks above */
    dcgmRet = mpGroupManager->CreateDefaultGroups();
    if (dcgmRet != DCGM_ST_OK)
    {
        throw std::runtime_error("DCGM was unable to create default groups for the group manager.");
    }

    /* Watch internal fields before we start the cache manager update thread */
    dcgmRet = WatchHostEngineFields();
    if (dcgmRet != 0)
    {
        throw std::runtime_error("WatchHostEngineFields failed.");
    }

    /* Start the cache manager update thread */
    ret = mpCacheManager->Start();
    if (ret != 0)
    {
        std::stringstream ss;
        ss << "CacheManager Start Failed. Error: " << ret;
        throw std::runtime_error(ss.str());
    }

    /* Wait for a round of updates to occur so that we can safely query values */
    ret = mpCacheManager->UpdateAllFields(1);
    if (ret != 0)
    {
        std::stringstream ss;
        ss << "CacheManager UpdateAllFields. Error: " << ret;
        throw std::runtime_error(ss.str());
    }

    ret = m_nvmlSystemEventThread->Start();
    if (ret != 0)
    {
        log_error("NVML System Event Thread Start Failed. Error: {}", ret);
    }
}

/*****************************************************************************
 Destructor for DCGM Host Engine Handler
 *****************************************************************************/
DcgmHostEngineHandler::~DcgmHostEngineHandler()
{
    /* Make sure that server is stopped first so that no new connections or
     * requests are accepted by the HostEngine.
     * Always keep this first */
    try
    {
        m_dcgmIpc->StopAndWait(60000);
    }
    catch (std::exception const &ex)
    {
        DCGM_LOG_ERROR << "Exception caught in DcgmHostEngineHandler::~DcgmHostEngineHandler(): " << ex.what();
    }
    catch (...)
    {
        DCGM_LOG_ERROR << "Unknown exception caught in DcgmHostEngineHandler::~DcgmHostEngineHandler()";
    }

    if (auto ret = m_nvmlSystemEventThread->StopAndWait(45000); ret != 0)
    {
        log_error("Failed to stop NVML system event thread: {}", ret);
    }

    auto lock = Lock();
    /**
     * Free sub-modules before we unload core modules.
     *
     * Wait until after ~CacheManager() to close the modules.
     */
    UnloadModules();
    deleteNotNull(mpCacheManager);
    deleteNotNull(mpFieldGroupManager);

    for (auto &m_module : m_modules)
    {
        /* now that modules and CacheManager are stopped, close the modules */
        if (m_module.dlopenPtr != nullptr)
        {
            dlclose(m_module.dlopenPtr);
            m_module.dlopenPtr = nullptr;
        }
    }

    // DcgmFieldsTerm(); //Not doing this for now due to bug 1787570, comment 1

    /* Remove all the connections. Keep it after modules referencing the connections */
    deleteNotNull(mpGroupManager);

    /* Remove lingering tracked rquests */
    RemoveAllTrackedRequests();

    ShutdownNvml();
}

dcgmReturn_t DcgmHostEngineHandler::SendModuleMessage(dcgmModuleId_t id, dcgm_module_command_header_t *moduleCommand)
{
    if (moduleCommand == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }

    if ((m_modules[id].ptr == nullptr) || (m_modules[id].msgCB == nullptr))
    {
        return DCGM_ST_BADPARAM;
    }

    return m_modules[id].msgCB(m_modules[id].ptr, moduleCommand);
}

namespace
{
char const *ModuleIdToName(dcgmModuleId_t moduleId)
{
    char const *moduleName;
    if (auto ret = dcgmModuleIdToName(moduleId, &moduleName); ret != DCGM_ST_OK)
    {
        return "Unknown";
    }
    return moduleName;
}
} //namespace

/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::LoadModule(dcgmModuleId_t moduleId)
{
    if (moduleId <= DcgmModuleIdCore || moduleId >= DcgmModuleIdCount)
    {
        log_error("Invalid moduleId {}", moduleId);
        return DCGM_ST_BADPARAM;
    }

    /* Is the module marked reloadable */
    if (m_modules[moduleId].status == DcgmModuleStatusReloadable)
    {
        m_modules[moduleId].status = DcgmModuleStatusLoaded;

        /* We fall through to make sure it is loaded. */
    }

    /* Is the module already loaded? */
    if (m_modules[moduleId].ptr != nullptr)
    {
        return DCGM_ST_OK;
    }

    if (m_modules[moduleId].status == DcgmModuleStatusDenylisted || m_modules[moduleId].status == DcgmModuleStatusFailed
        || m_modules[moduleId].status == DcgmModuleStatusUnloaded)
    {
        log_warning("Skipping loading of module {} in status {}", moduleId, m_modules[moduleId].status);
        return DCGM_ST_MODULE_NOT_LOADED;
    }

    /* Get the lock so we don't try to load the module from two threads */

    auto lock = Lock();

    if (m_modules[moduleId].ptr != nullptr)
    {
        /* Module was loaded by another thread while we were getting the lock */
        return DCGM_ST_OK;
    }

    /* Do we have a library name to open? */
    if (m_modules[moduleId].filename == nullptr)
    {
        m_modules[moduleId].status = DcgmModuleStatusFailed;
        log_error("Failed to load module {} - no filename", moduleId);
        return DCGM_ST_MODULE_NOT_LOADED;
    }

    /* Try to load the library */

    // If GPUs are detached, we abort module initialization
    if (m_gpusDetached.load(std::memory_order_acquire))
    {
        log_error("GPUs are detached, module {} cannot be loaded", moduleId);
        return DCGM_ST_GPUS_DETACHED;
    }

    {
        /* Lock hostengine logging severity to avoid module severity falling out
         * of sync with hostengine severity */
        std::unique_lock<std::mutex> loggingSeverityLock = LoggerLockSeverity();
        m_modules[moduleId].dlopenPtr                    = dlopen(m_modules[moduleId].filename, RTLD_NOW);
    }

    if (m_modules[moduleId].dlopenPtr == nullptr)
    {
        m_modules[moduleId].status = DcgmModuleStatusFailed;
        log_error(
            "Failed to load module {} - dlopen({}) returned: {}", moduleId, m_modules[moduleId].filename, dlerror());
        return DCGM_ST_MODULE_NOT_LOADED;
    }

    /* Get all of the function pointers we need */
    m_modules[moduleId].allocCB = (dcgmModuleAlloc_f)dlsym(m_modules[moduleId].dlopenPtr, "dcgm_alloc_module_instance");
    m_modules[moduleId].freeCB  = (dcgmModuleFree_f)dlsym(m_modules[moduleId].dlopenPtr, "dcgm_free_module_instance");
    m_modules[moduleId].msgCB
        = (dcgmModuleProcessMessage_f)dlsym(m_modules[moduleId].dlopenPtr, "dcgm_module_process_message");
    if ((m_modules[moduleId].allocCB == nullptr) || (m_modules[moduleId].freeCB == nullptr)
        || (m_modules[moduleId].msgCB == nullptr))
    {
        log_error(
            "dcgm_alloc_module_instance ({}), dcgm_free_module_instance ({}), or dcgm_module_process_message ({}) was missing from {}",
            (void *)m_modules[moduleId].allocCB,
            (void *)m_modules[moduleId].freeCB,
            (void *)m_modules[moduleId].msgCB,
            m_modules[moduleId].filename);
        m_modules[moduleId].status = DcgmModuleStatusFailed;
        dlclose(m_modules[moduleId].dlopenPtr);
        m_modules[moduleId].dlopenPtr = nullptr;
        return DCGM_ST_MODULE_NOT_LOADED;
    }

    /* Call the constructor (finally). Note that constructors can throw runtime errors. We should treat that as
       the constructor failing and mark the module as failing to load. */
    try
    {
        m_modules[moduleId].ptr = m_modules[moduleId].allocCB(&m_coreCallbacks);
    }
    catch (const std::runtime_error &e)
    {
        log_error("Caught std::runtime error from allocCB()");
        /* m_modules[moduleId].ptr will remain null, which is handled below */
    }

    if (m_modules[moduleId].ptr == nullptr)
    {
        m_modules[moduleId].status = DcgmModuleStatusFailed;
        dlclose(m_modules[moduleId].dlopenPtr);
        m_modules[moduleId].dlopenPtr = nullptr;
        log_error("Failed to load module {}", moduleId);
    }
    else
    {
        m_modules[moduleId].status = DcgmModuleStatusLoaded;
        log_info("Loaded module {}", moduleId);
    }

    if (m_modules[moduleId].status == DcgmModuleStatusLoaded)
    {
        /*
         * If the DCGM was paused, we instantly pause any loaded module.
         * If pausing the module fails, we unload the module as it's dangerous to leave in a loaded and running
         * state when the rest of the DCGM is supposed to be paused.
         */
        if (m_modules[DcgmModuleIdCore].status == DcgmModuleStatusPaused)
        {
            if (auto ret = PauseModule(moduleId); ret != DCGM_ST_OK)
            {
                char const *moduleName = ModuleIdToName(moduleId);
                log_error(
                    "Unable to pause the module ({}){} on load: ({}){}", moduleId, moduleName, ret, errorString(ret));
                log_error("The module ({}){} will be unloaded", moduleId, moduleName);

                m_modules[moduleId].status = DcgmModuleStatusFailed;
                dlclose(m_modules[moduleId].dlopenPtr);
                m_modules[moduleId].dlopenPtr = nullptr;
                return DCGM_ST_MODULE_NOT_LOADED;
            }
        }
        return DCGM_ST_OK;
    }

    return DCGM_ST_MODULE_NOT_LOADED;
}


/*****************************************************************************/
DcgmLockGuard DcgmHostEngineHandler::Lock()
{
    auto result = DcgmLockGuard(&m_lock);
    return result;
}

/*****************************************************************************/
void DcgmHostEngineHandler::Unlock(DcgmLockGuard /*guard*/)
{
    /*
     * This function is intentionally empty as the whole logic is to take the DcgmLockGuard ownership and call its
     * destructor. To do so, it's enough to get the DcgmLockGuard by value as an argument.
     * The DcgmLockGuard is not copyable and the only way to call this function is not move previously acquired lock
     * into this function like Unlock(std::move(lock));
     *
     * This functionality is needed to allow releasing a lock inside a scope earlier than the lock drops out of
     * scope and destroyed automatically.
     */
}

/*
 * Pass pointer by reference so that we can set it to NULL afterwards
 */
template <typename T>
void DcgmHostEngineHandler::deleteNotNull(T *&obj)
{
    if (nullptr != obj)
    {
        delete obj;
        obj = nullptr;
    }
}

/*****************************************************************************
 This method initializes and returns the singleton instance to DCGM Host Engine Handler
 *****************************************************************************/
DcgmHostEngineHandler *DcgmHostEngineHandler::Init(dcgmStartEmbeddedV2Params_v1 params)
{
    if (mpHostEngineHandlerInstance == nullptr)
    {
        try
        {
            mpHostEngineHandlerInstance = new DcgmHostEngineHandler(params);
        }
        catch (const std::runtime_error &e)
        {
            DCGM_LOG_ERROR << "Cannot initialize the hostengine: " << e.what();
            fprintf(stderr, "%s\n", e.what());
            /* Don't delete here. It wasn't allocated if we got an exception */
            mpHostEngineHandlerInstance = nullptr;
            return nullptr;
        }
    }
    return mpHostEngineHandlerInstance;
}

/*****************************************************************************
 This method returns the singleton instance to DCGM Host Engine Handler
 *****************************************************************************/
DcgmHostEngineHandler *DcgmHostEngineHandler::Instance()
{
    return mpHostEngineHandlerInstance;
}

/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::GetDcgmGpuIds(std::vector<unsigned int> &gpuIds, int onlySupported)
{
    return (dcgmReturn_t)mpCacheManager->GetGpuIds(onlySupported, gpuIds);
}


/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::GetDcgmGpuArch(dcgm_field_eid_t entityId, dcgmChipArchitecture_t &arch)
{
    return (dcgmReturn_t)mpCacheManager->GetGpuArch(entityId, arch);
}

/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::GetLatestSample(dcgm_field_entity_group_t entityGroupId,
                                                    dcgm_field_eid_t entityId,
                                                    unsigned short dcgmFieldId,
                                                    dcgmcm_sample_p sample)
{
    return (dcgmReturn_t)mpCacheManager->GetLatestSample(entityGroupId, entityId, dcgmFieldId, sample, nullptr);
}

/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::HelperGetInt64StatSummary(dcgm_field_entity_group_t entityGroupId,
                                                              dcgm_field_eid_t entityId,
                                                              unsigned short fieldId,
                                                              dcgmStatSummaryInt64_t *summary,
                                                              long long startTime,
                                                              long long endTime)
{
    dcgmReturn_t dcgmReturn;
    DcgmcmSummaryType_t summaryTypes[DcgmcmSummaryTypeSize];
    long long summaryValues[DcgmcmSummaryTypeSize];

    int numSummaryTypes = 3; /* Should match count below */
    summaryTypes[0]     = DcgmcmSummaryTypeMinimum;
    summaryTypes[1]     = DcgmcmSummaryTypeMaximum;
    summaryTypes[2]     = DcgmcmSummaryTypeAverage;

    dcgmReturn = mpCacheManager->GetInt64SummaryData(entityGroupId,
                                                     entityId,
                                                     fieldId,
                                                     numSummaryTypes,
                                                     summaryTypes,
                                                     summaryValues,
                                                     startTime,
                                                     endTime,
                                                     nullptr,
                                                     nullptr);
    if (dcgmReturn != DCGM_ST_OK)
    {
        return dcgmReturn;
    }

    /* Should be same indexes as summaryTypes assignments above */
    summary->minValue = summaryValues[0];
    summary->maxValue = summaryValues[1];
    summary->average  = summaryValues[2];

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::HelperGetInt32StatSummary(dcgm_field_entity_group_t entityGroupId,
                                                              dcgm_field_eid_t entityId,
                                                              unsigned short fieldId,
                                                              dcgmStatSummaryInt32_t *summary,
                                                              long long startTime,
                                                              long long endTime)
{
    dcgmStatSummaryInt64_t summary64;

    dcgmReturn_t dcgmReturn
        = HelperGetInt64StatSummary(entityGroupId, entityId, fieldId, &summary64, startTime, endTime);
    if (dcgmReturn != DCGM_ST_OK)
    {
        return dcgmReturn;
    }

    summary->average  = nvcmvalue_int64_to_int32(summary64.average);
    summary->maxValue = nvcmvalue_int64_to_int32(summary64.maxValue);
    summary->minValue = nvcmvalue_int64_to_int32(summary64.minValue);
    return DCGM_ST_OK;
}

/*************************************************************************************/
/* Helper to fill destPids[] with unique entries from srcPids it doesn't have already */
static void mergeUniquePids(unsigned int *destPids,
                            int *destPidsSize,
                            int maxDestPids,
                            const unsigned int *const srcPids,
                            int srcPidsSize)
{
    int i;
    int j;
    bool havePid;

    if ((*destPidsSize) >= maxDestPids)
    {
        return; /* destPids is already full */
    }

    for (i = 0; i < srcPidsSize; i++)
    {
        havePid = false;
        for (j = 0; j < (*destPidsSize); j++)
        {
            if (srcPids[i] == destPids[j])
            {
                havePid = true;
                break;
            }
        }

        if (havePid)
        {
            continue;
        }

        destPids[*destPidsSize] = srcPids[i];
        (*destPidsSize)++;

        if ((*destPidsSize) >= maxDestPids)
        {
            return; /* destPids is full */
        }
    }
}

/*************************************************************************************/
/* Helper to fill destPidInfo[] with unique entries from srcPidInfo it doesn't have already */

static void mergeUniquePidInfo(dcgmProcessUtilInfo_t *destPidInfo,
                               int *destPidInfoSize,
                               int maxDestPidInfo,
                               dcgmProcessUtilInfo_t *srcPidInfo,
                               int srcPidInfoSize)
{
    int i;
    int j;
    bool havePid;

    if ((*destPidInfoSize) >= maxDestPidInfo)
    {
        return; /* destPids is already full */
    }

    for (i = 0; i < srcPidInfoSize; i++)
    {
        havePid = false;
        for (j = 0; j < (*destPidInfoSize); j++)
        {
            if (srcPidInfo[i].pid == destPidInfo[j].pid)
            {
                havePid = true;
                break;
            }
        }

        if (havePid)
        {
            continue;
        }

        destPidInfo[*destPidInfoSize].pid     = srcPidInfo[i].pid;
        destPidInfo[*destPidInfoSize].smUtil  = srcPidInfo[i].smUtil;
        destPidInfo[*destPidInfoSize].memUtil = srcPidInfo[i].memUtil;
        (*destPidInfoSize)++;

        if ((*destPidInfoSize) >= maxDestPidInfo)
        {
            return; /* destPids is full */
        }
    }
}


/*************************************************************************************/
/* Helper to find and fill the Utilization rates in pidInfo for the pid in pidInfo*/

static void findPidUtilInfo(dcgmProcessUtilSample_t *smUtil,
                            unsigned int numSmUtilVal,
                            dcgmProcessUtilSample_t *memUtil,
                            unsigned int numMemUtilVal,
                            dcgmProcessUtilInfo_t *pidInfo)
{
    unsigned int smUtilIter  = 0;
    unsigned int memUtilIter = 0;
    bool pidFound            = false;

    /* Copy the SM Util first*/
    for (smUtilIter = 0; smUtilIter < numSmUtilVal; smUtilIter++)
    {
        if (pidInfo->pid == smUtil[smUtilIter].pid)
        {
            pidInfo->smUtil = smUtil[smUtilIter].util;
            pidFound        = true;
            break;
        }
    }

    if (!pidFound)
    {
        pidInfo->smUtil = DCGM_INT32_BLANK;
    }

    /* Reset pidFound Variable */
    pidFound = false;

    /* Update the Mem Util */
    for (memUtilIter = 0; memUtilIter < numMemUtilVal; memUtilIter++)
    {
        if (pidInfo->pid == memUtil[memUtilIter].pid)
        {
            pidInfo->memUtil = memUtil[memUtilIter].util;
            pidFound         = true;
            break;
        }
    }

    if (!pidFound)
    {
        pidInfo->memUtil = DCGM_INT32_BLANK;
    }
}


/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::GetProcessInfo(unsigned int groupId, dcgmPidInfo_t *pidInfo)
{
    std::vector<dcgmGroupEntityPair_t> entities;
    std::vector<unsigned int> gpuIds;
    std::vector<unsigned int>::iterator gpuIdIt;
    dcgmPidSingleInfo_t *singleInfo;
    dcgmReturn_t dcgmReturn = DCGM_ST_OK;
    dcgmDevicePidAccountingStats_t accountingInfo;
    long long startTime;
    long long endTime;
    long long i64Val;
    DcgmcmSummaryType_t summaryTypes[DcgmcmSummaryTypeSize];
    int i;
    double doubleVal;
    const int Msamples = 10; /* Should match size of samples[] */
    dcgmcm_sample_t samples[Msamples];
    dcgmStatSummaryInt32_t blankSummary32 = { DCGM_INT32_BLANK, DCGM_INT32_BLANK, DCGM_INT32_BLANK };
    dcgmStatSummaryInt64_t blankSummary64 = { DCGM_INT64_BLANK, DCGM_INT64_BLANK, DCGM_INT64_BLANK };

    if (!m_nvmlLoaded.load(std::memory_order_acquire))
    {
        log_debug("Cannot get process stats: NVML is not loaded");
        return DCGM_ST_NVML_NOT_LOADED;
    }

    /* Sanity check the incoming parameters */
    if (pidInfo->pid == 0)
    {
        log_warning("No PID provided in request");
        return DCGM_ST_BADPARAM;
    }

    if (pidInfo->version != dcgmPidInfo_version)
    {
        log_warning("Version mismatch. expected {}. Got {}", dcgmPidInfo_version, pidInfo->version);
        return DCGM_ST_VER_MISMATCH;
    }


    /* Verify group id is valid */
    dcgmReturn = mpGroupManager->verifyAndUpdateGroupId(&groupId);
    if (DCGM_ST_OK != dcgmReturn)
    {
        log_error("Error: Bad group id parameter");
        return dcgmReturn;
    }

    /* Resolve the groupId -> entities[] -> gpuIds[] */
    dcgmReturn = mpGroupManager->GetGroupEntities(groupId, DcgmGroupOption::All, entities);
    if (dcgmReturn != DCGM_ST_OK)
    {
        log_error("Error {} from GetGroupEntities()", (int)dcgmReturn);
        return dcgmReturn;
    }

    /* Process stats are only supported for GPUs for now */
    for (i = 0; i < (int)entities.size(); i++)
    {
        if (entities[i].entityGroupId != DCGM_FE_GPU)
        {
            continue;
        }

        gpuIds.push_back(entities[i].entityId);
    }

    /* Prepare a health response to be populated once we have startTime and endTime */
    std::unique_ptr<dcgmHealthResponse_v5> response = std::make_unique<dcgmHealthResponse_v5>();
    memset(response.get(), 0, sizeof(*response));

    /* Zero the structures */
    memset(&pidInfo->gpus[0], 0, sizeof(pidInfo->gpus));
    memset(&pidInfo->summary, 0, sizeof(pidInfo->summary));
    pidInfo->numGpus = 0;

    /* Initialize summary information */
    pidInfo->summary.pcieRxBandwidth      = blankSummary64;
    pidInfo->summary.pcieTxBandwidth      = blankSummary64;
    pidInfo->summary.powerViolationTime   = DCGM_INT64_NOT_SUPPORTED;
    pidInfo->summary.thermalViolationTime = DCGM_INT64_NOT_SUPPORTED;
    pidInfo->summary.energyConsumed       = DCGM_INT64_BLANK;
    pidInfo->summary.pcieReplays          = 0;
    pidInfo->summary.smUtilization        = blankSummary32;
    pidInfo->summary.memoryUtilization    = blankSummary32;
    pidInfo->summary.eccSingleBit         = DCGM_INT32_BLANK;
    pidInfo->summary.eccDoubleBit         = 0;
    pidInfo->summary.memoryClock          = blankSummary32;
    pidInfo->summary.smClock              = blankSummary32;

    for (gpuIdIt = gpuIds.begin(); gpuIdIt != gpuIds.end(); ++gpuIdIt)
    {
        singleInfo        = &pidInfo->gpus[pidInfo->numGpus];
        singleInfo->gpuId = *gpuIdIt;

        dcgmReturn = mpCacheManager->GetLatestProcessInfo(singleInfo->gpuId, pidInfo->pid, &accountingInfo);
        if (dcgmReturn == DCGM_ST_NO_DATA)
        {
            log_debug("Pid {} did not run on gpuId {}", pidInfo->pid, singleInfo->gpuId);
            continue;
        }

        if (dcgmReturn == DCGM_ST_NOT_WATCHED)
        {
            log_debug("Fields are not watched. Cannot get info for pid {} on GPU {}", pidInfo->pid, singleInfo->gpuId);
            continue;
        }

        /* Increment GPU count now that we know the process ran on this GPU */
        pidInfo->numGpus++;

        startTime = (long long)accountingInfo.startTimestamp;
        if (pidInfo->summary.startTime == 0 || startTime < pidInfo->summary.startTime)
        {
            pidInfo->summary.startTime = startTime;
        }


        if (0 == accountingInfo.activeTimeUsec) // Implies that the process is still running
        {
            endTime                  = (long long)timelib_usecSince1970();
            pidInfo->summary.endTime = 0; // Set end-time to 0 if the process is act
        }
        else
        {
            endTime                  = (long long)(accountingInfo.startTimestamp + accountingInfo.activeTimeUsec);
            pidInfo->summary.endTime = endTime;
        }

        singleInfo->startTime = pidInfo->summary.startTime;
        singleInfo->endTime   = pidInfo->summary.endTime;

        /* See if the energy counter is supported. If so, use that rather than integrating the power usage */
        summaryTypes[0] = DcgmcmSummaryTypeDifference;
        mpCacheManager->GetInt64SummaryData(DCGM_FE_GPU,
                                            singleInfo->gpuId,
                                            DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION,
                                            1,
                                            &summaryTypes[0],
                                            &i64Val,
                                            startTime,
                                            endTime,
                                            nullptr,
                                            nullptr);
        if (!DCGM_INT64_IS_BLANK(i64Val))
        {
            singleInfo->energyConsumed = i64Val;
        }
        else
        {
            /* No energy counter. Integrate power usage */
            log_debug("No energy counter. Using power_usage");
            summaryTypes[0] = DcgmcmSummaryTypeIntegral;
            mpCacheManager->GetFp64SummaryData(DCGM_FE_GPU,
                                               singleInfo->gpuId,
                                               DCGM_FI_DEV_POWER_USAGE,
                                               1,
                                               &summaryTypes[0],
                                               &doubleVal,
                                               startTime,
                                               endTime,
                                               nullptr,
                                               nullptr);
            if (!DCGM_FP64_IS_BLANK(doubleVal))
            {
                doubleVal /= 1000.0; /* convert from usec watts to milliwatt seconds */
            }
            singleInfo->energyConsumed = nvcmvalue_double_to_int64(doubleVal);
        }

        /* Update summary value, handling blank case */
        if (!DCGM_INT64_IS_BLANK(singleInfo->energyConsumed))
        {
            if (!DCGM_INT64_IS_BLANK(pidInfo->summary.energyConsumed))
            {
                pidInfo->summary.energyConsumed += singleInfo->energyConsumed;
            }
            else
            {
                pidInfo->summary.energyConsumed = singleInfo->energyConsumed;
            }
        }

        HelperGetInt64StatSummary(DCGM_FE_GPU,
                                  singleInfo->gpuId,
                                  DCGM_FI_DEV_PCIE_RX_THROUGHPUT,
                                  &singleInfo->pcieRxBandwidth,
                                  startTime,
                                  endTime);
        HelperGetInt64StatSummary(DCGM_FE_GPU,
                                  singleInfo->gpuId,
                                  DCGM_FI_DEV_PCIE_TX_THROUGHPUT,
                                  &singleInfo->pcieTxBandwidth,
                                  startTime,
                                  endTime);

        summaryTypes[0] = DcgmcmSummaryTypeMaximum;
        mpCacheManager->GetInt64SummaryData(DCGM_FE_GPU,
                                            singleInfo->gpuId,
                                            DCGM_FI_DEV_PCIE_REPLAY_COUNTER,
                                            1,
                                            &summaryTypes[0],
                                            &singleInfo->pcieReplays,
                                            startTime,
                                            endTime,
                                            nullptr,
                                            nullptr);
        if (!DCGM_INT64_IS_BLANK(singleInfo->pcieReplays))
        {
            pidInfo->summary.pcieReplays += singleInfo->pcieReplays;
        }


        HelperGetInt32StatSummary(
            DCGM_FE_GPU, singleInfo->gpuId, DCGM_FI_DEV_GPU_UTIL, &singleInfo->smUtilization, startTime, endTime);
        HelperGetInt32StatSummary(DCGM_FE_GPU,
                                  singleInfo->gpuId,
                                  DCGM_FI_DEV_MEM_COPY_UTIL,
                                  &singleInfo->memoryUtilization,
                                  startTime,
                                  endTime);

        summaryTypes[0] = DcgmcmSummaryTypeMaximum;
        mpCacheManager->GetInt64SummaryData(DCGM_FE_GPU,
                                            singleInfo->gpuId,
                                            DCGM_FI_DEV_ECC_DBE_VOL_TOTAL,
                                            1,
                                            &summaryTypes[0],
                                            &i64Val,
                                            startTime,
                                            endTime,
                                            nullptr,
                                            nullptr);
        singleInfo->eccDoubleBit = nvcmvalue_int64_to_int32(i64Val);
        if (!DCGM_INT32_IS_BLANK(singleInfo->eccDoubleBit))
        {
            pidInfo->summary.eccDoubleBit += singleInfo->eccDoubleBit;
        }

        HelperGetInt32StatSummary(
            DCGM_FE_GPU, singleInfo->gpuId, DCGM_FI_DEV_SM_CLOCK, &singleInfo->smClock, startTime, endTime);

        HelperGetInt32StatSummary(
            DCGM_FE_GPU, singleInfo->gpuId, DCGM_FI_DEV_MEM_CLOCK, &singleInfo->memoryClock, startTime, endTime);

        singleInfo->numXidCriticalErrors = Msamples;
        dcgmReturn                       = mpCacheManager->GetSamples(DCGM_FE_GPU,
                                                singleInfo->gpuId,
                                                DCGM_FI_DEV_XID_ERRORS,
                                                samples,
                                                &singleInfo->numXidCriticalErrors,
                                                startTime,
                                                endTime,
                                                DCGM_ORDER_ASCENDING,
                                                nullptr);

        if (dcgmReturn != DCGM_ST_OK)
        {
            DCGM_LOG_ERROR << "Got " << dcgmReturn << " from GetSamples()";
            /* Keep going. We used to just ignore this return */
        }

        for (i = 0; i < singleInfo->numXidCriticalErrors; i++)
        {
            singleInfo->xidCriticalErrorsTs[i] = samples[i].timestamp;
            if (pidInfo->summary.numXidCriticalErrors < (int)DCGM_ARRAY_CAPACITY(pidInfo->summary.xidCriticalErrorsTs))
            {
                pidInfo->summary.xidCriticalErrorsTs[pidInfo->summary.numXidCriticalErrors] = samples[i].timestamp;
                pidInfo->summary.numXidCriticalErrors++;
            }
        }
        mpCacheManager->FreeSamples(samples, singleInfo->numXidCriticalErrors, DCGM_FI_DEV_XID_ERRORS);

        singleInfo->numOtherComputePids = (int)DCGM_ARRAY_CAPACITY(singleInfo->otherComputePids);
        dcgmReturn                      = mpCacheManager->GetUniquePidLists(DCGM_FE_GPU,
                                                       singleInfo->gpuId,
                                                       DCGM_FI_DEV_COMPUTE_PIDS,
                                                       pidInfo->pid,
                                                       singleInfo->otherComputePids,
                                                       (unsigned int *)&singleInfo->numOtherComputePids,
                                                       startTime,
                                                       endTime);
        if (dcgmReturn != DCGM_ST_OK)
        {
            DCGM_LOG_ERROR << "Got " << dcgmReturn << " from GetUniquePidLists()";
            /* Keep going. We used to just ignore this return */
        }

        mergeUniquePids(pidInfo->summary.otherComputePids,
                        &pidInfo->summary.numOtherComputePids,
                        (int)DCGM_ARRAY_CAPACITY(pidInfo->summary.otherComputePids),
                        singleInfo->otherComputePids,
                        singleInfo->numOtherComputePids);

        singleInfo->numOtherGraphicsPids = (int)DCGM_ARRAY_CAPACITY(singleInfo->otherGraphicsPids);
        dcgmReturn                       = mpCacheManager->GetUniquePidLists(DCGM_FE_GPU,
                                                       singleInfo->gpuId,
                                                       DCGM_FI_DEV_GRAPHICS_PIDS,
                                                       pidInfo->pid,
                                                       singleInfo->otherGraphicsPids,
                                                       (unsigned int *)&singleInfo->numOtherGraphicsPids,
                                                       startTime,
                                                       endTime);

        mergeUniquePids(pidInfo->summary.otherGraphicsPids,
                        &pidInfo->summary.numOtherGraphicsPids,
                        (int)DCGM_ARRAY_CAPACITY(pidInfo->summary.otherGraphicsPids),
                        singleInfo->otherGraphicsPids,
                        singleInfo->numOtherGraphicsPids);

        singleInfo->maxGpuMemoryUsed      = (long long)accountingInfo.maxMemoryUsage;
        pidInfo->summary.maxGpuMemoryUsed = (long long)accountingInfo.maxMemoryUsage;

        /* Get the unique utilization sample for PIDs from the utilization Sample */
        dcgmProcessUtilSample_t smUtil[DCGM_MAX_PID_INFO_NUM];
        unsigned int numUniqueSmSamples = DCGM_MAX_PID_INFO_NUM;
        mpCacheManager->GetUniquePidUtilLists(DCGM_FE_GPU,
                                              singleInfo->gpuId,
                                              DCGM_FI_DEV_GPU_UTIL_SAMPLES,
                                              pidInfo->pid,
                                              smUtil,
                                              &numUniqueSmSamples,
                                              startTime,
                                              endTime);

        dcgmProcessUtilSample_t memUtil[DCGM_MAX_PID_INFO_NUM];
        unsigned int numUniqueMemSamples = DCGM_MAX_PID_INFO_NUM;
        mpCacheManager->GetUniquePidUtilLists(DCGM_FE_GPU,
                                              singleInfo->gpuId,
                                              DCGM_FI_DEV_MEM_COPY_UTIL_SAMPLES,
                                              pidInfo->pid,
                                              memUtil,
                                              &numUniqueMemSamples,
                                              startTime,
                                              endTime);

        /* Update the process utilization in the pidInfo*/
        singleInfo->processUtilization.pid     = pidInfo->pid;
        singleInfo->processUtilization.smUtil  = smUtil[0].util;
        singleInfo->processUtilization.memUtil = memUtil[0].util;

        summaryTypes[0] = DcgmcmSummaryTypeDifference;
        mpCacheManager->GetInt64SummaryData(DCGM_FE_GPU,
                                            singleInfo->gpuId,
                                            DCGM_FI_DEV_POWER_VIOLATION,
                                            1,
                                            &summaryTypes[0],
                                            &i64Val,
                                            startTime,
                                            endTime,
                                            nullptr,
                                            nullptr);
        singleInfo->powerViolationTime = i64Val;
        if (!DCGM_INT64_IS_BLANK(i64Val))
        {
            if (!DCGM_INT64_IS_BLANK(pidInfo->summary.powerViolationTime))
            {
                pidInfo->summary.powerViolationTime += i64Val;
            }
            else
            {
                pidInfo->summary.powerViolationTime = i64Val;
            }
        }

        summaryTypes[0] = DcgmcmSummaryTypeDifference;
        mpCacheManager->GetInt64SummaryData(DCGM_FE_GPU,
                                            singleInfo->gpuId,
                                            DCGM_FI_DEV_THERMAL_VIOLATION,
                                            1,
                                            &summaryTypes[0],
                                            &i64Val,
                                            startTime,
                                            endTime,
                                            nullptr,
                                            nullptr);
        singleInfo->thermalViolationTime = i64Val;
        if (!DCGM_INT64_IS_BLANK(i64Val))
        {
            if (!DCGM_INT64_IS_BLANK(pidInfo->summary.thermalViolationTime))
            {
                pidInfo->summary.thermalViolationTime += i64Val;
            }
            else
            {
                pidInfo->summary.thermalViolationTime = i64Val;
            }
        }

        summaryTypes[0] = DcgmcmSummaryTypeDifference;
        mpCacheManager->GetInt64SummaryData(DCGM_FE_GPU,
                                            singleInfo->gpuId,
                                            DCGM_FI_DEV_BOARD_LIMIT_VIOLATION,
                                            1,
                                            &summaryTypes[0],
                                            &i64Val,
                                            startTime,
                                            endTime,
                                            nullptr,
                                            nullptr);
        singleInfo->boardLimitViolationTime = i64Val;
        if (!DCGM_INT64_IS_BLANK(i64Val))
        {
            if (!DCGM_INT64_IS_BLANK(pidInfo->summary.boardLimitViolationTime))
            {
                pidInfo->summary.boardLimitViolationTime += i64Val;
            }
            else
            {
                pidInfo->summary.boardLimitViolationTime = i64Val;
            }
        }

        summaryTypes[0] = DcgmcmSummaryTypeDifference;
        mpCacheManager->GetInt64SummaryData(DCGM_FE_GPU,
                                            singleInfo->gpuId,
                                            DCGM_FI_DEV_LOW_UTIL_VIOLATION,
                                            1,
                                            &summaryTypes[0],
                                            &i64Val,
                                            startTime,
                                            endTime,
                                            nullptr,
                                            nullptr);
        singleInfo->lowUtilizationTime = i64Val;
        if (!DCGM_INT64_IS_BLANK(i64Val))
        {
            if (!DCGM_INT64_IS_BLANK(pidInfo->summary.lowUtilizationTime))
            {
                pidInfo->summary.lowUtilizationTime += i64Val;
            }
            else
            {
                pidInfo->summary.lowUtilizationTime = i64Val;
            }
        }

        summaryTypes[0] = DcgmcmSummaryTypeDifference;
        mpCacheManager->GetInt64SummaryData(DCGM_FE_GPU,
                                            singleInfo->gpuId,
                                            DCGM_FI_DEV_RELIABILITY_VIOLATION,
                                            1,
                                            &summaryTypes[0],
                                            &i64Val,
                                            startTime,
                                            endTime,
                                            nullptr,
                                            nullptr);
        singleInfo->reliabilityViolationTime = i64Val;
        if (!DCGM_INT64_IS_BLANK(i64Val))
        {
            if (!DCGM_INT64_IS_BLANK(pidInfo->summary.reliabilityViolationTime))
            {
                pidInfo->summary.reliabilityViolationTime += i64Val;
            }
            else
            {
                pidInfo->summary.reliabilityViolationTime = i64Val;
            }
        }

        summaryTypes[0] = DcgmcmSummaryTypeDifference;
        mpCacheManager->GetInt64SummaryData(DCGM_FE_GPU,
                                            singleInfo->gpuId,
                                            DCGM_FI_DEV_SYNC_BOOST_VIOLATION,
                                            1,
                                            &summaryTypes[0],
                                            &i64Val,
                                            startTime,
                                            endTime,
                                            nullptr,
                                            nullptr);
        singleInfo->syncBoostTime = i64Val;
        if (!DCGM_INT64_IS_BLANK(i64Val))
        {
            if (!DCGM_INT64_IS_BLANK(pidInfo->summary.syncBoostTime))
            {
                pidInfo->summary.syncBoostTime += i64Val;
            }
            else
            {
                pidInfo->summary.syncBoostTime = i64Val;
            }
        }

        /* Update the Health Response if we haven't retrieved it yet */
        if (response->version == 0)
        {
            HelperHealthCheck(groupId, startTime, endTime, *(response));
        }

        /* Update the overallHealth of the system */
        pidInfo->summary.overallHealth = response->overallHealth;

        unsigned int incidentCount = 0;

        singleInfo->overallHealth = DCGM_HEALTH_RESULT_PASS;

        for (unsigned int incidentIndex = 0; incidentIndex < response->incidentCount; incidentIndex++)
        {
            if (response->incidents[incidentIndex].entityInfo.entityId == singleInfo->gpuId
                && response->incidents[incidentIndex].entityInfo.entityGroupId == DCGM_FE_GPU)
            {
                if (response->incidents[incidentIndex].health > singleInfo->overallHealth)
                {
                    singleInfo->overallHealth = response->incidents[incidentIndex].health;
                }

                singleInfo->systems[incidentCount].system = response->incidents[incidentIndex].system;
                singleInfo->systems[incidentCount].health = response->incidents[incidentIndex].health;

                incidentCount++;
            }
        }

        // Update the Incident Count
        singleInfo->incidentCount = incidentCount;
    }

    if (pidInfo->numGpus == 0)
    {
        if (dcgmReturn == DCGM_ST_NOT_WATCHED)
        {
            return DCGM_ST_NOT_WATCHED;
        }


        log_debug("Pid {} ran on no GPUs", pidInfo->pid);
        return DCGM_ST_NO_DATA;
    }

    return DCGM_ST_OK;
}


/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::JobStartStats(std::string const &jobId, unsigned int groupId)
{
    jobIdMap_t::iterator it;

    /* If the entry already exists return error to provide unique key. Override it with */
    auto lock = Lock();

    it = mJobIdMap.find(jobId);
    if (it == mJobIdMap.end())
    {
        /* Insert it as a record */
        jobRecord_t record;
        record.startTime = timelib_usecSince1970();
        record.endTime   = 0;
        record.groupId   = groupId;
        mJobIdMap.insert(make_pair(jobId, record));
    }
    else
    {
        log_error("Duplicate JobId as input : {}", jobId.c_str());
        /* Implies that the entry corresponding to the job id already exists */
        return DCGM_ST_DUPLICATE_KEY;
    }

    return DCGM_ST_OK;
}


/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::JobStopStats(std::string const &jobId)
{
    jobIdMap_t::iterator it;

    /* If the entry already exists return error to provide unique key. Override it with */
    auto lock = Lock();

    it = mJobIdMap.find(jobId);
    if (it == mJobIdMap.end())
    {
        log_error("Can't find entry corresponding to the Job Id : {}", jobId.c_str());
        return DCGM_ST_NO_DATA;
    }

    jobRecord_t *pRecord = &(it->second);
    pRecord->endTime     = timelib_usecSince1970();

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::HelperHealthCheck(unsigned int groupId,
                                                      long long startTime,
                                                      long long endTime,
                                                      dcgmHealthResponse_t &response)
{
    std::unique_ptr<dcgm_health_msg_check_v5> msg = std::make_unique<dcgm_health_msg_check_v5>();

    memset(msg.get(), 0, sizeof(*msg));
    msg->header.length     = sizeof(*msg);
    msg->header.moduleId   = DcgmModuleIdHealth;
    msg->header.subCommand = DCGM_HEALTH_SR_CHECK_V5;
    msg->header.version    = dcgm_health_msg_check_version5;

    msg->groupId   = (dcgmGpuGrp_t)(uintptr_t)groupId;
    msg->startTime = startTime;
    msg->endTime   = endTime;

    dcgmReturn_t dcgmReturn = ProcessModuleCommand(&msg->header);
    if (dcgmReturn != DCGM_ST_OK)
    {
        if (dcgmReturn == DCGM_ST_MODULE_NOT_LOADED)
        {
            log_debug("Health check skipped due to module not being loaded.");
        }
        else
        {
            log_error("Health check failed with {}", (int)dcgmReturn);
        }
        return dcgmReturn;
    }

    memcpy(&response, &msg->response, sizeof(response));
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::JobGetStats(const std::string &jobId, dcgmJobInfo_t *pJobInfo)
{
    jobIdMap_t::iterator it;
    jobRecord_t *pRecord;
    unsigned int groupId;
    std::vector<dcgmGroupEntityPair_t> entities;
    std::vector<unsigned int> gpuIds;
    std::vector<unsigned int>::iterator gpuIdIt;
    dcgmGpuUsageInfo_t *singleInfo;
    dcgmReturn_t dcgmReturn = DCGM_ST_OK;
    dcgmDevicePidAccountingStats_t accountingInfo;
    long long startTime;
    long long endTime;
    long long i64Val;
    DcgmcmSummaryType_t summaryTypes[DcgmcmSummaryTypeSize];
    int i;
    double doubleVals[DcgmcmSummaryTypeSize];
    int Msamples = 10; /* Should match size of samples[] */
    dcgmcm_sample_t samples[10];
    dcgmStatSummaryInt32_t blankSummary32  = { DCGM_INT32_BLANK, DCGM_INT32_BLANK, DCGM_INT32_BLANK };
    dcgmStatSummaryInt64_t blankSummary64  = { DCGM_INT64_BLANK, DCGM_INT64_BLANK, DCGM_INT64_BLANK };
    dcgmStatSummaryFp64_t blankSummaryFP64 = { DCGM_FP64_BLANK, DCGM_FP64_BLANK, DCGM_FP64_BLANK };
    int fieldValue;

    if (pJobInfo->version != dcgmJobInfo_version)
    {
        log_warning("Version mismatch. expected {}. Got {}", dcgmJobInfo_version, pJobInfo->version);
        return DCGM_ST_VER_MISMATCH;
    }

    /* If entry can't be found then return error back to the caller */
    auto lock = Lock();

    it = mJobIdMap.find(jobId);
    if (it == mJobIdMap.end())
    {
        log_error("Can't find entry corresponding to the Job Id : {}", jobId.c_str());
        return DCGM_ST_NO_DATA;
    }

    pRecord   = &it->second;
    groupId   = pRecord->groupId;
    startTime = pRecord->startTime;

    if (pRecord->endTime == 0)
    {
        endTime = (long long)timelib_usecSince1970();
    }
    else
    {
        endTime = (long long)pRecord->endTime;
    }
    Unlock(std::move(lock));

    if (startTime > endTime)
    {
        log_error(
            "Get job stats. Start time is greater than end time. start time: {} end time: {}", startTime, endTime);
        return DCGM_ST_GENERIC_ERROR;
    }

    /* Resolve the groupId -> entities[] -> gpuIds[] */
    dcgmReturn = mpGroupManager->GetGroupEntities(groupId, DcgmGroupOption::All, entities);
    if (dcgmReturn != DCGM_ST_OK)
    {
        log_error("Error {} from GetGroupEntities()", (int)dcgmReturn);
        return dcgmReturn;
    }

    /* Process stats are only supported for GPUs for now */
    for (i = 0; i < (int)entities.size(); i++)
    {
        if (entities[i].entityGroupId != DCGM_FE_GPU)
        {
            continue;
        }

        gpuIds.push_back(entities[i].entityId);
    }

    /* Initialize a health response to be populated later */
    std::unique_ptr<dcgmHealthResponse_t> response = std::make_unique<dcgmHealthResponse_t>();

    /* Zero the structures */
    pJobInfo->numGpus = 0;
    memset(&pJobInfo->gpus[0], 0, sizeof(pJobInfo->gpus));
    memset(&pJobInfo->summary, 0, sizeof(pJobInfo->summary));

    /* Initialize summary information */
    pJobInfo->summary.gpuId                    = DCGM_INT32_BLANK;
    pJobInfo->summary.pcieRxBandwidth          = blankSummary64;
    pJobInfo->summary.pcieTxBandwidth          = blankSummary64;
    pJobInfo->summary.powerViolationTime       = DCGM_INT64_NOT_SUPPORTED;
    pJobInfo->summary.thermalViolationTime     = DCGM_INT64_NOT_SUPPORTED;
    pJobInfo->summary.reliabilityViolationTime = DCGM_INT64_NOT_SUPPORTED;
    pJobInfo->summary.boardLimitViolationTime  = DCGM_INT64_NOT_SUPPORTED;
    pJobInfo->summary.lowUtilizationTime       = DCGM_INT64_NOT_SUPPORTED;
    pJobInfo->summary.syncBoostTime            = DCGM_INT64_BLANK;
    pJobInfo->summary.energyConsumed           = DCGM_INT64_BLANK;
    pJobInfo->summary.pcieReplays              = DCGM_INT64_BLANK;
    pJobInfo->summary.smUtilization            = blankSummary32;
    pJobInfo->summary.memoryUtilization        = blankSummary32;
    pJobInfo->summary.eccSingleBit             = DCGM_INT32_BLANK;
    pJobInfo->summary.eccDoubleBit             = DCGM_INT32_BLANK;
    pJobInfo->summary.memoryClock              = blankSummary32;
    pJobInfo->summary.smClock                  = blankSummary32;
    pJobInfo->summary.powerUsage               = blankSummaryFP64;

    /* Update the start and end time in the summary*/
    pJobInfo->summary.startTime = startTime;
    pJobInfo->summary.endTime   = endTime;

    for (gpuIdIt = gpuIds.begin(); gpuIdIt != gpuIds.end(); ++gpuIdIt)
    {
        singleInfo        = &pJobInfo->gpus[pJobInfo->numGpus];
        singleInfo->gpuId = *gpuIdIt;

        /* Increment GPU count now that we know the process ran on this GPU */
        pJobInfo->numGpus++;

        summaryTypes[0] = DcgmcmSummaryTypeIntegral;
        summaryTypes[1] = DcgmcmSummaryTypeMinimum;
        summaryTypes[2] = DcgmcmSummaryTypeMaximum;
        summaryTypes[3] = DcgmcmSummaryTypeAverage;

        mpCacheManager->GetFp64SummaryData(DCGM_FE_GPU,
                                           singleInfo->gpuId,
                                           DCGM_FI_DEV_POWER_USAGE,
                                           4,
                                           &summaryTypes[0],
                                           &doubleVals[0],
                                           startTime,
                                           endTime,
                                           nullptr,
                                           nullptr);

        /* See if the energy counter is supported. If so, use that rather than integrating the power usage */
        summaryTypes[0] = DcgmcmSummaryTypeDifference;
        mpCacheManager->GetInt64SummaryData(DCGM_FE_GPU,
                                            singleInfo->gpuId,
                                            DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION,
                                            1,
                                            &summaryTypes[0],
                                            &i64Val,
                                            startTime,
                                            endTime,
                                            nullptr,
                                            nullptr);
        if (!DCGM_INT64_IS_BLANK(i64Val))
        {
            singleInfo->energyConsumed = i64Val;
        }
        else
        {
            /* No energy counter. Integrate power usage */
            log_debug("No energy counter. Using power_usage");

            if (!DCGM_FP64_IS_BLANK(doubleVals[0]))
            {
                doubleVals[0] /= 1000.0; /* convert from usec watts to milliwatt seconds */
            }
            singleInfo->energyConsumed = nvcmvalue_double_to_int64(doubleVals[0]);
        }

        /* Update summary value, handling blank case */
        if (!DCGM_INT64_IS_BLANK(singleInfo->energyConsumed))
        {
            if (!DCGM_INT64_IS_BLANK(pJobInfo->summary.energyConsumed))
            {
                pJobInfo->summary.energyConsumed += singleInfo->energyConsumed;
            }
            else
            {
                pJobInfo->summary.energyConsumed = singleInfo->energyConsumed;
            }
        }

        singleInfo->powerUsage.minValue = doubleVals[1]; /* Same indexes as summaryTypes[] */
        singleInfo->powerUsage.maxValue = doubleVals[2];
        singleInfo->powerUsage.average  = doubleVals[3];

        /* Update summary value for average, handling blank case */
        if (!DCGM_FP64_IS_BLANK(singleInfo->powerUsage.average))
        {
            if (!DCGM_FP64_IS_BLANK(pJobInfo->summary.powerUsage.average))
            {
                pJobInfo->summary.powerUsage.average += singleInfo->powerUsage.average;
            }
            else
            {
                pJobInfo->summary.powerUsage.average = singleInfo->powerUsage.average;
            }
        }

        /* Note: we aren't populating minimum and maximum summary values because they don't make sense across
         * GPUS. One GPUs minimum could occur at a different time than another GPU's minimum
         */

        DcgmHostEngineHandler::Instance()->HelperGetInt64StatSummary(DCGM_FE_GPU,
                                                                     singleInfo->gpuId,
                                                                     DCGM_FI_DEV_PCIE_RX_THROUGHPUT,
                                                                     &singleInfo->pcieRxBandwidth,
                                                                     startTime,
                                                                     endTime);
        DcgmHostEngineHandler::Instance()->HelperGetInt64StatSummary(DCGM_FE_GPU,
                                                                     singleInfo->gpuId,
                                                                     DCGM_FI_DEV_PCIE_TX_THROUGHPUT,
                                                                     &singleInfo->pcieTxBandwidth,
                                                                     startTime,
                                                                     endTime);

        /* If the PCIE Tx BW is blank, update the average with the PCIE Tx BW value as 0 for this GPU*/
        if (DCGM_INT64_IS_BLANK(singleInfo->pcieTxBandwidth.average))
        {
            fieldValue = 0;
        }
        else
        {
            fieldValue = (int)singleInfo->pcieTxBandwidth.average;
        }

        pJobInfo->summary.pcieTxBandwidth.average
            = (pJobInfo->summary.pcieTxBandwidth.average * (pJobInfo->numGpus - 1) + fieldValue) / (pJobInfo->numGpus);

        /* If the PCIE Rx BW is blank, update the average with the PCIE Rx BW value as 0 for this GPU*/
        if (DCGM_INT64_IS_BLANK(singleInfo->pcieRxBandwidth.average))
        {
            fieldValue = 0;
        }
        else
        {
            fieldValue = (int)singleInfo->pcieRxBandwidth.average;
        }

        pJobInfo->summary.pcieRxBandwidth.average
            = (pJobInfo->summary.pcieRxBandwidth.average * (pJobInfo->numGpus - 1) + fieldValue) / (pJobInfo->numGpus);

        summaryTypes[0] = DcgmcmSummaryTypeMaximum;
        mpCacheManager->GetInt64SummaryData(DCGM_FE_GPU,
                                            singleInfo->gpuId,
                                            DCGM_FI_DEV_PCIE_REPLAY_COUNTER,
                                            1,
                                            &summaryTypes[0],
                                            &singleInfo->pcieReplays,
                                            startTime,
                                            endTime,
                                            nullptr,
                                            nullptr);
        if (!DCGM_INT64_IS_BLANK(singleInfo->pcieReplays))
        {
            if (DCGM_INT64_IS_BLANK(pJobInfo->summary.pcieReplays))
            {
                pJobInfo->summary.pcieReplays = singleInfo->pcieReplays;
            }
            else
            {
                pJobInfo->summary.pcieReplays += singleInfo->pcieReplays;
            }
        }

        singleInfo->startTime = startTime;
        singleInfo->endTime   = endTime;

        DcgmHostEngineHandler::Instance()->HelperGetInt32StatSummary(
            DCGM_FE_GPU, singleInfo->gpuId, DCGM_FI_DEV_GPU_UTIL, &singleInfo->smUtilization, startTime, endTime);

        /* If the SM utilization is blank, update the average with the SM utilization value as 0 for this GPU*/
        if (DCGM_INT32_IS_BLANK(singleInfo->smUtilization.average))
        {
            fieldValue = 0;
        }
        else
        {
            fieldValue = singleInfo->smUtilization.average;
        }

        pJobInfo->summary.smUtilization.average
            = (pJobInfo->summary.smUtilization.average * (pJobInfo->numGpus - 1) + fieldValue) / (pJobInfo->numGpus);

        DcgmHostEngineHandler::Instance()->HelperGetInt32StatSummary(DCGM_FE_GPU,
                                                                     singleInfo->gpuId,
                                                                     DCGM_FI_DEV_MEM_COPY_UTIL,
                                                                     &singleInfo->memoryUtilization,
                                                                     startTime,
                                                                     endTime);

        /* If  mem utilization is blank, update the average with the mem utilization value as 0 for this GPU*/
        if (DCGM_INT32_IS_BLANK(singleInfo->memoryUtilization.average))
        {
            fieldValue = 0;
        }
        else
        {
            fieldValue = singleInfo->memoryUtilization.average;
        }

        pJobInfo->summary.memoryUtilization.average
            = (pJobInfo->summary.memoryUtilization.average * (pJobInfo->numGpus - 1) + fieldValue)
              / (pJobInfo->numGpus);

        summaryTypes[0] = DcgmcmSummaryTypeMaximum;
        mpCacheManager->GetInt64SummaryData(DCGM_FE_GPU,
                                            singleInfo->gpuId,
                                            DCGM_FI_DEV_ECC_DBE_VOL_TOTAL,
                                            1,
                                            &summaryTypes[0],
                                            &i64Val,
                                            startTime,
                                            endTime,
                                            nullptr,
                                            nullptr);
        singleInfo->eccDoubleBit = nvcmvalue_int64_to_int32(i64Val);

        if (!DCGM_INT32_IS_BLANK(singleInfo->eccDoubleBit))
        {
            if (DCGM_INT32_IS_BLANK(pJobInfo->summary.eccDoubleBit))
            {
                pJobInfo->summary.eccDoubleBit = singleInfo->eccDoubleBit;
            }
            else
            {
                pJobInfo->summary.eccDoubleBit += singleInfo->eccDoubleBit;
            }
        }

        DcgmHostEngineHandler::Instance()->HelperGetInt32StatSummary(
            DCGM_FE_GPU, singleInfo->gpuId, DCGM_FI_DEV_SM_CLOCK, &singleInfo->smClock, startTime, endTime);

        /* If  SM clock is blank, update the average with the SM  clock value as 0 for this GPU*/
        if (DCGM_INT32_IS_BLANK(singleInfo->smClock.average))
        {
            fieldValue = 0;
        }
        else
        {
            fieldValue = singleInfo->smClock.average;
        }

        pJobInfo->summary.smClock.average
            = (pJobInfo->summary.smClock.average * (pJobInfo->numGpus - 1) + fieldValue) / (pJobInfo->numGpus);

        DcgmHostEngineHandler::Instance()->HelperGetInt32StatSummary(
            DCGM_FE_GPU, singleInfo->gpuId, DCGM_FI_DEV_MEM_CLOCK, &singleInfo->memoryClock, startTime, endTime);

        /* If memory clock is blank, update the average with the memory clock  value as 0 for this GPU*/
        if (DCGM_INT32_IS_BLANK(singleInfo->memoryClock.average))
        {
            fieldValue = 0;
        }
        else
        {
            fieldValue = singleInfo->memoryClock.average;
        }

        pJobInfo->summary.memoryClock.average
            = (pJobInfo->summary.memoryClock.average * (pJobInfo->numGpus - 1) + fieldValue) / (pJobInfo->numGpus);


        singleInfo->numXidCriticalErrors = Msamples;
        dcgmReturn                       = mpCacheManager->GetSamples(DCGM_FE_GPU,
                                                singleInfo->gpuId,
                                                DCGM_FI_DEV_XID_ERRORS,
                                                samples,
                                                &singleInfo->numXidCriticalErrors,
                                                startTime,
                                                endTime,
                                                DCGM_ORDER_ASCENDING,
                                                nullptr);
        if (dcgmReturn != DCGM_ST_OK)
        {
            DCGM_LOG_ERROR << "Got " << dcgmReturn << " from GetSamples()";
            /* Keep going. We used to just ignore this return */
        }

        for (i = 0; i < singleInfo->numXidCriticalErrors; i++)
        {
            singleInfo->xidCriticalErrorsTs[i] = samples[i].timestamp;
            if (pJobInfo->summary.numXidCriticalErrors
                < (int)DCGM_ARRAY_CAPACITY(pJobInfo->summary.xidCriticalErrorsTs))
            {
                pJobInfo->summary.xidCriticalErrorsTs[pJobInfo->summary.numXidCriticalErrors] = samples[i].timestamp;
                pJobInfo->summary.numXidCriticalErrors++;
            }
        }
        mpCacheManager->FreeSamples(samples, singleInfo->numXidCriticalErrors, DCGM_FI_DEV_XID_ERRORS);

        singleInfo->numComputePids = (int)DCGM_ARRAY_CAPACITY(singleInfo->computePidInfo);
        dcgmReturn                 = mpCacheManager->GetUniquePidLists(DCGM_FE_GPU,
                                                       singleInfo->gpuId,
                                                       DCGM_FI_DEV_COMPUTE_PIDS,
                                                       0,
                                                       singleInfo->computePidInfo,
                                                       (unsigned int *)&singleInfo->numComputePids,
                                                       startTime,
                                                       endTime);
        if (dcgmReturn != DCGM_ST_OK)
        {
            DCGM_LOG_ERROR << "Got " << dcgmReturn << " from GetUniquePidLists()";
            /* Keep going. We used to just ignore this return */
        }

        mergeUniquePidInfo(pJobInfo->summary.computePidInfo,
                           &pJobInfo->summary.numComputePids,
                           (int)DCGM_ARRAY_CAPACITY(pJobInfo->summary.computePidInfo),
                           singleInfo->computePidInfo,
                           singleInfo->numComputePids);

        singleInfo->numGraphicsPids = (int)DCGM_ARRAY_CAPACITY(singleInfo->graphicsPidInfo);
        dcgmReturn                  = mpCacheManager->GetUniquePidLists(DCGM_FE_GPU,
                                                       singleInfo->gpuId,
                                                       DCGM_FI_DEV_GRAPHICS_PIDS,
                                                       0,
                                                       singleInfo->graphicsPidInfo,
                                                       (unsigned int *)&singleInfo->numGraphicsPids,
                                                       startTime,
                                                       endTime);
        if (dcgmReturn != DCGM_ST_OK)
        {
            DCGM_LOG_ERROR << "Got " << dcgmReturn << " from GetUniquePidLists()";
            /* Keep going. We used to just ignore this return */
        }

        mergeUniquePidInfo(pJobInfo->summary.graphicsPidInfo,
                           &pJobInfo->summary.numGraphicsPids,
                           (int)DCGM_ARRAY_CAPACITY(pJobInfo->summary.graphicsPidInfo),
                           singleInfo->graphicsPidInfo,
                           singleInfo->numGraphicsPids);

        /* Get the max memory usage for the GPU and summary option for compute PIDs */
        for (i = 0; i < singleInfo->numComputePids; i++)
        {
            // Get max memory usage for all the processes on the GPU
            dcgmReturn = mpCacheManager->GetLatestProcessInfo(
                singleInfo->gpuId, singleInfo->computePidInfo[i].pid, &accountingInfo);
            if (DCGM_ST_OK == dcgmReturn)
            {
                if ((long long)accountingInfo.maxMemoryUsage > singleInfo->maxGpuMemoryUsed)
                {
                    singleInfo->maxGpuMemoryUsed = (long long)accountingInfo.maxMemoryUsage;
                }

                if ((long long)accountingInfo.maxMemoryUsage > pJobInfo->summary.maxGpuMemoryUsed)
                {
                    pJobInfo->summary.maxGpuMemoryUsed = (long long)accountingInfo.maxMemoryUsage;
                }
            }
        }

        /* Get the max memory usage for the GPU and summary option for Graphics PIDs */
        for (i = 0; i < singleInfo->numGraphicsPids; i++)
        {
            // Get max memory usage for all the processes on the GPU
            dcgmReturn = mpCacheManager->GetLatestProcessInfo(
                singleInfo->gpuId, singleInfo->graphicsPidInfo[i].pid, &accountingInfo);
            if (DCGM_ST_OK == dcgmReturn)
            {
                if ((long long)accountingInfo.maxMemoryUsage > singleInfo->maxGpuMemoryUsed)
                {
                    singleInfo->maxGpuMemoryUsed = (long long)accountingInfo.maxMemoryUsage;
                }

                if ((long long)accountingInfo.maxMemoryUsage > pJobInfo->summary.maxGpuMemoryUsed)
                {
                    pJobInfo->summary.maxGpuMemoryUsed = (long long)accountingInfo.maxMemoryUsage;
                }
            }
        }

        /* Get the unique utilization sample for PIDs from the utilization Sample */
        dcgmProcessUtilSample_t smUtil[DCGM_MAX_PID_INFO_NUM];
        unsigned int numUniqueSmSamples = DCGM_MAX_PID_INFO_NUM;
        mpCacheManager->GetUniquePidUtilLists(DCGM_FE_GPU,
                                              singleInfo->gpuId,
                                              DCGM_FI_DEV_GPU_UTIL_SAMPLES,
                                              0,
                                              smUtil,
                                              &numUniqueSmSamples,
                                              startTime,
                                              endTime);

        dcgmProcessUtilSample_t memUtil[DCGM_MAX_PID_INFO_NUM];
        unsigned int numUniqueMemSamples = DCGM_MAX_PID_INFO_NUM;
        mpCacheManager->GetUniquePidUtilLists(DCGM_FE_GPU,
                                              singleInfo->gpuId,
                                              DCGM_FI_DEV_MEM_COPY_UTIL_SAMPLES,
                                              0,
                                              memUtil,
                                              &numUniqueMemSamples,
                                              startTime,
                                              endTime);


        /* Merge the SM and MEM utilization rates for various PIDs */
        for (i = 0; i < singleInfo->numComputePids; i++)
        {
            findPidUtilInfo(smUtil, numUniqueSmSamples, memUtil, numUniqueMemSamples, &singleInfo->computePidInfo[i]);
        }

        for (i = 0; i < singleInfo->numGraphicsPids; i++)
        {
            findPidUtilInfo(smUtil, numUniqueSmSamples, memUtil, numUniqueMemSamples, &singleInfo->graphicsPidInfo[i]);
        }


        summaryTypes[0] = DcgmcmSummaryTypeDifference;
        mpCacheManager->GetInt64SummaryData(DCGM_FE_GPU,
                                            singleInfo->gpuId,
                                            DCGM_FI_DEV_POWER_VIOLATION,
                                            1,
                                            &summaryTypes[0],
                                            &i64Val,
                                            startTime,
                                            endTime,
                                            nullptr,
                                            nullptr);
        singleInfo->powerViolationTime = i64Val;
        if (!DCGM_INT64_IS_BLANK(i64Val))
        {
            if (!DCGM_INT64_IS_BLANK(pJobInfo->summary.powerViolationTime))
            {
                pJobInfo->summary.powerViolationTime += i64Val;
            }
            else
            {
                pJobInfo->summary.powerViolationTime = i64Val;
            }
        }

        summaryTypes[0] = DcgmcmSummaryTypeDifference;
        mpCacheManager->GetInt64SummaryData(DCGM_FE_GPU,
                                            singleInfo->gpuId,
                                            DCGM_FI_DEV_THERMAL_VIOLATION,
                                            1,
                                            &summaryTypes[0],
                                            &i64Val,
                                            startTime,
                                            endTime,
                                            nullptr,
                                            nullptr);
        singleInfo->thermalViolationTime = i64Val;
        if (!DCGM_INT64_IS_BLANK(i64Val))
        {
            if (!DCGM_INT64_IS_BLANK(pJobInfo->summary.thermalViolationTime))
            {
                pJobInfo->summary.thermalViolationTime += i64Val;
            }
            else
            {
                pJobInfo->summary.thermalViolationTime = i64Val;
            }
        }

        summaryTypes[0] = DcgmcmSummaryTypeDifference;
        mpCacheManager->GetInt64SummaryData(DCGM_FE_GPU,
                                            singleInfo->gpuId,
                                            DCGM_FI_DEV_RELIABILITY_VIOLATION,
                                            1,
                                            &summaryTypes[0],
                                            &i64Val,
                                            startTime,
                                            endTime,
                                            nullptr,
                                            nullptr);
        singleInfo->reliabilityViolationTime = i64Val;
        if (!DCGM_INT64_IS_BLANK(i64Val))
        {
            if (!DCGM_INT64_IS_BLANK(pJobInfo->summary.reliabilityViolationTime))
            {
                pJobInfo->summary.reliabilityViolationTime += i64Val;
            }
            else
            {
                pJobInfo->summary.reliabilityViolationTime = i64Val;
            }
        }

        summaryTypes[0] = DcgmcmSummaryTypeDifference;
        mpCacheManager->GetInt64SummaryData(DCGM_FE_GPU,
                                            singleInfo->gpuId,
                                            DCGM_FI_DEV_BOARD_LIMIT_VIOLATION,
                                            1,
                                            &summaryTypes[0],
                                            &i64Val,
                                            startTime,
                                            endTime,
                                            nullptr,
                                            nullptr);
        singleInfo->boardLimitViolationTime = i64Val;
        if (!DCGM_INT64_IS_BLANK(i64Val))
        {
            if (!DCGM_INT64_IS_BLANK(pJobInfo->summary.boardLimitViolationTime))
            {
                pJobInfo->summary.boardLimitViolationTime += i64Val;
            }
            else
            {
                pJobInfo->summary.boardLimitViolationTime = i64Val;
            }
        }

        summaryTypes[0] = DcgmcmSummaryTypeDifference;
        mpCacheManager->GetInt64SummaryData(DCGM_FE_GPU,
                                            singleInfo->gpuId,
                                            DCGM_FI_DEV_LOW_UTIL_VIOLATION,
                                            1,
                                            &summaryTypes[0],
                                            &i64Val,
                                            startTime,
                                            endTime,
                                            nullptr,
                                            nullptr);
        singleInfo->lowUtilizationTime = i64Val;
        if (!DCGM_INT64_IS_BLANK(i64Val))
        {
            if (!DCGM_INT64_IS_BLANK(pJobInfo->summary.lowUtilizationTime))
            {
                pJobInfo->summary.lowUtilizationTime += i64Val;
            }
            else
            {
                pJobInfo->summary.lowUtilizationTime = i64Val;
            }
        }

        summaryTypes[0] = DcgmcmSummaryTypeDifference;
        mpCacheManager->GetInt64SummaryData(DCGM_FE_GPU,
                                            singleInfo->gpuId,
                                            DCGM_FI_DEV_SYNC_BOOST_VIOLATION,
                                            1,
                                            &summaryTypes[0],
                                            &i64Val,
                                            startTime,
                                            endTime,
                                            nullptr,
                                            nullptr);
        singleInfo->syncBoostTime = i64Val;
        if (!DCGM_INT64_IS_BLANK(i64Val))
        {
            if (!DCGM_INT64_IS_BLANK(pJobInfo->summary.syncBoostTime))
            {
                pJobInfo->summary.syncBoostTime += i64Val;
            }
            else
            {
                pJobInfo->summary.syncBoostTime = i64Val;
            }
        }

        /* Update the Health Response if we haven't retrieved it yet */
        if (response->version == 0)
        {
            HelperHealthCheck(groupId, startTime, endTime, *(response));
        }

        /* Update the overallHealth of the system */
        pJobInfo->summary.overallHealth = response->overallHealth;

        unsigned int incidentCount = 0;

        singleInfo->overallHealth = DCGM_HEALTH_RESULT_PASS;

        for (unsigned int incidentIndex = 0; incidentIndex < response->incidentCount; incidentIndex++)
        {
            if (response->incidents[incidentIndex].entityInfo.entityId == singleInfo->gpuId
                && response->incidents[incidentIndex].entityInfo.entityGroupId == DCGM_FE_GPU)
            {
                if (response->incidents[incidentIndex].health > singleInfo->overallHealth)
                {
                    singleInfo->overallHealth = response->incidents[incidentIndex].health;
                }

                singleInfo->systems[incidentCount].system = response->incidents[incidentIndex].system;
                singleInfo->systems[incidentCount].health = response->incidents[incidentIndex].health;

                incidentCount++;
            }
        }

        // Update the Incident Count
        singleInfo->incidentCount = incidentCount;
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::JobRemove(std::string const &jobId)
{
    jobIdMap_t::iterator it;

    /* If the entry already exists return error to provide unique key. Override it with */
    auto lock = Lock();

    it = mJobIdMap.find(jobId);
    if (it == mJobIdMap.end())
    {
        log_error("JobRemove: Can't find jobId : {}", jobId);
        return DCGM_ST_NO_DATA;
    }

    mJobIdMap.erase(it);

    log_debug("JobRemove: Removed jobId {}", jobId);
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::JobRemoveAll()
{
    jobIdMap_t::iterator it;

    /* If the entry already exists return error to provide unique key. Override it with */
    auto lock = Lock();

    mJobIdMap.clear();

    log_debug("JobRemoveAll: Removed all jobs");
    return DCGM_ST_OK;
}

/*****************************************************************************/
static void helper_get_prof_field_ids(std::vector<unsigned short> &fieldIds, std::vector<unsigned short> &profFieldIds)
{
    profFieldIds.clear();

    for (unsigned short &fieldId : fieldIds)
    {
        if (fieldId >= DCGM_FI_PROF_FIRST_ID && fieldId <= DCGM_FI_PROF_LAST_ID)
        {
            profFieldIds.push_back(fieldId);
        }
    }
}

static bool helperIsSysmonField(unsigned short fieldId)
{
    return fieldId >= DCGM_FI_SYSMON_FIRST_ID && fieldId <= DCGM_FI_SYSMON_LAST_ID;
}

static dcgmReturn_t helperFilterSysmonEntitiesAndFields(dcgm_sysmon_msg_watch_fields_t &sysmonMsg,
                                                        std::vector<dcgmGroupEntityPair_t> const &entities,
                                                        std::vector<unsigned short> const &fields,
                                                        DcgmWatcher const &watcher,
                                                        timelib64_t const updateIntervalUsec,
                                                        double const maxSampleAge,
                                                        int const maxKeepSamples)
{
    // First check we have CPU entities and fields and that they fit in the message

    std::vector<dcgmGroupEntityPair_t> cpuEntities;
    std::vector<unsigned short> cpuFields;

    for (auto entity : entities)
    {
        if (entity.entityGroupId == DCGM_FE_CPU || entity.entityGroupId == DCGM_FE_CPU_CORE)
        {
            cpuEntities.emplace_back(entity);
        }
    }

    for (auto field : fields)
    {
        if (helperIsSysmonField(field))
        {
            cpuFields.emplace_back(field);
        }
    }

    if (cpuEntities.empty() || cpuFields.empty())
    {
        // Nothing to watch
        return DCGM_ST_OK;
    }

    if (cpuEntities.size() > SYSMON_MSG_WATCH_FIELDS_MAX_NUM_ENTITIES)
    {
        log_error(
            "cpuEntities.size {} exceeds max size {}", cpuEntities.size(), SYSMON_MSG_WATCH_FIELDS_MAX_NUM_ENTITIES);
        return DCGM_ST_INSUFFICIENT_RESOURCES;
    }

    if (cpuFields.size() > SYSMON_MSG_WATCH_FIELDS_MAX_NUM_FIELDS)
    {
        log_error("cpuFields.size {} exceeds max size {}", cpuFields.size(), SYSMON_MSG_WATCH_FIELDS_MAX_NUM_FIELDS);
        return DCGM_ST_INSUFFICIENT_RESOURCES;
    }

    sysmonMsg.header.length       = sizeof(sysmonMsg);
    sysmonMsg.header.moduleId     = DcgmModuleIdSysmon;
    sysmonMsg.header.subCommand   = DCGM_SYSMON_SR_WATCH_FIELDS;
    sysmonMsg.header.connectionId = watcher.connectionId;
    sysmonMsg.header.version      = dcgm_sysmon_msg_watch_fields_version;

    sysmonMsg.updateIntervalUsec = updateIntervalUsec;
    sysmonMsg.maxKeepAge         = maxSampleAge;
    sysmonMsg.maxKeepSamples     = maxKeepSamples;

    sysmonMsg.numEntities = cpuEntities.size();
    for (unsigned int i = 0; i < cpuEntities.size(); i++)
    {
        auto const &entity       = cpuEntities[i];
        sysmonMsg.entityPairs[i] = entity;
    }

    sysmonMsg.numFieldIds = cpuFields.size();
    for (unsigned int i = 0; i < cpuFields.size(); i++)
    {
        auto const &field     = cpuFields[i];
        sysmonMsg.fieldIds[i] = field;
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::WatchFieldGroup(unsigned int groupId,
                                                    dcgmFieldGrp_t fieldGroupId,
                                                    timelib64_t monitorIntervalUsec,
                                                    double maxSampleAge,
                                                    int maxKeepSamples,
                                                    DcgmWatcher const &watcher)
{
    int i;
    int j;
    dcgmReturn_t dcgmReturn;
    std::vector<dcgmGroupEntityPair_t> entities;
    std::vector<unsigned short> fieldIds;
    std::vector<unsigned short> profFieldIds;
    dcgm_sysmon_msg_watch_fields_t sysmonMsg = {};
    dcgmReturn_t retSt                       = DCGM_ST_OK;

    dcgmReturn = mpGroupManager->GetGroupEntities(groupId, DcgmGroupOption::All, entities);
    if (dcgmReturn != DCGM_ST_OK)
    {
        log_error("Error {} from GetGroupEntities()", (int)dcgmReturn);
        return dcgmReturn;
    }

    dcgmReturn = mpFieldGroupManager->GetFieldGroupFields(fieldGroupId, fieldIds);
    if (dcgmReturn != DCGM_ST_OK)
    {
        log_error("Got {} from mpFieldGroupManager->GetFieldGroupFields()", (int)dcgmReturn);
        return dcgmReturn;
    }

    // Remove detached entities
    DcgmNs::Utils::EraseIf(entities, [this](const dcgmGroupEntityPair_t &entity) {
        auto status = mpCacheManager->GetEntityStatus(entity.entityGroupId, entity.entityId);
        return status == DcgmEntityStatusDetached || status == DcgmEntityStatusInaccessible;
    });

    log_debug("Got {} entities and {} fields", (int)entities.size(), (int)fieldIds.size());

    bool wasFirstWatcher    = false;
    bool updateOnFirstWatch = false; /* Don't have the cache manager update after every watch. Instead,
                                        we will set shouldUpdateAllFields and UpdateAllFields at the end */
    bool shouldUpdateAllFields = false;

    for (i = 0; i < (int)entities.size(); i++)
    {
        for (j = 0; j < (int)fieldIds.size(); j++)
        {
            dcgmReturn = mpCacheManager->AddFieldWatch(entities[i].entityGroupId,
                                                       entities[i].entityId,
                                                       fieldIds[j],
                                                       monitorIntervalUsec,
                                                       maxSampleAge,
                                                       maxKeepSamples,
                                                       watcher,
                                                       false,
                                                       updateOnFirstWatch,
                                                       wasFirstWatcher);
            if (dcgmReturn != DCGM_ST_OK)
            {
                log_error("AddFieldWatch({}, {}, {}) returned {}",
                          entities[i].entityGroupId,
                          entities[i].entityId,
                          (int)fieldIds[j],
                          (int)dcgmReturn);
                retSt = dcgmReturn;
                goto GETOUT;
            }

            if (wasFirstWatcher)
            {
                shouldUpdateAllFields = true;
            }
        }
    }

    for (auto const field : fieldIds)
    {
        mpCacheManager->AddMetaGroupWatchedField(groupId == mpGroupManager->GetAllGpusGroup() ? DCGM_GROUP_ALL_GPUS
                                                                                              : groupId,
                                                 field,
                                                 monitorIntervalUsec,
                                                 maxSampleAge,
                                                 maxKeepSamples,
                                                 watcher);
    }

    if (shouldUpdateAllFields)
    {
        dcgmReturn = mpCacheManager->UpdateAllFields(true);
        if (dcgmReturn != DCGM_ST_OK)
        {
            DCGM_LOG_ERROR << "Got dcgmReturn " << dcgmReturn << " from UpdateAllFields()";
            return dcgmReturn;
        }
    }

    /* Add profiling watches after the watches exist in the cache manager so that
       quota policy is in place */
    helper_get_prof_field_ids(fieldIds, profFieldIds);

    if (!profFieldIds.empty())
    {
        dcgm_profiling_msg_watch_fields_t msg;
        memset(&msg, 0, sizeof(msg));

        if (profFieldIds.size() > DCGM_ARRAY_CAPACITY(msg.watchFields.fieldIds))
        {
            log_error("Too many prof field IDs {} for request DCGM_PROFILING_SR_WATCH_FIELDS",
                      (int)profFieldIds.size());

            retSt = DCGM_ST_GENERIC_ERROR;
            goto GETOUT;
        }

        /* Do we need to forward on a profiling watch request to the profiling module */
        mpCacheManager->GetProfModuleServicedEntities(entities);

        if (!entities.empty())
        {
            msg.header.length           = sizeof(msg);
            msg.header.moduleId         = DcgmModuleIdProfiling;
            msg.header.subCommand       = DCGM_PROFILING_SR_WATCH_FIELDS;
            msg.header.connectionId     = watcher.connectionId;
            msg.header.version          = dcgm_profiling_msg_watch_fields_version;
            msg.watchFields.version     = dcgmProfWatchFields_version;
            msg.watchFields.groupId     = (dcgmGpuGrp_t)groupId;
            msg.watchFields.numFieldIds = profFieldIds.size();
            memcpy(&msg.watchFields.fieldIds[0],
                   &profFieldIds[0],
                   profFieldIds.size() * sizeof(msg.watchFields.fieldIds[0]));
            msg.watchFields.updateFreq     = monitorIntervalUsec;
            msg.watchFields.maxKeepAge     = maxSampleAge;
            msg.watchFields.maxKeepSamples = maxKeepSamples;

            dcgmReturn = ProcessModuleCommand((dcgm_module_command_header_t *)&msg);
            if (dcgmReturn != DCGM_ST_OK)
            {
                log_error("DCGM_PROFILING_SR_WATCH_FIELDS failed with {}", dcgmReturn);
                retSt = dcgmReturn;
                goto GETOUT;
            }
        }
    }

    helperFilterSysmonEntitiesAndFields(
        sysmonMsg, entities, fieldIds, watcher, monitorIntervalUsec, maxSampleAge, maxKeepSamples);
    if (sysmonMsg.numEntities > 0 && sysmonMsg.numFieldIds > 0)
    {
        dcgmReturn = ProcessModuleCommand((dcgm_module_command_header_t *)&sysmonMsg);
        if (dcgmReturn != DCGM_ST_OK)
        {
            log_error("DCGM_SYSMON_SR_WATCH_FIELD failed with {}", dcgmReturn);
            retSt = dcgmReturn;
            goto GETOUT;
        }
    }

GETOUT:
    if (retSt != DCGM_ST_OK)
    {
        /* Clean up any watches that were established since at least one failed */
        UnwatchFieldGroup(groupId, fieldGroupId, watcher);
    }

    return retSt;
}

/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::UnwatchFieldGroup(unsigned int groupId,
                                                      dcgmFieldGrp_t fieldGroupId,
                                                      DcgmWatcher const &watcher)
{
    int i;
    int j;
    dcgmReturn_t dcgmReturn;
    dcgmReturn_t retSt = DCGM_ST_OK;
    std::vector<dcgmGroupEntityPair_t> entities;
    std::vector<unsigned short> fieldIds;
    std::vector<unsigned short> profFieldIds;
    dcgm_sysmon_msg_watch_fields_t sysmonMsg = {};

    dcgmReturn = mpGroupManager->GetGroupEntities(groupId, DcgmGroupOption::All, entities);
    if (dcgmReturn != DCGM_ST_OK)
    {
        log_error("Error {} from GetGroupEntities()", (int)dcgmReturn);
        return dcgmReturn;
    }

    dcgmReturn = mpFieldGroupManager->GetFieldGroupFields(fieldGroupId, fieldIds);
    if (dcgmReturn != DCGM_ST_OK)
    {
        log_error("Got {} from mpFieldGroupManager->GetFieldGroupFields()", (int)dcgmReturn);
        return dcgmReturn;
    }

    // Remove detached entities
    DcgmNs::Utils::EraseIf(entities, [this](const dcgmGroupEntityPair_t &entity) {
        auto status = mpCacheManager->GetEntityStatus(entity.entityGroupId, entity.entityId);
        return status == DcgmEntityStatusDetached || status == DcgmEntityStatusInaccessible;
    });

    log_debug("Got {} entities and {} fields", (int)entities.size(), (int)fieldIds.size());

    for (i = 0; i < (int)entities.size(); i++)
    {
        for (j = 0; j < (int)fieldIds.size(); j++)
        {
            dcgmReturn = mpCacheManager->RemoveFieldWatch(
                entities[i].entityGroupId, entities[i].entityId, fieldIds[j], 0, watcher);
            if (dcgmReturn != DCGM_ST_OK)
            {
                log_error("RemoveFieldWatch({}, {}, {}) returned {}",
                          entities[i].entityGroupId,
                          entities[i].entityId,
                          (int)fieldIds[j],
                          (int)dcgmReturn);
                retSt = dcgmReturn;
                /* Keep going so we don't leave watches active */
            }
        }
    }

    for (auto const field : fieldIds)
    {
        mpCacheManager->RemoveMetaGroupWatchedField(
            groupId == mpGroupManager->GetAllGpusGroup() ? DCGM_GROUP_ALL_GPUS : groupId, field, watcher);
    }

    /* Send a module command to the profiling module to unwatch any fieldIds */
    helper_get_prof_field_ids(fieldIds, profFieldIds);

    if (!profFieldIds.empty())
    {
        /* Do we need to forward on a profiling unwatch request to the profiling module? */
        mpCacheManager->GetProfModuleServicedEntities(entities);

        if (entities.empty())
        {
            DCGM_LOG_DEBUG << "No entities were serviced by the profiling module";
        }
        else
        {
            dcgm_profiling_msg_unwatch_fields_t msg;
            memset(&msg, 0, sizeof(msg));

            msg.header.length         = sizeof(msg);
            msg.header.moduleId       = DcgmModuleIdProfiling;
            msg.header.subCommand     = DCGM_PROFILING_SR_UNWATCH_FIELDS;
            msg.header.connectionId   = watcher.connectionId;
            msg.header.version        = dcgm_profiling_msg_unwatch_fields_version;
            msg.unwatchFields.version = dcgmProfUnwatchFields_version;
            msg.unwatchFields.groupId = (dcgmGpuGrp_t)groupId;

            dcgmReturn = ProcessModuleCommand((dcgm_module_command_header_t *)&msg);
            if (dcgmReturn != DCGM_ST_OK)
            {
                log_error("DCGM_PROFILING_SR_UNWATCH_FIELDS failed with {}", dcgmReturn);
                retSt = dcgmReturn;
            }
        }
    }

    // Sysmon watches
    for (auto field : fieldIds)
    {
        if (helperIsSysmonField(field))
        {
            sysmonMsg.header.length       = sizeof(sysmonMsg);
            sysmonMsg.header.moduleId     = DcgmModuleIdSysmon;
            sysmonMsg.header.subCommand   = DCGM_SYSMON_SR_UNWATCH_FIELDS;
            sysmonMsg.header.connectionId = watcher.connectionId;
            sysmonMsg.header.version      = dcgm_sysmon_msg_unwatch_fields_version;
            sysmonMsg.watcher             = watcher;

            dcgmReturn = ProcessModuleCommand((dcgm_module_command_header_t *)&sysmonMsg);
            if (dcgmReturn != DCGM_ST_OK)
            {
                log_error("DCGM_SYSMON_SR_UNWATCH_FIELD failed with {}", errorString(dcgmReturn));
                retSt = dcgmReturn;
            }
            // Only need to send message once. Exit loop
            break;
        }
    }

    return retSt;
}

/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::WatchFieldGroupAllGpus(dcgmFieldGrp_t fieldGroupId,
                                                           timelib64_t monitorIntervalUsec,
                                                           double maxSampleAge,
                                                           int maxKeepSamples,
                                                           int activeOnly,
                                                           DcgmWatcher const &watcher)
{
    int i;
    int j;
    dcgmReturn_t dcgmReturn;
    std::vector<unsigned int> gpuIds;
    std::vector<unsigned short> fieldIds;

    dcgmReturn = mpCacheManager->GetGpuIds(activeOnly, gpuIds);
    if (dcgmReturn != DCGM_ST_OK)
    {
        DCGM_LOG_ERROR << "Got " << dcgmReturn << " from mpFieldGroupManager->GetGpuIds()";
        return dcgmReturn;
    }

    dcgmReturn = mpFieldGroupManager->GetFieldGroupFields(fieldGroupId, fieldIds);
    if (dcgmReturn != DCGM_ST_OK)
    {
        log_error("Got {} from mpFieldGroupManager->GetFieldGroupFields()", (int)dcgmReturn);
        return dcgmReturn;
    }


    log_debug("Got {} gpus and {} fields", (int)gpuIds.size(), (int)fieldIds.size());

    bool wasFirstWatcher    = false;
    bool updateOnFirstWatch = false; /* Don't have the cache manager update after every watch. Instead,
                                        we will set shouldUpdateAllFields and UpdateAllFields at the end */
    bool shouldUpdateAllFields = false;

    for (i = 0; i < (int)gpuIds.size(); i++)
    {
        for (j = 0; j < (int)fieldIds.size(); j++)
        {
            dcgmReturn = mpCacheManager->AddFieldWatch(DCGM_FE_GPU,
                                                       gpuIds[i],
                                                       fieldIds[j],
                                                       monitorIntervalUsec,
                                                       maxSampleAge,
                                                       maxKeepSamples,
                                                       watcher,
                                                       false,
                                                       updateOnFirstWatch,
                                                       wasFirstWatcher);
            if (dcgmReturn != DCGM_ST_OK)
            {
                log_error("AddFieldWatch({}, {}) returned {}", (int)gpuIds[i], (int)fieldIds[j], (int)dcgmReturn);
                return DCGM_ST_GENERIC_ERROR;
            }

            if (wasFirstWatcher)
            {
                shouldUpdateAllFields = true;
            }
        }
    }

    if (shouldUpdateAllFields)
    {
        dcgmReturn = mpCacheManager->UpdateAllFields(1);
        if (dcgmReturn != DCGM_ST_OK)
        {
            DCGM_LOG_ERROR << "Got dcgmReturn " << dcgmReturn << " from UpdateAllFields()";
            return dcgmReturn;
        }
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::AddRequestWatcher(std::unique_ptr<DcgmRequest> request,
                                                      dcgm_request_id_t &requestId)
{
    if (request == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }

    std::lock_guard<std::mutex> lock(m_watchedRequestsMutex);

    m_nextWatchedRequestId++;

    /* Search for a nonzero, unused request ID. This should only take more than one
       loop if we've served more than 4 billion requests */
    while (m_nextWatchedRequestId == DCGM_REQUEST_ID_NONE
           || m_watchedRequests.find(m_nextWatchedRequestId) != m_watchedRequests.end())
    {
        m_nextWatchedRequestId++;
    }

    request->SetRequestId(m_nextWatchedRequestId);
    requestId = m_nextWatchedRequestId;

    m_watchedRequests[m_nextWatchedRequestId] = std::shared_ptr<DcgmRequest>(std::move(request));

    /* Log while we still have the lock. */
    DCGM_LOG_DEBUG << "Assigned requestId " << m_nextWatchedRequestId << " to request " << std::hex
                   << m_watchedRequests[m_nextWatchedRequestId].get();

    return DCGM_ST_OK;
}

/*****************************************************************************/
void DcgmHostEngineHandler::NotifyRequestOfCompletion(dcgm_connection_id_t connectionId, dcgm_request_id_t requestId)
{
    if (connectionId == DCGM_CONNECTION_ID_NONE)
    {
        /* Local request. Remove from map with scoped lock. */
        std::shared_ptr<DcgmRequest> removedRequest;
        {
            std::lock_guard<std::mutex> lock(m_watchedRequestsMutex);

            watchedRequests_t::iterator it = m_watchedRequests.find(requestId);
            if (it == m_watchedRequests.end())
            {
                log_error("Unable to find requestId {}", requestId);
            }
            else
            {
                /* Move the shared_ptr out so destruction happens after lock release */
                removedRequest = std::move(it->second);
                m_watchedRequests.erase(it);
                log_debug("Removed requestId {}", requestId);
            }
        }
        return;
    }

    dcgm_msg_request_notify_t msg;
    memset(&msg, 0, sizeof(msg));
    msg.requestId = requestId;

    SendRawMessageToClient(connectionId, DCGM_MSG_REQUEST_NOTIFY, requestId, &msg, sizeof(msg), DCGM_ST_OK);
}

/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::RemoveAllTrackedRequests()
{
    log_debug("Entering RemoveAllTrackedRequests");

    std::lock_guard<std::mutex> lock(m_watchedRequestsMutex);
    m_watchedRequests.clear();

    return DCGM_ST_OK;
}

/*****************************************************************************/

/*****************************************************************************
 This method is used to start DCGM Host Engine in listening mode
 *****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::RunServer(unsigned short portNumber,
                                              char const *socketPath,
                                              dcgmConnectionType_t connectionType)
{
    dcgmReturn_t dcgmReturn;

    switch (connectionType)
    {
        case DcgmConnectionTypeDomainSocket:
        {
            DcgmIpcDomainServerParams_t domainParams {};
            domainParams.domainSocketPath = socketPath;
            dcgmReturn                    = m_dcgmIpc->Init(domainParams,
                                         DcgmHostEngineHandler::StaticProcessMessage,
                                         this,
                                         DcgmHostEngineHandler::StaticProcessDisconnect,
                                         this);
            break;
        }
        case DcgmConnectionTypeTcp:
        {
            DcgmIpcTcpServerParams_t tcpParams {};
            tcpParams.bindIPAddress = socketPath;
            tcpParams.port          = portNumber;
            dcgmReturn              = m_dcgmIpc->Init(tcpParams,
                                         DcgmHostEngineHandler::StaticProcessMessage,
                                         this,
                                         DcgmHostEngineHandler::StaticProcessDisconnect,
                                         this);
            break;
        }
        case DcgmConnectionTypeVsock:
        {
            DcgmIpcVsockServerParams_t vsockParams {};
            vsockParams.port = portNumber;
            if (socketPath == nullptr || socketPath[0] == '\0')
            {
                vsockParams.bindAny = true;
            }
            else
            {
                std::string_view vsockPath(socketPath);
                unsigned int cid   = 0;
                auto [ptr, result] = std::from_chars(vsockPath.data(), vsockPath.data() + vsockPath.size(), cid);
                if (result != std::errc())
                {
                    auto error = std::make_error_code(result);
                    log_error(
                        "Invalid vsock path: ({}){}. Provided value: {}", error.value(), error.message(), socketPath);
                    return DCGM_ST_BADPARAM;
                }
                vsockParams.cid = cid;
            }
            dcgmReturn = m_dcgmIpc->Init(vsockParams,
                                         DcgmHostEngineHandler::StaticProcessMessage,
                                         this,
                                         DcgmHostEngineHandler::StaticProcessDisconnect,
                                         this);
            break;
        }
        default:
            log_error("Invalid connection type {}", std::to_underlying(connectionType));
            dcgmReturn = DCGM_ST_BADPARAM;
            break;
    }

    if (dcgmReturn != DCGM_ST_OK)
    {
        log_error("Got error {} from m_dcgmIpc->Init", errorString(dcgmReturn));
        return DCGM_ST_INIT_ERROR;
    }

    return DCGM_ST_OK;
}

/*****************************************************************************
 This method deletes the DCGM Host Engine Handler Instance
 *****************************************************************************/
void DcgmHostEngineHandler::Cleanup()
{
    if (nullptr != mpHostEngineHandlerInstance)
    {
        delete mpHostEngineHandlerInstance;
        mpHostEngineHandlerInstance = nullptr;
    }
}

/*****************************************************************************/
DcgmEntityStatus_t DcgmHostEngineHandler::GetNvSwitchEntityStatus(dcgm_field_entity_group_t entityGroupId,
                                                                  dcgm_field_eid_t entityId)
{
    // Route to NvSwitch module with proper switch ID
    dcgm_nvswitch_msg_get_entity_status_t nvsMsg {};
    nvsMsg.header.length     = sizeof(nvsMsg);
    nvsMsg.header.moduleId   = DcgmModuleIdNvSwitch;
    nvsMsg.header.subCommand = DCGM_NVSWITCH_SR_GET_ENTITY_STATUS;
    nvsMsg.header.version    = dcgm_nvswitch_msg_get_entity_status_version;
    nvsMsg.entityId          = entityId;
    nvsMsg.entityGroupId     = entityGroupId;

    if (dcgmReturn_t dcgmReturn = ProcessModuleCommand(&nvsMsg.header); dcgmReturn == DCGM_ST_OK)
    {
        log_verbose("Switch link validation successful, eg {} entityId {}, status={}",
                    entityGroupId,
                    entityId,
                    nvsMsg.entityStatus);
        return nvsMsg.entityStatus;
    }
    else
    {
        log_error("Got {} from DCGM_NVSWITCH_SR_GET_ENTITY_STATUS of entityId {}", errorString(dcgmReturn), entityId);
        return DcgmEntityStatusUnknown;
    }
}

/*****************************************************************************/
DcgmEntityStatus_t DcgmHostEngineHandler::GetEntityStatus(dcgm_field_entity_group_t entityGroupId,
                                                          dcgm_field_eid_t entityId)
{
    if (entityGroupId == DCGM_FE_LINK)
    {
        // Decode the link entity to determine type
        dcgm_link_t link {};
        link.raw = entityId;

        log_debug("DCGM_FE_LINK entityId {} -> type {} gpuId {} index {} ",
                  entityId,
                  static_cast<uint8_t>(link.parsed.type),
                  static_cast<uint8_t>(link.parsed.gpuId),
                  static_cast<uint16_t>(link.parsed.index));

        switch (link.parsed.type)
        {
            case DCGM_FE_GPU:
            {
                // Validate GPU exists using CacheManager
                if (!mpCacheManager->GetIsValidEntityId(DCGM_FE_GPU, link.parsed.gpuId))
                {
                    log_warning("GPU {} is invalid", static_cast<uint8_t>(link.parsed.gpuId));
                    return DcgmEntityStatusUnknown;
                }

                log_verbose("Valid GPU link");
                return DcgmEntityStatusOk;
            }

            /* Other link types currently handled by the NvSwitch module */
            default:
                return GetNvSwitchEntityStatus(entityGroupId, entityId);
        }
    }
    else if ((entityGroupId == DCGM_FE_SWITCH) || (entityGroupId == DCGM_FE_CONNECTX))
    {
        return GetNvSwitchEntityStatus(entityGroupId, entityId);
    }
    if ((entityGroupId == DCGM_FE_CPU) || (entityGroupId == DCGM_FE_CPU_CORE))
    {
        dcgm_sysmon_msg_get_entity_status_t msg {};
        msg.header.length       = sizeof(msg);
        msg.header.moduleId     = DcgmModuleIdSysmon;
        msg.header.subCommand   = DCGM_SYSMON_SR_GET_ENTITY_STATUS;
        msg.header.version      = dcgm_sysmon_msg_get_entity_status_version;
        msg.entityId            = entityId;
        msg.entityGroupId       = entityGroupId;
        dcgmReturn_t dcgmReturn = ProcessModuleCommand(&msg.header);
        if (dcgmReturn == DCGM_ST_OK)
        {
            return msg.entityStatus;
        }
        else
        {
            DCGM_LOG_ERROR << "Got " << errorString(dcgmReturn)
                           << " from DCGM_NVSWITCH_SR_GET_ENTITY_STATUS of entityId " << entityId;
            return DcgmEntityStatusUnknown;
        }
    }
    else
    {
        return mpCacheManager->GetEntityStatus(entityGroupId, entityId);
    }
}

/*****************************************************************************/
void DcgmHostEngineHandler::StaticProcessDisconnect(dcgm_connection_id_t connectionId, void *userData)
{
    DcgmHostEngineHandler *he = (DcgmHostEngineHandler *)userData;
    he->OnConnectionRemove(connectionId);
}

/*****************************************************************************/
void DcgmHostEngineHandler::SetServiceAccount(char const *serviceAccount)
{
    if (serviceAccount != nullptr)
    {
        m_serviceAccount = std::string(serviceAccount);
    }
    else
    {
        m_serviceAccount.clear();
    }
}


std::string const &DcgmHostEngineHandler::GetServiceAccount() const
{
    return m_serviceAccount;
}

bool DcgmHostEngineHandler::UsingInjectionNvml() const
{
    return m_usingInjectionNvml;
}

dcgmReturn_t DcgmHostEngineHandler::ProcessPauseResume(dcgm_core_msg_pause_resume_v1 *msg)
{
    if (msg == nullptr)
    {
        log_error("Invalid parameter: msg is nullptr");
        return DCGM_ST_BADPARAM;
    }
    if (msg->header.version != dcgm_core_msg_pause_resume_version1)
    {
        log_error("Invalid parameter: msg version is %d, expected %d",
                  msg->header.version,
                  dcgm_core_msg_pause_resume_version1);
        return DCGM_ST_BADPARAM;
    }
    return msg->pause ? Pause() : Resume();
}

dcgmReturn_t DcgmHostEngineHandler::Pause()
{
    bool allPaused = true;

    /* The core module should be the last one to pause */
    for (auto &module : m_modules | std::views::filter([](auto const &m) { return m.id != DcgmModuleIdCore; }))
    {
        if (auto ret = PauseModule(module.id); ret != DCGM_ST_OK)
        {
            allPaused = false;
        }
    }

    if (auto ret = PauseModule(DcgmModuleIdCore); ret != DCGM_ST_OK)
    {
        allPaused = false;
    }

    return allPaused ? DCGM_ST_OK : DCGM_ST_GENERIC_ERROR;
}

dcgmReturn_t DcgmHostEngineHandler::Resume()
{
    bool allResumed = true;

    for (auto &module : m_modules)
    {
        /* The core module must be resumed first */
        if (auto ret = ResumeModule(module.id); ret != DCGM_ST_OK)
        {
            allResumed = false;
        }
    }

    return allResumed ? DCGM_ST_OK : DCGM_ST_GENERIC_ERROR;
}

dcgmReturn_t DcgmHostEngineHandler::PauseModule(dcgmModuleId_t moduleId)
{
    auto &module = m_modules[moduleId];

    if ((module.status != DcgmModuleStatusLoaded) && (module.status != DcgmModuleStatusReloadable))
    {
        return DCGM_ST_OK;
    }

    module.status = DcgmModuleStatusLoaded; /* If it was reloadable. */

    dcgm_core_msg_pause_resume_v1 msg;
    memset(&msg, 0, sizeof(msg));
    msg.header.length       = sizeof(msg);
    msg.header.moduleId     = DcgmModuleIdCore;
    msg.header.version      = dcgm_core_msg_pause_resume_version1;
    msg.header.subCommand   = DCGM_CORE_SR_PAUSE_RESUME;
    msg.pause               = true;
    msg.header.connectionId = 0; // Not used

    if (auto const ret = SendModuleMessage(module.id, &msg.header); ret != DCGM_ST_OK)
    {
        if (ret == DCGM_ST_UNINITIALIZED)
        {
            log_debug("Skipping pause of module {} because it is not initialized", module.id);
            return DCGM_ST_OK;
        }

        log_error("Failed to send pause message to module {}: ({}){}", module.id, ret, errorString(ret));
        return ret;
    }

    module.status = DcgmModuleStatusPaused;

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmHostEngineHandler::ResumeModule(dcgmModuleId_t moduleId)
{
    auto &module = m_modules[moduleId];

    if (module.status != DcgmModuleStatusPaused)
    {
        log_debug("Skip resume of module {} because it is not paused", module.id);
        return DCGM_ST_OK;
    }

    dcgm_core_msg_pause_resume_v1 msg;
    memset(&msg, 0, sizeof(msg));
    msg.header.length       = sizeof(msg);
    msg.header.moduleId     = DcgmModuleIdCore;
    msg.header.version      = dcgm_core_msg_pause_resume_version1;
    msg.header.subCommand   = DCGM_CORE_SR_PAUSE_RESUME;
    msg.pause               = false;
    msg.header.connectionId = 0; // Not used

    if (auto ret = SendModuleMessage(module.id, &msg.header); ret != DCGM_ST_OK)
    {
        log_error("Failed to send resume message to module {}: ({}){}", module.id, ret, errorString(ret));
        return ret;
    }

    module.status = DcgmModuleStatusLoaded;

    return DCGM_ST_OK;
}


std::optional<unsigned int> DcgmHostEngineHandler::ReserveResources()
{
    return m_resourceManager.ReserveResources();
}

bool DcgmHostEngineHandler::FreeResources(unsigned int const &token)
{
    return m_resourceManager.FreeResources(token);
}

dcgmReturn_t DcgmHostEngineHandler::ApplyModuleDenylist(unsigned int const *denyList, unsigned int denyListCount)
{
    if (denyList == nullptr)
    {
        log_error("Invalid parameter: denyList is nullptr");
        return DCGM_ST_BADPARAM;
    }

    for (unsigned int i = 0; i < denyListCount; i++)
    {
        if (denyList[i] == DcgmModuleIdCore)
        {
            log_error("Ignored denylist request for core module.");
            continue;
        }

        if (denyList[i] >= DcgmModuleIdCount)
        {
            log_error("Out of range module ID given: {}", denyList[i]);
            return DCGM_ST_BADPARAM;
        }

        log_error("Module {} was added to the denylist.", denyList[i]);
        m_modules[denyList[i]].status = DcgmModuleStatusDenylisted;
    }

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmHostEngineHandler::ChildProcessSpawn(dcgmChildProcessParams_t const &params,
                                                      ChildProcessHandle_t &handle,
                                                      int &pid)
{
    return m_childProcessManager.Spawn(params, handle, pid);
}

dcgmReturn_t DcgmHostEngineHandler::ChildProcessStop(ChildProcessHandle_t handle, bool force)
{
    return m_childProcessManager.Stop(handle, force);
}

dcgmReturn_t DcgmHostEngineHandler::ChildProcessGetStatus(ChildProcessHandle_t handle, dcgmChildProcessStatus_t &status)
{
    return m_childProcessManager.GetStatus(handle, status);
}

dcgmReturn_t DcgmHostEngineHandler::ChildProcessWait(ChildProcessHandle_t handle, int timeoutSec)
{
    return m_childProcessManager.Wait(handle, timeoutSec);
}

dcgmReturn_t DcgmHostEngineHandler::ChildProcessDestroy(ChildProcessHandle_t handle, int sigTermTimeoutSec)
{
    return m_childProcessManager.Destroy(handle, sigTermTimeoutSec);
}

dcgmReturn_t DcgmHostEngineHandler::ChildProcessGetStdErrHandle(ChildProcessHandle_t handle, int &fd)
{
    return m_childProcessManager.GetStdErrHandle(handle, fd);
}

dcgmReturn_t DcgmHostEngineHandler::ChildProcessGetStdOutHandle(ChildProcessHandle_t handle, int &fd)
{
    return m_childProcessManager.GetStdOutHandle(handle, fd);
}

dcgmReturn_t DcgmHostEngineHandler::ChildProcessGetDataChannelHandle(ChildProcessHandle_t handle, int &fd)
{
    return m_childProcessManager.GetDataChannelHandle(handle, fd);
}

dcgmReturn_t DcgmHostEngineHandler::ChildProcessManagerReset()
{
    return m_childProcessManager.Reset();
}

bool DcgmHostEngineHandler::IsNvmlLoaded() const
{
    return m_nvmlLoaded.load(std::memory_order_acquire);
}

dcgmReturn_t DcgmHostEngineHandler::AttachDriver()
{
    auto task = m_nvmlSystemEventThread->AttachDriver();
    if (!task.has_value())
    {
        log_error("Failed to attach driver");
        return DCGM_ST_GENERIC_ERROR;
    }
    return task->get();
}

dcgmReturn_t DcgmHostEngineHandler::DetachDriver()
{
    auto task = m_nvmlSystemEventThread->DetachDriver();
    if (!task.has_value())
    {
        log_error("Failed to detach driver");
        return DCGM_ST_GENERIC_ERROR;
    }
    return task->get();
}

void DcgmHostEngineHandler::InitializeHangDetection()
{
    if (std::getenv(DCGM_HANGDETECT_DISABLE) != nullptr)
    {
        m_hangDetectDisabled = true;
        log_info("Hang detection disabled");
        return;
    }

    bool terminateOnHang                      = false;
    bool constexpr reportUserspaceHangs       = false;
    std::chrono::seconds hangDetectExpirySec  = std::chrono::seconds(DEFAULT_HANG_DETECT_EXPIRY_SEC);
    std::chrono::seconds hangCheckIntervalSec = std::chrono::seconds(DEFAULT_HANG_DETECT_CHECK_INTERVAL);

    if (std::getenv(DCGM_HANGDETECT_TERMINATE) != nullptr)
    {
        terminateOnHang = true;
    }

    if (auto const expirySecStr = std::getenv(DCGM_HANGDETECT_EXPIRY_SEC); expirySecStr != nullptr)
    {
        unsigned int expirySec {};
        auto const [_, ec] = std::from_chars(expirySecStr, expirySecStr + std::strlen(expirySecStr), expirySec);
        if (ec != std::errc())
        {
            log_warning(
                "Invalid {} value '{}'. Must be a non-negative number.", DCGM_HANGDETECT_EXPIRY_SEC, expirySecStr);
        }
        else
        {
            hangDetectExpirySec = std::chrono::seconds(expirySec);
        }
    }

    m_hangDetectHandler = std::make_unique<EngineHangDetectHandler>();
    m_hangDetectHandler->SetTaskContextManager(static_cast<class TaskContextManager const *>(GetTaskContextManager()));
    m_hangDetectHandler->SetTerminateOnHang(terminateOnHang);

    m_detector = std::make_unique<HangDetect>();

    m_monitor = std::make_unique<HangDetectMonitor>(*m_detector, hangCheckIntervalSec, hangDetectExpirySec);
    m_monitor->SetReportUserspaceHangs(reportUserspaceHangs);
    m_monitor->SetHangDetectedHandler(m_hangDetectHandler.get());
    m_monitor->StartMonitoring();
    log_debug("Hang detection enabled");
}
