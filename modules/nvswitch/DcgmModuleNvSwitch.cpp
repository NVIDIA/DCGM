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
#include "DcgmModuleNvSwitch.h"

#include <DcgmLogging.h>
#include <cstdlib>
#include <dcgm_api_export.h>
#include <dcgm_structs.h>

namespace DcgmNs
{
static std::unique_ptr<DcgmNvSwitchManagerBase> createSwitchManager(dcgmCoreCallbacks_t *dcc)
{
    std::unique_ptr<DcgmNvSwitchManagerBase> nvswitchMgr;

    log_debug("Loading NVSDM");
    nvswitchMgr      = std::make_unique<DcgmNvsdmManager>(dcc);
    dcgmReturn_t ret = nvswitchMgr->Init();
    if (ret == DCGM_ST_OK)
    {
        return nvswitchMgr;
    }

    log_debug("Loading NSCQ");
    nvswitchMgr = std::make_unique<DcgmNscqManager>(dcc);
    ret         = nvswitchMgr->Init();
    if (ret == DCGM_ST_OK)
    {
        return nvswitchMgr;
    }

    log_warning("Could not initialize NSCQ. Ret: {}", errorString(ret));
    return nvswitchMgr;
}

DcgmModuleNvSwitch::DcgmModuleNvSwitch(dcgmCoreCallbacks_t &dcc)
    : DcgmModuleWithCoreProxy(dcc)
    , m_nvswitchMgrPtr(createSwitchManager(&dcc))
    , m_nvswitchMgr(*m_nvswitchMgrPtr)
    , m_lastLinkStatusUpdateUsec(0)
{
    DCGM_LOG_DEBUG << "Constructing NvSwitch Module";

    /* Start our TaskRunner now that we've survived initialization
     *
     * Start **must** come after Init. Otherwise two threads will attempt to
     *   - load NSCQ
     *   - use NSCQ to rescan devices
     * That is undefined behaviour and causes lockups most of the time.
     */
    int st = Start();
    if (st)
    {
        DCGM_LOG_ERROR << "Got error " << st << " when trying to start the task runner";
        throw std::runtime_error("Unable to start a DcgmTaskRunner");
    }
}

DcgmModuleNvSwitch::~DcgmModuleNvSwitch()
{
    try
    {
        if (StopAndWait(60000))
        {
            DCGM_LOG_WARNING << "Not all threads for the NVSwitch module exited correctly; exiting anyway";
            Kill();
        }
    }
    catch (std::exception const &ex)
    {
        DCGM_LOG_ERROR << "Exception caught while stopping the NVSwitch module: " << ex.what();
    }
    catch (...)
    {
        DCGM_LOG_ERROR << "Unknown exception caught while stopping the NVSwitch module";
    }
}

/*************************************************************************/
dcgmReturn_t DcgmModuleNvSwitch::ProcessMessage(dcgm_module_command_header_t *moduleCommand)
{
    dcgmReturn_t retSt       = DCGM_ST_OK;
    bool processInTaskRunner = false;

    if (moduleCommand->moduleId == DcgmModuleIdCore)
    {
        processInTaskRunner = true;
    }
    else if (moduleCommand->moduleId != DcgmModuleIdNvSwitch)
    {
        DCGM_LOG_ERROR << "Unexpected module command for module " << moduleCommand->moduleId;
        return DCGM_ST_BADPARAM;
    }
    else /* NvSwitch module request */
    {
        switch (moduleCommand->subCommand)
        {
            /* Messages to process on the task runner */
            case DCGM_NVSWITCH_SR_GET_SWITCH_IDS:
            case DCGM_NVSWITCH_SR_CREATE_FAKE_SWITCH:
            case DCGM_NVSWITCH_SR_WATCH_FIELD:
            case DCGM_NVSWITCH_SR_UNWATCH_FIELD:
            case DCGM_NVSWITCH_SR_GET_LINK_STATES:
            case DCGM_NVSWITCH_SR_GET_ALL_LINK_STATES:
            case DCGM_NVSWITCH_SR_SET_LINK_STATE:
            case DCGM_NVSWITCH_SR_GET_ENTITY_STATUS:
            case DCGM_NVSWITCH_SR_GET_LINK_IDS:
            case DCGM_NVSWITCH_SR_GET_BACKEND:
            case DCGM_NVSWITCH_SR_GET_ENTITIES_IDS:
            {
                processInTaskRunner = true;
            }

            break;
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
            DCGM_LOG_ERROR << "Unable to enqueue NvSwitch Module task";
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
dcgmReturn_t DcgmModuleNvSwitch::ProcessMessageFromTaskRunner(dcgm_module_command_header_t *moduleCommand)
{
    dcgmReturn_t retSt = DCGM_ST_OK;

    if (moduleCommand->moduleId == DcgmModuleIdCore)
    {
        retSt = ProcessCoreMessage(moduleCommand);
    }
    else if (moduleCommand->moduleId != DcgmModuleIdNvSwitch)
    {
        log_error("Unexpected module command for module {}", moduleCommand->moduleId);
        return DCGM_ST_BADPARAM;
    }
    else /* NvSwitch module request */
    {
        switch (moduleCommand->subCommand)
        {
            case DCGM_NVSWITCH_SR_GET_SWITCH_IDS:
            {
                retSt = CheckVersion(moduleCommand, dcgm_nvswitch_msg_get_switches_version);
                if (retSt != DCGM_ST_OK)
                {
                    DCGM_LOG_ERROR << "Version mismatch " << moduleCommand->version
                                   << " != " << dcgm_nvswitch_msg_get_switches_version;
                    return retSt;
                }

                retSt = ProcessGetSwitchIds((dcgm_nvswitch_msg_get_switches_v1 *)moduleCommand);
                break;
            }
            case DCGM_NVSWITCH_SR_CREATE_FAKE_SWITCH:
            {
                retSt = CheckVersion(moduleCommand, dcgm_nvswitch_msg_create_fake_switch_version);
                if (retSt != DCGM_ST_OK)
                {
                    DCGM_LOG_ERROR << "Version mismatch " << moduleCommand->version
                                   << " != " << dcgm_nvswitch_msg_create_fake_switch_version;
                    return retSt;
                }

                dcgm_nvswitch_msg_create_fake_switch_v1 *cfs = (dcgm_nvswitch_msg_create_fake_switch_v1 *)moduleCommand;
                cfs->numCreated                              = cfs->numToCreate;
                retSt = m_nvswitchMgr.CreateFakeSwitches(cfs->numCreated, cfs->switchIds);
                break;
            }

            case DCGM_NVSWITCH_SR_WATCH_FIELD:
            {
                retSt = CheckVersion(moduleCommand, dcgm_nvswitch_msg_watch_field_version);
                if (retSt != DCGM_ST_OK)
                {
                    DCGM_LOG_ERROR << "Version mismatch " << moduleCommand->version
                                   << " != " << dcgm_nvswitch_msg_unwatch_field_version;
                    return retSt;
                }

                retSt = ProcessWatchField((dcgm_nvswitch_msg_watch_field_t *)moduleCommand);
                break;
            }
            case DCGM_NVSWITCH_SR_UNWATCH_FIELD:
            {
                retSt = CheckVersion(moduleCommand, dcgm_nvswitch_msg_unwatch_field_version);
                if (retSt != DCGM_ST_OK)
                {
                    DCGM_LOG_ERROR << "Version mismatch " << moduleCommand->version
                                   << " != " << dcgm_nvswitch_msg_unwatch_field_version;
                    return retSt;
                }

                retSt = ProcessUnwatchField((dcgm_nvswitch_msg_unwatch_field_t *)moduleCommand);
                break;
            }

            case DCGM_NVSWITCH_SR_GET_LINK_STATES:
            {
                retSt = CheckVersion(moduleCommand, dcgm_nvswitch_msg_get_link_states_version);
                if (retSt != DCGM_ST_OK)
                {
                    DCGM_LOG_ERROR << "Version mismatch " << moduleCommand->version
                                   << " != " << dcgm_nvswitch_msg_get_link_states_version;
                    return retSt;
                }

                retSt = ProcessGetLinkStates((dcgm_nvswitch_msg_get_link_states_t *)moduleCommand);
                break;
            }

            case DCGM_NVSWITCH_SR_GET_ALL_LINK_STATES:
            {
                retSt = CheckVersion(moduleCommand, dcgm_nvswitch_msg_get_all_link_states_version);
                if (retSt != DCGM_ST_OK)
                {
                    DCGM_LOG_ERROR << "Version mismatch " << moduleCommand->version
                                   << " != " << dcgm_nvswitch_msg_get_all_link_states_version;
                    return retSt;
                }

                retSt = ProcessGetAllLinkStates((dcgm_nvswitch_msg_get_all_link_states_t *)moduleCommand);
                break;
            }

            case DCGM_NVSWITCH_SR_SET_LINK_STATE:
            {
                retSt = CheckVersion(moduleCommand, dcgm_nvswitch_msg_set_link_state_version);
                if (retSt != DCGM_ST_OK)
                {
                    DCGM_LOG_ERROR << "Version mismatch " << moduleCommand->version
                                   << " != " << dcgm_nvswitch_msg_set_link_state_version;
                    return retSt;
                }

                retSt = ProcessSetEntityNvLinkLinkState((dcgm_nvswitch_msg_set_link_state_t *)moduleCommand);
                break;
            }

            case DCGM_NVSWITCH_SR_GET_ENTITY_STATUS:
            {
                retSt = CheckVersion(moduleCommand, dcgm_nvswitch_msg_get_entity_status_version);
                if (retSt != DCGM_ST_OK)
                {
                    DCGM_LOG_ERROR << "Version mismatch " << moduleCommand->version
                                   << " != " << dcgm_nvswitch_msg_get_entity_status_version;
                    return retSt;
                }

                retSt = ProcessGetEntityStatus((dcgm_nvswitch_msg_get_entity_status_t *)moduleCommand);
                break;
            }
            case DCGM_NVSWITCH_SR_GET_LINK_IDS:
            {
                retSt = CheckVersion(moduleCommand, dcgm_nvswitch_msg_get_links_version);
                if (retSt != DCGM_ST_OK)
                {
                    DCGM_LOG_ERROR << "Version mismatch " << moduleCommand->version
                                   << " != " << dcgm_nvswitch_msg_get_links_version;
                    return retSt;
                }

                retSt = ProcessGetLinkIds((dcgm_nvswitch_msg_get_links_v1 *)moduleCommand);
                break;
            }

            case DCGM_NVSWITCH_SR_GET_BACKEND:
            {
                retSt = CheckVersion(moduleCommand, dcgm_nvswitch_msg_get_backend_version);
                if (retSt != DCGM_ST_OK)
                {
                    DCGM_LOG_ERROR << "Version mismatch " << moduleCommand->version
                                   << " != " << dcgm_nvswitch_msg_get_backend_version;
                    return retSt;
                }

                retSt = ProcessGetBackend((dcgm_nvswitch_msg_get_backend_t *)moduleCommand);
                break;
            }

            case DCGM_NVSWITCH_SR_GET_ENTITIES_IDS:
            {
                retSt = CheckVersion(moduleCommand, dcgm_nvswitch_msg_get_entities_ids_version);
                if (retSt != DCGM_ST_OK)
                {
                    log_error("Version mismatch, [{}] != [{}].",
                              moduleCommand->version,
                              dcgm_nvswitch_msg_get_entities_ids_version);
                    return retSt;
                }

                retSt = ProcessGetEntityIds(reinterpret_cast<dcgm_nvswitch_msg_get_entities_ids_t *>(moduleCommand));
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
            [[maybe_unused]] auto _ = Enqueue(
                DcgmNs::make_task("Call RunOnce after all events are processed", [this]() { TryRunOnce(true); }));
        }
    }

    return retSt;
}

/*************************************************************************/
dcgmReturn_t DcgmModuleNvSwitch::ProcessGetSwitchIds(dcgm_nvswitch_msg_get_switches_v1 *msg)
{
    msg->switchCount = DCGM_MAX_NUM_SWITCHES;
    return m_nvswitchMgr.GetNvSwitchList(msg->switchCount, msg->switchIds, msg->flags);
}

/*************************************************************************/
dcgmReturn_t DcgmModuleNvSwitch::ProcessGetLinkIds(dcgm_nvswitch_msg_get_links_v1 *msg)
{
    msg->linkCount = DCGM_NVLINK_MAX_LINKS_PER_NVSWITCH;
    return m_nvswitchMgr.GetNvLinkList(msg->linkCount, msg->linkIds, msg->flags);
}

/*************************************************************************/
dcgmReturn_t DcgmModuleNvSwitch::ProcessGetBackend(dcgm_nvswitch_msg_get_backend_t *msg)
{
    return m_nvswitchMgr.GetBackend(msg);
}

dcgmReturn_t DcgmModuleNvSwitch::ProcessGetEntityIds(dcgm_nvswitch_msg_get_entities_ids_t *msg)
{
    msg->entitiesCount = DCGM_NVLINK_MAX_ENTITIES;
    return m_nvswitchMgr.GetEntityList(msg->entitiesCount, msg->entities, msg->entityGroup, msg->flags);
}

/*****************************************************************************/
dcgmReturn_t DcgmModuleNvSwitch::ProcessWatchField(WatchFieldMessage msg)
{
    if (msg->numFieldIds >= NVSWITCH_MSG_MAX_WATCH_FIELD_IDS)
    {
        DCGM_LOG_ERROR << "Invalid numFieldIds";
        return DCGM_ST_BADPARAM;
    }

    return m_nvswitchMgr.WatchField(msg->entityGroupId,
                                    msg->entityId,
                                    msg->numFieldIds,
                                    msg->fieldIds,
                                    msg->updateIntervalUsec,
                                    msg->watcherType,
                                    msg->connectionId,
                                    false);
}

/*****************************************************************************/
dcgmReturn_t DcgmModuleNvSwitch::ProcessUnwatchField(UnwatchFieldMessage msg)
{
    return m_nvswitchMgr.UnwatchField(msg->watcherType, msg->connectionId);
}

/*****************************************************************************/
dcgmReturn_t DcgmModuleNvSwitch::ProcessGetLinkStates(dcgm_nvswitch_msg_get_link_states_t *msg)
{
    return m_nvswitchMgr.GetLinkStates(msg);
}

/*****************************************************************************/
dcgmReturn_t DcgmModuleNvSwitch::ProcessGetAllLinkStates(dcgm_nvswitch_msg_get_all_link_states_t *msg)
{
    return m_nvswitchMgr.GetAllLinkStates(msg);
}

/*****************************************************************************/
dcgmReturn_t DcgmModuleNvSwitch::SetEntityNvLinkLinkState(dcgm_nvswitch_msg_set_link_state_t *msg)
{
    return m_nvswitchMgr.SetEntityNvLinkLinkState(msg);
}

/*****************************************************************************/
dcgmReturn_t DcgmModuleNvSwitch::ProcessGetEntityStatus(dcgm_nvswitch_msg_get_entity_status_t *msg)
{
    return m_nvswitchMgr.GetEntityStatus(msg);
}

/*****************************************************************************/
dcgmReturn_t DcgmModuleNvSwitch::ProcessSetEntityNvLinkLinkState(dcgm_nvswitch_msg_set_link_state_t *msg)
{
    return m_nvswitchMgr.SetEntityNvLinkLinkState(msg);
}

/*****************************************************************************/
dcgmReturn_t DcgmModuleNvSwitch::ProcessClientDisconnect(dcgm_core_msg_client_disconnect_t *msg)
{
    DCGM_LOG_INFO << "Unwatching fields watched by connection " << msg->connectionId;
    return m_nvswitchMgr.UnwatchField(DcgmWatcherTypeClient, msg->connectionId);
}

/*****************************************************************************/
dcgmReturn_t DcgmModuleNvSwitch::ProcessCoreMessage(dcgm_module_command_header_t *moduleCommand)
{
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

void DcgmModuleNvSwitch::RescanDevicesState()
{
    log_debug("Rescanning devices states");
    auto dcgmReturn = m_nvswitchMgr.ReadNvSwitchStatusAllSwitches();
    if (dcgmReturn != DCGM_ST_OK && dcgmReturn != DCGM_ST_PAUSED)
    {
        log_warning("ReadNvSwitchStatusAllSwitches() returned: [{}].", errorString(dcgmReturn));
    }
    dcgmReturn = m_nvswitchMgr.ReadIbCxStatusAllIbCxCards();
    if (dcgmReturn != DCGM_ST_OK && dcgmReturn != DCGM_ST_PAUSED && dcgmReturn != DCGM_ST_NOT_SUPPORTED)
    {
        log_warning("ReadIbCxStatusAllIbCxCards() returned: [{}].", errorString(dcgmReturn));
    }
}

/*****************************************************************************/
unsigned int DcgmModuleNvSwitch::RunOnce()
{
    timelib64_t nextUpdateTimeUsec;
    timelib64_t untilNextLinkStatusUsec;                       /* How long until our next switch status rescan */
    timelib64_t const linkStatusRescanIntervalUsec = 30000000; /* How long until our next switch status rescan */
    timelib64_t const minUpdateIntervalUsec        = 1000;     /* Minimum update interval in microseconds */

    /* Update link statuses every 30 seconds */
    timelib64_t now         = timelib_usecSince1970();
    untilNextLinkStatusUsec = -((now - m_lastLinkStatusUpdateUsec) - linkStatusRescanIntervalUsec);
    if (untilNextLinkStatusUsec <= 0)
    {
        RescanDevicesState();
        m_lastLinkStatusUpdateUsec = now;
        untilNextLinkStatusUsec    = linkStatusRescanIntervalUsec;
    }

    m_nvswitchMgr.UpdateFields(nextUpdateTimeUsec, now);
    if (nextUpdateTimeUsec == 0)
    {
        /*
         * If there are no watches, returned value for nextUpdateTimeUsec will be 0. Set it equal to
         * linkStatusRescanIntervalUsec i.e 30 seconds.
         */
        nextUpdateTimeUsec = linkStatusRescanIntervalUsec;
    }
    else
    {
        /*
         * Determine the absolute time of the next update, then convert to relative time.
         * If nextUpdateTimeUsec is lower than the minimum update interval, set it to the minimum update interval.
         */
        nextUpdateTimeUsec = nextUpdateTimeUsec - now;
        if (nextUpdateTimeUsec < minUpdateIntervalUsec)
        {
            log_info(
                "Next update time {} usec, is below minimum update interval {} usec, setting it to minimum update interval.",
                nextUpdateTimeUsec,
                minUpdateIntervalUsec);
            nextUpdateTimeUsec = minUpdateIntervalUsec;
        }
    }

    /* Adjust our nextUpdateTimeUsec for whichever is sooner between when watches will
      update again or when we need to rescan links */
    nextUpdateTimeUsec = std::min(nextUpdateTimeUsec, untilNextLinkStatusUsec);

    log_verbose("Next update in {} usec.", nextUpdateTimeUsec);
    return nextUpdateTimeUsec / 1000;
}

/*****************************************************************************/
std::chrono::system_clock::time_point DcgmModuleNvSwitch::TryRunOnce(bool forceRun)
{
    if (forceRun || (std::chrono::system_clock::now() > m_nextWakeup))
    {
        m_runInterval = std::chrono::milliseconds(RunOnce());
        m_nextWakeup  = std::chrono::system_clock::now() + m_runInterval;
        SetRunInterval(m_runInterval);
    }
    return m_nextWakeup;
}

/*****************************************************************************/
void DcgmModuleNvSwitch::run()
{
    using DcgmNs::TaskRunner;
    TryRunOnce(true);
    while (ShouldStop() == 0)
    {
        if (TaskRunner::Run(true) != TaskRunner::RunResult::Ok)
        {
            break;
        }

        TryRunOnce(false);
    }
}
dcgmReturn_t DcgmModuleNvSwitch::ProcessPauseResumeMessage(PauseResumeMessage msg)
{
    log_debug("Got {} message", (msg->pause ? "pause" : "resume"));
    return msg->pause ? m_nvswitchMgr.Pause() : m_nvswitchMgr.Resume();
}

extern "C" {
/*************************************************************************/
DCGM_PUBLIC_API DcgmModule *dcgm_alloc_module_instance(dcgmCoreCallbacks_t *dcc)
{
    return SafeWrapper([=] { return new DcgmModuleNvSwitch(*dcc); });
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
