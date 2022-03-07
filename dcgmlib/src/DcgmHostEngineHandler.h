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
 * File:   DcgmHostEngineHandler.h
 */

#ifndef DCGMHOSTENGINEHANDLER_H
#define DCGMHOSTENGINEHANDLER_H

#include "DcgmCacheManager.h"
#include "DcgmCoreCommunication.h"
#include "DcgmFieldGroup.h"
#include "DcgmGroupManager.h"
#include "DcgmIpc.h"
#include "DcgmModule.h"
#include "DcgmProtobuf.h"
#include "DcgmRequest.h"
#include "DcgmWatcher.h"
#include "dcgm_agent.h"
#include <core/DcgmModuleCore.h>
#include <dcgm_core_communication.h>
#include <iostream>
#include <mutex>
#include <unordered_map>

/* Module status structure */
typedef struct dcgmhe_module_info_t
{
    dcgmModuleId_t id;                /* ID of this module  */
    dcgmModuleStatus_t status;        /* Status of this module */
    DcgmModule *ptr;                  /* Pointer to the loaded class of this module */
    const char *filename;             /* Filename for this module like libdcgmmodulehealth.so */
    void *dlopenPtr;                  /* Pointer to this loaded module returned by dlopen(). NULL if not loaded */
    dcgmModuleAlloc_f allocCB;        /* Module function for allocating a DcgmModule object. NULL if not set */
    dcgmModuleFree_f freeCB;          /* Module function for freeing a DcgmModule object. NULL if not set */
    dcgmModuleProcessMessage_f msgCB; /* Module function for receiving/processing messages. NULL if not set */
} dcgmhe_module_info_t, *dcgmhe_module_info_p;

/* Job definition */
typedef struct
{
    unsigned int groupId;
    timelib64_t startTime;
    timelib64_t endTime;
} jobRecord_t;


class DcgmHostEngineHandler
{
private:
    static const int DCGM_HE_NUM_WORKERS = 2; /* How many worker threads to use for processing
                                                 user data */

public:
    /*****************************************************************************
     * This method is used to initialize DCGM HostEngineHandler
     * @param mode
     * @return
     *****************************************************************************/
    static DcgmHostEngineHandler *Init(dcgmStartEmbeddedV2Params_v1 params);

    /*****************************************************************************
     This method is used to get Instance of HostEngine Handler.
     There will be just one instance for the node
     *****************************************************************************/
    static DcgmHostEngineHandler *Instance();

    /*****************************************************************************
     This method is used to run the server on host engine side.
     Must be invoked from HostEngine to start the server to listen for
     connections.
     The corresponding "C" API will not be in the public header for the agent.
     The "C" API for this method will be part of Internal control APIs which can
     be invoked by NV Host Engine
     *****************************************************************************/
    dcgmReturn_t RunServer(unsigned short portNumber, char const *socketPath, unsigned int isConnectionTCP);

    /*****************************************************************************
     * This method is used to process one or more commands at the host engine.
     * For Host Engine, this method is intended to be a common processing method
     * for both Embedded and Stand-alone use case.
     * @param pVecCmdsToProcess : Serves as both Input and Output argument
     * @param pIsComplete       : Notifies if the command is complete with its processing
     * @param pConnection       : Pointer to the actual connection object this request came
     *                            from or NULL if this is an embedded request
     * @param requestId         : ID of the request this came from
     *
     *
     * @return
     * 0        On Success
     * <0       On Error
     *****************************************************************************/
    int HandleCommands(std::vector<dcgm::Command *> *pVecCmdsToProcess,
                       dcgm_connection_id_t connectionId,
                       dcgm_request_id_t requestId);

    /*****************************************************************************
     * This method is used to handle a client disconnecting from the host engine
     *****************************************************************************/
    void OnConnectionRemove(dcgm_connection_id_t connectionId);

    /*****************************************************************************
     This method retrieves the instance of the current cache manager
     *****************************************************************************/
    DcgmCacheManager *GetCacheManager()
    {
        return mpCacheManager;
    }

    /*****************************************************************************
     This method retrieves the instance of the group manager
     *****************************************************************************/
    DcgmGroupManager *GetGroupManager()
    {
        return mpGroupManager;
    }

    /*****************************************************************************
     This method retrieves the instance of the field group manager
     *****************************************************************************/
    DcgmFieldGroupManager *GetFieldGroupManager()
    {
        return mpFieldGroupManager;
    }

    /*****************************************************************************/
    dcgmReturn_t ProcessProtobufMessage(dcgm_connection_id_t connectionId, std::unique_ptr<DcgmMessage> message);
    dcgmReturn_t ProcessModuleCommandMsg(dcgm_connection_id_t connectionId, std::unique_ptr<DcgmMessage> message);
    dcgmReturn_t ProcessModuleCommand(dcgm_module_command_header_t *moduleCommand);

    /*****************************************************************************
     Get the status for an entity
     *****************************************************************************/
    DcgmEntityStatus_t GetEntityStatus(dcgm_field_entity_group_t entityGroupId, dcgm_field_eid_t entityId);

    /*************************************************************************/
    /*
     * Get the entities seen by DCGM of a given entityGroupId
     *
     * activeOnly     IN: If set to 1, will not count GPUs that are inaccessible for any
     *                    reason. Reasons include (but are not limited to):
     *                    - not being whitelisted
     *                    - being blocked by cgroups
     *                    - fallen off bus
     * entityGroupId IN: Which entity group to fetch the entities of
     * entities     OUT: Entity IDs
     *
     * Returns: 0 on success
     *          DCGM_ST_? #define on error
     */
    dcgmReturn_t GetAllEntitiesOfEntityGroup(int activeOnly,
                                             dcgm_field_entity_group_t entityGroupId,
                                             std::vector<dcgmGroupEntityPair_t> &entities);

    /*****************************************************************************
     This method is used to cleanup the Host Engine Handler Instance
     *****************************************************************************/
    static void Cleanup();

    /*****************************************************************************
     * This method is used to get GPU Ids corresponding to all the devices on
     * the node. The GPU Ids are valid for a life-span of hostengine and cannot
     * be assumed to get same value across the reboots.
     *****************************************************************************/
    dcgmReturn_t GetDcgmGpuIds(std::vector<unsigned int> &gpuIds, int onlySupported);

    dcgmReturn_t GetDcgmGpuArch(dcgm_field_eid_t entityId, dcgmChipArchitecture_t &arch);

    /*****************************************************************************
     * Process a WATCH_FIELD_VALUE message
     *****************************************************************************/
    dcgmReturn_t WatchFieldValue(dcgm_field_entity_group_t entityGroupId,
                                 dcgm_field_eid_t entityId,
                                 unsigned short dcgmFieldId,
                                 timelib64_t monitorFrequencyUsec,
                                 double maxSampleAge,
                                 int maxKeepSamples,
                                 const DcgmWatcher &watcher);

    /*****************************************************************************
     * Process an UNWATCH_FIELD_VALUE message
     *****************************************************************************/
    dcgmReturn_t UnwatchFieldValue(dcgm_field_entity_group_t entityGroupId,
                                   dcgm_field_eid_t entityId,
                                   unsigned short dcgmFieldId,
                                   int clearCache,
                                   const DcgmWatcher &watcher);

    /****************************************************************************
     * Get the most recent sample of a field
     *****************************************************************************/

    dcgmReturn_t GetLatestSample(dcgm_field_entity_group_t entityGroupId,
                                 dcgm_field_eid_t entityId,
                                 unsigned short dcgmFieldId,
                                 dcgmcm_sample_p sample);

    /*****************************************************************************
     Notify this object that a group was removed from the group manager
     *****************************************************************************/
    void OnGroupRemove(unsigned int groupId);

    /*****************************************************************************
     Notify this object that field values we subscribed for updated.
     *****************************************************************************/
    void OnFvUpdates(DcgmFvBuffer *fvBuffer, DcgmWatcherType_t *watcherTypes, int numWatcherTypes, void *userData);

    /*****************************************************************************
     Notify this object that mig configuration has updated.
     *****************************************************************************/
    void OnMigUpdates(unsigned int gpuId);

    /*****************************************************************************
     * Add a watcher to a local request. This watcher will be assigned a requestId
     * and will receive a ProcessMessage() call every time a message is sent from
     * the host engine to connectionId 0 (local) with its request ID.
     *
     * Note that request->requestId will be assigned by this call.
     *
     * request    IN: A DcgmRequest instance that was allocated with make_unique and
     *                will now belong to the host engine.
     * requestId OUT: The request ID that was assigned to this request on success.
     *
     *****************************************************************************/
    dcgmReturn_t AddRequestWatcher(std::unique_ptr<DcgmRequest> request, dcgm_request_id_t &requestId);

    /*****************************************************************************
     * Notify a DcgmRequest object that it has received its last response and
     * thus should be cleaned up by its owner.
     */
    void NotifyRequestOfCompletion(dcgm_connection_id_t connectionId, dcgm_request_id_t requestId);

    /*****************************************************************************
     * Send a raw message to a connected client
     *
     *****************************************************************************/
    dcgmReturn_t SendRawMessageToClient(dcgm_connection_id_t connectionId,
                                        unsigned int msgType,
                                        dcgm_request_id_t requestId,
                                        void *msgData,
                                        int msgLength,
                                        dcgmReturn_t status);
    dcgmReturn_t SendRawMessageToEmbeddedClient(unsigned int msgType,
                                                dcgm_request_id_t requestId,
                                                void *msgData,
                                                int msgLength,
                                                dcgmReturn_t status);

    void NotifyLoggingSeverityChange();

    /* Helpers to access m_persistAfterDisconnect. These all need to lock m_lock to
       maintain thread safety */
    void SetPersistAfterDisconnect(dcgm_connection_id_t connectionId)
    {
        std::scoped_lock lock(m_lock);

        m_persistAfterDisconnect.insert(connectionId);
    }

    bool GetPersistAfterDisconnect(dcgm_connection_id_t connectionId)
    {
        std::scoped_lock lock(m_lock);

        auto it = m_persistAfterDisconnect.find(connectionId);
        return it != m_persistAfterDisconnect.end();
    }

    /*****************************************************************************/
    dcgmReturn_t JobStartStats(std::string const &jobId, unsigned int groupId);

    /*****************************************************************************/
    dcgmReturn_t JobStopStats(std::string const &jobId);

    /*****************************************************************************/
    dcgmReturn_t JobGetStats(const std::string &jobId, dcgmJobInfo_t *pJobInfo);

    /*****************************************************************************/
    dcgmReturn_t JobRemove(std::string const &jobId);

    /*****************************************************************************/
    dcgmReturn_t JobRemoveAll(void);

    /*****************************************************************************
     * Add a watch on a field group
     *
     * This helper is used both internally and externally
     *
     ****************************************************************************/
    dcgmReturn_t WatchFieldGroup(unsigned int groupId,
                                 dcgmFieldGrp_t fieldGroupId,
                                 timelib64_t monitorFrequencyUsec,
                                 double maxSampleAge,
                                 int maxKeepSamples,
                                 DcgmWatcher const &watcher);

    /*****************************************************************************
     * Remove a watch on a field group
     *
     * This helper is used both internally and externally
     *
     ****************************************************************************/
    dcgmReturn_t UnwatchFieldGroup(unsigned int groupId, dcgmFieldGrp_t fieldGroupId, DcgmWatcher const &watcher);

    dcgmReturn_t HelperGetTopologyIO(unsigned int groupid, dcgmTopology_t &gpuTopology);
    dcgmReturn_t HelperGetTopologyAffinity(unsigned int groupid, dcgmAffinity_t &gpuAffinity);
    dcgmReturn_t HelperSelectGpusByTopology(uint32_t numGpus, uint64_t inputGpus, uint64_t hints, uint64_t &outputGpus);
    dcgmReturn_t HelperGetFieldSummary(dcgmFieldSummaryRequest_t &fieldSummary);
    dcgmReturn_t HelperCreateFakeEntities(dcgmCreateFakeEntities_t *fakeEntities);
    dcgmReturn_t HelperWatchPredefined(dcgmWatchPredefined_t *watchPredef, DcgmWatcher &dcgmWatcher);
    dcgmReturn_t HelperModuleBlacklist(dcgmModuleId_t moduleId);
    dcgmReturn_t HelperModuleStatus(dcgmModuleGetStatuses_v1 &msg);
    unsigned int GetHostEngineHealth() const;

    /*****************************************************************************
     * Process a GET_PROCESS_INFO message
     *****************************************************************************/
    dcgmReturn_t GetProcessInfo(unsigned int groupId, dcgmPidInfo_t *pidInfo);

private:
    std::mutex m_lock; /* Lock used for accessing table of job stats and the objects within them */

    /**************************************************************************
     * Lock/Unlocks methods
     **************************************************************************/
    int Lock();
    int Unlock();

    /*****************************************************************************
     * This method is used to process a single command on the host engine
     *****************************************************************************/
    int ProcessRequest(dcgm::Command *pCmd,
                       bool *pIsComplete,
                       dcgm_connection_id_t connectionId,
                       dcgm_request_id_t requestId);

    /*****************************************************************************
     Deletes an object if it is not null then sets its pointer to null
     *****************************************************************************/
    template <typename T>
    static void deleteNotNull(T *&obj);

    /*****************************************************************************
     * This method is used to get GPU Ids corresponding to all the devices on
     * the node. The GPU Ids are valid for a life-span of hostengine and cannot
     * be assumed to get same value across the reboots.
     *****************************************************************************/
    dcgmReturn_t GetDcgmGpuIds(dcgm::FieldMultiValues *pDcgmFieldMultiValues, int onlySupported);

    /*****************************************************************************
     * This method is used to query Cache Manager to get latest sample for a field
     *****************************************************************************/
    dcgmReturn_t GetFieldValue(dcgm_field_entity_group_t entityGroupId,
                               dcgm_field_eid_t entityId,
                               unsigned int fieldId,
                               dcgm::FieldValue *pDcgmFieldValue);

    /*****************************************************************************
     * This method is used to get values corresponding to the fields
     *****************************************************************************/
    dcgmReturn_t GetValuesForFields(dcgm_field_entity_group_t entityGroupId,
                                    dcgm_field_eid_t entityId,
                                    unsigned int fieldIds[],
                                    unsigned int count,
                                    dcgm::FieldValue values[]);

    /*****************************************************************************
     * This method is used to inject a field value
     *****************************************************************************/
    dcgmReturn_t InjectFieldValue(dcgm_field_entity_group_t entityGroupId,
                                  dcgm_field_eid_t entityId,
                                  dcgm::InjectFieldValue *injectFieldValue);

    /*****************************************************************************
     * This method is get information for a field in the cache manager
     *****************************************************************************/
    dcgmReturn_t GetCacheManagerFieldInfo(dcgmCacheManagerFieldInfo_t *fieldInfo);

    /*****************************************************************************
     * Process a WATCH_FIELD_VALUE message
     *****************************************************************************/
    dcgmReturn_t WatchFieldValue(dcgm_field_entity_group_t entityGroupId,
                                 dcgm_field_eid_t entityId,
                                 const dcgm::WatchFieldValue *watchFieldValue,
                                 const DcgmWatcher &watcher);

    /*****************************************************************************
     * Process an UNWATCH_FIELD_VALUE message
     *****************************************************************************/
    dcgmReturn_t UnwatchFieldValue(dcgm_field_entity_group_t entityGroupId,
                                   dcgm_field_eid_t entityId,
                                   const dcgm::UnwatchFieldValue *unwatchFieldValue,
                                   const DcgmWatcher &watcher);

    /*****************************************************************************
     * Process an UPDATE_ALL_FIELDS message
     *****************************************************************************/
    dcgmReturn_t UpdateAllFields(const dcgm::UpdateAllFields *updateAllFields);

    /*****************************************************************************
     * Process an GET_FIELD_MULTIPLE_VALUES message
     *****************************************************************************/
    dcgmReturn_t GetFieldMultipleValues(dcgm_field_entity_group_t entityGroupId,
                                        dcgm_field_eid_t entityId,
                                        dcgm::FieldMultiValues *pFieldMultiValues);

    /*****************************************************************************
     * This method is used to try to load a module of DCGM
     *
     * Returns DCGM_ST_OK on success or if the module is already loaded
     *         DCGM_ST_MODULE_NOT_LOADED on error
     *****************************************************************************/
    dcgmReturn_t LoadModule(dcgmModuleId_t moduleId);

    dcgmReturn_t SendModuleMessage(dcgmModuleId_t moduleId, dcgm_module_command_header_t *moduleCommand);

    /*****************************************************************************/
    /* Helper methods */
    dcgmReturn_t HelperGetInt64StatSummary(dcgm_field_entity_group_t entityGroupId,
                                           dcgm_field_eid_t entityId,
                                           unsigned short fieldId,
                                           dcgmStatSummaryInt64_t *summary,
                                           long long startTime,
                                           long long endTime);
    dcgmReturn_t HelperGetInt32StatSummary(dcgm_field_entity_group_t entityGroupId,
                                           dcgm_field_eid_t entityId,
                                           unsigned short fieldId,
                                           dcgmStatSummaryInt32_t *summary,
                                           long long startTime,
                                           long long endTime);


    /*****************************************************************************
     * Add a watch on a field group for all GPUs
     *
     * activeOnly: Whether or not to only watch the field group on GPUs that
     *             are active. Inactive GPUs include GPUs that are not whitelisted
     *
     * This helper is used both internally and externally
     *
     ****************************************************************************/
    dcgmReturn_t WatchFieldGroupAllGpus(dcgmFieldGrp_t fieldGroupId,
                                        timelib64_t monitorFrequencyUsec,
                                        double maxSampleAge,
                                        int maxKeepSamples,
                                        int activeOnly,
                                        DcgmWatcher const &watcher);

    /*****************************************************************************
     Helper functions for the scheduler hint API
     *****************************************************************************/
    dcgmReturn_t TranslateBitmapToGpuVector(uint64_t gpuBitmap, std::vector<unsigned int> &gpuIds);

    void RemoveUnhealthyGpus(std::vector<unsigned int> &gpuIds);

    dcgmReturn_t ProcessSelectGpusByTopology(dcgm::Command *pCmd, bool *pIsComplete);

    /*****************************************************************************
     Helper method to RPC to the health module for a health check
     *****************************************************************************/
    dcgmReturn_t HelperHealthCheck(unsigned int groupId,
                                   long long startTime,
                                   long long endTime,
                                   dcgmHealthResponse_v4 &response);

    /*****************************************************************************
     Helper method for watching the fields that the host engine cares about
     *****************************************************************************/
    dcgmReturn_t WatchHostEngineFields(void);

    /*****************************************************************************
     Helper methods for processing the different commands on the host engine
     *****************************************************************************/
    dcgmReturn_t ProcessClientLogin(dcgm::Command *pCmd, bool *pIsComplete, dcgm_connection_id_t connectionId);
    dcgmReturn_t ProcessGroupCreate(dcgm::Command *pCmd, bool *pIsComplete, dcgm_connection_id_t connectionId);
    dcgmReturn_t ProcessAddRemoveGroup(dcgm::Command *pCmd, bool *pIsComplete, dcgm_connection_id_t connectionId);
    dcgmReturn_t ProcessGroupDestroy(dcgm::Command *pCmd, bool *pIsComplete, dcgm_connection_id_t connectionId);
    dcgmReturn_t ProcessGroupInfo(dcgm::Command *pCmd, bool *pIsComplete, dcgm_connection_id_t connectionId);
    dcgmReturn_t ProcessGroupGetallIds(dcgm::Command *pCmd, bool *pIsComplete, dcgm_connection_id_t connectionId);
    static dcgmReturn_t ProcessDiscoverDevices(dcgm::Command *pCmd, bool *pIsComplete);
    dcgmReturn_t ProcessGetEntityList(dcgm::Command *pCmd, bool *pIsComplete);
    dcgmReturn_t ProcessRegForPolicyUpdate(dcgm::Command *pCmd,
                                           bool *pIsComplete,
                                           dcgm_connection_id_t pConnectionId,
                                           dcgm_request_id_t requestId);
    dcgmReturn_t ProcessUnregForPolicyUpdate(dcgm::Command *pCmd, bool *pIsComplete, dcgm_connection_id_t connectionId);
    dcgmReturn_t ProcessSetCurrentViolPolicy(dcgm::Command *pCmd,
                                             bool *pIsComplete,
                                             dcgm_connection_id_t pConnectionId);
    dcgmReturn_t ProcessGetCurrentViolPolicy(dcgm::Command *pCmd, bool *pIsComplete);
    static dcgmReturn_t ProcessInjectFieldValue(dcgm::Command *pCmd, bool *pIsComplete);
    static dcgmReturn_t ProcessGetFieldLatestValue(dcgm::Command *pCmd, bool *pIsComplete);
    static dcgmReturn_t ProcessGetFieldMultipleValues(dcgm::Command *pCmd, bool *pIsComplete);
    static dcgmReturn_t ProcessWatchFieldValue(dcgm::Command *pCmd, bool *pIsComplete, DcgmWatcher &dcgmWatcher);
    static dcgmReturn_t ProcessUnwatchFieldValue(dcgm::Command *pCmd, bool *pIsComplete, DcgmWatcher &dcgmWatcher);
    static dcgmReturn_t ProcessUpdateAllFields(dcgm::Command *pCmd, bool *pIsComplete);
    static dcgmReturn_t ProcessCacheManagerFieldInfo(dcgm::Command *pCmd, bool *pIsComplete);
    dcgmReturn_t ProcessWatchFields(dcgm::Command *pCmd, bool *pIsComplete, DcgmWatcher &dcgmWatcher);
    dcgmReturn_t ProcessUnwatchFields(dcgm::Command *pCmd, bool *pIsComplete, DcgmWatcher &dcgmWatcher);
    dcgmReturn_t ProcessGetPidInfo(dcgm::Command *pCmd, bool *pIsComplete);
    dcgmReturn_t ProcessFieldGroupCreate(dcgm::Command *pCmd, bool *pIsComplete, DcgmWatcher &dcgmWatcher);
    dcgmReturn_t ProcessFieldGroupDestroy(dcgm::Command *pCmd, bool *pIsComplete, DcgmWatcher &dcgmWatcher);
    dcgmReturn_t ProcessFieldGroupGetOne(dcgm::Command *pCmd, bool *pIsComplete);
    dcgmReturn_t ProcessFieldGroupGetAll(dcgm::Command *pCmd, bool *pIsComplete);
    dcgmReturn_t ProcessWatchPredefined(dcgm::Command *pCmd, bool *pIsComplete, DcgmWatcher &dcgmWatcher);
    dcgmReturn_t ProcessJobStartStats(dcgm::Command *pCmd, bool *pIsComplete);
    dcgmReturn_t ProcessJobStopStats(dcgm::Command *pCmd, bool *pIsComplete);
    dcgmReturn_t ProcessJobRemove(dcgm::Command *pCmd, bool *pIsComplete);
    dcgmReturn_t ProcessJobGetInfo(dcgm::Command *pCmd, bool *pIsComplete);
    dcgmReturn_t ProcessGetTopologyAffinity(dcgm::Command *pCmd, bool *pIsComplete);
    dcgmReturn_t ProcessGetTopologyIO(dcgm::Command *pCmd, bool *pIsComplete);
    dcgmReturn_t ProcessCreateFakeEntities(dcgm::Command *pCmd, bool *pIsComplete);
    dcgmReturn_t ProcessGetNvLinkLinkStatus(dcgm::Command *pCmd, bool *pIsComplete);
    dcgmReturn_t ProcessGetMultipleLatestValues(dcgm::Command *pCmd, bool *pIsComplete);
    dcgmReturn_t ProcessGetGpuInstanceHierarchy(dcgm::Command *pCmd, bool *pIsComplete);
    dcgmReturn_t ProcessSetNvLinkLinkStatus(dcgm::Command *pCmd, bool *pIsComplete);
    dcgmReturn_t ProcessGetFieldSummary(dcgm::Command *pCmd, bool *pIsComplete);
    dcgmReturn_t ProcessModuleBlacklist(dcgm::Command *pCmd);
    dcgmReturn_t ProcessModuleGetStatuses(dcgm::Command *pCmd);
    dcgmReturn_t ProcessIsHostengineHealthy(dcgm::Command *pCmd, bool *pIsComplete);

    /*****************************************************************************/
    /* Remove any requests that the host engine was tracking */
    dcgmReturn_t RemoveAllTrackedRequests(void);

    /*****************************************************************************/
    /* Helper to retrieve a cached value or live value and return it in fvBuffer */
    dcgmReturn_t GetCachedOrLiveValueForEntity(dcgmGroupEntityPair_t entity,
                                               unsigned short fieldId,
                                               DcgmFvBuffer &fvBuffer);

    /*****************************************************************************/
    /* Set of connectionId as to if this connection's watches should persist after disconnect.
       If this is unset for a connectionId, we should assume false. Being present = true */
    std::unordered_set<dcgm_connection_id_t> m_persistAfterDisconnect;

    void ClearPersistAfterDisconnect(dcgm_connection_id_t connectionId)
    {
        std::scoped_lock lock(m_lock);

        auto it = m_persistAfterDisconnect.find(connectionId);
        if (it != m_persistAfterDisconnect.end())
        {
            m_persistAfterDisconnect.erase(it);
        }
    }

    /*****************************************************************************/
    /* DcgmIpc callbacks */
    static void StaticProcessMessage(dcgm_connection_id_t connectionId,
                                     std::unique_ptr<DcgmMessage> message,
                                     void *userData);
    void ProcessMessage(dcgm_connection_id_t connectionId, std::unique_ptr<DcgmMessage> message);

    static void StaticProcessDisconnect(dcgm_connection_id_t connectionId, void *userData);

    /*****************************************************************************/

    /*****************************************************************************
     Private Constructor and Destructor to achieve Singelton design
     *****************************************************************************/
    DcgmHostEngineHandler()
        : m_dcgmIpc(DCGM_HE_NUM_WORKERS)
    {}
    DcgmHostEngineHandler(dcgmStartEmbeddedV2Params_v1 params);
    virtual ~DcgmHostEngineHandler();

    /* This data structure is used to store user provided job id information and associates start
       and stop timestamp with the user provided start/stop notification. */
    typedef std::map<std::string, jobRecord_t> jobIdMap_t;
    jobIdMap_t mJobIdMap;

    /* Core module is always loaded. We create a static object for Core with this class */
    static DcgmModuleCore mModuleCoreObj;

    static DcgmHostEngineHandler *mpHostEngineHandlerInstance; // HostEngine Handler Instance

    DcgmCacheManager *mpCacheManager;
    DcgmGroupManager *mpGroupManager;
    DcgmCoreCommunication m_communicator; // The object which processes module core requests
    dcgmCoreCallbacks_t m_coreCallbacks;  // The callback function passed to modules for core requests
    DcgmFieldGroupManager *mpFieldGroupManager;

    DcgmIpc m_dcgmIpc; /* IPC object */

    /* Field Groups */
    dcgmFieldGrp_t mFieldGroup1Sec;
    dcgmFieldGrp_t mFieldGroup30Sec;
    dcgmFieldGrp_t mFieldGroupHourly;
    dcgmFieldGrp_t mFieldGroupPidAndJobStats;

    void HandleAddWatchError(int ret, std::string field);
    static void finalizeCmd(dcgm::Command *pCmd,
                            dcgmReturn_t cmdStatus,
                            bool *&pIsComplete,
                            void *returnArg,
                            size_t returnArgSize);

    /* This data structure stores pluggable modules for handling client requests */
    dcgmhe_module_info_t m_modules[DcgmModuleIdCount];

    /* Watched requests. Currently used to track policy management callbacks. Protected by Lock()/Unlock() */
    dcgm_request_id_t m_nextWatchedRequestId;
    typedef std::unordered_map<dcgm_request_id_t, std::unique_ptr<DcgmRequest>> watchedRequests_t;
    watchedRequests_t m_watchedRequests;

    unsigned int m_hostengineHealth;
};

#endif /* DCGMHOSTENGINEHANDLER_H */
