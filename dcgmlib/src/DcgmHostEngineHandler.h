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
/*
 * File:   DcgmHostEngineHandler.h
 */

#ifndef DCGMHOSTENGINEHANDLER_H
#define DCGMHOSTENGINEHANDLER_H

#include "DcgmCacheManager.h"
#include "DcgmChildProcessManager.hpp"
#include "DcgmCoreCommunication.h"
#include "DcgmFieldGroup.h"
#include "DcgmGroupManager.h"
#include "DcgmIpc.h"
#include "DcgmModule.h"
#include "DcgmRequest.h"
#include "DcgmResourceManager.h"
#include "DcgmWatcher.h"
#include "dcgm_agent.h"
#include <core/DcgmModuleCore.h>
#include <dcgm_core_communication.h>
#include <iostream>
#include <mutex>
#include <unordered_map>

#define INJECTION_MODE_ENV_VAR "NVML_INJECTION_MODE"

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
     *                    - not on the allowlist
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

    /*************************************************************************/
    /*
     * Get if the given entityGroupId and entityId are known by DCGM.
     *
     * Returns true if the entities are valid and known
     *         false if not
     *
     */
    bool GetIsValidEntityId(dcgm_field_entity_group_t entityGroupId, dcgm_field_eid_t entityId);

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
        auto lock = Lock();

        m_persistAfterDisconnect.insert(connectionId);
    }

    bool GetPersistAfterDisconnect(dcgm_connection_id_t connectionId)
    {
        auto lock = Lock();

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
                                 timelib64_t monitorIntervalUsec,
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
    dcgmReturn_t HelperModuleDenylist(dcgmModuleId_t moduleId);
    dcgmReturn_t HelperModuleStatus(dcgmModuleGetStatuses_v1 &msg);
    unsigned int GetHostEngineHealth() const;

    /*****************************************************************************
     * Process a GET_PROCESS_INFO message
     *****************************************************************************/
    dcgmReturn_t GetProcessInfo(unsigned int groupId, dcgmPidInfo_t *pidInfo);

    void SetServiceAccount(const char *serviceAccout);
    std::string const &GetServiceAccount() const;

    /**
     * Apply a denylist to modules
     *
     * @param denyList Array of module IDs to add to the denylist
     * @param denyListCount Number of entries in denyList
     * @return DCGM_ST_OK if successful, error code otherwise
     */
    dcgmReturn_t ApplyModuleDenylist(unsigned int const *denyList, unsigned int denyListCount);

    bool UsingInjectionNvml() const;

    dcgmReturn_t NvmlInjectFieldValue(dcgm_field_eid_t gpuId, const nvmlFieldValue_t &value);

    dcgmReturn_t ProcessPauseResume(dcgm_core_msg_pause_resume_v1 *msg);

    /**
     * @brief Pauses all modules
     *
     * @return
     *      - \ref DCGM_ST_OK               All modules successfully handled the Pause message
     *      - \ref DCGM_ST_GENERIC_ERROR    At least one module failed to handle the Pause message
     */
    dcgmReturn_t Pause();

    /**
     * @brief Resumes all modules
     * @return
     *      - \ref DCGM_ST_OK               All modules successfully handled the Resume message
     *      - \ref DCGM_ST_GENERIC_ERROR    At least one module failed to handle the Resume message
     */
    dcgmReturn_t Resume();

    /**
     * @brief Pauses a loaded module.
     * @param[in] moduleId Module ID to pause
     * @return
     *      - \ref DCGM_ST_OK   A module successfully handled the pause message or the module is not loaded or not
     *                          initialized.
     *      - \ref DCGM_ST_*    An error occurred while trying to pause the module.
     */
    dcgmReturn_t PauseModule(dcgmModuleId_t moduleId);

    /**
     * @brief Resumes a loaded module.
     * @param[in] moduleId Module ID to resume
     * @return
     *      - \ref DCGM_ST_OK   A module successfully handled the resume message or the module is not loaded or not
     *                          initialized.
     *      - \ref DCGM_ST_*    An error occurred while trying to resume the module.
     */
    dcgmReturn_t ResumeModule(dcgmModuleId_t moduleId);

    /**
     * @brief Callback to reserve resources for a module
     * @return Token if reservation was successful or NullOpt
     */
    std::optional<unsigned int> ReserveResources();

    /**
     * @brief Callback to free resources for a module
     * @param[in] token Token to free
     * @return True if resources were freed, false if not
     */
    bool FreeResources(unsigned int const &token);

    /**
     * Spawn a new child process.
     *
     * @param[in] params       Parameters required to spawn the child process
     * @param[out] handle      Handle to the ChildProcess instance
     * @param[out] pid         Process ID of the spawned process
     *
     * @returns DCGM_ST_OK if successful, error code otherwise
     */
    dcgmReturn_t ChildProcessSpawn(dcgmChildProcessParams_t const &params, ChildProcessHandle_t &handle, int &pid);

    /**
     * Stop the child process (issues a SIGTERM, unless specified otherwise)
     *
     * @param[in] handle      Handle to the ChildProcess instance
     * @param[in] force       Whether to force the process to stop with SIGKILL
     *
     * @returns DCGM_ST_OK if successful, error code otherwise
     */
    dcgmReturn_t ChildProcessStop(ChildProcessHandle_t handle, bool force);

    /**
     * Get the status of the child process.
     *
     * @param[in] handle      Handle to the ChildProcess instance
     * @param[out] status     Status of the child process
     *
     * @returns DCGM_ST_OK if successful, error code otherwise
     */
    dcgmReturn_t ChildProcessGetStatus(ChildProcessHandle_t handle, dcgmChildProcessStatus_t &status);

    /**
     * Wait for the child process to exit.
     *
     * @param[in] handle      Handle to the ChildProcess instance
     * @param[in] timeoutSec  Timeout in seconds
     *
     * @returns DCGM_ST_OK if successful, error code otherwise
     */
    dcgmReturn_t ChildProcessWait(ChildProcessHandle_t handle, int timeoutSec);

    /**
     * Destroy the child process.
     *
     * @param[in] handle             Handle to the ChildProcess instance
     * @param[in] sigTermTimeoutSec  Timeout in seconds for SIGTERM
     *
     * @returns DCGM_ST_OK if successful, error code otherwise
     */
    dcgmReturn_t ChildProcessDestroy(ChildProcessHandle_t handle, int sigTermTimeoutSec);

    /**
     * Get the standard error handle of the child process.
     *
     * @param[in] handle      Handle to the ChildProcess instance
     * @param[out] fd         File descriptor of the standard error
     *
     * @returns DCGM_ST_OK if successful, error code otherwise
     */
    dcgmReturn_t ChildProcessGetStdErrHandle(ChildProcessHandle_t handle, int &fd);

    /**
     * Get the standard output handle of the child process.
     *
     * @param[in] handle      Handle to the ChildProcess instance
     * @param[out] fd         File descriptor of the standard output
     *
     * @returns DCGM_ST_OK if successful, error code otherwise
     */
    dcgmReturn_t ChildProcessGetStdOutHandle(ChildProcessHandle_t handle, int &fd);

    /**
     * Get the data channel handle of the child process.
     *
     * @param[in] handle      Handle to the ChildProcess instance
     * @param[out] fd         File descriptor of the data channel
     *
     * @returns DCGM_ST_OK if successful, error code otherwise
     */
    dcgmReturn_t ChildProcessGetDataChannelHandle(ChildProcessHandle_t handle, int &fd);

    /**
     * Reset the ChildProcessManager to clean state.
     *
     * Cleans up all processes and resets internal state to prevent
     * signal handling corruption between runs.
     *
     * @returns DCGM_ST_OK if successful, error code otherwise
     */
    dcgmReturn_t ChildProcessManagerReset();

private:
    DcgmMutex m_lock = DcgmMutex(0);

    /**************************************************************************
     * Lock/Unlocks methods
     **************************************************************************/
    DcgmLockGuard Lock();
    /**
     * @brief Takes a previously acquired lock guard and releases it by calling its destructor.
     * @param[in] lock  The lock guard that needs to be released.
     * @note This function is only necessary if you want to release the lock before it goes out of a scope
     *       and gets destroyed/released automatically.
     * @code{cpp}
     * {
     *  auto lock = Lock(); // acquiring the lock guard
     *  Unlock(std::move(lock));    // release the lock earlier that the scope (code block) is ended
     * } // here the lock would have been released if the Unlock function hasn't been called
     * @endcode
     */
    static void Unlock(DcgmLockGuard lock);

    /*****************************************************************************
     Deletes an object if it is not null then sets its pointer to null
     *****************************************************************************/
    template <typename T>
    static void deleteNotNull(T *&obj);

    /*****************************************************************************
     * This method is get information for a field in the cache manager
     *****************************************************************************/
    dcgmReturn_t GetCacheManagerFieldInfo(dcgmCacheManagerFieldInfo_v4_t *fieldInfo);

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
     *             are active. Inactive GPUs include GPUs that are not on the
     *             allowlist.
     *
     * This helper is used both internally and externally
     *
     ****************************************************************************/
    dcgmReturn_t WatchFieldGroupAllGpus(dcgmFieldGrp_t fieldGroupId,
                                        timelib64_t monitorIntervalUsec,
                                        double maxSampleAge,
                                        int maxKeepSamples,
                                        int activeOnly,
                                        DcgmWatcher const &watcher);

    /*****************************************************************************
     Helper functions for the scheduler hint API
     *****************************************************************************/
    dcgmReturn_t TranslateBitmapToGpuVector(uint64_t gpuBitmap, std::vector<unsigned int> &gpuIds);

    void RemoveUnhealthyGpus(std::vector<unsigned int> &gpuIds);

    /*****************************************************************************
     Helper method to RPC to the health module for a health check
     *****************************************************************************/
    dcgmReturn_t HelperHealthCheck(unsigned int groupId,
                                   long long startTime,
                                   long long endTime,
                                   dcgmHealthResponse_v5 &response);

    /*****************************************************************************
     Helper method for watching the fields that the host engine cares about
     *****************************************************************************/
    dcgmReturn_t WatchHostEngineFields(void);

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
        auto lock = Lock();

        auto it = m_persistAfterDisconnect.find(connectionId);
        if (it != m_persistAfterDisconnect.end())
        {
            m_persistAfterDisconnect.erase(it);
        }
    }

    /*****************************************************************************/
    void LoadNvml();

    /*****************************************************************************/
    void ShutdownNvml();

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
        , m_nvmlLoaded(false)
    {}
    explicit DcgmHostEngineHandler(dcgmStartEmbeddedV2Params_v1 params);
    virtual ~DcgmHostEngineHandler();

    /* This data structure is used to store user provided job id information and associates start
       and stop timestamp with the user provided start/stop notification. */
    typedef std::map<std::string, jobRecord_t> jobIdMap_t;
    jobIdMap_t mJobIdMap;

    /* Core module is always loaded. We create a static object for Core with this class */
    static DcgmModuleCore mModuleCoreObj;

    static DcgmHostEngineHandler *mpHostEngineHandlerInstance; // HostEngine Handler Instance

    DcgmCacheManager *mpCacheManager {};
    DcgmGroupManager *mpGroupManager {};
    DcgmCoreCommunication m_communicator;   // The object which processes module core requests
    dcgmCoreCallbacks_t m_coreCallbacks {}; // The callback function passed to modules for core requests
    DcgmFieldGroupManager *mpFieldGroupManager {};

    DcgmIpc m_dcgmIpc; /* IPC object */

    /* Field Groups */
    dcgmFieldGrp_t mFieldGroup1Sec {};
    dcgmFieldGrp_t mFieldGroup30Sec {};
    dcgmFieldGrp_t mFieldGroupHourly {};
    dcgmFieldGrp_t mFieldGroupPidAndJobStats {};

    void HandleAddWatchError(int ret, std::string field);

    /* This data structure stores pluggable modules for handling client requests */
    dcgmhe_module_info_t m_modules[DcgmModuleIdCount] {};

    /* Watched requests. Currently used to track policy management callbacks. Protected by Lock()/Unlock() */
    dcgm_request_id_t m_nextWatchedRequestId {};
    typedef std::unordered_map<dcgm_request_id_t, std::unique_ptr<DcgmRequest>> watchedRequests_t;
    watchedRequests_t m_watchedRequests;

    unsigned int m_hostengineHealth {};
    std::string m_serviceAccount;
    bool m_usingInjectionNvml {};
    bool m_nvmlLoaded {};

    DcgmResourceManager m_resourceManager;
    DcgmChildProcessManager m_childProcessManager;
};

#endif /* DCGMHOSTENGINEHANDLER_H */
