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
#ifndef DCGMPOLICYMANAGER_H
#define DCGMPOLICYMANAGER_H

#include "DcgmMutex.h"
#include "DcgmPolicyCallbackRAII.h"
#include "DcgmPolicyEvaluator.h"
#include "DcgmPolicyNotificationRouter.h"
#include "DcgmPolicyPersistent.h"
#include "DcgmPolicyRegistry.h"
#include "DcgmProtocol.h"
#include "dcgm_policy_structs.h"
#include <DcgmCoreProxy.h>
#include <dcgm_fields.h>

#include <expected>
#include <list>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

/* These are array indexes that correspond with DCGM_POLICY_COND_* bitmasks */
typedef enum DcgmViolationPolicyAlert_enum
{
    DCGM_VIOLATION_POLICY_FAIL_ECC_DBE = 0,
    DCGM_VIOLATION_POLICY_FAIL_PCIE,
    DCGM_VIOLATION_POLICY_FAIL_MAX_RETIRED_PAGES,
    DCGM_VIOLATION_POLICY_FAIL_THERMAL,
    DCGM_VIOLATION_POLICY_FAIL_POWER,
    DCGM_VIOLATION_POLICY_FAIL_NVLINK,
    DCGM_VIOLATION_POLICY_FAIL_XID,

    /* this should be last */
    DCGM_VIOLATION_POLICY_FAIL_COUNT,
} DcgmViolationPolicyAlert_t;

/* The number of bitmask entries and the count of their corresponding indexes must be the same */
DCGM_CASSERT(DCGM_POLICY_COND_MAX == DCGM_VIOLATION_POLICY_FAIL_COUNT, DCGM_POLICY_COND_MAX);

/* A watcher of policy */
typedef struct
{
    /* Key parts */
    dcgm_connection_id_t connectionId; /* Associated connection */
    dcgm_request_id_t requestId;       /* Request ID that owns this watch */
    /* Attributes */
    int64_t lastSentTimestamp[DCGM_VIOLATION_POLICY_FAIL_COUNT];
    /* What was the timestamp of the last
                                          fv that caused us to trigger this.
                                          Use this to make sure we don't spam
                                          notifications */
    dcgmPolicyCondition_t conditions; /* A mask of policy conditions that this
                                          connection+request wants callbacks for */
} dpm_watcher_t;

/* Per-GPU Policy context information */
typedef struct
{
    bool policiesHaveBeenSet;            /* Has currrentPolicies been set yet? If not,
                                     we have no thresholds to alert on. */
    dcgmPolicy_t currentPolicies;        /* Current policy thresholds for this GPU.
                                     Note that these get fully-overwritten every
                                     time another user sets policies for this GPU.
                                     Conditions are global to a GPU. Which conditions
                                     trigger callbacks are per-watcher in watchers[] */
    std::vector<dpm_watcher_t> watchers; /* connectionId+requestIds that care about this */
    bool alive;                          /* Whether the GPU is alive (i.e., attached) */
} dpm_gpu_t;

/******************************************************************
 * Class to implement the compute side policy manager
 ******************************************************************/
class DcgmPolicyManager
{
public:
    DcgmPolicyManager(dcgmCoreCallbacks_t &pDcc);
    virtual ~DcgmPolicyManager();

    dcgmReturn_t Init();

    /*************************************************************************/
    /*
     * Register for policy updates if a violation occurs
     */
    dcgmReturn_t RegisterForPolicy(dcgm_policy_msg_register_t *msg);

    /*************************************************************************/
    /*
     * Unregister for policy updates
     */
    dcgmReturn_t UnregisterForPolicy(dcgm_policy_msg_unregister_t *msg);

    /*************************************************************************/
    /*
     * Set the current violation policy for a given group (affects all registered watchers)
     */
    dcgmReturn_t ProcessSetPolicy(dcgm_policy_msg_set_policy_t *msg);

    /*************************************************************************/
    /*
     * Get the current violation policy
     */
    dcgmReturn_t ProcessGetPolicies(dcgm_policy_msg_get_policies_t *msg);

    /*************************************************************************/
    /*
     * Helper method to remove any watchers associated with a connection ID
     */
    void RemoveWatchersForConnection(dcgm_connection_id_t connectionId);

    /*************************************************************************/
    /*
     * Process a client disconnecting
     */
    void OnClientDisconnect(dcgm_connection_id_t connectionId);

    /*************************************************************************/
    /*
     * Process a field value we care about being updated
     */
    void OnFieldValuesUpdate(DcgmFvBuffer *fvBuffer);

    /*************************************************************************/
    /*
     * Attach GPUs to the policy manager
     */
    dcgmReturn_t AttachGpus();

    /*************************************************************************/
    /*
     * Detach GPUs from the policy manager
     */
    dcgmReturn_t DetachGpus();

    /**
     * Create a field-based policy and refresh field watches/persistence.
     *
     * Runs on the policy module task runner. Caller-provided runtime fields in msg->policy
     * are ignored and replaced with module-owned values on success.
     *
     * @param[in,out] msg Create request and created policy readback.
     * @return DCGM_ST_OK on success, or a registry/watch/persistence error code.
     */
    dcgmReturn_t CreatePolicy(dcgm_policy_msg_create_t *msg);

    /**
     * Modify an existing field-based policy and refresh field watches/persistence.
     *
     * Runs on the policy module task runner. Runtime fields from the existing policy are
     * preserved instead of trusting caller-provided values.
     *
     * @param[in,out] msg Modify request and modified policy readback.
     * @return DCGM_ST_OK on success, DCGM_ST_NO_DATA if the policy is missing, or a
     *         registry/watch/persistence error code.
     */
    dcgmReturn_t ModifyPolicy(dcgm_policy_msg_modify_t *msg);

    /**
     * Delete one or all field-based policies and refresh field watches/persistence.
     *
     * Runs on the policy module task runner.
     *
     * @param[in] msg Delete request.
     * @return DCGM_ST_OK on success, DCGM_ST_NO_DATA if the policy is missing, or a
     *         registry/watch/persistence error code.
     */
    dcgmReturn_t DeletePolicy(dcgm_policy_msg_delete_t *msg);

    /**
     * Enable a field-based policy and refresh field watches/persistence.
     *
     * Runs on the policy module task runner.
     *
     * @param[in] msg Policy identity to enable.
     * @return DCGM_ST_OK on success, DCGM_ST_NO_DATA if the policy is missing, or a
     *         registry/watch/persistence error code.
     */
    dcgmReturn_t EnablePolicy(dcgm_policy_msg_set_state_t *msg);

    /**
     * Disable a field-based policy and refresh field watches/persistence.
     *
     * Runs on the policy module task runner.
     *
     * @param[in] msg Policy identity to disable.
     * @return DCGM_ST_OK on success, DCGM_ST_NO_DATA if the policy is missing, or a
     *         registry/watch/persistence error code.
     */
    dcgmReturn_t DisablePolicy(dcgm_policy_msg_set_state_t *msg);

    /**
     * Get one field-based policy by policy ID.
     *
     * Runs on the policy module task runner.
     *
     * @param[in,out] msg Lookup request and policy output.
     * @return DCGM_ST_OK on success, DCGM_ST_NO_DATA if the policy is missing.
     */
    dcgmReturn_t GetOnePolicy(dcgm_policy_msg_get_one_t *msg);

    /**
     * Get all field-based policies.
     *
     * Runs on the policy module task runner.
     *
     * @param[in,out] msg Paged output request and populated policy array on return.
     * @return DCGM_ST_OK on success, DCGM_ST_INSUFFICIENT_SIZE if another page remains.
     */
    dcgmReturn_t GetAllPolicies(dcgm_policy_msg_get_all_t *msg);

    /**
     * Import field-based policies from a YAML file and refresh field watches/persistence.
     *
     * Runs on the policy module task runner.
     *
     * @param[in] path Import path copied from the module message.
     * @return DCGM_ST_OK on success, or a load/watch/persistence error code.
     */
    dcgmReturn_t ImportPolicies(std::string const &path);

    /**
     * Get recent field-based policy violations.
     *
     * Runs on the policy module task runner.
     *
     * @param[in,out] msg Query request and violation output array.
     * @return DCGM_ST_OK on success, DCGM_ST_INSUFFICIENT_SIZE if more violations exist than fit.
     */
    dcgmReturn_t GetPolicyViolations(dcgm_policy_msg_get_violations_t *msg);

    /**
     * Register for field-based policy violation callbacks.
     *
     * Runs on the policy module task runner.
     *
     * @param[in] msg Callback registration request.
     * @return DCGM_ST_OK on success, or a registry/routing error code.
     */
    dcgmReturn_t RegisterPolicyCallback(dcgm_policy_msg_register_callback_t *msg);

    /**
     * Unregister field-based policy violation callbacks.
     *
     * Runs on the policy module task runner.
     *
     * @param[in] msg Callback unregistration request.
     * @return DCGM_ST_OK on success.
     */
    dcgmReturn_t UnregisterPolicyCallback(dcgm_policy_msg_unregister_callback_t *msg);

    /**
     * Remove field-based policy callback registrations matching a connection/filter.
     *
     * @param[in] connectionId Connection whose registrations should be considered.
     * @param[in] fieldId Field filter; ignored when allForConnection is true.
     * @param[in] policyId Policy filter; ignored when allForConnection is true.
     * @param[in] allForConnection When true, remove all registrations for connectionId.
     * @return Number of registrations removed.
     */
    size_t RemovePolicyCallbackRegistrations(dcgm_connection_id_t connectionId,
                                             unsigned short fieldId,
                                             uint64_t policyId,
                                             bool allForConnection);

    /**
     * Evaluate field-based policies for updated field values and route violations.
     *
     * Runs on the policy module task runner.
     *
     * @param[in] fvBuffer Copied field values from Cache Manager.
     * @return DCGM_ST_OK on success, or the first registry/routing error code.
     */
    dcgmReturn_t OnPolicyFieldValuesUpdate(DcgmFvBuffer *fvBuffer);

private:
    /* variables */
    using PolicyWatch = std::tuple<dcgm_field_entity_group_t, dcgm_field_eid_t, unsigned short>;

    struct CallbackRegistration
    {
        CallbackRegistration(dcgm_connection_id_t connectionIdIn,
                             dcgm_request_id_t requestIdIn,
                             unsigned short fieldIdIn,
                             uint64_t policyIdIn,
                             DcgmPolicyCallbackRAII guardIn);

        dcgm_connection_id_t connectionId;
        dcgm_request_id_t requestId;
        unsigned short fieldId;
        uint64_t policyId;
        DcgmPolicyCallbackRAII guard;
    };

    DcgmCoreProxy mpCoreProxy;
    DcgmPolicyRegistry m_registry;
    DcgmPolicyEvaluator m_evaluator;
    DcgmPolicyNotificationRouter m_router;
    DcgmPolicyPersistence m_persistence;
    std::set<PolicyWatch> m_activeWatches;
    std::list<CallbackRegistration> m_callbackRegistrations;

    /**
     * Record violations in the registry and route them through configured channels.
     *
     * @param[in] violations Violations produced by the evaluator.
     * @return DCGM_ST_OK on success, or the first registry/routing error code.
     */
    dcgmReturn_t RoutePolicyViolations(std::vector<dcgm_policy_violation_t> const &violations);

    // mutex for use when manipulating members of this object
    DcgmMutex *m_mutex;

    /* Have we watched our fields in the cache manager yet for a given connection ID?.
       Don't set this directly. Call WatchFields() */
    std::map<dcgm_connection_id_t, int> m_haveWatchedFields;

    /* How many GPUs are currently valid in m_gpus? */
    int m_numGpus;
    dpm_gpu_t m_gpus[DCGM_MAX_NUM_DEVICES]; /* Per-GPU information */

    std::vector<dpm_watcher_t> m_metaGroupWatchers; /* Watchers for the meta group (DCGM_GROUP_ALL_GPUS), used to add
                                                       back when GPUs are attached */
    dcgmPolicy_t m_metaGroupPolicy; /* Policy for the meta group (DCGM_GROUP_ALL_GPUS), used to add back when GPUs are
                                       attached */

    /* methods */
    void SetViolation(DcgmViolationPolicyAlert_t alertType,
                      unsigned int gpuId,
                      int64_t timestamp,
                      dcgmPolicyCallbackResponse_t *callbackResponse);

    /* error checking functions */
    dcgmReturn_t CheckEccErrors(dcgmBufferedFv_t *fv);
    dcgmReturn_t CheckPcieErrors(dcgmBufferedFv_t *fv);
    dcgmReturn_t CheckRetiredPages(dcgmBufferedFv_t *fv);
    dcgmReturn_t CheckThermalValues(dcgmBufferedFv_t *fv);
    dcgmReturn_t CheckPowerValues(dcgmBufferedFv_t *fv);
    dcgmReturn_t CheckNVLinkErrors(dcgmBufferedFv_t *fv);
    dcgmReturn_t CheckXIDErrors(dcgmBufferedFv_t *fv);

    /* Helper function to convert Nvlink counters fieldIds to string */
    char *ConvertNVLinkCounterTypeToString(unsigned short fieldId);

    /**
     * Gets delta between current and previous sample from cache.
     *
     * @param[in]  entityId      GPU entity ID
     * @param[in]  fieldId       Field to query
     * @param[in]  timestamp     Current sample timestamp
     * @param[out] currentCount  Current counter value
     * @param[out] prevCount     Previous counter value
     *
     * @return true if delta available, false if not enough samples or error
     */
    bool GetCounterDelta(unsigned int entityId,
                         unsigned short fieldId,
                         int64_t timestamp,
                         unsigned int &currentCount,
                         unsigned int &prevCount);

    /*****************************************************************************
     * Method to watch all fields that need to be watched in order for the policy
     * manager to do its job. If the cache manager is already watching fields, this is a
     * no-op.
     *
     * @param connectionId IN: Which connection is this for?
     *
     *****************************************************************************/
    dcgmReturn_t WatchFields(dcgm_connection_id_t connectionId);

    /*************************************************************************/
    /*
     * Add a watcher to the m_metaGroupWatchers, it will ignore if the watcher is already in the list.
     * It is expected to be called with m_mutex locked.
     *
     * @param watcher IN: The watcher to add
     */
    void AddMetaGroupWatcher(dpm_watcher_t watcher);

    /*************************************************************************/
    /*
     * Remove a watcher from m_metaGroupWatchers, it will ignore if the watcher is not in the list.
     * It is expected to be called with m_mutex locked.
     *
     * @param connectionId IN: The connection ID of the watcher to remove
     */
    void RemoveMetaGroupWatcher(dcgm_connection_id_t connectionId);

    /*************************************************************************/
    /*
     * Add the meta group policy to the watchers
     */
    void AddMetaGroupPolicyToWatchers(std::vector<dcgmGroupEntityPair_t> const &entities);

    /**
     * Refresh Cache Manager field watches for enabled field policies and save policies.
     *
     * @return DCGM_ST_OK on success, or the first watch/persistence error code.
     */
    dcgmReturn_t RefreshFieldPolicyWatchesAndSave();

    /**
     * Reconcile active Cache Manager watches with currently enabled field policies.
     *
     * @return DCGM_ST_OK on success, or the first watch update error code.
     */
    dcgmReturn_t ApplyPolicyWatchDelta();

    /**
     * Apply a desired field-watch set without restoring on failure.
     *
     * @param[in] desired Desired (entityGroupId, entityId, fieldId) watch set.
     * @return DCGM_ST_OK on success, or the first Cache Manager watch update error code.
     */
    dcgmReturn_t ApplyPolicyWatchDeltaNoRollback(std::set<PolicyWatch> const &desired);

    /**
     * Build the Cache Manager field-watch set required by currently enabled policies.
     *
     * @param[in] registry  Policy registry to inspect.
     * @param[in] coreProxy Core proxy used to resolve active GPU entities.
     * @return Desired policy watch set, or the active-GPU lookup error.
     */
    static std::expected<std::set<PolicyWatch>, dcgmReturn_t> BuildDesiredPolicyWatches(DcgmPolicyRegistry &registry,
                                                                                        DcgmCoreProxy &coreProxy);

    /**
     * Take a copy of all registered field policies.
     *
     * @return Snapshot of all field policies.
     */
    std::vector<DcgmPolicy> SnapshotPolicies() const;

    /**
     * Replace all registry policies with a supplied snapshot.
     * Takes ownership of @p policies; callers should pass snapshots with std::move().
     *
     * @param[in] policies Replacement policy set.
     * @return DCGM_ST_OK on success, or the first registry error code.
     */
    dcgmReturn_t ReplaceRegistryPolicies(std::vector<DcgmPolicy> policies);

    /**
     * Restore registry, watches, and persisted file from a previous policy snapshot.
     * Takes ownership of @p policies; callers should pass snapshots with std::move().
     *
     * @param[in] policies Previous policy set.
     */
    void RestorePoliciesAndWatches(std::vector<DcgmPolicy> policies);

    /**
     * Restore one policy and refresh watches/persistence after a failed single-policy mutation.
     *
     * @param[in] policy Previous policy value.
     */
    void RestorePolicyAndWatches(DcgmPolicy const &policy);

    /**
     * Remove one policy and refresh watches/persistence after a failed create.
     *
     * @param[in] fieldId  Field ID the policy belongs to.
     * @param[in] policyId Policy ID to remove.
     */
    void RemovePolicyAndRefreshWatches(unsigned short fieldId, uint64_t policyId);
};

#endif // DCGMPOLICYMANAGER_H
