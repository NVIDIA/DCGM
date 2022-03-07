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
#ifndef DCGMPOLICYMANAGER_H
#define DCGMPOLICYMANAGER_H

#include "DcgmMutex.h"
#include "DcgmProtocol.h"
#include "dcgm_policy_structs.h"
#include <DcgmCoreProxy.h>

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

private:
    /* variables */
    DcgmCoreProxy mpCoreProxy;

    // mutex for use when manipulating members of this object
    DcgmMutex *m_mutex;

    /* Have we watched our fields in the cache manager yet for a given connection ID?.
       Don't set this directly. Call WatchFields() */
    std::map<dcgm_connection_id_t, int> m_haveWatchedFields;

    /* How many GPUs are currently valid in m_gpus? */
    int m_numGpus;
    dpm_gpu_t m_gpus[DCGM_MAX_NUM_DEVICES]; /* Per-GPU information */

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

    /*****************************************************************************
     * Method to watch all fields that need to be watched in order for the policy
     * manager to do its job. If the cache manager is already watching fields, this is a
     * no-op.
     *
     * @param connectionId IN: Which connection is this for?
     *
     *****************************************************************************/
    dcgmReturn_t WatchFields(dcgm_connection_id_t connectionId);
};

#endif // DCGMPOLICYMANAGER_H
