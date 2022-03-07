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
 * File:   DcgmConfigManager.h
 */

#ifndef DCGMCONFIGMANAGER_H
#define DCGMCONFIGMANAGER_H

#include "DcgmMutex.h"
#include "DcgmProtobuf.h"
#include "DcgmSettings.h"
#include "dcgm_agent.h"
#include "dcgm_config_structs.h"
#include <DcgmCoreProxy.h>
#include <iostream>
#include <string>


#include <DcgmModule.h>

/* Helper class for aggregating request statuses of variable sizes without excessive copies */
class DcgmConfigManagerStatusList
{
public:
    unsigned int *m_errorCount;
    unsigned int m_maxNumErrors;
    dcgm_config_status_t *m_statuses;

    DcgmConfigManagerStatusList(unsigned int maxNumErrors, unsigned int *errorCount, dcgm_config_status_t *statuses)
    {
        m_errorCount   = errorCount;
        m_maxNumErrors = maxNumErrors;
        m_statuses     = statuses;

        *errorCount = 0;
    }
    ~DcgmConfigManagerStatusList() {};

    void AddStatus(unsigned int gpuId, int fieldId, dcgmReturn_t errorCode)
    {
        if ((*m_errorCount) >= m_maxNumErrors)
            return; /* Don't overflow */

        m_statuses[*m_errorCount].gpuId     = gpuId;
        m_statuses[*m_errorCount].fieldId   = fieldId;
        m_statuses[*m_errorCount].errorCode = errorCode;
        (*m_errorCount)++;
    }
};


class DcgmConfigManager
{
public:
    DcgmConfigManager(dcgmCoreCallbacks_t &pDcc);
    virtual ~DcgmConfigManager();

    /*****************************************************************************
     * This method is used to set configuration for a group.
     * Best effort to configure all the properties.
     *
     * @param groupId       IN : Group Identifier
     * @param pDeviceConfig IN : The device configuration to be set
     * @param statuses     OUT : Per-GPU status codes resulting from this operation
     *
     * @return
     * DCGM_ST_OK               :   On Success
     * DCGM_ST_GENERIC_ERROR    :   If any of the property is not configured. See statuses for details.
     *
     *****************************************************************************/
    dcgmReturn_t SetConfig(unsigned int groupId, dcgmConfig_t *setConfig, DcgmConfigManagerStatusList *statusList);

    /**
     * Used to get the target configuration for the group
     * @param groupId       IN: Group ID
     * @param numConfigs   OUT: How many configs were written to configs[]
     * @param statuses     OUT: Per-GPU status codes resulting from this operation
     *
     * @return
     */
    dcgmReturn_t GetTargetConfig(unsigned int groupId,
                                 unsigned int *numConfigs,
                                 dcgmConfig_t *configs,
                                 DcgmConfigManagerStatusList *statusList);

    /**
     * Used to get the actual configuration for the group
     * @param groupId       IN: Group ID
     * @param numConfigs   OUT: How many configs were written to configs[]
     * @param statuses     OUT: Per-GPU status codes resulting from this operation
     *
     * @return
     */
    dcgmReturn_t GetCurrentConfig(unsigned int groupId,
                                  unsigned int *numConfigs,
                                  dcgmConfig_t *configs,
                                  DcgmConfigManagerStatusList *statusList);

    /*****************************************************************************
     * Used to enforce previously set configuration for the specified GPU or group. The method is to enforce
     * device configuration such as ecc mode, power limits, clocks and compute mode.
     * Must be called after GPU reset is called in order to retain the configuration before reset.
     *
     * @return
     *****************************************************************************/
    dcgmReturn_t EnforceConfigGroup(unsigned int groupId, DcgmConfigManagerStatusList *statusList);
    dcgmReturn_t EnforceConfigGpu(unsigned int gpuId, DcgmConfigManagerStatusList *statusList);

private:
    /*****************************************************************************
     * Helper method to get target configuration for a GPU ID
     *****************************************************************************/
    dcgmConfig_t *HelperGetTargetConfig(unsigned int gpuId);

    /*****************************************************************************
     * This method is used to merge setConfig into the target configuration for a GPU
     * for a given field. If fieldId is non-blank in setConfig, it will be applied
     * to our local configuration
     *****************************************************************************/
    void HelperMergeTargetConfiguration(unsigned int gpuId, unsigned int fieldId, dcgmConfig_t *setConfig);

    /*****************************************************************************
     * Helper method to configure ECC Mode
     * If Reset is needed, it's returned in pIsResetNeeded
     *****************************************************************************/
    dcgmReturn_t HelperSetEccMode(unsigned int gpuId,
                                  dcgmConfig_t *setConfig,
                                  dcgmConfig_t *currentConfig,
                                  bool *pIsResetNeeded);

    /*****************************************************************************
     * Helper method to set power limit for the GPU
     *****************************************************************************/
    dcgmReturn_t HelperSetPowerLimit(unsigned int gpuId, dcgmConfig_t *setConfig);

    /*****************************************************************************
     * Helper method to set the perf state
     *****************************************************************************/
    dcgmReturn_t HelperSetPerfState(unsigned int gpuId, dcgmConfig_t *setConfig);

    /*****************************************************************************
     * Helper method to set Compute Mode
     *****************************************************************************/
    dcgmReturn_t HelperSetComputeMode(unsigned int gpuId, dcgmConfig_t *setConfig);

    /*****************************************************************************
     * This method is used to set sync boost on a group of GPUs
     *****************************************************************************/
    dcgmReturn_t SetSyncBoost(unsigned int gpuIdList[],
                              unsigned int count,
                              dcgmConfig_t *setConfig,
                              DcgmConfigManagerStatusList *statusList);

    /*****************************************************************************
     * This method works as a helper method to enforce configuration on a GPU ID
     *****************************************************************************/
    dcgmReturn_t HelperEnforceConfig(unsigned int gpuId, DcgmConfigManagerStatusList *statusList);

    /*****************************************************************************
     * Populate a dcgmConfig_t with the current LIVE config for a GPU from the cache manager
     *****************************************************************************/
    dcgmReturn_t GetCurrentConfigGpu(unsigned int gpuId, dcgmConfig_t *config);

    /******************************************************************************
     * Helper to set the config for a single GPU
     *****************************************************************************/
    dcgmReturn_t SetConfigGpu(unsigned int gpuId, dcgmConfig_t *setConfig, DcgmConfigManagerStatusList *statusList);

    /*****************************************************************************
     * This method returns nonzero if the host engine is running as root and 0 if the host
     * engine is running as non-root.
     *****************************************************************************/
    bool RunningAsRoot(void);

    /* Array of currently-active target configs. These can be null, so you may have to alloc them */
    dcgmConfig_t *m_activeConfig[DCGM_MAX_NUM_DEVICES];

    DcgmCoreProxy mpCoreProxy;
    unsigned int mClocksConfigured;

    DcgmMutex *m_mutex; /* Lock used for accessing default config data structure */
};

#endif /* DCGMCONFIGMANAGER_H */
