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

#include "DcgmModuleDiag.h"
#include "dcgm_diag_structs.h"

#include <DcgmConfigManager.h>
#include <DcgmDiagResponseWrapper.h>
#include <DcgmLogging.h>
#include <DcgmStringHelpers.h>
#include <dcgm_api_export.h>
#include <dcgm_structs.h>

#include <bit>

namespace
{
void CustomRunDiagSettings(void const *legacyDrd, unsigned int legacyDrdVersion, dcgmRunDiag_v10 &newDrd)
{
    auto setGroupAndEntityIds = [&](auto legacyDrdPtr) {
        // In drd7 and drd8, gpuList takes precedence over groupId. Conversely, in drd9 and drd10,
        // groupId has higher priority than entityId. To maintain legacy drd's prioritization scheme,
        // we set groupId to DCGM_GROUP_NULL unless the gpuList is empty.
        newDrd.groupId = DCGM_GROUP_NULL;
        if (legacyDrdPtr->gpuList[0] == '\0')
        {
            newDrd.groupId = legacyDrdPtr->groupId;
        }
        SafeCopyTo<sizeof(newDrd.entityIds), sizeof(legacyDrdPtr->gpuList)>(newDrd.entityIds, legacyDrdPtr->gpuList);
    };
    auto setExpectedNumEntities = [&](auto legacyDrdPtr) {
        // expectedNumEntities exists in runDiag_v8, runDiag_v9, and runDiag_v10 only.
        SafeCopyTo<sizeof(newDrd.expectedNumEntities), sizeof(legacyDrdPtr->expectedNumEntities)>(
            newDrd.expectedNumEntities, legacyDrdPtr->expectedNumEntities);
    };
    auto setEntityIds = [&](auto legacyDrdPtr) {
        SafeCopyTo<sizeof(newDrd.entityIds), sizeof(legacyDrdPtr->entityIds)>(newDrd.entityIds,
                                                                              legacyDrdPtr->entityIds);
    };
    auto setWatchFrequency = [&](auto legacyDrdPtr) {
        newDrd.watchFrequency = legacyDrdPtr->watchFrequency;
    };

    switch (legacyDrdVersion)
    {
        case dcgmRunDiag_version7:
        {
            auto legacyDrdPtr = static_cast<const dcgmRunDiag_v7 *>(legacyDrd);
            setGroupAndEntityIds(legacyDrdPtr);
            break;
        }
        break;
        case dcgmRunDiag_version8:
        {
            auto legacyDrdPtr = static_cast<const dcgmRunDiag_v8 *>(legacyDrd);
            setGroupAndEntityIds(legacyDrdPtr);
            setExpectedNumEntities(legacyDrdPtr);
        }
        break;
        case dcgmRunDiag_version9:
        {
            auto legacyDrdPtr = static_cast<const dcgmRunDiag_v9 *>(legacyDrd);
            setExpectedNumEntities(legacyDrdPtr);
            setEntityIds(legacyDrdPtr);
            setWatchFrequency(legacyDrdPtr);
        }
        break;
        default:
            log_debug("RunDiag version {} not supported in this function.", legacyDrdVersion);
    }
}

template <typename T>
dcgmRunDiag_v10 ProduceLatestDcgmRunDiag(const T &legacyDrd)
    requires std::is_same_v<T, dcgmRunDiag_v9> || std::is_same_v<T, dcgmRunDiag_v8> || std::is_same_v<T, dcgmRunDiag_v7>
{
    dcgmRunDiag_v10 newDrd {};

    newDrd.version    = dcgmRunDiag_version10;
    newDrd.flags      = legacyDrd.flags;
    newDrd.debugLevel = legacyDrd.debugLevel;
    newDrd.groupId    = legacyDrd.groupId;
    newDrd.validate   = legacyDrd.validate;
    std::memcpy(&newDrd.testNames, &legacyDrd.testNames, sizeof(newDrd.testNames));
    for (unsigned int i = 0; i < DCGM_MAX_TEST_PARMS; ++i)
    {
        SafeCopyTo(newDrd.testParms[i], legacyDrd.testParms[i]);
    }
    SafeCopyTo<sizeof(newDrd.fakeGpuList), sizeof(legacyDrd.fakeGpuList)>(newDrd.fakeGpuList, legacyDrd.fakeGpuList);
    SafeCopyTo<sizeof(newDrd.debugLogFile), sizeof(legacyDrd.debugLogFile)>(newDrd.debugLogFile,
                                                                            legacyDrd.debugLogFile);
    SafeCopyTo<sizeof(newDrd.statsPath), sizeof(legacyDrd.statsPath)>(newDrd.statsPath, legacyDrd.statsPath);
    SafeCopyTo<sizeof(newDrd.configFileContents), sizeof(legacyDrd.configFileContents)>(newDrd.configFileContents,
                                                                                        legacyDrd.configFileContents);
    std::memcpy(&newDrd.clocksEventMask,
                &legacyDrd.clocksEventMask,
                std::min(sizeof(newDrd.clocksEventMask), sizeof(legacyDrd.clocksEventMask)));
    SafeCopyTo<sizeof(newDrd.pluginPath), sizeof(legacyDrd.pluginPath)>(newDrd.pluginPath, legacyDrd.pluginPath);
    newDrd.currentIteration  = legacyDrd.currentIteration;
    newDrd.totalIterations   = legacyDrd.totalIterations;
    newDrd.timeoutSeconds    = legacyDrd.timeoutSeconds;
    newDrd.failCheckInterval = legacyDrd.failCheckInterval;

    CustomRunDiagSettings(&legacyDrd, legacyDrd.version, newDrd);
    return newDrd;
}
} // namespace

/*****************************************************************************/
DcgmModuleDiag::DcgmModuleDiag(dcgmCoreCallbacks_t &dcc)
    : DcgmModuleWithCoreProxy(dcc)
{
    mpDiagManager = std::make_unique<DcgmDiagManager>(dcc);
}

/*****************************************************************************/
DcgmModuleDiag::~DcgmModuleDiag() = default;

/*****************************************************************************/
/**
 * Process a diagnostic run request version 12
 * Compatibility:
 * - Supports diagResponse_v12, v11, v10
 * - Supports diagRun_v10
 */
dcgmReturn_t DcgmModuleDiag::ProcessRun_v12(dcgm_diag_msg_run_v12 *msg)
{
    dcgmReturn_t dcgmReturn;
    DcgmDiagResponseWrapper drw;

    dcgmReturn = CheckVersion(&msg->header, dcgm_diag_msg_run_version12);
    if (DCGM_ST_OK != dcgmReturn)
    {
        return dcgmReturn; /* Logging handled by helper method */
    }

    drw.SetVersion(&msg->diagResponse);

    /* Sanitize the inputs */
    dcgmTerminateCharBuffer(msg->runDiag.fakeGpuList);
    dcgmTerminateCharBuffer(msg->runDiag.debugLogFile);
    dcgmTerminateCharBuffer(msg->runDiag.statsPath);
    dcgmTerminateCharBuffer(msg->runDiag.configFileContents);
    dcgmTerminateCharBuffer(msg->runDiag.clocksEventMask);
    dcgmTerminateCharBuffer(msg->runDiag.pluginPath);
    dcgmTerminateCharBuffer(msg->runDiag._unusedBuf);
    dcgmTerminateCharBuffer(msg->runDiag.entityIds);
    dcgmTerminateCharBuffer(msg->runDiag.expectedNumEntities);
    dcgmTerminateCharBuffer(msg->runDiag.ignoreErrorCodes);

    size_t i;
    for (i = 0; i < DCGM_ARRAY_CAPACITY(msg->runDiag.testNames); i++)
    {
        dcgmTerminateCharBuffer(msg->runDiag.testNames[i]);
    }
    for (i = 0; i < DCGM_ARRAY_CAPACITY(msg->runDiag.testParms); i++)
    {
        dcgmTerminateCharBuffer(msg->runDiag.testParms[i]);
    }

    /* Run the diag */
    dcgmReturn = mpDiagManager->RunDiagAndAction(&msg->runDiag, msg->action, drw, msg->header.connectionId);
    if (DCGM_ST_OK != dcgmReturn)
    {
        log_error("RunDiagAndAction returned {}", dcgmReturn);
    }

    return dcgmReturn;
}

/*****************************************************************************/
/**
 * Process a diagnostic run request version 11
 * Compatibility:
 * - Supports diagResponse_v11, v10
 * - Supports diagRun_v10
 */
dcgmReturn_t DcgmModuleDiag::ProcessRun_v11(dcgm_diag_msg_run_v11 *msg)
{
    dcgmReturn_t dcgmReturn;
    DcgmDiagResponseWrapper drw;

    dcgmReturn = CheckVersion(&msg->header, dcgm_diag_msg_run_version11);
    if (DCGM_ST_OK != dcgmReturn)
    {
        return dcgmReturn; /* Logging handled by helper method */
    }

    drw.SetVersion(&msg->diagResponse);

    /* Sanitize the inputs */
    dcgmTerminateCharBuffer(msg->runDiag.fakeGpuList);
    dcgmTerminateCharBuffer(msg->runDiag.debugLogFile);
    dcgmTerminateCharBuffer(msg->runDiag.statsPath);
    dcgmTerminateCharBuffer(msg->runDiag.configFileContents);
    dcgmTerminateCharBuffer(msg->runDiag.clocksEventMask);
    dcgmTerminateCharBuffer(msg->runDiag.pluginPath);
    dcgmTerminateCharBuffer(msg->runDiag._unusedBuf);
    dcgmTerminateCharBuffer(msg->runDiag.entityIds);
    dcgmTerminateCharBuffer(msg->runDiag.expectedNumEntities);
    dcgmTerminateCharBuffer(msg->runDiag.ignoreErrorCodes);

    size_t i;
    for (i = 0; i < DCGM_ARRAY_CAPACITY(msg->runDiag.testNames); i++)
    {
        dcgmTerminateCharBuffer(msg->runDiag.testNames[i]);
    }
    for (i = 0; i < DCGM_ARRAY_CAPACITY(msg->runDiag.testParms); i++)
    {
        dcgmTerminateCharBuffer(msg->runDiag.testParms[i]);
    }

    /* Run the diag */
    dcgmReturn = mpDiagManager->RunDiagAndAction(&msg->runDiag, msg->action, drw, msg->header.connectionId);
    if (DCGM_ST_OK != dcgmReturn)
    {
        log_error("RunDiagAndAction returned {}", dcgmReturn);
    }

    return dcgmReturn;
}

/*****************************************************************************/
/**
 * Process a diagnostic run request version 10
 * Compatibility:
 * - Supports diagResponse_v11, v10, v9
 * - Supports diagRun_v9
 */
dcgmReturn_t DcgmModuleDiag::ProcessRun_v10(dcgm_diag_msg_run_v10 *msg)
{
    dcgmReturn_t dcgmReturn;
    DcgmDiagResponseWrapper drw;
    dcgmRunDiag_v10 drd10 = ProduceLatestDcgmRunDiag(msg->runDiag);

    dcgmReturn = CheckVersion(&msg->header, dcgm_diag_msg_run_version10);
    if (DCGM_ST_OK != dcgmReturn)
    {
        return dcgmReturn; /* Logging handled by helper method */
    }

    // Upgrade v10 -> v11
    drw.SetVersion<dcgmDiagResponse_v11>(std::bit_cast<dcgmDiagResponse_v11 *>(&msg->diagResponse));

    /* Sanitize the inputs */
    dcgmTerminateCharBuffer(drd10.fakeGpuList);
    dcgmTerminateCharBuffer(drd10.debugLogFile);
    dcgmTerminateCharBuffer(drd10.statsPath);
    dcgmTerminateCharBuffer(drd10.configFileContents);
    dcgmTerminateCharBuffer(drd10.clocksEventMask);
    dcgmTerminateCharBuffer(drd10.pluginPath);
    dcgmTerminateCharBuffer(drd10._unusedBuf);
    dcgmTerminateCharBuffer(drd10.entityIds);
    dcgmTerminateCharBuffer(drd10.expectedNumEntities);
    dcgmTerminateCharBuffer(drd10.ignoreErrorCodes);

    size_t i;
    for (i = 0; i < DCGM_ARRAY_CAPACITY(drd10.testNames); i++)
    {
        dcgmTerminateCharBuffer(drd10.testNames[i]);
    }
    for (i = 0; i < DCGM_ARRAY_CAPACITY(drd10.testParms); i++)
    {
        dcgmTerminateCharBuffer(drd10.testParms[i]);
    }

    /* Run the diag */
    dcgmReturn = mpDiagManager->RunDiagAndAction(&drd10, msg->action, drw, msg->header.connectionId);
    if (DCGM_ST_OK != dcgmReturn)
    {
        log_error("RunDiagAndAction returned {}", dcgmReturn);
    }

    return dcgmReturn;
}

/*****************************************************************************/
/**
 * Process a diagnostic run request version 9
 * Compatibility:
 * - Supports diagResponse_v10, v9, v8
 * - Supports diagRun_v8
 */
dcgmReturn_t DcgmModuleDiag::ProcessRun_v9(dcgm_diag_msg_run_v9 *msg)
{
    dcgmReturn_t dcgmReturn;
    DcgmDiagResponseWrapper drw;
    dcgmRunDiag_v10 drd10 = ProduceLatestDcgmRunDiag(msg->runDiag);

    dcgmReturn = CheckVersion(&msg->header, dcgm_diag_msg_run_version9);
    if (DCGM_ST_OK != dcgmReturn)
    {
        return dcgmReturn; /* Logging handled by helper method */
    }

    // Upgrade to v10
    drw.SetVersion<dcgmDiagResponse_v10>(std::bit_cast<dcgmDiagResponse_v10 *>(&msg->diagResponse));

    /* Sanitize the inputs */
    dcgmTerminateCharBuffer(drd10.fakeGpuList);
    dcgmTerminateCharBuffer(drd10.debugLogFile);
    dcgmTerminateCharBuffer(drd10.statsPath);
    dcgmTerminateCharBuffer(drd10.configFileContents);
    dcgmTerminateCharBuffer(drd10.clocksEventMask);
    dcgmTerminateCharBuffer(drd10.pluginPath);
    dcgmTerminateCharBuffer(drd10._unusedBuf);
    dcgmTerminateCharBuffer(drd10.entityIds);
    dcgmTerminateCharBuffer(drd10.expectedNumEntities);
    dcgmTerminateCharBuffer(drd10.ignoreErrorCodes);

    size_t i;
    for (i = 0; i < DCGM_ARRAY_CAPACITY(drd10.testNames); i++)
    {
        dcgmTerminateCharBuffer(drd10.testNames[i]);
    }
    for (i = 0; i < DCGM_ARRAY_CAPACITY(drd10.testParms); i++)
    {
        dcgmTerminateCharBuffer(drd10.testParms[i]);
    }

    /* Run the diag */
    dcgmReturn = mpDiagManager->RunDiagAndAction(&drd10, msg->action, drw, msg->header.connectionId);
    if (DCGM_ST_OK != dcgmReturn)
    {
        log_error("RunDiagAndAction returned {}", dcgmReturn);
    }

    return dcgmReturn;
}

/*****************************************************************************/
/**
 * Process a diagnostic run request version 8
 * Compatibility:
 * - Supports diagResponse_v10, v9, v8, v7
 * - Supports diagRun_v7
 */
dcgmReturn_t DcgmModuleDiag::ProcessRun_v8(dcgm_diag_msg_run_v8 *msg)
{
    dcgmReturn_t dcgmReturn;
    DcgmDiagResponseWrapper drw;
    dcgmRunDiag_v10 drd10 = ProduceLatestDcgmRunDiag(msg->runDiag);

    dcgmReturn = CheckVersion(&msg->header, dcgm_diag_msg_run_version8);
    if (DCGM_ST_OK != dcgmReturn)
    {
        return dcgmReturn; /* Logging handled by helper method */
    }
    else
    {
        // Upgrade to v10
        drw.SetVersion<dcgmDiagResponse_v10>(std::bit_cast<dcgmDiagResponse_v10 *>(&msg->diagResponse));
    }

    /* Sanitize the inputs */
    dcgmTerminateCharBuffer(drd10.fakeGpuList);
    dcgmTerminateCharBuffer(drd10.debugLogFile);
    dcgmTerminateCharBuffer(drd10.statsPath);
    dcgmTerminateCharBuffer(drd10.configFileContents);
    dcgmTerminateCharBuffer(drd10.clocksEventMask);
    dcgmTerminateCharBuffer(drd10.pluginPath);
    dcgmTerminateCharBuffer(drd10._unusedBuf);
    dcgmTerminateCharBuffer(drd10.entityIds);
    dcgmTerminateCharBuffer(drd10.expectedNumEntities);
    dcgmTerminateCharBuffer(drd10.ignoreErrorCodes);

    size_t i;
    for (i = 0; i < DCGM_ARRAY_CAPACITY(drd10.testNames); i++)
    {
        dcgmTerminateCharBuffer(drd10.testNames[i]);
    }
    for (i = 0; i < DCGM_ARRAY_CAPACITY(drd10.testParms); i++)
    {
        dcgmTerminateCharBuffer(drd10.testParms[i]);
    }

    /* Run the diag */
    dcgmReturn = mpDiagManager->RunDiagAndAction(&drd10, msg->action, drw, msg->header.connectionId);
    if (DCGM_ST_OK != dcgmReturn)
    {
        log_error("RunDiagAndAction returned {}", dcgmReturn);
    }

    return dcgmReturn;
}

/*****************************************************************************/
/**
 * Process a diagnostic run request version 7
 * Compatibility:
 * - Supports diagResponse_v9, v8, v7
 * - Supports diagRun_v7
 */
dcgmReturn_t DcgmModuleDiag::ProcessRun_v7(dcgm_diag_msg_run_v7 *msg)
{
    dcgmReturn_t dcgmReturn;
    DcgmDiagResponseWrapper drw;
    dcgmRunDiag_v10 drd10 = ProduceLatestDcgmRunDiag(msg->runDiag);

    dcgmReturn = CheckVersion(&msg->header, dcgm_diag_msg_run_version7);
    if (DCGM_ST_OK != dcgmReturn)
    {
        return dcgmReturn; /* Logging handled by helper method */
    }
    else
    {
        // Upgrade to v9
        drw.SetVersion<dcgmDiagResponse_v9>(std::bit_cast<dcgmDiagResponse_v9 *>(&msg->diagResponse));
    }

    /* Sanitize the inputs */
    dcgmTerminateCharBuffer(drd10.fakeGpuList);
    dcgmTerminateCharBuffer(drd10.debugLogFile);
    dcgmTerminateCharBuffer(drd10.statsPath);
    dcgmTerminateCharBuffer(drd10.configFileContents);
    dcgmTerminateCharBuffer(drd10.clocksEventMask);
    dcgmTerminateCharBuffer(drd10.pluginPath);
    dcgmTerminateCharBuffer(drd10._unusedBuf);
    dcgmTerminateCharBuffer(drd10.entityIds);
    dcgmTerminateCharBuffer(drd10.expectedNumEntities);
    dcgmTerminateCharBuffer(drd10.ignoreErrorCodes);

    size_t i;
    for (i = 0; i < DCGM_ARRAY_CAPACITY(drd10.testNames); i++)
    {
        dcgmTerminateCharBuffer(drd10.testNames[i]);
    }
    for (i = 0; i < DCGM_ARRAY_CAPACITY(drd10.testParms); i++)
    {
        dcgmTerminateCharBuffer(drd10.testParms[i]);
    }

    /* Run the diag */
    dcgmReturn = mpDiagManager->RunDiagAndAction(&drd10, msg->action, drw, msg->header.connectionId);
    if (DCGM_ST_OK != dcgmReturn)
    {
        log_error("RunDiagAndAction returned {}", dcgmReturn);
    }

    return dcgmReturn;
}

/*****************************************************************************/
/**
 * Process a diagnostic run request version 6
 * Compatibility:
 * - Supports diagResponse_v8, v7
 * - Supports diagRun_v7
 */
dcgmReturn_t DcgmModuleDiag::ProcessRun_v6(dcgm_diag_msg_run_v6 *msg)
{
    dcgmReturn_t dcgmReturn;
    DcgmDiagResponseWrapper drw;
    dcgmRunDiag_v10 drd10 = ProduceLatestDcgmRunDiag(msg->runDiag);

    dcgmReturn = CheckVersion(&msg->header, dcgm_diag_msg_run_version6);
    if (DCGM_ST_OK != dcgmReturn)
    {
        return dcgmReturn; /* Logging handled by helper method */
    }
    else
    {
        // Upgrade to v8
        drw.SetVersion<dcgmDiagResponse_v8>(std::bit_cast<dcgmDiagResponse_v8 *>(&msg->diagResponse));
    }

    /* Sanitize the inputs */
    dcgmTerminateCharBuffer(drd10.fakeGpuList);
    dcgmTerminateCharBuffer(drd10.debugLogFile);
    dcgmTerminateCharBuffer(drd10.statsPath);
    dcgmTerminateCharBuffer(drd10.configFileContents);
    dcgmTerminateCharBuffer(drd10.clocksEventMask);
    dcgmTerminateCharBuffer(drd10.pluginPath);
    dcgmTerminateCharBuffer(drd10._unusedBuf);
    dcgmTerminateCharBuffer(drd10.entityIds);
    dcgmTerminateCharBuffer(drd10.expectedNumEntities);
    dcgmTerminateCharBuffer(drd10.ignoreErrorCodes);

    size_t i;
    for (i = 0; i < DCGM_ARRAY_CAPACITY(drd10.testNames); i++)
    {
        dcgmTerminateCharBuffer(drd10.testNames[i]);
    }
    for (i = 0; i < DCGM_ARRAY_CAPACITY(drd10.testParms); i++)
    {
        dcgmTerminateCharBuffer(drd10.testParms[i]);
    }

    /* Run the diag */
    dcgmReturn = mpDiagManager->RunDiagAndAction(&drd10, msg->action, drw, msg->header.connectionId);
    if (DCGM_ST_OK != dcgmReturn)
    {
        log_error("RunDiagAndAction returned {}", dcgmReturn);
    }

    return dcgmReturn;
}

/*****************************************************************************/
/**
 * Process a diagnostic run request version 5
 * Compatibility:
 * - Supports diagResponse_v7
 * - Supports diagRun_v7
 */
dcgmReturn_t DcgmModuleDiag::ProcessRun_v5(dcgm_diag_msg_run_v5 *msg)
{
    dcgmReturn_t dcgmReturn;
    DcgmDiagResponseWrapper drw;
    dcgmRunDiag_v10 drd10 = ProduceLatestDcgmRunDiag(msg->runDiag);

    dcgmReturn = CheckVersion(&msg->header, dcgm_diag_msg_run_version5);
    if (DCGM_ST_OK != dcgmReturn)
    {
        return dcgmReturn; /* Logging handled by helper method */
    }
    else
    {
        // Upgrade to v7
        drw.SetVersion<dcgmDiagResponse_v7>(std::bit_cast<dcgmDiagResponse_v7 *>(&msg->diagResponse));
    }

    /* Sanitize the inputs */
    dcgmTerminateCharBuffer(drd10.fakeGpuList);
    dcgmTerminateCharBuffer(drd10.debugLogFile);
    dcgmTerminateCharBuffer(drd10.statsPath);
    dcgmTerminateCharBuffer(drd10.configFileContents);
    dcgmTerminateCharBuffer(drd10.clocksEventMask);
    dcgmTerminateCharBuffer(drd10.pluginPath);
    dcgmTerminateCharBuffer(drd10._unusedBuf);
    dcgmTerminateCharBuffer(drd10.entityIds);
    dcgmTerminateCharBuffer(drd10.expectedNumEntities);
    dcgmTerminateCharBuffer(drd10.ignoreErrorCodes);

    size_t i;
    for (i = 0; i < DCGM_ARRAY_CAPACITY(drd10.testNames); i++)
    {
        dcgmTerminateCharBuffer(drd10.testNames[i]);
    }
    for (i = 0; i < DCGM_ARRAY_CAPACITY(drd10.testParms); i++)
    {
        dcgmTerminateCharBuffer(drd10.testParms[i]);
    }

    /* Run the diag */
    dcgmReturn = mpDiagManager->RunDiagAndAction(&drd10, msg->action, drw, msg->header.connectionId);
    if (DCGM_ST_OK != dcgmReturn)
    {
        DCGM_LOG_ERROR << "RunDiagAndAction returned " << dcgmReturn;
    }

    return dcgmReturn;
}

/*****************************************************************************/

inline dcgmReturn_t DcgmModuleDiag::ProcessRun(dcgm_module_command_header_t *moduleCommand)
{
    if (moduleCommand->version == dcgm_diag_msg_run_version12)
    {
        return ProcessRun_v12(std::bit_cast<dcgm_diag_msg_run_v12 *>(moduleCommand));
    }
    else if (moduleCommand->version == dcgm_diag_msg_run_version11)
    {
        return ProcessRun_v11(std::bit_cast<dcgm_diag_msg_run_v11 *>(moduleCommand));
    }
    else if (moduleCommand->version == dcgm_diag_msg_run_version10)
    {
        return ProcessRun_v10(std::bit_cast<dcgm_diag_msg_run_v10 *>(moduleCommand));
    }
    else if (moduleCommand->version == dcgm_diag_msg_run_version9)
    {
        return ProcessRun_v9(std::bit_cast<dcgm_diag_msg_run_v9 *>(moduleCommand));
    }
    else if (moduleCommand->version == dcgm_diag_msg_run_version8)
    {
        return ProcessRun_v8(std::bit_cast<dcgm_diag_msg_run_v8 *>(moduleCommand));
    }
    else if (moduleCommand->version == dcgm_diag_msg_run_version7)
    {
        return ProcessRun_v7((dcgm_diag_msg_run_v7 *)moduleCommand);
    }
    else if (moduleCommand->version == dcgm_diag_msg_run_version6)
    {
        return ProcessRun_v6((dcgm_diag_msg_run_v6 *)moduleCommand);
    }
    else if (moduleCommand->version == dcgm_diag_msg_run_version5)
    {
        return ProcessRun_v5((dcgm_diag_msg_run_v5 *)moduleCommand);
    }

    log_error("Version mismatch {:x} != {:x}", moduleCommand->version, dcgm_diag_msg_run_version);
    return DCGM_ST_VER_MISMATCH;
}

/*****************************************************************************/
dcgmReturn_t DcgmModuleDiag::ProcessStop(dcgm_diag_msg_stop_t * /* msg */)
{
    return mpDiagManager->StopRunningDiag();
}

dcgmReturn_t DcgmModuleDiag::ProcessCoreMessage(dcgm_module_command_header_t *moduleCommand)
{
    dcgmReturn_t retSt = DCGM_ST_OK;

    switch (moduleCommand->subCommand)
    {
        case DCGM_CORE_SR_LOGGING_CHANGED:
            OnLoggingSeverityChange((dcgm_core_msg_logging_changed_t *)moduleCommand);
            break;

        case DCGM_CORE_SR_PAUSE_RESUME:
            log_debug("Received Pause/Resume subcommand");
            m_isPaused.store(((dcgm_core_msg_pause_resume_v1 *)moduleCommand)->pause, std::memory_order_relaxed);
            break;

        default:
            DCGM_LOG_DEBUG << "Unknown subcommand: " << static_cast<int>(moduleCommand->subCommand);
            return DCGM_ST_FUNCTION_NOT_FOUND;
    }

    return retSt;
}

/*****************************************************************************/
dcgmReturn_t DcgmModuleDiag::ProcessMessage(dcgm_module_command_header_t *moduleCommand)
{
    dcgmReturn_t retSt = DCGM_ST_OK;

    if (moduleCommand == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }

    if (moduleCommand->moduleId == DcgmModuleIdCore)
    {
        retSt = ProcessCoreMessage(moduleCommand);
    }
    else
    {
        /*
         * If the module is paused, we prevent accidental running of the diagnostic.
         * This is a safety net for EUD that pauses all DCGM modules before running the EUD binary
         * to prevent unwanted side effects.
         * Commands other than SR_RUN are still allowed so that we are able to interrupt a diagnostic even if the
         * module is paused. The use case is to be able to interrupt EUD tests.
         */
        switch (moduleCommand->subCommand)
        {
            case DCGM_DIAG_SR_RUN:
                if (m_isPaused.load(std::memory_order_relaxed))
                {
                    log_info("The Diag module is paused. Ignoring the run command.");
                    retSt = DCGM_ST_PAUSED;
                }
                else
                {
                    retSt = ProcessRun(moduleCommand);
                }
                break;

            case DCGM_DIAG_SR_STOP:
                retSt = ProcessStop((dcgm_diag_msg_stop_t *)moduleCommand);
                break;

            default:
                log_debug("Unknown subcommand: {}", moduleCommand->subCommand);
                retSt = DCGM_ST_FUNCTION_NOT_FOUND;
                break;
        }
    }

    return retSt;
}

extern "C" {
/*****************************************************************************/
DCGM_PUBLIC_API DcgmModule *dcgm_alloc_module_instance(dcgmCoreCallbacks_t *dcc)
{
    if (dcc == nullptr)
    {
        log_error("Cannot instantiate the diag class without libdcgm callback functions!");
        return nullptr;
    }

    return SafeWrapper([=] { return new DcgmModuleDiag(*dcc); });
}

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
