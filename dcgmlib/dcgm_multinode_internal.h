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
#include "dcgm_structs.h"


#ifdef __cplusplus
extern "C" {
#endif

typedef enum dcgmMultinodeTestType_enum
{
    mnubergemm
} dcgmMultinodeTestType_t;

typedef enum dcgmMultinodeRequestType_enum
{
    ReserveResources,
    ReleaseResources,
    DetectProcess,
    AuthorizeConnection,
    RevokeAuthorization,
    BroadcastRunParameters,
    GetNodeInfo,
} dcgmMultinodeRequestType_t;

typedef enum dcgmMultinodeStatus_enum
{
    UNKNOWN,
    READY,
    RUNNING,
    RESERVED,
    COMPLETED,
    FAILED
} dcgmMultinodeStatus_t;

typedef struct
{
    char driverVersion[DCGM_MAX_STR_LENGTH];
    char dcgmVersion[DCGM_MAX_STR_LENGTH];
} dcgmMultinodeNodeInfo_v1;

typedef struct
{
    unsigned int headNodeId;
    dcgmMultinodeStatus_t response;
} dcgmMultinodeResource_v1;

typedef struct
{
    unsigned int headNodeId;
} dcgmMultinodeAuthorization_v1;


typedef struct
{
    unsigned int headNodeId;
    dcgmRunMnDiag_v1 runMnDiag;
    char mnubergemmPath[DCGM_MAX_STR_LENGTH];
} dcgmMultinodeRunParams_v1;

typedef struct
{
    unsigned int version;
    dcgmMultinodeTestType_t testType;
    dcgmMultinodeRequestType_t requestType;
    union
    {
        dcgmMultinodeResource_v1 resource;
        dcgmMultinodeAuthorization_v1 authorization;
        dcgmMultinodeRunParams_v1 runParams;
        dcgmMultinodeNodeInfo_v1 nodeInfo;
    } requestData;
} dcgmMultinodeRequest_v1;

typedef dcgmMultinodeRequest_v1 dcgmMultinodeRequest_t;
#define dcgmMultinodeRequest_version1 MAKE_DCGM_VERSION(dcgmMultinodeRequest_v1, 1)

dcgmReturn_t DCGM_PUBLIC_API dcgmMultinodeRequest(dcgmHandle_t pDcgmHandle, dcgmMultinodeRequest_t *pRequest);

#ifdef __cplusplus
}
#endif