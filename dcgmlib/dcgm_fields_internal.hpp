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

#include <vector>

/********************************************************************************/
/* Definitions of internal DCGM fuctions */
/********************************************************************************/

/********************************************************************************/
/*
 * Get the NVML_GPM_METRIC_? ID for a given DCGM_FI_PROF_ fieldId.
 *
 * @param[in] fieldId           : Dcgm field ID to convert to NVML ID
 * @param[in] isPercentageField : Is the returned NVML field ID a percentage? Yes=true. No=false.
 *                                DCGM tracks these as 0.0-1.0 so we need to divide percentages
 *                                by 100.0.
 *
 * @return Nonzero Nvml metric ID on success.
 *         0 on failure
 */
unsigned int DcgmFieldIdToNvmlGpmMetricId(unsigned int fieldId, bool &isPercentageField);

/********************************************************************************/
/*
 * Get all valid field IDs
 *
 * @param[out] fieldIds         : Vector in which to place all valid field IDs
 *
 * @return 0 on success.
 *         Nonzero on error
 */
int DcgmFieldGetAllFieldIds(std::vector<unsigned short> &fieldIds);

/********************************************************************************/
/* Definitions of all the internal DCGM fields                                  */
/********************************************************************************/

/**
 * Memory utilization samples
 */
#define DCGM_FI_DEV_MEM_COPY_UTIL_SAMPLES 210

/*
 * SM utilization samples
 */
#define DCGM_FI_DEV_GPU_UTIL_SAMPLES 211

/**
 * Graphics processes running on the GPU.
 */
#define DCGM_FI_DEV_GRAPHICS_PIDS 220

/**
 * Compute processes running on the GPU.
 */
#define DCGM_FI_DEV_COMPUTE_PIDS 221


/********************************************************************************/
/* Field ID ranges                                                              */
/********************************************************************************/

/* Profiling field IDs */
#define DCGM_FI_PROF_FIRST_ID DCGM_FI_PROF_GR_ENGINE_ACTIVE
#define DCGM_FI_PROF_LAST_ID  DCGM_FI_PROF_C2C_RX_DATA_BYTES

/* Sysmon field IDs */
#define DCGM_FI_SYSMON_FIRST_ID DCGM_FI_DEV_CPU_UTIL_TOTAL
#define DCGM_FI_SYSMON_LAST_ID  DCGM_FI_DEV_CPU_MODEL

/* Macro to get whether a given fieldId is a profiling field or not */
#define DCGM_FIELD_ID_IS_PROF_FIELD(fieldId) ((fieldId) >= DCGM_FI_PROF_FIRST_ID && (fieldId) <= DCGM_FI_PROF_LAST_ID)
