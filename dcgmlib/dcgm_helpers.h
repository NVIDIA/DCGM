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

#ifndef DCGM_HELPERS_H
#define DCGM_HELPERS_H

#include "dcgm_api_export.h"
#include "dcgm_structs.h"

#ifdef __cplusplus
extern "C" {
#endif

/***************************************************************************************************/
/** @defgroup DCGMAPI_HELPERS Helpers
 *
 *  This chapter describes helpers packaged with the DCGM library for convenience
 *  @{
 */
/***************************************************************************************************/

/**
 * Check whether a given core belongs to a CPU
 *
 * The CPU's cores are provided in a dcgmCpuHierarchyOwnedCores_* structure,
 * which is populated by dcgmGetCpuHierarchy. dcgmCpuHierarchyOwnedCores_t lists
 * a CPU's cores in a bitmask; this function is a helper for working with that bitmask.
 *
 * @param coreId      IN: Check whether this core belongs to a CPU
 * @param ownedCores  IN: A structure listing a CPU's cores in a bitmask
 *
 * @return
 *        - 1         if the bitmask indicates that the CPU owns the provided core
 *        - 0         if the bitmask does not indicate the above or if an invalid argument is provided
 */
DCGM_PUBLIC_API unsigned int dcgmCpuHierarchyCpuOwnsCore(unsigned int coreId,
                                                         dcgmCpuHierarchyOwnedCores_v1 const *ownedCores);

/** @} */ // Closing for DCGMAPI_HELPERS
#ifdef __cplusplus
}
#endif

#endif // DCGM_HELPERS_H
