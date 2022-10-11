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

#pragma once

#include "dcgm_api_export.h"
#include "dcgm_structs.h"

#ifdef __cplusplus
extern "C" {
#endif

/*************************************************************************/
/**
 * Set a module to be blacklisted. This module will be prevented from being loaded
 * if it hasn't been loaded already. Modules are lazy-loaded as they are used by
 * DCGM APIs, so it's important to call this API soon after the host engine has been started.
 * You can also pass --blacklist-modules to the nv-hostengine binary to make sure modules
 * get blacklisted immediately after the host engine starts up.
 *
 * @param pDcgmHandle        IN: DCGM Handle
 * @param moduleId           IN: ID of the module to blacklist. Use \ref dcgmModuleGetStatuses to get a list of valid
 *                               module IDs.
 *
 * @return
 *        - \ref DCGM_ST_OK         if the module has been blacklisted.
 *        - \ref DCGM_ST_IN_USE     if the module has already been loaded and cannot be blacklisted.
 *        - \ref DCGM_ST_BADPARAM   if a parameter is missing or bad.
 *
 */
dcgmReturn_t DCGM_PUBLIC_API dcgmModuleBlacklist(dcgmHandle_t pDcgmHandle, dcgmModuleId_t moduleId);

/*************************************************************************/
/**
 * Toggle the state of introspection metadata gathering in DCGM.  Metadata gathering will increase the memory usage
 * of DCGM so that it can store the metadata it gathers.
 *
 * @param pDcgmHandle                   IN: DCGM Handle
 * @param enabledState                  IN: The state to set gathering of introspection data to
 *
 * @return
 *        - \ref DCGM_ST_OK                   if the call was successful
 *        - \ref DCGM_ST_BADPARAM             enabledState is an invalid state for metadata gathering
 *
 */
dcgmReturn_t DCGM_PUBLIC_API dcgmIntrospectToggleState(dcgmHandle_t pDcgmHandle, dcgmIntrospectState_t enabledState);

/*************************************************************************/
/**
 * Get the current amount of memory used to store the given field collection.
 *
 * @param pDcgmHandle                   IN: DCGM Handle
 * @param context                       IN: see \ref dcgmIntrospectContext_t.  This identifies the level of fields to do
 *                                          introspection for (ex: all fields, field groups) context->version must be
 *                                          set to dcgmIntrospectContext_version prior to this call.
 * @param memoryInfo                IN/OUT: see \ref dcgmIntrospectFullMemory_t. memoryInfo->version must be set to
 *                                          dcgmIntrospectFullMemory_version prior to this call.
 * @param waitIfNoData                  IN: if no metadata has been gathered, should this call block until data has been
 *                                          gathered (1), or should this call just return DCGM_ST_NO_DATA (0).
 * @return
 *        - \ref DCGM_ST_OK                   if the call was successful
 *        - \ref DCGM_ST_NOT_CONFIGURED       if metadata gathering state is \a DCGM_INTROSPECT_STATE_DISABLED
 *        - \ref DCGM_ST_NO_DATA              if \a waitIfNoData is false and metadata has not been gathered yet
 *        - \ref DCGM_ST_VER_MISMATCH         if context->version or memoryInfo->version is 0 or invalid.
 *
 */
dcgmReturn_t DCGM_PUBLIC_API dcgmIntrospectGetFieldsMemoryUsage(dcgmHandle_t pDcgmHandle,
                                                                dcgmIntrospectContext_t *context,
                                                                dcgmIntrospectFullMemory_t *memoryInfo,
                                                                int waitIfNoData);

/*************************************************************************/
/**
 * Get introspection info relating to execution time needed to update the fields
 * identified by \a context.
 *
 * @param pDcgmHandle        IN: DCGM Handle
 * @param context            IN: see \ref dcgmIntrospectContext_t.  This identifies the level of fields to do
 *                               introspection for (ex: all fields, field group ) context->version must be set to
 *                               dcgmIntrospectContext_version prior to this call.
 * @param execTime       IN/OUT: see \ref dcgmIntrospectFullFieldsExecTime_t. execTime->version must be set to
 *                               dcgmIntrospectFullFieldsExecTime_version prior to this call.
 * @param waitIfNoData       IN: if no metadata is gathered, wait until data has been gathered (1) or return
 *                               DCGM_ST_NO_DATA (0)
 * @return
 *        - \ref DCGM_ST_OK                   if the call was successful
 *        - \ref DCGM_ST_NOT_CONFIGURED       if metadata gathering state is \a DCGM_INTROSPECT_STATE_DISABLED
 *        - \ref DCGM_ST_NO_DATA              if \a waitIfNoData is false and metadata has not been gathered yet
 *        - \ref DCGM_ST_VER_MISMATCH         if context->version or execTime->version is 0 or invalid.
 *
 */
dcgmReturn_t DCGM_PUBLIC_API dcgmIntrospectGetFieldsExecTime(dcgmHandle_t pDcgmHandle,
                                                             dcgmIntrospectContext_t *context,
                                                             dcgmIntrospectFullFieldsExecTime_t *execTime,
                                                             int waitIfNoData);

/*************************************************************************/
/**
 * This method is used to manually tell the the introspection module to update
 * all DCGM introspection data. This is normally performed automatically on an
 * interval of 1 second.
 *
 * @param pDcgmHandle        IN: DCGM Handle
 * @param waitForUpdate      IN: Whether or not to wait for the update loop to complete before returning to the caller
 *
 * @return
 *        - \ref DCGM_ST_OK                   if the call was successful
 *        - \ref DCGM_ST_BADPARAM             if \a waitForUpdate is invalid
 *
 */
dcgmReturn_t DCGM_PUBLIC_API dcgmIntrospectUpdateAll(dcgmHandle_t pDcgmHandle, int waitForUpdate);

/*************************************************************************/
/**
 * Get the current amount of memory used to store the given field.
 *
 * @param pDcgmHandle                   IN: DCGM Handle
 * @param fieldId                       IN: DCGM field ID
 * @param pDcgmMetadataMemory          OUT: Total memory usage information for all field values in DCGM
 * @param waitIfNoData                  IN: if no metadata is gathered wait till this occurs (!0)
 *                                          or return DCGM_ST_NO_DATA (0)
 * @return
 *        - \ref DCGM_ST_OK                   if the call was successful
 *        - \ref DCGM_ST_NOT_CONFIGURED       if metadata gathering state is \ref DCGM_METADATA_STATE_DISABLED
 *        - \ref DCGM_ST_NO_DATA              if \a waitIfNoData is false and metadata has not been gathered yet
 */
dcgmReturn_t DCGM_PUBLIC_API dcgmIntrospectGetFieldMemoryUsage(dcgmHandle_t pDcgmHandle,
                                                               unsigned short fieldId,
                                                               dcgmIntrospectFullMemory_t *memoryInfo,
                                                               int waitIfNoData);

/*************************************************************************/
/**
 * Set the interval (in milliseconds) for when the metadata manager should
 * do its collection runs.
 *
 * @param pDcgmHandle             IN: DCGM Handle
 * @param runIntervalMs          OUT: interval duration to set
 */
dcgmReturn_t DCGM_PUBLIC_API dcgmMetadataStateSetRunInterval(dcgmHandle_t pDcgmHandle, unsigned int runIntervalMs);

/*************************************************************************/
/**
 * Get the total execution time since startup used to update a field in DCGM.
 *
 * @param pDcgmHandle                   IN: DCGM Handle
 * @param fieldId                       IN: field ID
 * @param execTime                     OUT: see \ref dcgmFieldExecTime_t
 * @param waitIfNoData                  IN: if no metadata is gathered wait till this occurs (!0)
 *                                          or return DCGM_ST_NO_DATA (0)
 * @return
 *        - \ref DCGM_ST_OK                   if the call was successful
 *        - \ref DCGM_ST_NOT_CONFIGURED       if metadata gathering state is \ref DCGM_METADATA_STATE_DISABLED
 *        - \ref DCGM_ST_NO_DATA              if \a waitIfNoData is false and metadata has not been gathered yet
 */
dcgmReturn_t DCGM_PUBLIC_API dcgmIntrospectGetFieldExecTime(dcgmHandle_t pDcgmHandle,
                                                            unsigned short fieldId,
                                                            dcgmIntrospectFullFieldsExecTime_t *execTime,
                                                            int waitIfNoData);


#ifdef __cplusplus
}
#endif
