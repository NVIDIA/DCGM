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

#include <DcgmFvBuffer.h>
#include <DcgmWatchTable.h>
#include <dcgm_structs.h>

#include "DcgmNscqManager.h"
#include "FieldIds.h"

namespace DcgmNs
{
/**
 * Here, we wrap the actual update functions in convenient templated classes.
 * The appropriate one is used to statically construct a constexpr pointer for
 * each fully specialized FieldIdControlType. The various specialized
 * FieldIdControlType UpdateFunc methods are coded to return the updateFunc
 * member of the appropriate class.
 */

using phys_id_t      = uint32_t;
using uuid_p         = nscq_uuid_t *;
using label_t        = nscq_label_t;
using link_id_t      = uint32_t;
using lane_vc_id_t   = uint32_t;
using nvlink_state_t = nscq_nvlink_state_t;

/**
 * This handles Update functions for switches with scalar returned data.
 */
template <unsigned short fieldId>
class UpdateNvSwitchScalarFieldType
{
public:
    static constexpr UpdateFuncType updateFunc
        = &DcgmNscqManager::UpdateFields<typename FieldIdControlType<fieldId>::nscqFieldType,
                                         FieldIdStorageType<fieldId>,
                                         false,
                                         uuid_p>;
};

/**
 * This handles Update functions for switches with vactor returned data.
 */
template <unsigned short fieldId>
class UpdateNvSwitchVectorFieldType
{
public:
    static constexpr UpdateFuncType updateFunc
        = &DcgmNscqManager::UpdateFields<typename FieldIdControlType<fieldId>::nscqFieldType,
                                         FieldIdStorageType<fieldId>,
                                         true,
                                         uuid_p>;
};

/**
 * This handles Update functions for switches and nvlinks with scalar returned
 * data.
 */
template <unsigned short fieldId>
class UpdateLinkScalarFieldType
{
public:
    static constexpr UpdateFuncType updateFunc
        = &DcgmNscqManager::UpdateFields<typename FieldIdControlType<fieldId>::nscqFieldType,
                                         FieldIdStorageType<fieldId>,
                                         false,
                                         uuid_p,
                                         link_id_t>;
};

/**
 * This handles Update functions for switches and nvlinks with vector returned
 * data.
 */
template <unsigned short fieldId>
class UpdateLinkVectorFieldType
{
public:
    static constexpr UpdateFuncType updateFunc
        = &DcgmNscqManager::UpdateFields<typename FieldIdControlType<fieldId>::nscqFieldType,
                                         FieldIdStorageType<fieldId>,
                                         true,
                                         uuid_p,
                                         link_id_t>;
};

/**
 * This handles Update functions for switches, nvlinks, and lanes (or virtual
 * circuits)  with scalar returned data.
 */
template <unsigned short fieldId>
class UpdateLaneScalarFieldType
{
public:
    static constexpr UpdateFuncType updateFunc
        = &DcgmNscqManager::UpdateFields<typename FieldIdControlType<fieldId>::nscqFieldType,
                                         FieldIdStorageType<fieldId>,
                                         false,
                                         uuid_p,
                                         link_id_t,
                                         lane_vc_id_t>;
};

} // namespace DcgmNs
