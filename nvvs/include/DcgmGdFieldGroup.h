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

#include <string>
#include <vector>

#include "dcgm_structs.h"

class DcgmGdFieldGroup
{
public:
    DcgmGdFieldGroup();
    ~DcgmGdFieldGroup();

    /*
     * Get the field group id associated with this object
     */
    dcgmFieldGrp_t GetFieldGroupId();

    /*
     * Create a DCGM field group
     */
    dcgmReturn_t Init(dcgmHandle_t handle,
                      const std::vector<unsigned short> &fieldIds,
                      const std::string &fieldGroupName);

    dcgmReturn_t Cleanup();

private:
    dcgmFieldGrp_t m_fieldGroupId;
    dcgmHandle_t m_handle; // Not owned here
};
