/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

#include <map>
#include <nvml.h>
#include <unordered_map>

#include "AttributeQueryInfo.h"
#include "InjectionArgument.h"

#define MAX_KEY_LENGTH 128

class FieldHelpers
{
public:
    FieldHelpers();

    void Initialize();

    /**
     * Retrieves the field value associated with the key.
     *
     * @param aqi - The query information we are trying to associate with a field ID
     * @return    - The field ID associated with the query information
     *              0 if there is no field ID associated with the key
     */
    unsigned int GetFieldId(const AttributeQueryInfo &aqi);

    /**
     * Retrieves the field value associated with the key.
     *
     * @param errorType   - the type of error (single or double precision)
     * @param counterType - the type of ecc counter
     * @param memLocation - the memory location
     * @return    - The field ID associated with the key.
     *              0 if there is no field ID associated with the key
     */
    unsigned int GetFieldId(nvmlMemoryErrorType_t errorType,
                            nvmlEccCounterType_t counterType,
                            nvmlMemoryLocation_t memLocation);

    /**
     * Returns the NVML injection library key associated with the DCGM field ID
     *
     * @param  fieldId   - The DCGM field ID we are translating to an NVML string
     * @return           - The attribute query info, or an empty attribute query info if nothing is found.
     */
    AttributeQueryInfo GetKeyFromDcgmFieldId(unsigned short fieldId);

    /**
     * Returns the NVML injection library key associated with the NVML field ID
     *
     * @param  fieldId   - The NVML field ID we are translating to an NVML string
     * @return           - The attribute query info, or an empty attribute query info if nothing is found.
     */
    AttributeQueryInfo GetKeyFromFieldId(unsigned int fieldId);

    double GetDouble(nvmlValue_t &fieldValue);
    unsigned int GetUInt(nvmlValue_t &fieldValue);
    unsigned long GetULong(nvmlValue_t &fieldValue);
    unsigned long long GetULongLong(nvmlValue_t &fieldValue);
    long long GetLongLong(nvmlValue_t &fieldValue);

private:
    std::unordered_map<unsigned int, AttributeQueryInfo> m_fieldIdToQueryInfo;
    std::map<AttributeQueryInfo, unsigned int> m_queryInfoToFieldId;

    std::unordered_map<unsigned int, AttributeQueryInfo> m_dcgmFieldIdToQueryInfo;
    std::map<AttributeQueryInfo, unsigned int> m_queryInfoToDcgmFieldId;
    // Add 1 more memory location to distinguish DRAM and DEVICE_MEMORY
    unsigned int m_eccFieldIds[NVML_MEMORY_ERROR_TYPE_COUNT][NVML_ECC_COUNTER_TYPE_COUNT][NVML_MEMORY_LOCATION_COUNT];

    void InitializeEccFieldMap();
};
