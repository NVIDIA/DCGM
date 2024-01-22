/*
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
#include <FieldHelpers.h>
#include <InjectionKeys.h>

unsigned int FieldHelpers::GetFieldId(const AttributeQueryInfo &aqi)
{
    return m_queryInfoToFieldId[aqi];
}

unsigned int FieldHelpers::GetFieldId(nvmlMemoryErrorType_t errorType,
                                      nvmlEccCounterType_t counterType,
                                      nvmlMemoryLocation_t memLocation)
{
    if (errorType >= NVML_MEMORY_ERROR_TYPE_COUNT || counterType >= NVML_ECC_COUNTER_TYPE_COUNT
        || memLocation >= NVML_MEMORY_LOCATION_COUNT)
    {
        return 0;
    }

    return m_eccFieldIds[errorType][counterType][memLocation];
}

/*AttributeQueryInfo FieldHelpers::GetKeyFromDcgmFieldId(unsigned short fieldId)
{
}*/

AttributeQueryInfo FieldHelpers::GetKeyFromFieldId(unsigned int fieldId)
{
    return m_fieldIdToQueryInfo[fieldId];
}

double FieldHelpers::GetDouble(nvmlValue_t &fieldValue)
{
    return fieldValue.dVal;
}

unsigned int FieldHelpers::GetUInt(nvmlValue_t &fieldValue)
{
    return fieldValue.uiVal;
}

unsigned long FieldHelpers::GetULong(nvmlValue_t &fieldValue)
{
    return fieldValue.ulVal;
}

unsigned long long FieldHelpers::GetULongLong(nvmlValue_t &fieldValue)
{
    return fieldValue.ullVal;
}

long long FieldHelpers::GetLongLong(nvmlValue_t &fieldValue)
{
    return fieldValue.sllVal;
}

FieldHelpers::FieldHelpers()
    : m_fieldIdToQueryInfo()
    , m_queryInfoToFieldId()
{
    Initialize();
}

void FieldHelpers::Initialize()
{
    m_fieldIdToQueryInfo[NVML_FI_DEV_ECC_CURRENT] = AttributeQueryInfo(INJECTION_ECCMODE_KEY, 0);
    m_fieldIdToQueryInfo[NVML_FI_DEV_ECC_PENDING] = AttributeQueryInfo(INJECTION_ECCMODE_KEY, 1);

    InitializeEccFieldMap();
}

void FieldHelpers::InitializeEccFieldMap()
{
    m_eccFieldIds[NVML_MEMORY_ERROR_TYPE_CORRECTED][NVML_VOLATILE_ECC][NVML_MEMORY_LOCATION_L1_CACHE]
        = NVML_FI_DEV_ECC_SBE_VOL_L1;
    m_eccFieldIds[NVML_MEMORY_ERROR_TYPE_CORRECTED][NVML_VOLATILE_ECC][NVML_MEMORY_LOCATION_L2_CACHE]
        = NVML_FI_DEV_ECC_SBE_VOL_L2;
    m_eccFieldIds[NVML_MEMORY_ERROR_TYPE_CORRECTED][NVML_VOLATILE_ECC][NVML_MEMORY_LOCATION_DEVICE_MEMORY]
        = NVML_FI_DEV_ECC_SBE_VOL_DEV;
    m_eccFieldIds[NVML_MEMORY_ERROR_TYPE_CORRECTED][NVML_VOLATILE_ECC][NVML_MEMORY_LOCATION_REGISTER_FILE]
        = NVML_FI_DEV_ECC_SBE_VOL_REG;
    m_eccFieldIds[NVML_MEMORY_ERROR_TYPE_CORRECTED][NVML_VOLATILE_ECC][NVML_MEMORY_LOCATION_TEXTURE_MEMORY]
        = NVML_FI_DEV_ECC_SBE_VOL_TEX;
    m_eccFieldIds[NVML_MEMORY_ERROR_TYPE_CORRECTED][NVML_VOLATILE_ECC][NVML_MEMORY_LOCATION_TEXTURE_SHM] = 0;
    m_eccFieldIds[NVML_MEMORY_ERROR_TYPE_CORRECTED][NVML_VOLATILE_ECC][NVML_MEMORY_LOCATION_CBU]         = 0;
    m_eccFieldIds[NVML_MEMORY_ERROR_TYPE_CORRECTED][NVML_VOLATILE_ECC][NVML_MEMORY_LOCATION_SRAM]        = 0;

    m_eccFieldIds[NVML_MEMORY_ERROR_TYPE_CORRECTED][NVML_AGGREGATE_ECC][NVML_MEMORY_LOCATION_L1_CACHE]
        = NVML_FI_DEV_ECC_SBE_AGG_L1;
    m_eccFieldIds[NVML_MEMORY_ERROR_TYPE_CORRECTED][NVML_AGGREGATE_ECC][NVML_MEMORY_LOCATION_L2_CACHE]
        = NVML_FI_DEV_ECC_SBE_AGG_L2;
    m_eccFieldIds[NVML_MEMORY_ERROR_TYPE_CORRECTED][NVML_AGGREGATE_ECC][NVML_MEMORY_LOCATION_DEVICE_MEMORY]
        = NVML_FI_DEV_ECC_SBE_AGG_DEV;
    m_eccFieldIds[NVML_MEMORY_ERROR_TYPE_CORRECTED][NVML_AGGREGATE_ECC][NVML_MEMORY_LOCATION_REGISTER_FILE]
        = NVML_FI_DEV_ECC_SBE_AGG_REG;
    m_eccFieldIds[NVML_MEMORY_ERROR_TYPE_CORRECTED][NVML_AGGREGATE_ECC][NVML_MEMORY_LOCATION_TEXTURE_MEMORY]
        = NVML_FI_DEV_ECC_SBE_AGG_TEX;
    m_eccFieldIds[NVML_MEMORY_ERROR_TYPE_CORRECTED][NVML_AGGREGATE_ECC][NVML_MEMORY_LOCATION_TEXTURE_SHM] = 0;
    m_eccFieldIds[NVML_MEMORY_ERROR_TYPE_CORRECTED][NVML_AGGREGATE_ECC][NVML_MEMORY_LOCATION_CBU]         = 0;
    m_eccFieldIds[NVML_MEMORY_ERROR_TYPE_CORRECTED][NVML_AGGREGATE_ECC][NVML_MEMORY_LOCATION_SRAM]        = 0;

    m_eccFieldIds[NVML_MEMORY_ERROR_TYPE_UNCORRECTED][NVML_VOLATILE_ECC][NVML_MEMORY_LOCATION_L1_CACHE]
        = NVML_FI_DEV_ECC_DBE_VOL_L1;
    m_eccFieldIds[NVML_MEMORY_ERROR_TYPE_UNCORRECTED][NVML_VOLATILE_ECC][NVML_MEMORY_LOCATION_L2_CACHE]
        = NVML_FI_DEV_ECC_DBE_VOL_L2;
    m_eccFieldIds[NVML_MEMORY_ERROR_TYPE_UNCORRECTED][NVML_VOLATILE_ECC][NVML_MEMORY_LOCATION_DEVICE_MEMORY]
        = NVML_FI_DEV_ECC_DBE_VOL_DEV;
    m_eccFieldIds[NVML_MEMORY_ERROR_TYPE_UNCORRECTED][NVML_VOLATILE_ECC][NVML_MEMORY_LOCATION_REGISTER_FILE]
        = NVML_FI_DEV_ECC_DBE_VOL_REG;
    m_eccFieldIds[NVML_MEMORY_ERROR_TYPE_UNCORRECTED][NVML_VOLATILE_ECC][NVML_MEMORY_LOCATION_TEXTURE_MEMORY]
        = NVML_FI_DEV_ECC_DBE_VOL_TEX;
    m_eccFieldIds[NVML_MEMORY_ERROR_TYPE_UNCORRECTED][NVML_VOLATILE_ECC][NVML_MEMORY_LOCATION_TEXTURE_SHM] = 0;
    m_eccFieldIds[NVML_MEMORY_ERROR_TYPE_UNCORRECTED][NVML_VOLATILE_ECC][NVML_MEMORY_LOCATION_CBU]
        = NVML_FI_DEV_ECC_DBE_VOL_CBU;
    m_eccFieldIds[NVML_MEMORY_ERROR_TYPE_UNCORRECTED][NVML_VOLATILE_ECC][NVML_MEMORY_LOCATION_SRAM] = 0;

    m_eccFieldIds[NVML_MEMORY_ERROR_TYPE_UNCORRECTED][NVML_AGGREGATE_ECC][NVML_MEMORY_LOCATION_L1_CACHE]
        = NVML_FI_DEV_ECC_DBE_AGG_L1;
    m_eccFieldIds[NVML_MEMORY_ERROR_TYPE_UNCORRECTED][NVML_AGGREGATE_ECC][NVML_MEMORY_LOCATION_L2_CACHE]
        = NVML_FI_DEV_ECC_DBE_AGG_L2;
    m_eccFieldIds[NVML_MEMORY_ERROR_TYPE_UNCORRECTED][NVML_AGGREGATE_ECC][NVML_MEMORY_LOCATION_DEVICE_MEMORY]
        = NVML_FI_DEV_ECC_DBE_AGG_DEV;
    m_eccFieldIds[NVML_MEMORY_ERROR_TYPE_UNCORRECTED][NVML_AGGREGATE_ECC][NVML_MEMORY_LOCATION_REGISTER_FILE]
        = NVML_FI_DEV_ECC_DBE_AGG_REG;
    m_eccFieldIds[NVML_MEMORY_ERROR_TYPE_UNCORRECTED][NVML_AGGREGATE_ECC][NVML_MEMORY_LOCATION_TEXTURE_MEMORY]
        = NVML_FI_DEV_ECC_DBE_AGG_TEX;
    m_eccFieldIds[NVML_MEMORY_ERROR_TYPE_UNCORRECTED][NVML_AGGREGATE_ECC][NVML_MEMORY_LOCATION_TEXTURE_SHM] = 0;
    m_eccFieldIds[NVML_MEMORY_ERROR_TYPE_UNCORRECTED][NVML_AGGREGATE_ECC][NVML_MEMORY_LOCATION_CBU]
        = NVML_FI_DEV_ECC_DBE_AGG_CBU;
    m_eccFieldIds[NVML_MEMORY_ERROR_TYPE_UNCORRECTED][NVML_AGGREGATE_ECC][NVML_MEMORY_LOCATION_SRAM] = 0;
}
