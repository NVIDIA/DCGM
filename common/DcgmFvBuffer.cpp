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
#include "DcgmFvBuffer.h"
#include "DcgmLogging.h"

/******************************************************************************/
DcgmFvBuffer::DcgmFvBuffer(size_t initialCapacity, bool growExponentially)
{
    m_buffer            = 0;
    m_bufferCapacity    = 0;
    m_bufferUsed        = 0;
    m_numEntries        = 0;
    m_growExponentially = growExponentially;
    /* Allow this to be initialized without a capacity for efficient setting from a buffer */
    if (initialCapacity > 0)
        Resize(initialCapacity);
}

/******************************************************************************/
DcgmFvBuffer::~DcgmFvBuffer()
{
    if (m_buffer)
    {
        free(m_buffer);
        m_buffer = 0;
    }

    m_bufferUsed     = 0;
    m_bufferCapacity = 0;
    m_numEntries     = 0;
}

/******************************************************************************/
dcgmReturn_t DcgmFvBuffer::Resize(size_t newCapacity)
{
    if (!newCapacity)
        return DCGM_ST_BADPARAM;
    if (newCapacity <= m_bufferCapacity)
        return DCGM_ST_OK;

    char *tmp_buffer = (char *)realloc(m_buffer, newCapacity);
    if (!tmp_buffer)
    {
        log_error("Unable to resize buffer to {}", (int)newCapacity);
        m_bufferUsed     = 0;
        m_bufferCapacity = 0;
        m_numEntries     = 0;
        free(m_buffer);
        m_buffer = nullptr;
        return DCGM_ST_MEMORY;
    }
    m_buffer = tmp_buffer;
    memset(m_buffer + m_bufferCapacity, 0, newCapacity - m_bufferCapacity);

    m_bufferCapacity = newCapacity;
    return DCGM_ST_OK;
}

/******************************************************************************/
void DcgmFvBuffer::Clear(void)
{
    m_bufferUsed = 0;
    m_numEntries = 0;
}

/******************************************************************************/
dcgmBufferedFv_t *DcgmFvBuffer::AddFvReally(size_t bytesNeeded)
{
    dcgmReturn_t dcgmReturn;
    dcgmBufferedFv_t *retPtr;

    size_t spaceUsedAfter = m_bufferUsed + bytesNeeded;
    if (spaceUsedAfter > m_bufferCapacity)
    {
        size_t newBufferCapacity;

        // Find nearest value that is a product of 512
        newBufferCapacity = (spaceUsedAfter + 511U) & (~511U);
        if (m_growExponentially)
        {
            /* Using 1.6x to avoid heap fragmentation */
            newBufferCapacity = (size_t)(newBufferCapacity * 1.6);
        }

        dcgmReturn = Resize(newBufferCapacity);
        if (dcgmReturn != DCGM_ST_OK)
        {
            return nullptr;
        }
    }

    /* We have space for our new field. Save it */
    retPtr          = (dcgmBufferedFv_t *)&m_buffer[m_bufferUsed];
    retPtr->length  = bytesNeeded;
    retPtr->version = dcgmBufferedFv_version;
    m_bufferUsed    = spaceUsedAfter;
    m_numEntries++;
    return retPtr;
}

/******************************************************************************/
dcgmBufferedFv_t *DcgmFvBuffer::AddInt64Value(dcgm_field_entity_group_t entityGroupId,
                                              dcgm_field_eid_t entityId,
                                              unsigned short fieldId,
                                              long long value,
                                              long long timestamp,
                                              dcgmReturn_t status)
{
    dcgmBufferedFv_t *retPtr;
    size_t spaceNeeded = (sizeof(*retPtr) - sizeof(retPtr->value)) + sizeof(retPtr->value.i64);

    retPtr = AddFvReally(spaceNeeded);
    if (!retPtr)
    {
        return nullptr;
    }

    retPtr->fieldType     = DCGM_FT_INT64;
    retPtr->status        = status;
    retPtr->entityGroupId = entityGroupId;
    retPtr->entityId      = entityId;
    retPtr->fieldId       = fieldId;
    retPtr->timestamp     = timestamp;
    retPtr->value.i64     = value;
    return retPtr;
}

/******************************************************************************/
dcgmBufferedFv_t *DcgmFvBuffer::AddDoubleValue(dcgm_field_entity_group_t entityGroupId,
                                               dcgm_field_eid_t entityId,
                                               unsigned short fieldId,
                                               double value,
                                               long long timestamp,
                                               dcgmReturn_t status)
{
    dcgmBufferedFv_t *retPtr;
    size_t spaceNeeded = (sizeof(*retPtr) - sizeof(retPtr->value)) + sizeof(retPtr->value.dbl);

    retPtr = AddFvReally(spaceNeeded);
    if (!retPtr)
    {
        return nullptr;
    }

    retPtr->fieldType     = DCGM_FT_DOUBLE;
    retPtr->status        = status;
    retPtr->entityGroupId = entityGroupId;
    retPtr->entityId      = entityId;
    retPtr->fieldId       = fieldId;
    retPtr->timestamp     = timestamp;
    retPtr->value.dbl     = value;
    return retPtr;
}

/******************************************************************************/
dcgmBufferedFv_t *DcgmFvBuffer::AddStringValue(dcgm_field_entity_group_t entityGroupId,
                                               dcgm_field_eid_t entityId,
                                               unsigned short fieldId,
                                               const char *value,
                                               long long timestamp,
                                               dcgmReturn_t status)
{
    dcgmBufferedFv_t *retPtr;
    if (!value || !(*value))
    {
        log_error("Bad parameter");
        return 0;
    }

    size_t stringLength = strlen(value);

    /* There are +1s below because we're including the terminating null character */

    if (stringLength + 1 > sizeof(retPtr->value.str))
    {
        log_error("String {} is too big to buffer. (> {})", value, (int)sizeof(retPtr->value.str));
        return 0;
    }

    size_t spaceNeeded = (sizeof(*retPtr) - sizeof(retPtr->value)) + stringLength + 1;

    retPtr = AddFvReally(spaceNeeded);
    if (!retPtr)
    {
        return nullptr;
    }

    retPtr->fieldType     = DCGM_FT_STRING;
    retPtr->status        = status;
    retPtr->entityGroupId = entityGroupId;
    retPtr->entityId      = entityId;
    retPtr->fieldId       = fieldId;
    retPtr->timestamp     = timestamp;
    memmove(retPtr->value.str, value, stringLength + 1);
    return retPtr;
}

/******************************************************************************/
dcgmBufferedFv_t *DcgmFvBuffer::AddBlobValue(dcgm_field_entity_group_t entityGroupId,
                                             dcgm_field_eid_t entityId,
                                             unsigned short fieldId,
                                             void *value,
                                             size_t valueSize,
                                             long long timestamp,
                                             dcgmReturn_t status)
{
    dcgmBufferedFv_t *retPtr;

    if (!value || !valueSize)
    {
        log_error("Bad parameter");
        return 0;
    }
    if (valueSize > sizeof(retPtr->value.blob))
    {
        log_error("Blob is too big to buffer. (> {})", (int)sizeof(retPtr->value.blob));
        return 0;
    }

    size_t spaceNeeded = (sizeof(*retPtr) - sizeof(retPtr->value)) + valueSize;

    retPtr = AddFvReally(spaceNeeded);
    if (!retPtr)
    {
        return nullptr;
    }

    retPtr->fieldType     = DCGM_FT_BINARY;
    retPtr->status        = status;
    retPtr->entityGroupId = entityGroupId;
    retPtr->entityId      = entityId;
    retPtr->fieldId       = fieldId;
    retPtr->timestamp     = timestamp;
    memmove(retPtr->value.blob, value, valueSize);
    return retPtr;
}

/******************************************************************************/
dcgmBufferedFv_t *DcgmFvBuffer::AddBlankValue(dcgm_field_entity_group_t entityGroupId,
                                              dcgm_field_eid_t entityId,
                                              unsigned short fieldId,
                                              dcgmReturn_t status)
{
    dcgm_field_meta_p fieldMeta = DcgmFieldGetById(fieldId);
    if (fieldMeta == nullptr)
    {
        DCGM_LOG_ERROR << "Invalid fieldId " << fieldId;
        return nullptr;
    }

    switch (fieldMeta->fieldType)
    {
        case DCGM_FT_INT64:
            return AddInt64Value(entityGroupId, entityId, fieldId, DCGM_INT64_BLANK, 0, status);

        case DCGM_FT_DOUBLE:
            return AddDoubleValue(entityGroupId, entityId, fieldId, DCGM_FP64_BLANK, 0, status);

        case DCGM_FT_STRING:
            return AddStringValue(entityGroupId, entityId, fieldId, DCGM_STR_BLANK, 0, status);

        default:
            DCGM_LOG_ERROR << "Unhandled fieldType: " << fieldMeta->fieldType;
            return nullptr;
    }

    return nullptr;
}

/******************************************************************************/
dcgmBufferedFv_t *DcgmFvBuffer::AddFV1Value(dcgm_field_entity_group_t entityGroupId,
                                            dcgm_field_eid_t entityId,
                                            dcgmFieldValue_v1 *fv1)
{
    assert(fv1 != nullptr);

    switch (fv1->fieldType)
    {
        case DCGM_FT_INT64:
            return AddInt64Value(
                entityGroupId, entityId, fv1->fieldId, fv1->value.i64, fv1->ts, (dcgmReturn_t)fv1->status);

        case DCGM_FT_DOUBLE:
            return AddDoubleValue(
                entityGroupId, entityId, fv1->fieldId, fv1->value.dbl, fv1->ts, (dcgmReturn_t)fv1->status);

        case DCGM_FT_STRING:
            return AddStringValue(
                entityGroupId, entityId, fv1->fieldId, fv1->value.str, fv1->ts, (dcgmReturn_t)fv1->status);

        default:
            DCGM_LOG_ERROR << "Unhandled fieldType: " << fv1->fieldType;
            return nullptr;
    }

    return nullptr;
}

/******************************************************************************/
dcgmBufferedFv_t *DcgmFvBuffer::GetNextFv(dcgmBufferedFvCursor_t *cursor)
{
    dcgmBufferedFv_t *retPtr;

    if (!m_buffer || !m_bufferUsed)
        return 0;

    if ((*cursor) >= m_bufferUsed)
        return 0;

    retPtr = (dcgmBufferedFv_t *)&m_buffer[*cursor];

    /* Do some basic sanity on the FV */
    if (retPtr->version != dcgmBufferedFv_version)
    {
        log_error("Corrupt fv. version {} found.", (int)retPtr->version);
        return 0;
    }
    if (retPtr->length + (*cursor) > m_bufferUsed)
    {
        log_error("Corrupt fv length {} at {} / {}", retPtr->length, (int)(*cursor), (int)m_bufferUsed);
        return 0;
    }

    /* Advance the cursor */
    (*cursor) += retPtr->length;
    return retPtr;
}

/******************************************************************************/
dcgmReturn_t DcgmFvBuffer::SetFromBuffer(const char *buffer, size_t bufferSize)
{
    dcgmBufferedFv_t *fv;
    size_t bufferIndex;

    if (!buffer || !bufferSize)
        return DCGM_ST_BADPARAM;

    /* Make space for the new data */
    dcgmReturn_t dcgmReturn = Resize(bufferSize);
    if (dcgmReturn != DCGM_ST_OK)
        return dcgmReturn;

    /* Copy the data into place */
    memcpy(m_buffer, buffer, bufferSize);
    m_bufferUsed = bufferSize;
    m_numEntries = 0;

    /* Count entries and do basic sanity */
    for (bufferIndex = 0; bufferIndex < m_bufferUsed; bufferIndex += fv->length)
    {
        fv = (dcgmBufferedFv_t *)&m_buffer[bufferIndex];
        if (fv->version != dcgmBufferedFv_version)
        {
            log_error(
                "Corrupt fv. version {} found at {} / {}.", (int)fv->version, (int)bufferIndex, (int)m_bufferUsed);
            return DCGM_ST_GENERIC_ERROR;
        }
        if (fv->length + bufferIndex > m_bufferUsed)
        {
            log_error("Corrupt fv length {} at {} / {}", fv->length, (int)bufferIndex, (int)m_bufferUsed);
            return DCGM_ST_GENERIC_ERROR;
        }

        m_numEntries++;
    }

    return DCGM_ST_OK;
}

/******************************************************************************/
void DcgmFvBuffer::ConvertBufferedFvToFv1(dcgmBufferedFv_t *fv, dcgmFieldValue_v1 *fv1)
{
    if (!fv || !fv1)
        return;

    fv1->version   = dcgmFieldValue_version1;
    fv1->fieldId   = fv->fieldId;
    fv1->fieldType = fv->fieldType;
    fv1->status    = fv->status;
    fv1->ts        = fv->timestamp;
    switch (fv->fieldType)
    {
        case DCGM_FT_DOUBLE:
            fv1->value.dbl = fv->value.dbl;
            break;

        case DCGM_FT_INT64:
            fv1->value.i64 = fv->value.i64;
            break;

        case DCGM_FT_BINARY:
        case DCGM_FT_STRING:
        {
            /* Compute the string/binary size by subtracting off the rest of the fv */
            size_t dataSize = fv->length - (sizeof(*fv) - sizeof(fv->value));
            memmove(&fv1->value, &fv->value, dataSize);
            break;
        }

        default:
            log_error("Unhandled field type {}", fv->fieldType);
            break;
    }
}

/******************************************************************************/
void DcgmFvBuffer::ConvertBufferedFvToFv2(dcgmBufferedFv_t *fv, dcgmFieldValue_v2 *fv2)
{
    if (!fv || !fv2)
        return;

    fv2->version       = dcgmFieldValue_version2;
    fv2->entityGroupId = (dcgm_field_entity_group_t)fv->entityGroupId;
    fv2->entityId      = fv->entityId;
    fv2->fieldId       = fv->fieldId;
    fv2->fieldType     = fv->fieldType;
    fv2->unused        = 0;
    fv2->status        = fv->status;
    fv2->ts            = fv->timestamp;
    switch (fv->fieldType)
    {
        case DCGM_FT_DOUBLE:
            fv2->value.dbl = fv->value.dbl;
            break;

        case DCGM_FT_INT64:
            fv2->value.i64 = fv->value.i64;
            break;

        case DCGM_FT_BINARY:
        case DCGM_FT_STRING:
        {
            /* Compute the string/binary size by subtracting off the rest of the fv */
            size_t dataSize = fv->length - (sizeof(*fv) - sizeof(fv->value));
            memmove(&fv2->value, &fv->value, dataSize);
            break;
        }

        default:
            log_error("Unhandled field type {}", fv->fieldType);
            break;
    }
}

/******************************************************************************/
dcgmReturn_t DcgmFvBuffer::GetAllAsFv1(dcgmFieldValue_v1 *fv1, size_t fv1Capacity, size_t *numStored)
{
    dcgmBufferedFv_t *fv;

    if (!fv1 || !fv1Capacity)
        return DCGM_ST_BADPARAM;

    size_t numStoredTemp          = 0;
    dcgmBufferedFvCursor_t cursor = 0;

    for (fv = GetNextFv(&cursor); fv && numStoredTemp < fv1Capacity; fv = GetNextFv(&cursor))
    {
        ConvertBufferedFvToFv1(fv, &fv1[numStoredTemp]);
        numStoredTemp++;
    }

    /* Did we store exactly as many fields as were in this FV buffer? */
    if (fv1Capacity != numStoredTemp)
    {
        log_warning("GetAllAsFv1 capacity {}, stored {}", (int)fv1Capacity, (int)numStoredTemp);
        /* Not returning an error for now */
    }

    if (numStored)
        *numStored = numStoredTemp;

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmFvBuffer::GetSize(size_t *bufferSize, size_t *elementCount)
{
    if (!bufferSize && !elementCount)
        return DCGM_ST_BADPARAM;

    if (bufferSize)
        *bufferSize = m_bufferUsed;
    if (elementCount)
        *elementCount = m_numEntries;
    return DCGM_ST_OK;
}

/******************************************************************************/
void DcgmFvBuffer::SetGrowExponentially(bool growExponentially)
{
    m_growExponentially = growExponentially;
}

/******************************************************************************/
