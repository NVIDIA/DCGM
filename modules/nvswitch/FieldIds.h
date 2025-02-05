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

#include <dcgm_structs.h>

#include "DcgmNscqManager.h"

namespace DcgmNs
{
/**
 * UpdateFuncType describes the signature of a function to be called to return a
 * DcgmFvBuffer for the storageType template.
 *
 * Constexpr pointers to functions with this signature are used for each
 * FieldIdControlType template instantiation on a FieldId. But, different
 * fields may require different update functions for data specific to
 * NvSwitches, NvLinks, Links, Lanes, and VCs, whether scalar or vector,
 * and these are assigned in the fully specialized FieldIdInteralType
 * definition.
 */
using UpdateFuncType = dcgmReturn_t (DcgmNscqManager::*)(unsigned short fieldId,
                                                         DcgmFvBuffer &buf,
                                                         const std::vector<dcgm_field_update_info_t> &entities,
                                                         timelib64_t now);

/**
 * We template the data storage type for holding retrieved data from the NSCQ
 * callback on the storage type for an individial item along with any
 * anclilliary entity indexing information (the taggedType). taggedType must
 * provide a storageType "type" definition for the actual storage type.
 *
 * Construction: NSCQ path and field ID parameters are included so they can be
 * logged in the lambda callback.
 */
template <typename taggedType>
struct NscqDataCollector
{
    unsigned int callCounter = 0;
    unsigned short fieldId;
    const char *nscqPath;
    std::vector<taggedType> data;

    NscqDataCollector(unsigned short fieldId, const char *nscqPath)
        : fieldId(fieldId)
        , nscqPath(nscqPath)
    {}
};

/**
 * This affords a mapping from a FieldID to the type of the internal data
 * type of the object used to hold the selected returned value from an NSCQ
 * callback (dcgmFieldType) and the type of data returned by the NCSQ callback
 * (nscqFieldType)  itself.
 *
 * The NSCQ callback may return a structure, but depending on the FieldID,
 * we may only desire a particular member of the struture. *That*
 * selection is made via the FieldIdStorageType template.
 *
 * This is intended to be fully specialized for each FieldId, with specification
 * of the internal data type member dcgmFieldType, and the NSCQ callback data type
 * member nscqFieldType. Also an UpdateFuncTyp UpdateFunc(void) is expected to return
 * update function. Finally a const char* NscqPath(void) is expected to return
 * the nscqPath to retrieve the data necessary for this fieldId. These are
 * polymorphic functions in classes to be derived from
 * FieldIdControlType<DCGM_FI_UNKNOWN>.
 *
 * Each dcgmFieldType is expected to have a
 *
 *   void BufferAdd(dcgm_field_entity_group_t entityGroupId,
 *                  dcgm_field_eid_t entityId,
 *                  unsigned short fieldId,
 *                  timelib64_t now,
 *                  DcgmFvBuffer &buf) const
 *
 * function to add itself to the DcgmFvBuffer reference passed and a
 *
 * an std::string Str(void) function to return the item as a string (ostensibly
 * for logging).
 *
 * In addition, it should have a default constructor to initialize it to the
 * error value of it's type (generally DCGM_INT64_BLANK or DCGM_FP64_BLANK).
 */
template <unsigned short fieldId>
class FieldIdControlType
{
public:
    using dcgmFieldType = void;
    using nscqFieldType = void;

    const char *NscqPath(void) const
    {
        return nullptr;
    }

    UpdateFuncType UpdateFunc(void) const
    {
        return nullptr;
    }

    FieldIdControlType()  = delete;
    ~FieldIdControlType() = delete;
};

/**
 * Here, we define a base class for all fully specialized FieldIdControlType
 * classes (one for each Field Id). This provides for virtual NscqPath and
 * UpdateFunc functions to return the NSCQ path to query the field data and
 * a pointer to a DcgmNscqManager member function to Find NSCQ
 * callback indicies among supplied entities.
 */
template <>
class FieldIdControlType<DCGM_FI_UNKNOWN>
{
public:
    using dcgmFieldType = void;
    using nscqFieldType = void;

    virtual const char *NscqPath(void) const      = 0;
    virtual UpdateFuncType UpdateFunc(void) const = 0;

    FieldIdControlType()  = default;
    ~FieldIdControlType() = default;
};

/**
 * This affords storage for data specific to
 * FieldIdControlType<fieldId>::dcgmFieldType.
 *
 * The intent is to define constructors taking different different types
 * of data and picking out the fieldID specific bit or storing a compound
 * structured type.
 *
 * So, depending on fieldId, we may select either a single member of
 * the provided FieldIdControlType<fieldId>::nscqFieldType (when a struct is
 * returned of which the fieldId selects a single member), or multiple members,
 * usually when a vector of items is returned from NSCQ with timestamps on each
 * item. (FieldIdControlType<fieldId>::nscqFieldType still names the vector
 * element and not the vector itself.
 *
 * By default, the FieldIdControlType<fieldId>::nscqFieldType is just passed as
 * a construction argument to the FieldIdControlType<fieldId>::dcgmFieldType
 * base class and it is expected that implicit conversion just "works".
 */
template <unsigned short fieldId>
class FieldIdStorageType : public FieldIdControlType<fieldId>::dcgmFieldType
{
public:
    FieldIdStorageType(void)
        : FieldIdControlType<fieldId>::dcgmFieldType()
    {}

    FieldIdStorageType(const FieldIdControlType<fieldId>::nscqFieldType &data)
        : FieldIdControlType<fieldId>::dcgmFieldType(data)
    {}

    ~FieldIdStorageType() = default;
};

/**
 * Here we define a selective vector/non-vector NSCQ callback class with
 * the appropriate function to inject the called back item or vector items
 * into a collection vector. NSCQ callback data is collected, along with
 * indicies, in these objects.
 *
 * Basically, we capture the called back data as well as the indicies provided.
 */
template <typename nscqFieldType, typename storageType, bool is_vector, typename... indexTypes>
class TempData;

/**
 * This is the non-vector case of the NSCQ callback data and index object.
 */
template <typename nscqFieldType, typename storageType, typename... indexTypes>
class TempData<nscqFieldType, storageType, false, indexTypes...>
{
public:
    std::tuple<indexTypes...> index;
    storageType data;

    using cbType = const nscqFieldType;

    void CollectFunc(NscqDataCollector<TempData> *dest, indexTypes... indicies)
    {
        index = std::tuple<indexTypes...>(indicies...);
        data  = storageType();
        dest->data.push_back(*this);
    }

    void CollectFunc(NscqDataCollector<TempData> *dest, const nscqFieldType in, indexTypes... indicies)
    {
        index = std::tuple<indexTypes...>(indicies...);
        data  = storageType(in);
        dest->data.push_back(*this);
    }
};

/**
 * This is the vector case of the NSCQ callback data and index object.
 */
template <typename nscqFieldType, typename storageType, typename... indexTypes>
class TempData<nscqFieldType, storageType, true, indexTypes...>
{
public:
    std::tuple<indexTypes...> index;
    storageType data;

    using cbType = const std::vector<nscqFieldType>;

    void CollectFunc(NscqDataCollector<TempData> *dest, indexTypes... indicies)
    {
        index = std::tuple<indexTypes...>(indicies...);
        data  = storageType();

        dest->data.push_back(*this);
    }

    void CollectFunc(NscqDataCollector<TempData> *dest, const std::vector<nscqFieldType> in, indexTypes... indicies)
    {
        index = std::tuple<indexTypes...>(indicies...);

        for (auto item : in)
        {
            data = storageType(item);
            dest->data.push_back(*this);
        }
    }
};

/**
 * Map fieldId to FieldIdControlType<fieldId> singleton reference.
 */
const FieldIdControlType<DCGM_FI_UNKNOWN> *FieldIdFind(unsigned short fieldId);

} // namespace DcgmNs