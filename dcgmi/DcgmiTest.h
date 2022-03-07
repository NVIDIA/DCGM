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
/*
 * DcgmTest.h
 *
 */
#pragma once

#include "Command.h"
#include "dcgm_structs_internal.h"

class DcgmiTest : public Command
{
public:
    DcgmiTest() = default;

    /* View information in cache on host specified by mDcgmHandle */
    dcgmReturn_t IntrospectCache(dcgmHandle_t mDcgmHandle,
                                 unsigned int gpuId,
                                 std::string const &fieldId,
                                 bool isGroup);

    /* Inject errors and information into the cache on host specified by mDcgmHandle */
    dcgmReturn_t InjectCache(dcgmHandle_t mDcgmHandle,
                             unsigned int gpuId,
                             std::string const &fieldId,
                             unsigned int pTime,
                             std::string &injectValue);

protected:
    dcgmReturn_t DoExecuteConnected() override;

private:
    /* Helper function to display field info to stdout */
    void HelperDisplayField(dcgmCacheManagerFieldInfo_t &fieldInfo);

    /* Helper function to initialize and populate the needed data in the field value */
    dcgmReturn_t HelperInitFieldValue(dcgmInjectFieldValue_t &injectFieldValue, std::string &injectValue);

    /* Helper function to parse user input into field id  */
    dcgmReturn_t HelperParseForFieldId(std::string const &str, unsigned short &fieldId, dcgmHandle_t mDcgmHandle);

    /* Helper function to format timestamp into human readable format */
    std::string HelperFormatTimestamp(long long timestamp);
};


/*****************************************************************************
 * Define classes to extend commands
 ****************************************************************************/

/**
 * Introspect Cache Invoker
 */
class IntrospectCache : public Command
{
public:
    IntrospectCache(std::string hostname, unsigned int gpuId, std::string fieldId, bool isGroup);

protected:
    dcgmReturn_t DoExecuteConnected() override;

private:
    DcgmiTest adminObj;
    unsigned int mGpuId;
    std::string mFieldId;
    bool mIDisGroup;
};

/**
 * Inject Cache Invoker
 */
class InjectCache : public Command
{
public:
    InjectCache(std::string hostname,
                unsigned int gId,
                std::string fieldId,
                unsigned int pTime,
                std::string injectValue);

protected:
    dcgmReturn_t DoExecuteConnected() override;

private:
    DcgmiTest adminObj;
    unsigned int mGId;
    std::string mFieldId;
    unsigned int mTime;
    std::string mInjectValue;
};
