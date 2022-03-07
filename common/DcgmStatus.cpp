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
 * File:   DcgmStatus.cpp
 */

#include "DcgmStatus.h"

DcgmStatus::DcgmStatus()
    : mLock()
{}

DcgmStatus::~DcgmStatus()
{
    RemoveAll();
}

/*****************************************************************************/
int DcgmStatus::Lock()
{
    mLock.lock();
    return DCGM_ST_OK;
}

/*****************************************************************************/
int DcgmStatus::UnLock()
{
    mLock.unlock();
    return DCGM_ST_OK;
}

/*****************************************************************************/
int DcgmStatus::Enqueue(unsigned int gpuId, short fieldId, int errorCode)
{
    Lock();
    dcgmErrorInfo_t st;
    st.gpuId   = gpuId;
    st.fieldId = fieldId;
    st.status  = errorCode;
    mStatusList.push_back(st);
    UnLock();
    return 0;
}

/*****************************************************************************/
int DcgmStatus::Dequeue(dcgmErrorInfo_t *pDcgmStatus)
{
    Lock();

    if (NULL == pDcgmStatus)
    {
        UnLock();
        return -1;
    }

    if (mStatusList.empty())
    {
        UnLock();
        return -1;
    }

    *pDcgmStatus = mStatusList.front();
    mStatusList.pop_front();

    UnLock();
    return 0;
}

/*****************************************************************************/
int DcgmStatus::RemoveAll()
{
    Lock();
    mStatusList.clear();
    UnLock();

    return 0;
}

/*****************************************************************************/
bool DcgmStatus::IsEmpty()
{
    Lock();

    if (mStatusList.size())
    {
        UnLock();
        return false;
    }

    UnLock();
    return true;
}

/*****************************************************************************/
unsigned int DcgmStatus::GetNumErrors()
{
    unsigned int size;

    Lock();
    size = mStatusList.size();
    UnLock();

    return size;
}
