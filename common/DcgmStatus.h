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
 * File:   DcgmStatus.h
 */

#ifndef DCGMSTATUS_H
#define DCGMSTATUS_H

#include "dcgm_structs.h"
#include <iostream>
#include <list>
#include <mutex>

class DcgmStatus
{
public:
    DcgmStatus();

    virtual ~DcgmStatus();

    /*****************************************************************************
     * This method checks if the status list is empty or not
     *****************************************************************************/
    bool IsEmpty();

    /*****************************************************************************
     * This method is used to get the number of errors in the status list
     *****************************************************************************/
    unsigned int GetNumErrors();

    /*****************************************************************************
     * Add status to the list
     * @param gpuId         IN  : Represents GPU ID
     * @param fieldId       IN  : Field ID corresponding to which error is reported
     * @param errorCode     IN  : Error code corresponding to the GPU ID and Field ID
     * @return
     *****************************************************************************/
    int Enqueue(unsigned int gpuId, short fieldId, int errorCode);

    /*****************************************************************************
     * Removes status from the list
     * @param pDcgmStatus   OUT :   Status to be returned to the caller
     * @return
     *****************************************************************************/
    int Dequeue(dcgmErrorInfo_t *pDcgmStatus);

    /*****************************************************************************
     * Clears all the status from the list
     *****************************************************************************/
    int RemoveAll();

private:
    /*****************************************************************************
     * Lock/Unlock methods
     *****************************************************************************/
    int Lock();
    int UnLock();

    std::mutex mLock; /* Lock used for accessing status list */
    std::list<dcgmErrorInfo_t> mStatusList;
};

#endif /* DCGMSTATUS_H */
