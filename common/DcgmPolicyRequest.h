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
#ifndef DCGMPOLICYREQUEST_H
#define DCGMPOLICYREQUEST_H

#include "DcgmRequest.h"
#include "dcgm_structs.h"
#include <iostream>

class DcgmPolicyRequest : public DcgmRequest
{
public:
    DcgmPolicyRequest(fpRecvUpdates callback, uint64_t userData, std::mutex &cbMutex);
    virtual ~DcgmPolicyRequest();
    int ProcessMessage(std::unique_ptr<DcgmMessage> msg) override;

private:
    bool mIsAckRecvd;
    fpRecvUpdates mCallback;
    uint64_t mUserData;
    std::mutex &mCbMutex;
};

#endif /* DCGMPOLICYREQUEST_H */
