/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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
    DcgmPolicyRequest(fpRecvUpdates callback, uint64_t userData);
    virtual ~DcgmPolicyRequest();

    /**
     * Processes an incoming DCGM message for policy requests.
     *
     * @param[in] msg The message to process
     * @return Status code indicating success (0) or error (non-zero)
     */
    [[nodiscard]] int ProcessMessage(std::unique_ptr<DcgmMessage> msg) override;

private:
    bool mIsAckRecvd;
    fpRecvUpdates mCallback;
    uint64_t mUserData;
};

/**
 * Handles field-policy violation notifications for a registered client callback.
 */
class DcgmPolicyViolationRequest : public DcgmRequest
{
public:
    /**
     * Create a violation request handler.
     *
     * @param[in] callback Callback invoked for matching policy violations.
     * @param[in] userData Opaque user data passed back to callback.
     */
    DcgmPolicyViolationRequest(fpRecvPolicyViolation callback, uint64_t userData);

    /**
     * Destroy the violation request handler.
     */
    virtual ~DcgmPolicyViolationRequest();

    /**
     * Process one incoming message for this request.
     *
     * @param[in] msg Incoming DCGM message.
     * @return DCGM_ST_OK on success, or a DCGM_ST_* error code.
     */
    [[nodiscard]] int ProcessMessage(std::unique_ptr<DcgmMessage> msg) override;

private:
    bool mIsAckRecvd;
    fpRecvPolicyViolation mCallback;
    uint64_t mUserData;
};

#endif /* DCGMPOLICYREQUEST_H */
