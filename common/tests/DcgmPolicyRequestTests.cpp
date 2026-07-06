/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
#include <DcgmPolicyRequest.h>
#include <DcgmProtocol.h>

#include <catch2/catch_all.hpp>
#include <dcgm_structs.h>

TEST_CASE("DcgmPolicyRequest::ProcessMessage rejects truncated policy notify")
{
    DcgmPolicyRequest request(nullptr, 0);

    auto msg = std::make_unique<DcgmMessage>();
    msg->UpdateMsgHdr(DCGM_MSG_POLICY_NOTIFY, 1, DCGM_ST_OK, 0);

    // Simulate a truncated message by making the buffer one byte too small
    auto msgBytes = msg->GetMsgBytesPtr();
    msgBytes->resize(sizeof(dcgm_msg_policy_notify_t) - 1);

    int ret = request.ProcessMessage(std::move(msg));
    REQUIRE(ret == DCGM_ST_BADPARAM);
}
