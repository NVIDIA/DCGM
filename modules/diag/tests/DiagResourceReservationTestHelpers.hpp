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

#ifndef DIAG_RESOURCE_RESERVATION_TEST_HELPERS_HPP
#define DIAG_RESOURCE_RESERVATION_TEST_HELPERS_HPP

#include <dcgm_core_structs.h>

namespace DcgmNs::Tests
{

struct ResourceReservationContext
{
    dcgmReturn_t reserveReturn = DCGM_ST_IN_USE;
    unsigned int reserveCalls  = 0;
    unsigned int freeCalls     = 0;
};

inline dcgmCoreCallbacks_t MakeResourceReservationCallbacks(ResourceReservationContext &context)
{
    return { .version  = dcgmCoreCallbacks_version,
             .postfunc = [](dcgm_module_command_header_t *header, void *poster) -> dcgmReturn_t {
                 auto *context = static_cast<ResourceReservationContext *>(poster);
                 if (header == nullptr || context == nullptr)
                 {
                     return DCGM_ST_BADPARAM;
                 }

                 if (header->subCommand == DcgmCoreReqIdResourceReserve)
                 {
                     auto *msg = reinterpret_cast<dcgmCoreResourceReserve_t *>(header);
                     context->reserveCalls++;
                     msg->response.ret   = context->reserveReturn;
                     msg->response.token = 4242;
                     return DCGM_ST_OK;
                 }

                 if (header->subCommand == DcgmCoreReqIdResourceFree)
                 {
                     auto *msg = reinterpret_cast<dcgmCoreResourceFree_t *>(header);
                     context->freeCalls++;
                     msg->ret = DCGM_ST_OK;
                     return DCGM_ST_OK;
                 }

                 return DCGM_ST_NOT_SUPPORTED;
             },
             .poster     = &context,
             .loggerfunc = [](void const *) { /* do nothing */ } };
}

} // namespace DcgmNs::Tests

#endif // DIAG_RESOURCE_RESERVATION_TEST_HELPERS_HPP
