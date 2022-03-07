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
#ifndef DCGM_MODULE_STRUCTS_H
#define DCGM_MODULE_STRUCTS_H

#include <dcgm_structs_internal.h>

/* Definitions for DCGM modules */

// Use this structure for requests and responses

/* Base structure for all module commands. Put this structure at the front of your messages */
typedef struct
{
    unsigned int length; /* Total length of this module command. Should be sizeof(yourstruct) that has a header at the
                            front of it */
    dcgmModuleId_t moduleId; /* Which module to dispatch to. See DcgmModuleId* #defines in dcgm_structs.h */
    unsigned int subCommand; /* Sub-command number for the module to switch on. These are defined by each module */
    dcgm_connection_id_t
        connectionId;       /* The connectionId that generated this request. This can be used to key per-client info or
                                          as a handle to send async responses back to clients. 0=Internal or don't care */
    unsigned int requestId; /* Unique request ID that the sender is using to uniquely identify
                               this request. This can be 0 if the receiving module or the sender
                               doesn't care. This is a dcgm_request_id_t but is an unsigned int here
                               for sanity */
    unsigned int version;   /* Version of this module sub-command, to be checked by the module */
} dcgm_module_command_header_t;


#endif // DCGM_MODULE_STRUCTS_H
