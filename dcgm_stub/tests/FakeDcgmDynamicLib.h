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
#ifndef DCGM_FAKE_DCGM_DYNAMIC_LIB_H
#define DCGM_FAKE_DCGM_DYNAMIC_LIB_H

#include <dcgm_structs.h>

#define FAKE_DCGM_HANDLE           ((dcgmHandle_t)0x778899U)
#define FAKE_DCGM_GPU_ID_0         101U
#define FAKE_DCGM_GPU_ID_1         202U
#define FAKE_DCGM_GPU_COUNT        2
#define FAKE_DCGM_GROUP_ID         ((dcgmGpuGrp_t)0xABC)
#define FAKE_DCGM_GROUP_COUNT      1U
#define FAKE_DCGM_OWNED_CORE_ID    7U
#define FAKE_DCGM_DEVICE_NAME      "fake-gpu-101"
#define FAKE_DCGM_CORE_MODULE_NAME "core"
#define FAKE_DCGM_BADPARAM_STRING  "fake bad parameter"

#endif // DCGM_FAKE_DCGM_DYNAMIC_LIB_H
