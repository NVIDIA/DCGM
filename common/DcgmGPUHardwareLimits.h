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
#ifndef DCGM_GPU_HARDWARE_LIMITS_H
#define DCGM_GPU_HARDWARE_LIMITS_H

// 8 replays per minute is the maximum recommended by RM
#define DCGM_LIMIT_MAX_PCIREPLAY_RATE 8


// defined via bug 1665722
// Max retired pages updated to match the field diag: http://nvbugs/3024088
#define DCGM_LIMIT_MAX_RETIRED_PAGES            63
#define DCGM_LIMIT_MAX_RETIRED_PAGES_SOFT_LIMIT 15

// dummy value until a real one is determined. JIRA: DCGM-445
#define DCGM_LIMIT_MAX_SBE_RATE 10


#define DCGM_LIMIT_MAX_NVLINK_ERROR 1
// RM has informed us that CRC errors only matter at rates of 100+ per second
#define DCGM_LIMIT_MAX_NVLINK_CRC_ERROR 100.0

#endif // DCGM_GPU_HARDWARE_LIMITS_H
