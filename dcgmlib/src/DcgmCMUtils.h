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
#pragma once

#include "dcgm_fields.h"
#include "dcgm_nvml.h"
#include "dcgm_structs.h"

#include "DcgmLogging.h"

/*************************************************************************/
bool DcgmFieldIsMappedToNvmlField(dcgm_field_meta_p fieldMeta, bool driver520OrNewer);

/*************************************************************************/
/* Convert a NVML return code to an appropriate null value */
const char *NvmlErrorToStringValue(nvmlReturn_t nvmlReturn);
long long NvmlErrorToInt64Value(nvmlReturn_t nvmlReturn);
int NvmlErrorToInt32Value(nvmlReturn_t nvmlReturn);
double NvmlErrorToDoubleValue(nvmlReturn_t nvmlReturn);
