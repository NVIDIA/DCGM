# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import nvml_injection_structs
from ctypes import Structure, ARRAY, c_uint, c_char, c_uint32, c_uint

NVML_INJECTION_MAX_EXTRA_KEYS = 4
NVML_INJECTION_MAX_VALUES = 4
NVML_INJECTION_MAX_RETURNS = 8
NVML_MAX_FUNCS = 1024
NVML_MAX_FUNC_NAME_LENGTH = 1024

if nvml_injection_structs.nvml_injection_usable:
    class c_injectNvmlRet_t(Structure):
        _fields_ = [
            ("nvmlRet", c_uint),
            ("values", ARRAY(nvml_injection_structs.c_injectNvmlVal_t, (NVML_INJECTION_MAX_VALUES + 1))),
            ("valueCount", c_uint),
        ]
    
    class c_injectNvmlFuncCallInfo_t(Structure):
        _fields_ = [
            ("funcName", c_char * NVML_MAX_FUNC_NAME_LENGTH),
            ("funcCallCount", c_uint32)
        ]

    class c_injectNvmlFuncCallCounts_t(Structure):
        _fields_ = [
            ("funcCallInfo", ARRAY(c_injectNvmlFuncCallInfo_t, NVML_MAX_FUNCS)),
            ("numFuncs", c_uint),
        ]
