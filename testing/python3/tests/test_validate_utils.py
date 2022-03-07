# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
'''
Tests written for test_utils.py
'''
import test_utils
import dcgm_agent_internal

@test_utils.run_with_embedded_host_engine()
def test_utils_run_with_embedded_host_engine(handle):
    '''
    Sanity test for running with an embedded host engine
    '''
    assert(handle.value == dcgm_agent_internal.DCGM_EMBEDDED_HANDLE.value), \
            "Expected embedded handle %s but got %s" % \
            (hex(dcgm_agent_internal.DCGM_EMBEDDED_HANDLE.value), hex(handle.value))
            
@test_utils.run_with_standalone_host_engine()
@test_utils.run_with_initialized_client()
def test_utils_run_with_standalone_host_engine(handle):
    '''
    Sanity test for running with a standalone host engine
    '''
    assert(handle.value != dcgm_agent_internal.DCGM_EMBEDDED_HANDLE.value), \
            "Expected a handle different from the embedded one %s" % \
            hex(dcgm_agent_internal.DCGM_EMBEDDED_HANDLE.value)
