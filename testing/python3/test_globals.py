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
#
# This file contains globals shared between tests and the accelerated test
# wrapper produced by test_compile.py. When a test decorator refers to a symbol
# it must be available in the module where the decorator originally appears AND
# the generated test_compiled_all.py file. The easiest way to do this is to
# define the symbols in a common file, that both can include, hence
# test_globals.py.
#
# This does require that such globals be uniuque across ALL tests, but this is
# generally not a problem as there are few of them that appear in decorators.


DEV_MODE_MSG = "Manual test for verifying plugin output. Use developer mode to enable."

# The sample scripts can potentially take a long time to run since they perform 
# a health check.
SAMPLE_SCRIPT_TIMEOUT = 120.0

DCGM_SKIP_SYSMON_HARDWARE_CHECK = "DCGM_SKIP_SYSMON_HARDWARE_CHECK"

# duration to gather data for when we limit the record count for DCGM to store  
# This time needs to be long enough for memory usage to level off.
BOUNDED_TEST_DURATION = 40
