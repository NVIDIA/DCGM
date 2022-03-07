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

================
DCGM SDK Samples
================

Building the samples
--------------------

In order to build the samples:

- ensure that DCGM is installed
- copy the samples directory to a location you have write access to
- run the following commands:
  - mkdir build
  - cd build
  - cmake .. # you might need to provide CMAKE_PREFIX_PATH=/usr/lib/x86_64-linux-gnu/cmake
             # or the appropriate prefix according to your platform
  - make

-----------------------------------------------
Sample: Group Configuration 
Folder: 0_configuration_sample

This sample goes through the process of creating a group, adding GPUs to it and then getting, setting and enforcing a configuration
on that group. Some error handling through status handles is also shown.

Key concepts:
- Querying for GPUs on system
- Group creation
- Managing group configurations
- Status handles


-----------------------------------------------
Sample: Group Health, Watches and Diagnostics
Folder: 1_healthAndDiagnostics_sample

This sample demonstrates the process of creating a group and managing health watches for that group. Demonstrates setting watches, 
querying them for information and also running group diagnostics.

Key concepts:
- Group creation
- Managing group health watches
- Managing group field watches
- Running group diagnostics
- DCGM fields and field collections


-----------------------------------------------
Sample: Process statistics
Folder: 2_processStatistics_sample

This sample goes through the process of enabling process watches on a group, running a process and viewing the statistics of the 
group while the process ran.

Key concepts:
- Using the default group (All GPUs)
- Managing process watches


-----------------------------------------------
Sample: Group Policy
Folder: 3_policy_sample

This sample demonstrates the process of creating a group and getting/setting the policies of the GPUs in that group. Policy 
registration is also shown.

Key concepts:
- Group Creation
- Managing group policies
- Registering for policy violation callbacks
- Status handles
