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
# test the metadata API calls for DCGM
import time
from sys import float_info

import dcgm_fields
import pydcgm
import logger
import test_utils
    
@test_utils.run_with_standalone_host_engine()
@test_utils.run_only_on_architecture('amd64')
def test_dcgm_standalone_metadata_memory_get_hostengine_sane(handle):
    """
    Sanity test for API that gets memory usage of the hostengine process
    """
    handle = pydcgm.DcgmHandle(handle)
    system = pydcgm.DcgmSystem(handle)
    
    bytesUsed = system.introspect.memory.GetForHostengine().bytesUsed 
    
    logger.debug('the hostengine process is using %.2f MB' % (bytesUsed / 1024. / 1024.))
    
    assert(1*1024*1024 < bytesUsed < 100*1024*1024), bytesUsed        # 1MB to 100MB

def helper_watch_fields(handle, system):
    fieldIds = [
        dcgm_fields.DCGM_FI_DEV_NAME,
        dcgm_fields.DCGM_FI_DEV_BRAND,
        dcgm_fields.DCGM_FI_DEV_SERIAL,
        dcgm_fields.DCGM_FI_DEV_UUID,
        dcgm_fields.DCGM_FI_DEV_PCI_BUSID,
        dcgm_fields.DCGM_FI_DEV_MEMORY_TEMP,
        dcgm_fields.DCGM_FI_DEV_GPU_TEMP,
        dcgm_fields.DCGM_FI_DEV_POWER_USAGE,
        dcgm_fields.DCGM_FI_DEV_FABRIC_MANAGER_STATUS,
        dcgm_fields.DCGM_FI_DEV_FAN_SPEED,
    ]
    fieldGroupName = "my_field_group"
    updateFreq = 100000  # in usec, 0.1 sec
    maxKeepAge = 86400.0 # in seconds, 24 hours
    maxKeepSamples = 0   # no limit

    fieldGroup = pydcgm.DcgmFieldGroup(handle, fieldGroupName, fieldIds)
    dcgmGroup = system.GetDefaultGroup()
    dcgmGroup.samples.WatchFields(fieldGroup, updateFreq, maxKeepAge, maxKeepSamples)

@test_utils.run_with_embedded_host_engine()
def test_dcgm_embedded_metadata_cpuutil_get_hostengine_sane(handle):
    """
    Sanity test for API that gets CPU Utilization of the hostengine process.
    """
    from multiprocessing import cpu_count
    handle = pydcgm.DcgmHandle(handle)
    system = pydcgm.DcgmSystem(handle)

    def get_current_process_cpu_util():
        """
        Return a tuple representing CPU user-time and system-time and total for the current process
        """
        import os
        with open('/proc/self/stat', "rb", buffering=0) as f:
            data = f.readline()
        values = data[data.rfind(b')') + 2:].split()
        utime = float(values[11]) / os.sysconf("SC_CLK_TCK")
        stime = float(values[12]) / os.sysconf("SC_CLK_TCK")
        return utime, stime, utime + stime

    # HostEngine lacks initial data on CPU utilization. By calling the following API and incorporating a sleep,
    # the metadata is gathered, establishing a baseline for subsequent calls to retrieve CPU utilization.
    cpuUtil = system.introspect.cpuUtil.GetForHostengine()
    logger.debug('DCGM CPU Util before watching fields: %f' % (cpuUtil.total * cpu_count()))
    sleepTimeInSec = 1
    time.sleep(sleepTimeInSec)

    # Monitor multiple fields and record CPU utilization before and after a sleep period.
    # With an update frequency of 0.1 seconds, sleeping for one second will monitor each field 10 times.
    # CPU utilization for this activity should remain within acceptable limits.
    # Additionally, measure CPU utilization outside of the host-engine and verify that it falls within the expected range.
    helper_watch_fields(handle, system)
    startTime = time.time()
    startCpuUtil = get_current_process_cpu_util()
    hostEngineCpuUtilBefore = system.introspect.cpuUtil.GetForHostengine(False)
    time.sleep(sleepTimeInSec) # Sleep for sometime for host-engine to gather clock ticks from watchers.
    stopTime = time.time()
    stopCpuUtil = get_current_process_cpu_util()
    hostEngineCpuUtilAfter = system.introspect.cpuUtil.GetForHostengine(False)

    # assert that CPU utilization stays within acceptable limits.
    cpuUtilBefore = hostEngineCpuUtilBefore.total * cpu_count()
    cpuUtilAfter = hostEngineCpuUtilAfter.total * cpu_count()
    cpuUtilDifference = cpuUtilAfter - cpuUtilBefore
    cpuUtilrange = 0.1
    cpuUtilLimit = cpuUtilrange * cpu_count()
    assert cpuUtilDifference < cpuUtilLimit, \
        'CPU Utilization increased over acceptable limits of %.2f%%. CPU util before: %f, after: %f, difference: %f' \
        % (cpuUtilLimit * 100, cpuUtilBefore, cpuUtilAfter, cpuUtilDifference)

    # assert that CPU utilization remains within the expected range by verifying it against external statistics.
    diffTotal = stopCpuUtil[2] - startCpuUtil[2]
    diffTime = stopTime - startTime
    overallCpuUtil = diffTotal / diffTime
    absoluteDiff = abs(overallCpuUtil - cpuUtilAfter)
    assert absoluteDiff < cpuUtilrange, \
        'CPU Utilization was not within %.2f%% of expected range. DCGM CPU util: %f, Stats CPU util: %f, absolute diff: %f' \
        % (cpuUtilrange * 100, cpuUtilAfter, overallCpuUtil, absoluteDiff)

    # test that user and kernel add to total (with rough float accuracy)
    assert abs(hostEngineCpuUtilAfter.total - (hostEngineCpuUtilAfter.user + hostEngineCpuUtilAfter.kernel)) <= 4*float_info.epsilon, \
        'CPU kernel and user utilization did not add up to total. Kernel: %f, User: %f, Total: %f' \
        % (hostEngineCpuUtilAfter.kernel, hostEngineCpuUtilAfter.user, hostEngineCpuUtilAfter.total)
