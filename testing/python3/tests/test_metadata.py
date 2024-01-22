# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import dcgm_structs
import dcgm_agent
import dcgm_agent_internal
import dcgm_fields
import pydcgm
import logger
import test_utils
import stats
    
@test_utils.run_with_standalone_host_engine()
@test_utils.run_with_initialized_client()
@test_utils.run_only_on_architecture('amd64|ppc64le')
def test_dcgm_standalone_metadata_memory_get_hostengine_sane(handle):
    """
    Sanity test for API that gets memory usage of the hostengine process
    """
    handle = pydcgm.DcgmHandle(handle)
    system = pydcgm.DcgmSystem(handle)
    
    bytesUsed = system.introspect.memory.GetForHostengine().bytesUsed 
    
    logger.debug('the hostengine process is using %.2f MB' % (bytesUsed / 1024. / 1024.))
    
    assert(1*1024*1024 < bytesUsed < 100*1024*1024), bytesUsed        # 1MB to 100MB

def _cpu_load(start_time, duration_sec, x):
    while time.time() - start_time < duration_sec:
        x*x

def _cpu_load_star(arg1, arg2):
    """ Convert arguments from (start_time, duration_sec), x to start_time, duration_sec, x"""
    _cpu_load(*arg1, arg2)

@test_utils.run_with_embedded_host_engine()
def test_dcgm_embedded_metadata_cpuutil_get_hostengine_sane(handle):
    """
    Sanity test for API that gets CPU Utilization of the hostengine process.
    """
    from multiprocessing import cpu_count
    handle = pydcgm.DcgmHandle(handle)
    system = pydcgm.DcgmSystem(handle)

    def generate_cpu_load(duration_sec):
        """
        Generate a CPU load for a given duration.
        """
        from multiprocessing import Pool
        from itertools import repeat

        start_time = time.time()
        processes = cpu_count()
        with Pool(processes) as pool:
            pool.starmap(_cpu_load_star, zip(repeat((start_time, duration_sec)), range(processes)))

    def get_current_process_cpu_util():
        """Return a tuple representing CPU user-time and system-time and total for the current process
        """
        import os
        with open('/proc/self/stat', "rb", buffering=0) as f:
            data = f.readline()
        values = data[data.rfind(b')') + 2:].split()
        utime = float(values[11]) / os.sysconf("SC_CLK_TCK")
        stime = float(values[12]) / os.sysconf("SC_CLK_TCK")
        return utime, stime, utime + stime

    start = time.time()
    start_cpu_util = get_current_process_cpu_util()
    system.introspect.cpuUtil.GetForHostengine()
    generate_cpu_load(1)
    stop = time.time()
    stop_cpu_util = get_current_process_cpu_util()
    cpuUtil = system.introspect.cpuUtil.GetForHostengine(False)

    #diff_utime = stop_cpu_util[0] - start_cpu_util[0]
    #diff_stime = stop_cpu_util[1] - start_cpu_util[1]
    diff_total = stop_cpu_util[2] - start_cpu_util[2]
    diff_time = stop - start
    overall_cpu_util = diff_total / diff_time
    logger.debug("DCGM CPU Util: %f" % (cpuUtil.total * cpu_count()))
    logger.debug('Stats CPU Util: %f' % overall_cpu_util)
    assert abs(overall_cpu_util - (cpu_count() * cpuUtil.total)) < 0.05, "CPU Utilization was not within 5% of expected value"

    # test that user and kernel add to total (with rough float accuracy)
    assert abs(cpuUtil.total - (cpuUtil.user + cpuUtil.kernel)) <= 4*float_info.epsilon, \
           'CPU kernel and user utilization did not add up to total. Kernel: %f, User: %f, Total: %f' \
           % (cpuUtil.kernel, cpuUtil.user, cpuUtil.total)

