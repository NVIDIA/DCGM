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
import sys
import subprocess
import signal
import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)

import pydcgm
import dcgm_fields
import dcgm_structs
import threading
from DcgmReader import DcgmReader

if 'DCGM_TESTING_FRAMEWORK' in os.environ:
    try:
        import collectd_tester_api as collectd
    except:
        import collectd
else:
    import collectd

# Set default values for the hostname and the library path
g_dcgmLibPath = '/usr/lib'
g_dcgmHostName = 'localhost'

# Add overriding through the environment instead of through the
if 'DCGM_HOSTNAME' in os.environ:
    g_dcgmHostName = os.environ['DCGM_HOSTNAME']

if 'DCGMLIBPATH' in os.environ:
    g_dcgmLibPath = os.environ['DCGMLIBPATH']

g_intervalSec = 10

g_dcgmIgnoreFields = [dcgm_fields.DCGM_FI_DEV_UUID] #Fields not to publish

g_publishFieldIds = [
    dcgm_fields.DCGM_FI_DEV_UUID, #Needed for plugin instance
    dcgm_fields.DCGM_FI_DEV_POWER_USAGE,
    dcgm_fields.DCGM_FI_DEV_GPU_TEMP,
    dcgm_fields.DCGM_FI_DEV_SM_CLOCK,
    dcgm_fields.DCGM_FI_DEV_GPU_UTIL,
    dcgm_fields.DCGM_FI_DEV_RETIRED_PENDING,
    dcgm_fields.DCGM_FI_DEV_RETIRED_SBE,
    dcgm_fields.DCGM_FI_DEV_RETIRED_DBE,
    dcgm_fields.DCGM_FI_DEV_ECC_SBE_VOL_TOTAL,
    dcgm_fields.DCGM_FI_DEV_ECC_DBE_VOL_TOTAL,
    dcgm_fields.DCGM_FI_DEV_ECC_SBE_AGG_TOTAL,
    dcgm_fields.DCGM_FI_DEV_ECC_DBE_AGG_TOTAL,
    dcgm_fields.DCGM_FI_DEV_FB_TOTAL,
    dcgm_fields.DCGM_FI_DEV_FB_FREE,
    dcgm_fields.DCGM_FI_DEV_FB_USED,
    dcgm_fields.DCGM_FI_DEV_PCIE_REPLAY_COUNTER,
    dcgm_fields.DCGM_FI_DEV_POWER_VIOLATION,
    dcgm_fields.DCGM_FI_DEV_THERMAL_VIOLATION,
    dcgm_fields.DCGM_FI_DEV_XID_ERRORS,
    dcgm_fields.DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_TOTAL,
    dcgm_fields.DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_TOTAL,
    dcgm_fields.DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_TOTAL,
    dcgm_fields.DCGM_FI_DEV_MEM_CLOCK,
    dcgm_fields.DCGM_FI_DEV_MEMORY_TEMP,
    dcgm_fields.DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION,
    dcgm_fields.DCGM_FI_DEV_MEM_COPY_UTIL,
    dcgm_fields.DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_TOTAL,
    dcgm_fields.DCGM_FI_DEV_NVLINK_BANDWIDTH_TOTAL,
    dcgm_fields.DCGM_FI_DEV_PCIE_TX_THROUGHPUT,
    dcgm_fields.DCGM_FI_DEV_PCIE_RX_THROUGHPUT
    ]

class DcgmCollectdPlugin(DcgmReader):
    ###########################################################################
    def __init__(self):
        collectd.debug('Initializing DCGM with interval={}s'.format(g_intervalSec))
        DcgmReader.__init__(self, fieldIds=g_publishFieldIds, ignoreList=g_dcgmIgnoreFields, fieldGroupName='collectd_plugin', updateFrequency=g_intervalSec*1000000)

    ###########################################################################
    def CustomDataHandler(self, fvs):
        value = collectd.Values(type='gauge')  # pylint: disable=no-member
        value.plugin = 'dcgm_collectd'

        for gpuId in fvs.keys():
            gpuFv = fvs[gpuId]

            uuid = self.m_gpuIdToUUId[gpuId]
            value.plugin_instance = '%s' % (uuid)

            typeInstance = str(gpuId)

            for fieldId in gpuFv.keys():
                # Skip ignore list
                if fieldId in self.m_dcgmIgnoreFields:
                    continue

                fieldTag = self.m_fieldIdToInfo[fieldId].tag
                val = gpuFv[fieldId][-1]

                #Skip blank values. Otherwise, we'd have to insert a placeholder blank value based on the fieldId
                if val.isBlank:
                    continue

                valTimeSec1970 = (val.ts / 1000000) #Round down to 1-second for now
                valueArray = [val.value, ]

                value.dispatch(type=fieldTag, type_instance=typeInstance, time=valTimeSec1970, values=valueArray, plugin=value.plugin)

                collectd.debug("gpuId %d, tag %s, value %s" % (gpuId, fieldTag, str(val.value)))  # pylint: disable=no-member

    ###########################################################################
    def LogInfo(self, msg):
        collectd.info(msg)  # pylint: disable=no-member

    ###########################################################################
    def LogError(self, msg):
        collectd.error(msg)  # pylint: disable=no-member

###############################################################################
##### Wrapper the Class methods for collectd callbacks
###############################################################################
def config_dcgm(config):
    global g_intervalSec

    for node in config.children:
        if node.key == 'Interval':
            g_intervalSec = int(node.values[0])

###############################################################################
def init_dcgm():
    global g_dcgmCollectd

    # restore default SIGCHLD behavior to avoid exceptions with new processes
    signal.signal(signal.SIGCHLD, signal.SIG_DFL)

    g_dcgmCollectd = DcgmCollectdPlugin()
    g_dcgmCollectd.Init()

###############################################################################
def shutdown_dcgm():
    g_dcgmCollectd.Shutdown()

###############################################################################
def read_dcgm(data=None):
    g_dcgmCollectd.Process()

def register_collectd_callbacks():
    collectd.register_config(config_dcgm)  # pylint: disable=no-member
    collectd.register_init(init_dcgm)  # pylint: disable=no-member
    collectd.register_read(read_dcgm)  # pylint: disable=no-member
    collectd.register_shutdown(shutdown_dcgm)  # pylint: disable=no-member

###############################################################################
##### Main
###############################################################################
register_collectd_callbacks()
