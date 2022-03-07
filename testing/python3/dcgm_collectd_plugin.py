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
import re
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)

import dcgm_fields_collectd
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

# Add overriding through the environment instead of hard coded.
if 'DCGM_HOSTNAME' in os.environ:
    g_dcgmHostName = os.environ['DCGM_HOSTNAME']

if 'DCGMLIBPATH' in os.environ:
    g_dcgmLibPath = os.environ['DCGMLIBPATH']

c_ONE_SEC_IN_USEC = 1000000

g_intervalSec = 10 # Default

g_dcgmIgnoreFields = [dcgm_fields.DCGM_FI_DEV_UUID] # Fields not to publish

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

g_fieldIntervalMap = None
g_parseRegEx = None
g_fieldRegEx = None

# We build up a regex to match field IDs. These can be numeric IDs, or
# names. We start with field_regex that matches either as a string (as
# well as names that might start with digits, but we do not worry about
# this over-generation of valid IDs at this point).
#
# Basically a field is an integral number or a textual name. A field
# list is a field, or a list of fields separated by commas and enclosed
# in parenthssis. A field list may be optionally followed by a colon,
# indicating a possible non-default interval if also followed by a
# floating point interval value. This is a complete field list.
# Multiple complete field lists may appear, separated by commas.
#
# For example: (1001,tensor_active):5,1002:10
#
# This specifies that fields 1001 and tensor_active are to be sampled
# at a rate of every 5 seconds, and 1002 every ten seconds.
#
# For example: (1001,tensor_active):5,1002:
#
# This is the same, but field 1002 is to be sampled at the default rate
# (and the colon in entirely unnecessary, but not illegal).

field_regex = r"[0-9a-zA-Z_]+"
g_fieldRegEx = re.compile("((" + field_regex + "),?)")

# We now generate a list of field regular expressions, separated by a
# comma, and enclosed with parenthesis, for grouping.

fields_regex = r"\(" + field_regex + "(," + field_regex + ")*" + r"\)"

# This is an optional interval specification, allowing an optional :,
# followed by an optional floating point dcgm sampling interval. If any
# are missing, the default collectd sampling interval is used.

interval_regex = r"(:[0-9]*(\.[0-9]+)?)?,?"

# Here, we combine a field regex or field list regex with an optional
# interval regex. Multiple of these may appear in succession.

g_parseRegEx = re.compile("((" + field_regex + "|(" + fields_regex + "))" + interval_regex + ")")

class DcgmCollectdPlugin(DcgmReader):
    ###########################################################################
    def __init__(self):
        global c_ONE_SEC_IN_USEC

        collectd.debug('Initializing DCGM with interval={}s'.format(g_intervalSec))
        DcgmReader.__init__(self, fieldIds=g_publishFieldIds, ignoreList=g_dcgmIgnoreFields, fieldGroupName='collectd_plugin', updateFrequency=g_intervalSec*c_ONE_SEC_IN_USEC, fieldIntervalMap = g_fieldIntervalMap)


###########################################################################
    def CustomDataHandler(self, fvs):
        global c_ONE_SEC_IN_USEC

        value = collectd.Values(type='gauge')  # pylint: disable=no-member
        value.plugin = 'dcgm_collectd'

        for gpuId in list(fvs.keys()):
            gpuFv = fvs[gpuId]

            uuid = self.m_gpuIdToUUId[gpuId]
            collectd.debug('CustomDataHandler uuid: ' + '%s' % (uuid) + '\n')
            value.plugin_instance = '%s' % (uuid)

            typeInstance = str(gpuId)

            for fieldId in list(gpuFv.keys()):
                # Skip ignore list
                if fieldId in self.m_dcgmIgnoreFields:
                    continue

                fieldTag = self.m_fieldIdToInfo[fieldId].tag
                lastValTime = float("inf")

                # Filter out times too close together (< 1.0 sec) but always
                # include latest one.

                for val in gpuFv[fieldId][::-1]:
                    # Skip blank values. Otherwise, we'd have to insert a placeholder blank value based on the fieldId
                    if val.isBlank:
                        continue

                    valTimeSec1970 = (val.ts / c_ONE_SEC_IN_USEC) #Round down to 1-second for now
                    if (lastValTime - valTimeSec1970) < 1.0:
                        collectd.debug("DCGM sample for field ID %d too soon  at %f, last one sampled at %f" % (fieldId, valTimeSec1970, lastValTime))
                        val.isBlank = True # Filter this one out
                        continue

                    lastValTime = valTimeSec1970

                i = 0

                for val in gpuFv[fieldId]:
                    # Skip blank values. Otherwise, we'd have to insert a placeholder blank value based on the fieldId
                    if val.isBlank:
                        continue

                    # Round down to 1-second for now
                    valTimeSec1970 = (val.ts / c_ONE_SEC_IN_USEC)
                    valueArray = [val.value, ]
                    value.dispatch(type=fieldTag, type_instance=typeInstance, time=valTimeSec1970, values=valueArray, plugin=value.plugin)

                    collectd.debug("    gpuId %d, tag %s, sample %d, value %s, time %s" % (gpuId, fieldTag, i, str(val.value), str(val.ts)))  # pylint: disable=no-member
                    i += 1

    ###########################################################################
    def LogInfo(self, msg):
        collectd.info(msg)  # pylint: disable=no-member

    ###########################################################################
    def LogError(self, msg):
        collectd.error(msg)  # pylint: disable=no-member

###############################################################################
##### Parse supplied collectd configuration object.
###############################################################################
def parse_config(config):
    global c_ONE_SEC_IN_USEC
    global g_intervalSec
    global g_fieldIntervalMap
    global g_parseRegEx
    global g_fieldRegEx

    g_fieldIntervalMap = {}

    for node in config.children:
        if node.key == 'Interval':
            g_intervalSec = float(node.values[0])
        elif node.key == 'FieldIds':
            fieldIds = node.values[0]

            # And we parse out the field ID list with this regex.
            field_set_list = g_parseRegEx.finditer(fieldIds)

            for field_set in field_set_list:
                # We get the list of fields...
                fields = field_set.group(2)

                # ... and the optional interval.
                interval_str = field_set.group(5)

                # We figure out if the default collectd sampling interval is
                # to be used, or a different one.
                if (interval_str == None) or (interval_str == ":"):
                    interval = int(g_intervalSec * c_ONE_SEC_IN_USEC)
                else:
                    interval = int(float(interval_str[1:]) * c_ONE_SEC_IN_USEC) # strip :

                # We keep a set of fields for each unique interval
                if interval not in g_fieldIntervalMap.keys():
                    g_fieldIntervalMap[interval] = []

                # Here we parse out either miltiple fields sharing an
                # interval, or a single field.
                if fields[0:1] == "(": # a true field set
                    fields = fields[1:-1]
                    field_list = g_fieldRegEx.finditer(fields)
                    for field_group in field_list:

                        # We map any field names to field numbers, and add
                        # them to the list for the interval
                        field = dcgm_fields_collectd.GetFieldByName(field_group.group(2))
                        g_fieldIntervalMap[interval] += [field]
                else: # just one field
                    # Map field name to number.
                    field = dcgm_fields_collectd.GetFieldByName(fields)
                    g_fieldIntervalMap[interval] += [field]


###############################################################################
##### Wrapper the Class methods for collectd callbacks
###############################################################################
def config_dcgm(config = None):
    """
    collectd config for dcgm is in the form of a dcgm.conf file, usually
    installed in /etc/collectd/collectd.conf.d/dcgm.conf.

    An example is:

    LoadPlugin python
    <Plugin python>
        ModulePath "/usr/lib64/collectd/dcgm"
        LogTraces true
        Interactive false
        Import "dcgm_collectd_plugin"
        <Module dcgm_collectd_plugin>
            Interval 2
            FieldIds "(1001,tensor_active):5,1002:10,1004:.1,1010:"
            FieldIds "1007"
        </Module>
    </Plugin>

    ModulePath indicates where the plugin and supporting files are installed
    (generally copied from /usr/local/dcgm/bindings/python3).

    Interval is the default collectd sampling interval in seconds.

    FieldIds may appear several times. One is either a field ID by name or
    number. A field ID list is either a single field ID or a list of same,
    separated by commas (,) and bounded by parenthesis ( ( and ) ). Each field
    ID list can be followed by an optional colon (:) and a floating point
    DCGM sampling interval. If no sampling interval is specified the default
    collectd sampling interval is used (and the colon is redundant but not
    illegal). Multiple field ID lists can appear on one FieldIds entry,
    separated by commas (,). FieldIDs are strings and must be enclosed in
    quotes ("). Multiple FieldIds lines are permitted.

    DCGM will sample the fields at the interval(s) indicated, and collectd will
    collect the samples asynchronously at the Interval specified. Because this
    is asynchronous sometimes one less than expected will be collected and other
    times one more than expected will be collected.
    """

    # If we throw an exception here, collectd config will terminate loading the
    # plugin.
    if config is not None:
        parse_config(config)

    # Register the read function with the default collectd sampling interval.
    collectd.register_read(read_dcgm, interval=g_intervalSec)  # pylint: disable=no-member

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
    collectd.register_config(config_dcgm, name="dcgm_collectd_plugin")  # pylint: disable=no-member
    # config_dcgm registers read since it needs to parse the sampling interval.
    collectd.register_init(init_dcgm)  # pylint: disable=no-member
    collectd.register_shutdown(shutdown_dcgm)  # pylint: disable=no-member

###############################################################################
##### Main
###############################################################################
register_collectd_callbacks()
