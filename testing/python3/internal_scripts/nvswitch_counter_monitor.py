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
try:
    import pydcgm
except ImportError:
    print("Unable to find pydcgm. You need to add the location of "
          "pydcgm.py to your environment as PYTHONPATH=$PYTHONPATH:[path-to-pydcgm.py]")

import sys
import os
import time
import dcgm_field_helpers
import dcgm_fields
import dcgm_structs

class NvSwitchCounterMonitor:
    def __init__(self, hostname):
        self._pidPostfix = "_" + str(os.getpid()) #Add this to any names so we can run multiple instances
        self._updateIntervalSecs = 30.0 #How often to print out new rows
        self._hostname = hostname
        self.NVSWITCH_NUM_LINKS = 18
        self._InitFieldLists()
        self._InitHandles()

    def _InitFieldLists(self):
        self._nvSwitchLatencyFieldIds = []
        #get the low/medium/high/max latency bucket field ids, each switch port has 4 values.
        #the field ids are contiguous, where first 4 ids are for port0, next 4 for port1 and so on.
        for i in range(dcgm_fields.DCGM_FI_DEV_NVSWITCH_LATENCY_LOW_P00, dcgm_fields.DCGM_FI_DEV_NVSWITCH_LATENCY_MAX_P17+1, 1):
            self._nvSwitchLatencyFieldIds.append(i)

        #need two lists because there is gap between bandwidth0 and bandwidth1 field Ids.
        #each counter has two values, TX_0 and RX_0. 
        #the field ids are contiguous, where first 2 ids are for port0, next 2 for port1 and so on.
        self._nvSwitchBandwidth0FieldIds = []
        for i in range(dcgm_fields.DCGM_FI_DEV_NVSWITCH_BANDWIDTH_TX_0_P00, dcgm_fields.DCGM_FI_DEV_NVSWITCH_BANDWIDTH_RX_0_P17+1, 1):
            self._nvSwitchBandwidth0FieldIds.append(i)

        #get bandwidth counter1 field ids, ie TX_1, RX_1
        self._nvSwitchBandwidth1FieldIds = []
        for i in range(dcgm_fields.DCGM_FI_DEV_NVSWITCH_BANDWIDTH_TX_1_P00, dcgm_fields.DCGM_FI_DEV_NVSWITCH_BANDWIDTH_RX_1_P17+1, 1):
            self._nvSwitchBandwidth1FieldIds.append(i)

    def _InitHandles(self):
        self._dcgmHandle = pydcgm.DcgmHandle(ipAddress=self._hostname)

        groupName = "bandwidth_mon_nvswitches" + self._pidPostfix
        self._allNvSwitchesGroup = pydcgm.DcgmGroup(self._dcgmHandle, groupName=groupName, groupType=dcgm_structs.DCGM_GROUP_DEFAULT_NVSWITCHES)
        print(("Found %d NVSwitches" % len(self._allNvSwitchesGroup.GetEntities())))

        fgName = "latency_mon_nvswitches" + self._pidPostfix
        self._nvSwitchLatencyFieldGroup = pydcgm.DcgmFieldGroup(self._dcgmHandle, name=fgName, fieldIds=self._nvSwitchLatencyFieldIds)

        fgName = "bandwidth0_mon_nvswitches" + self._pidPostfix
        self._nvSwitchBandwidth0FieldGroup = pydcgm.DcgmFieldGroup(self._dcgmHandle, name=fgName, fieldIds=self._nvSwitchBandwidth0FieldIds)

        fgName = "bandwidth1_mon_nvswitches" + self._pidPostfix
        self._nvSwitchBandwidth1FieldGroup = pydcgm.DcgmFieldGroup(self._dcgmHandle, name=fgName, fieldIds=self._nvSwitchBandwidth1FieldIds)

        updateFreq = int(self._updateIntervalSecs / 2.0) * 1000000
        maxKeepAge = 3600.0 #1 hour
        maxKeepSamples = 0 #Rely on maxKeepAge

        self._nvSwitchLatencyWatcher = dcgm_field_helpers.DcgmFieldGroupEntityWatcher(
            self._dcgmHandle.handle, self._allNvSwitchesGroup.GetId(), 
            self._nvSwitchLatencyFieldGroup, dcgm_structs.DCGM_OPERATION_MODE_AUTO,
            updateFreq, maxKeepAge, maxKeepSamples, 0)
        self._nvSwitchBandwidth0Watcher = dcgm_field_helpers.DcgmFieldGroupEntityWatcher(
            self._dcgmHandle.handle, self._allNvSwitchesGroup.GetId(), 
            self._nvSwitchBandwidth0FieldGroup, dcgm_structs.DCGM_OPERATION_MODE_AUTO,
            updateFreq, maxKeepAge, maxKeepSamples, 0)
        self._nvSwitchBandwidth1Watcher = dcgm_field_helpers.DcgmFieldGroupEntityWatcher(
            self._dcgmHandle.handle, self._allNvSwitchesGroup.GetId(), 
            self._nvSwitchBandwidth1FieldGroup, dcgm_structs.DCGM_OPERATION_MODE_AUTO,
            updateFreq, maxKeepAge, maxKeepSamples, 0)

    def _MonitorOneCycle(self):
        numErrors = 0
        nowStr = time.strftime("%m/%d/%Y %H:%M:%S") 
        self._nvSwitchLatencyWatcher.GetMore()
        self._nvSwitchBandwidth0Watcher.GetMore()
        self._nvSwitchBandwidth1Watcher.GetMore()
        #3D dictionary of [entityGroupId][entityId][fieldId](DcgmFieldValueTimeSeries)
        # where entityId = SwitchID
        for entityGroupId in list(self._nvSwitchLatencyWatcher.values.keys()):
            for entityId in self._nvSwitchLatencyWatcher.values[entityGroupId]:
                latencyFieldId = dcgm_fields.DCGM_FI_DEV_NVSWITCH_LATENCY_LOW_P00
                for linkIdx in range(0, self.NVSWITCH_NUM_LINKS):
                    # if the link is not enabled, then the corresponding latencyFieldId key value will be
                    # empty, so skip those links.
                    if latencyFieldId in self._nvSwitchLatencyWatcher.values[entityGroupId][entityId]:
                        latencyLow = self._nvSwitchLatencyWatcher.values[entityGroupId][entityId][latencyFieldId].values[-1].value
                        latencyFieldId += 1
                        latencyMed = self._nvSwitchLatencyWatcher.values[entityGroupId][entityId][latencyFieldId].values[-1].value
                        latencyFieldId += 1
                        latencyHigh = self._nvSwitchLatencyWatcher.values[entityGroupId][entityId][latencyFieldId].values[-1].value
                        latencyFieldId += 1
                        latencyMax = self._nvSwitchLatencyWatcher.values[entityGroupId][entityId][latencyFieldId].values[-1].value
                        latencyFieldId += 1
                        print(("SwitchID %d LinkIdx %d Latency Low %d Medium %d High %d Max %d"
                                % (entityId, linkIdx, latencyLow, latencyMed, latencyHigh, latencyMax)))
                    else:
                        latencyFieldId += 4;
        for entityGroupId in list(self._nvSwitchBandwidth0Watcher.values.keys()):
            for entityId in self._nvSwitchBandwidth0Watcher.values[entityGroupId]:
                bandwidth0FieldId = dcgm_fields.DCGM_FI_DEV_NVSWITCH_BANDWIDTH_TX_0_P00
                bandwidth1FieldId = dcgm_fields.DCGM_FI_DEV_NVSWITCH_BANDWIDTH_TX_1_P00
                for linkIdx in range(0, self.NVSWITCH_NUM_LINKS):
                    # if the link is not enabled, then the corresponding bandwidth0FieldId and
                    # bandwidth1FieldId key values will be empty, so skip those links.
                    if bandwidth0FieldId in self._nvSwitchBandwidth0Watcher.values[entityGroupId][entityId]:
                        counter0Tx = self._nvSwitchBandwidth0Watcher.values[entityGroupId][entityId][bandwidth0FieldId].values[-1].value
                        counter1Tx = self._nvSwitchBandwidth1Watcher.values[entityGroupId][entityId][bandwidth1FieldId].values[-1].value
                        bandwidth0FieldId += 1
                        bandwidth1FieldId += 1
                        counter0Rx = self._nvSwitchBandwidth0Watcher.values[entityGroupId][entityId][bandwidth0FieldId].values[-1].value
                        counter1Rx = self._nvSwitchBandwidth1Watcher.values[entityGroupId][entityId][bandwidth1FieldId].values[-1].value
                        bandwidth0FieldId += 1
                        bandwidth1FieldId += 1
                        print(("SwitchID %d LinkIdx %d counter0Tx %d counter0Rx %d counter1Tx %d counter1Rx %d"
                                % (entityId, linkIdx, counter0Tx, counter0Rx, counter1Tx, counter1Rx)))
                    else:
                        bandwidth0FieldId += 2
                        bandwidth1FieldId += 2

        self._nvSwitchLatencyWatcher.EmptyValues()
        self._nvSwitchBandwidth0Watcher.EmptyValues()
        self._nvSwitchBandwidth1Watcher.EmptyValues()

    def Monitor(self):
        self._nvSwitchLatencyWatcher.EmptyValues()
        self._nvSwitchBandwidth0Watcher.EmptyValues()
        self._nvSwitchBandwidth1Watcher.EmptyValues()

        try:
            while True:
                self._MonitorOneCycle()
                time.sleep(self._updateIntervalSecs)
        except KeyboardInterrupt:
            print ("Got CTRL-C. Exiting")
            return



def main():
    if len(sys.argv) > 1:
        hostname = sys.argv[1]
    else:
        hostname = "localhost"

    counterMonitor = NvSwitchCounterMonitor(hostname)
    print(("Using hostname %s and update interval as %d secs " % (hostname, counterMonitor._updateIntervalSecs)))
    counterMonitor.Monitor()

if __name__ == "__main__":
    main()
