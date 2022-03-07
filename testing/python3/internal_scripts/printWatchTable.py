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
import pydcgm
import dcgm_fields
import dcgm_agent_internal
import dcgm_structs
import time

dcgmHandle = pydcgm.DcgmHandle(ipAddress="127.0.0.1")
dcgmSystem = dcgmHandle.GetSystem()
dcgmGroup = dcgmSystem.GetDefaultGroup()

#Discover which fieldIds are valid
g_fieldTags = {}
for fieldId in range(1, dcgm_fields.DCGM_FI_MAX_FIELDS):
    fieldMeta = dcgm_fields.DcgmFieldGetById(fieldId)
    if fieldMeta is None:
        continue
    
    g_fieldTags[fieldId] = fieldMeta.tag

#print("Found field tags: " + str(g_fieldTags))

fieldIds = sorted(g_fieldTags.keys())

gpuIds = dcgmGroup.GetGpuIds()

totalSampleCount = 0

cycleCount = 0

while True:
    cycleCount += 1
    print(("Cycle %d. Fields that updated in the last 60 seconds" % cycleCount))

    driverTimeByFieldId = {}
    watchIntervalByFieldId = {}

    for gpuId in gpuIds:
        for fieldId in fieldIds:
            watchInfo = None
            try:
                watchInfo = dcgm_agent_internal.dcgmGetCacheManagerFieldInfo(dcgmHandle.handle, gpuId, fieldId)
            except:
                pass
            
            if watchInfo is None:
                continue
            
            nowTs = int(time.time() * 1000000)
            oneMinuteAgoTs = nowTs - 60000000

            if watchInfo.newestTimestamp < oneMinuteAgoTs:
                continue
            
            perUpdate = 0
            if watchInfo.fetchCount > 0:
                perUpdate = watchInfo.execTimeUsec / watchInfo.fetchCount

            if fieldId not in driverTimeByFieldId:
                driverTimeByFieldId[fieldId] = [perUpdate, ]
            else:
                driverTimeByFieldId[fieldId].append(perUpdate)

            lastUpdateSec = (nowTs - watchInfo.newestTimestamp) / 1000000.0

            if fieldId not in watchIntervalByFieldId:
                watchIntervalByFieldId[fieldId] = watchInfo.monitorFrequencyUsec
            else:
                watchIntervalByFieldId[fieldId] = max(watchInfo.monitorFrequencyUsec, watchIntervalByFieldId[fieldId])

            #print("gpuId %d, fieldId %d (%s). lastUpdate: %f s, execTime %d, fetchCount %d, perUpdate: %d" % 
            #      (gpuId, fieldId, g_fieldTags[fieldId], lastUpdateSec, 
            #      watchInfo.execTimeUsec, watchInfo.fetchCount, perUpdate))
    
    totalDriverTime = 0
    for fieldId in list(driverTimeByFieldId.keys()):
        numGpus = len(driverTimeByFieldId[fieldId])
        fieldDriverTime = sum(driverTimeByFieldId[fieldId])
        totalDriverTime += fieldDriverTime
        driverTimeAvg = fieldDriverTime / numGpus
        print(("fieldId %d (%s), numGpus %u, driverTimePerUpdate %d usec, watchInterval %d usec" %
             (fieldId, g_fieldTags[fieldId], numGpus, driverTimeAvg, watchIntervalByFieldId[fieldId])))
    
    print(("Total Driver Time: %d usec" % totalDriverTime))
    print("")

    time.sleep(5.0)


