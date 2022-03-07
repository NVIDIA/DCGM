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
sys.path.insert(0,'..')
import test_utils
import apps
import time
import dcgm_agent
import dcgm_agent_internal
import dcgm_structs
import dcgm_fields
import logger
import option_parser
import random

from dcgm_structs import dcgmExceptionClass

g_processesPerSecond = 5        #How many processes should we launch per second
g_processRunTime = 0.1          #How long should each process run in seconds
g_runHostEngineLockStep = False #Should we run the host engine in lock step or auto mode?
g_embeddedMode = True           #Run the host engine embedded in python = True. Remote = False

if g_runHostEngineLockStep:
    g_engineMode = dcgm_structs.DCGM_OPERATION_MODE_MANUAL
else:
    g_engineMode = dcgm_structs.DCGM_OPERATION_MODE_AUTO

class ProcessStatsStressPid:
    def __init__(self):
        self.pid = 0
        self.gpuId = 0
        self.appObj = None

class ProcessStatsStressGpu:
    def __init__(self):
        self.gpuId = -1 #DCGM gpu ID
        self.busId = "" #Bus ID string

class ProcessStatsStress:
    def __init__(self, embeddedMode, heHandle):
        self.gpus = [] #Array of ProcessStatsStressGpu objects
        self.groupName = "pss_group"
        self.groupId = None
        self.addedPids = []
        self.embeddedMode = embeddedMode
        self.heHandle = heHandle

    def __del__(self):
        if self.groupId is not None:
            dcgm_agent.dcgmGroupDestroy(self.heHandle, self.groupId)
            self.groupId = None

        self.heHandle = None

    def Log(self, strVal):
        print(strVal) #Just print for now. Can do more later

    def GetGpus(self):
        """
        Populate self.gpus
        """
        self.groupId = dcgm_agent.dcgmGroupCreate(self.heHandle, dcgm_structs.DCGM_GROUP_DEFAULT, self.groupName)
        groupInfo = dcgm_agent.dcgmGroupGetInfo(self.heHandle, self.groupId, dcgm_structs.c_dcgmGroupInfo_version2)

        gpuIds = groupInfo.gpuIdList[0:groupInfo.count]

        self.Log("Running on %d GPUs" % len(gpuIds))

        for gpuId in gpuIds:
            newGpu = ProcessStatsStressGpu()
            newGpu.gpuId = gpuId
            self.gpus.append(newGpu)

            #Get the busid of the GPU
            fieldId = dcgm_fields.DCGM_FI_DEV_PCI_BUSID
            updateFreq = 100000
            maxKeepAge = 3600.0 #one hour
            maxKeepEntries = 0 #no limit

            dcgm_agent_internal.dcgmWatchFieldValue(self.heHandle, gpuId, fieldId, updateFreq, maxKeepAge, maxKeepEntries)

        #Update all of the new watches
        dcgm_agent.dcgmUpdateAllFields(self.heHandle, 1)

        for gpu in self.gpus:
            values = dcgm_agent_internal.dcgmGetLatestValuesForFields(self.heHandle, gpuId, [fieldId,])
            busId = values[0].value.str.decode('utf-8')
            gpu.busId = busId

            self.Log("    GPUID %d, busId %s" % (gpu.gpuId, gpu.busId))

    def WatchProcessStats(self):

        #watch the process info fields
        updateFreq = 1000000
        maxKeepAge = 3600.0
        maxKeepEntries = 0
        dcgm_agent.dcgmWatchPidFields(self.heHandle, self.groupId, updateFreq, maxKeepAge, maxKeepEntries)


    def StartAppOnGpus(self):

        for gpu in self.gpus:
            pidObj = ProcessStatsStressPid()

            appTimeout = int(1000 * g_processRunTime)

            #Start a cuda app so we have something to accounted
            appParams = ["--ctxCreate", gpu.busId,
                         "--busyGpu", gpu.busId, str(appTimeout),
                         "--ctxDestroy", gpu.busId]
            app = apps.CudaCtxCreateAdvancedApp(appParams)
            app.start(appTimeout*2)
            pidObj.pid = app.getpid()
            pidObj.appObj = app
            self.addedPids.append(pidObj)
            self.Log("Started PID %d. Runtime %d ms" % (pidObj.pid, appTimeout))


    def LoopOneIteration(self):
        for i in range(g_processesPerSecond):
            self.StartAppOnGpus()

        #How many PIDs should we buffer by? Below is 3 seconds worth
        pidBuffer = (3 * g_processesPerSecond * len(self.gpus)) + 1

        #Do we have any pids that have finished yet? Clean them up
        while len(self.addedPids) > pidBuffer:

            #Look up PID info on a random PID that should be done. Assuming 3 seconds is enough
            pidIndex = random.randint(0, len(self.addedPids) - pidBuffer)

            pidObj = self.addedPids[pidIndex]

            try:
                processStats = dcgm_agent.dcgmGetPidInfo(self.heHandle, self.groupId, pidObj.pid)
                self.Log("Found pid stats for pid %d. gpuId %d. returned pid %d" % (pidObj.pid, pidObj.gpuId, processStats.pid))
            except dcgmExceptionClass(dcgm_structs.DCGM_ST_NO_DATA):
                self.Log("Pid %d hasn't finished yet. Sleeping to allow cuda to catch up" % pidObj.pid)
                time.sleep(1.0)
                break

            #Finalize the resources the app object watches
            pidObj.appObj.wait()
            #Delete the found pid so we don't run out of file handles
            del self.addedPids[pidIndex]
            pidObj = None

    def RunLoop(self):

        while True:
            loopStart = time.time()

            #Do the loop work
            self.LoopOneIteration()

            loopEnd = time.time()

            loopDuration = loopEnd - loopStart
            if loopDuration > 1.0:
                self.Log("Warning: Loop took %.2f seconds" % loopDuration)
                continue #Keep on going

            sleepFor = 1.0 - loopDuration
            time.sleep(sleepFor)

    def Run(self):
        self.GetGpus()
        self.WatchProcessStats()
        self.RunLoop()

def processMatchFn(stdoutStr):
    '''
    Callback passed to HostEngineApp.stdout_readtillmatch to see if the host engine has started
    '''
    if stdoutStr.find("Host Engine Listener Started") >= 0:
        return True
    else:
        return False

def main():
    #Make sure logging stuff is bootstrapped
    try:
        option_parser.parse_options()
        option_parser.options.no_logging = True #Don't log anything
        heHandle = None
        heAppRunner = None

        dcgm_structs._LoadDcgmLibrary()

        if g_embeddedMode:
            host = 0
        else:
            #Start host engine
            heAppRunner = apps.NvHostEngineApp()
            heAppRunner.start(timeout=1000000000)
            time.sleep(2.0)
            host = "127.0.0.1"

        heHandle = dcgm_agent.dcgmInit()

        pssObj = ProcessStatsStress(g_embeddedMode, heHandle)
        pssObj.Run()
        del(pssObj) #Force destructor
        heAppRunner.wait()
    except Exception as e:
        raise
    finally:
        apps.AppRunner.clean_all()
        if heHandle is not None:
            dcgm_agent.dcgmShutdown()

if __name__ == "__main__":
    main()
