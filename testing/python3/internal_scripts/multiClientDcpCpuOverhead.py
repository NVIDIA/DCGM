import os
import sys
import time

try:
    import pydcgm
    import dcgm_structs
except:
    print("The DCGM modules were not found. Make sure to provide the 'PYTHONPATH=../' environmental variable")
    sys.exit(1)

try:
    import psutil
except:
    print("psutil is missing. Install with 'pip3 install psutil'")
    sys.exit(1)

numClients = 4 #How many clients to simulate
watchInterval = 10.0 #How often to update watches in seconds
prodWatchInterval = 30.0 #How often DCGM-exporter updates watches in seconds
prodDivisor = prodWatchInterval / watchInterval
gpuIds = None #Either set this to None to use all gpus or a list of gpuIds like [0,1]


fieldIds = [1001,1004,1005,1009,1010,1011,1012] #DCGM-exporter default list

print("Watch list: %s" % (str(fieldIds)))

dcgm_structs._dcgmInit('../apps/amd64')

def getNvHostEngineProcessObject():
    for proc in psutil.process_iter(['name', 'pid']):
        if proc.info['name'] == 'nv-hostengine':
            return psutil.Process(proc.info['pid'])
    
    return None

dcgmProcess = getNvHostEngineProcessObject()
if dcgmProcess is None:
    print("nv-hostengine was not running")
    sys.exit(1)

def getProcessPct(process):
    '''
    Get CPU usage of the passed in process object since last call. Call this
    once at the start and then at the end of each test iteration

    Note that this is % of a single core. so 100% of 1 core in a 12 core system is 100.0.
    '''
    return process.cpu_percent(None)


print("DCGM's PID is %d" % dcgmProcess.pid)

discard = getProcessPct(dcgmProcess)
time.sleep(1.0)
noClientsPct = getProcessPct(dcgmProcess)
print("DCGM used %.f%% CPU with no clients (idle)" % noClientsPct)

clientHandles = []
for i in range(numClients):
    clientHandles.append(pydcgm.DcgmHandle(ipAddress="127.0.0.1"))
discard = getProcessPct(dcgmProcess)
time.sleep(1.0)
idleClientsPct = getProcessPct(dcgmProcess)
print("DCGM used %.f%% CPU with %d idle clients" % (idleClientsPct, numClients))

nameIncrement = 0

class FieldWatcher:
    def __init__(self, dcgmHandle, gpuIds, fieldIds, watchIntervalSecs):
        global nameIncrement
        
        self._dcgmHandle = dcgmHandle
        self._dcgmSystem = dcgmHandle.GetSystem()
        gpuGroupName = "%d_%d" % (os.getpid(), nameIncrement)
        nameIncrement += 1

        if gpuIds is None:
            self._dcgmGroup = self._dcgmSystem.GetDefaultGroup()
        else:
            self._dcgmGroup = self._dcgmSystem.GetGroupWithGpuIds(gpuGroupName, gpuIds)
        self._watchIntervalSecs = watchIntervalSecs
        fieldGroupName = "%d_%d" % (os.getpid(), nameIncrement)
        nameIncrement += 1
        self._dcgmFieldGroup = pydcgm.DcgmFieldGroup(dcgmHandle, fieldGroupName, fieldIds, None)   

    def Watch(self):
        self._dcgmGroup.samples.WatchFields(self._dcgmFieldGroup, int(self._watchIntervalSecs * 1000000), 0, 1)
    
    def Unwatch(self):
        self._dcgmGroup.samples.UnwatchFields(self._dcgmFieldGroup, self._dcgmFieldGroup)
    
    def GetLatestSample(self):
        return self._dcgmGroup.samples.GetLatest(self._dcgmFieldGroup)

    def __del__(self):
        del self._dcgmFieldGroup
        del self._dcgmGroup

watchers = []

print ("====Starting DCP overhead test ====")

for i in range(numClients):
    watchers.append(FieldWatcher(clientHandles[i], gpuIds, fieldIds, watchInterval))
    watchers[-1].Watch()
    
    discard = getProcessPct(dcgmProcess) #Don't measure watch start-up overhead

    sleepTime = watchInterval * 2
    #print("Sleeping %.1f seconds to allow DCP to work in the background" % sleepTime)
    time.sleep(sleepTime)
    pct = getProcessPct(dcgmProcess)
    print("DCGM used %.2f%% CPU (%.2f%% PROD cpu) with %d clients with watches but no field retrieval" % (pct, (pct / prodDivisor), len(watchers)))
    time.sleep(0.5) #psutil suggests this as a minimum polling interval
    

print ("====Starting DCP with polling overhead test ====")
discard = getProcessPct(dcgmProcess)

for numClientsToPoll in range(numClients):
    for i in range(3):
        for j in range(numClientsToPoll):
            sample = watchers[j].GetLatestSample()
            #print(str(sample))
        time.sleep(watchInterval)

    pct = getProcessPct(dcgmProcess)
    print("DCGM used %.2f%% CPU (%.2f%% PROD cpu) with %d clients with watches + samples" % (pct, (pct / prodDivisor), numClientsToPoll+1))


del watchers
