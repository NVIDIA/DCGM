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
import argparse
import sys
import logging
import json
import os

try:
    import pydcgm
    import dcgm_agent
    import dcgm_structs
    import dcgm_errors
    import dcgm_fields
    import DcgmSystem
except:
    # If we don't find the bindings, add the default path and try again
    if 'PYTHONPATH' in os.environ:
        os.environ['PYTHONPATH'] = os.environ['PYTHONPATH'] + ":/usr/local/dcgm/bindings"
    else:
        os.environ['PYTHONPATH'] = '/usr/local/dcgm/bindings'

    import pydcgm
    import dcgm_agent
    import dcgm_structs
    import dcgm_fields
    import DcgmSystem

BR_ST_HEALTHY               = 0x0000
BR_ST_NOT_DETECTED          = 0x0001
BR_ST_FAILED_PASSIVE_HEALTH = 0x0002
BR_ST_FAILED_ACTIVE_HEALTH  = 0x0004

BR_HEALTH_WATCH_BITMAP   = dcgm_structs.DCGM_HEALTH_WATCH_ALL

DIAG_SM_STRESS_DURATION       = 90.0
DIAG_CONSTANT_POWER_DURATION  = 120.0
DIAG_CONSTANT_STRESS_DURATION = 120.0
DIAG_DIAGNOSTIC_DURATION      = 300.0

global g_gpus
global g_switches
g_gpus = []
g_switches = []

class Entity(object):
    def __init__(self, entityId, entityType=dcgm_fields.DCGM_FE_GPU, uuid=None, bdf=None):
        self.health = BR_ST_HEALTHY
        self.entityType = entityType
        self.entityId = entityId
        self.reasonsUnhealthy = []
        if uuid:
            self.uuid = uuid
        if bdf:
            self.bdf = bdf

    def IsHealthy(self):
        return self.health == BR_ST_HEALTHY

    def MarkUnhealthy(self, failCondition, reason):
        self.health = self.health | failCondition
        self.reasonsUnhealthy.append(reason)

    def WhyUnhealthy(self):
        return self.reasonsUnhealthy

    def SetEntityId(self, entityId):
        self.entityId = entityId

    def GetEntityId(self):
        return self.entityId

    def GetUUID(self):
        return self.uuid

    def GetBDF(self):
        return self.bdf

def mark_entity_unhealthy(entities, entityId, code, reason):
    found = False
    for entity in entities:
        if entityId == entity.GetEntityId():
            entity.MarkUnhealthy(code, reason)
            found = True

    return found

def addParamString(runDiagInfo, paramIndex, paramStr):
    strIndex = 0
    for c in paramStr:
        runDiagInfo.testParms[paramIndex][strIndex] = c
        strIndex = strIndex + 1


def setTestDurations(runDiagInfo, timePercentage):
    # We only are reducing the test time for the default case
    if runDiagInfo.validate != 3:
        return

    stressDuration = int(DIAG_SM_STRESS_DURATION * timePercentage)
    powerDuration = int(DIAG_CONSTANT_POWER_DURATION * timePercentage)
    constantStressDuration = int(DIAG_CONSTANT_STRESS_DURATION * timePercentage)
    diagDuration = int(DIAG_DIAGNOSTIC_DURATION * timePercentage)

    smParam = "sm stress.test_duration=%d" % (stressDuration)
    powerParam = "targeted power.test_duration=%d" % (powerDuration)
    constantStressParam = "targeted stress.test_duration=%d" % (constantStressDuration)
    diagParam = "diagnostic.test_duration=%d" % (diagDuration)

    addParamString(runDiagInfo, 0, diagParam)
    addParamString(runDiagInfo, 1, smParam)
    addParamString(runDiagInfo, 2, constantStressParam)
    addParamString(runDiagInfo, 3, powerParam)

def initialize_run_diag_info(settings):
    runDiagInfo = dcgm_structs.c_dcgmRunDiag_v7()
    runDiagInfo.version = dcgm_structs.dcgmRunDiag_version7
    runDiagInfo.flags = dcgm_structs.DCGM_RUN_FLAGS_VERBOSE
    testNamesStr = settings['testNames']
    if testNamesStr == '1':
        runDiagInfo.validate = 1
    elif testNamesStr == '2':
        runDiagInfo.validate = 2
    elif testNamesStr == '3':
        runDiagInfo.validate = 3
    else:
        # Make sure no number other that 1-3 were submitted
        if testNamesStr.isdigit():
            raise ValueError("'%s' is not a valid test name" % testNamesStr)

        # Copy to the testNames portion of the object
        names = testNamesStr.split(',')
        testIndex = 0
        if len(names) > dcgm_structs.DCGM_MAX_TEST_NAMES:
            err = 'Aborting DCGM Diag because %d test names were specified exceeding the limit of %d' %\
                  (len(names), dcgm_structs.DCGM_MAX_TEST_NAMES)
            raise ValueError(err)

        for testName in names:
            testNameIndex = 0
            if len(testName) >= dcgm_structs.DCGM_MAX_TEST_NAMES_LEN:
                err = 'Aborting DCGM Diag because test name %s exceeds max length %d' % \
                      (testName, dcgm_structs.DCGM_MAX_TEST_NAMES_LEN)
                raise ValueError(err)

            for c in testName:
                runDiagInfo.testNames[testIndex][testNameIndex] = c
                testNameIndex = testNameIndex + 1

            testIndex = testIndex + 1

    if 'timePercentage' in settings:
        setTestDurations(runDiagInfo, settings['timePercentage'])

    activeGpuIds = []

    first = True
    for gpuObj in g_gpus:
        if gpuObj.IsHealthy():
            activeGpuIds.append(gpuObj.GetEntityId())
            if first:
                runDiagInfo.gpuList = str(gpuObj.GetEntityId())
                first = False
            else:
                to_append = ',%s' % (str(gpuObj.GetEntityId()))
                runDiagInfo.gpuList = runDiagInfo.gpuList + to_append

    return runDiagInfo, activeGpuIds

def mark_all_unhealthy(activeGpuIds, reason):
    for gpuId in activeGpuIds:
        mark_entity_unhealthy(g_gpus, gpuId, BR_ST_FAILED_ACTIVE_HEALTH, reason)

def result_to_str(result):
    if result == dcgm_structs.DCGM_DIAG_RESULT_PASS:
        return 'PASS'
    elif result == dcgm_structs.DCGM_DIAG_RESULT_SKIP:
        return 'SKIP'
    elif result == dcgm_structs.DCGM_DIAG_RESULT_WARN:
        return 'WARN'
    elif result == dcgm_structs.DCGM_DIAG_RESULT_FAIL:
        return 'FAIL'
    else:
        return 'NOT RUN'

def check_passive_health_checks(response, activeGpuIds):
    unhealthy = False
    for i in range(0, dcgm_structs.DCGM_SWTEST_COUNT):
        if response.levelOneResults[i].result == dcgm_structs.DCGM_DIAG_RESULT_FAIL:
            mark_all_unhealthy(activeGpuIds, response.levelOneResults[i].error.msg)
            unhealthy = True
            break

    return unhealthy

def check_gpu_diagnostic(handleObj, settings):
    runDiagInfo, activeGpuIds = initialize_run_diag_info(settings)
    if len(activeGpuIds) == 0:
        return

    response = dcgm_agent.dcgmActionValidate_v2(handleObj.handle, runDiagInfo)

    sysError = response.systemError
    if (sysError.code != dcgm_errors.DCGM_FR_OK):
        raise ValueError(sysError)

    if check_passive_health_checks(response, activeGpuIds) == False:
        for gpuIndex in range(response.gpuCount):
            for testIndex in range(dcgm_structs.DCGM_PER_GPU_TEST_COUNT):
                if response.perGpuResponses[gpuIndex].results[testIndex].result == dcgm_structs.DCGM_DIAG_RESULT_FAIL:
                    gpuId = response.perGpuResponses[gpuIndex].gpuId
                    mark_entity_unhealthy(g_gpus, gpuId, BR_ST_FAILED_ACTIVE_HEALTH,
                                              response.perGpuResponses[gpuIndex].results[testIndex].warning)

                    # NVVS marks all subsequent tests as failed so there's no point in continuing
                    break 


def query_passive_health(handleObj, desired_watches):
    dcgmGroup = handleObj.GetSystem().GetDefaultGroup()
    watches = dcgmGroup.health.Get()

    # Check for the correct watches to be set and set them if necessary
    if watches != desired_watches:
        dcgmGroup.health.Set(desired_watches)

    return dcgmGroup.health.Check()

def blacklist_from_passive_health_check(response):
    for incidentIndex in range(response.incidentCount):
        if response.incidents[incidentIndex].health != dcgm_structs.DCGM_HEALTH_RESULT_FAIL:
            # Only blacklist for failures; ignore warnings
            continue

        entityId = response.incidents[incidentIndex].entityInfo.entityId
        entityGroupId = response.incidents[incidentIndex].entityInfo.entityGroupId
        errorString = response.incidents[incidentIndex].error.msg

        if entityGroupId == dcgm_fields.DCGM_FE_GPU:
            mark_entity_unhealthy(g_gpus, entityId, BR_ST_FAILED_PASSIVE_HEALTH, errorString)
        else:
            mark_entity_unhealthy(g_switches, entityId, BR_ST_FAILED_PASSIVE_HEALTH, errorString)

def check_passive_health(handleObj, watches):
    response = query_passive_health(handleObj, watches)

    if response.overallHealth != dcgm_structs.DCGM_HEALTH_RESULT_PASS:
        blacklist_from_passive_health_check(response)

def initialize_devices(handle, flags):
    gpuIds = dcgm_agent.dcgmGetEntityGroupEntities(handle, dcgm_fields.DCGM_FE_GPU, flags)
    switchIds = dcgm_agent.dcgmGetEntityGroupEntities(handle, dcgm_fields.DCGM_FE_SWITCH, flags)

    i = 0
    for gpuId in gpuIds:
        attributes = dcgm_agent.dcgmGetDeviceAttributes(handle, gpuId)
        gpuObj = Entity(gpuId, entityType=dcgm_fields.DCGM_FE_GPU, uuid=attributes.identifiers.uuid, bdf=attributes.identifiers.pciBusId)
        g_gpus.append(gpuObj)
        i = i + 1

    i = 0
    for switchId in switchIds:
        switchObj = Entity(switchId, entityType=dcgm_fields.DCGM_FE_SWITCH)
        g_switches.append(switchObj)
        i = i + 1

# Process command line arguments
def __process_command_line__(settings):
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--num-gpus', dest='num_gpus', type=int,
                        help='The expected number of GPUs.')
    parser.add_argument('-s', '--num-switches', dest='num_switches', type=int,
                        help='The expected number of NvSwitches.')
    parser.add_argument('-n', '--hostname', dest='hostname', type=str,
                        help='The hostname of the nv-hostengine we want to query.')
    parser.add_argument('-d', '--detect', dest='detect', action='store_true',
                        help='Run on whatever GPUs can be detected. Do not check counts.')
    parser.add_argument('-l', '--log-file', dest='logfileName', type=str,
                        help='The name of the log file where details should be stored. Default is stdout')
    parser.add_argument('-u', '--unsupported-too', dest='unsupported', action='store_true',
                        help='Get unsupported devices in addition to the ones DCGM supports')
    parser.add_argument('-f', '--full-report', dest='fullReport', action='store_true',
                        help='Print a health status for each GPU')
    parser.add_argument('-c', '--csv', dest='outfmtCSV', action='store_true',
                        help='Write output in csv format. By default, output is in json format.')
    parser.add_argument('-w', '--watches', dest='watches', type=str,
                        help='Specify which health watches to monitor. By default, all are watched. Any list of the following may be specified:\n\ta = All watches\n\tp = PCIE\n\tm = Memory\n\ti = Inforom\n\tt = Thermal and Power\n\tn = NVLINK')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('-r', '--specified-test', dest='testNames', type=str,
                       help='Option to specify what tests are run in dcgmi diag.')
    group.add_argument('-i', '--instantaneous', dest='instant', action='store_true',
                       help='Specify to skip the longer tests and run instantaneously')
    group.add_argument('-t', '--time-limit', dest='timeLimit', type=int,
                        help='The time limit in seconds that all the tests should not exceed. Diagnostics will be reduced in their time to meet this boundary.')

    parser.set_defaults(instant=False, detect=False, fullReport=False)
    args = parser.parse_args()

    if args.num_gpus is not None and args.num_switches is not None:
        settings['numGpus'] = args.num_gpus
        settings['numSwitches'] = args.num_switches
    elif args.detect == False:
        raise ValueError('Must specify either a number of gpus and switches with -g and -s or auto-detect with -d')

    if args.hostname:
        settings['hostname'] = args.hostname
    else:
        settings['hostname'] = 'localhost'

    if args.unsupported:
        settings['entity_get_flags'] = 0
    else:
        settings['entity_get_flags'] = dcgm_structs.DCGM_GEGE_FLAG_ONLY_SUPPORTED

    settings['instant'] = args.instant
    settings['fullReport'] = args.fullReport
    
    if args.testNames:
        settings['testNames'] = args.testNames
    else:
        settings['testNames'] = '3'

    if args.timeLimit:
        settings['timePercentage'] = float(args.timeLimit) / 840.0

    if args.logfileName:
        logging.basicConfig(filename=args.logfileName)

    if args.outfmtCSV:
        settings['outfmtCSV'] = 1

    if args.watches:
        health_watches = 0
        for c in args.watches:
            if c == 'p':
                health_watches |= dcgm_structs.DCGM_HEALTH_WATCH_PCIE
            elif c == 'm':
                health_watches |= dcgm_structs.DCGM_HEALTH_WATCH_MEM
            elif c == 'i':
                health_watches |= dcgm_structs.DCGM_HEALTH_WATCH_INFOROM
            elif c == 't':
                health_watches |= dcgm_structs.DCGM_HEALTH_WATCH_THERMAL
                health_watches |= dcgm_structs.DCGM_HEALTH_WATCH_POWER
            elif c == 'n':
                health_watches |= dcgm_structs.DCGM_HEALTH_WATCH_NVLINK
            elif c == 'a':
                health_watches |= dcgm_structs.DCGM_HEALTH_WATCH_ALL
            else:
                print(("Unrecognized character %s found in watch string '%s'" % (c, args.watches)))
                sys.exit(-1)
        settings['watches'] = health_watches
    else:
        settings['watches'] = BR_HEALTH_WATCH_BITMAP

def get_entity_id_list(entities):
    ids = ""
    first = True
    for entity in entities:
        if first:
            ids = str(entity.GetEntityId())
        else:
            ids += ",%d" % (entity.GetEntityId())
        first = False

    return ids

def check_health(handleObj, settings, error_list):
    initialize_devices(handleObj.handle, settings['entity_get_flags'])

    if 'numGpus' in settings:
        if len(g_gpus) != settings['numGpus']:
            error_list.append("%d GPUs were specified but only %d were detected with ids '%s'" % 
                    (settings['numGpus'], len(g_gpus), get_entity_id_list(g_gpus)))

    if 'numSwitches' in settings:
        if len(g_switches) != settings['numSwitches']:
            error_list.append("%d switches were specified but only %d were detected with ids '%s'" %
                    (settings['numSwitches'], len(g_switches), get_entity_id_list(g_switches)))

    check_passive_health(handleObj, settings['watches'])        # quick check

    if settings['instant'] == False:
        check_gpu_diagnostic(handleObj, settings)

def process_command_line(settings):
    try:
        __process_command_line__(settings)
    except ValueError as e:
        return str(e)

def main():
    # Parse the command line
    settings = {}
    error_list = []

    exitCode = 0
    jsonTop = {}

    error = process_command_line(settings)
    if error:
        # If we had an error processing the command line, don't attempt to check anything
        error_list.append(error)
    else:
        try:
            handleObj = pydcgm.DcgmHandle(None, settings['hostname'], dcgm_structs.DCGM_OPERATION_MODE_AUTO)

            check_health(handleObj, settings, error_list)
        except dcgm_structs.DCGMError as e:
            # Catch any exceptions from DCGM and add them to the error_list so they'll be printed as JSON
            error_list.append(str(e))
        except ValueError as e:
            error_list.append(str(e))

        if 'outfmtCSV' in settings:       # show all health, then all un-healthy
            for gpuObj in g_gpus:
                if gpuObj.IsHealthy() == True:
                    print("healthy,%s,%s" %(gpuObj.GetBDF(), gpuObj.GetUUID()))
            for gpuObj in g_gpus:
                if gpuObj.IsHealthy() == False:
                    print("unhealthy,%s,%s,\"%s\"" %(gpuObj.GetBDF(), gpuObj.GetUUID(),gpuObj.WhyUnhealthy()))

        else:              # build obj that can be output in json
            blacklistGpus = {}
            healthyGpus = {}
            for gpuObj in g_gpus:
                if gpuObj.IsHealthy() == False:
                    details = {}
                    details['UUID'] = gpuObj.GetUUID()
                    details['BDF']  = gpuObj.GetBDF()
                    details['Failure Explanation'] = gpuObj.WhyUnhealthy()
                    blacklistGpus[gpuObj.GetEntityId()] = details
                elif settings['fullReport']:
                    details = {}
                    details['UUID'] = gpuObj.GetUUID()
                    details['BDF']  = gpuObj.GetBDF()
                    healthyGpus[gpuObj.GetEntityId()] = details
    
            jsonTop['blacklistedGpus'] = blacklistGpus
            if settings['fullReport']:
                jsonTop['Healthy GPUs'] = healthyGpus

    if len(error_list):         # had error processing the command line
        exitCode = 1
        if 'outfmtCSV' in settings:   # json output 
            if len(error_list):
                for errObj in error_list:
                    print("errors,\"%s\"" %(errObj))
        else:
            jsonTop['errors'] = error_list

    if 'outfmtCSV' in settings:       # show all health, then all un-healthy
        pass
    else:
        print(json.dumps(jsonTop, indent=4, separators=(',', ': ')))

    sys.exit(exitCode)

if __name__ == '__main__':
    main()

