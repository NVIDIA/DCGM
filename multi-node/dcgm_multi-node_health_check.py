#!/usr/bin/env python3

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

import argparse
import json
import os
import multiprocessing
import psutil
import socket
import subprocess
import sys
import time

import dcgm_fields
import dcgm_structs
import pydcgm
import DcgmDiag
import re

RESERVED_SYSTEM = "_system_"

OUT_OF_PLACE = 0
IN_PLACE = 1

g_dcgmHandle = None

'''
    Json output example
{
    "HealthCheck" : {
        "result" : PASS / FAIL,
        "system" : [ "error1", "error2", ... ],
        "nccl_tests" : {
            "result" : PASS / FAIL,
            "errors" : [ "error1", "error2", ... ],
        },
        "DCGMDiagnostic" : {
            "result" : PASS / FAIL
            "global_errors" : "timeout msg",
            "host_results" : [
                {
                    "hostname" : "name",
                    "result" : PASS / FAIL,
                    "errors" : [
                        "GPU" : id,
                        "test_name" : "name",
                        "error_msg" : "msg",
                    ]
                },
                ...
            ]
        },
    }
}
'''

JS_FAIL          = "FAIL"
JS_PASS          = "PASS"
JS_RESULT        = "result"
JS_ERRORS        = "errors"
JS_HOSTNAME      = "hostname"
JS_GLOBAL_ERRORS = "global_errors"
def build_and_print_json(nccl_errors, system_errors, diag_error_map, node_list):
    top          = {}
    health_check = {}
    nccl_tests   = {}
    diag         = {}

    errorCount = 0

    errorCount = errorCount + len(system_errors)
    health_check["system_errors"] = system_errors

    errorCount = errorCount + len(nccl_errors)
    nccl_tests[JS_ERRORS] = nccl_errors
    if len(nccl_errors):
        nccl_tests[JS_RESULT] = JS_FAIL
    else:
        nccl_tests[JS_RESULT] = JS_PASS

    health_check["nccl_tests"] = nccl_tests

    hosts_accounted_for = []
    diag_host_results = []
    diagFailed = False
    for hostname in diag_error_map:
        if hostname == RESERVED_SYSTEM:
            # Process diag system errors
            diag[JS_GLOBAL_ERRORS] = []
            for error in diag_error_map[hostname]:
                diag[JS_GLOBAL_ERRORS].append(error.GetErrorMsg())
        else:
            diag_host_entry = {}
            diag_host_entry[JS_HOSTNAME] = hostname
            hosts_accounted_for.append(hostname)
                        
            diag_entry_errors = []
            for error in diag_error_map[hostname]:
                diagFailed = True
                gpuId      = error.GetGpuId()
                errmsg     = error.GetErrorMsg()
                testname   = error.GetTestName()

                error_entry = {}
                if gpuId and gpuId >= 0:
                    error_entry["GPU"] = gpuId
                else:
                    error_entry["GPU"] = -1

                error_entry["test_name"] = testname
                error_entry["error_msg"] = errmsg
                diag_entry_errors.append(error_entry)
            diag_host_entry[JS_ERRORS] = diag_entry_errors

            if len(diag_entry_errors):
                diag_host_entry[JS_RESULT] = JS_FAIL
            else:
                diag_host_entry[JS_RESULT] = JS_PASS
            diag_host_results.append(diag_host_entry)

            errorCount = errorCount + len(diag_entry_errors)

    for node in node_list:
        hostname = node.hostname
        if hostname not in hosts_accounted_for:
            # Add a passing entry for this host
            diag_host_entry = {}
            diag_host_entry[JS_HOSTNAME] = hostname
            diag_host_entry[JS_RESULT]   = JS_PASS
            diag_host_entry[JS_ERRORS]   = []
            diag_host_results.append(diag_host_entry)

    diag["host_results"] = diag_host_results

    if diagFailed:
        diag[JS_RESULT] = JS_FAIL
    else:
        diag[JS_RESULT] = JS_PASS
    if JS_GLOBAL_ERRORS not in diag:
        diag[JS_GLOBAL_ERRORS] = []
    health_check["DCGMDiagnostic"] = diag

    if errorCount > 0:
        health_check[JS_RESULT] = JS_FAIL
    else:
        health_check[JS_RESULT] = JS_PASS

    top["HealthCheck"] = health_check

    output = json.dumps(top, indent=4)
    print(output)

    return errorCount
        
def basic_print_errors(nccl_errors, system_errors, diag_error_map):
    errorCount = 0
    return errorCount

'''
    Error container object
'''
HC_TEST_TYPE_NCCL = 0
HC_TEST_TYPE_DIAG = 1
HC_TEST_TYPE_SYS  = 2
class HCError(object):
    def __init__(self, testType, errorMsg, hostname=None, gpuId=None, testName=None):
        self.testType = testType

        if not errorMsg:
            raise ValueError("Errors must include an error message")
        self.hostname = hostname
        self.gpuId    = gpuId
        self.testName = testName
        self.errorMsg = errorMsg

    def GetLongError(self):
        if self.testType == HC_TEST_TYPE_SYS:
            return "Health Check, %s" % self.errorMsg
        elif self.testType == HC_TEST_TYPE_DIAG:
            if not self.hostname:
                return "DCGM Diagnostics, System Error, %s" % self.errorMsg
            elif not self.testName:
                return "DCGM Diagnostics, %s, SystemError, %s" % (self.hostname, self.errorMsg)
            elif self.gpuId and self.gpuId != -1:
                return "DCGM Diagnostics, %s, %s, GPU %d, %s" % \
            (self.hostname, self.testName, self.gpuId, self.errorMsg)
            else:
                return "DCGM Diagnostics, %s, %s, %s" % (self.hostname, self.testName, self.errorMsg)
        else:
            return "nccl_tests: %s" % self.errorMsg

    def GetGpuId(self):
        return self.gpuId

    def GetTestName(self):
        return self.testName

    def GetHostname(self):
        return self.hostname

    def GetErrorMsg(self):
        return self.errorMsg

def getInt(intStr):
    if intStr == 'N/A':
        return 0

    return int(intStr)

def getFloat(floatStr):
    if floatStr == 'N/A':
        return 0

    return float(floatStr)

def parse_nccl_output(raw_output, minBandwidthRequired, minAlgBandwidthRequired):
    errors = []
    lines = raw_output.split('\n')
    output_lines = []
    comments = []
    for line in lines:
        stripped = line.strip()
        if line != '':
            if line[0] == '#':
                comments.append(stripped)
            else:
                output_lines.append(stripped)

    
    maxAlgBW = [0, 0]
    maxBusBW = [0, 0]
    numWrong = [0, 0]

    '''
    Each line in the output will have 13 tokens; Here's what each token / position is:
    size  = 0
    count = 1
    type  = 2
    redop = 3
    root = 4
    out of place
    time = 5
    algbw = 6
    busbw = 7
    Number wrong = 8
    in place
    time = 9
    algbw = 10
    busbw = 11
    Number wrong = 12
    '''
    
    for line in output_lines:
        tokens = line.split() # split on whitespace

        if len(tokens) == 13:
            # Correct number of elements
            maxAlgBW[OUT_OF_PLACE] = max(maxAlgBW[OUT_OF_PLACE], getFloat(tokens[6]))
            maxAlgBW[IN_PLACE] = max(maxAlgBW[IN_PLACE], getFloat(tokens[10]))
            maxBusBW[OUT_OF_PLACE] = max(maxBusBW[OUT_OF_PLACE], getFloat(tokens[7]))
            maxBusBW[IN_PLACE] = max(maxBusBW[IN_PLACE], getFloat(tokens[11]))
            err1 = getInt(tokens[8])
            numWrong[OUT_OF_PLACE] = numWrong[OUT_OF_PLACE] + err1
            err2 = getInt(tokens[12])
            numWrong[IN_PLACE] = numWrong[IN_PLACE] + err2 
            if err1 > 0:
                errors.append("Found %d errors with size %s out of place" % (err1, tokens[0]))
            if err2 > 0: 
                errors.append("Found %d errors with size %s in place" % (err2, tokens[0]))

    if numWrong[OUT_OF_PLACE] > 0:
        errors.append("%d total errors found while performing out of place tests." % numWrong[OUT_OF_PLACE])
    if numWrong[IN_PLACE] > 0:
        errors.append("%d total errors found while performing in place tests." % numWrong[IN_PLACE])

    if maxAlgBW[OUT_OF_PLACE] < minAlgBandwidthRequired:
        errors.append("Maximum achieved out of place alg bandwidth %f is less than required minimum %d" % \
                (maxAlgBW[OUT_OF_PLACE], minAlgBandwidthRequired))
    if maxAlgBW[IN_PLACE] < minAlgBandwidthRequired:
        errors.append("Maximum achieved in place alg bandwidth %f is less than required minimum %d" % \
                (maxAlgBW[IN_PLACE], minAlgBandwidthRequired))

    if maxBusBW[OUT_OF_PLACE] < minBandwidthRequired:
        errors.append("Maximum achieved out of place bus bandwidth %f is less than required minimum %d" % \
                (maxBusBW[OUT_OF_PLACE], minBandwidthRequired))
    if maxBusBW[IN_PLACE] < minBandwidthRequired:
        errors.append("Maximum achieved in place bus bandwidth %f is less than required minimum %d" % \
                (maxBusBW[IN_PLACE], minBandwidthRequired))

    return errors

OPERATION_ALL_REDUCE = 0
OPERATION_ALL_TO_ALL = 1

# These arguments are used for every nccl_tests command
NCCL_ARGS = "-b8 -e16G -f2"

def get_node_counts(node_list):
    node_counts = {}
    for node in node_list:
        if node.hostname in node_counts:
            node_counts[node.hostname] = node_counts[node] + len(node.gpuIds)
        else:
            node_counts[node.hostname] = len(node.gpuIds)

    return node_counts

def get_binary_name(args, operation):
    prefix = ""
    if args.ncclPath:
        prefix = "%s/" % args.ncclPath

    if operation == OPERATION_ALL_REDUCE:
        return '%sall_reduce_perf' % prefix
    elif operation == OPERATION_ALL_TO_ALL:
        return '%salltoall_perf' % prefix

    raise ValueError('Invalid operation: %s' % str(operation))

def get_network_interface_name():
    # Bash command to get interface: `$(ip route get 8.8.8.8 | sed -E 's/.*?dev (\S+) .*/\1/;t;d')`
    dummyIpAddr = '8.8.8.8'
    raw_output = subprocess.check_output(['ip', '-o', 'route', 'get', dummyIpAddr], universal_newlines=True)
    output = raw_output.split()
    if len(output) < 10 or output[0] != dummyIpAddr:
        raise ValueError("Couldn't find the IP address! ip -o route get %s returned '%s'. Please set the network interface name for MPI communications using --network-interface." % (dummyIpAddr, raw_output))
    return output[4]

def create_nccl_command(args, node_list, operation):
    node_counts = get_node_counts(node_list)

    if len(node_counts) > 1:
        # Multi-node job
        total_np = 0
        nodes_str = ''
        for node in node_counts:
            if nodes_str == '':
                nodes_str = "%s:%d" % (node, node_counts[node])
            else:
                tmp = "%s,%s:%d" % (nodes_str, node, node_counts[node])
                nodes_str = tmp
            total_np = total_np + node_counts[node]

        cmd = "mpirun -np %d --oversubscribe --bind-to numa -H %s -x NCCL_DEBUG=WARN --mca btl_tcp_if_include %s %s %s -g1" % \
               (total_np, nodes_str, args.networkInterface, get_binary_name(args, operation), NCCL_ARGS)
        return cmd
    else:
        # Single-node job
        cmd = '%s %s' % (get_binary_name(args, operation), NCCL_ARGS)
        gpuCount = 0
        if args.gpuIds == 'all':
            global g_dcgmHandle
            if g_dcgmHandle == None:
                g_dcgmHandle = pydcgm.DcgmHandle(None, args.hostname, dcgm_structs.DCGM_OPERATION_MODE_AUTO)

            dcgmSystem = g_dcgmHandle.GetSystem()
            gpuIds = dcgmSystem.discovery.GetAllSupportedGpuIds()
            gpuCount = len(gpuIds)

        else:
            gpuIds = args.gpuIds.split(',')
            gpuCount = len(gpuIds)
            for gpuId in gpuIds:
                # Make sure we have only integers
                try:
                    if int(gpuId) < 0:
                        print("The GPU ids argument must be positive integers, but found %s" % gpuId)
                        sys.exit(1)
                except ValueError:
                    print("The GPU ids argument must be a comma-separated list of integers, but found %s" % gpuId)
                    sys.exit(1)
            os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuIds
        return "%s -g%d" % (cmd, gpuCount)

def get_nccl_commands(args, node_list):
    return [ create_nccl_command(args, node_list, OPERATION_ALL_REDUCE),
             create_nccl_command(args, node_list, OPERATION_ALL_TO_ALL) ]

def launch_nccl_tests(args, node_list):
    cmds = get_nccl_commands(args, node_list)
    errors = []
    for command in cmds:
        try:
            runner = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            (bOutput, bError) = runner.communicate()
            output = bOutput.decode("utf-8")
            error = bError.decode("utf-8")
            if len(error):
                print("Stderr: %s" % error)

            errors.extend(parse_nccl_output(output, args.minBandwidth, args.minAlgBandwidth))
        except FileNotFoundError as e:
            errors.append("Cannot execute nccl. Command %s failed because the binary doesn't exist at that path" \
                % command)
            return errors, True
        except PermissionError as e:
            errors.append("Cannot execute nccl. Command %s failed because the binary isn't executable." \
                % command)
            return errors, True

    return errors, False

def get_test_name(index):
    # This is valid only for diagResponse_v10 and earlier
    if index == dcgm_structs.DCGM_MEMORY_INDEX:
        return "Memory"
    elif index == dcgm_structs.DCGM_DIAGNOSTIC_INDEX:
        return "Diagnostic"
    elif index == dcgm_structs.DCGM_PCI_INDEX:
        return "PCIe"
    elif index == dcgm_structs.DCGM_SM_STRESS_INDEX:
        return "SM Stress"
    elif index == dcgm_structs.DCGM_TARGETED_STRESS_INDEX:
        return "Targeted Stress"
    elif index == dcgm_structs.DCGM_TARGETED_POWER_INDEX:
        return "Targeted Power"
    elif index == dcgm_structs.DCGM_MEMORY_BANDWIDTH_INDEX:
        return "Memory Bandwidth"
    elif index == dcgm_structs.DCGM_MEMTEST_INDEX:
        return "Memtest"
    elif index == dcgm_structs.DCGM_PULSE_TEST_INDEX:
        return "Pulse Test"
    elif index == dcgm_structs.DCGM_EUD_TEST_INDEX:
        return "EUD"
    else:
        return "Unknown %d" % index

def inspect_response_for_errors(response, errors, hostname):
    """Collect any errors from all plugins, including system errors"""
    for errIdx in range(response.numErrors):
        curErr = response.errors[errIdx]
        if curErr.testId == dcgm_structs.DCGM_DIAG_RESPONSE_SYSTEM_ERROR:
            testName = "System"
            errors.append(HCError(HC_TEST_TYPE_DIAG,
                                  curErr.msg,
                                  hostname=hostname,
                                  testName=testName))
        elif curErr.entity.entityGroupId == dcgm_fields.DCGM_FE_GPU:
            testName = response.tests[curErr.testId].name
            errors.append(HCError(HC_TEST_TYPE_DIAG,
                                  curErr.msg,
                                  hostname=hostname,
                                  gpuId=curErr.entity.entityId,
                                  testName=testName))
        else:
            testName = response.tests[curErr.testId].name
            errors.append(HCError(HC_TEST_TYPE_DIAG,
                                  curErr.msg,
                                  hostname=hostname,
                                  testName=testName))
    return

def run_diagnostic_on_host(args, hostname, gpuIds, queue):
    dcgmHandle = pydcgm.DcgmHandle(None, hostname, dcgm_structs.DCGM_OPERATION_MODE_AUTO)
    dd = DcgmDiag.DcgmDiag(gpuIds=gpuIds, testNamesStr=args.testNames, paramsStr=args.diagParameters)
    errors = []
    try:
        response = dd.Execute(dcgmHandle.handle)
        inspect_response_for_errors(response, errors, hostname)
    except dcgm_structs.DCGMError as de:
        errors.append(HCError(HC_TEST_TYPE_DIAG, str(de), hostname=hostname))

    queue.put(errors)

def launch_dcgm_diagnostics(args, node_list):
    diag_error_map = {}
    queue = multiprocessing.Queue()
    processes = []
    for node in node_list:
        process = multiprocessing.Process(target=run_diagnostic_on_host, args=(args, node.hostname, node.gpuIds, queue))
        process.start()
        processes.append(process)

    total_timeout = args.timeout

    for process in processes:
        # NOTE: these are not guaranteed to be in order, but we just need the correct count
        if total_timeout == None:
            # No timeout specified
            process_errors = queue.get()
            if len(process_errors):
                diag_error_map[process_errors[0].GetHostname()] = process_errors
        else:
            try:
                st = time.time()
                process_errors = queue.get(timeout=total_timeout)
                et = time.time()
                total_timeout = total_timeout - (et - st)
                if len(process_errors):
                    diag_error_map[process_errors[0].GetHostname()] = process_errors
            except TimeoutError as te:
                error_msg  = "Couldn't execute the diagnostic because we timed out: %s" % str(te)

                diag_error_map[RESERVED_SYSTEM] = [ HCError(HC_TEST_TYPE_DIAG, error_msg) ]
                # We have timed out. Do not continue.
                return diag_error_map

    for process in processes:
        if total_timeout == None:
            process.join()
        else:
            try:
                st = time.time()
                process.join(total_timeout)
                et = time.time()
                total_timeout = total_timeout - (et - st)
            except TimeoutError as te:
                error_msg  = "Timed out while waiting for diagnostic processes to end: %s" % str(te)

                diag_error_map[RESERVED_SYSTEM] = [ HCError(HC_TEST_TYPE_DIAG, error_msg) ]
                # We have timed out. Do not continue.
                return diag_error_map

    return diag_error_map


def is_valid_hostname(hostname):
    if hostname[-1] == ".":
        # strip exactly one dot from the right, if present
        hostname = hostname[:-1]
    if len(hostname) > 253:
        return False

    ipaddr_check = re.compile(r"\d\d?\d?\.\d\d?\d?\.\d\d?\d?\.\d\d?\d?")
    if ipaddr_check.match(hostname):
        return True

    labels = hostname.split(".")

    # the TLD must be not all-numeric
    if re.match(r"[0-9]+$", labels[-1]):
        return False

    hostname_check = re.compile(r"(?!-)[a-z0-9-]{1,63}(?<!-)$", re.IGNORECASE)
    return all(hostname_check.match(label) for label in labels)

def parse_gpu_ids(gpuIdStr):
    gpuIds = []
    comma_tokens = gpuIdStr.split(',')
    for comma_token in comma_tokens:
        range_tokens = comma_token.split('-')
        if len(range_tokens) == 2:
            upper = int(range_tokens[1])
            lower = int(range_tokens[0])
            for i in range(lower, upper+1):
                gpuIds.append(i)
        elif len(range_tokens) == 1:
            gpuIds.append(int(comma_token))
    return gpuIds
            
class Node(object):
    def __init__(self, hostname, gpuIds=[0]):
        self.hostname = hostname
        self.gpuIds = gpuIds

def get_node_list(args):
    param_node_list = None
    arg_value = None
    file_contents = None
    if args.node_list:
        arg_value = args.node_list
        param_node_list = args.node_list.split(',')
    elif args.nodefile:
        original_arg = '--nodes-file'
        with open(args.nodefile, 'r') as nodefile:
            raw_data = nodefile.read()
            arg_value = args.nodefile
            file_contents = raw_data
            param_node_list = raw_data.split()
    else:
        param_node_list = [ args.hostname ]
        arg_value = args.hostname

    node_list = []

    # Raise an error if an invalid hostname was specified.
    for param_node in param_node_list:
        tokens = param_node.split(':')
        if len(tokens) == 2:
            node_list.append(Node(tokens[0], parse_gpu_ids(tokens[1])))
        elif len(tokens) == 1:
            node_list.append(Node(tokens[0]))
        else:
            raise ValueError("Invalid hostname:gpu_list specification '%s'" % param_node)

        if not is_valid_hostname(node_list[-1].hostname):
            if file_contents:
                raise ValueError("Found invalid hostname '%s' in the specified file '%s' with contents:\n %s" \
                        % (node_list[-1].hostname, arg_value, file_contents))
            else:
                raise ValueError("Found invalid hostname '%s' in the specified node list '%s'" \
                        % (node_list[-1].hostname, arg_value))

    return node_list

def process_command_line():
    parser = argparse.ArgumentParser()
    general_group = parser.add_argument_group('Main Group', 'Arguments that are all acceptable together')
    # Specify the minimum bandwith required for nccl-tests' busbw output. Lower bandwidth is considered a failure.
    # Default is 0
    general_group.add_argument('-b', '--min-bandwidth', dest='minBandwidth', type=int, default=0)
    # Specify the minimum bandwidth required for nccl-tests' algbw output. Lower bandwidth is considered a failure.
    # Default is 0
    general_group.add_argument('-a', '--min-alg-bandwidth', dest='minAlgBandwidth', type=int, default=0)
    # Specify the test names / run mode to run for the diagnostic, this is passed to the -r parameter. Default is 3.
    general_group.add_argument('--test-names', dest='testNames', type=str, default='3')
    # Specify the parameters to pass to the DCGM diagnostic. Default is no parameters.
    general_group.add_argument('-p', '--diag-parameters', dest='diagParameters', type=str, default='')
    # Specify the list of GPU ids to use for the nccl-tests and diagnostic, default is all
    general_group.add_argument('-i', '--gpu-ids', dest='gpuIds', type=str, default='all')
    general_group.add_argument('-t', '--timeout', dest='timeout', type=int)
    general_group.add_argument('--nccl-bin-path', dest='ncclPath', type=str)
    general_group.add_argument('-j', '--json', dest='json', type=bool, default=True)
    general_group.add_argument('--network-interface', dest='networkInterface', default=get_network_interface_name())

    me_group = parser.add_mutually_exclusive_group(required=True)
    # Specify the hostname for the nv-hostengine connection. Default is None, resulting in an embedded hostengine.
    me_group.add_argument('-s', '--hostname', dest='hostname', type=str)
    me_group.add_argument('-n', '--node-list', dest='node_list', type=str)
    me_group.add_argument('-f', '--nodes-file', dest='nodefile', type=str)
    args = parser.parse_args()

    return args

def main():
    args = process_command_line()

    node_list = get_node_list(args)

    nccl_errors, abort = launch_nccl_tests(args, node_list)
    system_errors = []

    if not abort:
        diag_error_map = launch_dcgm_diagnostics(args, node_list)
    else:
        system_errors.append("Skipping the diagnostic because nccl tests couldn't be executed.")

    errorCount = 0
    if args.json:
        errorCount = build_and_print_json(nccl_errors, system_errors, diag_error_map, node_list)
    else:
        errorCount = basic_print_errors(nccl_errors, system_errors, diag_error_map)
        if errorCount == 0:
            print("No errors detected during tests")

    if errorCount > 0:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == '__main__':
    main()
