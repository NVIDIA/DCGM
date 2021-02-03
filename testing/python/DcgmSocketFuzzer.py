# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

import errno
import os
import socket
import struct
import random
import time
import dcgm_structs
import multiprocessing

'''
/**
 * DCGM Message Header
 */
typedef struct
{
    int msgId;                   /* Identifier to represent DCGM protocol (DCGM_PROTO_MAGIC) */
    dcgm_request_id_t requestId; /* Represent Message by a request ID */
    int length;                  /* Length of encoded message */
    int msgType;                 /* Type of encoded message. One of DCGM_MSG_? */
    int status;                  /* Status. One of DCGM_PROTO_ST_? */
} dcgm_message_header_t;

/* Base structure for all module commands. Put this structure at the front of your messages */
typedef struct
{
    unsigned int length; /* Total length of this module command. Should be sizeof(yourstruct) that has a header at the
                            front of it */
    dcgmModuleId_t moduleId; /* Which module to dispatch to. See DCGM_MODULE_ID_* #defines in dcgm_structs.h */
    unsigned int subCommand; /* Sub-command number for the module to switch on. These are defined by each module */
    dcgm_connection_id_t
        connectionId;       /* The connectionId that generated this request. This can be used to key per-client info or
                                          as a handle to send async responses back to clients. 0=Internal or don't care */
    unsigned int requestId; /* Unique request ID that the sender is using to uniquely identify
                               this request. This can be 0 if the receiving module or the sender
                               doesn't care. This is a dcgm_request_id_t but is an unsigned int here
                               for sanity */
    unsigned int version;   /* Version of this module sub-command, to be checked by the module */
} dcgm_module_command_header_t;
'''

DCGM_PROTO_MAGIC = 0xadbcbcad

#DCGM Message types
DCGM_MSG_PROTO_REQUEST = 0x0100 # A Google protobuf-based request
DCGM_MSG_PROTO_RESPONSE = 0x0200 # A Google protobuf-based response to a request
DCGM_MSG_MODULE_COMMAND = 0x0300 # A module command message 
DCGM_MSG_POLICY_NOTIFY = 0x0400 # Async notification of a policy violation
DCGM_MSG_REQUEST_NOTIFY = 0x0500 # Notify an async request that it will receive no further updates

allMsgTypes = [
    DCGM_MSG_PROTO_REQUEST, 
    DCGM_MSG_PROTO_RESPONSE, 
    DCGM_MSG_MODULE_COMMAND, 
    DCGM_MSG_POLICY_NOTIFY, 
    DCGM_MSG_REQUEST_NOTIFY
    ]

def getDcgmMessageHeaderBytes(msgId, requestId, length, msgType, status):
    return struct.pack("IIiii", msgId, requestId, length, msgType, status)

def getRandomBytes(length):
    messageBytes = ""
    for i in range(length):
        messageBytes += struct.pack("B", random.randint(0, 255))
    return messageBytes

# Get an entire DCGM packet
def getDcgmMessageBytes(msgId=DCGM_PROTO_MAGIC, requestId=None, length=20, msgType=DCGM_MSG_PROTO_REQUEST, status=0, fillMessageToSize=True):
    if requestId is None:
        requestId = random.randint(0, 2000000000)
    
    messageBytes = getDcgmMessageHeaderBytes(msgId, requestId, length-20, msgType, status)

    if length > 20 and fillMessageToSize :
        messageBytes += getRandomBytes(length-20)

    return messageBytes

# Get a DCGM packet containing a randomized module command
def getDcgmModuleCommandBytes(totalLength=None, moduleId=None, subCommand=None, moduleCommandVersion=None, moduleCommandLength=None):
    msgId = DCGM_PROTO_MAGIC
    requestId = random.randint(0, 2000000000)
    if totalLength is None:
        length = random.randint(24+1, 1000) #20 main header. 24 module header, at least 1 random byte
    else:
        length = totalLength

    #proto header
    messageBytes = getDcgmMessageHeaderBytes(msgId, requestId, length, DCGM_MSG_MODULE_COMMAND, 0)

    #Module command header
    if moduleId is None:
        moduleId = random.randint(0, dcgm_structs.DcgmModuleIdCount) #+1 so we send invalid values
    if subCommand is None:
        subCommand = random.randint(0,20) #Should cover most subcommands
    connectionId = random.randint(0,1000000000)
    if moduleCommandVersion is None:
        verNum = random.randint(0, 5)
        moduleCommandVersion = length | (verNum << 24)
    #The only reason to override moduleCommandLength is to send malformed packets
    if moduleCommandLength is None:
        moduleCommandLength = length
    messageBytes += struct.pack("IIIIII", moduleCommandLength, moduleId, subCommand, connectionId, requestId, moduleCommandVersion)

    #Random body
    #print("size %d : %s" % (len(messageBytes), str(messageBytes)))
    messageBytes += getRandomBytes(length-24)
    return messageBytes

def getSocketObj():
    socketObj = socket.create_connection(('127.0.0.1', 5555))
    return socketObj

def readAllFromSocketObj(socketObj):
    '''
    Read all available bytes from the given socket object so the socket doesn't back up on the
    host engine side
    '''
    while True:
        try:
            buf = socketObj.recv(65536, socket.MSG_DONTWAIT)
        except socket.error as e:
            if e.errno != errno.EAGAIN and e.errno != errno.ECONNRESET:
                raise Exception("Got unexpected errno: %s (%d)" % (os.strerror(e.errno), e.errno))
            else:
                return
        
        #print ("Read %d bytes." % len(buf))

def testCompleteGarbage():
    print("Sending random garbage and expecting a hang-up")
    numSocketErrors = 0
    numIterations = 50
    for i in range(0, numIterations):
        socketObj = getSocketObj()
        numBytesSent = 0
        while True:
            data = struct.pack("B", random.randint(0, 255))
            try:
                socketObj.sendall(data)
                readAllFromSocketObj(socketObj)
            except socket.error as serr:
                numSocketErrors += 1
                break
            numBytesSent += 1
        
    print("Got %d socket errors after %d connections" % (numSocketErrors, numIterations))

def testMessageContentsGarbage():
    print("Sending message body as random garbage expecting a hang-up")
    duration = 60.0
    startTime = time.time()
    while time.time() - startTime < duration:
        socketObj = getSocketObj()
        numBytesSent = 0
        numPacketsSent = 0
        while time.time() - startTime < duration:
            messageLength = random.randint(21, 1000)
            msgType = random.randint(0, len(allMsgTypes))
            data = getDcgmMessageBytes(length=messageLength, msgType=msgType)
            try:
                socketObj.sendall(data)
                readAllFromSocketObj(socketObj)
            except socket.error as serr:
                print("Got expected socket error after %d packets (%d bytes) sent" % (numPacketsSent, numBytesSent))
                break
            numPacketsSent += 1
            numBytesSent += messageLength

            if numPacketsSent % 1000 == 0:
                print("Sent %d packets. %u bytes so far." % (numPacketsSent, numBytesSent))

def testModuleCommandGarbage():
    print("Sending module commands with corrupt bodies")
    duration = 60.0
    startTime = time.time()
    while time.time() - startTime < duration:
        socketObj = getSocketObj()
        numBytesSent = 0
        numPacketsSent = 0
        while time.time() - startTime < duration:
            data = getDcgmModuleCommandBytes()
            try:
                socketObj.sendall(data)
                readAllFromSocketObj(socketObj)
            except socket.error as serr:
                print("Got expected socket error after %d packets (%d bytes) sent" % (numPacketsSent, numBytesSent))
                break
            numPacketsSent += 1
            numBytesSent += len(data)

            if numPacketsSent % 1000 == 0:
                print("Sent %d packets. %u bytes so far." % (numPacketsSent, numBytesSent))

def testModuleCommandBodyGarbage():
    print("Sending correct sized module commands with corrupt bodies")
    duration = 60.0
    startTime = time.time()

    # [totalLength, moduleId, subCommandId, moduleCommandVersion]
    moduleCommands = [
        #Diag
        [0x7e580, dcgm_structs.DcgmModuleIdDiag, 1, 0x407e580], #DCGM_DIAG_SR_RUN
        [0x100, dcgm_structs.DcgmModuleIdDiag,   2, 0x10000100], #DCGM_DIAG_SR_STOP
        #Config
        [0x6b0, dcgm_structs.DcgmModuleIdConfig, 1, 0x10006b0], #DCGM_CONFIG_SR_GET
        [0x1d0, dcgm_structs.DcgmModuleIdConfig, 2, 0x10001d0], #DCGM_CONFIG_SR_SET
        [0x1a8, dcgm_structs.DcgmModuleIdConfig, 3, 0x10001a8], #DCGM_CONFIG_SR_ENFORCE_GROUP
        [0x1a0, dcgm_structs.DcgmModuleIdConfig, 4, 0x10001a0], #DCGM_CONFIG_SR_ENFORCE_GPU
        #Health
        [0x28,  dcgm_structs.DcgmModuleIdHealth, 1, 0x1000028], #DCGM_HEALTH_SR_GET_SYSTEMS
        [0x105c0, dcgm_structs.DcgmModuleIdHealth, 5, 0x10105c0], #DCGM_HEALTH_SR_CHECK_GPUS
        [0x10540, dcgm_structs.DcgmModuleIdHealth, 7, 0x4010540], #DCGM_HEALTH_SR_CHECK_V4
        [0x40,  dcgm_structs.DcgmModuleIdHealth, 8, 0x1000040], #DCGM_HEALTH_SR_SET_SYSTEMS_V2
        #Policy
        [0x1228, dcgm_structs.DcgmModuleIdPolicy, 1, 0x2001228], #DCGM_POLICY_SR_GET_POLICIES
        [0xb0, dcgm_structs.DcgmModuleIdPolicy, 2, 0x10000b0], #DCGM_POLICY_SR_SET_POLICY
        [0x28,  dcgm_structs.DcgmModuleIdPolicy, 3, 0x1000028], #DCGM_POLICY_SR_REGISTER
        [0x28, dcgm_structs.DcgmModuleIdPolicy, 4, 0x1000028], #DCGM_POLICY_SR_UNREGISTER
        #Profiling
        [0x120, dcgm_structs.DcgmModuleIdProfiling, 1, 0x1000120], #DCGM_PROFILING_SR_GET_MGS
        [0x68,  dcgm_structs.DcgmModuleIdProfiling, 2, 0x1000068], #DCGM_PROFILING_SR_WATCH_FIELDS
        [0x30, dcgm_structs.DcgmModuleIdProfiling, 3,  0x1000030], #DCGM_PROFILING_SR_UNWATCH_FIELDS
        [0x1c, dcgm_structs.DcgmModuleIdProfiling, 4,  0x100001c], #DCGM_PROFILING_SR_PAUSE_RESUME
        #NvSwitch
        [0x58,  dcgm_structs.DcgmModuleIdNvSwitch, 1, 0x1000058], #DCGM_NVSWITCH_SR_GET_SWITCH_IDS
        [0x50,  dcgm_structs.DcgmModuleIdNvSwitch, 2, 0x1000050], #DCGM_NVSWITCH_SR_CREATE_FAKE_SWITCH
        [0x68,  dcgm_structs.DcgmModuleIdNvSwitch, 3, 0x1000068], #DCGM_NVSWITCH_SR_WATCH_FIELD
        [0x20,  dcgm_structs.DcgmModuleIdNvSwitch, 4, 0x1000020], #DCGM_NVSWITCH_SR_UNWATCH_FIELD
        [0xac,  dcgm_structs.DcgmModuleIdNvSwitch, 5, 0x10000ac], #DCGM_NVSWITCH_SR_GET_LINK_STATES
        [0xd98, dcgm_structs.DcgmModuleIdNvSwitch, 6, 0x1000d98], #DCGM_NVSWITCH_SR_GET_ALL_LINK_STATES
        [0x24,  dcgm_structs.DcgmModuleIdNvSwitch, 7, 0x1000024], #DCGM_NVSWITCH_SR_SET_LINK_STATE
        [0x20,  dcgm_structs.DcgmModuleIdNvSwitch, 8, 0x1000020], #DCGM_NVSWITCH_SR_GET_ENTITY_STATUS
        #vGPU
        [0x1c, dcgm_structs.DcgmModuleIdVGPU, 1, 0x100001c], #DCGM_VGPU_SR_START
        [0x1c, dcgm_structs.DcgmModuleIdVGPU, 2, 0x100001c], #DCGM_VGPU_SR_SHUTDOWN
        #Introspect
        [0x1c,  dcgm_structs.DcgmModuleIdIntrospect, 1, 0x100001c], #DCGM_INTROSPECT_SR_STATE_TOGGLE
        [0x1a0, dcgm_structs.DcgmModuleIdIntrospect, 2, 0x10001a0], #DCGM_INTROSPECT_SR_STATE_SET_RUN_INTERVAL
        [0x1a0, dcgm_structs.DcgmModuleIdIntrospect, 3, 0x10001a0], #DCGM_INTROSPECT_SR_UPDATE_ALL
        [0x1a0, dcgm_structs.DcgmModuleIdIntrospect, 4, 0x10001a0], #DCGM_INTROSPECT_SR_HOSTENGINE_MEM_USAGE
        [0x1a0, dcgm_structs.DcgmModuleIdIntrospect, 5, 0x10001a0], #DCGM_INTROSPECT_SR_HOSTENGINE_CPU_UTIL
        [0x1a0, dcgm_structs.DcgmModuleIdIntrospect, 6, 0x10001a0], #DCGM_INTROSPECT_SR_FIELDS_MEM_USAGE
        [0x1a0, dcgm_structs.DcgmModuleIdIntrospect, 7, 0x10001a0], #DCGM_INTROSPECT_SR_FIELDS_EXEC_TIME
    ]

    while time.time() - startTime < duration:
        socketObj = getSocketObj()
        numBytesSent = 0
        numPacketsSent = 0
        numPacketsWithBadMcLength = 0
        while time.time() - startTime < duration:
            commandIndex = random.randint(0, len(moduleCommands)-1)
            mc = moduleCommands[commandIndex]
            moduleCommandLength = None
            #5% of the time, throw in a bogus module command length
            if random.randint(0,100) < 5:
                moduleCommandLength = random.randint(1, 1000000000)
                numPacketsWithBadMcLength += 1

            data = getDcgmModuleCommandBytes(totalLength=mc[0], moduleId=mc[1], subCommand=mc[2], moduleCommandVersion=mc[3], moduleCommandLength=moduleCommandLength)
            try:
                socketObj.sendall(data)
                readAllFromSocketObj(socketObj)
            except socket.error as serr:
                print("Got expected socket error after %d packets (%d bytes) sent" % (numPacketsSent, numBytesSent))
                break
            numPacketsSent += 1
            numBytesSent += len(data)

            if numPacketsSent % 1000 == 0:
                print("Sent %d packets. %u bytes so far. Sent with bad MC length: %d" % (numPacketsSent, numBytesSent, numPacketsWithBadMcLength))

def testGiantAndNegativeMessages():
    print("Sending huge and negative message lengths")
    
    duration = 20.0
    startTime = time.time()

    numSocketErrors = 0
    i = 0
    while time.time() - startTime < duration:
        i += 1
        socketObj = getSocketObj()
        numBytesSent = 0
        numPacketsSent = 0
        while True:
            if i & 1:
                #Humungous
                messageLength = random.randint(1000000000, 2000000000)
            else:
                #Negative size
                messageLength = random.randint(-1000000000, -1)
            data = getDcgmMessageBytes(length=messageLength, fillMessageToSize=False)
            try:
                socketObj.sendall(data)
                readAllFromSocketObj(socketObj)
            except socket.error as serr:
                numSocketErrors += 1
                break
            numPacketsSent += 1
        if i % 1000 == 0:
            print("Got %d socket errors after %d packets (%d bytes) sent" % (numSocketErrors, numPacketsSent, numBytesSent))

def workerMain(workerNumber):
    #Seed our RNG with our worker number
    random.seed(workerNumber)
    
    try:
        print("Worker %d started" % workerNumber)
        while True:
            testGiantAndNegativeMessages()
            testCompleteGarbage()
            testModuleCommandGarbage()
            testMessageContentsGarbage()
            testModuleCommandBodyGarbage()
    except KeyboardInterrupt:
        print("Worker %d got a CTRL-C. Exiting." % workerNumber)
        return

if __name__ == '__main__':
    numCores = multiprocessing.cpu_count()
    numWorkers = numCores - 2 #Give DCGM and the system a little breathing room
    print("Found %d cores. Spawning %d worker processes." % (numCores, numWorkers))
    print("This script cycles through tests forever. Press CTRL-C to abort.")
    if numWorkers <= 1:
        #Just call workerMain if we have a single worker
        workerMain(0)
    else:
        try:
            pool = multiprocessing.Pool(numWorkers)
            pool.map(workerMain, range(0, numWorkers))
        except KeyboardInterrupt:
            print("Main process got CTRL-C. Exiting.")
            pass
        finally:
            pool.join()

    
