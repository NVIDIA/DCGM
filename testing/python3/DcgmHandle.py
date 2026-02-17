# Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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
import dcgm_structs
import dcgm_agent
import threading


class DcgmHandle:
    '''
    Class to encapsulate a handle to DCGM and global methods to control + query the host engine
    '''

    handleLock = threading.Lock()
    handles = []

    def __init__(self,
                 handle=None,
                 ipAddress=None,
                 opMode=dcgm_structs.DCGM_OPERATION_MODE_AUTO,
                 persistAfterDisconnect=False,
                 unixSocketPath=None,
                 timeoutMs=0,
                 decoratorHandle=None
                 ):
        '''
        Constructor

        handle is an existing handle from dcgmInit(). Pass None if you want this object to handle DCGM initialization for you
        ipAddress is the host to connect to. None = start embedded host engine
        opMode is a dcgm_structs.DCGM_OPERATION_MODE_* constant for how the host engine should run (embedded mode only)
        persistAfterDisconnect (TCP-IP connections only) is whether the host engine should persist all of our watches
                               after we disconnect. 1=persist our watches. 0=clean up after our connection
        unixSocketPath is a path to a path on the local filesystem that is a unix socket that the host engine is listening on.
                       This option is mutually exclusive with ipAddress
        timeoutMs is how long to wait for TCP/IP or Unix domain connections to establish in ms. 0=Default timeout (5000ms)
        decoratorHandle defaults to None. If a handle is supplied, it is presumed to come from a
        decorator. Set this to False if that is not the case and it comes from a test. If a handle
        is not supplied and it DOES come from a decorator, set this to True. If it is True, it will
        not be tracked. The idea is to track handles created in tests, and reap them between tests.
        '''
        self._handleCreated = False
        self._persistAfterDisconnect = persistAfterDisconnect

        if handle is not None:
            self.handle = handle
            if decoratorHandle is None:
                self._fromDecorator = True
            else:
                self._fromDecorator = decoratorHandle
            return
        else:
            if decoratorHandle is None:
                self._fromDecorator = False
            else:
                self._fromDecorator = decoratorHandle

        self._ipAddress = ipAddress

        # Can't provide both unix socket and ip address
        if ipAddress is not None and unixSocketPath is not None:
            raise dcgm_structs.dcgmExceptionClass(
                dcgm_structs.DCGM_ST_BADPARAM)

        # Initialize the DCGM client library
        dcgm_structs._dcgmInit()
        dcgm_agent.dcgmInit()  # Not harmful to call this multiple times in a process

        # If neither ipAddress nor unixSocketPath are present, start an embedded host engine
        if ipAddress is None and unixSocketPath is None:
            self.handle = dcgm_agent.dcgmStartEmbedded(opMode)
            self.isEmbedded = True
            self._handleCreated = True
            if not self._fromDecorator:
                with DcgmHandle.handleLock:
                    DcgmHandle.handles.append(self)
            return

        # Set up connection parameters. We're connecting to something
        connectParams = dcgm_structs.c_dcgmConnectV3Params_v1()
        connectParams.version = dcgm_structs.c_dcgmConnectV3Params_version
        connectParams.timeoutMs = timeoutMs
        if self._persistAfterDisconnect:
            connectParams.persistAfterDisconnect = 1
        else:
            connectParams.persistAfterDisconnect = 0

        if ipAddress is not None:
            connectToAddress = "tcp://" + ipAddress
        else:
            connectToAddress = "unix://" + unixSocketPath

        self.handle = dcgm_agent.dcgmConnect_v3(
            connectToAddress, connectParams)
        self.isEmbedded = False
        self._handleCreated = True

        if not self._fromDecorator:
            with DcgmHandle.handleLock:
                DcgmHandle.handles.append(self)

    @classmethod
    def reset(cls):
        with cls.handleLock:
            cleanup = cls.handles[:]
            cls.handles = []

        for handle in cleanup:
            # It may have already been shutdown.
            if handle._handleCreated:
                handle.Shutdown()

    def __del__(self):
        '''
        Destructor
        '''

        if self._handleCreated:
            self.Shutdown()

    def GetSystem(self):
        '''
        Get a DcgmSystem instance for this handle
        '''
        return pydcgm.DcgmSystem(self)

    def __StopDcgm__(self):
        '''
        Shuts down either the hostengine or the embedded server
        '''
        if self.isEmbedded:
            dcgm_agent.dcgmStopEmbedded(self.handle)
        else:
            dcgm_agent.dcgmDisconnect(self.handle)

    def FromDecorator(self):
        return self._fromDecorator

    def Shutdown(self):
        '''
        Shutdown DCGM hostengine
        '''
        if not self._handleCreated:
            return

        # Remove tracking of groups under this handle if under a decorator.
        if self.FromDecorator():
            pydcgm.DcgmGroup.clear(self)

        try:
            self.__StopDcgm__()
        except AttributeError as e:
            # Due to multi-threading, sometimes this is called after the modules have been unloaded, making
            # dcgm_agent effectively NoneType and resulting in this error being thrown.
            pass

        self._handleCreated = False
        self.handle = None

    @staticmethod
    def Unload():
        '''
        Unload DCGM, removing any memory it is pointing at. Use this if you really
        want DCGM gone from your process. Shutdown() only closes the connection/embedded host engine
        that was create in __init__().
        '''
        dcgm_agent.dcgmShutdown()

    def GetIpAddress(self):
        '''
        Returns the IP address associated with this handle. None=embedded connection
        '''
        return self._ipAddress
