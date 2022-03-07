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

from wsgiref.simple_server import make_server
import cgi
import os
import DcgmHandle, DcgmSystem, DcgmGroup
import dcgm_structs
import json
from dcgm_structs import dcgmExceptionClass, DCGM_ST_CONNECTION_NOT_VALID

###############################################################################
DCGM_HTTP_PORT = 1981
DCGM_HTTP_SERVE_FROM = None #'wwwroot/' #Local relative path to serve raw files from. None = Don't serve local files
DCGM_HTTP_JSON_DIR = 'dcgmjsonrest'
DCGM_JSON_VERSION = '1.0' #Change this major version to break compatibility. Change minor version to just inform of new versions
DCGM_IP_ADDRESS = "127.0.0.1" #Use None to run an embedded hostengine
DCGM_OP_MODE = dcgm_structs.DCGM_OPERATION_MODE_AUTO

###############################################################################
DCGM_MIME_TYPE_PLAIN = 0
DCGM_MIME_TYPE_JSON  = 1
DCGM_MIME_TYPE_HTML  = 2
DCGM_MIME_TYPE_PNG   = 3
DCGM_MIME_TYPE_GIF   = 4
DCGM_MIME_TYPE_JPG   = 5
DCGM_MIME_TYPE_JS    = 6

###############################################################################
DCGM_HTTP_CODE_OK           = 200
DCGM_HTTP_CODE_BAD_REQUEST  = 400
DCGM_HTTP_CODE_UNAUTHORIZED = 401
DCGM_HTTP_CODE_NOT_FOUND    = 404
DCGM_HTTP_CODE_INT_ERROR    = 500

###############################################################################
class DcgmHttpException(Exception):
    pass

###############################################################################
class DcgmHttpServer:
    ###########################################################################
    def __init__(self):
        #Initialize some defaults
        self.SetHttpResponseCode(200)
        self.SetMimeType(DCGM_MIME_TYPE_PLAIN)
        self._dcgmHandle = None
        self._dcgmSystem = None
        self._defaultGpuGroup = None
        self._haveWatchedHealth = False

    ###########################################################################
    def SetHttpResponseCode(self, codeId):
        codeId = int(codeId)

        if codeId == DCGM_HTTP_CODE_OK:
            self._httpResponseCode = "%d OK" % codeId
        elif codeId == DCGM_HTTP_CODE_BAD_REQUEST:
            self._httpResponseCode = '400 Bad Request'
        elif codeId == DCGM_HTTP_CODE_UNAUTHORIZED:
            self._httpResponseCode = '401 Unauthorized'
        elif codeId == DCGM_HTTP_CODE_NOT_FOUND:
            self._httpResponseCode = '404 Not Found'
        else: #DCGM_HTTP_CODE_INT_ERROR
            self._httpResponseCode = '500 Internal Server Error' #default

    ###########################################################################
    def SetMimeType(self, mimeType):
        if mimeType == DCGM_MIME_TYPE_PLAIN:
            self._httpMimeType = ('Content-Type','text/plain')
        elif mimeType == DCGM_MIME_TYPE_JSON:
            self._httpMimeType = ('Content-Type','application/json')
        elif mimeType == DCGM_MIME_TYPE_HTML:
            self._httpMimeType = ('Content-Type','text/html')
        elif mimeType == DCGM_MIME_TYPE_JPG:
            self._httpMimeType = ('Content-Type','image/jpeg')
        elif mimeType == DCGM_MIME_TYPE_GIF:
            self._httpMimeType = ('Content-Type','image/gif')
        elif mimeType == DCGM_MIME_TYPE_PNG:
            self._httpMimeType = ('Content-Type','image/png')
        elif mimeType == DCGM_MIME_TYPE_JS:
            self._httpMimeType = ('Content-Type','application/javascript')
        else:
            self._httpMimeType = ('Content-Type','text/plain')

    ###########################################################################
    def SetMimeTypeFromExtension(self, extension):
        extension = extension.lower()

        if extension == 'html' or extension == 'htm':
            self.SetMimeType(DCGM_MIME_TYPE_HTML)
        elif extension == 'json':
            self.SetMimeType(DCGM_MIME_TYPE_JSON)
        elif extension == 'png':
            self.SetMimeType(DCGM_MIME_TYPE_PNG)
        elif extension == 'gif':
            self.SetMimeType(DCGM_MIME_TYPE_GIF)
        elif extension == 'jpg' or extension == 'jpeg':
            self.SetMimeType(DCGM_MIME_TYPE_JPG)
        elif extension == 'js':
            self.SetMimeType(DCGM_MIME_TYPE_JS)
        else:
            self.SetMimeType(DCGM_MIME_TYPE_PLAIN)

    ###########################################################################
    def GetJsonError(self, errorString):
        responseObj = {'version':DCGM_JSON_VERSION,
                       'status':'ERROR',
                       'errorString':errorString}

        retString = json.JSONEncoder().encode(responseObj)
        self.SetMimeType(DCGM_MIME_TYPE_JSON)
        return retString

    ###########################################################################
    def GetJsonResponse(self, encodeObject):
        responseObj = {'version':DCGM_JSON_VERSION,
                       'status':'OK',
                       'responseData':encodeObject}

        #print str(responseObj)

        retString = json.dumps(responseObj, cls=dcgm_structs.DcgmJSONEncoder)
        self.SetMimeType(DCGM_MIME_TYPE_JSON)
        return retString

    ###########################################################################
    def WatchHealth(self):
        if self._haveWatchedHealth:
            return

        self._defaultGpuGroup.health.Set(dcgm_structs.DCGM_HEALTH_WATCH_ALL)
        #Make sure the health has updated at least once
        self._dcgmSystem.UpdateAllFields(1)

        self._haveWatchedHealth = True

    ###########################################################################
    def CheckDcgmConnection(self):
        '''
        Check if we are connected to DCGM or not and try to connect to DCGM if we aren't connected.

        Returns !0 on error. 0 on success
        '''
        if self._dcgmHandle != None:
            return 0

        self._dcgmHandle = DcgmHandle.DcgmHandle(handle=None, ipAddress=DCGM_IP_ADDRESS, opMode=DCGM_OP_MODE)
        self._dcgmSystem = self._dcgmHandle.GetSystem()
        self._defaultGpuGroup = self._dcgmSystem.GetDefaultGroup()

        #Clear other connection state we can no longer guarantee
        self._haveWatchedHealth = False
        return 0

    ###########################################################################
    def GetAllGpuIds(self, queryParams):
        responseObj = []
        gpuIds = self._dcgmSystem.discovery.GetAllGpuIds()
        return self.GetJsonResponse(gpuIds)

    ###########################################################################
    def GetGpuAttributes(self, queryParams):
        if not queryParams.has_key("gpuid"):
            self.SetHttpResponseCode(DCGM_HTTP_CODE_BAD_REQUEST)
            return self.GetJsonError("Missing 'gpuid' parameter")

        gpuId = int(cgi.escape(queryParams['gpuid'][0]))

        #Validate gpuId
        gpuIds = self._dcgmSystem.discovery.GetAllGpuIds()
        if not gpuId in gpuIds:
            self.SetHttpResponseCode(DCGM_HTTP_CODE_BAD_REQUEST)
            return self.GetJsonError("gpuid parameter is invalid")

        attributes = self._dcgmSystem.discovery.GetGpuAttributes(gpuId)
        return self.GetJsonResponse(attributes)

    ###########################################################################
    def CheckGpuHealth(self, queryParams):
        #Make sure we have a health watch
        self.WatchHealth()

        healthObj = self._defaultGpuGroup.health.Check()
        return self.GetJsonResponse(healthObj)

    ###########################################################################
    def RunDiagnostic(self, queryParams):
        validationLevel = 1

        if queryParams.has_key('level'):
            validationLevel = int(cgi.escape(queryParams['level'][0]))
            if validationLevel < dcgm_structs.DCGM_POLICY_VALID_SV_SHORT or validationLevel > dcgm_structs.DCGM_POLICY_VALID_SV_LONG:
                self.SetHttpResponseCode(DCGM_HTTP_CODE_BAD_REQUEST)
                return self.GetJsonError("\"level\" parameter must be between 1 and 3")

        try:
            diagResponse = self._defaultGpuGroup.action.Validate(validationLevel)
        except dcgmExceptionClass(dcgm_structs.DCGM_ST_NOT_SUPPORTED):
            return self.GetJsonError("The DCGM diagnostic program is not installed. Please install the Tesla-recommended driver.")

        return self.GetJsonResponse(diagResponse)

    ###########################################################################
    def GetJsonRestContents(self, queryParams):
        if not queryParams.has_key("action"):
            self.SetHttpResponseCode(DCGM_HTTP_CODE_BAD_REQUEST)
            return "Missing 'action' parameter"

        action = cgi.escape(queryParams['action'][0]).lower()

        if action == 'getallgpuids':
            return self.GetAllGpuIds(queryParams)
        elif action == 'getgpuattributes':
            return self.GetGpuAttributes(queryParams)
        elif action == 'checkgpuhealth':
            return self.CheckGpuHealth(queryParams)
        elif action == 'rundiagnostic':
            return self.RunDiagnostic(queryParams)
        else:
            self.SetMimeType(DCGM_MIME_TYPE_PLAIN)
            return self.GetJsonError("Unknown action: %s" % action)

    ###########################################################################
    def GetRawFile(self, filePath):
        serveFilePath = DCGM_HTTP_SERVE_FROM + filePath

        if os.path.exists(serveFilePath):
            fp = open(serveFilePath, 'rb')
            content = fp.read()
            fp.close()

            if filePath.find('.') >= 1:
                extension = filePath.split(".")[-1]
                self.SetMimeTypeFromExtension(extension)
            else:
                self.SetMimeType(DCGM_MIME_TYPE_PLAIN)

            self.SetHttpResponseCode(DCGM_HTTP_CODE_OK)
        else:
            content = "%s not found" % serveFilePath
            self.SetHttpResponseCode(DCGM_HTTP_CODE_NOT_FOUND)

        return content

    ###########################################################################
    def GetContents(self, queryParams, filePath):

        filePathList = filePath.split('/')

        print str(filePathList)
        if filePathList[0] == DCGM_HTTP_JSON_DIR:
            return self.GetJsonRestContents(queryParams)

        #default to return a raw file from the filesystem
        if DCGM_HTTP_SERVE_FROM != None:
            return self.GetRawFile(filePath)
        else:
            self.SetHttpResponseCode(DCGM_HTTP_CODE_NOT_FOUND)

    ###########################################################################
    '''
    Web server main entry point. Call from wsgi callback

    Returns string of http contents
    '''
    def WsgiMain(self, environ, start_response):
        responseStr = ""

        filePath = environ['PATH_INFO'].lstrip('/')

        queryParams = cgi.parse_qs(environ['QUERY_STRING'])

        #for k in environ.keys():
        #    responseStr += "%s => %s\n" % (k, environ[k])

        numRetries = 0
        retryLimit = 1
        gotResponse = False

        while (not gotResponse) and numRetries < retryLimit:
            try:
                self.CheckDcgmConnection()
                responseStr = self.GetContents(queryParams, filePath)
                gotResponse = True
            except dcgmExceptionClass(dcgm_structs.DCGM_ST_CONNECTION_NOT_VALID):
                #Just retry if we have a connection error
                self._dcgmHandle = None
                self._dcgmSystem = None
                numRetries += 1
                print "Got disconnected. Retrying"
                pass

        if not gotResponse:
            responseStr = self.GetJsonError("Unable to connect to the DCGM daemon")
            self.SetHttpResponseCode(DCGM_HTTP_CODE_INT_ERROR)

        responseHeaders = [self._httpMimeType]
        start_response(self._httpResponseCode, responseHeaders)
        return responseStr

###############################################################################
def dcmg_http_app(environ, start_response):
    '''
    Main entry point
    '''
    responseStr = g_dcgmServer.WsgiMain(environ, start_response)
    return [responseStr]

###############################################################################
def application(environ, start_response):
    '''
    Callback for uWSGI
    '''
    return dcmg_http_app(environ, start_response)

#Try to load the DCGM library
dcgm_structs._dcgmInit()
g_dcgmServer = DcgmHttpServer()

###############################################################################
if __name__ == '__main__':
    httpd = make_server('', DCGM_HTTP_PORT, dcmg_http_app)
    print "Serving HTTP on port %d..." % DCGM_HTTP_PORT

    # Respond to requests until process is killed
    httpd.serve_forever()

###############################################################################


