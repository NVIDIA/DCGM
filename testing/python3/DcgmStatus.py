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
import dcgm_agent
import dcgm_structs

class DcgmStatus:
    def __init__(self):
        self.handle = dcgm_agent.dcgmStatusCreate()
        self.errors = []

    def __del__(self):
        dcgm_agent.dcgmStatusDestroy(self.handle)

    '''
    Take any errors stored in our handle and update self.errors with them
    '''
    def UpdateErrors(self):
        errorCount = dcgm_agent.dcgmStatusGetCount(self.handle)
        if errorCount < 1:
            return

        for i in range(errorCount):
            self.errors.append(dcgm_agent.dcgmStatusPopError(self.handle))

    '''
    Throw an exception if any errors are stored in our status handle

    The exception text will contain all of the errors
    '''
    def ThrowExceptionOnErrors(self):
        #Make sure we've captured all errors before looking at them
        self.UpdateErrors()

        if len(self.errors) < 1:
            return

        errorString = "Errors: "
        for value in self.errors:
            errorString += "\"%s\"" % value
            raise dcgm_structs.DCGMError(value.status)

