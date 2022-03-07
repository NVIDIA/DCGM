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

import dcgm_agent
import dcgm_structs

'''
Class for managing a group of field IDs in the host engine.
'''
class DcgmFieldGroup:
    '''
    Constructor

    dcgmHandle - DcgmHandle() instance to use for communicating with the host engine
    name - Name of the field group to use within DCGM. This must be unique
    fieldIds - Fields that are part of this group
    fieldGroupId - If provided, this is used to initialize the object from an existing field group ID
    '''
    def __init__(self, dcgmHandle, name="", fieldIds=[], fieldGroupId=None):
        self.name = name
        self.fieldIds = fieldIds
        self._dcgmHandle = dcgmHandle
        if fieldGroupId is not None:
            self.fieldGroupId = fieldGroupId
        else:
            self.fieldGroupId = None #Assign here so the destructor doesn't fail if the call below fails
            self.fieldGroupId = dcgm_agent.dcgmFieldGroupCreate(self._dcgmHandle.handle, fieldIds, name)

    '''
    Remove this field group from DCGM. This object can no longer be passed to other APIs after this call.
    '''
    def Delete(self):
        if self.fieldGroupId is not None:
            try:
                try:
                    dcgm_agent.dcgmFieldGroupDestroy(self._dcgmHandle.handle, self.fieldGroupId)
                except dcgm_structs.dcgmExceptionClass(dcgm_structs.DCGM_ST_NO_DATA):
                    #someone may have deleted the group under us. That's ok.
                    pass
            except AttributeError as ae:
                # When we're cleaning up at the end, dcgm_agent and dcgm_structs have been unloaded and we'll 
                # get an AttributeError: "'NoneType' object has no 'dcgmExceptionClass'" Ignore this
                pass
            self.fieldGroupId = None

    #Destructor
    def __del__(self):
        self.Delete()
