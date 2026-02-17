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

import dcgm_agent
import dcgm_structs
import logger
import threading

'''
Class for managing a group of field IDs in the host engine.
'''


class DcgmFieldGroup:
    '''
    Static members
    '''

    groups = {}  # Dictionary of per-handle named group name to group.
    groupLock = threading.RLock()  # lock for above

    '''
    Constructor

    dcgmHandle - DcgmHandle() instance to use for communicating with the host engine
    name - Name of the field group to use within DCGM. This must be unique
    fieldIds - Fields that are part of this group
    fieldGroupId - If provided, this is used to initialize the object from an existing field group ID
    '''

    def __init__(self, dcgmHandle, name="", fieldIds=None, fieldGroupId=None):
        fieldIds = fieldIds or []
        self.name = name
        self.fieldIds = fieldIds
        self._dcgmHandle = dcgmHandle
        self.wasCreated = False

        # If the user passed in an ID, the field group already exists. Fetch live info
        if fieldGroupId is not None:
            self.fieldGroupId = fieldGroupId
            fieldGroupInfo = dcgm_agent.dcgmFieldGroupGetInfo(
                self._dcgmHandle.handle, self.fieldGroupId)
            self.name = fieldGroupInfo.fieldGroupName
            self.fieldIds = fieldGroupInfo.fieldIds
        else:
            # Assign here so the destructor doesn't fail if the call below fails
            self.fieldGroupId = None
            self.fieldGroupId = dcgm_agent.dcgmFieldGroupCreate(
                self._dcgmHandle.handle, fieldIds, name)
            self.wasCreated = True

            with DcgmFieldGroup.groupLock:
                if self.fieldGroupId.value not in DcgmFieldGroup.groups:
                    DcgmFieldGroup.groups[self.fieldGroupId.value] = self

    '''
    Remove this field group from DCGM. This object can no longer be passed to other APIs after this call.
    '''

    def Delete(self):
        """
        We need to see if this shadows a FieldGroupId that was already created
        by us. If so, we need to actually have the hostengine delete the
        FieldGroup.
        """
        # Remove from tracked list
        with DcgmFieldGroup.groupLock:
            if self.fieldGroupId.value in DcgmFieldGroup.groups:
                del DcgmFieldGroup.groups[self.fieldGroupId.value]
                destroy = True
            else:
                destroy = False

        if destroy:
            dcgm_agent.dcgmFieldGroupDestroy(
                self._dcgmHandle.handle, self.fieldGroupId)

        self.fieldGroupId = None
        self._dcgmHandle = None

    # This clears the groups that were created. It is to be called in the
    # hostengine decorators to ensure only groups created in tests under the
    # decorators are actually tracked. Of course, this means that no groups
    # can be created IN decorators. This deletes tracking them locally. Call
    # reset if you wish to delete them in a running standalone hostengine as
    # well
    @classmethod
    def clear(cls):
        with cls.groupLock:
            cls.groups.clear()

    # Besides calling clear() above, this also calls Delete() on the groups,
    # which has them deleted in the running standlone hostengine.
    @classmethod
    def reset(cls):
        with cls.groupLock:
            remove = list(cls.groups)

            for fieldGroupIdValue in remove:
                cls.groups[fieldGroupIdValue].Delete()

        cls.clear()

    # Destructor
    def __del__(self):
        if self._dcgmHandle != None:
            try:
                self.Delete()
            except dcgm_structs.dcgmExceptionClass(dcgm_structs.DCGM_ST_NO_DATA):
                # someone may have deleted the group under us. That's ok.
                pass
            except dcgm_structs.dcgmExceptionClass(dcgm_structs.DCGM_ST_CONNECTION_NOT_VALID):
                # We lost our connection, but we're destructing this object anyway.
                pass
            except dcgm_structs.dcgmExceptionClass(dcgm_structs.DCGM_ST_BADPARAM):
                # Handle or field group ID has already been expired.
                pass
            except AttributeError as ae:
                # When we're cleaning up at the end, dcgm_agent and dcgm_structs have been unloaded and we'll
                # get an AttributeError: "'NoneType' object has no 'dcgmExceptionClass'" Ignore this
                pass
            except TypeError as te:
                # When we're cleaning up at the end, dcgm_agent and dcgm_structs have been unloaded and we might
                # get a TypeError: "'NoneType' object is not callable'" Ignore this
                pass
