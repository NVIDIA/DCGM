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
import test_utils
import dcgm_structs
import dcgm_fields
import dcgm_agent
from dcgm_structs import dcgmExceptionClass

@test_utils.run_with_embedded_host_engine()
def test_dcgm_field_group_duplicate_name(handle):
    fieldIds = [dcgm_fields.DCGM_FI_DRIVER_VERSION, ]
    handle = pydcgm.DcgmHandle(handle)
    fieldGroup = pydcgm.DcgmFieldGroup(handle, "dupeme", fieldIds)
    with test_utils.assert_raises(dcgmExceptionClass(dcgm_structs.DCGM_ST_DUPLICATE_KEY)):
        fieldGroup2 = pydcgm.DcgmFieldGroup(handle, "dupeme", fieldIds)

@test_utils.run_with_embedded_host_engine()
def test_dcgm_field_group_add_remove(handle):
    fieldIds = [dcgm_fields.DCGM_FI_DRIVER_VERSION, dcgm_fields.DCGM_FI_DEV_NAME, dcgm_fields.DCGM_FI_DEV_BRAND]
    handle = pydcgm.DcgmHandle(handle)
    fieldGroup = pydcgm.DcgmFieldGroup(handle, "mygroup", fieldIds)

    #Save this ID before we mess with the object
    fieldGroupId = fieldGroup.fieldGroupId

    #This will assert on error
    fieldGroupInfo = dcgm_agent.dcgmFieldGroupGetInfo(handle.handle, fieldGroupId)

    #Delete the field group and make sure it's gone from the host engine
    del(fieldGroup)
    fieldGroup = None

    with test_utils.assert_raises(dcgmExceptionClass(dcgm_structs.DCGM_ST_NO_DATA)):
        fieldGroupInfo = dcgm_agent.dcgmFieldGroupGetInfo(handle.handle, fieldGroupId)

@test_utils.run_with_embedded_host_engine()
def test_dcgm_field_group_info(handle):
    fieldIds = [dcgm_fields.DCGM_FI_DRIVER_VERSION, dcgm_fields.DCGM_FI_DEV_NAME, dcgm_fields.DCGM_FI_DEV_BRAND]
    handle = pydcgm.DcgmHandle(handle)
    fieldGroup = pydcgm.DcgmFieldGroup(handle, "mygroup", fieldIds)

    #Get the field group we just added to verify it was added and the metadata is correct
    fieldGroupInfo = dcgm_agent.dcgmFieldGroupGetInfo(handle.handle, fieldGroup.fieldGroupId)
    assert fieldGroupInfo.version == dcgm_structs.dcgmFieldGroupInfo_version1, fieldGroupInfo.version
    assert fieldGroupInfo.fieldGroupId == int(fieldGroup.fieldGroupId.value), "%s != %s" %(str(fieldGroupInfo.fieldGroupId), str(fieldGroup.fieldGroupId))
    assert fieldGroupInfo.fieldGroupName == fieldGroup.name, str(fieldGroupInfo.name)
    assert fieldGroupInfo.numFieldIds == len(fieldIds), fieldGroupInfo.numFieldIds
    for i, fieldId in enumerate(fieldIds):
        assert fieldGroupInfo.fieldIds[i] == fieldId, "i = %d, %d != %d" % (i, fieldGroupInfo.fieldIds[i], fieldId)


@test_utils.run_with_embedded_host_engine()
def test_dcgm_field_group_get_by_name(handle):
    fieldIds = [dcgm_fields.DCGM_FI_DRIVER_VERSION, dcgm_fields.DCGM_FI_DEV_NAME, dcgm_fields.DCGM_FI_DEV_BRAND]
    handle = pydcgm.DcgmHandle(handle)

    fieldGroupName = "mygroup"
    fieldGroupObj = pydcgm.DcgmFieldGroup(handle, "mygroup", fieldIds)

    findByNameId = handle.GetSystem().GetFieldGroupIdByName(fieldGroupName)

    assert findByNameId is not None, "Expected field group ID. Got None"
    assert int(findByNameId.value) == int(fieldGroupObj.fieldGroupId.value), "Got field group ID handle mismatch %s != %s" % (findByNameId, fieldGroupObj.fieldGroupId)

    #Make sure we can create an object from our found id and delete it
    fieldGroupObj2 = pydcgm.DcgmFieldGroup(dcgmHandle=handle, fieldGroupId=findByNameId)
    fieldGroupObj2.Delete()
