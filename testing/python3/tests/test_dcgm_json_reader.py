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
from _test_helpers import maybemock
from dcgm_structs import dcgmExceptionClass
from json import loads
import pydcgm
import dcgm_structs
import dcgm_structs_internal
import dcgm_agent_internal
import dcgm_fields
import test_utils
import time
import os
import sys

from common.Struct import Struct
from DcgmJsonReader import DcgmJsonReader

def create_fv(key, values):
    fv_values = [Struct(fieldId=key, value=val) for val in values] # Struct(values=values)
    return {key: Struct(values=fv_values)}

def test_convert_field_id_to_tag():
    fieldTagMap = {
        1: Struct(tag='field1'),
        2: Struct(tag='field2'),
        3: Struct(tag='field3'),
    }

    dr = DcgmJsonReader()
    dr.m_fieldIdToInfo = fieldTagMap
    for key in list(fieldTagMap.keys()):
        assert (dr.ConvertFieldIdToTag(key) == fieldTagMap[key].tag) # pylint: disable=no-member

def test_prepare_json():
    obj = {
        'star wars': 'overrated'
    }

    gpuUuidMap = {
        0: 'uuid0',
        1: 'uuid1'
    }

    dr = DcgmJsonReader()
    dr.m_gpuIdToUUId = gpuUuidMap

    for gpuId in gpuUuidMap:
        outJson = dr.PrepareJson(gpuId, obj)
        outObj = loads(outJson)
        assert(outObj['star wars'] == 'overrated')
        assert(outObj['gpu_uuid'] == gpuUuidMap[gpuId])

def test_custom_data_handler():
    namespace = Struct(called=False, result=None)

    expected = {
        'fieldName': 'value',
        'gpu_uuid': 'this'
    }

    # This function tells us that the json callback is called by CustomDataHandler
    # with the correct data
    def setCalled(json):
        namespace.called = True
        namespace.result = loads(json)

    gpuUuidMap = {0: 'notthis', 1: 'this'}
    fvs = {1: create_fv('key', ['not last value', 'value'])}

    dr = DcgmJsonReader()
    dr.m_gpuIdToUUId = gpuUuidMap
    dr.m_fieldIdToInfo = {'key': Struct(tag='fieldName')}
    dr.CustomJsonHandler = setCalled

    dr.CustomDataHandler(fvs)
    assert namespace.called
    assert expected == namespace.result

@maybemock.patch.multiple('logging', info=maybemock.DEFAULT, warning=maybemock.DEFAULT)
def test_custom_json_handler(info, warning):
    dr = DcgmJsonReader()
    dr.CustomJsonHandler(1)
    info.assert_called_with(1)
    warning.assert_called_with('CustomJsonHandler has not been overriden')
