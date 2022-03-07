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
from DcgmReader import DcgmReader
from json import dumps as toJson
from os import environ
from socket import socket, AF_INET, SOCK_DGRAM
from time import sleep
import dcgm_fields
import logging

class DcgmJsonReader(DcgmReader):

    ###########################################################################
    def ConvertFieldIdToTag(self, fieldId):
        return self.m_fieldIdToInfo[fieldId].tag

    ###########################################################################
    def PrepareJson(self, gpuId, obj):
        '''
        Receive an object with measurements turn it into an equivalent JSON. We
        add the GPU UUID first.
        '''
        uuid = self.m_gpuIdToUUId[gpuId]
        # This mutates the original object, but it shouldn't be a problem here
        obj['gpu_uuid'] = uuid
        return toJson(obj)

    ###########################################################################
    def CustomDataHandler(self, fvs):
        for gpuId in list(fvs.keys()):
            # We don't need the keys because each value has a `fieldId`
            # So just get the values
            gpuData = list(fvs[gpuId].values())

            # Get the values from FV (which is a list of values)
            valuesListOfLists = [datum.values for datum in gpuData]

            # We only want the last measurement
            lastValueList = [l[-1] for l in valuesListOfLists]

            # Turn FV into a conventional Python Object which can be converted to JSON
            outObject = {self.ConvertFieldIdToTag(i.fieldId): i.value for i in lastValueList}
            outJson = self.PrepareJson(gpuId, outObject)

            self.CustomJsonHandler(outJson)

    ###########################################################################
    def CustomJsonHandler(self, outJson):
        '''
        This method should be overriden by subclasses to handle the JSON objects
        received.
        '''
        logging.warning('CustomJsonHandler has not been overriden')
        logging.info(outJson)
