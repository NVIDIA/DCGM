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
from dcgm_field_helpers import FieldValueEncoder
from DcgmReader import DcgmReader

import abc
import dcgm_fields
import json
import dcgm_structs
import time
import logger

try:
    from prometheus_client import start_http_server, Gauge
except ImportError:
    pass
    logger.warning("prometheus_client not installed, please run: \"pip install prometheus_client\"")

ignore_List = [dcgm_fields.DCGM_FI_DEV_PCI_BUSID, dcgm_fields.DCGM_FI_DEV_UUID]


publishFieldIds = [
            dcgm_fields.DCGM_FI_DEV_PCI_BUSID, #Needed for plugin_instance
            dcgm_fields.DCGM_FI_DEV_POWER_USAGE,
            dcgm_fields.DCGM_FI_DEV_GPU_TEMP,
            dcgm_fields.DCGM_FI_DEV_SM_CLOCK,
            dcgm_fields.DCGM_FI_DEV_GPU_UTIL,
            dcgm_fields.DCGM_FI_DEV_RETIRED_PENDING,
            dcgm_fields.DCGM_FI_DEV_RETIRED_SBE,
            dcgm_fields.DCGM_FI_DEV_RETIRED_DBE,
            dcgm_fields.DCGM_FI_DEV_ECC_SBE_AGG_TOTAL,
            dcgm_fields.DCGM_FI_DEV_ECC_DBE_AGG_TOTAL,
            dcgm_fields.DCGM_FI_DEV_FB_TOTAL,
            dcgm_fields.DCGM_FI_DEV_FB_FREE,
            dcgm_fields.DCGM_FI_DEV_FB_USED,
            dcgm_fields.DCGM_FI_DEV_PCIE_REPLAY_COUNTER,
            dcgm_fields.DCGM_FI_DEV_UUID
            ]

class DcgmJson(DcgmReader):
    def __init__(self):
        DcgmReader.__init__(self, fieldIds=publishFieldIds, ignoreList=ignore_List)
        self.m_jsonData = {} #Json data for each field.
        self.m_list=[] # list of jsons of all the fields.
    ###########################################################################
    
    '''
    The customDataHandler creates a json from the fvs dictionary. All jsons are appended to a list which is then returned from
    the function.

    @params: 
    fvs : The fieldvalue dictionary that contains info about the values of field Ids for each gpuId.

    @return :
    list of all the jsons for each gpuID. 

    '''
    def CustomDataHandler(self,fvs):
        for gpuId in list(fvs.keys()):
            gpuFv = fvs[gpuId]
            typeInstance = str(gpuId)

            for fieldId in list(gpuFv.keys()):
                if fieldId in self.m_dcgmIgnoreFields:
                    continue

                self.m_jsonData = {
                    "GpuId": typeInstance,
                    "UUID": (gpuFv[dcgm_fields.DCGM_FI_DEV_UUID][-1]).value,
                    "FieldTag": self.m_fieldIdToInfo[fieldId].tag,
                    "FieldValues": json.dumps(gpuFv[fieldId], cls=FieldValueEncoder),
                }
                self.m_list.append(json.dumps(self.m_jsonData))

    ###########################################################################
    
    '''
    function to create json from the field value dictionary.
    '''
    def CreateJson(self,data=None):
        self.Process()
        return self.m_list
    
    ###########################################################################

###############################################################################
# Usage:                                                                      #
#                                                                             #
# obj = DcgmJson()                                                            #
#                                                                             #
# obj.createJson()                                                            #
#                                                                             #
# obj.shutdown()                                                              #
###############################################################################
