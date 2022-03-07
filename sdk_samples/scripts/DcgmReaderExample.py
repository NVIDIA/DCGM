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
from DcgmReader import *
import dcgm_fields

class FieldHandlerReader(DcgmReader):
    '''
        Override just this method to do something different per field. 
        This method is called once for each field for each GPU each 
        time that its Process() method is invoked, and it will be skipped
        for blank values and fields in the ignore list.
    '''
    def CustomFieldHandler(self, gpuId, fieldId, fieldTag, val):
        print('GPU %d %s(%d) = %s' % (gpuId, fieldTag, fieldId, val.value))

class DataHandlerReader(DcgmReader):
    '''
        Override just this method to handle the entire map of data in your own way. This 
        might be used if you want to iterate by field id and then GPU or something like that.
        This method is called once for each time the Process() method is invoked.
    '''
    def CustomDataHandler(self, fvs):
        for fieldId in self.m_publishFieldIds:
            if fieldId in self.m_dcgmIgnoreFields:
                continue
        
            out = 'Values for %s:' % (self.m_fieldIdToInfo[fieldId].tag)
            wasBlank = True
            for gpuId in list(fvs.keys()):
                gpuFv = fvs[gpuId]
                val = gpuFv[fieldId][-1]

                #Skip blank values. Otherwise, we'd have to insert a placeholder blank value based on the fieldId
                if val.isBlank:
                    continue

                wasBlank = False
                append = " GPU%d=%s" % (gpuId, val.value)
                out = out + append

            if wasBlank == False:
                print(out)

'''
    field_ids        : List of the field ids to publish. If it isn't specified, our default list is used.
    update_frequency : Frequency of update in microseconds. Defauls to 10 seconds or 10000000 microseconds
    keep_time        : Max time to keep data from NVML, in seconds. Default is 3600.0 (1 hour)
    ignores          : List of the field ids we want to query but not publish.
'''
def DcgmReaderDictionary(field_ids=defaultFieldIds, update_frequency=10000000, keep_time=3600.0, ignores=[], field_groups='dcgm_fieldgroupdata'):
    # Instantiate a DcgmReader object
    dr = DcgmReader(fieldIds=field_ids, updateFrequency=update_frequency, maxKeepAge=keep_time, ignoreList=ignores, fieldGroupName=field_groups)

    # Get the default list of fields as a dictionary of dictionaries:
    # gpuId -> field name -> field value
    data = dr.GetLatestGpuValuesAsFieldNameDict()

    # Print the dictionary
    for gpuId in data:
        for fieldName in data[gpuId]:
            print("For gpu %s field %s=%s" % (str(gpuId), fieldName, data[gpuId][fieldName]))


def main(): 
    print('Some examples of different DcgmReader usages')
    
    print('\n\nThe default interaction')
    dr = DcgmReader()
    dr.Process()

    print('\n\nUsing custom fields through the dictionary interface...')
    customFields = [dcgm_fields.DCGM_FI_DEV_MEM_COPY_UTIL, dcgm_fields.DCGM_FI_DEV_GPU_UTIL, dcgm_fields.DCGM_FI_DEV_POWER_USAGE]
    DcgmReaderDictionary(field_ids=customFields)

    print('\n\nProcessing in field order by overriding the CustomerDataHandler() method')
    cdr = DataHandlerReader()
    cdr.Process()

    print('\n\nPrinting a little differently by overriding the CustomFieldHandler() method')
    fhr = FieldHandlerReader()
    fhr.Process()

if __name__ == '__main__':
    main()
