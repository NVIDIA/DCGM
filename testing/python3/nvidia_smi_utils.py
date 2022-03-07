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
import threading
import xml.etree.ElementTree as ET
import time

import option_parser
import logger
import pydcgm
import dcgm_structs
import dcgm_fields
import dcgm_agent
import shlex
import subprocess

################################################################################
### XML tags to search for
################################################################################
THROTTLE_FN      = "clocks_throttle_reasons"
ECC_FN           = "ecc_errors"
ECC_ENABLED_FN   = "ecc_mode"
GPU_FN           = "gpu"
MINOR_NUM_FN     = "minor_number"
ECC_CURRENT_FN   = "current_ecc"
RETIRED_PAGES_FN = "retired_pages"
RETIRED_COUNT_FN = "retired_count"
RETIRED_SBE_FN   = "multiple_single_bit_retirement"
RETIRED_DBE_FN   = "double_bit_retirement"
VOLATILE_FN      = "volatile"
AGGREGATE_FN     = "aggregate"
TOTAL_FN         = "total"
DB_FN            = 'double_bit'
SB_FN            = 'single_bit'

# list of relevant throttle reasons
relevant_throttling = ["clocks_throttle_reason_hw_slowdown",
                       "clocks_throttle_reason_hw_thermal_slowdown",
                       "clocks_throttle_reason_hw_power_brake_slowdown",
                       "clocks_throttle_reason_sw_thermal_slowdown"]

################################################################################
### Supported field ids
### Each is a tuple of the field id, the type of check for error, and the ideal 
### value. Set each of these values in order to add support for a new field.
################################################################################
CHECKER_ANY_VALUE = 0
CHECKER_MAX_VALUE = 1
CHECKER_LAST_VALUE = 2
CHECKER_INFOROM = 3
supportedFields = [ (dcgm_fields.DCGM_FI_DEV_CLOCK_THROTTLE_REASONS, CHECKER_ANY_VALUE, ''),
                (dcgm_fields.DCGM_FI_DEV_ECC_CURRENT, CHECKER_ANY_VALUE, 'Active'),
                (dcgm_fields.DCGM_FI_DEV_THERMAL_VIOLATION, CHECKER_MAX_VALUE, 0),
                (dcgm_fields.DCGM_FI_DEV_ECC_DBE_VOL_TOTAL, CHECKER_LAST_VALUE, 0),
                (dcgm_fields.DCGM_FI_DEV_INFOROM_CONFIG_VALID, CHECKER_INFOROM, True),
    ]
# field ids where any value is an error
anyCheckFields = [ dcgm_fields.DCGM_FI_DEV_CLOCK_THROTTLE_REASONS, dcgm_fields.DCGM_FI_DEV_ECC_CURRENT ]
# field ids where the max value should be returned as an error
maxCheckFields = [ dcgm_fields.DCGM_FI_DEV_THERMAL_VIOLATION ]
# field ids where the last value should be returned as an error
lastCheckFields = [ dcgm_fields.DCGM_FI_DEV_ECC_DBE_VOL_TOTAL ]

# field ids where the ideal value is 0
zeroIdealField = [ dcgm_fields.DCGM_FI_DEV_ECC_DBE_VOL_TOTAL, dcgm_fields.DCGM_FI_DEV_THERMAL_VIOLATION ]

# field ids where the ideal value is an empty string
emptyStrIdealField = [dcgm_fields.DCGM_FI_DEV_CLOCK_THROTTLE_REASONS ]

# field ids where False is the ideal value
falseIdealField = [ dcgm_fields.DCGM_FI_DEV_INFOROM_CONFIG_VALID ]

def parse_int_from_nvml_xml(text):
    if text == 'N/A':
        return 0
    try:
        return int(text)
    except ValueError as e:
        return 0

################################################################################
class NvidiaSmiJob(threading.Thread):
    ################################################################################
    ### Constructor
    ################################################################################
    def __init__(self):
        threading.Thread.__init__(self)
        self.m_shutdownFlag = threading.Event()
        self.m_data = {}
        self.m_sleepInterval = 1
        self.m_inforomCorrupt = None
        self.m_supportedFields = {}
        self.InitializeSupportedFields()

    ################################################################################
    ### Map the fieldId to the information for supporting that field id
    ################################################################################
    def InitializeSupportedFields(self):
        for fieldInfo in supportedFields:
            self.m_supportedFields[fieldInfo[0]] = fieldInfo

    ################################################################################
    ### Sets the sleep interval between querying nvidia-smi
    ################################################################################
    def SetIterationInterval(self, interval):
        self.m_sleepInterval = interval

    ################################################################################
    ### Looks at the volatile XML node to find the total double bit errors
    ################################################################################
    def ParseEccErrors(self, ecc_subnode, gpudata, isVolatile):
        for child in ecc_subnode:
            if child.tag == DB_FN:
                for grandchild in child:
                    if grandchild.tag == TOTAL_FN:
                        if isVolatile:
                            gpudata[dcgm_fields.DCGM_FI_DEV_ECC_DBE_VOL_TOTAL] = parse_int_from_nvml_xml(grandchild.text)
                        else:
                            gpudata[dcgm_fields.DCGM_FI_DEV_ECC_DBE_AGG_TOTAL] = parse_int_from_nvml_xml(grandchild.text)
            elif child.tag == SB_FN:
                for grandchild in child:
                    if grandchild.tag == TOTAL_FN:
                        if isVolatile:
                            gpudata[dcgm_fields.DCGM_FI_DEV_ECC_SBE_VOL_TOTAL] = parse_int_from_nvml_xml(grandchild.text)
                        else:
                            gpudata[dcgm_fields.DCGM_FI_DEV_ECC_SBE_AGG_TOTAL] = parse_int_from_nvml_xml(grandchild.text)
                            

    def ParseRetiredPagesCount(self, retired_sbe_node, gpudata, fieldId):
        for child in retired_sbe_node:
            if child.tag == RETIRED_COUNT_FN:
                gpudata[fieldId] = parse_int_from_nvml_xml(child.text)
                break

    ################################################################################
    ### Reads the common failure conditions from the XML for this GPU
    ### All non-error values are set to None to make it easier to read the map
    ################################################################################
    def ParseSingleGpuDataFromXml(self, gpuxml_node):
        gpudata = {}
        gpu_id = -1
        for child in gpuxml_node:
            if child.tag == THROTTLE_FN:
                reasons = ''
                for grandchild in child:
                    if grandchild.tag in relevant_throttling and grandchild.text == 'Active':
                        if not reasons:
                            reasons = grandchild.tag
                        else:
                            reasons += ",%s" % (grandchild.tag)

                gpudata[dcgm_fields.DCGM_FI_DEV_CLOCK_THROTTLE_REASONS] = reasons
            elif child.tag == MINOR_NUM_FN:
                gpu_id = parse_int_from_nvml_xml(child.text)
                gpudata[dcgm_fields.DCGM_FI_DEV_NVML_INDEX] = gpu_id
            elif child.tag == ECC_ENABLED_FN:
                for grandchild in child:
                    if grandchild.tag == ECC_CURRENT_FN:
                        gpudata[dcgm_fields.DCGM_FI_DEV_ECC_CURRENT] = grandchild.text
            elif child.tag == ECC_FN:
                for grandchild in child:
                    if grandchild.tag == VOLATILE_FN:
                        self.ParseEccErrors(grandchild, gpudata, True)
                    elif grandchild.tag == AGGREGATE_FN:
                        self.ParseEccErrors(grandchild, gpudata, False)
            elif child.tag == RETIRED_PAGES_FN:
                for grandchild in child:
                    if grandchild.tag == RETIRED_SBE_FN:
                        self.ParseRetiredPagesCount(grandchild, gpudata, dcgm_fields.DCGM_FI_DEV_RETIRED_SBE)
                    elif grandchild.tag == RETIRED_DBE_FN:
                        self.ParseRetiredPagesCount(grandchild, gpudata, dcgm_fields.DCGM_FI_DEV_RETIRED_DBE)

        if gpu_id not in self.m_data:
            self.m_data[gpu_id] = {}

        for key in gpudata:
            if key not in self.m_data[gpu_id]:
                self.m_data[gpu_id][key] = []
                
            self.m_data[gpu_id][key].append(gpudata[key])

    ################################################################################
    ### Finds each GPU's xml entry and passes it off to be read
    ################################################################################
    def ParseDataFromXml(self, root):
        for child in root:
            if child.tag == GPU_FN:
                self.ParseSingleGpuDataFromXml(child)

    ################################################################################
    ### Reads some common failure condition values from nvidia-smi -q -x
    ### Returns an XML ElementTree object on success
    ###         None on failure
    ################################################################################
    def QueryNvidiaSmiXml(self, parseData=None):
        if parseData is None:
            parseData = True
        
        output = ""
        nvsmi_cmd = "nvidia-smi -q -x"
        try:
            runner = subprocess.Popen(nvsmi_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            (output, error) = runner.communicate()
            root = ET.fromstring(output)
            if parseData:
                self.ParseDataFromXml(root)
            return root
        except OSError as e:
            logger.error("Failed to run nvidia-smi.\nError: %s" % e)
            return None
        except ET.ParseError as e:
            logger.error("Got exception %s while parsing XML: %s" % (str(e), str(output)))
            return None

    ################################################################################
    ### Reads thermal violation values from nvidia-smi stats 
    ################################################################################
    def QueryNvidiaSmiStats(self):
        nvsmi_cmd = "nvidia-smi stats -c 1 -d violThm"
        # Initialize lines as an empty list so we don't do anything if IO fails
        lines = []
        try:
            runner = subprocess.Popen(nvsmi_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            (output_buf, error_buf) = runner.communicate()
            output = output_buf.decode('utf-8')
            error = error_buf.decode('utf-8')
            lines = output.split('\n')
        except OSError as e:
            logger.error("Failed to query nvidia-smi stats.\nError: %s" % e)

        for line in lines:
            if not line:
                continue

            try:
                values = line.split(',')
                if len(values) != 4:
                    continue
                gpu_id = int(values[0])
                violation = int(values[-1].strip())
            except ValueError as e:
                # Sometimes there is output with comments in it from nvidia-smi. These will throw
                # exceptions that we should ignore.
                continue

            if gpu_id not in self.m_data:
                self.m_data[gpu_id] = {}

            if dcgm_fields.DCGM_FI_DEV_THERMAL_VIOLATION not in self.m_data[gpu_id]:
                self.m_data[gpu_id][dcgm_fields.DCGM_FI_DEV_THERMAL_VIOLATION] = []

            self.m_data[gpu_id][dcgm_fields.DCGM_FI_DEV_THERMAL_VIOLATION].append(violation)

    ################################################################################
    def CheckInforom(self):
        nvsmi_cmd = "nvidia-smi"
        try:
            runner = subprocess.Popen(nvsmi_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            (outputbytes, errorbytes) = runner.communicate()
            output = outputbytes.decode('utf-8')
            error = errorbytes.decode('utf-8')
            if output.find("infoROM is corrupted") != -1:
                return True
            else:
                return False
        except OSError as e:
            logger.error("Failed to query for corrupt inforom.\nError: %s" % e)

        return None


    ################################################################################
    def run(self):
        while not self.m_shutdownFlag.is_set():
            self.QueryNvidiaSmiXml(parseData=True)
            self.QueryNvidiaSmiStats()
            time.sleep(self.m_sleepInterval)

    ################################################################################
    def CheckField(self, fieldId, values):
        # Return None for an empty list or an unsupported field id
        if not values or fieldId not in self.m_supportedFields:
            return None

        if self.m_supportedFields[fieldId][1] == CHECKER_ANY_VALUE:
            for val in values:
                if val:
                    return val
            return None
        elif self.m_supportedFields[fieldId][1] == CHECKER_MAX_VALUE:
            maxVal = values[0]
            for num, val in enumerate(values, start=1):
                if val > maxVal:
                    maxVal = val

            return maxVal
        elif self.m_supportedFields[fieldId][1] == CHECKER_LAST_VALUE:
            return values[-1]
        else:
            return None

    ################################################################################
    ### Determine if the NvidiaSmiChecker object also found an error for the specified
    ### gpuId and fieldId. If so, return a valid value.
    ### Returning None means no error was found for that fieldId and gpuId
    ################################################################################
    def GetErrorValue(self, gpuId, fieldId):
        if not gpuId:
            if fieldId == dcgm_fields.DCGM_FI_DEV_INFOROM_CONFIG_VALID:
                if self.m_inforomCorrupt is None:
                    self.m_inforomCorrupt = self.CheckInforom()
                # Only valid if we're looking for inforom errors
                return self.m_inforomCorrupt, self.GetCorrectValue(fieldId)
        elif gpuId in self.m_data:
            if fieldId in self.m_data[gpuId]:
                return self.CheckField(fieldId, self.m_data[gpuId][fieldId]), self.GetCorrectValue(fieldId)

        return None, None

    def GetCorrectValue(self, fieldId):
        if fieldId not in self.m_supportedFields:
            return 'Unknown'
        else:
            return self.m_supportedFields[fieldId][2]


    ################################################################################
    ### Checks for multiple page retirement issues within the nvidia-smi xml output
    ### returns True if there are page retirement issues according to criteria
    ### described in JIRA DCGM-1009
    ################################################################################
    def CheckPageRetirementErrors(self):

        elemTree = self.QueryNvidiaSmiXml()
        if elemTree is None:
            logger.warning("We were unable to query nvidia-smi XML successfully. Ignoring the page retirement check.")
            return False

        totals = {}
        totals[dcgm_fields.DCGM_FI_DEV_ECC_SBE_VOL_TOTAL] = 0
        totals[dcgm_fields.DCGM_FI_DEV_ECC_DBE_VOL_TOTAL] = 0
        totals[dcgm_fields.DCGM_FI_DEV_ECC_SBE_AGG_TOTAL] = 0
        totals[dcgm_fields.DCGM_FI_DEV_ECC_DBE_AGG_TOTAL] = 0
        totals[dcgm_fields.DCGM_FI_DEV_RETIRED_SBE] = 0
        totals[dcgm_fields.DCGM_FI_DEV_RETIRED_DBE] = 0

        for gpuId in self.m_data:
            for fieldId in totals:
                if fieldId in self.m_data[gpuId]:
                    if self.m_data[gpuId][fieldId]:
                        totals[fieldId] += self.m_data[gpuId][fieldId][-1]

        if (totals[dcgm_fields.DCGM_FI_DEV_RETIRED_SBE] + totals[dcgm_fields.DCGM_FI_DEV_RETIRED_DBE]) > 64:
            logger.warning("More than 64 page retirements were found")
            return True

        if totals[dcgm_fields.DCGM_FI_DEV_ECC_SBE_VOL_TOTAL] + totals[dcgm_fields.DCGM_FI_DEV_ECC_DBE_VOL_TOTAL] > 50000:
            logger.warning("Too many ECC errors found: %d volatile SBE and %d volatile DBE errors" % \
                    (totals[dcgm_fields.DCGM_FI_DEV_ECC_SBE_VOL_TOTAL], \
                     totals[dcgm_fields.DCGM_FI_DEV_ECC_DBE_VOL_TOTAL]))
            return True

        if totals[dcgm_fields.DCGM_FI_DEV_ECC_SBE_AGG_TOTAL] + totals[dcgm_fields.DCGM_FI_DEV_ECC_DBE_AGG_TOTAL] > 50000:
            logger.warning("Too many ECC errors found: %d aggregate SBE and %d aggregate DBE errors" % \
                    (totals[dcgm_fields.DCGM_FI_DEV_ECC_SBE_AGG_TOTAL], \
                     totals[dcgm_fields.DCGM_FI_DEV_ECC_DBE_AGG_TOTAL]))
            return True

        return False

def are_gpus_free():
    """
    Parses nvidia-smi xml output and discovers if any processes are using  the GPUs,
    returns  whether or not the GPUs are in use or not. True = GPUs are not being used.
    False = GPUs are in use by one or more processes
    """

    cmd = "nvidia-smi -q -x"
    try:
        nvsmiData = subprocess.check_output(shlex.split(cmd)).decode('utf-8')
    except subprocess.CalledProcessError:
        logger.info("The nvidia-smi XML output was malformed.")
        return True

    nvsmiData = subprocess.check_output(shlex.split(cmd)).decode('utf-8')
    try:
        tree = ET.fromstring(nvsmiData)
    except ET.ParseError:
        logger.info("The nvidia-smi XML output was malformed.")
        return True

    pidList = []
    processList = []
    # Goes deep into the XML Element Tree to get PID and Process Name
    for node in tree.iter('gpu'):
        for proc in node.iterfind('processes'):
            for pr in proc.iterfind('process_info'):
                for pid in pr.iterfind('pid'):
                    pidList.append(pid.text)
                for name in pr.iterfind('process_name'):
                    processList.append(name.text)

    if len(pidList) != 0:
        logger.warning("UNABLE TO CONTINUE, GPUs ARE IN USE! MAKE SURE THAT THE GPUS ARE FREE AND TRY AGAIN!")
        logger.info("Gpus are being used by processes below: ")
        logger.info("Process ID: %s" % pidList)
        logger.info("Process Name: %s" % processList)
        logger.info()

        return False

    return True

def get_output():
    """
    Executes nvidia-smi -q and returns the output
    """
    cmd = "nvidia-smi -q"
    try:
        nvsmiData = subprocess.check_output(shlex.split(cmd)).decode('utf-8')
        return (nvsmiData, None)
    except subprocess.CalledProcessError as e:
        print(("Unable to collect \"nvidia-smi -q\" output. Error:\n%s" % e))
        return (None, e.output)

def enable_persistence_mode():
    """
    Attempts to enable persistence mode via nvidia-smi
    """
    cmd = "nvidia-smi -pm 1"
    try:
        nvsmiData = subprocess.check_output(shlex.split(cmd)).decode('utf-8')
        return (nvsmiData, None)
    except subprocess.CalledProcessError as e:
        print(("Failed to enable persistence mode. Error:\n%s" % e))
        return (None, e.output)

################################################################################
### Simple function to print out some values from nvidia-smi commands
################################################################################
def main():
    #sc = check_sanity_nvml.SanityChecker()
    j =  NvidiaSmiJob()
    j.start()
    time.sleep(20)
    j.m_shutdownFlag.set()
    j.join()
    print("Data:")
    print((j.m_data))

if __name__ == '__main__':
    main()
