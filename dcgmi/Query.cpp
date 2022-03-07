/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
/*
 * Query.cpp
 *
 */

#include "Query.h"
#include "CommandOutputController.h"
#include "DcgmiOutput.h"
#include "dcgm_agent.h"
#include "dcgm_structs.h"
#include <ctype.h>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string.h>


/**************************************************************************************/

const static char c_GpuIdTag[]      = "GPU ID";
const static char c_DeviceInfoTag[] = "Device Information";

#define GPU_BUS_ID_TAG "<GPU_PCI_BUS_ID"
#define GPU_UUID_TAG   "<GPU_UUID"
#define GPU_ID_TAG     "<GPUID"
#define GPU_NAME_TAG   "<GPU_NAME"

static const char c_name[]       = "Name";
static const char c_busId[]      = "PCI Bus ID";
static const char c_deviceUuid[] = "Device UUID";

#define ERROR_RETURN_TAG  "<ERROR_RETURN_STRING"
#define ERROR_MESSAGE_TAG "<ERROR_MESSAGE"


/* Device Info */
static char QUERY_DEVICE_HEADER[] = "+--------------------------+-------------------------------------------------+\n"
                                    "| <HEADER_INFO           > | Device Information                              |\n"
                                    "+==========================+=================================================+\n";

static char QUERY_ATTRIBUTE_DATA[] = "| <ATTRIBUTE              >| <DEVICE_ATTRIBUTE_INFO                         >|\n";

static char QUERY_ATTRIBUTE_FOOTER[]
    = "+--------------------------+-------------------------------------------------+\n";

#define HEADER_TAG         "<HEADER_INFO"
#define ATTRIBUTE_TAG      "<ATTRIBUTE"
#define ATTRIBUTE_DATA_TAG "<DEVICE_ATTRIBUTE_INFO"

static const char c_gpuId[] = "GPU ID";
static const char c_info[]  = "Device Information";

static const char c_switchId[] = "Switch ID";

/*****************************************************************************************/

Query::Query()
{
    // TODO Auto-generated constructor stub
}

Query::~Query()
{
    // TODO Auto-generated destructor stub
}

/********************************************************************************/
dcgmReturn_t Query::DisplayDiscoveredDevices(dcgmHandle_t dcgmHandle)
{
    dcgmReturn_t result;
    dcgmDeviceAttributes_t stDeviceAttributes;
    std::vector<dcgm_field_eid_t> entityIds;

    std::string entityId;

    {
        DcgmiOutputColumns outColumns;
        DcgmiOutput &out = outColumns;

        DcgmiOutputFieldSelector gpuIdSelector   = DcgmiOutputFieldSelector().child(c_gpuId);
        DcgmiOutputFieldSelector gpuInfoSelector = DcgmiOutputFieldSelector().child(c_info);

        out.setOption(DCGMI_OUTPUT_OPTIONS_SEPARATE_SECTIONS, true);

        out.addColumn(8, c_gpuId, gpuIdSelector);
        out.addColumn(70, c_info, gpuInfoSelector);

        /* Display the GPUs in the system */

        result = HelperGetEntityList(dcgmHandle, DCGM_FE_GPU, entityIds);
        if (DCGM_ST_OK != result)
        {
            std::cout << "Error: Cannot get GPU list from remote node. Return: " << errorString(result) << std::endl;
            PRINT_ERROR("%d", "Cannot get GPU list from remote node. Return: %d", result);
            return result;
        }

        std::cout << entityIds.size() << " GPU" << (entityIds.size() == 1 ? "" : "s") << " found." << std::endl;
        for (unsigned int i = 0; i < entityIds.size(); i++)
        {
            stDeviceAttributes.version = dcgmDeviceAttributes_version;
            result                     = dcgmGetDeviceAttributes(dcgmHandle, entityIds[i], &stDeviceAttributes);

            entityId = std::to_string(entityIds[i]);

            out[entityId][c_gpuId] = entityIds[i];

            if (DCGM_ST_OK != result)
            {
                out[entityId][c_info].setOrAppend("Error: Cannot get device attributes for GPU.");
                out[entityId][c_info].setOrAppend(std::string("Return: ") + errorString(result));
                PRINT_ERROR(
                    "%d %d", "Error getting device attributes with GPU ID: %d. Return: %d", entityIds[i], result);
            }
            else
            {
                out[entityId][c_info].setOrAppend(std::string(c_name) + ": "
                                                  + stDeviceAttributes.identifiers.deviceName);
                out[entityId][c_info].setOrAppend(std::string(c_busId) + ": "
                                                  + stDeviceAttributes.identifiers.pciBusId);
                out[entityId][c_info].setOrAppend(std::string(c_deviceUuid) + ": "
                                                  + stDeviceAttributes.identifiers.uuid);
            }
        }
        std::cout << out.str();
    }

    {
        DcgmiOutputColumns outColumns;
        DcgmiOutput &out = outColumns;

        DcgmiOutputFieldSelector switchIdSelector = DcgmiOutputFieldSelector().child(c_switchId);

        out.addColumn(11, c_switchId, switchIdSelector);
        /* display the NvSwitches in the system */

        result = HelperGetEntityList(dcgmHandle, DCGM_FE_SWITCH, entityIds);
        if (DCGM_ST_OK != result)
        {
            std::cout << "Error: Cannot get NvSwitch list from remote node. Return: " << errorString(result)
                      << std::endl;
            PRINT_ERROR("%d", "Cannot get NvSwitch list from remote node. Return: %d", result);
            return result;
        }

        std::cout << entityIds.size() << " NvSwitch" << (entityIds.size() == 1 ? "" : "es") << " found." << std::endl;
        for (unsigned int i = 0; i < entityIds.size(); i++)
        {
            out[std::to_string(entityIds[i])][c_switchId] = entityIds[i];
        }
        std::cout << out.str();
    }
    return DCGM_ST_OK;
}
/********************************************************************************/
dcgmReturn_t Query::DisplayDeviceInfo(dcgmHandle_t dcgmHandle,
                                      unsigned int requestedGpuId,
                                      std::string const &attributes)
{
    dcgmDeviceAttributes_t stDeviceAttributes;
    stDeviceAttributes.version      = dcgmDeviceAttributes_version;
    CommandOutputController cmdView = CommandOutputController();
    std::stringstream ss;

    // Check if input attribute flags are valid
    dcgmReturn_t result = HelperValidInput(attributes);
    if (DCGM_ST_OK != result)
    {
        std::cout << "Error: Invalid flags detected. Return: " << errorString(result) << std::endl;
        return result;
    }

    result = dcgmGetDeviceAttributes(dcgmHandle, requestedGpuId, &stDeviceAttributes);

    if (result != DCGM_ST_OK)
    {
        std::cout << "Error: Unable to get GPU info. Return: " << errorString(result) << std::endl;
        DCGM_LOG_ERROR << "Error getting device attributes with GPU ID: " << requestedGpuId << ". Return: " << result;
        return result;
    }
    else if (!stDeviceAttributes.identifiers.brandName[0])
    { // This should be there if the gpu was found
        std::cout << "Error: Unable to get GPU info. Return: Bad parameter passed to function.\n";
        DCGM_LOG_ERROR << "Error getting device attributes with GPU ID: " << requestedGpuId << ". Return: " << result;
        return DCGM_ST_BADPARAM;
    }
    else
    {
        // Parse tags and output selected parameters
        cmdView.setDisplayStencil(QUERY_DEVICE_HEADER);
        ss << "GPU ID: " << requestedGpuId;
        cmdView.addDisplayParameter(HEADER_TAG, ss.str());
        cmdView.display();

        for (char attribute : attributes)
        {
            switch (attribute)
            {
                case 'p':

                    HelperDisplayPowerLimits(stDeviceAttributes.powerLimits, 0);
                    std::cout << QUERY_ATTRIBUTE_FOOTER;

                    break;
                case 't':

                    HelperDisplayThermals(stDeviceAttributes.thermalSettings, 0);
                    std::cout << QUERY_ATTRIBUTE_FOOTER;

                    break;
                case 'c':

                    HelperDisplayClocks(stDeviceAttributes.clockSets);
                    std::cout << QUERY_ATTRIBUTE_FOOTER;

                    break;
                case 'a':

                    HelperDisplayIdentifiers(stDeviceAttributes.identifiers, 0);
                    std::cout << QUERY_ATTRIBUTE_FOOTER;

                    break;
                default:
                    // Should never run
                    DCGM_LOG_ERROR << "Unexpected error in querying GPU " << requestedGpuId << ".";
                    break;
            }
        }
    }

    return DCGM_ST_OK;
}

/********************************************************************************/

dcgmReturn_t Query::DisplayGroupInfo(dcgmHandle_t mNvcmHandle,
                                     unsigned int requestedGroupId,
                                     std::string const &attributes,
                                     bool verbose)
{
    dcgmReturn_t result = DCGM_ST_OK;
    dcgmGroupInfo_t stNvcmGroupInfo;

    stNvcmGroupInfo.version = dcgmGroupInfo_version;
    result = dcgmGroupGetInfo(mNvcmHandle, (dcgmGpuGrp_t)(long long)requestedGroupId, &stNvcmGroupInfo);
    if (DCGM_ST_OK != result)
    {
        std::string error = (result == DCGM_ST_NOT_CONFIGURED) ? "The Group is not found" : errorString(result);
        std::cout << "Error: Cannot get group info from remote node. Return: " << error << std::endl;
        PRINT_ERROR("%d %d", "Error getting group info with Group ID: %d. Return: %d", requestedGroupId, result);
        return result;
    }

    if (stNvcmGroupInfo.count == 0)
    {
        std::cout << "No devices in group.\n";
    }
    else if (!verbose)
    {
        result = HelperDisplayNonVerboseGroup(mNvcmHandle, stNvcmGroupInfo, attributes);
    }
    else
    {
        std::cout << "Device info: " << std::endl;
        for (unsigned int i = 0; i < stNvcmGroupInfo.count; i++)
        {
            if (stNvcmGroupInfo.entityList[i].entityGroupId != DCGM_FE_GPU)
            {
                std::cout << DcgmFieldsGetEntityGroupString(stNvcmGroupInfo.entityList[i].entityGroupId)
                          << " id: " << stNvcmGroupInfo.entityList[i].entityId << std::endl;
                continue;
            }

            result = DisplayDeviceInfo(mNvcmHandle, stNvcmGroupInfo.entityList[i].entityId, attributes);

            if (result != DCGM_ST_OK)
            {
                break;
            }
        }
    }

    return result;
}

/********************************************************************************/
dcgmReturn_t Query::HelperDisplayNonVerboseGroup(dcgmHandle_t mNvcmHandle,
                                                 dcgmGroupInfo_t &stNvcmGroupInfo,
                                                 std::string attributes)
{
    dcgmReturn_t result;
    std::unique_ptr<dcgmDeviceAttributes_t[]> stDeviceAttributes { new dcgmDeviceAttributes_t[stNvcmGroupInfo.count] };
    CommandOutputController cmdView = CommandOutputController();
    std::stringstream ss;
    bool allTheSame = true;
    unsigned int bitvector;

    // Check if input attribute flags are valid
    result = HelperValidInput(attributes);
    if (DCGM_ST_OK != result)
    {
        std::cout << "Error: Invalid flags detected. Return: " << errorString(result) << std::endl;
        return result;
    }

    for (unsigned int i = 0; i < stNvcmGroupInfo.count; i++)
    {
        if (stNvcmGroupInfo.entityList[i].entityGroupId != DCGM_FE_GPU)
        {
            std::cout << DcgmFieldsGetEntityGroupString(stNvcmGroupInfo.entityList[i].entityGroupId)
                      << " id: " << stNvcmGroupInfo.entityList[i].entityId << std::endl;
            continue;
        }

        stDeviceAttributes[i].version = dcgmDeviceAttributes_version;
        result = dcgmGetDeviceAttributes(mNvcmHandle, stNvcmGroupInfo.entityList[i].entityId, &stDeviceAttributes[i]);

        if (result != DCGM_ST_OK)
        {
            std::cout << "Error: Unable to get GPU info. Return: " << errorString(result) << std::endl;
            PRINT_ERROR("%d %d",
                        "Error getting device attributes with GPU ID: %d. Return: %d",
                        stNvcmGroupInfo.entityList[i].entityId,
                        result);
            return result;
        }
    }

    // Parse tags and output selected parameters
    cmdView.setDisplayStencil(QUERY_DEVICE_HEADER);
    ss << "Group of " << stNvcmGroupInfo.count << " GPUs";
    cmdView.addDisplayParameter(HEADER_TAG, ss.str());
    cmdView.display();

    cmdView.setDisplayStencil(QUERY_ATTRIBUTE_DATA);
    // Display Info
    for (unsigned int i = 0; i < attributes.length(); i++)
    {
        switch (attributes.at(i))
        {
            case 'p':
                bitvector = 0;

                // See if all GPUS match
                for (unsigned int i = 1; i < stNvcmGroupInfo.count; i++)
                {
                    if (stDeviceAttributes[0].powerLimits.curPowerLimit
                        != stDeviceAttributes[i].powerLimits.curPowerLimit)
                    {
                        bitvector |= (1 << 0); // flip bit for curPower limit to be replaced with **** in display
                    }
                    if (stDeviceAttributes[0].powerLimits.defaultPowerLimit
                        != stDeviceAttributes[i].powerLimits.defaultPowerLimit)
                    {
                        bitvector |= (1 << 1);
                    }
                    if (stDeviceAttributes[0].powerLimits.maxPowerLimit
                        != stDeviceAttributes[i].powerLimits.maxPowerLimit)
                    {
                        bitvector |= (1 << 2);
                    }
                    if (stDeviceAttributes[0].powerLimits.minPowerLimit
                        != stDeviceAttributes[i].powerLimits.minPowerLimit)
                    {
                        bitvector |= (1 << 3);
                    }
                    if (stDeviceAttributes[0].powerLimits.enforcedPowerLimit
                        != stDeviceAttributes[i].powerLimits.enforcedPowerLimit)
                    {
                        bitvector |= (1 << 4);
                    }
                }

                HelperDisplayPowerLimits(stDeviceAttributes[0].powerLimits, bitvector);

                std::cout << QUERY_ATTRIBUTE_FOOTER;

                break;

            case 't':
                bitvector = 0;

                // See if all GPUS match
                for (unsigned int i = 1; i < stNvcmGroupInfo.count; i++)
                {
                    if (stDeviceAttributes[0].thermalSettings.shutdownTemp
                        != stDeviceAttributes[i].thermalSettings.shutdownTemp)
                    {
                        bitvector |= (1 << 0); // flip bit for shutdown limit to be replaced with **** in display
                    }
                    if (stDeviceAttributes[0].thermalSettings.slowdownTemp
                        != stDeviceAttributes[i].thermalSettings.slowdownTemp)
                    {
                        bitvector |= (1 << 1);
                    }
                }

                HelperDisplayThermals(stDeviceAttributes[0].thermalSettings, bitvector);

                std::cout << QUERY_ATTRIBUTE_FOOTER;

                break;

            case 'c':

                allTheSame = true;

                // See if all GPUS match
                for (unsigned int i = 1; i < stNvcmGroupInfo.count; i++)
                {
                    if (stDeviceAttributes[0].clockSets.count != stDeviceAttributes[i].clockSets.count)
                    {
                        allTheSame = false;
                        break;
                    }
                    else if (stDeviceAttributes[0].clockSets.version != stDeviceAttributes[i].clockSets.version)
                    {
                        allTheSame = false;
                        break;
                    }
                }

                // Now check if all clocks match
                if (allTheSame)
                {
                    std::multimap<unsigned int, unsigned int> clocksMap;
                    for (unsigned int j = 0; j < stDeviceAttributes[0].clockSets.count; j++)
                    {
                        clocksMap.insert(std::make_pair(stDeviceAttributes[0].clockSets.clockSet[j].memClock,
                                                        stDeviceAttributes[0].clockSets.clockSet[j].smClock));
                    }

                    for (unsigned int i = 1; i < stNvcmGroupInfo.count; i++)
                    {
                        for (unsigned int j = 0; j < stDeviceAttributes[i].clockSets.count; j++)
                        {
                            if (clocksMap.find(stDeviceAttributes[i].clockSets.clockSet[j].memClock) == clocksMap.end())
                            {
                                allTheSame = false;
                                break;
                            }
                            else
                            {
                                std::pair<std::multimap<unsigned int, unsigned int>::iterator,
                                          std::multimap<unsigned int, unsigned int>::iterator>
                                    ret;
                                ret = clocksMap.equal_range(stDeviceAttributes[i].clockSets.clockSet[j].memClock);
                                bool matchedClock = false;
                                for (std::multimap<unsigned int, unsigned int>::iterator it = ret.first;
                                     it != ret.second;
                                     ++it)
                                {
                                    if (stDeviceAttributes[i].clockSets.clockSet[j].smClock == it->second)
                                    {
                                        matchedClock = true;
                                        break;
                                    }
                                }

                                if (!matchedClock)
                                {
                                    allTheSame = false;
                                    break;
                                }
                            }
                        }
                    }
                }


                if (allTheSame)
                {
                    HelperDisplayClocks(stDeviceAttributes[0].clockSets);
                }
                else
                {
                    cmdView.addDisplayParameter(ATTRIBUTE_TAG, "Supported Clocks (MHz)");
                    cmdView.addDisplayParameter(ATTRIBUTE_DATA_TAG, "****");
                    cmdView.display();
                }

                std::cout << QUERY_ATTRIBUTE_FOOTER;

                break;

            case 'a':
                bitvector = 0;
                // See if all GPUS match
                for (unsigned int i = 1; i < stNvcmGroupInfo.count; i++)
                {
                    if (strcmp(stDeviceAttributes[0].identifiers.deviceName,
                               stDeviceAttributes[i].identifiers.deviceName))
                    {
                        bitvector |= (1 << 0); // flip bit for deviceName to be replaced with **** in display
                    }
                    if (strcmp(stDeviceAttributes[0].identifiers.serial, stDeviceAttributes[i].identifiers.serial))
                    {
                        bitvector |= (1 << 3);
                    }
                    if (strcmp(stDeviceAttributes[0].identifiers.inforomImageVersion,
                               stDeviceAttributes[i].identifiers.inforomImageVersion))
                    {
                        bitvector |= (1 << 4);
                    }
                    if (strcmp(stDeviceAttributes[0].identifiers.vbios, stDeviceAttributes[i].identifiers.vbios))
                    {
                        bitvector |= (1 << 5);
                    }
                }
                // UUID and BusID will always be different so we switch on their bits
                if (stNvcmGroupInfo.count >= 2)
                {
                    bitvector |= ((1 << 1) | (1 << 2));
                }

                HelperDisplayIdentifiers(stDeviceAttributes[0].identifiers, bitvector);
                std::cout << QUERY_ATTRIBUTE_FOOTER;

                break;

            default:
                // Should never run
                PRINT_ERROR("", "Unexpected Error.");
                break;
        }
    }

    std::cout << "**** Non-homogenous settings across group. Use with –v flag to see details.\n";

    return DCGM_ST_OK;
}


void Query::HelperDisplayClocks(dcgmDeviceSupportedClockSets_t &clocks)
{
    CommandOutputController cmdView = CommandOutputController();
    cmdView.setDisplayStencil(QUERY_ATTRIBUTE_DATA);
    cmdView.addDisplayParameter(ATTRIBUTE_TAG, "Supported Clocks (MHz)");
    cmdView.addDisplayParameter(ATTRIBUTE_DATA_TAG, "Memory Clock, SM Clock");
    cmdView.display();

    if (clocks.count == 0)
    {
        cmdView.addDisplayParameter(ATTRIBUTE_TAG, "");
        cmdView.addDisplayParameter(ATTRIBUTE_DATA_TAG, DCGM_INT32_NOT_SUPPORTED);
        cmdView.display();
    }

    for (unsigned int k = 0; k < clocks.count; k++)
    {
        cmdView.addDisplayParameter(ATTRIBUTE_TAG, "");
        cmdView.addDisplayParameter(ATTRIBUTE_DATA_TAG, HelperFormatClock(clocks.clockSet[k]));
        cmdView.display();
    }
}

void Query::HelperDisplayThermals(dcgmDeviceThermals_t thermals, unsigned int bitvector)
{
    CommandOutputController cmdView = CommandOutputController();
    cmdView.setDisplayStencil(QUERY_ATTRIBUTE_DATA);

    cmdView.addDisplayParameter(ATTRIBUTE_TAG, "Shutdown Temperature (C)");
    if (bitvector & (1 << 0))
    { // check if bit for shutdown temp is flipped
        cmdView.addDisplayParameter(ATTRIBUTE_DATA_TAG, "****");
    }
    else
    {
        cmdView.addDisplayParameter(ATTRIBUTE_DATA_TAG, thermals.shutdownTemp);
    }
    cmdView.display();

    cmdView.addDisplayParameter(ATTRIBUTE_TAG, "Slowdown Temperature (C)");
    if (bitvector & (1 << 1))
    {
        cmdView.addDisplayParameter(ATTRIBUTE_DATA_TAG, "****");
    }
    else
    {
        cmdView.addDisplayParameter(ATTRIBUTE_DATA_TAG, thermals.slowdownTemp);
    }
    cmdView.display();
}
void Query::HelperDisplayPowerLimits(dcgmDevicePowerLimits_t powerLimits, unsigned int bitvector)
{
    CommandOutputController cmdView = CommandOutputController();
    cmdView.setDisplayStencil(QUERY_ATTRIBUTE_DATA);

    cmdView.addDisplayParameter(ATTRIBUTE_TAG, "Current Power Limit (W)");
    if (bitvector & (1 << 0))
    { // check if bit for pwr limit is flipped
        cmdView.addDisplayParameter(ATTRIBUTE_DATA_TAG, "****");
    }
    else
    {
        cmdView.addDisplayParameter(ATTRIBUTE_DATA_TAG, powerLimits.curPowerLimit);
    }
    cmdView.display();

    cmdView.addDisplayParameter(ATTRIBUTE_TAG, "Default Power Limit (W)");
    if (bitvector & (1 << 1))
    {
        cmdView.addDisplayParameter(ATTRIBUTE_DATA_TAG, "****");
    }
    else
    {
        cmdView.addDisplayParameter(ATTRIBUTE_DATA_TAG, powerLimits.defaultPowerLimit);
    }
    cmdView.display();

    cmdView.addDisplayParameter(ATTRIBUTE_TAG, "Max Power Limit (W)");
    if (bitvector & (1 << 2))
    {
        cmdView.addDisplayParameter(ATTRIBUTE_DATA_TAG, "****");
    }
    else
    {
        cmdView.addDisplayParameter(ATTRIBUTE_DATA_TAG, powerLimits.maxPowerLimit);
    }
    cmdView.display();

    cmdView.addDisplayParameter(ATTRIBUTE_TAG, "Min Power Limit (W)");
    if (bitvector & (1 << 3))
    {
        cmdView.addDisplayParameter(ATTRIBUTE_DATA_TAG, "****");
    }
    else
    {
        cmdView.addDisplayParameter(ATTRIBUTE_DATA_TAG, powerLimits.minPowerLimit);
    }
    cmdView.display();

    cmdView.addDisplayParameter(ATTRIBUTE_TAG, "Enforced Power Limit (W)");
    if (bitvector & (1 << 4))
    {
        cmdView.addDisplayParameter(ATTRIBUTE_DATA_TAG, "****");
    }
    else
    {
        cmdView.addDisplayParameter(ATTRIBUTE_DATA_TAG, powerLimits.enforcedPowerLimit);
    }
    cmdView.display();
}
void Query::HelperDisplayIdentifiers(dcgmDeviceIdentifiers_t &identifiers, unsigned int bitvector)
{
    CommandOutputController cmdView = CommandOutputController();
    cmdView.setDisplayStencil(QUERY_ATTRIBUTE_DATA);

    cmdView.addDisplayParameter(ATTRIBUTE_TAG, "Device Name");
    if (bitvector & (1 << 0))
    { // check if bit for device name is flipped
        cmdView.addDisplayParameter(ATTRIBUTE_DATA_TAG, "****");
    }
    else
    {
        cmdView.addDisplayParameter(ATTRIBUTE_DATA_TAG, identifiers.deviceName);
    }
    cmdView.display();

    cmdView.addDisplayParameter(ATTRIBUTE_TAG, "PCI Bus ID");
    if (bitvector & (1 << 1))
    {
        cmdView.addDisplayParameter(ATTRIBUTE_DATA_TAG, "****");
    }
    else
    {
        cmdView.addDisplayParameter(ATTRIBUTE_DATA_TAG, identifiers.pciBusId);
    }
    cmdView.display();

    cmdView.addDisplayParameter(ATTRIBUTE_TAG, "UUID");
    if (bitvector & (1 << 2))
    {
        cmdView.addDisplayParameter(ATTRIBUTE_DATA_TAG, "****");
    }
    else
    {
        cmdView.addDisplayParameter(ATTRIBUTE_DATA_TAG, identifiers.uuid);
    }
    cmdView.display();

    cmdView.addDisplayParameter(ATTRIBUTE_TAG, "Serial Number");
    if (bitvector & (1 << 3))
    {
        cmdView.addDisplayParameter(ATTRIBUTE_DATA_TAG, "****");
    }
    else
    {
        cmdView.addDisplayParameter(ATTRIBUTE_DATA_TAG, identifiers.serial);
    }
    cmdView.display();

    cmdView.addDisplayParameter(ATTRIBUTE_TAG, "InfoROM Version");
    if (bitvector & (1 << 4))
    {
        cmdView.addDisplayParameter(ATTRIBUTE_DATA_TAG, "****");
    }
    else
    {
        cmdView.addDisplayParameter(ATTRIBUTE_DATA_TAG, identifiers.inforomImageVersion);
    }
    cmdView.display();

    cmdView.addDisplayParameter(ATTRIBUTE_TAG, "VBIOS");
    if (bitvector & (1 << 5))
    {
        cmdView.addDisplayParameter(ATTRIBUTE_DATA_TAG, "****");
    }
    else
    {
        cmdView.addDisplayParameter(ATTRIBUTE_DATA_TAG, identifiers.vbios);
    }
    cmdView.display();
}

/********************************************************************************/
dcgmReturn_t Query::DisplayHierarchyInfo(dcgmHandle_t handle)
{
    dcgmMigHierarchy_v2 hierarchy {};
    hierarchy.version = dcgmMigHierarchy_version2;

    dcgmReturn_t ret = dcgmGetGpuInstanceHierarchy(handle, &hierarchy);
    if (ret != DCGM_ST_OK)
    {
        std::cout << "Error: Cannot get hierarchy information from the hostengine. Return: " << errorString(ret)
                  << std::endl;
        DCGM_LOG_ERROR << "Error discovering hierarchy information from the hostengine. Return: " << ret;
        return ret;
    }

    std::cout << FormatMigHierarchy(hierarchy);

    return DCGM_ST_OK;
}


/********************************************************************************/
dcgmReturn_t Query::HelperGetEntityList(dcgmHandle_t dcgmHandle,
                                        dcgm_field_entity_group_t entityGroup,
                                        std::vector<dcgm_field_eid_t> &entityIds)
{
    dcgmReturn_t result;
    dcgm_field_eid_t entities[DCGM_MAX_NUM_DEVICES];
    int numItems = DCGM_MAX_NUM_DEVICES;
    int i;

    entityIds.clear();

    result = dcgmGetEntityGroupEntities(dcgmHandle, entityGroup, entities, &numItems, 0);
    if (DCGM_ST_OK != result)
    {
        std::cout << "Error: Cannot get devices from remote node. Return: " << errorString(result) << std::endl;
        PRINT_ERROR("%d", "Error discovering devices from remote node. Return: %d", result);
        return result;
    }

    for (i = 0; i < numItems; i++)
    {
        entityIds.push_back(entities[i]);
    }

    return DCGM_ST_OK;
}

/********************************************************************************/
std::string Query::HelperFormatClock(dcgmClockSet_t clock)
{
    std::stringstream ss;

    ss << clock.memClock;
    ss << ", ";
    ss << clock.smClock;

    return ss.str();
}

/********************************************************************************/
dcgmReturn_t Query::HelperValidInput(std::string const &attributes)
{
    char matches[] = "aptc";

    // Check for valid input
    if (attributes.length() > strlen(matches))
    {
        std::cout
            << "Error: Invalid input. Please include only one of each valid tag.\n Example:./dcgmi discovery --gpuid 1 -i apt\n";
        PRINT_ERROR("%s", "Error parsing for attributes. Invalid input detected: %s", attributes.c_str());
        return DCGM_ST_BADPARAM;
    }

    bool hasBeenSeen[5] = { false };
    unsigned int count  = 0;
    for (unsigned int j = 0; j < attributes.length(); j++)
    {
        for (unsigned int i = 0; i < strlen(matches); i++)
        {
            if (attributes.at(j) == matches[i])
            {
                if (hasBeenSeen[i])
                {
                    std::cout
                        << "Error: Invalid input. Please include only one of each tag.\n Example:./dcgmi discovery --gpuid 1 -i apt\n";
                    PRINT_ERROR("%s", "Error parsing for attributes. Invalid input detected: %s", attributes.c_str());
                    return DCGM_ST_BADPARAM;
                }
                else
                {
                    hasBeenSeen[i] = true;
                    count++;
                }
            }
        }
    }

    if (count != attributes.length())
    {
        std::cout
            << "Invalid input. Please include only valid tags.\n Example:./dcgmi discovery --gpuid 1 -i a \n Type ./dcgmi discovery -h for more help.\n";
        PRINT_ERROR("%s", "Error parsing for attributes. Invalid input detected: %s", attributes.c_str());
        return DCGM_ST_BADPARAM;
    }

    return DCGM_ST_OK;
}

/*****************************************************************************
 *****************************************************************************
 *Query Device Info Invoker
 *****************************************************************************
 *****************************************************************************/

/*****************************************************************************/
QueryDeviceInfo::QueryDeviceInfo(std::string hostname, unsigned int device, std::string attributes)
    : deviceNum(device)
    , attributes(std::move(attributes))
{
    m_hostName = std::move(hostname);
}

/*****************************************************************************/
dcgmReturn_t QueryDeviceInfo::DoExecuteConnected()
{
    return queryObj.DisplayDeviceInfo(m_dcgmHandle, deviceNum, attributes);
}

/*****************************************************************************
 *****************************************************************************
 *Query Group Info Invoker
 *****************************************************************************
 *****************************************************************************/

/*****************************************************************************/
QueryGroupInfo::QueryGroupInfo(std::string hostname, unsigned int group, std::string attributes, bool verbose)
    : groupNum(group)
    , attributes(std::move(attributes))
    , verbose(verbose)
{
    m_hostName = std::move(hostname);
}

/*****************************************************************************/
dcgmReturn_t QueryGroupInfo::DoExecuteConnected()
{
    return queryObj.DisplayGroupInfo(m_dcgmHandle, groupNum, attributes, verbose);
}


/*****************************************************************************
 *****************************************************************************
 * Query Device List
 *****************************************************************************
 *****************************************************************************/

/*****************************************************************************/
QueryDeviceList::QueryDeviceList(std::string hostname)
{
    m_hostName = std::move(hostname);
}

/*****************************************************************************/
dcgmReturn_t QueryDeviceList::DoExecuteConnected()
{
    return queryObj.DisplayDiscoveredDevices(m_dcgmHandle);
}

/*****************************************************************************
 *****************************************************************************
 * Query Hierarchy Info
 *****************************************************************************
 *****************************************************************************/

/*****************************************************************************/
QueryHierarchyInfo::QueryHierarchyInfo(std::string hostname)
{
    m_hostName = std::move(hostname);
}

/*****************************************************************************/
dcgmReturn_t QueryHierarchyInfo::DoExecuteConnected()
{
    return m_queryObj.DisplayHierarchyInfo(m_dcgmHandle);
}

void TopologicalSort(dcgmMigHierarchy_v2 &hierarchy)
{
    /*
     * In the dcgmMigHierarchy_v2 we store only GPU_I and GPU_CI. There are no GPUs.
     * To sort them properly, we need the following order:
     *      GPU 0
     *          GPU_I 0
     *              GPU_CI 0
     *              GPU_CI 1
     *          GPU_I 1
     *              GPU_CI 0
     *              GPU_CI 1
     *      GPU 1
     *          ...
     * Where indices after GPU_CI and GPU_I are NVML indices, not entityIds.
     */

    std::sort(&hierarchy.entityList[0],
              &hierarchy.entityList[hierarchy.count],
              [&](dcgmMigHierarchyInfo_v2 const &left, dcgmMigHierarchyInfo_v2 const &right) {
                  return std::tie(left.info.nvmlGpuIndex,
                                  left.info.nvmlInstanceId,
                                  left.entity.entityGroupId,
                                  left.info.nvmlComputeInstanceId)
                         < std::tie(right.info.nvmlGpuIndex,
                                    right.info.nvmlInstanceId,
                                    right.entity.entityGroupId,
                                    right.info.nvmlComputeInstanceId);
              });
}

std::string FormatMigHierarchy(dcgmMigHierarchy_v2 &hierarchy)
{
    DcgmiOutputTree outTree(20, 70);
    DcgmiOutput &out = outTree;

    out.setOption(DCGMI_OUTPUT_OPTIONS_SEPARATE_SECTIONS, true);

    TopologicalSort(hierarchy);

    out.addHeader("Instance Hierarchy");

    std::set<unsigned int> printedGpus;
    std::string gpuId;
    std::string instanceId;

    for (unsigned int i = 0; i < hierarchy.count; i++)
    {
        auto const &entity   = hierarchy.entityList[i];
        const auto &entityId = entity.entity.entityId;
        switch (entity.entity.entityGroupId)
        {
            case DCGM_FE_GPU_I:
            {
                // Display the GPU the first time we see it
                if (printedGpus.insert(entity.parent.entityId).second == true)
                {
                    gpuId      = "GPU " + std::to_string(entity.info.nvmlGpuIndex);
                    out[gpuId] = "GPU " + std::string(entity.info.gpuUuid)
                                 + " (EntityID: " + std::to_string(entity.parent.entityId) + ")";
                }

                // Display the Instance
                instanceId = "I " + std::to_string(entity.info.nvmlGpuIndex) + "/"
                             + std::to_string(entity.info.nvmlInstanceId);
                out[gpuId][instanceId] = "GPU Instance (EntityID: " + std::to_string(entityId) + ")";
                break;
            }
            case DCGM_FE_GPU_CI:
            {
                const std::string computeInstanceId = "CI " + std::to_string(entity.info.nvmlGpuIndex) + "/"
                                                      + std::to_string(entity.info.nvmlInstanceId) + "/"
                                                      + std::to_string(entity.info.nvmlComputeInstanceId);
                out[gpuId][instanceId][computeInstanceId]
                    = "Compute Instance (EntityID: " + std::to_string(entityId) + ")";
                break;
            }
            default:
                // Should never get here
                DCGM_LOG_ERROR << "Incorrect entity group " << entity.entity.entityGroupId
                               << " detected when displaying hierarchy";
                break;
        }
    }

    return out.str();
}
