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
#include "Topo.h"
#include "CommandOutputController.h"
#include "DcgmiOutput.h"
#include "dcgm_agent.h"
#include "dcgm_structs.h"
#include <bitset>
#include <cstring>
#include <iostream>
#include <sstream>
#include <stdexcept>


/* Device Info */
#define MAX_SIZE_OF_AFFINITY_STRING 54 /* Used for overflow (ATTRIBUTE_DATA_TAG tag) */

#define HEADER_NAME "Topology Information"


/********************************************************************************/
dcgmReturn_t Topo::DisplayGPUTopology(dcgmHandle_t mNvcmHandle, unsigned int requestedGPUId, bool json)
{
    dcgmReturn_t result = DCGM_ST_OK;
    dcgmDeviceTopology_t gpuTopo;
    gpuTopo.version = dcgmDeviceTopology_version;
    DcgmiOutputTree outTree(20, 80);
    DcgmiOutputJson outJson;
    DcgmiOutput &out = json ? (DcgmiOutput &)outJson : (DcgmiOutput &)outTree;
    std::stringstream ss;

    // Get topology
    result = dcgmGetDeviceTopology(mNvcmHandle, requestedGPUId, &gpuTopo);
    if (result == DCGM_ST_NOT_SUPPORTED)
    {
        std::cout << "Getting topology is not supported for GPU " << requestedGPUId << std::endl;
        PRINT_INFO("%u", "Getting topology is not supported for GPU: %u", requestedGPUId);
        return result;
    }
    else if (result != DCGM_ST_OK)
    {
        std::cout << "Error: unable to get topology for GPU " << requestedGPUId << ". Return: " << errorString(result)
                  << "." << std::endl;
        PRINT_ERROR("%u %d", "Error gettting topology for gpu: %u. Return: %d", requestedGPUId, result);
        return result;
    }

    // Header
    ss << "GPU ID: " << requestedGPUId;
    out.addHeader(HEADER_NAME);
    out.addHeader(ss.str());

    // Affinity

    std::string strHold = HelperGetAffinity(gpuTopo.cpuAffinityMask);
    unsigned int p      = 0;
    unsigned int start  = 0;

    if (strHold.length() > MAX_SIZE_OF_AFFINITY_STRING)
    {
        while (start < strHold.length())
        {
            p += MAX_SIZE_OF_AFFINITY_STRING;
            if (p >= strHold.length())
                p = strHold.length() - 1;

            else
            { // Put pointer to last available digit
                while (isdigit(strHold.at(p)))
                {
                    if (p + 1 < strHold.length() && !isdigit(strHold.at(p + 1)))
                        break; // check if landed on end of a digit
                    p--;
                }
                while (!isdigit(strHold.at(p)))
                {
                    p--;
                }
            }
            // p is now the index of a the last digit of a CPU

            // Comma case
            if (p + 1 < strHold.length() && strHold.at(p + 1) == ',')
            {
                ss.str(strHold.substr(start, p - start + 2));
            }
            else
            {
                // Hyphen case
                ss.str(strHold.substr(start, p - start + 1));
            }

            // Need to only print CPU Core affinity in first line
            if (start == 0)
            {
                out["CPU Core Affinity"] = ss.str();
            }
            else
            {
                out["CPU Core Affinity"].addOverflow(ss.str());
            }

            start = p + 2; // move ahead two characters
        }
    }
    else
    {
        out["CPU Core Affinity"] = strHold;
    }

    for (unsigned int i = 0; i < gpuTopo.numGpus; i++)
    {
        ss.str(""); // clear
        ss << "To GPU " << gpuTopo.gpuPaths[i].gpuId;
        out[ss.str()] = HelperGetPciPath(gpuTopo.gpuPaths[i].path);
        if (gpuTopo.gpuPaths[i].localNvLinkIds != 0)
        {
            out[ss.str()].addOverflow(
                HelperGetNvLinkPath(gpuTopo.gpuPaths[i].path, gpuTopo.gpuPaths[i].localNvLinkIds));
        }
    }

    std::cout << out.str();

    return DCGM_ST_OK;
}


/********************************************************************************/
dcgmReturn_t Topo::DisplayGroupTopology(dcgmHandle_t mNvcmHandle, dcgmGpuGrp_t requestedGroupId, bool json)
{
    dcgmReturn_t result = DCGM_ST_OK;
    dcgmGroupTopology_t groupTopo;
    groupTopo.version = dcgmGroupTopology_version;
    DcgmiOutputTree outTree(20, 80);
    DcgmiOutputJson outJson;
    DcgmiOutput &out = json ? (DcgmiOutput &)outJson : (DcgmiOutput &)outTree;
    std::stringstream ss;
    dcgmGroupInfo_t stNvcmGroupInfo;

    // Get group name
    stNvcmGroupInfo.version = dcgmGroupInfo_version;
    result                  = dcgmGroupGetInfo(mNvcmHandle, requestedGroupId, &stNvcmGroupInfo);
    if (DCGM_ST_OK != result)
    {
        std::string error = (result == DCGM_ST_NOT_CONFIGURED) ? "The Group is not found" : errorString(result);
        std::cout << "Error: Unable to get group information. Return: " << error << std::endl;
        PRINT_ERROR("%u,%d",
                    "Error: GroupGetInfo for GroupId: %u. Return: %d",
                    (unsigned int)(uintptr_t)requestedGroupId,
                    result);
        return DCGM_ST_GENERIC_ERROR;
    }

    // Get topology
    result = dcgmGetGroupTopology(mNvcmHandle, requestedGroupId, &groupTopo);
    if (result == DCGM_ST_NOT_SUPPORTED)
    {
        std::cout << "Getting topology is not supported for group " << requestedGroupId << std::endl;
        PRINT_INFO("%u",
                   "Getting topology is not supported for this configuration of group %u",
                   (unsigned int)(uintptr_t)requestedGroupId);
        return result;
    }
    else if (result != DCGM_ST_OK)
    {
        std::cout << "Error: unable to get topology for Group. Return: " << errorString(result) << "." << std::endl;
        PRINT_ERROR("%u %d",
                    "Error gettting topology for group: %u. Return: %d",
                    (unsigned int)(uintptr_t)requestedGroupId,
                    result);
        return result;
    }

    // Header
    out.addHeader(HEADER_NAME);
    ss << stNvcmGroupInfo.groupName;
    out.addHeader(ss.str());

    // Affinity

    std::string strHold = HelperGetAffinity(groupTopo.groupCpuAffinityMask);
    unsigned int p      = 0;
    unsigned int start  = 0;

    if (strHold.length() > MAX_SIZE_OF_AFFINITY_STRING)
    {
        while (start < strHold.length())
        {
            p += MAX_SIZE_OF_AFFINITY_STRING;
            if (p >= strHold.length())
                p = strHold.length() - 1;

            else
            { // Put pointer to last available digit
                while (isdigit(strHold.at(p)))
                {
                    if (p + 1 < strHold.length() && !isdigit(strHold.at(p + 1)))
                        break; // check if landed on end of a digit
                    p--;
                }
                while (!isdigit(strHold.at(p)))
                {
                    p--;
                }
            }
            // p is now the index of a the last digit of a CPU

            // Comma case
            if (p + 1 < strHold.length() && strHold.at(p + 1) == ',')
            {
                ss.str(strHold.substr(start, p - start + 2));
            }
            else
            {
                // Hyphen case
                ss.str(strHold.substr(start, p - start + 1));
            }

            // Need to only print CPU Core affinity in first line
            if (start == 0)
            {
                out["CPU Core Affinity"] = ss.str();
            }
            else
            {
                out["CPU Core Affinity"].addOverflow(ss.str());
            }

            start = p + 2; // move ahead two characters
        }
    }
    else
    {
        out["CPU Core Affinity"] = strHold;
    }

    // Numa optimal

    out["NUMA Optimal"] = groupTopo.numaOptimalFlag ? "True" : "False";

    // Worst path

    out["Worst Path"] = HelperGetPciPath(groupTopo.slowestPath);

    std::cout << out.str();

    return DCGM_ST_OK;
}

// **************************************************************************************************
std::string Topo::HelperGetAffinity(unsigned long const *cpuAffinity)
{
    using Bits = std::bitset<sizeof(unsigned long) * DCGM_AFFINITY_BITMASK_ARRAY_SIZE * 8>;
    Bits bits;
    for (size_t i = 1; i <= DCGM_AFFINITY_BITMASK_ARRAY_SIZE; ++i)
    {
        bits <<= sizeof(unsigned long) * 8;
        Bits tmp_bit { cpuAffinity[DCGM_AFFINITY_BITMASK_ARRAY_SIZE - i] };
        bits |= tmp_bit;
    }

    std::stringstream ss_;
    bool bitSet     = false;
    size_t firstBit = 0;
    size_t lastBit  = 0;
    for (size_t i = 0; i < bits.size(); ++i)
    {
        if (bits.test(i))
        {
            if (bitSet)
            {
                lastBit = i;
            }
            else
            {
                firstBit = i;
                lastBit  = i;
                bitSet   = true;
            }
        }
        else
        {
            if (bitSet)
            {
                if (firstBit == lastBit)
                {
                    ss_ << firstBit << ", ";
                }
                else
                {
                    ss_ << firstBit << " - " << lastBit << ", ";
                }
                bitSet = false;
            }
        }
    }
    auto result = ss_.str();
    result.resize(result.length() - 2);
    return result;
}


// **************************************************************************************************
std::string Topo::HelperGetPciPath(dcgmGpuTopologyLevel_t &path)
{
    dcgmGpuTopologyLevel_t pciPath = DCGM_TOPOLOGY_PATH_PCI(path);
    switch (pciPath)
    {
        case DCGM_TOPOLOGY_BOARD:
            return "Connected via an on-board PCIe switch";
        case DCGM_TOPOLOGY_SINGLE:
            return "Connected via a single PCIe switch";
        case DCGM_TOPOLOGY_MULTIPLE:
            return "Connected via multiple PCIe switches";
        case DCGM_TOPOLOGY_HOSTBRIDGE:
            return "Connected via a PCIe host bridge";
        case DCGM_TOPOLOGY_CPU:
            return "Connected via a CPU-level link";
        case DCGM_TOPOLOGY_SYSTEM:
            return "Connected via a CPU-level link";
        default:
            return "Unknown";
    }
}

// **************************************************************************************************
std::string Topo::HelperGetNvLinkPath(dcgmGpuTopologyLevel_t &path, unsigned int linkMask)
{
    std::stringstream pathSS;
    unsigned int maxLinks = DCGM_NVLINK_MAX_LINKS_PER_GPU;

    pathSS << "Connected via ";

    dcgmGpuTopologyLevel_t nvLinkPath = DCGM_TOPOLOGY_PATH_NVLINK(path);
    switch (nvLinkPath)
    {
        case DCGM_TOPOLOGY_NVLINK1:
            pathSS << "one NVLINK ";
            break;
        case DCGM_TOPOLOGY_NVLINK2:
            pathSS << "two NVLINKs ";
            break;
        case DCGM_TOPOLOGY_NVLINK3:
            pathSS << "three NVLINKs ";
            break;
        case DCGM_TOPOLOGY_NVLINK4:
            pathSS << "four NVLINKs ";
            break;
        case DCGM_TOPOLOGY_NVLINK5:
            pathSS << "five NVLINKs ";
            break;
        case DCGM_TOPOLOGY_NVLINK6:
            pathSS << "six NVLINKs ";
            break;
        case DCGM_TOPOLOGY_NVLINK7:
            pathSS << "seven NVLINKs ";
            break;
        case DCGM_TOPOLOGY_NVLINK8:
            pathSS << "eight NVLINKs ";
            break;
        case DCGM_TOPOLOGY_NVLINK9:
            pathSS << "nine NVLINKs ";
            break;
        case DCGM_TOPOLOGY_NVLINK10:
            pathSS << "ten NVLINKs ";
            break;
        case DCGM_TOPOLOGY_NVLINK11:
            pathSS << "eleven NVLINKs ";
            break;
        case DCGM_TOPOLOGY_NVLINK12:
            pathSS << "twelve NVLINKs ";
            break;
        default:
            return "Unknown";
    }

    if (nvLinkPath == DCGM_TOPOLOGY_NVLINK1)
        pathSS << "(Link: ";
    else
        pathSS << "(Links: ";

    bool startedLinkList = false;
    for (unsigned int i = 0; i < maxLinks; i++)
    {
        unsigned int mask = 1 << i;

        if (mask & linkMask)
        {
            if (startedLinkList)
            {
                pathSS << ", ";
            }
            pathSS << i;
            startedLinkList = true;
        }
    }

    pathSS << ")";
    return pathSS.str();
}

/*****************************************************************************
 *****************************************************************************
 * Get GPU Topology
 *****************************************************************************
 *****************************************************************************/

/*****************************************************************************/
GetGPUTopo::GetGPUTopo(std::string hostname, unsigned int gpu, bool json)
{
    m_hostName = std::move(hostname);
    mGpuId     = gpu;
    m_json     = json;
}

/*****************************************************************************/
dcgmReturn_t GetGPUTopo::DoExecuteConnected()
{
    return topoObj.DisplayGPUTopology(m_dcgmHandle, mGpuId, m_json);
}


/*****************************************************************************
 *****************************************************************************
 * Get Group Topology
 *****************************************************************************
 *****************************************************************************/

/*****************************************************************************/
GetGroupTopo::GetGroupTopo(std::string hostname, unsigned int groupId, bool json)
{
    m_hostName = std::move(hostname);
    mGroupId   = (dcgmGpuGrp_t)(long long)groupId;
    m_json     = json;
}

/*****************************************************************************/
dcgmReturn_t GetGroupTopo::DoExecuteConnected()
{
    return topoObj.DisplayGroupTopology(m_dcgmHandle, mGroupId, m_json);
}
