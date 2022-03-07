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

#include "dcgm_agent.h"
#include "dcgm_fields.h"
#include "dcgm_structs.h"
#include "string.h"
#include <iostream>

////////////////////////////////////////////////////////////////////////////////
// This sample goes through the process of enabling process watches on a group,
// running a process and viewing the statistics of the group while the process ran.
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    // DCGM calls return a dcgmReturn_t which can be useful for error handling and control flow.
    // Whenever we call DCGM we will store the return in result and check it for errors.
    dcgmReturn_t result;

    // Variables to be used later, descriptions to follow as they become needed.
    dcgmHandle_t dcgmHandle = (dcgmHandle_t)NULL;
    bool standalone         = false;
    char hostIpAddress[16]  = { 0 };
    dcgmPidInfo_t pidInfo;


    // In our case we do not know whether to run in standalone or embedded so we will get the user
    // to set the mode. This will also allow us to see the differences and similarities between
    // embedded and standalone modes in one place.
    std::cout << "Start DCGM Host Engine in: \n";
    std::cout << "0 - Embedded mode \n";
    std::cout << "1 - Standalone mode \n";
    while (!(std::cin >> standalone))
    {
        std::cout << "Invalid input.\n";
        std::cin.clear();
        std::cin.ignore();
    }
    std::cout << std::endl;

    std::cout << (standalone ? "Standalone mode selected.\n" : "Embedded mode selected.\n");

    // Now we need an IP address if we are running in standalone mode. Let's take in whatever string
    // of length 15 and let DCGM handle the error if the IP address is invalid for simplicity.
    if (standalone)
    {
        std::cout << "Please input an IP address to connect to. (Localhost = 127.0.0.1) \n";

        std::string buffer;
        std::cin >> buffer;
        if (buffer.length() > sizeof(hostIpAddress) - 1)
        {
            std::cout << "Error: Invalid IP address given.\n";
            result = DCGM_ST_BADPARAM; // set this before we go to cleanup since we were given a bad IP address.
            goto cleanup;
        }
        else
        {
            buffer += '\0';
            strncpy(hostIpAddress, buffer.c_str(), 15);
        }
    }

    // Now that we have all of our settings sorted out we need to initialize DCGM.
    result = dcgmInit();

    // Check the result to see if our DCGM operation was successful. In this example we will simply print
    // out our error, cleanup and exit DCGM if any part of our program fails. This may not be the
    // desired functionality for other programs.
    if (result != DCGM_ST_OK)
    {
        std::cout << "Error initializing DCGM engine. Return: " << errorString(result) << std::endl;
        goto cleanup;
    }
    else
    {
        std::cout << "DCGM Initialized.\n";
    }

    // We now need to either connect to the remote host engine in standalone mode or start a host
    // engine in embedded mode. If we are in embedded mode we need to manually control our watches
    // and periodically call DCGM to wake up and update our fields.
    // If we are in standalone mode we can set the operation mode to auto and
    // let DCGM worry about updating fields itself. This fucntion will store a handle to the requested
    // DCGM server if it is able to connect. To learn more please refer to the DCGM User Guide.
    if (standalone)
    {
        result = dcgmConnect(hostIpAddress, &dcgmHandle);
        if (result != DCGM_ST_OK)
        {
            std::cout << "Error connecting to remote DCGM engine. Return: " << errorString(result) << std::endl;
            goto cleanup;
        }
    }
    else // Embedded
    {
        result = dcgmStartEmbedded(DCGM_OPERATION_MODE_AUTO, &dcgmHandle);
        if (result != DCGM_ST_OK)
        {
            std::cout << "Error starting embedded DCGM engine. Return: " << errorString(result) << std::endl;
            goto cleanup;
        }
    }

    // One of the core concepts of DCGM is managing groups of GPUs. Groups allow operations
    // to be performed uniformly across multiple GPUs including setting configurations.
    // In this example we are going to use the DCGM default group, which contains all GPUs.
    // We could always make our own group with any subset of GPUs that we wanted.
    // See DCGM_GROUPING.


    // First we need to enable our process watches so we can record process information. In this example we
    // will set the sample frequency to once per second, the max keep age to one hour and the maximum number of
    // samples to keep to unlimited. Always remember to enable watches before starting the process or DCGM will
    // not record any information for it.
    result = dcgmWatchPidFields(dcgmHandle, (dcgmGpuGrp_t)DCGM_GROUP_ALL_GPUS, 1000000, 3600, 0);

    // Check the result to see if our DCGM operation was successful.
    if (result != DCGM_ST_OK)
    {
        std::cout << "Error setting process watches. Return: " << errorString(result);
        goto cleanup;
    }

    // It's important to remember if we are in manual mode (set when we initialized DCGM) we have to use
    // dcgmEngineUpdateAllFields every time we want a sample to be taken. If we call dcgmEngineUpdateAllFields
    // with a higher frequency than the sampling frequency of a watch, we will still get a recorded frequency of
    // the watch. However if we call dcgmEngineUpdateAllFields with a lower frequency than the sampling frequency
    // we will get a recorded frequency of the dcgmEngineUpdateAllFields call.
    if (!standalone)
    {
        dcgmUpdateAllFields(dcgmHandle, 0);
    }

    // Since process watches are enabled, we can run our process and and fetch the process execution statistics
    // from DCGM. We will ask the user for a PID in this example.

    // Before making DCGM calls that fill in object/struct information then we need to set the version of
    // our structs. This will tell the Host Engine what type of struct to fill in. If the client has
    // a greater version number than the Host Engine then an error will be returned since the host engine
    // will unaware of how to handle that type of request.
    pidInfo.version = dcgmPidInfo_version;

    // Get process info for the specified PID. This call will fill in our pidInfo struct with the information
    // corresponding to the PID given. Right now we do not have a CUDA application to start so the PID we pass
    // in will return no data. We handle that in our error check and exit this example.
    pidInfo.pid = 1;
    result      = dcgmGetPidInfo(dcgmHandle, (dcgmGpuGrp_t)DCGM_GROUP_ALL_GPUS, &pidInfo);

    // Check the result to see if our DCGM operation was successful. DCGM_ST_NO data is checked for here as well
    // since it is the expected result when we pass in a random PID.
    if (result != DCGM_ST_OK && result != DCGM_ST_NO_DATA)
    {
        std::cout << "Error getting process info. Return: " << errorString(result) << std::endl;
        goto cleanup;
    }

    // Let's output some information we think is useful. We should also test if each value is blank so we don't
    // output raw error codes. The blank values may actually be "Not Supported", "Not Found", etc. and checking
    // for these may be desired. However in this example let's use -1 to display a blank value which covers all
    // of these cases. See dcgm_structs.h for more info.
    if (result == DCGM_ST_OK)
    {
        std::cout << "PID Info\n";
        std::cout << "Start Time timestamp: "
                  << (DCGM_INT64_IS_BLANK(pidInfo.summary.startTime) ? -1 : pidInfo.summary.startTime) << std::endl;
        std::cout << "End Time timestamp: "
                  << (DCGM_INT64_IS_BLANK(pidInfo.summary.endTime) ? -1 : pidInfo.summary.endTime) << std::endl;
        std::cout << "Energy Consumed: "
                  << (DCGM_INT64_IS_BLANK(pidInfo.summary.energyConsumed) ? -1 : pidInfo.summary.energyConsumed)
                  << std::endl;
        std::cout << "Max GPU Memory Used: "
                  << (DCGM_INT64_IS_BLANK(pidInfo.summary.maxGpuMemoryUsed) ? -1 : pidInfo.summary.maxGpuMemoryUsed)
                  << std::endl;
        std::cout << "Average Memory Utilization: "
                  << (DCGM_INT32_IS_BLANK(pidInfo.summary.memoryUtilization.average)
                          ? -1
                          : pidInfo.summary.memoryUtilization.average)
                  << std::endl;
        std::cout << "No. Other Compute PIDs: "
                  << (DCGM_INT32_IS_BLANK(pidInfo.summary.numOtherComputePids) ? -1
                                                                               : pidInfo.summary.numOtherComputePids)
                  << std::endl;
        std::cout << "No. PCIe Replays: "
                  << (DCGM_INT64_IS_BLANK(pidInfo.summary.pcieReplays) ? -1 : pidInfo.summary.pcieReplays) << std::endl;
        std::cout << "No. XID Critical Errors: "
                  << (DCGM_INT32_IS_BLANK(pidInfo.summary.numXidCriticalErrors) ? -1
                                                                                : pidInfo.summary.numXidCriticalErrors)
                  << std::endl;
    }
    else
    {
        std::cout << "No data for PID found." << std::endl;
    }

// Cleanup consists of shutting down DCGM.
cleanup:
    std::cout << "Cleaning up.\n";
    if (standalone)
        dcgmDisconnect(dcgmHandle);
    else
        dcgmStopEmbedded(dcgmHandle);
    dcgmShutdown();
    return result;
}
