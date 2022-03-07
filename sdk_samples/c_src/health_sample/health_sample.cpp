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

// See function description at bottom of file.
int displayFieldValue(unsigned int gpuId, dcgmFieldValue_v1 *values, int numValues, void *userData);

bool ReceivedIncompatibleMigConfigurationMessage(dcgmDiagResponse_t &response)
{
    return (strstr(response.systemError.msg, "MIG configuration is incompatible with the diagnostic")
            || strstr(response.systemError.msg,
                      "Cannot run diagnostic: CUDA does not support enumerating GPUs with MIG mode"));
}

////////////////////////////////////////////////////////////////////////////////
// In this program we will go through creating a group, enabling health watches
// and checking group health. We will additionally demonstrate running a diagnostic
// on the group.
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    // DCGM calls return a dcgmReturn_t which can be useful for error handling and control flow.
    // Whenever we call DCGM we will store the return in result and check it for errors.
    dcgmReturn_t result;

    // Variables to be used later, descriptions to follow as they become needed.
    bool standalone         = false;
    char hostIpAddress[16]  = { 0 };
    dcgmHandle_t dcgmHandle = (dcgmHandle_t)NULL;
    dcgmGpuGrp_t myGroupId  = (dcgmGpuGrp_t)NULL;
    char groupName[]        = "myGroup";
    dcgmHealthSystems_t healthSystems;
    dcgmHealthResponse_v4 results;
    char myTestString[] = "Outputting watches for DCGM_FC_DEV_INFO";
    dcgmDiagResponse_t diagnosticResults;

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
    // In this example we are going to create a group using DCGM_GROUP_DEFAULT as the group type.
    // This adds all GPUs on the system to our group. See DCGM_GROUPING.

    // Let's create our group with all available GPUs as an example using our groupName and storing our
    // group ID in our previously allocated dcgmGpuGrp_t myGroupId.

    result = dcgmGroupCreate(dcgmHandle, DCGM_GROUP_DEFAULT, groupName, &myGroupId);

    // Check the result to see if our DCGM operation was successful.
    if (result != DCGM_ST_OK)
    {
        std::cout << "Error creating group. Return: " << errorString(result) << std::endl;
        ;
        goto cleanup;
    }
    else
    {
        std::cout << "Successfully created group with group ID: " << (unsigned long)myGroupId << std::endl;
        ;
    }

    // We have now set up our group and can set our watches, run diagnostics and more on the GPUs in
    // within the group.


    // It's important to remember that if we are in manual mode (set when we initialized DCGM) we have to use
    // dcgmEngineUpdateAllFields every time we want fresh data. However we do not have to use it in two cases:
    // 1) DCGM has just been initialized, in this case DCGM automatically takes one sample for all the GPUs so
    // when we create our group the data is still fresh.
    // 2) When a set function has just been used, such as dcgmEngineConfigSet. This automatically updates the latest
    // sample as well.

    if (!standalone)
    {
        dcgmUpdateAllFields(dcgmHandle, 0);
    }

    // First, let's output our group's current health systems for PCIe and memory to the screen.
    // dcgmHealthSystems_t is a bit vector where each bit refers to a particular health watch.

    result = dcgmHealthGet(dcgmHandle, myGroupId, &healthSystems);

    // Check result to see if DCGM operation was successful.
    if (result != DCGM_ST_OK)
    {
        std::cout << "Error getting health systems information. Result: " << errorString(result) << std::endl;
        ;
        goto cleanup;
    }

    // Output the watches we care about, which is PCIe and memory. We could have gone through all of these
    // but there is no need in this example.
    std::cout << "PCIe Watches : " << ((healthSystems & DCGM_HEALTH_WATCH_PCIE) ? " On" : " Off") << std::endl;
    std::cout << "Memory Watches : " << ((healthSystems & DCGM_HEALTH_WATCH_MEM) ? " On" : " Off") << std::endl;

    // Let's enable these watches (assuming they weren't already enabled). Since healthSystems represents
    // a bit vector we logically OR both the PCIe and the memory health watch together to enable both.
    healthSystems = (dcgmHealthSystems_t)(DCGM_HEALTH_WATCH_PCIE | DCGM_HEALTH_WATCH_MEM);

    result = dcgmHealthSet(dcgmHandle, myGroupId, healthSystems);

    // Check result to see if DCGM operation was successful.
    if (result != DCGM_ST_OK)
    {
        std::cout << "Error setting health systems. Result: " << errorString(result) << std::endl;
        ;
        goto cleanup;
    }

    // Both PCIe and memory watches are now enabled and any errors involving them will be recorded.
    // If we would like to sample and record information on fields rather just just checking the field
    // for errors we can use WatchFields. We are going to watch the GPU temperature and power usage in
    // this example. We want to record information at 1 second intervals and hold onto that information
    // for 5 minutes with no limit on the number of samples.

    dcgmFieldGrp_t fieldGroupId;
    unsigned short fieldIds[2];

    fieldIds[0] = DCGM_FI_DEV_POWER_USAGE;
    fieldIds[1] = DCGM_FI_DEV_GPU_TEMP;

    result = dcgmFieldGroupCreate(dcgmHandle, 2, &fieldIds[0], (char *)"interesting_fields", &fieldGroupId);
    if (result != DCGM_ST_OK)
    {
        std::cout << "Error creating field group. Result: " << errorString(result) << std::endl;
        ;
        goto cleanup;
    }

    result = dcgmWatchFields(dcgmHandle, myGroupId, fieldGroupId, 1000000, 300, 0);

    // Check result to see if DCGM operation was successful.
    if (result != DCGM_ST_OK)
    {
        std::cout << "Error setting watches. Result: " << errorString(result) << std::endl;
        ;
        goto cleanup;
    }

    // Let's run another application, while its running any errors in memory and PCIe will be recorded
    // by our health watches which we can check after our process is complete. Our watched fields will
    // also record information. If we are in manual mode we have to call dcgmEngineUpdateAllFields
    // less than or equal to our sampling frequency to ensure we catch all events. If the sample frequency
    // was set to 1 second but dcgmEngineUpdateAllFields was called every 5 then we would only get a
    // recorded frequency of 5 seconds.

    if (!standalone)
    {
        dcgmUpdateAllFields(dcgmHandle, 0);
    }

    /***********************************************
     *    Run process here while updating fields
     ***********************************************/

    if (!standalone)
    {
        dcgmUpdateAllFields(dcgmHandle, 0);
    }

    // Now that our process is complete let's check our watches and output any errors that were caught
    // during it's execution. To do this we need our previously allocated dcgmHealthResponse_t results.

    // Before making DCGM calls that fill in struct information then we need to set the version of
    // our struct. This will tell the Host Engine what type of struct to fill in. If the client has
    // a greater version number than the Host Engine then an error will be returned since the host engine
    // will unaware of how to handle that type of request. So we set the version for the health response struct.
    results.version = dcgmHealthResponse_version4;

    result = dcgmHealthCheck(dcgmHandle, myGroupId, (dcgmHealthResponse_t *)&results);

    // Check result to see if DCGM operation was successful.
    if (result != DCGM_ST_OK)
    {
        std::cout << "Error checking health systems. Result: " << errorString(result) << std::endl;
        ;
        goto cleanup;
    }

    // Let's display any errors caught by the health watches.
    if (results.overallHealth == DCGM_HEALTH_RESULT_PASS)
    {
        std::cout << "Group is healthy.\n";
    }
    else
    {
        std::cout << "Group has a "
                  << ((results.overallHealth == DCGM_HEALTH_RESULT_WARN) ? "warning.\n" : "failure.\n");
        std::cout << "GPU ID : Health \n";
        for (unsigned int i = 0; i < results.incidentCount; i++)
        {
            if (results.incidents[i].entityInfo.entityGroupId != DCGM_FE_GPU)
            {
                continue;
            }

            std::cout << results.incidents[i].entityInfo.entityId << " : ";

            switch (results.incidents[i].health)
            {
                case DCGM_HEALTH_RESULT_PASS:
                    std::cout << "Pass\n";
                    break;
                case DCGM_HEALTH_RESULT_WARN:
                    std::cout << "Warn\n";
                    break;
                default:
                    std::cout << "Fail\n";
            }


            // A more in depth case check may be required here, but since we are only interested in PCIe and memory
            // watches This is all we are going to check for here.
            std::cout << "Error: " << ((results.incidents[i].system == DCGM_HEALTH_WATCH_PCIE) ? "PCIe " : "Memory ");
            std::cout << "watches detected a "
                      << ((results.incidents[i].health == DCGM_HEALTH_RESULT_WARN) ? "warning.\n" : "failure.\n");

            std::cout << results.incidents[i].error.msg << "\n";
        }
        std::cout << std::endl;
    }

    // And let's also display some of the samples recorded by our watches on the temperature and power usage of the
    // GPUs. We demonstrate how we can pass through information using the userDate void pointer parameter in this
    // function by passing our testString through. More information on this can be seen in the function below.

    result = dcgmGetLatestValues(dcgmHandle, myGroupId, fieldGroupId, &displayFieldValue, (void *)myTestString);

    // Check result to see if DCGM operation was successful.
    if (result != DCGM_ST_OK)
    {
        std::cout << "Error fetching latest values for watches. Result: " << errorString(result) << std::endl;
        goto cleanup;
    }

    // Now that we have run our process, let's say we want to perform a short diagnostic on the system to ensure that
    // the group is still healthy and ready for the next process to be run. We will then print out the results of the
    // diagnostic. Medium and long NVVS diagnostics can offer per GPU results and it's likely that a more thorough
    // diagnostic would be preferred, but this is just an example.
    std::cout << "Running diagnostic.\n";
    // Update version
    diagnosticResults.version = dcgmDiagResponse_version;
    result                    = dcgmRunDiagnostic(dcgmHandle, myGroupId, DCGM_DIAG_LVL_MED, &diagnosticResults);

    // Check result to see if DCGM operation was successful.
    if (result != DCGM_ST_OK)
    {
        if (result == DCGM_ST_GROUP_INCOMPATIBLE)
            std::cout << "GPUs in the group are incompatible with each other to run diagnostics" << std::endl;
        else
        {
            // Ignore message for incompatible MIG configuration
            if (ReceivedIncompatibleMigConfigurationMessage(diagnosticResults))
            {
                std::cout << "GPU configuation is not MIG compatible: " << diagnosticResults.systemError.msg
                          << std::endl;
                result = DCGM_ST_OK;
            }
            else
            {
                std::cout << "Error running diagnostic. Result: " << errorString(result) << " : "
                          << diagnosticResults.systemError.msg << std::endl;
            }
        }
    }
    else
    {
        std::cout << "Diagnostic Results (0 = Pass, 1 = Skip, 2 = Warn, 3 = Fail)\n";
        std::cout << "Blacklist: " << diagnosticResults.levelOneResults[DCGM_SWTEST_BLACKLIST].status << std::endl;
        std::cout << "CUDA Main Library: " << diagnosticResults.levelOneResults[DCGM_SWTEST_CUDA_MAIN_LIBRARY].status
                  << std::endl;
        std::cout << "CUDA Runtime Library: "
                  << diagnosticResults.levelOneResults[DCGM_SWTEST_CUDA_RUNTIME_LIBRARY].status << std::endl;
        std::cout << "Environment: " << diagnosticResults.levelOneResults[DCGM_SWTEST_ENVIRONMENT].status << std::endl;
        std::cout << "Graphics Processes: " << diagnosticResults.levelOneResults[DCGM_SWTEST_GRAPHICS_PROCESSES].status
                  << std::endl;
        std::cout << "NVML Library: " << diagnosticResults.levelOneResults[DCGM_SWTEST_NVML_LIBRARY].status
                  << std::endl;
        std::cout << "Page Retirement: " << diagnosticResults.levelOneResults[DCGM_SWTEST_PAGE_RETIREMENT].status
                  << std::endl;
        std::cout << "Permissions: " << diagnosticResults.levelOneResults[DCGM_SWTEST_PERMISSIONS].status << std::endl;
        std::cout << "Persistence Mode: " << diagnosticResults.levelOneResults[DCGM_SWTEST_PERSISTENCE_MODE].status
                  << std::endl;
    }

// Cleanup consists of destroying our group and shutting down DCGM.
cleanup:
    std::cout << "Cleaning up. \n";
    dcgmFieldGroupDestroy(dcgmHandle, fieldGroupId);
    dcgmGroupDestroy(dcgmHandle, myGroupId);
    if (standalone)
        dcgmDisconnect(dcgmHandle);
    else
        dcgmStopEmbedded(dcgmHandle);
    dcgmShutdown();
    return result;
}


// In this function we simply print out the information in the callback of the watched field.
int displayFieldValue(unsigned int gpuId, dcgmFieldValue_v1 *values, int numValues, void *userData)
{
    // The void pointer at the end allows a pointer to be passed to this function. Here we know
    // that we are passing in a null terminated C string, so I can cast it as such. This pointer
    // can be useful if you need a reference to something inside your function.
    std::cout << (char *)userData << std::endl;
    std::cout << std::endl;

    // If we want to query for more information on the particular field ID we have to initialize
    // DCGM fields. We can then query using a field ID and get a pointer to a dcgm_field_meta_t
    // struct. This will allow us to output the name of the DCGM field.

    // Output the information to screen.
    for (int i = 0; i < numValues; i++)
    {
        std::cout << "GPU: " << gpuId << std::endl;

        // Get the field ID tag. This is why we needed to initialize the DCGM fields with DcgmFieldsInit.
        std::cout << "Field ID: "
                  << (DcgmFieldGetById(values[i].fieldId) ? DcgmFieldGetById(values[i].fieldId)->tag : "Unknown")
                  << std::endl;
        std::cout << "Query Status: " << errorString((dcgmReturn_t)values[i].status) << std::endl;
        std::cout << "Time stamp: " << values[i].ts << std::endl;

        // Including a switch statement here even though I know that both power and temperature come
        // back as doubles and 64 bit integers respectively. This will demonstrate handling different
        // types of values (except binary blobs).

        switch (DcgmFieldGetById(values[i].fieldId)->fieldType)
        {
            case DCGM_FT_BINARY:
                // Handle binary data
                break;
            case DCGM_FT_DOUBLE:
                std::cout << "Value: " << values[i].value.dbl;
                break;
            case DCGM_FT_INT64:
                std::cout << "Value: " << values[i].value.i64;
                break;
            case DCGM_FT_STRING:
                std::cout << "Value: " << values[i].value.str;
                break;
            case DCGM_FT_TIMESTAMP:
                std::cout << "Value: " << values[i].value.i64;
                break;
            default:
                std::cout << "Error in field types. " << values[i].fieldType << " Exiting.\n";
                // Error, return > 0 error code.
                return 1;
        }
        std::cout << std::endl;
    }

    std::cout << std::endl;
    // Shutdown DCGM fields. This takes care of the memory initialized when we called DcgmFieldsInit.

    // Program executed correctly. Return 0 to notify DCGM (callee) that it was successful.
    return 0;
}
