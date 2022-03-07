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
#include "dcgm_structs.h"
#include "string.h"
#include "unistd.h"
#include <iostream>
#include <map>
#include <time.h>
#include <vector>


#define MAX_GPUS_IN_GROUP 2


int list_field_values_since(unsigned int gpuId, dcgmFieldValue_v1 *values, int numValues, void *userData);
int list_field_values(unsigned int gpuId, dcgmFieldValue_v1 *values, int numValues, void *userData);


int main(int argc, char **argv)
{
    // DCGM calls return a dcgmReturn_t which can be useful for error handling
    // and control flow.Whenever we call DCGM we will store the return in result
    // and check it for errors.
    dcgmReturn_t result;

    // Variables to be used later, descriptions to follow as they become needed.
    dcgmHandle_t dcgmHandle = (dcgmHandle_t)NULL;
    bool standalone         = false;
    char hostIpAddress[16]  = { 0 };
    unsigned int gpuIdList[DCGM_MAX_NUM_DEVICES];
    int count;
    char groupName[]       = "MyGroup";
    dcgmGpuGrp_t myGroupId = (dcgmGpuGrp_t)NULL;
    int max_loop_run_sec   = 10;
    std::map<unsigned int, dcgmFieldValue_v1> field_val_map;
    std::map<unsigned int, std::vector<dcgmFieldValue_v1>> field_val_vec_map;
    long long next_since_timestamp = 0;


    // In our case we do not know whether to run in standalone or embedded so we
    // will get the user to set the mode. This will also allow us to see the
    // differences and similarities between embedded and standalone modes in
    // one place.
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

    std::cout << (standalone ? "Standalone mode selected.\n"
                             : "Embedded mode "
                               "selected.\n");

    // Now we need an IP address if we are running in standalone mode. Let's
    // take in whatever string of length 15 and let DCGM handle the error if the
    // IP address is invalid for simplicity.
    if (standalone)
    {
        std::string buffer;

        std::cout << "Please input an IP address to connect to "
                     "(Localhost = 127.0.0.1) \n";

        std::cin >> buffer;
        if (buffer.length() > sizeof(hostIpAddress) - 1)
        {
            std::cout << "Error: Invalid IP address given.\n";
            result = DCGM_ST_BADPARAM; // set this before we go to cleanup since
            // we were given a bad IP address.
            goto cleanup;
        }
        else
        {
            buffer += '\0';
            strncpy(hostIpAddress, buffer.c_str(), 15);
        }
    }

    // Now that we have all of our settings sorted out we need to initialize
    // DCGM.
    result = dcgmInit();

    // Check the result to see if our DCGM operation was successful. In this
    // example we will simply print out our error, cleanup and exit DCGM if any
    // part of our program fails. This may not be the desired functionality for
    // other programs.
    if (result != DCGM_ST_OK)
    {
        std::cout << "Error initializing DCGM engine. Return: " << errorString(result) << std::endl;
        goto cleanup;
    }
    else
    {
        std::cout << "DCGM Initialized.\n";
    }

    // We now need to either connect to the remote host engine in standalone
    // mode or start a host engine in embedded mode. If we are in embedded mode
    // we need to manually control our watches and periodically call DCGM to
    // wake up and update our fields. If we are in standalone mode we can set
    // the operation mode to auto and let DCGM worry about updating fields
    // itself. This function will store a handle to the requested DCGM server if
    // it is able to connect. To learn more please refer to the DCGM User Guide.
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


    // One of the core concepts of DCGM is managing groups of GPUs. Groups allow
    // operations to be performed uniformly across multiple GPUs including
    // setting configurations. In this example we are going to create a group
    // with the first two GPUs listed (If only one GPU on the system it will
    // form a group of one GPU). See DCGM_GROUPING for more information.

    // We can query DCGM to give us a list of all the GPUs on the system and
    // select from those available.

    // Call DCGM to return a list of all GPUs on the system. Storing the GPU IDs
    // in our array and the total number of GPUs in count.

    result = dcgmGetAllSupportedDevices(dcgmHandle, gpuIdList, &count);

    // Check the result to see if our DCGM operation was successful.
    if (result != DCGM_ST_OK)
    {
        std::cout << "Error fetching devices. Return: " << errorString(result) << std::endl;
        goto cleanup;
    }

    // If we have no GPUs on our system then there isn't much else for us to do.
    // We will treat this as if it is an error and cleanup. Otherwise we are
    // going to output our available GPUs.
    if (count == 0)
    {
        std::cout << "None\n";
        std::cout << "Error: No Supported GPUs.\n";
        result = DCGM_ST_GPU_NOT_SUPPORTED; // set this before we go to cleanup
        // since we were given a bad IP address.
        goto cleanup;
    }
    else
    {
        std::cout << "Available DCGM-Supported GPUs: ";
        for (int i = 0; i < count; i++)
        {
            std::cout << gpuIdList[i] << "   ";
        }
        std::cout << std::endl;
    }

    // We now know what GPUs are available to us, but let's say we want a group
    // with the first two GPUs listed as an example. We pass in a reference to
    // our dcgmGpuGrp_t to store our group's ID.

    // Create our group which we would like to start off empty.
    result = dcgmGroupCreate(dcgmHandle, DCGM_GROUP_DEFAULT, groupName, &myGroupId);

    // Check the result to see if our DCGM operation was successful.
    if (result != DCGM_ST_OK)
    {
        std::cout << "Error creating group. Return: " << errorString(result) << std::endl;
        goto cleanup;
    }
    else
    {
        std::cout << "Successfully created group with group ID: " << (unsigned long)myGroupId << std::endl;
    }

    // We have now set up our group and can set our watches, run diagnostics
    // and more on the GPUs in within the group.

    // We are going to watch the GPU temperature and power usage in
    // this example. We want to record information at 1 second intervals and
    // hold onto that information for 5 minutes with no limit on the number of
    // samples.

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

    result = dcgmWatchFields(dcgmHandle, myGroupId, fieldGroupId, 1000000, 3600.0, 3600);

    // Check result to see if DCGM operation was successful.
    if (result != DCGM_ST_OK)
    {
        std::cout << "Error setting watches. Result: " << errorString(result) << std::endl;
        ;
        goto cleanup;
    }

    // It's important to remember that if we are in manual mode (set when we
    // initialized DCGM) we have to use dcgmEngineUpdateAllFields every time we
    // want fresh data. However we do not have to use it in two cases:
    // 1) DCGM has just been initialized, in this case DCGM automatically takes
    // one sample for all the GPUs so when we create our group the data is still
    // fresh.
    // 2) When a set function has just been used, such as dcgmEngineConfigSet.
    // This automatically updates the latest sample as well.

    if (!standalone)
    {
        dcgmUpdateAllFields(dcgmHandle, 1);
    }

    // Displaying some of the samples recorded by our watches on the temperature
    // and power usage of the GPUs. We demonstrate how we can pass through
    // information using the userDate void pointer parameter in this function by
    // passing our testString through. More information on this can be seen in
    // the function below.
    int i;
    i = 0;
    while (i < max_loop_run_sec)
    {
        result = dcgmGetLatestValues(dcgmHandle, myGroupId, fieldGroupId, &list_field_values, &field_val_map);

        // Check the result to see if our DCGM operation was successful.
        if (result != DCGM_ST_OK)
        {
            std::cout << "Error getValues information. "
                         "Return: "
                      << errorString(result) << std::endl;
            goto cleanup;
        }
        else
        {
            std::cout << "getValues information successfully stored.\n";
        }

        usleep(1000000);
        i++;
    }

    i = 0;
    long long since_timestamp;
    since_timestamp = static_cast<long long>(time(NULL));

    while (i < 3)
    {
        result = dcgmGetValuesSince(dcgmHandle,
                                    myGroupId,
                                    fieldGroupId,
                                    since_timestamp,
                                    &next_since_timestamp,
                                    &list_field_values_since,
                                    &field_val_vec_map);

        // Check the result to see if our DCGM operation was successful.
        if (result != DCGM_ST_OK)
        {
            std::cout << "Error GetValuesSince information. "
                         "Return: "
                      << errorString(result) << std::endl;
            goto cleanup;
        }
        else
        {
            std::cout << "\n\nGetValuesSince information successfully stored.\n";
        }

        usleep(5 * 1000000);
        i++;
    }

    std::cout << "Clearing the map" << std::endl;
    field_val_map.clear();
    field_val_vec_map.clear();

// Clean up consists of memory management and shutting down the host engine.
// This may not be desired in all applications.
cleanup:
    std::cout << "Cleaning up.\n";
    dcgmGroupDestroy(dcgmHandle, myGroupId);
    if (standalone)
        dcgmDisconnect(dcgmHandle);
    else
        dcgmStopEmbedded(dcgmHandle);
    dcgmShutdown();
    return result;
}

/**
 *
 * This function prints the collected value over a time period. It is invoked
 * every 5 seconds and aggregates the values for this time.
 *
 * @param gpuId
 * @param values
 * @param numValues
 * @param userdata
 * @return int
 */
int list_field_values_since(unsigned int gpuId, dcgmFieldValue_v1 *values, int numValues, void *userdata)
{
    int i = 0;
    std::vector<dcgmFieldValue_v1> dcgm_field_vals_vec;

    std::map<unsigned int, std::vector<dcgmFieldValue_v1>> field_val_vec_map;
    // note this is a pointer to a map.
    field_val_vec_map = *static_cast<std::map<unsigned int, std::vector<dcgmFieldValue_v1>> *>(userdata);


    for (i = 0; i < numValues; i++)
    {
        // Check if element exists in the map
        if (field_val_vec_map.count((values[i].fieldId)) > 0)
        {
            dcgm_field_vals_vec = field_val_vec_map[values[i].fieldId];

            dcgm_field_vals_vec.push_back(values[i]);

            field_val_vec_map[values[i].fieldId] = dcgm_field_vals_vec;
            std::cout << "Pushed value for the existing field id => " << (values[i].fieldId) << std::endl;
        }
        // Element is not present in map. Create a new vector and push values
        // into it.
        else
        {
            dcgm_field_vals_vec.push_back(values[i]);
            field_val_vec_map[values[i].fieldId] = dcgm_field_vals_vec;
            std::cout << "\n\n\n\nPushed value for the non - existing field id => " << values[i].fieldId << std::endl;
        }
    }

    return 0;
}

/**
 * In this function we simply print out the information in the callback of
 * the  watched field.
 * @param gpuId
 * @param values
 * @param numValues
 * @param userdata
 * @return
 */
int list_field_values(unsigned int gpuId, dcgmFieldValue_v1 *values, int numValues, void *userdata)
{
    // The void pointer at the end allows a pointer to be passed to this
    // function. Here we know that we are passing in a null terminated C
    // string, so I can cast it as such. This pointer can be useful if you
    // need a reference to something inside your function.
    std::cout << std::endl;
    std::map<unsigned int, dcgmFieldValue_v1> field_val_map;
    // note this is a pointer to a map.
    field_val_map = *static_cast<std::map<unsigned int, dcgmFieldValue_v1> *>(userdata);

    // Storing the values in the map where key is field Id and the value is
    // the corresponding data for the field.
    for (int i = 0; i < numValues; i++)
    {
        field_val_map[values[i].fieldId] = values[i];
    }


    // Output the information to screen.
    for (std::map<unsigned int, dcgmFieldValue_v1>::iterator it = field_val_map.begin(); it != field_val_map.end();
         ++it)
    {
        std::cout << "Field ID => " << it->first << std::endl;
        std::cout << "Value => ";

        dcgm_field_meta_p field = DcgmFieldGetById((it->second).fieldId);
        unsigned char fieldType = field == nullptr ? DCGM_FI_UNKNOWN : field->fieldType;
        switch (fieldType)
        {
            case DCGM_FT_BINARY:
                // Handle binary data
                break;
            case DCGM_FT_DOUBLE:
                std::cout << (it->second).value.dbl;
                break;
            case DCGM_FT_INT64:
                std::cout << (it->second).value.i64;
                break;
            case DCGM_FT_STRING:
                std::cout << (it->second).value.str;
                break;
            case DCGM_FT_TIMESTAMP:
                std::cout << (it->second).value.i64;
                break;
            default:
                std::cout << "Error in field types. " << (it->second).fieldType << " Exiting.\n";
                // Error, return > 0 error code.
                return 1;
                break;
        }

        std::cout << std::endl;
        // Shutdown DCGM fields. This takes care of the memory initialized
        // when we called DcgmFieldsInit.
    }

    // Program executed correctly. Return 0 to notify DCGM (callee) that it
    // was successful.
    return 0;
}
