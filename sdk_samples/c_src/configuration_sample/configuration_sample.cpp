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
#include <iostream>

void displayError(dcgmStatus_t &statusHandle);

////////////////////////////////////////////////////////////////////////////////
// In this program we will create a group, get the configuration of the GPUs in
// in the group, set a configuration on those GPUs and demonstrate enforcing.
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
    unsigned int gpuIdList[DCGM_MAX_NUM_DEVICES];
    int count;
    char groupName[]       = "MyGroup";
    dcgmGpuGrp_t myGroupId = (dcgmGpuGrp_t)NULL;
    dcgmGroupInfo_t myGroupInfo;
    dcgmStatus_t statusHandle      = (dcgmStatus_t)NULL;
    dcgmConfig_t *deviceConfigList = NULL;
    dcgmConfig_t myGroupConfig;


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
    // let DCGM worry about updating fields itself. This function will store a handle to the requested
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
    // In this example we are going to create a group with all of the evenly numbered GPUs but other
    // groups may be desired. We could also simply use the DCGM default group which contains all GPUs.
    // See DCGM_GROUPING.

    // We can query DCGM to give us a list of all the GPUs on the system and select from
    // those available.

    // Use gpuIdList, count and call DCGM to return a list of all GPUs on the system.

    result = dcgmGetAllSupportedDevices(dcgmHandle, gpuIdList, &count);

    // Check the result to see if our DCGM operation was successful.
    if (result != DCGM_ST_OK)
    {
        std::cout << "Error fetching devices. Return: " << errorString(result) << std::endl;
        goto cleanup;
    }

    std::cout << "Available DCGM-Supported GPUs: ";

    // If we have no GPUs on our system then there isn't much else for us to do. We will treat this as if
    // it is an error and cleanup. Otherwise we are going to output our available GPUs.
    if (count == 0)
    {
        std::cout << "None\n";
        std::cout << "Error: No further action possible.\n";
        result = DCGM_ST_GENERIC_ERROR; // set this before we go to cleanup since we have no GPUs.
        goto cleanup;
    }
    else
    {
        for (int i = 0; i < count; i++)
        {
            std::cout << gpuIdList[i] << "   ";
        }
        std::cout << std::endl;
    }

    // Using groupName and myGroupId, let's go ahead and make our group with only the GPUs with even GPU IDs.

    // Create our group which we would like to start off empty since we will be adding GPUs to it. Likewise we
    // could have used DCGM_GROUP_DEFAULT and created the group with all GPUs already added. Then to get our
    // evenly numbered group we would have just removed the odd numbered GPUs.
    result = dcgmGroupCreate(dcgmHandle, DCGM_GROUP_EMPTY, groupName, &myGroupId);

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

    // Add the even numbered GPUs
    for (int i = 0; i < count; i++)
    {
        if (gpuIdList[i] % 2 == 0)
        {
            result = dcgmGroupAddDevice(dcgmHandle, myGroupId, gpuIdList[i]);

            // Check the result to see if our DCGM operation was successful.
            if (result != DCGM_ST_OK)
            {
                std::cout << "Error adding device to group. Return: " << errorString(result) << std::endl;
                goto cleanup;
            }
            else
            {
                std::cout << "Successfully added GPU " << gpuIdList[i] << " to group.\n";
            }
        }
    }

    // Using myGroupInfo, let's check and make sure we have some GPUs in our group. In our example, if we
    // don't have any GPUs in our group then we will exit.

    // Before making DCGM calls that fill in object/struct information then we need to set the version of
    // our object/struct. This will tell the Host Engine what type of struct to fill in. If the client has
    // a greater version number than the Host Engine then an error will be returned since the host engine
    // will unaware of how to handle that type of request.
    myGroupInfo.version = dcgmGroupInfo_version;

    result = dcgmGroupGetInfo(dcgmHandle, myGroupId, &myGroupInfo);

    // Check the result to see if our DCGM operation was successful.
    if (result != DCGM_ST_OK)
    {
        std::cout << "Error getting group information. Return: " << errorString(result) << std::endl;
        goto cleanup;
    }

    // Check to see if our group has some GPUs in it. This error shouldn't happen since we have already checked
    // if we had GPUs on our system and normally the first GPU should be indexed at 0. This may not be the case
    // if a GPU falls off the bus however, so it may be best to check for this here.
    if (!myGroupInfo.count)
    {
        std::cout << "No GPUs in group. Exiting.\n";
        goto cleanup;
    }

    // We now have our group the even numbered GPUs.

    // However before we can set a group configuration we need a DCGM status handle. The Status handle will allow
    // us to look at per GPU error reporting rather than just the dcgmStatus_t result we were using before. If we
    // didn't want or need this kind of error reporting, we can simply pass NULL in anywhere a status handle is
    // used as a parameter.

    result = dcgmStatusCreate(&statusHandle);

    // Check the result to see if our DCGM operation was successful.
    if (result != DCGM_ST_OK)
    {
        std::cout << "Error creating status handler. Return: " << errorString(result) << std::endl;
        goto cleanup;
    }
    else
    {
        std::cout << "Status handle created. \n";
    }

    // Let's initialize our device config list and output our current configuration's compute mode. We create
    // an array of dcgmConfig_t structs that is the size of our group and update the version of each struct in
    // the array.

    deviceConfigList = new dcgmConfig_t[myGroupInfo.count];

    // Update version for each dcmgConfig_t struct.
    for (unsigned int i = 0; i < myGroupInfo.count; i++)
    {
        deviceConfigList[i].version = dcgmConfig_version;
    }

    // Here we fetch the current state of our group. This will give us a current configuration of
    // each GPU in our group. Likewise we could have gotten the target configuration for this group,
    // but we will be setting the configuration next so we don't need it right now.
    result = dcgmConfigGet(
        dcgmHandle, myGroupId, DCGM_CONFIG_CURRENT_STATE, myGroupInfo.count, deviceConfigList, statusHandle);

    // Check the result to see if our DCGM operation was successful. However we now have a status
    // handle which we will use to display our errors per GPU. We will simply pass our status handle
    // to the function handleError which is defined at the bottom of this file.
    if (result != DCGM_ST_OK)
    {
        std::cout << "Error getting group configuration. Return: " << errorString(result) << std::endl;
        displayError(statusHandle);
        goto cleanup;
    }
    else
    {
        // If successful we will display the compute mode of each GPU in our group. This isn't the best
        // formatting for outputting the compute mode but it gets the job done.
        std::cout << "GPU : Compute Mode\n";
        for (unsigned int i = 0; i < myGroupInfo.count; i++)
        {
            if ((DCGM_INT32_NOT_SUPPORTED == deviceConfigList[i].computeMode))
            {
                std::cout << deviceConfigList[i].gpuId << "  : Not Supported" << std::endl;
            }
            else
            {
                std::cout << deviceConfigList[i].gpuId << "   : " << deviceConfigList[i].computeMode << std::endl;
            }
        }
    }

    // It's important to remember that if are in manual mode (set when we initialized DCGM) we have to use
    // dcgmEngineUpdateAllFields every time we want fresh data. However we do not have to use it in two cases:
    // 1) DCGM has just been initialized, in this case DCGM automatically takes one sample for all the GPUs so
    // when we create our group the data is still fresh.
    // 2) When a set function has just been used, such as dcgmEngineConfigSet. This automatically updates the latest
    // sample as well.

    if (!standalone)
    {
        dcgmUpdateAllFields(dcgmHandle, 0);
    }

    // Now let's set the compute mode of the group to unrestricted.
    // We could add more configuration settings here here but let's keep it at this for simplicity.
    // Any setting not included in mConfigVal will maintain its previous setting, however the target
    // configuration will only include the compute mode. For more information see configurations in
    // the DCGM documentation.

    // Set version and default all configuration settings to blank. By setting these to blank it is essentially
    // telling DCGM that we do not want to specify the values for this setting. These settings will remain
    // unchanged on the GPUs in the group.
    myGroupConfig.version = dcgmConfig_version;

    myGroupConfig.eccMode                         = DCGM_INT32_BLANK;
    myGroupConfig.perfState.syncBoost             = DCGM_INT32_BLANK;
    myGroupConfig.perfState.targetClocks.memClock = DCGM_INT32_BLANK;
    myGroupConfig.perfState.targetClocks.smClock  = DCGM_INT32_BLANK;
    myGroupConfig.powerLimit.val                  = DCGM_INT32_BLANK;
    myGroupConfig.computeMode                     = DCGM_INT32_BLANK;
    myGroupConfig.gpuId                           = DCGM_INT32_BLANK;

    // Set compute mode
    myGroupConfig.computeMode = DCGM_CONFIG_COMPUTEMODE_DEFAULT;

    // Set myGroup to use the configuration
    result = dcgmConfigSet(dcgmHandle, myGroupId, &myGroupConfig, statusHandle);

    // Check result to check for errors. Again using the status handle to give per GPU errors.
    if (result != DCGM_ST_OK)
    {
        std::cout << "Error setting group configuration. Return: " << errorString(result) << std::endl;
        displayError(statusHandle);
        goto cleanup;
    }

    std::cout << "Successfully set group configuration.\n";
    // DCGM starts maintaining the target/desired configuration for a group after Set operation. In the event
    // of a GPU reset DCGM will automatically enforce the last target configuration set by the user on the GPU.

    // For demonstration purposes let's say after some time the group settings have changed,
    // maybe a few GPUs were set to another compute mode through some other means (either from a GPU
    // reset or perhaps a GPU had its configuration changed since it was included in another group).
    // We could ensure that our group has the settings we previously configured by enforcing that
    // configuration.

    result = dcgmConfigEnforce(dcgmHandle, myGroupId, statusHandle);

    // Again check for errors cleanup.
    if (result != DCGM_ST_OK)
    {
        std::cout << "Error enforcing group configuration. Return: " << errorString(result) << std::endl;
        displayError(statusHandle);
        goto cleanup;
    }

    std::cout << "Group configuration enforced.\n";

// Clean up consists of memory management and shutting down the host engine.
// This may not be desired in all applications.
cleanup:
    std::cout << "Cleaning up. \n";
    if (deviceConfigList)
        delete[] deviceConfigList;
    dcgmGroupDestroy(dcgmHandle, myGroupId);
    dcgmStatusDestroy(statusHandle);
    if (standalone)
        dcgmDisconnect(dcgmHandle);
    else
        dcgmStopEmbedded(dcgmHandle);
    dcgmShutdown();
    return result;
}

// This function will output information contained in our status handle. The handle contains
// errors which map to individual GPUs on the group. In this function we simply display those
// errors.
void displayError(dcgmStatus_t &statusHandle)
{
    dcgmReturn_t result;
    dcgmErrorInfo_t errorInfo;
    unsigned int numberOfErrors;

    // Get the number of errors in the statusHandle
    result = dcgmStatusGetCount(statusHandle, &numberOfErrors);

    if (result == DCGM_ST_OK)
    {
        for (unsigned int i = 0; i < numberOfErrors; i++)
        {
            result = dcgmStatusPopError(statusHandle, &errorInfo);
            if (result == DCGM_ST_OK)
            {
                std::cout << "Error on GPU: " << errorInfo.gpuId << ", Field ID: " << errorInfo.fieldId << std::endl;
                std::cout << "Error message: " << errorString((dcgmReturn_t)errorInfo.status) << std::endl;
                std::cout << std::endl;
            }
            else
            {
                std::cout << "Error processing an error in status handle.";
                break;
            }
        }
    }
    else
    {
        std::cout << "Error accessing status handle. Return: " << errorString(result) << std::endl;
    }

    dcgmStatusClear(statusHandle);
}
