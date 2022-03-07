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

// Definitions at bottom of file
void displayError(dcgmStatus_t &statusHandle);
int violationRegistration(void *data);

////////////////////////////////////////////////////////////////////////////////
// This sample demonstrates the process of creating a group and getting/setting
// the policies of the GPUs in that group. Policy registration is also shown.
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
    char groupName[]          = "MyGroup";
    dcgmGpuGrp_t myGroupId    = (dcgmGpuGrp_t)NULL;
    dcgmStatus_t statusHandle = (dcgmStatus_t)NULL;
    dcgmPolicy_t devicePolicyList[2];
    int conditionBuffer = 0;
    dcgmPolicy_t myGroupPolicy;

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
    // In this example we are going to create a group with the first two GPUs listed (If only one GPU
    // on the system it will form a group of one GPU). See DCGM_GROUPING for more information.

    // We can query DCGM to give us a list of all the GPUs on the system and select from
    // those available.

    // Call DCGM to return a list of all GPUs on the system. Storing the GPU IDs in our array and the
    // total number of GPUs in count.

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
        result = DCGM_ST_GENERIC_ERROR; // set this before we go to cleanup since we were given a bad IP address.
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

    // We now know what GPUs are available to us, but let's say we want a group with the first two GPUs
    // listed as an example. We pass in a reference to our dcgmGpuGrp_t to store our group's ID.

    // Create our group which we would like to start off empty.
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

    // Add the first two GPUs

    for (int i = 0; i < count; i++)
    {
        // Only add up to 2 GPUs
        if (i >= 2)
        {
            count = 2;
            break;
        }

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

    // We now have a group with two (or one) GPUs.
    // However before we can work with group policies we need a DCGM status handle. The status handle
    // gives us per GPU errors which can be useful for control flow and error handling. An example
    // function to demonstrate popping errors from a status handle is implemented at the end of the file.
    // If we didn't want or need this kind of error reporting, we can simply pass NULL in anywhere a
    // status handle is used as a parameter.

    result = dcgmStatusCreate(&statusHandle);

    // Check the result to see if our DCGM operation was successful.
    if (result != DCGM_ST_OK)
    {
        std::cout << "Error creating status handler. Return: " << errorString(result) << std::endl;
        goto cleanup;
    }

    // Now we have all the pieces in place to get and set policies for our group. Using devicePolicyList
    // let's query for and then output the current configuration of our group. I am using 2 here as the
    // array length because we added 2 GPUs to the group. It is often a best practice to use
    // DCGM_MAX_NUM_DEVICES as the array length to ensure that you will never run into memory problems,
    // but it is not needed in this example.

    // Before making DCGM calls that fill in object/struct information then we need to set the version of
    // our structs. This will tell the Host Engine what type of struct to fill in. If the client has
    // a greater version number than the Host Engine then an error will be returned since the host engine
    // will unaware of how to handle that type of request.
    for (int i = 0; i < count; i++)
    {
        devicePolicyList[i].version = dcgmPolicy_version;
    }

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

    result = dcgmPolicyGet(dcgmHandle, myGroupId, count, devicePolicyList, statusHandle);

    // Check the result to see if our DCGM operation was successful. However we now
    // have a status handle which can be used for per GPU error checking.
    if (result != DCGM_ST_OK)
    {
        std::cout << "Error getting group policy information. Return: " << errorString(result) << std::endl;
        displayError(statusHandle);
        goto cleanup;
    }
    else
    {
        // If successful output our policy information.
        std::cout << "Policy Information\n";
        for (int i = 0; i < count; i++)
        {
            std::cout << "GPU: " << i << std::endl;
            std::cout << "Condition: " << devicePolicyList[i].condition << std::endl;
            std::cout << "Action: " << ((devicePolicyList[i].action) ? "Reset GPU" : "None") << std::endl;
            std::cout << "Validation: " << devicePolicyList[i].validation << std::endl;
            std::cout << "Mode: " << ((devicePolicyList[i].mode) ? "Manual" : "Auto") << std::endl;
            std::cout << std::endl;
        }
    }

    // Now let's go through setting a group policy. We fill in the information for myGroupPolicy and set it
    // on our group ID.

    // Again set the version for reasons mentioned above.
    myGroupPolicy.version = dcgmPolicy_version;

    // Set mode, validation and action here.
    myGroupPolicy.mode       = (dcgmPolicyMode_t)DCGM_OPERATION_MODE_AUTO;
    myGroupPolicy.action     = DCGM_POLICY_ACTION_NONE;
    myGroupPolicy.validation = DCGM_POLICY_VALID_SV_SHORT;

    // Set violation conditions here. Note that the tag must be set to refelect the type of the value.
    conditionBuffer |= DCGM_POLICY_COND_DBE;
    myGroupPolicy.parms[0].tag         = dcgmPolicyConditionParams_t::BOOL;
    myGroupPolicy.parms[0].val.boolean = true;

    conditionBuffer |= DCGM_POLICY_COND_PCI;
    myGroupPolicy.parms[1].tag         = dcgmPolicyConditionParams_t::BOOL;
    myGroupPolicy.parms[1].val.boolean = true;

    conditionBuffer |= DCGM_POLICY_COND_THERMAL;
    myGroupPolicy.parms[3].tag       = dcgmPolicyConditionParams_t::LLONG;
    myGroupPolicy.parms[3].val.llval = 90;

    conditionBuffer |= DCGM_POLICY_COND_NVLINK;
    myGroupPolicy.parms[5].tag         = dcgmPolicyConditionParams_t::BOOL;
    myGroupPolicy.parms[5].val.boolean = true;

    myGroupPolicy.condition = (dcgmPolicyCondition_t)conditionBuffer;

    // Set our group policy.
    result = dcgmPolicySet(dcgmHandle, myGroupId, &myGroupPolicy, statusHandle);

    // Check the result to see if our DCGM operation was successful.
    if (result == DCGM_ST_NOT_SUPPORTED)
    {
        std::cout << "Policy setting was not supported. This is expected for non-Tesla GPUs. " << std::endl;
        std::cout << "Skipping the rest of this sample." << std::endl;
        result = DCGM_ST_OK;
        goto cleanup;
    }
    else if (result != DCGM_ST_OK)
    {
        std::cout << "Error setting group policy information. Return: " << errorString(result) << std::endl;
        displayError(statusHandle);
        goto cleanup;
    }
    else
    {
        std::cout << "Policy successfully set.\n";
    }

    // Now that our policy has been set for our group any violations of that policy will cause the
    // action and validation to be performed. If we want additional functionality when one of the
    // violations occurs we can register for policy updates. This allows us to handle violations
    // in our own way. We could hand in a new set of conditions, but let's use the conditions for the
    // policy we just set on the group. In this example we will simply print some of the information
    // to the screen when a violation occurs.

    // Note: if we wanted a condition of double bit and PCIe errors we would use
    // dcgmPolicyCondition_t condition = (DCGM_POLICY_COND_DBE |  DCGM_POLICY_COND_PCI);

    result = dcgmPolicyRegister(
        dcgmHandle, myGroupId, myGroupPolicy.condition, violationRegistration, violationRegistration);

    // Check the result to see if our DCGM operation was successful.
    if (result != DCGM_ST_OK)
    {
        std::cout << "Error registering policy. Return: " << errorString(result) << std::endl;
        goto cleanup;
    }
    else
    {
        std::cout << "Policy successfully registered.\n";
    }

    // We could now run a process and any errors / events in our conditions would be caught and
    // would trigger our policy action, our policy validation as well as our callback. We used the same
    // callback function twice above, so our message will be printed twice.

    result = dcgmPolicyUnregister(dcgmHandle, myGroupId, myGroupPolicy.condition);

    // Check the result to see if our DCGM operation was successful.
    if (result != DCGM_ST_OK)
    {
        std::cout << "Error setting group policy information. Return: " << errorString(result) << std::endl;
        goto cleanup;
    }
    else
    {
        std::cout << "Policy successfully unregistered.\n";
    }

// Clean up consists of memory management and shutting down the host engine.
// This may not be desired in all applications.
cleanup:
    std::cout << "Cleaning up.\n";
    dcgmStatusDestroy(statusHandle);
    dcgmGroupDestroy(dcgmHandle, myGroupId);
    if (standalone)
        dcgmDisconnect(dcgmHandle);
    else
        dcgmStopEmbedded(dcgmHandle);
    dcgmShutdown();
    return result;
}

// This function will output information contained in our status handle. In this
// case we shutdown DCGM as well, however that may not be needed.
void displayError(dcgmStatus_t &statusHandle)
{
    dcgmReturn_t result;
    unsigned int numberOfErrors;
    dcgmErrorInfo_t errorInfo;

    // Get the number of erros within the status handle
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
        }
    }
}

// In this function we simply print what kind of condition is in the response. The callback response
// struct has a lot more useful information here that we could use but let's keep it simple.
int violationRegistration(void *data)
{
    std::string errorMessage;
    dcgmPolicyCallbackResponse_t *response = (dcgmPolicyCallbackResponse_t *)data;
    switch (response->condition)
    {
        case DCGM_POLICY_COND_DBE:
            errorMessage = "Double-bit ECC error.";
            break;
        case DCGM_POLICY_COND_PCI:
            errorMessage = "PCIe error.";
            break;
        case DCGM_POLICY_COND_MAX_PAGES_RETIRED:
            errorMessage = "Maximum number of retired pages error.";
            break;
        case DCGM_POLICY_COND_THERMAL:
            errorMessage = "Thermal policy violation.";
            break;
        case DCGM_POLICY_COND_POWER:
            errorMessage = "Power policy violation..";
            break;
        case DCGM_POLICY_COND_NVLINK:
            errorMessage = "Nvlink policy violation";
            break;
        default:
            errorMessage = "Unknown error.";
            break;
    }

    std::cout << "Detected callback: " << errorMessage << std::endl;

    return 0;
}
