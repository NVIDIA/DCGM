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
#include <iomanip>
#include <iostream>
#include <stdexcept>

// Module names
#define MODULE_CORE_NAME       "Core"
#define MODULE_NVSWITCH_NAME   "NvSwitch"
#define MODULE_VGPU_NAME       "VGPU"
#define MODULE_INTROSPECT_NAME "Introspection"
#define MODULE_HEALTH_NAME     "Health"
#define MODULE_POLICY_NAME     "Policy"
#define MODULE_CONFIG_NAME     "Config"
#define MODULE_DIAG_NAME       "Diag"
#define MODULE_PROFILING_NAME  "Profiling"

// Helper for displaying module names
std::string moduleIdToName(dcgmModuleId_t moduleId)
{
    // If adding a case here, you very likely also need to add one in the
    // BlacklistModule::Execute method below
    switch (moduleId)
    {
        case DcgmModuleIdCore:
            return MODULE_CORE_NAME;
        case DcgmModuleIdNvSwitch:
            return MODULE_NVSWITCH_NAME;
        case DcgmModuleIdVGPU:
            return MODULE_VGPU_NAME;
        case DcgmModuleIdIntrospect:
            return MODULE_INTROSPECT_NAME;
        case DcgmModuleIdHealth:
            return MODULE_HEALTH_NAME;
        case DcgmModuleIdPolicy:
            return MODULE_POLICY_NAME;
        case DcgmModuleIdConfig:
            return MODULE_CONFIG_NAME;
        case DcgmModuleIdDiag:
            return MODULE_DIAG_NAME;
        case DcgmModuleIdProfiling:
            return MODULE_PROFILING_NAME;
        // Invalid module
        case DcgmModuleIdCount:
            return "Invalid module";

            // No default case in the hopes the compiler will complain about missing cases
    }

    // So the program compiles
    return "";
}

// Helper for displaying module status
std::string statusToStr(dcgmModuleStatus_t status)
{
    switch (status)
    {
        case DcgmModuleStatusNotLoaded:
            return "Not loaded";
        case DcgmModuleStatusBlacklisted:
            return "Blacklisted";
        case DcgmModuleStatusFailed:
            return "Failed to load";
        case DcgmModuleStatusLoaded:
            return "Loaded";
        case DcgmModuleStatusUnloaded:
            return "Unloaded";

            // No default case in the hopes the compiler will complain about missing cases
    }

    return "Invalid status";
}

////////////////////////////////////////////////////////////////////////////////
// This sample goes through the process of listing and blacklisting modules
// running a process and viewing the statistics of the group while the process ran.
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    (void)argc;
    (void)argv;

    // DCGM calls return a dcgmReturn_t which can be useful for error handling and control flow.
    // Whenever we call DCGM we will store the return in result and check it for errors.
    dcgmReturn_t result;

    dcgmHandle_t dcgmHandle = (dcgmHandle_t)NULL;
    bool standalone         = false;
    char hostIpAddress[16]  = { 0 };
    // We will need to cast to dcgmModuleId_t, but we need to read into a
    // unsigned int
    unsigned int blacklistModuleId;
    dcgmModuleGetStatuses_t getStatuses;

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

    std::cout << "Blacklist module. Enter module ID:\n";

    while (!(std::cin >> blacklistModuleId))
    {
        std::cout << "Invalid input.\n";
        std::cin.clear();
        std::cin.ignore();
    }

    // Need to validate the input
    if (blacklistModuleId >= DcgmModuleIdCount)
    {
        std::cout << "Invalid module ID. Aborting. \n";
        goto cleanup;
    }

    result = dcgmModuleBlacklist(dcgmHandle, (dcgmModuleId_t)blacklistModuleId);
    if (result != DCGM_ST_OK)
    {
        std::cout << "Could not blacklist module. Return: " << errorString(result) << std::endl;
        // Don't need to go to cleanup. This is an expected error. It could mean
        // that the module is already loaded
    }

    // We need to zero getStatuses so we don't pass unintended information in
    // the function calls below
    memset(&getStatuses, 0, sizeof(getStatuses));

    // Set version to the current version
    getStatuses.version = dcgmModuleGetStatuses_version;

    // Get statuses
    result = dcgmModuleGetStatuses(dcgmHandle, &getStatuses);
    if (result != DCGM_ST_OK)
    {
        std::cout << "Could not get statuses. Return: " << errorString(result) << std::endl;
        goto cleanup;
    }

    std::cout << std::left << std::setw(12) << "Module ID" << std::left << std::setw(16) << "Module name"
              << "Status" << std::endl;

    for (unsigned int i = 0; i < getStatuses.numStatuses; i++)
    {
        std::cout << std::left << std::setw(12) << getStatuses.statuses[i].id << std::left << std::setw(16)
                  << moduleIdToName(getStatuses.statuses[i].id) << statusToStr(getStatuses.statuses[i].status)
                  << std::endl;
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
