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
#pragma once

#include "dcgm_core_communication.h"
#include "dcgm_core_structs.h"
#include "dcgm_module_structs.h"
#include "dcgm_structs.h"
#include <DcgmCoreProxy.h>
#include <DcgmFvBuffer.h>
#include <DcgmProtobuf.h>
#include <stdexcept>
#include <string>

/* Base class for a DCGM Module that will be plugged into the host engine to service
 * requests. Extend this class with your own.
 */

class DcgmModule
{
public:
    /*************************************************************************/
    virtual ~DcgmModule(); /* Virtual destructor because of ancient compiler */

    /*************************************************************************/
    /*
     * Helper method to look at the first 4 bytes of the blob field of a moduleCommand
     * and compare the version against the expected version of the message
     *
     * Returns: DCGM_ST_OK if the versions match
     *          DCGM_ST_VER_MISMATCH if the versions mismatch
     *          DCGM_ST_? on other error
     *
     */
    static dcgmReturn_t CheckVersion(dcgm_module_command_header_t const *moduleCommand, unsigned int compareVersion);

    /*************************************************************************/
    /*
     * Virtual method to process a DCGM Module command
     *
     * moduleCommand contains the command for this module. Call moduleCommand->set_blob()
     * to set the bytes that will be returned to the caller on the client side.
     *
     * Returns: DCGM_ST_OK if processing the command succeeded
     *          DCGM_ST_? enum value on error. Will be returned to the caller
     */
    virtual dcgmReturn_t ProcessMessage(dcgm_module_command_header_t *moduleCommand) = 0;

    /*************************************************************************/
    /*
     * Virtual method that instructs module to update its severity form
     * Hostengine
     */
    virtual void OnLoggingSeverityChange(dcgm_core_msg_logging_changed_t *msg)
    {}
};


/* This class is a template so that it's generated inside the module and its
 * logger does not share Hostengine's logger
 */

template <unsigned int moduleId>
class DcgmModuleWithCoreProxy : public DcgmModule
{
public:
    DcgmModuleWithCoreProxy(const dcgmCoreCallbacks_t &dcc)
        : m_coreCallbacks(dcc)
        , m_coreProxy(dcc)
    {
        DcgmLogging::initLogToHostengine(m_coreProxy.GetLoggerSeverity(0));
        DcgmLogging::setHostEngineCallback((hostEngineAppenderCallbackFp_t)m_coreCallbacks.loggerfunc);
        char const *moduleName;
        if (DCGM_ST_OK == dcgmModuleIdToName(static_cast<dcgmModuleId_t>(moduleId), &moduleName))
        {
            DcgmLogging::setHostEngineComponentName(moduleName);
        }
        DCGM_LOG_DEBUG << "Initialized logging for module " << moduleId;
        DCGM_LOG_DEBUG << "Logger address " << DcgmLogging::getLoggerAddress();
    }

    void OnLoggingSeverityChange(dcgm_core_msg_logging_changed_t *msg) override
    {
        dcgmReturn_t dcgmReturn = CheckVersion(&msg->header, dcgm_core_msg_logging_changed_version);
        if (DCGM_ST_OK != dcgmReturn)
        {
            return;
        }

        DcgmLoggingSeverity_t severity = m_coreProxy.GetLoggerSeverity(0, static_cast<loggerCategory_t>(BASE_LOGGER));
        if (severity == DcgmLoggingSeverityUnspecified)
        {
            DCGM_LOG_ERROR << "Encountered error while fetching severity for BASE_LOGGER";
        }
        DcgmLogging::setLoggerSeverity<BASE_LOGGER>(severity);

        severity = m_coreProxy.GetLoggerSeverity(0, SYSLOG_LOGGER);
        if (severity == DcgmLoggingSeverityUnspecified)
        {
            DCGM_LOG_ERROR << "Encountered error while fetching severity for SYSLOG_LOGGER";
        }
        DcgmLogging::setLoggerSeverity<SYSLOG_LOGGER>(severity);
    }

protected:
    dcgmCoreCallbacks_t m_coreCallbacks;
    DcgmCoreProxy m_coreProxy;
};

/* Callback functions for allocating and freeing DcgmModules. These are found
   in the modules' shared library with dlsym */
typedef DcgmModule *(*dcgmModuleAlloc_f)(dcgmCoreCallbacks_t *dcc);
typedef void (*dcgmModuleFree_f)(DcgmModule *);
typedef dcgmReturn_t (*dcgmModuleProcessMessage_f)(DcgmModule *, dcgm_module_command_header_t *moduleCommand);

dcgmReturn_t PassMessageToModule(DcgmModule *module, dcgm_module_command_header_t *moduleCommand);

namespace
{
/* Helper function to wrap class instantiation inside try/catch. Try to avoid throwing exceptions
 * across module boundaries. */
template <class T>
DcgmModule *SafeWrapper(T func)
{
    try
    {
        return std::invoke(func);
    }
    catch (std::bad_alloc &ex)
    {
        DCGM_LOG_ERROR << "An exception occurred when allocating memory for the module";
    }
    catch (std::runtime_error const &ex)
    {
        DCGM_LOG_ERROR << "A runtime exception occured when creating module. Ex: " << ex.what();
    }
    catch (...)
    {
        DCGM_LOG_ERROR << "An unknown exception occured when creating module";
    }
    return nullptr;
}
} // namespace
