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

#include <string>

#include "dcgm_agent.h"
#include "dcgm_structs.h"

class DcgmHandle
{
public:
    DcgmHandle();
    DcgmHandle(dcgmHandle_t handle)
        : m_lastReturn(DCGM_ST_OK)
        , m_handle(handle)
        , m_ownHandle(false)
    {}

    DcgmHandle(DcgmHandle &&other) noexcept;
    DcgmHandle(const DcgmHandle &other) = delete;
    DcgmHandle &operator=(const DcgmHandle &other) = delete;
    DcgmHandle &operator                           =(dcgmHandle_t handle);
    ~DcgmHandle();

    /*
     * Translate the DCGM return code to a human-readable string
     *
     * @param ret      IN : the return code to translate to a string
     * @return
     * A string translation of the error code : On Success
     * 'Unknown error from DCGM: <ret>'       : On Failure
     */
    std::string RetToString(dcgmReturn_t ret);

    /*
     * Establishes a connection with DCGM, saving the handle internally. This MUST be called before using
     * the other methods
     *
     * @param dcgmHostname    IN : the hostname of the nv-hostengine we should connect to. If "" is specified,
     *                             will connect to localhost.
     * @return
     * DCGM_ST_OK : On Success
     * DCGM_ST_*  : On Failure
     */
    dcgmReturn_t ConnectToDcgm(const std::string &dcgmHostname);

    /*
     * Get a string representation of the last error
     *
     * @return
     * The last DCGM return code as a string, or "" if the last call was successful
     */
    std::string GetLastError();

    /*
     * Return the DCGM handle - it will still be owned by this class
     *
     * @return
     * the handle or 0 if it hasn't been initialized
     */
    dcgmHandle_t GetHandle();

    /*
     * Destroys the allocated members of this class if they exist
     */
    void Cleanup();

private:
    dcgmReturn_t m_lastReturn; // The last return code from DCGM
    dcgmHandle_t m_handle;     // The handle for the DCGM connection
    bool m_ownHandle;          // True if we created the connection to DCGM, false otherwise
};
