/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
#ifndef DCGM_DIAG_RESPONSE_WRAPPER_H
#define DCGM_DIAG_RESPONSE_WRAPPER_H

#include "dcgm_structs.h"
#include "dcgm_structs_internal.h"

#include <DcgmLogging.h>
#include <DiagResponseUtils.h>
#include <json/json.h>
#include <string>

extern const std::string_view denylistName;
extern const std::string_view nvmlLibName;
extern const std::string_view cudaMainLibName;
extern const std::string_view cudaTkLibName;
extern const std::string_view permissionsName;
extern const std::string_view persistenceName;
extern const std::string_view envName;
extern const std::string_view pageRetirementName;
extern const std::string_view graphicsName;
extern const std::string_view inforomName;
extern const std::string_view fabricManagerName;

extern const std::string_view swTestNames[];

/**
 * Used to differentiate different response message types.
 */
enum class MsgType
{
    Info, //!< Informational message
    Error //!< Error message
};

// Forward declaration of friend DcgmModuleDiag
class DcgmModuleDiag;

/*****************************************************************************/
/*
 * Class for handling the different versions of the diag response
 */
class DcgmDiagResponseWrapper
{
public:
    /*****************************************************************************/
    DcgmDiagResponseWrapper();

    /*****************************************************************************/
    void RecordSystemError(std::string const &errorStr) const;

    /*****************************************************************************/

    /**
     * Set the version of the response
     * @param response The response to set
     * @return DCGM_ST_OK if the version was set, DCGM_ST_NOT_SUPPORTED if the version is already set, or
     * DCGM_ST_GENERIC_ERROR if the version is not supported
     */
    template <typename DiagResponseType>
        requires DcgmNs::IsDiagResponse<DiagResponseType>
    dcgmReturn_t SetVersion(DiagResponseType *response)
    {
        if (m_version != 0)
        {
            // Message should match DDRW_VER_ALREADY_SET_FMT
            log_warning("Unable to set response version to {}, already set to {}.", m_version, m_version);
            return DCGM_ST_NOT_SUPPORTED;
        }

        if constexpr (std::is_same_v<DiagResponseType, dcgmDiagResponse_v12>)
        {
            return SetVersion12(response);
        }
        else if constexpr (std::is_same_v<DiagResponseType, dcgmDiagResponse_v11>)
        {
            return SetVersion11(response);
        }
        else if constexpr (std::is_same_v<DiagResponseType, dcgmDiagResponse_v10>)
        {
            return SetVersion10(response);
        }
        else if constexpr (std::is_same_v<DiagResponseType, dcgmDiagResponse_v9>)
        {
            return SetVersion9(response);
        }
        else if constexpr (std::is_same_v<DiagResponseType, dcgmDiagResponse_v8>)
        {
            return SetVersion8(response);
        }
        else if constexpr (std::is_same_v<DiagResponseType, dcgmDiagResponse_v7>)
        {
            return SetVersion7(response);
        }
    }

    /*****************************************************************************/
    dcgmReturn_t SetResult(std::span<std::byte> data) const;

    /*****************************************************************************/
    bool HasTest(const std::string &pluginName) const;

    /*****************************************************************************/
    dcgmReturn_t MergeEudResponse(DcgmDiagResponseWrapper &eudResponse);

    dcgmReturn_t AdoptEudResponse(DcgmDiagResponseWrapper &eudResponse);

    bool AddCpuSerials();

    std::string GetSystemErr() const;

    unsigned int GetVersion() const;

#ifndef __DIAG_UNIT_TESTING__
private:
#endif

    /* Deprecated methods. Use SetVersion() instead. */
    dcgmReturn_t SetVersion12(dcgmDiagResponse_v12 *response);
    dcgmReturn_t SetVersion11(dcgmDiagResponse_v11 *response);
    dcgmReturn_t SetVersion10(dcgmDiagResponse_v10 *response);
    dcgmReturn_t SetVersion9(dcgmDiagResponse_v9 *response);
    dcgmReturn_t SetVersion8(dcgmDiagResponse_v8 *response);
    dcgmReturn_t SetVersion7(dcgmDiagResponse_v7 *response);

    union
    {
        dcgmDiagResponse_v12 *v12ptr; // A pointer to the version12 struct
        dcgmDiagResponse_v11 *v11ptr; // A pointer to the version11 struct
        dcgmDiagResponse_v10 *v10ptr; //!< Deprecated. A pointer to the version10 struct
        dcgmDiagResponse_v9 *v9ptr;   //!< Deprecated. A pointer to the version9 struct
        dcgmDiagResponse_v8 *v8ptr;   //!< Deprecated. A pointer to the version8 struct
        dcgmDiagResponse_v7 *v7ptr;   //!< Deprecated. A pointer to the version7 struct
    } m_response;

    unsigned int m_version; //!< records the version of our dcgmDiagResponse_t

    /*****************************************************************************/
    bool StateIsValid() const;
    friend class DcgmModuleDiag;
};

#endif
