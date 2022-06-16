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
#ifndef DCGM_DIAG_RESPONSE_WRAPPER_H
#define DCGM_DIAG_RESPONSE_WRAPPER_H

#include <string>

#include "dcgm_structs.h"
#include "json/json.h"

extern const std::string_view blacklistName;
extern const std::string_view nvmlLibName;
extern const std::string_view cudaMainLibName;
extern const std::string_view cudaTkLibName;
extern const std::string_view permissionsName;
extern const std::string_view persistenceName;
extern const std::string_view envName;
extern const std::string_view pageRetirementName;
extern const std::string_view graphicsName;
extern const std::string_view inforomName;

extern const std::string_view swTestNames[];

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
    void InitializeResponseStruct(unsigned int numGpus);

    /*****************************************************************************/
    void SetPerGpuResponseState(unsigned int testIndex,
                                dcgmDiagResult_t result,
                                unsigned int gpuIndex,
                                unsigned int rc = 0);

    /*****************************************************************************/
    void AddPerGpuMessage(unsigned int testIndex, const std::string &msg, unsigned int gpuIndex, bool warning);

    /*****************************************************************************/
    void SetGpuIndex(unsigned int gpuIndex);

    /*****************************************************************************/
    void RecordSystemError(const std::string &errorStr) const;

    /*****************************************************************************/
    void SetGpuCount(unsigned int gpuCount) const;

    /*****************************************************************************/
    static unsigned int GetBasicTestResultIndex(std::string_view const &testname);

    /*****************************************************************************/
    dcgmReturn_t SetVersion6(dcgmDiagResponse_v6 *response);
    dcgmReturn_t SetVersion7(dcgmDiagResponse_v7 *response);

    /*****************************************************************************/
    dcgmReturn_t RecordTrainingMessage(const std::string &trainingMsg) const;

    /*****************************************************************************/
    dcgmReturn_t AddErrorDetail(unsigned int gpuIndex,
                                unsigned int testIndex,
                                const std::string &testname,
                                dcgmDiagErrorDetail_t &ed,
                                dcgmDiagResult_t result);

    /*****************************************************************************/
    bool IsValidGpuIndex(unsigned int gpuIndex);

private:
    union
    {
        dcgmDiagResponse_v6 *v6ptr; // A pointer to the version6 struct
        dcgmDiagResponse_v7 *v7ptr; // A pointer to the version7 struct
    } m_response;
    unsigned int m_version; // records the version of our dcgmDiagResponse_t

    /*****************************************************************************/
    bool StateIsValid() const;
};

#endif
