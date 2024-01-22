/*
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

#include <sstream>
#include <string>

#include <dcgm_agent.h>
#include <dcgm_errors.h>
#include <dcgm_structs.h>

/*****************************************************************************/
class DcgmError
{
    /***************************PUBLIC***********************************/
public:
    enum class GpuIdTag : std::uint8_t
    {
        Unknown
    };

    explicit DcgmError(GpuIdTag /*unused*/)
        : m_code(DCGM_FR_OK)
        , m_formatMessage(nullptr)
        , m_nextSteps()
        , m_severity(DCGM_ERROR_NONE)
        , m_message()
        , m_errorDetail()
        , m_fullError()
        , m_details()
        , m_gpuId(-1)
    {}

    explicit DcgmError(unsigned int gpuId)
        : m_code(DCGM_FR_OK)
        , m_formatMessage(nullptr)
        , m_nextSteps()
        , m_severity(DCGM_ERROR_NONE)
        , m_message()
        , m_errorDetail()
        , m_fullError()
        , m_details()
        , m_gpuId(gpuId)
    {}

    DcgmError(const DcgmError &other)
        : m_code(other.m_code)
        , m_formatMessage(other.m_formatMessage)
        , m_nextSteps(other.m_nextSteps)
        , m_severity(other.m_severity)
        , m_category(other.m_category)
        , m_message(other.m_message)
        , m_errorDetail(other.m_errorDetail)
        , m_fullError(other.m_fullError)
        , m_details(other.m_details)
        , m_gpuId(other.m_gpuId)
    {}

    /* Getters and Setters */
    /*****************************************************************************/
    dcgmError_t GetCode() const
    {
        return m_code;
    }

    DcgmError &SetCode(dcgmError_t code)
    {
        m_code     = code;
        m_category = dcgmErrorGetCategoryByCode(code);
        m_severity = dcgmErrorGetPriorityByCode(code);
        return *this;
    }

    dcgmErrorCategory_t GetCategory() const
    {
        return m_category;
    }

    dcgmErrorSeverity_t GetSeverity() const
    {
        return m_severity;
    }

    /*****************************************************************************/
    const char *GetFormatMessage() const
    {
        return m_formatMessage;
    }

    /*****************************************************************************/
    const std::string &GetMessage() const
    {
        return m_fullError;
    }

    /*****************************************************************************/
    DcgmError &SetMessage(const std::string &msg)
    {
        m_message = msg;

        SetFullError();
        return *this;
    }

    /*****************************************************************************/
    void SetFullError()
    {
        std::stringstream buf;
        buf << m_message;

        if (m_errorDetail.empty() == false)
        {
            buf << m_errorDetail;
        }

        if (m_nextSteps.empty() == false)
        {
            buf << " " << m_nextSteps;
        }

        if (m_details.empty() == false)
        {
            buf << " " << m_details;
        }

        m_fullError = buf.str();
    }

    /*****************************************************************************/
    DcgmError &AddDcgmError(dcgmReturn_t ret)
    {
        std::stringstream buf;
        const char *msg = errorString(ret);
        if (msg != NULL)
        {
            buf << ": '" << errorString(ret) << "' (" << ret << ").";
        }
        else
        {
            buf << ": unknown error code (" << ret << ").";
        }

        m_errorDetail = buf.str();

        SetFullError();
        return *this;
    }

    /*****************************************************************************/
    DcgmError &SetNextSteps(const std::string &nextSteps)
    {
        m_nextSteps = nextSteps;

        SetFullError();
        return *this;
    }

    DcgmError &AddDetail(const std::string &additionalDetail)
    {
        m_details = additionalDetail;

        SetFullError();
        return *this;
    }

    /*****************************************************************************/
    const char *GetNextSteps() const
    {
        return m_nextSteps.c_str();
    }

    int GetGpuId() const
    {
        return m_gpuId;
    }

    DcgmError &SetGpuId(unsigned int gpuId)
    {
        m_gpuId = gpuId;
        return *this;
    }

    bool operator==(const DcgmError &other) const
    {
        return (other.m_code == m_code && other.m_gpuId == m_gpuId && other.m_fullError == m_fullError);
    }

    /***************************PRIVATE**********************************/
private:
    /* Variables */
    dcgmError_t m_code          = DCGM_FR_OK;
    const char *m_formatMessage = nullptr;
    std::string m_nextSteps;
    dcgmErrorSeverity_t m_severity = DCGM_ERROR_NONE;
    dcgmErrorCategory_t m_category = DCGM_FR_EC_NONE;

    std::string m_message;
    std::string m_errorDetail;
    std::string m_fullError;
    std::string m_details;
    int m_gpuId = -1; // -1 means the gpuId wasn't set yet
};


// Convenience macro for formatting and setting the error message
#define DCGM_ERROR_FORMAT_MESSAGE(errCode, errorObj, ...)                          \
    do                                                                             \
    {                                                                              \
        char buf_2M8SD15I[1024];                                                   \
        const char *fmt_2M8SD15I = dcgmGetErrorMeta(errCode)->msgFormat;           \
        snprintf(buf_2M8SD15I, sizeof(buf_2M8SD15I), fmt_2M8SD15I, ##__VA_ARGS__); \
        errorObj.SetMessage(buf_2M8SD15I);                                         \
        errorObj.SetNextSteps(dcgmGetErrorMeta(errCode)->suggestion);              \
        errorObj.SetCode(errCode);                                                 \
    } while (0)

// Convenience macro for formatting and setting the error message and adding an nvml error
#define DCGM_ERROR_FORMAT_MESSAGE_DCGM(errCode, errorObj, dcgmRet, ...)            \
    do                                                                             \
    {                                                                              \
        char buf_2M8SD15I[1024];                                                   \
        const char *fmt_2M8SD15I = errCode##_MSG;                                  \
        snprintf(buf_2M8SD15I, sizeof(buf_2M8SD15I), fmt_2M8SD15I, ##__VA_ARGS__); \
        errorObj.SetMessage(std::string(buf_2M8SD15I));                            \
        errorObj.AddDcgmError(dcgmRet);                                            \
        errorObj.SetCode(errCode);                                                 \
    } while (0)
