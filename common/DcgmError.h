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

#include <sstream>
#include <string>

#include <dcgm_agent.h>
#include <dcgm_errors.h>
#include <dcgm_structs.h>

/*****************************************************************************/
/*
 * Priority levels for errors. FAILURE indicates that isolation is needed.
 */
typedef enum dcgmErrorPriority_enum
{
    DCGM_FR_LVL_WARNING = 1,
    DCGM_FR_LVL_FAILURE = 2
} dcgmErrorPriority_t;


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
        , m_priority(DCGM_FR_LVL_WARNING)
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
        , m_priority(DCGM_FR_LVL_WARNING)
        , m_message()
        , m_errorDetail()
        , m_fullError()
        , m_details()
        , m_gpuId(gpuId)
    {}
    /* Getters and Setters */
    /*****************************************************************************/
    dcgmError_t GetCode() const
    {
        return m_code;
    }

    void SetCode(dcgmError_t code)
    {
        m_code = code;
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
    void SetMessage(const std::string &msg)
    {
        m_message = msg;

        SetFullError();
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
    void AddDcgmError(dcgmReturn_t ret)
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
    }

    /*****************************************************************************/
    void SetNextSteps(const std::string &nextSteps)
    {
        m_nextSteps = nextSteps;

        SetFullError();
    }

    void AddDetail(const std::string &additionalDetail)
    {
        m_details = additionalDetail;

        SetFullError();
    }

    /*****************************************************************************/
    const char *GetNextSteps() const
    {
        return m_nextSteps.c_str();
    }

    /*****************************************************************************/
    dcgmErrorPriority_t GetPriority() const
    {
        return m_priority;
    }

    int GetGpuId() const
    {
        return m_gpuId;
    }

    void SetGpuId(unsigned int gpuId)
    {
        m_gpuId = gpuId;
    }

    /***************************PRIVATE**********************************/
private:
    /* Variables */
    dcgmError_t m_code          = DCGM_FR_OK;
    const char *m_formatMessage = nullptr;
    std::string m_nextSteps;
    dcgmErrorPriority_t m_priority = DCGM_FR_LVL_WARNING;

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
