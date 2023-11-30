/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
#ifndef _NVVS_NVVS_JsonOutput_H
#define _NVVS_NVVS_JsonOutput_H

#include "JsonResult.hpp"
#include "NvvsCommon.h"
#include "NvvsJsonStrings.h"
#include "Output.h"
#include "json/json.h"
#include <iostream>
#include <sstream>
#include <utility>
#include <vector>

class JsonOutput : public Output
{
    /***************************PUBLIC***********************************/
public:
    explicit JsonOutput(std::vector<unsigned int> gpuIndices);

    ~JsonOutput() override = default;

    void header(const std::string &headerSting) override;
    void Result(nvvsPluginResult_t overallResult,
                const std::vector<dcgmDiagSimpleResult_t> &perGpuResults,
                const std::vector<dcgmDiagErrorDetail_v2> &errors,
                const std::vector<dcgmDiagErrorDetail_v2> &info,
                const std::optional<std::any> &pluginSpecificData = std::nullopt) override;
    void prep(const std::string &testString) override;
    void updatePluginProgress(unsigned int progress, bool clear) override;
    void print() override;
    void addInfoStatement(const std::string &info) override;
    void AddGpusAndDriverVersion(std::vector<Gpu *> &gpuList) override;

    /***************************PRIVATE**********************************/
private:
    Json::Value m_root;
    unsigned int m_testIndex { 0 };
    unsigned int headerIndex { 0 };
    int m_gpuId { -1 };
    int m_globalInfoCount { 0 };
    std::string gpuList;
    std::vector<unsigned int> m_gpuIndices;
    bool softwareTest { false };

    static void AppendError(const dcgmDiagErrorDetail_v2 &error,
                            Json::Value &resultField,
                            const std::string &prefix = "");
    static void AppendInfo(const dcgmDiagErrorDetail_v2 &info,
                           Json::Value &resultField,
                           const std::string &prefix = "");
};

#endif // _NVVS_NVVS_JsonOutput_H
