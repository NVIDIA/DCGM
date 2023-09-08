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
#pragma once

#include "DcgmStringHelpers.h"
#include "NvvsCommon.h"
#include "NvvsJsonStrings.h"

#include <DcgmJsonSerialize.hpp>
#include <DcgmLogging.h>

#include <charconv>
#include <optional>
#include <string>
#include <vector>

/*
 * This file contains the JSON serialization and deserialization logic and structures for the NVVS JSON result format.
 * Here is an example of the JSON format:
 * {
 *      "DCGM GPU Diagnostic": {
 *          "version": "1.7",
 *          "runtime_error": "error message", # Optional, may not have test_categories then
 *          "Warning": "Deprecation message", # Optional
 *          "test_categories" : [
 *              {
 *                  "category": "Hardware", #Deployment|Hardware|Integration|Performance|Custom
 *                  "tests": [
 *                      {
 *                          "name": "Test Name",
 *                          "results": [
 *                              {
 *                                  "gpu_ids": "gpu0,gpu1,gpu2",
 *                                  "status": "PASS|FAIL|WARN|SKIPPED",
 *                                  "warnings": [
 *                                      {
 *                                          "warning": "text",
 *                                          "error_id": "id"
 *                                      }
 *                                  ],
 *                                  "info": [
 *                                      "info text",
 *                                      "info text",
 *                                  ...
 *                                  ]
 *                              }
 *                          ]
 *                      }
 *                  ]
 *              }
 *          ]
 *      }
 * }
 */
namespace DcgmNs::Nvvs::Json
{

struct Info
{
    std::vector<std::string> messages;
    friend auto operator<=>(Info const &, Info const &) = default;
};

inline auto ParseJson(::Json::Value const &json, DcgmNs::JsonSerialize::To<Info>) -> std::optional<Info>
{
    Info info;
    if (!json.isArray())
    {
        log_error("Failed to parse JSON: 'info' is not an array");
        log_debug("JSON: {}", json.toStyledString());
        return std::nullopt;
    }

    if (!json.isArray())
    {
        log_error("Failed to parse JSON: 'info' is not an array");
        log_debug("JSON: {}", json.toStyledString());
        return std::nullopt;
    }
    for (auto const &message : json)
    {
        if (message.isString())
        {
            info.messages.push_back(message.asString());
        }
        else
        {
            log_error("Failed to parse JSON: 'info' contains a non-string element");
            log_debug("JSON: {}", json.toStyledString());
            return std::nullopt;
        }
    }
    return info;
}

inline auto ToJson(Info const &value) -> ::Json::Value
{
    ::Json::Value json;
    for (auto const &message : value.messages)
    {
        json.append(message);
    }
    return json;
}

struct Warning
{
    std::string message;
    std::optional<int> error_code;

    friend auto operator<=>(Warning const &, Warning const &) = default;
};

inline auto ParseJson(::Json::Value const &json, DcgmNs::JsonSerialize::To<Warning>) -> std::optional<Warning>
{
    Warning warning;
    if (json.isString())
    {
        // Case where there is no error code and the warning is just a string
        warning.message = json.asString();
        return warning;
    }

    if (!json.isObject())
    {
        log_error("Failed to parse JSON: 'warning' is not an object");
        log_debug("JSON: {}", json.toStyledString());
        return std::nullopt;
    }

    if (json.isMember(NVVS_WARNING))
    {
        if (json[NVVS_WARNING].isString())
        {
            warning.message = json[NVVS_WARNING].asString();
        }
        else
        {
            log_error("Failed to parse JSON: '{}' is not a string", NVVS_WARNING);
            log_debug("JSON: {}", json.toStyledString());
            return std::nullopt;
        }
    }
    else
    {
        log_error("Failed to parse JSON: {} is missing", NVVS_WARNING);
        log_debug("JSON: {}", json.toStyledString());
        return std::nullopt;
    }

    if (json.isMember(NVVS_ERROR_ID))
    {
        if (json[NVVS_ERROR_ID].isInt())
        {
            warning.error_code = json[NVVS_ERROR_ID].asInt();
        }
        else
        {
            log_error("Failed to parse JSON: {} is not an integer", NVVS_ERROR_ID);
            log_debug("JSON: {}", json.toStyledString());
            return std::nullopt;
        }
    }
    else
    {
        log_error("Failed to parse JSON: {} is missing", NVVS_ERROR_ID);
        log_debug("JSON: {}", json.toStyledString());
        return std::nullopt;
    }

    return warning;
}

inline auto ToJson(Warning const &value) -> ::Json::Value
{
    ::Json::Value json;
    if (value.error_code.has_value())
    {
        json[NVVS_WARNING]  = value.message;
        json[NVVS_ERROR_ID] = *value.error_code;
    }
    else
    {
        json = value.message;
    }

    return json;
}

struct Status
{
    nvvsPluginResult_t result;
    friend auto operator<=>(Status const &, Status const &) = default;
};

inline auto ParseJson(::Json::Value const &json, DcgmNs::JsonSerialize::To<Status>) -> std::optional<Status>
{
    if (!json.isString())
    {
        log_error("Failed to parse JSON: {} is not a string", NVVS_STATUS);
        log_debug("JSON: {}", json.toStyledString());
        return std::nullopt;
    }

    auto const statusStr = json.asString();
    if (statusStr == "PASS")
    {
        return Status { .result = NVVS_RESULT_PASS };
    }

    if (statusStr == "WARN")
    {
        return Status { .result = NVVS_RESULT_WARN };
    }

    if (statusStr == "SKIP")
    {
        return Status { .result = NVVS_RESULT_SKIP };
    }

    if (statusStr == "FAIL")
    {
        return Status { .result = NVVS_RESULT_FAIL };
    }

    log_error("Failed to parse JSON: {} is not a valid status", statusStr);
    log_debug("JSON: {}", json.toStyledString());
    return std::nullopt;
}

inline auto ToJson(Status const &value) -> ::Json::Value
{
    switch (value.result)
    {
        case NVVS_RESULT_PASS:
            return "PASS";
        case NVVS_RESULT_WARN:
            return "WARN";
        case NVVS_RESULT_SKIP:
            return "SKIP";
        case NVVS_RESULT_FAIL:
        default:
            return "FAIL";
    }
}

struct GpuIds
{
    std::set<int> ids;
    friend auto operator<=>(GpuIds const &, GpuIds const &) = default;
};

auto inline ParseJson(::Json::Value const &json, DcgmNs::JsonSerialize::To<GpuIds>) -> std::optional<GpuIds>
{
    GpuIds gpuIds;

    if (json.isInt())
    {
        gpuIds.ids.insert(json.asInt());
        return gpuIds;
    }

    if (!json.isString())
    {
        log_error("Failed to parse JSON: {} is not a string", NVVS_GPU_IDS);
        log_debug("JSON: {}", json.toStyledString());
        return std::nullopt;
    }

    auto const idsStr = json.asString();
    for (auto const &idStr : DcgmNs::Split(idsStr, ','))
    {
        int gpuId      = -1;
        auto [ptr, ec] = std::from_chars(idStr.data(), idStr.data() + idStr.size(), gpuId);
        if (ec != std::errc {})
        {
            log_error("Failed to parse JSON: {} is not a valid GPU ID", idStr);
            log_debug("JSON: {}", json.toStyledString());
            return std::nullopt;
        }
        gpuIds.ids.insert(gpuId);
    }

    return gpuIds;
}

auto inline ToJson(GpuIds const &value) -> ::Json::Value
{
    std::string idsStr;
    idsStr.reserve(value.ids.size() * 2);
    for (auto const &id : value.ids)
    {
        if (!idsStr.empty())
        {
            idsStr += ',';
        }
        idsStr += std::to_string(id);
    }
    return idsStr;
}

struct Result
{
    GpuIds gpuIds;
    Status status;
    std::optional<std::vector<Warning>> warnings;
    std::optional<Info> info;
    friend auto operator<=>(Result const &, Result const &) = default;
};

auto inline ParseJson(::Json::Value const &json, DcgmNs::JsonSerialize::To<Result>) -> std::optional<Result>
{
    using DcgmNs::JsonSerialize::Deserialize;
    using DcgmNs::JsonSerialize::TryDeserialize;
    Result result;
    if (!json.isObject())
    {
        log_error("Failed to parse JSON: result is not an object");
        log_debug("JSON: {}", json.toStyledString());
        return std::nullopt;
    }

    if (!json.isMember(NVVS_GPU_IDS))
    {
        log_error("Failed to parse JSON: {} is missing", NVVS_GPU_IDS);
        log_debug("JSON: {}", json.toStyledString());
        return std::nullopt;
    }
    auto const gpuIds = TryDeserialize<GpuIds>(json[NVVS_GPU_IDS]);
    if (!gpuIds.has_value())
    {
        log_error("Failed to parse JSON: {} is not a valid GPU ID list", NVVS_GPU_IDS);
        log_debug("JSON: {}", json[NVVS_GPU_IDS].toStyledString());
        return std::nullopt;
    }
    result.gpuIds = *gpuIds;

    if (!json.isMember(NVVS_STATUS))
    {
        log_error("Failed to parse JSON: {} is missing", NVVS_STATUS);
        log_debug("JSON: {}", json.toStyledString());
        return std::nullopt;
    }
    auto const status = TryDeserialize<Status>(json[NVVS_STATUS]);
    if (!status.has_value())
    {
        log_error("Failed to parse JSON: {} is not a valid status", NVVS_STATUS);
        log_debug("JSON: {}", json.toStyledString());
        return std::nullopt;
    }
    result.status = *status;

    if (json.isMember(NVVS_WARNINGS))
    {
        if (!json[NVVS_WARNINGS].isArray())
        {
            log_error("Failed to parse JSON: {} is not an array", NVVS_WARNINGS);
            log_debug("JSON: {}", json.toStyledString());
            return std::nullopt;
        }
        result.warnings = std::vector<Warning> {};
        for (auto const &warning : json[NVVS_WARNINGS])
        {
            if (warning.isNull())
            {
                log_debug("Skipping null warning");
                continue;
            }
            auto const parsedWarning = TryDeserialize<Warning>(warning);
            if (!parsedWarning.has_value())
            {
                log_error("Failed to parse JSON: {} is not a valid warning", NVVS_WARNINGS);
                log_debug("JSON: {}", json.toStyledString());
                return std::nullopt;
            }
            (*result.warnings).push_back(*parsedWarning);
        }
    }

    if (json.isMember(NVVS_INFO))
    {
        if (!json[NVVS_INFO].isArray())
        {
            log_error("Failed to parse JSON: {} is not an array", NVVS_INFO);
            log_debug("JSON: {}", json.toStyledString());
            return std::nullopt;
        }
        auto const parsedInfo = TryDeserialize<Info>(json[NVVS_INFO]);
        if (!parsedInfo.has_value())
        {
            log_error("Failed to parse JSON: {} is not a valid info", NVVS_INFO);
            log_debug("JSON: {}", json.toStyledString());
            return std::nullopt;
        }
        result.info = *parsedInfo;
    }

    return result;
}

inline auto ToJson(Result const &value) -> ::Json::Value
{
    using DcgmNs::JsonSerialize::Serialize;
    ::Json::Value json;
    json[NVVS_GPU_IDS] = Serialize(value.gpuIds);
    json[NVVS_STATUS]  = Serialize(value.status);
    if (value.warnings.has_value())
    {
        for (auto const &warning : (*value.warnings))
        {
            json[NVVS_WARNINGS].append(Serialize(warning));
        }
    }
    if (value.info.has_value())
    {
        json[NVVS_INFO] = Serialize(*value.info);
    }
    return json;
}

struct Test
{
    std::string name;
    std::vector<Result> results;
    friend auto operator<=>(Test const &, Test const &) = default;
};

auto inline ParseJson(::Json::Value const &json, DcgmNs::JsonSerialize::To<Test>) -> std::optional<Test>
{
    using DcgmNs::JsonSerialize::Deserialize;
    using DcgmNs::JsonSerialize::TryDeserialize;
    Test test;
    if (!json.isObject())
    {
        log_error("Failed to parse JSON: test is not an object");
        log_debug("JSON: {}", json.toStyledString());
        return std::nullopt;
    }

    if (!json.isMember(NVVS_TEST_NAME))
    {
        log_error("Failed to parse JSON: {} is missing", NVVS_TEST_NAME);
        log_debug("JSON: {}", json.toStyledString());
        return std::nullopt;
    }
    test.name = json[NVVS_TEST_NAME].asString();

    if (!json.isMember(NVVS_RESULTS) || !json[NVVS_RESULTS].isArray())
    {
        log_error("Failed to parse JSON: {} is missing or not an array", NVVS_RESULTS);
        log_debug("JSON: {}", json.toStyledString());
        return std::nullopt;
    }
    for (auto const &result : json[NVVS_RESULTS])
    {
        if (result.isNull())
        {
            log_debug("Skipping null result");
            continue;
        }
        auto const parsedResult = TryDeserialize<Result>(result);
        if (!parsedResult.has_value())
        {
            log_error("Failed to parse JSON: {} has an invalid result item", NVVS_RESULTS);
            log_debug("JSON: {}", result.toStyledString());
            return std::nullopt;
        }
        test.results.push_back(*parsedResult);
    }

    return test;
}

inline auto ToJson(Test const &value) -> ::Json::Value
{
    using DcgmNs::JsonSerialize::Serialize;
    ::Json::Value json;
    json[NVVS_TEST_NAME] = value.name;
    for (auto const &result : value.results)
    {
        json[NVVS_RESULTS].append(Serialize(result));
    }
    return json;
}

struct Category
{
    std::string category;
    std::vector<Test> tests;
    friend auto operator<=>(Category const &, Category const &) = default;
};

auto inline ParseJson(::Json::Value const &json, DcgmNs::JsonSerialize::To<Category>) -> std::optional<Category>
{
    using DcgmNs::JsonSerialize::Deserialize;
    using DcgmNs::JsonSerialize::TryDeserialize;
    Category category;
    if (!json.isObject())
    {
        log_error("Failed to parse JSON: category is not an object");
        log_debug("JSON: {}", json.toStyledString());
        return std::nullopt;
    }

    if (!json.isMember(NVVS_HEADER))
    {
        log_error("Failed to parse JSON: {} is missing", NVVS_HEADER);
        log_debug("JSON: {}", json.toStyledString());
        return std::nullopt;
    }
    category.category = json[NVVS_HEADER].asString();

    if (!json.isMember(NVVS_TESTS) || !json[NVVS_TESTS].isArray())
    {
        log_error("Failed to parse JSON: {} is missing or not an array", NVVS_TESTS);
        log_debug("JSON: {}", json.toStyledString());
        return std::nullopt;
    }
    for (auto const &test : json[NVVS_TESTS])
    {
        if (test.isNull())
        {
            log_debug("Skipping test represented by null JSON value");
            continue;
        }
        auto const parsedTest = TryDeserialize<Test>(test);
        if (!parsedTest.has_value())
        {
            log_error("Failed to parse JSON: {} has an invalid test item", NVVS_TESTS);
            log_debug("JSON: {}", test.toStyledString());
            return std::nullopt;
        }
        category.tests.push_back(*parsedTest);
    }

    return category;
}

inline auto ToJson(Category const &value) -> ::Json::Value
{
    using DcgmNs::JsonSerialize::Serialize;
    ::Json::Value json;
    json[NVVS_HEADER] = value.category;
    for (auto const &test : value.tests)
    {
        json[NVVS_TESTS].append(Serialize(test));
    }
    return json;
}

struct DiagnosticResults
{
    std::optional<std::string> version;
    std::optional<std::string> runtimeError;
    std::optional<std::string> warning;
    std::optional<std::string> driverVersion;
    std::optional<std::vector<Category>> categories;
    std::optional<std::vector<std::string>> devIds;
    std::optional<nvvsReturn_t> errorCode;
    friend auto operator<=>(DiagnosticResults const &, DiagnosticResults const &) = default;
};

inline bool MergeResults(DiagnosticResults &result, DiagnosticResults &&other)
{
    if (result.version != other.version)
    {
        log_error("Failed to merge results: version mismatch. Destination version: {}, Source version: {}",
                  result.version.has_value() ? (*result.version) : "nullopt",
                  other.version.has_value() ? (*other.version) : "nullopt");
        return false;
    }

    if (other.runtimeError.has_value())
    {
        result.runtimeError = result.runtimeError.has_value()
                                  ? (*result.runtimeError) + "\n" + std::move(*other.runtimeError)
                                  : std::move(other.runtimeError);
    }

    if (other.errorCode.has_value() && !result.errorCode.has_value())
    {
        if (!result.errorCode.has_value())
        {
            result.errorCode = other.errorCode;
        }
        else
        {
            log_debug(
                "Skipping errorCode assignment because it is already set. Result errorCode: {}, other errorCode: {}",
                *result.errorCode,
                *other.errorCode);
        }
    }

    if (other.warning.has_value())
    {
        result.warning = result.warning.has_value() ? (*result.warning) + "\n" + std::move(*other.warning)
                                                    : std::move(other.warning);
    }

    if (other.driverVersion.has_value() && !result.driverVersion.has_value())
    {
        result.driverVersion = std::move(other.driverVersion);
    }

    if (other.categories.has_value())
    {
        if (result.categories.has_value())
        {
            result.categories->insert(result.categories->end(),
                                      std::make_move_iterator(other.categories->begin()),
                                      std::make_move_iterator(other.categories->end()));
        }
        else
        {
            result.categories = std::move(other.categories);
        }
    }

    if (other.devIds.has_value() && !result.devIds.has_value())
    {
        result.devIds = std::move(other.devIds);
    }

    return true;
}

inline auto RemoveSingleQuotes(std::string_view str) -> std::string
{
    if (str.size() > 1)
    {
        if ((str.front() == '\'' && str.back() == '\'') || (str.front() == '"' && str.back() == '"'))
        {
            return std::string { str.substr(1, str.size() - 2) };
        }
    }
    return std::string { str };
}

auto inline ParseJson(::Json::Value const &rootJson, DcgmNs::JsonSerialize::To<DiagnosticResults>)
    -> std::optional<DiagnosticResults>
{
    using DcgmNs::JsonSerialize::Deserialize;
    using DcgmNs::JsonSerialize::TryDeserialize;

    DiagnosticResults diagnosticResults;

    if (!rootJson.isObject())
    {
        log_error("Failed to parse JSON: not an object");
        log_debug("JSON: {}", rootJson.toStyledString());
        return std::nullopt;
    }

    if (!rootJson.isMember(NVVS_NAME))
    {
        log_error("Failed to parse JSON: {} is missing", NVVS_NAME);
        log_debug("JSON: {}", rootJson.toStyledString());
        return std::nullopt;
    }

    auto const &json = rootJson[NVVS_NAME];

    if (!json.isMember(NVVS_VERSION_STR))
    {
        log_error("Failed to parse JSON: {} is missing", NVVS_VERSION_STR);
        log_debug("JSON: {}", json.toStyledString());
        return std::nullopt;
    }

    if (json[NVVS_VERSION_STR].isString())
    {
        diagnosticResults.version = json[NVVS_VERSION_STR].asString();
    }
    else
    {
        log_error("Failed to parse JSON: {} is not a string", NVVS_VERSION_STR);
        log_debug("JSON: {}", json.toStyledString());
        return std::nullopt;
    }

    if (json.isMember(NVVS_RUNTIME_ERROR))
    {
        diagnosticResults.runtimeError = json[NVVS_RUNTIME_ERROR].asString();
    }

    if (json.isMember(NVVS_ERROR_CODE))
    {
        diagnosticResults.errorCode = static_cast<nvvsReturn_t>(json[NVVS_ERROR_CODE].asInt());
    }

    if (json.isMember(NVVS_GLOBAL_WARN))
    {
        diagnosticResults.warning = json[NVVS_GLOBAL_WARN].asString();
    }

    if (!json.isMember(NVVS_HEADERS) || !json[NVVS_HEADERS].isArray())
    {
        if (!diagnosticResults.runtimeError.has_value())
        {
            log_error("Failed to parse JSON: {} is missing or not an array and the {} is does not present",
                      NVVS_HEADERS,
                      NVVS_RUNTIME_ERROR);
            log_debug("JSON: {}", json.toStyledString());
            return std::nullopt;
        }
    }
    else
    {
        diagnosticResults.categories = std::vector<Category> {};
        for (auto const &category : json[NVVS_HEADERS])
        {
            auto const parsedCategory = TryDeserialize<Category>(category);
            if (!parsedCategory.has_value())
            {
                log_error("Failed to parse JSON: {} has an invalid category item", NVVS_HEADERS);
                log_debug("JSON: {}", category.toStyledString());
                return std::nullopt;
            }
            (*diagnosticResults.categories).push_back(*parsedCategory);
        }
    }

    if (json.isMember(NVVS_GPU_DEV_IDS))
    {
        diagnosticResults.devIds = std::vector<std::string> {};
        for (auto const &devId : json[NVVS_GPU_DEV_IDS])
        {
            (*diagnosticResults.devIds).push_back(devId.asString());
        }
    }

    if (json.isMember(NVVS_DRIVER_VERSION))
    {
        diagnosticResults.driverVersion = json[NVVS_DRIVER_VERSION].asString();
    }

    return diagnosticResults;
}

inline auto ToJson(DiagnosticResults const &value) -> ::Json::Value
{
    using DcgmNs::JsonSerialize::Serialize;
    ::Json::Value result;

    auto &json = result[NVVS_NAME];

    if (value.version.has_value())
    {
        json[NVVS_VERSION_STR] = *value.version;
    }

    if (value.runtimeError.has_value())
    {
        json[NVVS_RUNTIME_ERROR] = *value.runtimeError;
    }

    if (value.errorCode.has_value())
    {
        json[NVVS_ERROR_CODE] = static_cast<int>(*value.errorCode);
    }

    if (value.warning.has_value())
    {
        json[NVVS_GLOBAL_WARN] = *value.warning;
    }

    if (value.driverVersion.has_value())
    {
        json[NVVS_DRIVER_VERSION] = *value.driverVersion;
    }

    if (value.categories.has_value())
    {
        json[NVVS_HEADERS] = ::Json::arrayValue;
        for (auto const &category : *value.categories)
        {
            json[NVVS_HEADERS].append(Serialize(category));
        }
    }

    if (value.devIds.has_value())
    {
        json[NVVS_GPU_DEV_IDS] = ::Json::arrayValue;
        for (auto const &gpuDevId : *value.devIds)
        {
            json[NVVS_GPU_DEV_IDS].append(gpuDevId);
        }
    }
    return result;
}

} // namespace DcgmNs::Nvvs::Json
