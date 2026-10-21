/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

#include "NvBandwidthOutputParser.hpp"

#include <memory>

namespace
{

struct JsonObjectCandidate
{
    size_t startPosition;
    std::string_view object;
    size_t endPosition;
};

std::optional<JsonObjectCandidate> ExtractCompleteJsonObject(std::string_view output, size_t searchStart)
{
    while (true)
    {
        size_t const start = output.find('{', searchStart);
        if (start == std::string_view::npos)
        {
            return std::nullopt;
        }

        size_t depth    = 0;
        bool inString   = false;
        bool isEscaping = false;
        for (size_t i = start; i < output.size(); ++i)
        {
            char const ch = output[i];
            if (inString)
            {
                if (isEscaping)
                {
                    isEscaping = false;
                    continue;
                }
                if (ch == '\\')
                {
                    isEscaping = true;
                    continue;
                }
                if (ch == '"')
                {
                    inString = false;
                }
                continue;
            }

            if (ch == '"')
            {
                inString = true;
                continue;
            }
            if (ch == '{')
            {
                ++depth;
                continue;
            }
            if (ch == '}')
            {
                --depth;
                if (depth == 0)
                {
                    return JsonObjectCandidate {
                        .startPosition = start,
                        .object        = output.substr(start, i - start + 1),
                        .endPosition   = i + 1,
                    };
                }
            }
        }

        // An unmatched brace may be launcher noise, so retry at the next opening brace.
        searchStart = start + 1;
    }
}

std::optional<Json::Value> ParseJsonObject(std::string_view jsonObject)
{
    Json::Value root;
    Json::CharReaderBuilder builder;
    Json::String errors;
    builder["collectComments"]     = false;
    builder["allowComments"]       = true;
    builder["allowTrailingCommas"] = true;
    builder["allowSingleQuotes"]   = true;
    builder["failIfExtra"]         = true;
    builder["rejectDupKeys"]       = true;
    builder["allowSpecialFloats"]  = true;
    builder["skipBom"]             = false;
    std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
    if (!reader->parse(jsonObject.data(), jsonObject.data() + jsonObject.size(), &root, &errors))
    {
        return std::nullopt;
    }

    return root;
}

} // namespace

namespace DcgmNs::NvBandwidth
{

std::optional<Json::Value> ParseJsonFromOutput(std::string_view output)
{
    size_t searchStart = 0;
    while (true)
    {
        auto candidate = ExtractCompleteJsonObject(output, searchStart);
        if (!candidate.has_value())
        {
            return std::nullopt;
        }

        auto root = ParseJsonObject(candidate->object);
        if (!root.has_value())
        {
            searchStart = candidate->startPosition + 1;
            continue;
        }

        if (root->isObject() && root->isMember("nvbandwidth"))
        {
            return root;
        }

        searchStart = candidate->endPosition;
    }
}

} // namespace DcgmNs::NvBandwidth
