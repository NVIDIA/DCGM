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
#include "DcgmStringHelpers.h"

#include <cstring>
#include <set>
#include <string>

/*****************************************************************************/
void dcgmTokenizeString(const std::string &src, const std::string &delimiter, std::vector<std::string> &tokens)
{
    size_t pos      = 0;
    size_t prev_pos = 0;

    if (src.size() > 0)
    {
        while (pos != std::string::npos)
        {
            std::string token;
            pos = src.find(delimiter, prev_pos);

            if (pos == std::string::npos)
            {
                token = src.substr(prev_pos);
            }
            else
            {
                token    = src.substr(prev_pos, pos - prev_pos);
                prev_pos = pos + delimiter.size();
            }

            tokens.push_back(token);
        }
    }
}

/*****************************************************************************/
std::vector<std::string> dcgmTokenizeString(const std::string &src, const std::string &delimiter)
{
    std::vector<std::string> tokens;

    dcgmTokenizeString(src, delimiter, tokens);

    return tokens;
}


std::string dcgmStrToLower(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
    return s;
}

namespace DcgmNs
{
std::vector<std::string_view> Split(std::string_view value, char const separator)
{
    std::vector<std::string_view> result;
    std::string_view::size_type prevPos = 0;
    std::string_view::size_type curPos  = 0;
    while (std::string_view::npos != (curPos = value.find(separator, prevPos)))
    {
        result.push_back(value.substr(prevPos, curPos - prevPos));
        prevPos = curPos + 1;
    }

    result.push_back(value.substr(prevPos));

    return result;
}

/*****************************************************************************/
dcgmReturn_t ParseRangeString(const std::string &rangeStr, std::vector<unsigned int> &indices)
{
    auto cpuRanges = Split(rangeStr, ',');

    for (auto range : cpuRanges)
    {
        auto rangeTokens = DcgmNs::Split(range, '-');

        if (rangeTokens.size() == 2)
        {
            unsigned int start = 0;
            unsigned int end   = 0;

            if (rangeTokens[0].empty() || rangeTokens[1].empty())
            {
                return DCGM_ST_BADPARAM;
            }
            else if (!isdigit(rangeTokens[0].data()[0]) || !isdigit(rangeTokens[1].data()[0]))
            {
                return DCGM_ST_BADPARAM;
            }

            start = strtoul(rangeTokens[0].data(), nullptr, 10);
            end   = strtoul(rangeTokens[1].data(), nullptr, 10);

            for (unsigned int i = start; i <= end; i++)
            {
                indices.push_back(i);
            }
        }
        else if (rangeTokens.size() == 1)
        {
            if (rangeTokens[0].empty())
            {
                return DCGM_ST_BADPARAM;
            }
            else if (!isdigit(rangeTokens[0].data()[0]))
            {
                return DCGM_ST_BADPARAM;
            }
            indices.push_back(strtoul(rangeTokens[0].data(), nullptr, 10));
        }
        else
        {
            return DCGM_ST_BADPARAM;
        }
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
std::vector<std::string> TokenizeStringQuoted(std::string_view str, char delimiter, std::string_view quotes)
{
    std::vector<std::string> tokens;

    std::set<char> quote { quotes.begin(), quotes.end() };

    bool in = false;
    std::string_view::const_iterator s;
    for (auto it = str.begin(); it < str.end(); ++it)
    {
        if (in)
        {
            // unquoted
            if (quote.find(*s) == quote.end())
            {
                if (*it == delimiter)
                {
                    tokens.emplace_back(s, it);
                    in = false;
                }
            }
            // quoted
            else
            {
                if (*it == *s)
                {
                    ++s;
                    tokens.emplace_back(s, it);
                    in = false;
                }
            }
        }
        else
        {
            if (*it != delimiter)
            {
                s  = it;
                in = true;
            }
        }
    }

    // remaining entry
    if (in)
    {
        tokens.emplace_back(s, str.end());
    }

    return tokens;
}

} // namespace DcgmNs
