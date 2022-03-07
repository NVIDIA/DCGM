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
#include "DcgmStringHelpers.h"

#include <cstring>
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
/*****************************************************************************/
void dcgmStrncpy(char *destination, const char *source, size_t destinationSize)
{
    strncpy(destination, source, destinationSize);
    destination[destinationSize - 1] = '\0';
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

} // namespace DcgmNs
