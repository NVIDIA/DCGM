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
#ifndef DCGMSTRINGCONVERSIONS_H_
#define DCGMSTRINGCONVERSIONS_H_

#include <sstream>
#include <string>

/**
 * Convert a string to a type, T.  The type, T, must support the streaming operatior ">>".
 * If the conversion succeeds then \ref success will be set to true, otherwise false.
 * If the conversion does not succeed the return value is undefined.
 */
template <typename T>
T strTo(std::string const &str, bool *success)
{
    char extraChar;
    T val;
    std::stringstream ss(str);
    ss >> val;

    if (success != nullptr)
    {
        *success = !ss.fail() && !ss.get(extraChar);
    }

    return val;
}

/**
 * Convert a string to a type, T.  If the conversion fails, the return value
 * is undefined so only use this function if you know that the string will
 * definitely convert to the type you want.  To know if the conversion
 * succeeded or not use "strTo(std::string str, bool *success)" instead.
 */
template <typename T>
T strTo(std::string const &str)
{
    return strTo<T>(str, nullptr);
}

/**
 * Get the string representation of a value.
 * The type of the value must support the streaming operator "<<".
 */
template <typename T>
std::string toStr(T const &value)
{
    std::stringstream ss;
    ss << value;
    return ss.str();
}

#endif /* DCGMSTRINGCONVERSIONS_H_ */
