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

#pragma once

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <iterator>
#include <string>
#include <vector>

/**
 * Provides helpers for command-line argument assertions.
 */
class CommandLineArgUtils
{
public:
    /**
     * Checks whether a command-line argument vector contains a specific argument.
     *
     * @param args Command-line arguments to inspect.
     * @param value Argument value to find.
     * @return true if value is present, otherwise false.
     */
    static bool ContainsArg(std::vector<std::string> const &args, std::string const &value)
    {
        return std::find(args.begin(), args.end(), value) != args.end();
    }

    /**
     * Returns the argument immediately following a command-line option.
     *
     * @param args Command-line arguments to inspect.
     * @param key Option whose value should be returned.
     * @return The argument immediately following key.
     */
    static std::string ArgAfter(std::vector<std::string> const &args, std::string const &key)
    {
        auto const it = std::find(args.begin(), args.end(), key);
        REQUIRE(it != args.end());
        REQUIRE(std::next(it) != args.end());
        return *std::next(it);
    }
};
