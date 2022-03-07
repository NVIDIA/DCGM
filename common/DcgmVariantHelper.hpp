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

#include <utility>


namespace DcgmNs
{
/** Helper class to provide better std::visitor usability
 * @code{.cpp}
 *      std::visit(overloaded {
 *          [](Type1 const&){},
 *          [](Type2 const&){} },
 *      varObj);
 * @endcode
 */
template <typename... Ts>
struct overloaded : Ts...
{
    constexpr explicit overloaded(Ts &&...ts)
        : Ts(std::forward<Ts>(ts))...
    {}
    using Ts::operator()...;
};

template <typename... Ts>
overloaded(Ts...) -> overloaded<Ts...>;
} // namespace DcgmNs