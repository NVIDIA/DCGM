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

#pragma once

#include <cstring>
#include <memory>

template <typename T>
std::unique_ptr<T> MakeUniqueZero()
{
    auto ptr = std::make_unique<T>();
    std::memset(ptr.get(), 0, sizeof(T));
    return ptr;
}

template <typename T>
std::unique_ptr<T[]> MakeUniqueZero(size_t size)
{
    auto ptr = std::make_unique<T[]>(size);
    std::memset(ptr.get(), 0, size * sizeof(T));
    return ptr;
}