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

#include <concepts>
#include <dcgm_nvml.h>
#include <functional>
#include <string>
#include <utility>

namespace Nvml
{

enum class ApiResult
{
    SUCCESS,
    CALL_NOT_FOUND
};

struct Exception
{
    nvmlReturn_t return_code;
    std::string function_name;
    size_t function_addr;
};

template <typename F, typename Result, typename... Args>
concept InvokableWithResult = requires(F &&f, Args &&...args) {
    { std::invoke(std::forward<F>(f), std::forward<Args>(args)...) } -> std::same_as<Result>;
};

template <auto F, typename... Args>
concept Function = InvokableWithResult<decltype(F), nvmlReturn_t, Args...>;

template <auto F, typename... Args>
Exception Call(Args &&...args)
    requires Function<F, Args...>
{
    Exception ex;
    ex.return_code   = std::invoke(F, std::forward<Args>(args)...);
    ex.function_name = std::string(__PRETTY_FUNCTION__);
    ex.function_addr = reinterpret_cast<size_t>(F);

    return ex;
}

template <auto F, typename... Args>
void CallOrFail(Args &&...args)
{
    Exception ex = Call<F>(std::forward<Args>(args)...);
    if (ex.return_code != NVML_SUCCESS)
    {
        throw ex;
    }
}

template <auto F, typename... Args>
ApiResult CallIfExistsOrFail(Args &&...args)
{
    ApiResult result = ApiResult::SUCCESS;

    Exception ex = Call<F>(std::forward<Args>(args)...);
    if (ex.return_code != NVML_SUCCESS)
    {
        if (ex.return_code == NVML_ERROR_FUNCTION_NOT_FOUND)
        {
            result = ApiResult::CALL_NOT_FOUND;
        }
        else
        {
            throw ex;
        }
    }

    return result;
}

} //namespace Nvml
