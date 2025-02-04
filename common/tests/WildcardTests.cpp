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
#include "EntityListHelpers.h"
#include <MigIdParser.hpp>
#include <catch2/catch_all.hpp>

#include <tuple>
#include <unordered_set>


TEST_CASE("DCGMI: Wildcard entity parsing")
{
    using namespace DcgmNs;

    /*
     * Reproducing topology with 2 GPUs, 2 GPU_I in each and 2 GPU_CI in each GPU_I
     */
    EntityMap emap {
        { ParsedGpu { "0" }, { DCGM_FE_GPU, 0 } },
        { ParsedGpu { "1" }, { DCGM_FE_GPU, 1 } },
        { ParsedGpuI { "0", 0U }, { DCGM_FE_GPU_I, 0 } },
        { ParsedGpuI { "0", 1U }, { DCGM_FE_GPU_I, 1 } },
        { ParsedGpuI { "1", 0U }, { DCGM_FE_GPU_I, 2 } },
        { ParsedGpuI { "1", 1U }, { DCGM_FE_GPU_I, 3 } },

        { ParsedGpuCi { "0", 0U, 0U }, { DCGM_FE_GPU_CI, 0 } },
        { ParsedGpuCi { "0", 0U, 1U }, { DCGM_FE_GPU_CI, 1 } },
        { ParsedGpuCi { "0", 1U, 0U }, { DCGM_FE_GPU_CI, 2 } },
        { ParsedGpuCi { "0", 1U, 1U }, { DCGM_FE_GPU_CI, 3 } },
        { ParsedGpuCi { "1", 0U, 0U }, { DCGM_FE_GPU_CI, 4 } },
        { ParsedGpuCi { "1", 0U, 1U }, { DCGM_FE_GPU_CI, 5 } },
        { ParsedGpuCi { "1", 1U, 0U }, { DCGM_FE_GPU_CI, 6 } },
        { ParsedGpuCi { "1", 1U, 1U }, { DCGM_FE_GPU_CI, 7 } },
    };

    detail::EntityGroupContainer result;
    auto ret = detail::HandleWildcardResult::Error;

    {
        INFO("GPU_CI: Wildcarded GPU");
        ret = detail::HandleWildcard(ParsedGpuCi { Wildcarded {}, 1U, 1U }, emap, result);
        REQUIRE(ret == detail::HandleWildcardResult::Handled);
        REQUIRE(result.size() == 2);
        REQUIRE(result.count({ DCGM_FE_GPU_CI, 3 }) != 0);
        REQUIRE(result.count({ DCGM_FE_GPU_CI, 7 }) != 0);
    }

    result.clear();
    {
        INFO("GPU_CI: Wildcarded GPU_I and GPU_CI");
        ret = detail::HandleWildcard(ParsedGpuCi { "1", Wildcarded {}, Wildcarded {} }, emap, result);
        REQUIRE(ret == detail::HandleWildcardResult::Handled);
        REQUIRE(result.size() == 4);
        REQUIRE(result.count({ DCGM_FE_GPU_CI, 4 }) != 0);
        REQUIRE(result.count({ DCGM_FE_GPU_CI, 5 }) != 0);
        REQUIRE(result.count({ DCGM_FE_GPU_CI, 6 }) != 0);
        REQUIRE(result.count({ DCGM_FE_GPU_CI, 7 }) != 0);
    }

    result.clear();
    {
        INFO("GPU_CI: No wildcards");
        ret = detail::HandleWildcard(ParsedGpuCi { "1", 1U, 1U }, emap, result);
        REQUIRE(ret == detail::HandleWildcardResult::Unhandled);
        REQUIRE(result.empty());
    }

    {
        INFO("GPU_I: Wildcarded GPU");
        ret = detail::HandleWildcard(ParsedGpuI { Wildcarded {}, 1U }, emap, result);
        REQUIRE(ret == detail::HandleWildcardResult::Handled);
        REQUIRE(result.size() == 2);
        REQUIRE(result.count({ DCGM_FE_GPU_I, 1 }) != 0);
        REQUIRE(result.count({ DCGM_FE_GPU_I, 3 }) != 0);
    }

    result.clear();
    {
        INFO("GPU_I: Wildcarded GPU_I");
        ret = detail::HandleWildcard(ParsedGpuI { "1", Wildcarded {} }, emap, result);
        REQUIRE(ret == detail::HandleWildcardResult::Handled);
        REQUIRE(result.size() == 2);
        REQUIRE(result.count({ DCGM_FE_GPU_I, 2 }) != 0);
        REQUIRE(result.count({ DCGM_FE_GPU_I, 3 }) != 0);
    }

    result.clear();
    {
        INFO("GPU_I: No wildcards");
        ret = detail::HandleWildcard(ParsedGpuI { "1", 1U }, emap, result);
        REQUIRE(ret == detail::HandleWildcardResult::Unhandled);
        REQUIRE(result.empty());
    }

    result.clear();
    {
        INFO("GPU: Wildcarded");
        ret = detail::HandleWildcard(ParsedGpu { Wildcarded {} }, emap, result);
        REQUIRE(ret == detail::HandleWildcardResult::Handled);
        REQUIRE(result.size() == 2);
        REQUIRE(result.count({ DCGM_FE_GPU, 0 }) != 0);
        REQUIRE(result.count({ DCGM_FE_GPU, 1 }) != 0);
    }

    result.clear();
    {
        INFO("GPU: No wildcards");
        ret = detail::HandleWildcard(ParsedGpu { "1" }, emap, result);
        REQUIRE(ret == detail::HandleWildcardResult::Unhandled);
        REQUIRE(result.empty());
    }

    {
        INFO("Error");
        ret = detail::HandleWildcard(ParsedUnknown {}, emap, result);
        REQUIRE(ret == detail::HandleWildcardResult::Error);
    }
}

TEST_CASE("DCGMI: Wildcard comparison")
{
    using namespace DcgmNs;

    auto const left  = Wildcard<std::string> { "hello" };
    auto const right = Wildcard<std::string> { Wildcarded {} };
    auto const bad   = Wildcard<std::string> { NotInitialized {} };

    auto const result1 = left.Compare(right);
    REQUIRE(result1 == true);

    auto const result2 = right.Compare(left);
    REQUIRE(result2 == true);

    auto const result3 = std::tuple(compare(left, right), compare(right, left)) == std::tuple(true, true);
    REQUIRE(result3 == true);

    auto const result4 = left == Wildcarded {};
    REQUIRE(result4 == true);

    auto const result5 = compare(left, bad);
    REQUIRE(result5 == false);

    auto const result6 = left == NotInitialized {};
    REQUIRE(result6 == false);

    auto const result7 = compare(bad, bad);
    REQUIRE(result7 == true);
}
