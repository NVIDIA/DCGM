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

#include <ScopedAtomic.hpp>

#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <stdexcept>
#include <type_traits>

using Int32AtomicGuard = ScopedAtomic<std::atomic_int32_t>;

TEST_CASE("ScopedAtomic is neither copyable nor movable")
{
    STATIC_REQUIRE(!std::is_copy_constructible_v<Int32AtomicGuard>);
    STATIC_REQUIRE(!std::is_copy_assignable_v<Int32AtomicGuard>);
    STATIC_REQUIRE(!std::is_move_constructible_v<Int32AtomicGuard>);
    STATIC_REQUIRE(!std::is_move_assignable_v<Int32AtomicGuard>);
}

TEST_CASE("ScopedAtomic stores a temporary value and restores the previous value")
{
    std::atomic_int32_t value { 7 };
    {
        ScopedAtomic guard(value, 42);
        CHECK(value.load() == 42);
    }
    CHECK(value.load() == 7);
}

TEST_CASE("ScopedAtomic restores the previous value after an in-scope mutation")
{
    std::atomic_int32_t value { 7 };
    {
        ScopedAtomic guard(value, 42);
        CHECK(value.load() == 42);
        value.store(5);
        CHECK(value.load() == 5);
    }
    CHECK(value.load() == 7);
}

TEST_CASE("ScopedAtomic restores the previous value when an exception leaves the guarded scope")
{
    std::atomic_int32_t value { 7 };
    try
    {
        ScopedAtomic guard(value, 42);
        CHECK(value.load() == 42);
        throw std::runtime_error("unwind");
    }
    catch (std::runtime_error const &)
    {}
    CHECK(value.load() == 7);
}
