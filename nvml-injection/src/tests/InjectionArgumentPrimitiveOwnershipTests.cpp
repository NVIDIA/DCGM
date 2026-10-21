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

#include <InjectionArgument.h>

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstdlib>
#include <cstring>
#include <string>

namespace
{
template <typename T>
void CheckScalarSetValueFrom(T sourceValue, T differentValue)
{
    GIVEN("a scalar InjectionArgument and a writable scalar target")
    {
        T targetValue {};
        InjectionArgument source(sourceValue);
        InjectionArgument target(&targetValue);

        WHEN("the source value is copied into the pointer target")
        {
            auto const ret = target.SetValueFrom(source);

            THEN("the target storage receives the source value")
            {
                REQUIRE(ret == NVML_SUCCESS);
                CHECK(targetValue == sourceValue);
                CHECK(InjectionArgument(targetValue).Compare(source) == 0);
                CHECK(InjectionArgument(differentValue).Compare(source) != 0);
            }
        }
    }
}

template <typename T>
void CheckArraySetValueFrom(std::array<T, 3> sourceValues, std::array<T, 3> differentValues)
{
    GIVEN("array-backed InjectionArguments with caller-owned storage")
    {
        std::array<T, 3> targetValues {};
        InjectionArgument source(sourceValues.data(), static_cast<unsigned int>(sourceValues.size()));
        InjectionArgument target(targetValues.data(), static_cast<unsigned int>(targetValues.size()));

        WHEN("the array source is copied into the target array")
        {
            auto const ret = target.SetValueFrom(source);

            THEN("all declared elements are copied and comparison observes payload changes")
            {
                REQUIRE(ret == NVML_SUCCESS);
                CHECK(targetValues == sourceValues);
                CHECK(target.Compare(source) == 0);
                CHECK(InjectionArgument(differentValues.data(), static_cast<unsigned int>(differentValues.size()))
                          .Compare(source)
                      != 0);
            }
        }
    }
}
} //namespace

SCENARIO("InjectionArgument: copies primitive scalar values through pointer targets")
{
    CheckScalarSetValueFrom<char>('a', 'b');
    CheckScalarSetValueFrom<short>(-7, 8);
    CheckScalarSetValueFrom<int>(17, 19);
    CheckScalarSetValueFrom<long>(23L, 29L);
    CheckScalarSetValueFrom<long long>(31LL, 37LL);
    CheckScalarSetValueFrom<unsigned char>(3U, 4U);
    CheckScalarSetValueFrom<unsigned short>(5U, 6U);
    CheckScalarSetValueFrom<unsigned int>(7U, 8U);
    CheckScalarSetValueFrom<unsigned long>(11UL, 12UL);
    CheckScalarSetValueFrom<unsigned long long>(13ULL, 14ULL);
}

SCENARIO("InjectionArgument: copies primitive arrays through caller-owned buffers")
{
    CheckArraySetValueFrom<int>({ 1, 2, 3 }, { 1, 2, 4 });
    CheckArraySetValueFrom<unsigned int>({ 5U, 6U, 7U }, { 5U, 6U, 8U });
    CheckArraySetValueFrom<unsigned long long>({ 9ULL, 10ULL, 11ULL }, { 9ULL, 10ULL, 12ULL });
}

SCENARIO("InjectionArgument: ownership keeps heap-backed primitive values independent")
{
    GIVEN("an owning heap-backed integer argument")
    {
        auto *raw = static_cast<unsigned int *>(std::malloc(3 * sizeof(unsigned int)));
        REQUIRE(raw != nullptr);
        raw[0] = 10U;
        raw[1] = 20U;
        raw[2] = 30U;
        InjectionArgument source(raw, 3U, true);

        WHEN("the owning argument is copied and assigned")
        {
            InjectionArgument copied(source);
            InjectionArgument assigned;
            assigned = source;
            raw      = nullptr;

            THEN("both copies preserve the original payload without sharing caller storage")
            {
                unsigned int expectedValues[] = { 10U, 20U, 30U };
                InjectionArgument expected(expectedValues, 3U);
                CHECK(copied.Compare(expected) == 0);
                CHECK(assigned.Compare(expected) == 0);
            }
        }
    }
}

SCENARIO("InjectionArgument: string views expose null and non-null string contracts")
{
    GIVEN("mutable, const, owned, and null strings")
    {
        char mutableText[]        = "mutable";
        char *nullMutableText     = nullptr;
        char const *constText     = "const";
        char const *nullConstText = nullptr;
        InjectionArgument mutableString(mutableText);
        InjectionArgument nullMutableString(nullMutableText);
        InjectionArgument constString(constText);
        InjectionArgument nullConstString(nullConstText);
        InjectionArgument ownedString(std::string("owned"));

        WHEN("the values are read through public accessors")
        {
            THEN("null strings become empty strings and non-null storage remains observable")
            {
                CHECK(nullMutableString.AsString().empty());
                CHECK(nullConstString.AsString().empty());
                CHECK(std::string(constString.AsConstStr()) == "const");
                CHECK(mutableString.AsStr() == mutableText);
                CHECK(mutableString.Compare(ownedString) != 0);
                CHECK(ownedString.Compare(constString) != 0);
            }
        }
    }
}
