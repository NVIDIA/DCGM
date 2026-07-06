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

#include <DcgmStatus.h>

#include <catch2/catch_all.hpp>

namespace
{
void CheckStatus(dcgmErrorInfo_t const &status, unsigned int gpuId, short fieldId, int errorCode)
{
    CHECK(status.gpuId == gpuId);
    CHECK(status.fieldId == fieldId);
    CHECK(status.status == errorCode);
}
} // namespace

TEST_CASE("DcgmStatus starts empty and rejects invalid dequeue requests")
{
    DcgmStatus status;

    GIVEN("a new status queue")
    {
        THEN("it has no errors")
        {
            CHECK(status.IsEmpty());
            CHECK(status.GetNumErrors() == 0);
        }

        THEN("dequeue fails without an output pointer")
        {
            CHECK(status.Dequeue(nullptr) == -1);
            CHECK(status.IsEmpty());
            CHECK(status.GetNumErrors() == 0);
        }

        THEN("dequeue fails when the queue is empty")
        {
            dcgmErrorInfo_t popped {};
            CHECK(status.Dequeue(&popped) == -1);
            CHECK(status.IsEmpty());
            CHECK(status.GetNumErrors() == 0);
        }
    }
}

TEST_CASE("DcgmStatus stores and removes errors in FIFO order")
{
    DcgmStatus status;

    GIVEN("multiple enqueued errors")
    {
        REQUIRE(status.Enqueue(0, 10, DCGM_ST_BADPARAM) == 0);
        REQUIRE(status.Enqueue(1, 20, DCGM_ST_NOT_SUPPORTED) == 0);

        THEN("the queue reports the number of pending errors")
        {
            CHECK_FALSE(status.IsEmpty());
            CHECK(status.GetNumErrors() == 2);
        }

        THEN("dequeue returns the first error first")
        {
            dcgmErrorInfo_t popped {};
            REQUIRE(status.Dequeue(&popped) == 0);
            CheckStatus(popped, 0, 10, DCGM_ST_BADPARAM);
            CHECK(status.GetNumErrors() == 1);

            REQUIRE(status.Dequeue(&popped) == 0);
            CheckStatus(popped, 1, 20, DCGM_ST_NOT_SUPPORTED);
            CHECK(status.IsEmpty());
        }
    }
}

TEST_CASE("DcgmStatus RemoveAll clears pending errors")
{
    DcgmStatus status;

    GIVEN("a queue with pending errors")
    {
        REQUIRE(status.Enqueue(7, 30, DCGM_ST_GENERIC_ERROR) == 0);
        REQUIRE(status.Enqueue(8, 40, DCGM_ST_TIMEOUT) == 0);

        WHEN("all errors are removed")
        {
            CHECK(status.RemoveAll() == 0);

            THEN("the queue is empty and can be reused")
            {
                CHECK(status.IsEmpty());
                CHECK(status.GetNumErrors() == 0);

                REQUIRE(status.Enqueue(9, 50, DCGM_ST_OK) == 0);

                dcgmErrorInfo_t popped {};
                REQUIRE(status.Dequeue(&popped) == 0);
                CheckStatus(popped, 9, 50, DCGM_ST_OK);
                CHECK(status.IsEmpty());
            }
        }
    }
}
