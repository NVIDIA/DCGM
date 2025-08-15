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

#include "FingerprintStore.h"

#include <catch2/catch_all.hpp>
#include <dcgm_fields.h>
#include <filesystem>
#include <fmt/format.h>
#include <future>
#include <latch>
#include <random>
#include <thread>

class FingerprintStoreTest : public FingerprintStore
{
public:
    using FingerprintStore::GenerateUniqueSeed;
    using FingerprintStore::GetProcStatPath;
    using FingerprintStore::m_store;
};

TEST_CASE("FingerprintStore::Compute")
{
    FingerprintStoreTest fps {};

    SECTION("Compute")
    {
        auto ret1 = fps.Compute("");
        REQUIRE(ret1.first == DCGM_ST_NO_DATA);
        REQUIRE(!ret1.second.has_value());

        ret1 = fps.Compute(std::string(255, 'A'));
        REQUIRE(ret1.first == DCGM_ST_OK);
        REQUIRE(ret1.second.has_value());

        // Ensure the fingerprint is different
        auto ret2 = fps.Compute(std::string(255, 'B'));
        REQUIRE(ret2.first == DCGM_ST_OK);
        REQUIRE(ret2.second.has_value());
        REQUIRE(*(ret1.second) != *(ret2.second));

        // Linux 6.8.0. See proc_pid_stat(5) for details.
        std::string_view data
            = "186424 (cat) R 175206 186424 175206 34822 186424 4194304 92 0 0 0 0 0 0 0 20 0 1 0 718148738 8716288 448 18446744073709551615 98843718799360 98843718814593 140731843266560 0 0 0 0 0 0 0 0 0 17 6 0 0 0 0 0 98843718826672 98843718828136 98843746320384 140731843273929 140731843273956 140731843273956 140731843276779 0";
        auto ret3 = fps.Compute(data);
        REQUIRE(ret3.first == DCGM_ST_OK);
        REQUIRE(ret3.second.has_value());
        REQUIRE(*(ret1.second) != *(ret3.second));
        REQUIRE(*(ret2.second) != *(ret3.second));
    }

    SECTION("Edge Cases and Invalid Inputs")
    {
        // Test extremely large input
        auto ret = fps.Compute(std::string(1024 * 1024, 'X')); // 1MB of data
        CHECK(ret.first == DCGM_ST_OK);
        CHECK(ret.second.has_value());

        // Test Unicode input
        std::string unicode
            = "Hello \u4E16\u754C \U0001F30D \u043F\u0440\u0438\u0432\u0435\u0442 \u03B3\u03B5\u03B9\u03B1";
        ret = fps.Compute(unicode);
        CHECK(ret.first == DCGM_ST_OK);
        CHECK(ret.second.has_value());

        // Test non-printable ASCII
        std::string nonPrintable;
        for (unsigned char c = 1; c < 32; ++c)
        {
            nonPrintable.push_back(c);
        }
        ret = fps.Compute(nonPrintable);
        CHECK(ret.first == DCGM_ST_OK);
        CHECK(ret.second.has_value());

        // Test mixed Unicode, ASCII, and control characters
        ret = fps.Compute("Hello\x01\x02\u4E16\u754C\x1F\U0001F30D\x7F");
        CHECK(ret.first == DCGM_ST_OK);
        CHECK(ret.second.has_value());
    }
}

TEST_CASE("FingerprintStore::Operations")
{
    FingerprintStoreTest fps {};

    auto const absentKey = PidTidPair(0, 0);

    auto const testFp  = TaskFingerprint({ 0x123456789abcdef0, 0x123456789abcdef0 });
    auto const testKey = PidTidPair(65535, 32767);

    SECTION("Retrieve")
    {
        // Nothing has been populated in store
        auto [ret, fp] = fps.Retrieve(absentKey);
        REQUIRE(ret == DCGM_ST_NO_DATA);
        REQUIRE(!fp.has_value());

        // Store fingerprint with valid key
        fps.Update(testKey, testFp);

        // Retrieve Valid Key
        std::tie(ret, fp) = fps.Retrieve(testKey);
        REQUIRE(ret == DCGM_ST_OK);
        REQUIRE(fp.has_value());
        REQUIRE(*fp == testFp);
    }

    SECTION("Update")
    {
        auto const altFp = TaskFingerprint({ 0x2468acf013579bdf, 0x2468acf013579bdf });

        // Valid Update
        fps.Update(testKey, testFp);
        REQUIRE(fps.m_store.size() == 1);

        // Valid Update - replace value
        fps.Update(testKey, altFp);
        REQUIRE(fps.m_store.size() == 1);

        // Verify the update
        auto [ret, fp] = fps.Retrieve(testKey);
        REQUIRE(ret == DCGM_ST_OK);
        REQUIRE(fp.has_value());
        REQUIRE(*fp == altFp);
    }

    SECTION("Delete")
    {
        fps.Update(testKey, testFp);
        REQUIRE(fps.m_store.size() == 1);

        // Can't delete invalid key
        auto ret = fps.Delete(absentKey);
        REQUIRE(ret == DCGM_ST_NO_DATA);

        // Delete valid key
        ret = fps.Delete(testKey);
        REQUIRE(ret == DCGM_ST_OK);
        REQUIRE(fps.m_store.size() == 0);

        // No longer deletable
        ret = fps.Delete(testKey);
        REQUIRE(ret == DCGM_ST_NO_DATA);

        // Can't retrieve deleted entry
        auto [ret2, fp] = fps.Retrieve(testKey);
        REQUIRE(ret2 == DCGM_ST_NO_DATA);
        REQUIRE(!fp.has_value());
    }

    SECTION("Edge Cases")
    {
        // Test with maximum possible PID/TID values
        auto maxKey = PidTidPair(std::numeric_limits<pid_t>::max(), std::numeric_limits<pid_t>::max());
        fps.Update(maxKey, testFp);
        auto [ret, fp] = fps.Retrieve(maxKey);
        REQUIRE(ret == DCGM_ST_OK);
        REQUIRE(fp.has_value());
        REQUIRE(*fp == testFp);
    }
}

TEST_CASE("FingerprintStore::Concurrent Operations")
{
    FingerprintStoreTest fps {};
    auto const testKey = PidTidPair(65535, 32767);

    constexpr int NUM_THREADS        = 8;
    constexpr int UPDATES_PER_THREAD = 1000;

    // Synchronization primitives
    std::latch start_latch(NUM_THREADS + 1);
    std::atomic<int> completed_threads = 0;

    // Random number generation for realistic workload
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> pid_dist(1, 1000);

    // Test concurrent updates to same key
    {
        std::vector<std::thread> threads;
        for (int i = 0; i < NUM_THREADS; ++i)
        {
            threads.emplace_back([&, i]() {
                start_latch.arrive_and_wait();

                for (int j = 0; j < UPDATES_PER_THREAD; ++j)
                {
                    auto fp = TaskFingerprint(
                        { static_cast<uint64_t>(i * UPDATES_PER_THREAD + j), static_cast<uint64_t>(j) });
                    fps.Update(testKey, fp);

                    // Occasionally retrieve to mix operations
                    if (j % 100 == 0)
                    {
                        auto [ret, _] = fps.Retrieve(testKey);
                        CHECK(ret == DCGM_ST_OK);
                    }
                }
                completed_threads++;
            });
        }

        start_latch.arrive_and_wait(); // Start all threads simultaneously
        for (auto &t : threads)
        {
            t.join();
        }
        REQUIRE(fps.m_store.size() == 1);
    }

    // Test concurrent updates to different keys with mixed operations
    {
        std::vector<std::future<void>> futures;
        std::atomic<size_t> store_size = 0;

        for (int i = 0; i < NUM_THREADS; ++i)
        {
            futures.push_back(std::async(std::launch::async, [&]() {
                for (int j = 0; j < UPDATES_PER_THREAD; ++j)
                {
                    // Random PID for realistic workload
                    auto key = PidTidPair(pid_dist(gen), j);
                    auto fp  = TaskFingerprint({ static_cast<uint64_t>(j), static_cast<uint64_t>(j) });

                    // Mix different operations
                    switch (j % 3)
                    {
                        case 0:
                            fps.Update(key, fp);
                            store_size++;
                            break;
                        case 1:
                            if (auto [ret, _] = fps.Retrieve(key); ret == DCGM_ST_OK)
                            {
                                fps.Delete(key);
                                store_size--;
                            }
                            break;
                        case 2:
                            fps.Retrieve(key);
                            break;
                    }
                }
            }));
        }

        // Wait for all operations to complete
        for (auto &f : futures)
        {
            f.wait();
        }

        // Verify final state is consistent
        REQUIRE(fps.m_store.size() <= static_cast<size_t>(NUM_THREADS * UPDATES_PER_THREAD));
    }
}

TEST_CASE("FingerprintStore::Comparison")
{
    TaskFingerprint fp1({ 1, 1 });
    TaskFingerprint fp2({ 1, 1 });
    TaskFingerprint fp3({ 1, 2 });
    TaskFingerprint fp4({ 2, 1 });

    CHECK(fp1 == fp2);
    CHECK(fp1 != fp3);
    CHECK(fp1 != fp4);
    CHECK(fp3 != fp4);
}

TEST_CASE("FingerprintStore::Logging Helpers")
{
    auto fp = TaskFingerprint({ 0x00000000abcd1234, 0x000000009876fedc });
    REQUIRE(fmt::format("{}", fp) == "00000000abcd1234000000009876fedc");
}

TEST_CASE("FingerprintStore::GetProcStatPath")
{
    FingerprintStoreTest store;

    SECTION("Without thread ID")
    {
        auto path = store.GetProcStatPath(1234, {});
        REQUIRE(path == "/proc/1234/stat");
    }

    SECTION("With thread ID")
    {
        auto path = store.GetProcStatPath(1234, 5678);
        REQUIRE(path == "/proc/1234/task/5678/stat");
    }
}

TEST_CASE("FingerprintStore::PidTidPair Tests")
{
    PidTidPair pair1(1234, std::nullopt);
    PidTidPair pair2(1234, 5678);
    PidTidPair pair3(4321, std::nullopt);
    PidTidPair pair4(4321, 8765);

    SECTION("Comparisons")
    {
        // Compare pairs with and without tids
        CHECK(pair1 == PidTidPair(1234, std::nullopt)); // Same pid, no tid
        CHECK(pair1 != pair2);                          // Same pid, one with tid
        CHECK(pair2 == PidTidPair(1234, 5678));         // Same pid and tid
        CHECK(pair2 != pair3);                          // Different pid
        CHECK(pair1 != pair3);                          // Different pid, no tid
    }

    SECTION("Hash Avalanche Properties")
    {
        PidTidPairHash hasher;

        // Hash values for pairs
        auto hash1 = hasher(pair1);
        auto hash2 = hasher(pair2);
        auto hash3 = hasher(pair3);
        auto hash4 = hasher(pair4);

        // Check that small changes in input lead to large changes in hash
        CHECK(hash1 != hash2);
        CHECK(hash1 != hash3);
        CHECK(hash1 != hash4);
        CHECK(hash2 != hash3);
        CHECK(hash2 != hash4);
        CHECK(hash3 != hash4);

        // Worst-case scenario: single bit difference
        PidTidPair pair5(1234, 5679); // Single bit difference in tid
        PidTidPair pair6(1235, 5678); // Single bit difference in pid
        PidTidPair pair7(1235, 5679); // Single bit difference in both

        auto hash5 = hasher(pair5);
        auto hash6 = hasher(pair6);
        auto hash7 = hasher(pair7);

        // Ensure single bit changes result in different hashes
        CHECK(hash2 != hash5);
        CHECK(hash2 != hash6);
        CHECK(hash2 != hash7);
        CHECK(hash5 != hash6);
        CHECK(hash5 != hash7);
        CHECK(hash6 != hash7);
    }
}
