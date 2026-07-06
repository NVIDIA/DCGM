/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "DistributedCudaContext.h"
#include "PhysicalGpu.h"

#include <EnvVarGuard.hpp>
#include <dcgm_fields.h>
#include <dcgm_structs.h>

#include <cstdlib>
#include <fcntl.h>
#include <memory>
#include <string>
#include <unistd.h>

using namespace DcgmNs::ProfTester;
using namespace Catch::Matchers;

namespace DcgmNs::ProfTester
{

// Function to configure PhysicalGpu stub behavior for testing
void SetPhysicalGpuTestConfig(bool isMIG, std::string const &busId = "0000:01:00.0");
void SetPhysicalGpuSyncMode(bool isSynchronous);

/**
 * Friend test class for testing DistributedCudaContext private methods
 */
class DistributedCudaContextTests
{
public:
    DistributedCudaContextTests(DistributedCudaContext &context)
        : m_context(context)
    {}

    /**
     * Test the HandleCudaVisibleDevices helper method
     */
    dcgmReturn_t HandleCudaVisibleDevices()
    {
        return m_context.HandleCudaVisibleDevices();
    }

    int ReadLnCheck(unsigned int &activity, bool &earlyQuit)
    {
        return m_context.ReadLnCheck(activity, earlyQuit);
    }

    /**
     * Get the message stream content for verification
     */
    std::string GetMessageString() const
    {
        return m_context.m_message.str();
    }

    /**
     * Get the error stream content for verification
     */
    std::string GetErrorString() const
    {
        return m_context.m_error.str();
    }

    /**
     * Set the test field ID for testing different behaviors
     */
    void SetTestFieldId(unsigned int fieldId)
    {
        m_context.m_testFieldId = fieldId;
    }

    void SetInputFd(int fd)
    {
        m_context.m_inFd = fd;
    }

    void SetOutputFd(int fd)
    {
        m_context.m_outFd = fd;
    }

    void MarkInitialized()
    {
        m_context.m_isInitialized = true;
    }

    void SetStreams(std::string const &input, std::string const &message, std::string const &error)
    {
        m_context.m_input << input;
        m_context.m_message << message;
        m_context.m_error << error;
    }

private:
    DistributedCudaContext &m_context;
};

class ScopedFd
{
public:
    explicit ScopedFd(int fd)
        : m_fd(fd)
    {}

    ScopedFd(ScopedFd const &)            = delete;
    ScopedFd(ScopedFd &&)                 = delete;
    ScopedFd &operator=(ScopedFd const &) = delete;
    ScopedFd &operator=(ScopedFd &&)      = delete;

    ~ScopedFd()
    {
        if (m_fd >= 0)
        {
            close(m_fd);
        }
    }

    [[nodiscard]] int Close()
    {
        if (m_fd < 0)
        {
            return 0;
        }

        int const ret = close(m_fd);
        m_fd          = -1;
        return ret;
    }

private:
    int m_fd;
};

class InputFdGuard
{
public:
    InputFdGuard(DistributedCudaContextTests &tester, int fd)
        : m_tester(tester)
        , m_fd(fd)
    {
        m_tester.SetInputFd(m_fd);
    }

    InputFdGuard(InputFdGuard const &)            = delete;
    InputFdGuard(InputFdGuard &&)                 = delete;
    InputFdGuard &operator=(InputFdGuard const &) = delete;
    InputFdGuard &operator=(InputFdGuard &&)      = delete;

    ~InputFdGuard()
    {
        m_tester.SetInputFd(-1);
        if (m_fd >= 0)
        {
            close(m_fd);
        }
    }

private:
    DistributedCudaContextTests &m_tester;
    int m_fd;
};

class PhysicalGpuSyncModeGuard
{
public:
    PhysicalGpuSyncModeGuard() = default;

    PhysicalGpuSyncModeGuard(PhysicalGpuSyncModeGuard const &)            = delete;
    PhysicalGpuSyncModeGuard(PhysicalGpuSyncModeGuard &&)                 = delete;
    PhysicalGpuSyncModeGuard &operator=(PhysicalGpuSyncModeGuard const &) = delete;
    PhysicalGpuSyncModeGuard &operator=(PhysicalGpuSyncModeGuard &&)      = delete;

    ~PhysicalGpuSyncModeGuard()
    {
        SetPhysicalGpuSyncMode(false);
    }
};

DistributedCudaContext MakeContext(std::string const &device = "0", bool isSynchronous = false)
{
    SetPhysicalGpuTestConfig(false);
    SetPhysicalGpuSyncMode(isSynchronous);
    auto mockGpu                 = std::make_shared<PhysicalGpu>(nullptr, 0, Arguments_t::Parameters {});
    auto entities                = std::make_shared<std::map<dcgm_field_entity_group_t, dcgm_field_eid_t>>();
    (*entities)[DCGM_FE_GPU]     = 0;
    dcgmGroupEntityPair_t entity = { DCGM_FE_GPU, 0 };

    return DistributedCudaContext(mockGpu, entities, entity, device);
}

TEST_CASE("DistributedCudaContext::HandleCudaVisibleDevices")
{
    DcgmNs::Tests::EnvVarGuard envGuard("CUDA_VISIBLE_DEVICES");

    SECTION("No existing CUDA_VISIBLE_DEVICES - should set new value")
    {
        REQUIRE(envGuard.Unset() == 0);

        SetPhysicalGpuTestConfig(false); // Configure stub for non-MIG
        auto mockGpu                 = std::make_shared<PhysicalGpu>(nullptr, 0, Arguments_t::Parameters {});
        auto entities                = std::make_shared<std::map<dcgm_field_entity_group_t, dcgm_field_eid_t>>();
        (*entities)[DCGM_FE_GPU]     = 0;
        dcgmGroupEntityPair_t entity = { DCGM_FE_GPU, 0 };

        DistributedCudaContext context(mockGpu, entities, entity, "1");
        DistributedCudaContextTests tester(context);
        tester.SetTestFieldId(DCGM_FI_PROF_SM_UTIL_RATIO);

        dcgmReturn_t result = tester.HandleCudaVisibleDevices();

        REQUIRE(result == DCGM_ST_OK);
        REQUIRE(std::string(getenv("CUDA_VISIBLE_DEVICES") ? getenv("CUDA_VISIBLE_DEVICES") : "") == "1");
        REQUIRE_THAT(tester.GetMessageString(), ContainsSubstring("Set CUDA_VISIBLE_DEVICES to 1"));
    }

    SECTION("Empty existing CUDA_VISIBLE_DEVICES - should set new value")
    {
        REQUIRE(envGuard.Set("") == 0);

        SetPhysicalGpuTestConfig(false); // Configure stub for non-MIG
        auto mockGpu                 = std::make_shared<PhysicalGpu>(nullptr, 0, Arguments_t::Parameters {});
        auto entities                = std::make_shared<std::map<dcgm_field_entity_group_t, dcgm_field_eid_t>>();
        (*entities)[DCGM_FE_GPU]     = 0;
        dcgmGroupEntityPair_t entity = { DCGM_FE_GPU, 0 };

        DistributedCudaContext context(mockGpu, entities, entity, "1");
        DistributedCudaContextTests tester(context);
        tester.SetTestFieldId(DCGM_FI_PROF_SM_UTIL_RATIO);

        dcgmReturn_t result = tester.HandleCudaVisibleDevices();

        REQUIRE(result == DCGM_ST_OK);
        REQUIRE(std::string(getenv("CUDA_VISIBLE_DEVICES") ? getenv("CUDA_VISIBLE_DEVICES") : "") == "1");
        REQUIRE_THAT(tester.GetMessageString(), ContainsSubstring("Set CUDA_VISIBLE_DEVICES to 1"));
    }

    SECTION("Whitespace-only CUDA_VISIBLE_DEVICES - should respect existing value")
    {
        REQUIRE(envGuard.Set("   ") == 0);

        SetPhysicalGpuTestConfig(false); // Configure stub for non-MIG
        auto mockGpu                 = std::make_shared<PhysicalGpu>(nullptr, 0, Arguments_t::Parameters {});
        auto entities                = std::make_shared<std::map<dcgm_field_entity_group_t, dcgm_field_eid_t>>();
        (*entities)[DCGM_FE_GPU]     = 0;
        dcgmGroupEntityPair_t entity = { DCGM_FE_GPU, 0 };

        DistributedCudaContext context(mockGpu, entities, entity, "1");
        DistributedCudaContextTests tester(context);
        tester.SetTestFieldId(DCGM_FI_PROF_SM_UTIL_RATIO);

        dcgmReturn_t result = tester.HandleCudaVisibleDevices();

        REQUIRE(result == DCGM_ST_OK);
        REQUIRE(std::string(getenv("CUDA_VISIBLE_DEVICES") ? getenv("CUDA_VISIBLE_DEVICES") : "")
                == "   "); // Should remain unchanged
        REQUIRE_THAT(tester.GetMessageString(), ContainsSubstring("Respecting existing CUDA_VISIBLE_DEVICES:    "));
    }

    SECTION("MIG GPU - should always override existing CUDA_VISIBLE_DEVICES")
    {
        REQUIRE(envGuard.Set("0,1") == 0);

        SetPhysicalGpuTestConfig(true); // Configure stub for MIG
        auto mockMigGpu              = std::make_shared<PhysicalGpu>(nullptr, 0, Arguments_t::Parameters {});
        auto entities                = std::make_shared<std::map<dcgm_field_entity_group_t, dcgm_field_eid_t>>();
        (*entities)[DCGM_FE_GPU]     = 0;
        dcgmGroupEntityPair_t entity = { DCGM_FE_GPU, 0 };

        DistributedCudaContext context(mockMigGpu, entities, entity, "MIG-12345678-1234-1234-1234-123456789abc");
        DistributedCudaContextTests tester(context);
        tester.SetTestFieldId(DCGM_FI_PROF_SM_UTIL_RATIO);

        dcgmReturn_t result = tester.HandleCudaVisibleDevices();

        REQUIRE(result == DCGM_ST_OK);
        REQUIRE(std::string(getenv("CUDA_VISIBLE_DEVICES") ? getenv("CUDA_VISIBLE_DEVICES") : "")
                == "MIG-12345678-1234-1234-1234-123456789abc");
        REQUIRE_THAT(tester.GetMessageString(), ContainsSubstring("Overriding CUDA_VISIBLE_DEVICES"));
        REQUIRE_THAT(tester.GetMessageString(), ContainsSubstring("for MIG partition"));
    }

    SECTION("NvLink tests - should override existing CUDA_VISIBLE_DEVICES")
    {
        REQUIRE(envGuard.Set("1") == 0);

        SetPhysicalGpuTestConfig(false); // Configure stub for non-MIG
        auto mockGpu                 = std::make_shared<PhysicalGpu>(nullptr, 0, Arguments_t::Parameters {});
        auto entities                = std::make_shared<std::map<dcgm_field_entity_group_t, dcgm_field_eid_t>>();
        (*entities)[DCGM_FE_GPU]     = 0;
        dcgmGroupEntityPair_t entity = { DCGM_FE_GPU, 0 };

        DistributedCudaContext context(mockGpu, entities, entity, "0,1");
        DistributedCudaContextTests tester(context);
        tester.SetTestFieldId(DCGM_FI_PROF_NVLINK_RX_BYTES);

        dcgmReturn_t result = tester.HandleCudaVisibleDevices();

        REQUIRE(result == DCGM_ST_OK);
        REQUIRE(std::string(getenv("CUDA_VISIBLE_DEVICES") ? getenv("CUDA_VISIBLE_DEVICES") : "") == "0,1");
        REQUIRE_THAT(tester.GetMessageString(), ContainsSubstring("Overriding CUDA_VISIBLE_DEVICES"));
        REQUIRE_THAT(tester.GetMessageString(), ContainsSubstring("for NvLink test"));
    }

    SECTION("Other tests - should respect existing CUDA_VISIBLE_DEVICES")
    {
        REQUIRE(envGuard.Set("2") == 0);

        SetPhysicalGpuTestConfig(false); // Configure stub for non-MIG
        auto mockGpu                 = std::make_shared<PhysicalGpu>(nullptr, 0, Arguments_t::Parameters {});
        auto entities                = std::make_shared<std::map<dcgm_field_entity_group_t, dcgm_field_eid_t>>();
        (*entities)[DCGM_FE_GPU]     = 0;
        dcgmGroupEntityPair_t entity = { DCGM_FE_GPU, 0 };

        DistributedCudaContext context(mockGpu, entities, entity, "0,1");
        DistributedCudaContextTests tester(context);
        tester.SetTestFieldId(DCGM_FI_PROF_SM_UTIL_RATIO);

        dcgmReturn_t result = tester.HandleCudaVisibleDevices();

        REQUIRE(result == DCGM_ST_OK);
        REQUIRE(std::string(getenv("CUDA_VISIBLE_DEVICES") ? getenv("CUDA_VISIBLE_DEVICES") : "")
                == "2"); // Should remain unchanged
        REQUIRE_THAT(tester.GetMessageString(), ContainsSubstring("Respecting existing CUDA_VISIBLE_DEVICES: 2"));
    }
}

TEST_CASE("DistributedCudaContext state and pipe helpers")
{
    SECTION("GIVEN an already initialized context WHEN Init is called THEN it is a no-op")
    {
        auto context = MakeContext();
        DistributedCudaContextTests tester(context);
        tester.MarkInitialized();

        CHECK(context.Init(-1, -1) == DCGM_ST_OK);
        CHECK_THAT(tester.GetMessageString(), ContainsSubstring("already initialized"));
    }

    SECTION("GIVEN an invalid input fd WHEN Init is called THEN an error is returned before CUDA setup")
    {
        auto context = MakeContext();

        CHECK(context.Init(-1, -1) == DCGM_ST_GENERIC_ERROR);
    }

    SECTION("GIVEN valid pipe fds WHEN Init runs with CUDA stubs THEN attributes and messages are initialized")
    {
        int fds[2];
        REQUIRE(pipe(fds) == 0);
        DcgmNs::Tests::EnvVarGuard envGuard("CUDA_VISIBLE_DEVICES");
        REQUIRE(envGuard.Unset() == 0);
        auto context = MakeContext("2");

        REQUIRE(context.Init(fds[0], fds[1]) == DCGM_ST_OK);

        auto const &attributes = context.GetAttributes();
        CHECK(attributes.m_computeCapability == 0);
        CHECK(attributes.m_maxMemBandwidth == 0.0);
        CHECK(context.GetReadFD() == fds[0]);
        CHECK(context.Device() == "2");

        DistributedCudaContextTests tester(context);
        CHECK_THAT(tester.GetMessageString(), ContainsSubstring("mapped to cuda device ID"));
        CHECK_THAT(tester.GetMessageString(), ContainsSubstring("Init completed successfully"));
    }

    SECTION("GIVEN a MIG context WHEN Init runs THEN CUDA device zero is used")
    {
        int fds[2];
        REQUIRE(pipe(fds) == 0);
        SetPhysicalGpuTestConfig(true);
        auto mockGpu                 = std::make_shared<PhysicalGpu>(nullptr, 0, Arguments_t::Parameters {});
        auto entities                = std::make_shared<std::map<dcgm_field_entity_group_t, dcgm_field_eid_t>>();
        (*entities)[DCGM_FE_GPU_I]   = 7;
        dcgmGroupEntityPair_t entity = { DCGM_FE_GPU_I, 7 };

        DistributedCudaContext context(mockGpu, entities, entity, "MIG-test-device");

        REQUIRE(context.Init(fds[0], fds[1]) == DCGM_ST_OK);
        CHECK(context.Device() == "MIG-test-device");
        CHECK(context.EntityId().entityGroupId == DCGM_FE_GPU_I);
        CHECK(context.Entities().at(DCGM_FE_GPU_I) == 7);
    }

    SECTION("GIVEN part and status setters WHEN queried THEN state is updated")
    {
        auto context = MakeContext();
        unsigned int part {};
        unsigned int parts {};

        context.SetParts(2, 5);
        context.GetParts(part, parts);
        CHECK(part == 2);
        CHECK(parts == 5);
        CHECK(context.IsFirstTick());

        context.SetFinished();
        CHECK(context.Finished());
        context.SetFailed();
        CHECK(context.Failed());
        context.ClrFailed();
        CHECK_FALSE(context.Failed());
        context.SetTries(7);
        CHECK(context.GetTries() == 7);
        context.SetActivity(42);
        CHECK(context.GetActivity() == 42);
        context.SetValidated(false);
        CHECK_FALSE(context.GetValidated());
    }

    SECTION("GIVEN pipe input WHEN Read consumes requested bytes THEN input stream is populated")
    {
        int fds[2];
        REQUIRE(pipe(fds) == 0);
        auto context = MakeContext();
        DistributedCudaContextTests tester(context);
        [[maybe_unused]] InputFdGuard inputFdGuard(tester, fds[0]);
        ScopedFd writeFd(fds[1]);

        REQUIRE(write(fds[1], "abcdef", 6) == 6);
        REQUIRE(writeFd.Close() == 0);

        REQUIRE(context.Read(6));
        CHECK(context.Input().str() == "abcdef");
    }

    SECTION("GIVEN invalid input fd WHEN Read is called THEN false is returned")
    {
        auto context = MakeContext();

        CHECK_FALSE(context.Read(1));
    }

    SECTION("GIVEN nonblocking pipe WHEN ReadLn has no data THEN zero is returned")
    {
        int fds[2];
        REQUIRE(pipe2(fds, O_NONBLOCK) == 0);
        auto context = MakeContext();
        DistributedCudaContextTests tester(context);
        [[maybe_unused]] InputFdGuard inputFdGuard(tester, fds[0]);
        [[maybe_unused]] ScopedFd writeFd(fds[1]);

        CHECK(context.ReadLn() == 0);
    }

    SECTION("GIVEN line input WHEN ReadLn is called THEN one line is buffered")
    {
        int fds[2];
        REQUIRE(pipe(fds) == 0);
        auto context = MakeContext();
        DistributedCudaContextTests tester(context);
        [[maybe_unused]] InputFdGuard inputFdGuard(tester, fds[0]);
        ScopedFd writeFd(fds[1]);

        REQUIRE(write(fds[1], "A 7\nignored", 11) == 11);
        REQUIRE(writeFd.Close() == 0);

        CHECK(context.ReadLn() > 0);
        CHECK(context.Input().str() == "A 7\n");
    }

    SECTION("GIVEN synchronous mode with no data WHEN ReadLnCheck is called THEN activity is retried")
    {
        int fds[2];
        REQUIRE(pipe2(fds, O_NONBLOCK) == 0);
        auto context = MakeContext("0", true);
        [[maybe_unused]] PhysicalGpuSyncModeGuard syncModeGuard;
        DistributedCudaContextTests tester(context);
        [[maybe_unused]] InputFdGuard inputFdGuard(tester, fds[0]);
        [[maybe_unused]] ScopedFd writeFd(fds[1]);
        unsigned int activity = 5;
        bool earlyQuit        = true;

        CHECK(tester.ReadLnCheck(activity, earlyQuit) == 0);
        CHECK(activity == 4);
        CHECK_FALSE(earlyQuit);
    }

    SECTION("GIVEN synchronous commands WHEN ReadLnCheck is called THEN commands are interpreted")
    {
        auto runCommand = [](std::string const &line, bool expectedEarlyQuit, int expectedStatus) {
            int fds[2];
            REQUIRE(pipe(fds) == 0);
            auto context = MakeContext("0", true);
            [[maybe_unused]] PhysicalGpuSyncModeGuard syncModeGuard;
            DistributedCudaContextTests tester(context);
            [[maybe_unused]] InputFdGuard inputFdGuard(tester, fds[0]);
            ScopedFd writeFd(fds[1]);
            REQUIRE(write(fds[1], line.c_str(), line.size()) == static_cast<ssize_t>(line.size()));
            REQUIRE(writeFd.Close() == 0);

            unsigned int activity = 5;
            bool earlyQuit        = false;

            CHECK(tester.ReadLnCheck(activity, earlyQuit) == expectedStatus);
            CHECK(earlyQuit == expectedEarlyQuit);
            CHECK(activity == 5);
        };

        runCommand("A\n", false, 1);
        runCommand("Q\n", true, 1);
        runCommand("Z\n", false, -1);
    }

    SECTION("GIVEN output fd WHEN Command is called THEN formatted command is written")
    {
        int fds[2];
        REQUIRE(pipe(fds) == 0);
        auto context = MakeContext();
        DistributedCudaContextTests tester(context);
        tester.SetOutputFd(fds[1]);
        std::array<char, 8> buffer {};

        REQUIRE(context.Command("R %d\n", 9) > 0);
        close(fds[1]);
        tester.SetOutputFd(-1);

        REQUIRE(read(fds[0], buffer.data(), buffer.size()) > 0);
        CHECK(std::string(buffer.data()) == "R 9\n");
        close(fds[0]);
    }

    SECTION("GIVEN initialized context WHEN Reset keeps groups THEN runtime state and streams are cleared")
    {
        auto context = MakeContext();
        DistributedCudaContextTests tester(context);
        tester.MarkInitialized();
        tester.SetStreams("input", "message", "error");
        context.SetFinished();
        context.SetFailed();
        context.SetActivity(5);
        context.SetValidated(false);

        context.Reset(true);

        CHECK(context.Input().str().empty());
        CHECK(tester.GetMessageString().empty());
        CHECK(tester.GetErrorString().empty());
        CHECK_FALSE(context.Finished());
        CHECK_FALSE(context.Failed());
        CHECK(context.GetActivity() == 0);
        CHECK(context.GetValidated());
    }

    SECTION("GIVEN a populated context WHEN move constructed THEN state moves to the new context")
    {
        auto source = MakeContext("4");
        source.SetParts(3, 9);
        source.SetActivity(11);
        source.SetValidated(false);

        DistributedCudaContext moved(std::move(source));
        unsigned int part {};
        unsigned int parts {};
        moved.GetParts(part, parts);

        CHECK(moved.Device() == "4");
        CHECK(part == 3);
        CHECK(parts == 9);
        CHECK(moved.GetActivity() == 11);
        CHECK_FALSE(moved.GetValidated());
    }

    SECTION("GIVEN two contexts WHEN move assigned THEN destination receives source state")
    {
        auto source      = MakeContext("5");
        auto destination = MakeContext("6");
        source.SetFinished();
        source.SetFailed();
        source.SetActivity(12);

        destination = std::move(source);

        CHECK(destination.Device() == "5");
        CHECK(destination.Finished());
        CHECK(destination.Failed());
        CHECK(destination.GetActivity() == 12);
    }
}

} // namespace DcgmNs::ProfTester
