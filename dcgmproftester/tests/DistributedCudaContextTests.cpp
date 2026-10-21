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
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "DistributedCudaContext.h"
#include <MockCudaWorkerThreadFactory.hpp>
#include <MockPhysicalGpu.hpp>

#include <MockCudaOps.h>
#include <MockDcgmOps.h>
#include <ScopedEnvVars.hpp>
#include <ScopedFd.hpp>
#include <dcgm_fields.h>
#include <dcgm_structs.h>

#include <array>
#include <cerrno>
#include <fcntl.h>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <unistd.h>

using namespace DcgmNs::ProfTester;
using namespace Catch::Matchers;

namespace DcgmNs::ProfTester
{

struct DistributedCudaContextTestFixture
{
    DistributedCudaContextRuntime runtime;
    DcgmNs::MockDcgmOps mockDcgmOps;
    DcgmNs::DcgmOpsAnyPtr dcgmOps { &mockDcgmOps };
    MockCudaOps mockCudaOps;
    MockPhysicalGpu mockPhysicalGpu;
    MockCudaWorkerThreadFactory cudaWorkerFactory;
    DistributedCudaContext *m_context = nullptr;

    DistributedCudaContext MakeContext(std::string const &device = "0", bool isSynchronous = false)
    {
        mockPhysicalGpu.m_isMig         = false;
        mockPhysicalGpu.m_isSynchronous = isSynchronous;
        std::map<dcgm_field_entity_group_t, dcgm_field_eid_t> entities;
        entities[DCGM_FE_GPU]        = 0;
        dcgmGroupEntityPair_t entity = { DCGM_FE_GPU, 0 };

        return DistributedCudaContext(&mockPhysicalGpu,
                                      std::move(entities),
                                      entity,
                                      device,
                                      dcgmOps,
                                      &mockCudaOps,
                                      &runtime,
                                      cudaWorkerFactory.Create());
    }

    void BindContext(DistributedCudaContext &context)
    {
        m_context = &context;
    }

    dcgmReturn_t HandleCudaVisibleDevices()
    {
        return m_context->HandleCudaVisibleDevices();
    }

    int ReadLnCheck(unsigned int &activity, bool &earlyQuit)
    {
        return m_context->ReadLnCheck(activity, earlyQuit);
    }

    int RunTest()
    {
        return m_context->RunTest();
    }

    bool EccAffectsDramBandwidth()
    {
        return m_context->EccAffectsDramBandwidth();
    }

    std::string GetMessageString() const
    {
        return m_context->m_message.str();
    }

    std::string GetErrorString() const
    {
        return m_context->m_error.str();
    }

    void SetTestFieldId(unsigned int fieldId)
    {
        m_context->m_testFieldId = fieldId;
    }

    void SetInputFd(int fd)
    {
        m_context->m_inFd = fd;
    }

    void SetOutputFd(int fd)
    {
        m_context->m_outFd = fd;
    }

    void SetTiming(double duration, double reportInterval, bool targetMaxValue = false)
    {
        m_context->m_duration       = duration;
        m_context->m_reportInterval = reportInterval;
        m_context->m_targetMaxValue = targetMaxValue;
    }

    void SetAttributes(DistributedCudaContext::Attributes const &attributes)
    {
        m_context->m_attributes = attributes;
    }

    void MarkInitialized()
    {
        m_context->m_isInitialized = true;
    }

    void SetStreams(std::string const &input, std::string const &message, std::string const &error)
    {
        m_context->m_input << input;
        m_context->m_message << message;
        m_context->m_error << error;
    }

    DistributedCudaContext::Attributes MakePopulatedAttributes(int major = 9, int minor = 0, int eccSupport = 1)
    {
        DistributedCudaContext::Attributes attributes {};
        attributes.m_maxThreadsPerMultiProcessor = 2048;
        attributes.m_multiProcessorCount         = 16;
        attributes.m_sharedMemPerMultiprocessor  = 65536;
        attributes.m_computeCapabilityMajor      = major;
        attributes.m_computeCapabilityMinor      = minor;
        attributes.m_computeCapability           = static_cast<double>(major) + static_cast<double>(minor) / 10.0;
        attributes.m_memoryBusWidth              = 320;
        attributes.m_maxMemoryClockMhz           = 1215;
        attributes.m_maxMemBandwidth             = 97200000000.0;
        attributes.m_eccSupport                  = eccSupport;
        return attributes;
    }

    std::string ReadAllFromFd(int fd)
    {
        std::string output;
        std::array<char, 256> buffer {};

        for (;;)
        {
            ssize_t const bytes = read(fd, buffer.data(), buffer.size());
            if (bytes > 0)
            {
                output.append(buffer.data(), static_cast<std::size_t>(bytes));
                continue;
            }
            if (bytes < 0 && errno == EINTR)
            {
                continue;
            }
            REQUIRE(bytes == 0);
            break;
        }

        return output;
    }
};

TEST_CASE_METHOD(DistributedCudaContextTestFixture, "DistributedCudaContext::HandleCudaVisibleDevices")
{
    SECTION("No existing CUDA_VISIBLE_DEVICES - should set new value")
    {
        ScopedEnvVars envGuard("CUDA_VISIBLE_DEVICES", std::nullopt);

        MockPhysicalGpu mockGpu;
        std::map<dcgm_field_entity_group_t, dcgm_field_eid_t> entities;
        entities[DCGM_FE_GPU]        = 0;
        dcgmGroupEntityPair_t entity = { DCGM_FE_GPU, 0 };

        DistributedCudaContext context(
            &mockGpu, std::move(entities), entity, "1", dcgmOps, &mockCudaOps, &runtime, cudaWorkerFactory.Create());
        BindContext(context);
        SetTestFieldId(DCGM_FI_PROF_SM_UTIL_RATIO);

        dcgmReturn_t result = HandleCudaVisibleDevices();

        REQUIRE(result == DCGM_ST_OK);
        REQUIRE(ScopedEnvVars::Get("CUDA_VISIBLE_DEVICES").value_or("") == "1");
        REQUIRE_THAT(GetMessageString(), ContainsSubstring("Set CUDA_VISIBLE_DEVICES to 1"));
    }

    SECTION("Empty existing CUDA_VISIBLE_DEVICES - should set new value")
    {
        ScopedEnvVars envGuard("CUDA_VISIBLE_DEVICES", "");

        MockPhysicalGpu mockGpu;
        std::map<dcgm_field_entity_group_t, dcgm_field_eid_t> entities;
        entities[DCGM_FE_GPU]        = 0;
        dcgmGroupEntityPair_t entity = { DCGM_FE_GPU, 0 };

        DistributedCudaContext context(
            &mockGpu, std::move(entities), entity, "1", dcgmOps, &mockCudaOps, &runtime, cudaWorkerFactory.Create());
        BindContext(context);
        SetTestFieldId(DCGM_FI_PROF_SM_UTIL_RATIO);

        dcgmReturn_t result = HandleCudaVisibleDevices();

        REQUIRE(result == DCGM_ST_OK);
        REQUIRE(ScopedEnvVars::Get("CUDA_VISIBLE_DEVICES").value_or("") == "1");
        REQUIRE_THAT(GetMessageString(), ContainsSubstring("Set CUDA_VISIBLE_DEVICES to 1"));
    }

    SECTION("Whitespace-only CUDA_VISIBLE_DEVICES - should respect existing value")
    {
        ScopedEnvVars envGuard("CUDA_VISIBLE_DEVICES", "   ");

        MockPhysicalGpu mockGpu;
        std::map<dcgm_field_entity_group_t, dcgm_field_eid_t> entities;
        entities[DCGM_FE_GPU]        = 0;
        dcgmGroupEntityPair_t entity = { DCGM_FE_GPU, 0 };

        DistributedCudaContext context(
            &mockGpu, std::move(entities), entity, "1", dcgmOps, &mockCudaOps, &runtime, cudaWorkerFactory.Create());
        BindContext(context);
        SetTestFieldId(DCGM_FI_PROF_SM_UTIL_RATIO);

        dcgmReturn_t result = HandleCudaVisibleDevices();

        REQUIRE(result == DCGM_ST_OK);
        REQUIRE(ScopedEnvVars::Get("CUDA_VISIBLE_DEVICES").value_or("") == "   "); // Should remain unchanged
        REQUIRE_THAT(GetMessageString(), ContainsSubstring("Respecting existing CUDA_VISIBLE_DEVICES:    "));
    }

    SECTION("MIG GPU - should always override existing CUDA_VISIBLE_DEVICES")
    {
        ScopedEnvVars envGuard("CUDA_VISIBLE_DEVICES", "0,1");

        MockPhysicalGpu mockMigGpu;
        mockMigGpu.m_isMig = true;
        std::map<dcgm_field_entity_group_t, dcgm_field_eid_t> entities;
        entities[DCGM_FE_GPU]        = 0;
        dcgmGroupEntityPair_t entity = { DCGM_FE_GPU, 0 };

        DistributedCudaContext context(&mockMigGpu,
                                       std::move(entities),
                                       entity,
                                       "MIG-12345678-1234-1234-1234-123456789abc",
                                       dcgmOps,
                                       &mockCudaOps,
                                       &runtime,
                                       cudaWorkerFactory.Create());
        BindContext(context);
        SetTestFieldId(DCGM_FI_PROF_SM_UTIL_RATIO);

        dcgmReturn_t result = HandleCudaVisibleDevices();

        REQUIRE(result == DCGM_ST_OK);
        REQUIRE(ScopedEnvVars::Get("CUDA_VISIBLE_DEVICES").value_or("") == "MIG-12345678-1234-1234-1234-123456789abc");
        REQUIRE_THAT(GetMessageString(), ContainsSubstring("Overriding CUDA_VISIBLE_DEVICES"));
        REQUIRE_THAT(GetMessageString(), ContainsSubstring("for MIG partition"));
    }

    SECTION("NvLink tests - should override existing CUDA_VISIBLE_DEVICES")
    {
        ScopedEnvVars envGuard("CUDA_VISIBLE_DEVICES", "1");

        MockPhysicalGpu mockGpu;
        std::map<dcgm_field_entity_group_t, dcgm_field_eid_t> entities;
        entities[DCGM_FE_GPU]        = 0;
        dcgmGroupEntityPair_t entity = { DCGM_FE_GPU, 0 };

        DistributedCudaContext context(
            &mockGpu, std::move(entities), entity, "0,1", dcgmOps, &mockCudaOps, &runtime, cudaWorkerFactory.Create());
        BindContext(context);
        SetTestFieldId(DCGM_FI_PROF_NVLINK_RX_BYTES);

        dcgmReturn_t result = HandleCudaVisibleDevices();

        REQUIRE(result == DCGM_ST_OK);
        REQUIRE(ScopedEnvVars::Get("CUDA_VISIBLE_DEVICES").value_or("") == "0,1");
        REQUIRE_THAT(GetMessageString(), ContainsSubstring("Overriding CUDA_VISIBLE_DEVICES"));
        REQUIRE_THAT(GetMessageString(), ContainsSubstring("for NvLink test"));
    }

    SECTION("Other tests - should respect existing CUDA_VISIBLE_DEVICES")
    {
        ScopedEnvVars envGuard("CUDA_VISIBLE_DEVICES", "2");

        MockPhysicalGpu mockGpu;
        std::map<dcgm_field_entity_group_t, dcgm_field_eid_t> entities;
        entities[DCGM_FE_GPU]        = 0;
        dcgmGroupEntityPair_t entity = { DCGM_FE_GPU, 0 };

        DistributedCudaContext context(
            &mockGpu, std::move(entities), entity, "0,1", dcgmOps, &mockCudaOps, &runtime, cudaWorkerFactory.Create());
        BindContext(context);
        SetTestFieldId(DCGM_FI_PROF_SM_UTIL_RATIO);

        dcgmReturn_t result = HandleCudaVisibleDevices();

        REQUIRE(result == DCGM_ST_OK);
        REQUIRE(ScopedEnvVars::Get("CUDA_VISIBLE_DEVICES").value_or("") == "2"); // Should remain unchanged
        REQUIRE_THAT(GetMessageString(), ContainsSubstring("Respecting existing CUDA_VISIBLE_DEVICES: 2"));
    }
}

TEST_CASE_METHOD(DistributedCudaContextTestFixture, "DistributedCudaContext state and pipe helpers")
{
    SECTION("GIVEN an already initialized context WHEN Init is called THEN it is a no-op")
    {
        auto context = MakeContext();
        BindContext(context);
        MarkInitialized();

        CHECK(context.Init(-1, -1) == DCGM_ST_OK);
        CHECK_THAT(GetMessageString(), ContainsSubstring("already initialized"));
    }

    SECTION("GIVEN an invalid input fd WHEN Init is called THEN an error is returned before CUDA setup")
    {
        auto context = MakeContext();

        CHECK(context.Init(-1, -1) == DCGM_ST_GENERIC_ERROR);
        CHECK_FALSE(context.IsRunning());
    }

    SECTION("GIVEN valid pipe fds WHEN Init runs with CUDA stubs THEN attributes and messages are initialized")
    {
        int fds[2];
        REQUIRE(pipe(fds) == 0);
        ScopedEnvVars envGuard("CUDA_VISIBLE_DEVICES", std::nullopt);
        auto context = MakeContext("2");

        REQUIRE(context.Init(fds[0], fds[1]) == DCGM_ST_OK);

        auto const &attributes = context.GetAttributes();
        CHECK(attributes.m_computeCapability == 0);
        CHECK(attributes.m_maxMemBandwidth == 0.0);
        CHECK(context.GetReadFD() == fds[0]);
        CHECK(context.Device() == "2");

        BindContext(context);
        CHECK_THAT(GetMessageString(), ContainsSubstring("mapped to cuda device ID"));
        CHECK_THAT(GetMessageString(), ContainsSubstring("Init completed successfully"));
    }

    SECTION("GIVEN a MIG context WHEN Init runs THEN CUDA device zero is used")
    {
        int fds[2];
        REQUIRE(pipe(fds) == 0);
        MockPhysicalGpu mockGpu;
        mockGpu.m_isMig = true;
        std::map<dcgm_field_entity_group_t, dcgm_field_eid_t> entities;
        entities[DCGM_FE_GPU_I]      = 7;
        dcgmGroupEntityPair_t entity = { DCGM_FE_GPU_I, 7 };

        mockCudaOps.cuInitResult = CUDA_SUCCESS;

        DistributedCudaContext context(&mockGpu,
                                       std::move(entities),
                                       entity,
                                       "MIG-test-device",
                                       dcgmOps,
                                       &mockCudaOps,
                                       &runtime,
                                       cudaWorkerFactory.Create());

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
        BindContext(context);
        ScopedFd readFd(fds[0]);
        ScopedFd writeFd(fds[1]);
        SetInputFd(readFd.Get());

        REQUIRE(write(fds[1], "abcdef", 6) == 6);
        REQUIRE(writeFd.Close() == 0);

        REQUIRE(context.Read(6));
        CHECK(context.Input().str() == "abcdef");
        SetInputFd(-1);
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
        BindContext(context);
        ScopedFd readFd(fds[0]);
        [[maybe_unused]] ScopedFd writeFd(fds[1]);
        SetInputFd(readFd.Get());

        CHECK(context.ReadLn() == 0);
        SetInputFd(-1);
    }

    SECTION("GIVEN line input WHEN ReadLn is called THEN one line is buffered")
    {
        int fds[2];
        REQUIRE(pipe(fds) == 0);
        auto context = MakeContext();
        BindContext(context);
        ScopedFd readFd(fds[0]);
        ScopedFd writeFd(fds[1]);
        SetInputFd(readFd.Get());

        REQUIRE(write(fds[1], "A 7\nignored", 11) == 11);
        REQUIRE(writeFd.Close() == 0);

        CHECK(context.ReadLn() > 0);
        CHECK(context.Input().str() == "A 7\n");
        SetInputFd(-1);
    }

    SECTION("GIVEN synchronous mode with no data WHEN ReadLnCheck is called THEN activity is retried")
    {
        int fds[2];
        REQUIRE(pipe2(fds, O_NONBLOCK) == 0);
        auto context = MakeContext("0", true);
        BindContext(context);
        ScopedFd readFd(fds[0]);
        [[maybe_unused]] ScopedFd writeFd(fds[1]);
        SetInputFd(readFd.Get());
        unsigned int activity = 5;
        bool earlyQuit        = true;

        CHECK(ReadLnCheck(activity, earlyQuit) == 0);
        CHECK(activity == 4);
        CHECK_FALSE(earlyQuit);
        SetInputFd(-1);
    }

    SECTION("GIVEN synchronous commands WHEN ReadLnCheck is called THEN commands are interpreted")
    {
        auto runCommand = [this](std::string const &line, bool expectedEarlyQuit, int expectedStatus) {
            int fds[2];
            REQUIRE(pipe(fds) == 0);
            auto context = MakeContext("0", true);
            BindContext(context);
            ScopedFd readFd(fds[0]);
            ScopedFd writeFd(fds[1]);
            SetInputFd(readFd.Get());
            REQUIRE(write(fds[1], line.c_str(), line.size()) == static_cast<ssize_t>(line.size()));
            REQUIRE(writeFd.Close() == 0);

            unsigned int activity = 5;
            bool earlyQuit        = false;

            CHECK(ReadLnCheck(activity, earlyQuit) == expectedStatus);
            CHECK(earlyQuit == expectedEarlyQuit);
            CHECK(activity == 5);
            SetInputFd(-1);
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
        BindContext(context);
        ScopedFd readFd(fds[0]);
        ScopedFd writeFd(fds[1]);
        SetOutputFd(writeFd.Get());
        std::array<char, 8> buffer {};

        REQUIRE(context.Command("R %d\n", 9) > 0);
        REQUIRE(writeFd.Close() == 0);
        SetOutputFd(-1);

        REQUIRE(read(readFd.Get(), buffer.data(), buffer.size()) > 0);
        CHECK(std::string(buffer.data()) == "R 9\n");
    }

    SECTION("GIVEN invalid output fd WHEN Command is called THEN the write failure is reported")
    {
        auto context = MakeContext();
        BindContext(context);
        SetOutputFd(-1);

        CHECK(context.Command("R %d\n", 9) < 0);
    }

    SECTION("GIVEN initialized context WHEN Reset keeps groups THEN runtime state and streams are cleared")
    {
        auto context = MakeContext();
        BindContext(context);
        MarkInitialized();
        SetStreams("input", "message", "error");
        context.SetFinished();
        context.SetFailed();
        context.SetActivity(5);
        context.SetValidated(false);

        context.Reset(true);

        CHECK(context.Input().str().empty());
        CHECK(GetMessageString().empty());
        CHECK(GetErrorString().empty());
        CHECK_FALSE(context.Finished());
        CHECK_FALSE(context.Failed());
        CHECK(context.GetActivity() == 0);
        CHECK(context.GetValidated());
    }
}

TEST_CASE_METHOD(DistributedCudaContextTestFixture, "DistributedCudaContext worker-side dispatch and classification")
{
    SECTION("CUDA worker init failures stop RunTest before dispatch")
    {
        auto failingWorker          = std::make_unique<MockCudaWorkerThread>();
        failingWorker->m_initResult = DCGM_ST_GENERIC_ERROR;
        std::map<dcgm_field_entity_group_t, dcgm_field_eid_t> entities;
        entities[DCGM_FE_GPU] = 0;
        DistributedCudaContext context(&mockPhysicalGpu,
                                       std::move(entities),
                                       { DCGM_FE_GPU, 0 },
                                       "0",
                                       dcgmOps,
                                       &mockCudaOps,
                                       &runtime,
                                       std::move(failingWorker));
        BindContext(context);
        SetTestFieldId(DCGM_FI_PROF_SM_UTIL_RATIO);
        SetAttributes(MakePopulatedAttributes());

        CHECK(RunTest() == -1);
    }

    SECTION("Unsupported field ids are rejected with an explanatory error")
    {
        auto context = MakeContext();
        BindContext(context);
        SetTestFieldId(DCGM_FI_MAX_FIELDS);
        SetAttributes(MakePopulatedAttributes());
        SetStreams("R request\n", "", "");

        CHECK(RunTest() == -1);
        CHECK_THAT(GetErrorString(), ContainsSubstring("has not been implemented yet"));
    }

    SECTION("Selected supported fields emit start, completion, and pass responses")
    {
        auto const fieldId        = GENERATE(as<unsigned int> {},
                                             DCGM_FI_PROF_GR_ENGINE_UTIL_RATIO,
                                             DCGM_FI_PROF_SM_UTIL_RATIO,
                                             DCGM_FI_PROF_DRAM_UTIL_RATIO);
        auto const targetMaxValue = GENERATE(false, true);
        CAPTURE(fieldId, targetMaxValue);

        int fds[2];
        REQUIRE(pipe(fds) == 0);
        auto context = MakeContext();
        BindContext(context);
        ScopedFd readFd(fds[0]);
        ScopedFd writeFd(fds[1]);
        SetOutputFd(writeFd.Get());
        SetTestFieldId(fieldId);
        SetTiming(0.001, 0.001, targetMaxValue);
        SetAttributes(MakePopulatedAttributes(8, 6, 1));
        SetStreams("R request\n", "", "");

        CHECK(RunTest() == 0);
        REQUIRE(writeFd.Close() == 0);
        SetOutputFd(-1);
        auto const responses = ReadAllFromFd(readFd.Get());

        CHECK_THAT(responses, ContainsSubstring("S\n"));
        CHECK_THAT(responses, ContainsSubstring("D 1 1\nP\n"));
    }

    SECTION("ECC bandwidth classification follows compute capability boundaries")
    {
        auto context = MakeContext();
        BindContext(context);

        SetAttributes(MakePopulatedAttributes(8, 3, 1));
        CHECK(EccAffectsDramBandwidth());

        SetAttributes(MakePopulatedAttributes(8, 4, 1));
        CHECK(EccAffectsDramBandwidth());

        SetAttributes(MakePopulatedAttributes(8, 2, 1));
        CHECK_FALSE(EccAffectsDramBandwidth());

        SetAttributes(MakePopulatedAttributes(7, 5, 1));
        CHECK(EccAffectsDramBandwidth());

        SetAttributes(MakePopulatedAttributes(7, 6, 1));
        CHECK(EccAffectsDramBandwidth());

        SetAttributes(MakePopulatedAttributes(7, 4, 1));
        CHECK_FALSE(EccAffectsDramBandwidth());

        SetAttributes(MakePopulatedAttributes(9, 0, 1));
        CHECK_FALSE(EccAffectsDramBandwidth());
    }
}

} // namespace DcgmNs::ProfTester
