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

#include <DcgmModuleIntrospect.h>
#include <catch2/catch_all.hpp>

extern "C" DCGM_PUBLIC_API DcgmModule *dcgm_alloc_module_instance(dcgmCoreCallbacks_t *dcc);
extern "C" DCGM_PUBLIC_API void dcgm_free_module_instance(DcgmModule *freeMe);
extern "C" DCGM_PUBLIC_API dcgmReturn_t dcgm_module_process_message(DcgmModule *module,
                                                                    dcgm_module_command_header_t *moduleCommand);

namespace
{
dcgmCoreCallbacks_t MakeCallbacks()
{
    return { .version  = dcgmCoreCallbacks_version,
             .postfunc = [](dcgm_module_command_header_t *, void *) -> dcgmReturn_t { return DCGM_ST_OK; },
             .poster     = nullptr,
             .loggerfunc = [](void const *) { /* do nothing */ } };
}

void PopulateIntrospectHeader(dcgm_module_command_header_t &header, unsigned int subCommand, unsigned int version)
{
    header.moduleId   = DcgmModuleIdIntrospect;
    header.subCommand = subCommand;
    header.version    = version;
}

void PopulateCoreHeader(dcgm_module_command_header_t &header, unsigned int subCommand, unsigned int version)
{
    header.moduleId   = DcgmModuleIdCore;
    header.subCommand = subCommand;
    header.version    = version;
}

/**
 * Metadata manager test double with deterministic hostengine metadata responses.
 */
class FakeMetadataManager : public DcgmMetadataManager
{
public:
    /**
     * Constructor.
     *
     * @param[in] dcc callbacks forwarded to the base metadata manager.
     */
    explicit FakeMetadataManager(dcgmCoreCallbacks_t &dcc)
        : DcgmMetadataManager(dcc)
    {}

    /**
     * Return configured memory usage without reading procfs.
     *
     * @param[out] bytesUsed populated with configured byte count.
     * @param[in] waitIfNoData records caller wait preference.
     * @return configured memory status.
     */
    dcgmReturn_t GetHostEngineBytesUsed(long long &bytesUsed, bool waitIfNoData) override
    {
        lastMemoryWaitIfNoData = waitIfNoData;
        bytesUsed              = memoryBytes;
        return memoryReturn;
    }

    /**
     * Return configured CPU utilization without reading procfs.
     *
     * @param[out] cpuUtil populated with configured utilization values.
     * @param[in] waitIfNoData records caller wait preference.
     * @return configured CPU status.
     */
    dcgmReturn_t GetCpuUtilization(CpuUtil &cpuUtil, bool waitIfNoData) override
    {
        lastCpuWaitIfNoData = waitIfNoData;
        cpuUtil             = cpuUtilResponse;
        return cpuReturn;
    }

    long long memoryBytes       = 0;
    dcgmReturn_t memoryReturn   = DCGM_ST_OK;
    CpuUtil cpuUtilResponse     = {};
    dcgmReturn_t cpuReturn      = DCGM_ST_OK;
    bool lastMemoryWaitIfNoData = false;
    bool lastCpuWaitIfNoData    = false;
};
} //namespace

TEST_CASE("DcgmModuleIntrospect::ProcessMessage rejects null moduleCommand")
{
    auto dcc = MakeCallbacks();
    DcgmModuleIntrospect introspectModule(dcc);
    CHECK(introspectModule.ProcessMessage(nullptr) == DCGM_ST_BADPARAM);
}

TEST_CASE("DcgmModuleIntrospect::ProcessMessage rejects mismatched moduleId")
{
    auto dcc = MakeCallbacks();
    DcgmModuleIntrospect introspectModule(dcc);

    dcgm_module_command_header_t header {};
    header.moduleId = DcgmModuleIdNvSwitch;
    CHECK(introspectModule.ProcessMessage(&header) == DCGM_ST_BADPARAM);
}

TEST_CASE("DcgmModuleIntrospect::ProcessMessage handles introspect routing errors")
{
    auto dcc = MakeCallbacks();
    DcgmModuleIntrospect introspectModule(dcc);

    SECTION("unknown introspect subcommand")
    {
        dcgm_module_command_header_t header {};
        PopulateIntrospectHeader(header, DCGM_INTROSPECT_SR_COUNT + 10, 1);

        CHECK(introspectModule.ProcessMessage(&header) == DCGM_ST_FUNCTION_NOT_FOUND);
    }

    SECTION("introspect subcommands reject header version mismatches")
    {
        GIVEN("an introspect message with an invalid header version")
        {
            SECTION("hostengine memory usage")
            {
                dcgm_introspect_msg_he_mem_usage_v1 msg {};
                PopulateIntrospectHeader(
                    msg.header, DCGM_INTROSPECT_SR_HOSTENGINE_MEM_USAGE, dcgm_introspect_msg_he_mem_usage_version1 - 1);
                msg.memoryInfo.version = dcgmIntrospectMemory_version1;

                CHECK(introspectModule.ProcessMessage(&msg.header) == DCGM_ST_VER_MISMATCH);
            }

            SECTION("hostengine CPU utilization")
            {
                dcgm_introspect_msg_he_cpu_util_v1 msg {};
                PopulateIntrospectHeader(
                    msg.header, DCGM_INTROSPECT_SR_HOSTENGINE_CPU_UTIL, dcgm_introspect_msg_he_cpu_util_version1 - 1);
                msg.cpuUtil.version = dcgmIntrospectCpuUtil_version1;

                CHECK(introspectModule.ProcessMessage(&msg.header) == DCGM_ST_VER_MISMATCH);
            }
        }
    }

    SECTION("introspect subcommands reject payload version mismatches")
    {
        GIVEN("an introspect message with a valid header and invalid payload")
        {
            SECTION("hostengine memory usage")
            {
                dcgm_introspect_msg_he_mem_usage_v1 msg {};
                PopulateIntrospectHeader(
                    msg.header, DCGM_INTROSPECT_SR_HOSTENGINE_MEM_USAGE, dcgm_introspect_msg_he_mem_usage_version1);
                msg.memoryInfo.version = dcgmIntrospectMemory_version1 - 1;

                CHECK(introspectModule.ProcessMessage(&msg.header) == DCGM_ST_VER_MISMATCH);
            }

            SECTION("hostengine CPU utilization")
            {
                dcgm_introspect_msg_he_cpu_util_v1 msg {};
                PopulateIntrospectHeader(
                    msg.header, DCGM_INTROSPECT_SR_HOSTENGINE_CPU_UTIL, dcgm_introspect_msg_he_cpu_util_version1);
                msg.cpuUtil.version = dcgmIntrospectCpuUtil_version1 - 1;

                CHECK(introspectModule.ProcessMessage(&msg.header) == DCGM_ST_VER_MISMATCH);
            }
        }
    }
}

TEST_CASE("DcgmModuleIntrospect::ProcessMessage handles simple core routing")
{
    auto dcc = MakeCallbacks();
    DcgmModuleIntrospect introspectModule(dcc);

    SECTION("unknown core subcommand")
    {
        dcgm_module_command_header_t header {};
        PopulateCoreHeader(header, DCGM_CORE_SR_MARK_MODULES_RELOADABLE + 10, 1);

        CHECK(introspectModule.ProcessMessage(&header) == DCGM_ST_FUNCTION_NOT_FOUND);
    }

    SECTION("pause resume returns success")
    {
        dcgm_module_command_header_t header {};
        PopulateCoreHeader(header, DCGM_CORE_SR_PAUSE_RESUME, 1);

        CHECK(introspectModule.ProcessMessage(&header) == DCGM_ST_OK);
    }

    SECTION("logging changed returns success")
    {
        dcgm_core_msg_logging_changed_t msg {};
        PopulateCoreHeader(msg.header, DCGM_CORE_SR_LOGGING_CHANGED, dcgm_core_msg_logging_changed_version);

        CHECK(introspectModule.ProcessMessage(&msg.header) == DCGM_ST_OK);
    }
}

SCENARIO("DcgmModuleIntrospect routes hostengine metadata requests", "[introspect]")
{
    auto dcc                 = MakeCallbacks();
    auto fakeMetadataManager = std::make_unique<FakeMetadataManager>(dcc);
    auto *fakeMetadata       = fakeMetadataManager.get();
    DcgmModuleIntrospect introspectModule(dcc, std::move(fakeMetadataManager));

    GIVEN("a valid hostengine memory usage request")
    {
        fakeMetadata->memoryBytes = 8675309;

        dcgm_introspect_msg_he_mem_usage_v1 msg {};
        PopulateIntrospectHeader(
            msg.header, DCGM_INTROSPECT_SR_HOSTENGINE_MEM_USAGE, dcgm_introspect_msg_he_mem_usage_version1);
        msg.memoryInfo.version = dcgmIntrospectMemory_version1;
        msg.waitIfNoData       = 0;

        WHEN("the request is processed")
        {
            auto ret = introspectModule.ProcessMessage(&msg.header);

            THEN("memory usage is populated")
            {
                CHECK(ret == DCGM_ST_OK);
                CHECK(msg.memoryInfo.bytesUsed == 8675309);
                CHECK_FALSE(fakeMetadata->lastMemoryWaitIfNoData);
            }
        }
    }

    GIVEN("a valid hostengine CPU utilization request without waiting")
    {
        fakeMetadata->cpuReturn = DCGM_ST_NO_DATA;

        dcgm_introspect_msg_he_cpu_util_v1 msg {};
        PopulateIntrospectHeader(
            msg.header, DCGM_INTROSPECT_SR_HOSTENGINE_CPU_UTIL, dcgm_introspect_msg_he_cpu_util_version1);
        msg.cpuUtil.version = dcgmIntrospectCpuUtil_version1;
        msg.waitIfNoData    = 0;

        WHEN("no baseline has been recorded yet")
        {
            auto ret = introspectModule.ProcessMessage(&msg.header);

            THEN("the module reports no data")
            {
                CHECK(ret == DCGM_ST_NO_DATA);
                CHECK_FALSE(fakeMetadata->lastCpuWaitIfNoData);
            }
        }
    }

    GIVEN("a valid hostengine CPU utilization request that waits for data")
    {
        fakeMetadata->cpuUtilResponse = { .total = 0.75, .kernel = 0.25, .user = 0.5 };

        dcgm_introspect_msg_he_cpu_util_v1 msg {};
        PopulateIntrospectHeader(
            msg.header, DCGM_INTROSPECT_SR_HOSTENGINE_CPU_UTIL, dcgm_introspect_msg_he_cpu_util_version1);
        msg.cpuUtil.version = dcgmIntrospectCpuUtil_version1;
        msg.waitIfNoData    = 1;

        WHEN("the request is processed")
        {
            auto ret = introspectModule.ProcessMessage(&msg.header);

            THEN("CPU utilization fields are populated")
            {
                CHECK(ret == DCGM_ST_OK);
                CHECK(msg.cpuUtil.user == Catch::Approx(0.5));
                CHECK(msg.cpuUtil.kernel == Catch::Approx(0.25));
                CHECK(msg.cpuUtil.total == Catch::Approx(0.75));
                CHECK(fakeMetadata->lastCpuWaitIfNoData);
            }
        }
    }
}

SCENARIO("DcgmModuleIntrospect injection constructor handles null metadata manager", "[introspect]")
{
    auto dcc = MakeCallbacks();

    GIVEN("a null metadata manager")
    {
        std::unique_ptr<DcgmMetadataManager> metadataManager;
        DcgmModuleIntrospect introspectModule(dcc, std::move(metadataManager));

        WHEN("a core message is processed")
        {
            dcgm_module_command_header_t header {};
            PopulateCoreHeader(header, DCGM_CORE_SR_PAUSE_RESUME, 1);

            THEN("the module remains usable")
            {
                CHECK(introspectModule.ProcessMessage(&header) == DCGM_ST_OK);
            }
        }
    }
}

SCENARIO("DcgmModuleIntrospect C module API manages module instances", "[introspect]")
{
    auto dcc = MakeCallbacks();

    GIVEN("a module allocated through the exported C API")
    {
        DcgmModule *module = dcgm_alloc_module_instance(&dcc);
        REQUIRE(module != nullptr);

        WHEN("a message is passed through the exported process callback")
        {
            dcgm_module_command_header_t header {};
            PopulateCoreHeader(header, DCGM_CORE_SR_PAUSE_RESUME, 1);

            THEN("the message is processed by the module")
            {
                CHECK(dcgm_module_process_message(module, &header) == DCGM_ST_OK);
            }
        }

        WHEN("a null message is passed through the exported process callback")
        {
            THEN("module dispatch reports a bad parameter")
            {
                CHECK(dcgm_module_process_message(module, nullptr) == DCGM_ST_BADPARAM);
            }
        }

        dcgm_free_module_instance(module);
    }

    GIVEN("a null callback table")
    {
        THEN("allocation is guarded")
        {
            CHECK(dcgm_alloc_module_instance(nullptr) == nullptr);
        }
    }
}
