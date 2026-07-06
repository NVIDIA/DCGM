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

#include <catch2/catch_all.hpp>

#include <Nvlink.h>

#include "TestHelpers.hpp"

#include <cstring>

namespace
{
struct NvlinkApiState
{
    dcgmReturn_t statusReturn = DCGM_ST_OK;

    int statusCallCount = 0;

    dcgmHandle_t lastHandle = 0;
    dcgmNvLinkStatus_v5 status {};
};

NvlinkApiState g_nvlinkApi;

class TestGetNvLinkLinkStatuses : public GetNvLinkLinkStatuses
{
public:
    using GetNvLinkLinkStatuses::GetNvLinkLinkStatuses;

    dcgmReturn_t RunWithHandle(dcgmHandle_t handle)
    {
        m_dcgmHandle = handle;
        return DoExecuteConnected();
    }
};

void ResetNvlinkApi()
{
    g_nvlinkApi                                   = {};
    g_nvlinkApi.statusReturn                      = DCGM_ST_OK;
    g_nvlinkApi.status.version                    = dcgmNvLinkStatus_version5;
    g_nvlinkApi.status.numGpus                    = 1;
    g_nvlinkApi.status.gpus[0].entityId           = 2;
    g_nvlinkApi.status.gpus[0].linkState[0]       = DcgmNvLinkLinkStateUp;
    g_nvlinkApi.status.gpus[0].linkState[1]       = DcgmNvLinkLinkStateDown;
    g_nvlinkApi.status.gpus[0].linkState[2]       = DcgmNvLinkLinkStateDisabled;
    g_nvlinkApi.status.gpus[0].linkState[3]       = DcgmNvLinkLinkStateNotSupported;
    g_nvlinkApi.status.numNvSwitches              = 1;
    g_nvlinkApi.status.nvSwitches[0].entityId     = 4;
    g_nvlinkApi.status.nvSwitches[0].linkState[0] = DcgmNvLinkLinkStateUp;
    g_nvlinkApi.status.nvSwitches[0].linkState[1] = DcgmNvLinkLinkStateNotSupported;
}

void VerifyLinkEntityEncoding(dcgm_field_entity_group_t entityType,
                              dcgm_field_eid_t entityId,
                              uint16_t portIndex,
                              dcgm_field_eid_t expectedRaw)
{
    auto encodedId = HelperEncodeLinkEntity(entityType, entityId, portIndex);
    REQUIRE(encodedId == expectedRaw);

    dcgm_link_t link {};
    link.raw = encodedId;
    REQUIRE(link.parsed.type == entityType);
    REQUIRE(link.parsed.index == portIndex);

    if (entityType == DCGM_FE_GPU)
    {
        REQUIRE(link.parsed.gpuId == entityId);
    }
    else if (entityType == DCGM_FE_SWITCH)
    {
        REQUIRE(link.parsed.switchId == entityId);
    }
}
} // namespace

extern "C" dcgmReturn_t dcgmGetNvLinkLinkStatus(dcgmHandle_t handle, dcgmNvLinkStatus_v5 *linkStatus)
{
    g_nvlinkApi.statusCallCount++;
    g_nvlinkApi.lastHandle = handle;
    if (linkStatus == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }

    if (g_nvlinkApi.statusReturn != DCGM_ST_OK)
    {
        return g_nvlinkApi.statusReturn;
    }
    *linkStatus = g_nvlinkApi.status;
    return DCGM_ST_OK;
}

TEST_CASE("HelperEncodeLinkEntity encoding")
{
    struct TestCase
    {
        dcgm_field_entity_group_t entityType;
        dcgm_field_eid_t entityId;
        uint16_t portIndex;
        dcgm_field_eid_t expectedRaw;
    };

    TestCase const cases[] = {
        { DCGM_FE_GPU, 0, 0, 0x00000001 },     { DCGM_FE_GPU, 2, 5, 0x02000501 },
        { DCGM_FE_GPU, 1, 17, 0x01001101 },    { DCGM_FE_SWITCH, 0, 0, 0x00000003 },
        { DCGM_FE_SWITCH, 1, 10, 0x01000A03 }, { DCGM_FE_SWITCH, 2, 64, 0x02004003 },
    };

    for (auto const &tc : cases)
    {
        VerifyLinkEntityEncoding(tc.entityType, tc.entityId, tc.portIndex, tc.expectedRaw);
    }
}

TEST_CASE("Nvlink error count field names")
{
    GIVEN("Nvlink error-count field ids")
    {
        SECTION("Hopper and older per-link fields map to legacy error names")
        {
            CHECK(DcgmNs::Dcgmi::NvlinkDetail::GetErrorCountType(DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L0_TOTAL)
                  == "CRC FLIT Error");
            CHECK(DcgmNs::Dcgmi::NvlinkDetail::GetErrorCountType(DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L1_TOTAL)
                  == "CRC Data Error");
            CHECK(DcgmNs::Dcgmi::NvlinkDetail::GetErrorCountType(DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L2_TOTAL)
                  == "Replay Error");
            CHECK(DcgmNs::Dcgmi::NvlinkDetail::GetErrorCountType(DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L3_TOTAL)
                  == "Recovery Error");
        }

        SECTION("Blackwell and newer aggregate fields map to packet and BER names")
        {
            CHECK(DcgmNs::Dcgmi::NvlinkDetail::GetErrorCountType(DCGM_FI_DEV_NVLINK_RX_PACKET_MALFORMED_TOTAL)
                  == "Malformed Packet Error");
            CHECK(DcgmNs::Dcgmi::NvlinkDetail::GetErrorCountType(DCGM_FI_DEV_NVLINK_RX_PACKET_DROPPED_TOTAL)
                  == "Buffer Overrun Error");
            CHECK(DcgmNs::Dcgmi::NvlinkDetail::GetErrorCountType(DCGM_FI_DEV_NVLINK_RX_ERROR_TOTAL) == "Rx Error");
            CHECK(DcgmNs::Dcgmi::NvlinkDetail::GetErrorCountType(DCGM_FI_DEV_NVLINK_RX_REMOTE_ERROR_TOTAL)
                  == "Rx Remote Error");
            CHECK(DcgmNs::Dcgmi::NvlinkDetail::GetErrorCountType(DCGM_FI_DEV_NVLINK_RX_GENERAL_ERROR_TOTAL)
                  == "Rx General Error");
            CHECK(DcgmNs::Dcgmi::NvlinkDetail::GetErrorCountType(DCGM_FI_DEV_NVLINK_INTEGRITY_ERROR_TOTAL)
                  == "Link Integrity Error");
            CHECK(DcgmNs::Dcgmi::NvlinkDetail::GetErrorCountType(DCGM_FI_DEV_NVLINK_RX_SYMBOL_ERROR_TOTAL)
                  == "Rx Symbol Error");
            CHECK(DcgmNs::Dcgmi::NvlinkDetail::GetErrorCountType(DCGM_FI_DEV_NVLINK_SYMBOL_BER_RATIO) == "Symbol BER");
            CHECK(DcgmNs::Dcgmi::NvlinkDetail::GetErrorCountType(DCGM_FI_DEV_NVLINK_EFFECTIVE_BER_RATIO)
                  == "Effective BER");
            CHECK(DcgmNs::Dcgmi::NvlinkDetail::GetErrorCountType(DCGM_FI_DEV_NVLINK_EFFECTIVE_ERROR_TOTAL)
                  == "Effective Error");
            CHECK(DcgmNs::Dcgmi::NvlinkDetail::GetErrorCountType(DCGM_FI_DEV_NVLINK_COUNT_TX_DISCARDS)
                  == "Tx Discards");
        }

        SECTION("Unknown fields are labelled unknown")
        {
            CHECK(DcgmNs::Dcgmi::NvlinkDetail::GetErrorCountType(DCGM_FI_SYSTEM_FIELD_UNKNOWN) == "Unknown");
        }
    }
}

TEST_CASE("HelperFormatLinkEntityIds formatting")
{
    SECTION("GPU with 4 links formats correctly with port numbers")
    {
        dcgmNvLinkLinkState_t linkStates[4] = {
            DcgmNvLinkLinkStateUp, DcgmNvLinkLinkStateDown, DcgmNvLinkLinkStateNotSupported, DcgmNvLinkLinkStateUp
        };

        auto result = HelperFormatLinkEntityIds(DCGM_FE_GPU, 0, linkStates, 4);

        CHECK(result.find("0:001") != std::string::npos);
        CHECK(result.find("1:257") != std::string::npos);
        CHECK(result.find("3:769") != std::string::npos);
        CHECK(result.find("2:") == std::string::npos);
        CHECK(result.find("513") == std::string::npos);
    }

    SECTION("Switch with 2 links formats correctly with port numbers")
    {
        dcgmNvLinkLinkState_t linkStates[2] = { DcgmNvLinkLinkStateUp, DcgmNvLinkLinkStateUp };

        auto result = HelperFormatLinkEntityIds(DCGM_FE_SWITCH, 1, linkStates, 2);

        CHECK(result.find("0:16777219") != std::string::npos);
        CHECK(result.find("1:16777475") != std::string::npos);
    }

    SECTION("Device with no supported links returns empty string")
    {
        dcgmNvLinkLinkState_t linkStates[2] = { DcgmNvLinkLinkStateNotSupported, DcgmNvLinkLinkStateNotSupported };

        auto result = HelperFormatLinkEntityIds(DCGM_FE_GPU, 0, linkStates, 2);

        CHECK(result.empty());
    }

    SECTION("Many links format with line breaks")
    {
        dcgmNvLinkLinkState_t linkStates[10];
        for (int i = 0; i < 10; i++)
        {
            linkStates[i] = DcgmNvLinkLinkStateUp;
        }

        auto result = HelperFormatLinkEntityIds(DCGM_FE_GPU, 0, linkStates, 10);

        auto newlineCount = std::count(result.begin(), result.end(), '\n');
        CHECK(newlineCount == 0);

        CHECK(result.find("0:0001") != std::string::npos);
        CHECK(result.find("9:2305") != std::string::npos);
    }

    SECTION("NvSwitch with 75 links formats with proper line breaks")
    {
        dcgmNvLinkLinkState_t linkStates[75];
        for (int i = 0; i < 75; i++)
        {
            linkStates[i] = DcgmNvLinkLinkStateUp;
        }

        auto result = HelperFormatLinkEntityIds(DCGM_FE_SWITCH, 0, linkStates, 75);

        auto newlineCount = std::count(result.begin(), result.end(), '\n');
        CHECK(newlineCount == 6);

        CHECK(result.find(" 0:00003") != std::string::npos);
        CHECK(result.find("74:18947") != std::string::npos);

        auto firstNewline = result.find('\n');
        auto item11Pos    = result.find("11:02819");
        CHECK(item11Pos > firstNewline);
    }
}

TEST_CASE("Link entity item width scales with port numbers")
{
    struct TestCase
    {
        unsigned int numLinks;
        dcgm_field_entity_group_t entityType;
        std::string maxPortEntityId;
        int expectedNewlines;
    };

    TestCase const cases[] = {
        { 10, DCGM_FE_GPU, "9:2305", 0 },
        { 20, DCGM_FE_GPU, "19:4865", 1 },
        { 512, DCGM_FE_GPU, "511:130817", 56 },
        { 1024, DCGM_FE_SWITCH, "1023:261891", 127 },
    };

    for (auto const &tc : cases)
    {
        std::vector<dcgmNvLinkLinkState_t> linkStates(tc.numLinks, DcgmNvLinkLinkStateUp);

        auto result = HelperFormatLinkEntityIds(tc.entityType, 0, linkStates.data(), tc.numLinks);

        CHECK(result.find(tc.maxPortEntityId) != std::string::npos);

        auto newlineCount = std::count(result.begin(), result.end(), '\n');
        CHECK(newlineCount == tc.expectedNewlines);
    }
}

TEST_CASE("Link entity formatting handles sparse ports correctly")
{
    SECTION("Sparse 512-link set sizes width correctly")
    {
        dcgmNvLinkLinkState_t linkStates[512];

        for (int i = 0; i < 512; i++)
        {
            linkStates[i] = DcgmNvLinkLinkStateNotSupported;
        }
        linkStates[0] = linkStates[100] = linkStates[200] = DcgmNvLinkLinkStateUp;
        linkStates[300] = linkStates[400] = linkStates[500] = DcgmNvLinkLinkStateUp;

        auto result = HelperFormatLinkEntityIds(DCGM_FE_GPU, 0, linkStates, 512);

        CHECK(result.find("  0:000001") != std::string::npos);
        CHECK(result.find("500:128001") != std::string::npos);

        auto newlineCount = std::count(result.begin(), result.end(), '\n');
        CHECK(newlineCount == 0);
    }
}

TEST_CASE("Nvlink::DisplayNvLinkLinkStatus")
{
    GIVEN("mocked NvLink status")
    {
        ResetNvlinkApi();
        Nvlink nvlink;
        auto handle = static_cast<dcgmHandle_t>(0x9a);

        SECTION("GPU and NvSwitch link states are displayed")
        {
            CoutCapture capture;

            CHECK(nvlink.DisplayNvLinkLinkStatus(handle, false) == DCGM_ST_OK);
            CHECK(g_nvlinkApi.statusCallCount == 1);
            CHECK(g_nvlinkApi.lastHandle == handle);
            CHECK(capture.str().find("NvLink Link Status") != std::string::npos);
            CHECK(capture.str().find("gpuId 2") != std::string::npos);
            CHECK(capture.str().find("U D X _") != std::string::npos);
            CHECK(capture.str().find("physicalId 4") != std::string::npos);
            CHECK(capture.str().find("Key: Up=U") != std::string::npos);
        }

        SECTION("Entity IDs are displayed when requested")
        {
            CoutCapture capture;

            CHECK(nvlink.DisplayNvLinkLinkStatus(handle, true) == DCGM_ST_OK);
            CHECK(capture.str().find("Link Entities:") != std::string::npos);
            CHECK(capture.str().find("0:") != std::string::npos);
        }

        SECTION("Empty systems are reported")
        {
            g_nvlinkApi.status.numGpus       = 0;
            g_nvlinkApi.status.numNvSwitches = 0;
            CoutCapture capture;

            CHECK(nvlink.DisplayNvLinkLinkStatus(handle, true) == DCGM_ST_OK);
            CHECK(capture.str().find("No GPUs found") != std::string::npos);
            CHECK(capture.str().find("No NvSwitches found") != std::string::npos);
        }

        SECTION("DCGM failures are returned")
        {
            g_nvlinkApi.statusReturn = DCGM_ST_BADPARAM;
            CoutCapture capture;

            CHECK(nvlink.DisplayNvLinkLinkStatus(handle, false) == DCGM_ST_BADPARAM);
            CHECK(g_nvlinkApi.statusCallCount == 1);
            CHECK(capture.str().find("Unable to retrieve NvLink link status") != std::string::npos);
        }
    }
}

TEST_CASE("GetNvLinkLinkStatuses::DoExecuteConnected")
{
    ResetNvlinkApi();
    auto handle = static_cast<dcgmHandle_t>(0x9b);
    TestGetNvLinkLinkStatuses command("localhost", true);
    CoutCapture capture;

    CHECK(command.RunWithHandle(handle) == DCGM_ST_OK);
    CHECK(g_nvlinkApi.statusCallCount == 1);
    CHECK(g_nvlinkApi.lastHandle == handle);
    CHECK(capture.str().find("Link Entities:") != std::string::npos);
}
