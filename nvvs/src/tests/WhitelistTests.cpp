/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
#include <DcgmError.h>
#include <ParsingUtility.h>
#include <PluginStrings.h>
#include <catch2/catch.hpp>
#include <fstream>

#define private public // Hack to directly manipulate private members
#include <Whitelist.h>

#define VALID_ID   "1194"
#define INVALID_ID "1190"

using namespace DcgmNs::Nvvs;

class TestWhitelist : protected Whitelist
{
public:
    void WrapperUpdateGlobalsForDeviceId(const std::string &deviceId);
    TestWhitelist(const ConfigFileParser_v2 &parser)
        : Whitelist(parser)
    {}
};

static FrameworkConfig fwcfg;
static ConfigFileParser_v2 parser("", fwcfg);

// bool Whitelist::isWhitelisted(std::string deviceId)
SCENARIO("Whitelist is initialized when constructed", "[.]")
{
    Whitelist wl(parser);
    WHEN("Whitelist is initialized")
    {
        THEN("Whitelist correctly returns status of device IDs")
        {
            CHECK(wl.isWhitelisted(VALID_ID));
            CHECK(!wl.isWhitelisted(INVALID_ID));
        }
    }
}

SCENARIO("All valid GPUs are in whitelist", "[.]")
{
    Whitelist wl(parser);
    // Tesla K8
    CHECK(wl.isWhitelisted("1194"));
    // Tesla K80 ("Stella Duo" Gemini)
    CHECK(wl.isWhitelisted("102d"));
    // GP100 SKU 201
    CHECK(wl.isWhitelisted("15fa"));
    // GP100 SKU 200
    CHECK(wl.isWhitelisted("15fb"));
    // DGX Station GP100
    CHECK(wl.isWhitelisted("15fc"));
    // Serial Number : PH400-C01-P0203
    CHECK(wl.isWhitelisted("15f8"));
    // Serial Number : PH400 SKU 202
    CHECK(wl.isWhitelisted("15f7"));
    // GP100 Bringup - eris-l64-test12*
    CHECK(wl.isWhitelisted("15ff"));
    // Tesla P100-SXM2-16GB
    CHECK(wl.isWhitelisted("15f9"));
    // Tesla Stella Solo - Internal only
    CHECK(wl.isWhitelisted("102f"));
    // Tesla K40d ("Stella" DFF)
    CHECK(wl.isWhitelisted("102e"));
    // Tesla K40m
    CHECK(wl.isWhitelisted("1023"));
    // Tesla K40c
    CHECK(wl.isWhitelisted("1024"));
    // Tesla K40t ("Atlas" TTP) - values copied from Tesla K40c
    CHECK(wl.isWhitelisted("102a"));
    // Tesla K40s - values copied from Tesla K40c
    CHECK(wl.isWhitelisted("1029"));
    // Tesla K20X - values copied from Tesla K20xm
    CHECK(wl.isWhitelisted("101e"));
    // Tesla K20Xm
    CHECK(wl.isWhitelisted("1021"));
    // Tesla K20c - values copied from Tesla K20xm (except power)
    CHECK(wl.isWhitelisted("1022"));
    // Tesla K20m - values copied from Tesla K20xm
    CHECK(wl.isWhitelisted("1028"));
    // Tesla K20X
    CHECK(wl.isWhitelisted("1020"));
    // Tesla K10
    CHECK(wl.isWhitelisted("118f"));
    // Quadro M6000 //Not supported officially for NVVS
    CHECK(wl.isWhitelisted("17f0"));
    // Quadro M5000
    CHECK(wl.isWhitelisted("13f0"));
    // GeForce GTX TITAN X
    CHECK(wl.isWhitelisted("17c2"));
    // Tesla M60
    CHECK(wl.isWhitelisted("13f2"));
    // Tesla M6 (Copied from Tesla M60)
    CHECK(wl.isWhitelisted("13f3"));
    // Tesla M40
    CHECK(wl.isWhitelisted("17fd"));
    // Tesla M4
    CHECK(wl.isWhitelisted("1431"));
    // Mostly copied from Tesla M40
    CHECK(wl.isWhitelisted("13bd"));
    // Serial Number: PG503-A00-P0447
    CHECK(wl.isWhitelisted("1dbd"));
    // Tesla GV100 DGX Station - Part number 692-2G500-0201-300
    CHECK(wl.isWhitelisted("1db2"));
    // Tesla GV100 SXM2-32GB SKU 895
    CHECK(wl.isWhitelisted("1db1"));
    // Tesla GV100 SXM2-16GB SKU 890
    CHECK(wl.isWhitelisted("1db0"));
    // Tesla GV100 PCIE-16GB SKU 893
    CHECK(wl.isWhitelisted("1db4"));
    // Tesla V100-SXM2-32GB-LS - PG503 SKU 203
    CHECK(wl.isWhitelisted("1db5"));
    // Tesla V100 PCI-E-32GB (Passive) - PG500 SKU 202
    CHECK(wl.isWhitelisted("1db6"));
    // Tesla P40
    CHECK(wl.isWhitelisted("1b38"));
    // Tesla P10
    CHECK(wl.isWhitelisted("1b39"));
    // Tesla P6
    CHECK(wl.isWhitelisted("1bb4"));
    // Tesla P4
    CHECK(wl.isWhitelisted("1bb3"));
    // Tesla V100-HS
    CHECK(wl.isWhitelisted("1db3"));
    // Tesla V100 32GB
    CHECK(wl.isWhitelisted("1db5"));
    // Tesla V100 DGX-Station 32GB
    CHECK(wl.isWhitelisted("1db7"));
    // Tesla V100 DGX-Station 32GB
    CHECK(wl.isWhitelisted("1db8"));
    // Tesla T4 - TU104_PG183_SKU200
    CHECK(wl.isWhitelisted("1eb8"));
    // Tesla T40 - TU102-895
    CHECK(wl.isWhitelisted("1e38"));
    // PRIS TS1
    CHECK(wl.isWhitelisted("2236", "149d"));
    // A10c/noc
    CHECK(wl.isWhitelisted("2236", "1482"));
}

SCENARIO("Whitelist provides default values for device", "[.]")
{
    Whitelist wl(parser);
    TestParameters tp;
    GIVEN("valid id, valid test")
    {
        const std::string id       = "102d";
        const std::string testName = SMSTRESS_PLUGIN_NAME;


        // For some amazing reason, when we are getting the test parameters, we
        // free the memory too... so without setting them first, we get a segfault
        tp.AddString(SMSTRESS_STR_IS_ALLOWED, "False");
        tp.AddDouble(SMSTRESS_STR_TARGET_PERF, 0.0);
        tp.AddString(SMSTRESS_STR_USE_DGEMM, "True");
        tp.AddDouble(SMSTRESS_STR_TEMPERATURE_MAX, 0.0);

        wl.getDefaultsByDeviceId(testName, id, &tp);

        CHECK(tp.GetString(SMSTRESS_STR_IS_ALLOWED) == "True");
        CHECK(tp.GetDouble(SMSTRESS_STR_TARGET_PERF) == 950.0);
        CHECK(tp.GetString(SMSTRESS_STR_USE_DGEMM) == "False");
        CHECK(tp.GetDouble(SMSTRESS_STR_TEMPERATURE_MAX) == 0.0);
    }
}

SCENARIO("UpdateGlobalsForDeviceId sets the appropriate global flags", "[.]")
{
    TestWhitelist wl(parser);

    GIVEN("An ID that requires global change")
    {
        std::string id = "102d";
        WHEN("throttleIgnoreMask is blank")
        {
            nvvsCommon.throttleIgnoreMask = DCGM_INT64_BLANK;
            THEN("UpdateGlobalsForDeviceId sets throttle mask")
            {
                wl.WrapperUpdateGlobalsForDeviceId(id);
                CHECK(nvvsCommon.throttleIgnoreMask == MAX_THROTTLE_IGNORE_MASK_VALUE);
            }
        }

        WHEN("throttleIgnoreMask is not blank")
        {
            nvvsCommon.throttleIgnoreMask = 1;
            THEN("UpdateGlobalsForDeviceId does not set throttle mask")
            {
                wl.WrapperUpdateGlobalsForDeviceId(id);
                CHECK(nvvsCommon.throttleIgnoreMask == 1);
            }
        }
    }

    GIVEN("An ID that does not require global change")
    {
        std::string id = "0000";

        WHEN("throttleIgnoreMask is blank")
        {
            nvvsCommon.throttleIgnoreMask = DCGM_INT64_BLANK;
            THEN("UpdateGlobalsForDeviceId does not set throttle mask")
            {
                wl.WrapperUpdateGlobalsForDeviceId(id);
                CHECK(nvvsCommon.throttleIgnoreMask == DCGM_INT64_BLANK);
            }
        }

        WHEN("throttleIgnoreMask is not blank")
        {
            nvvsCommon.throttleIgnoreMask = 1;
            THEN("UpdateGlobalsForDeviceId does not set throttle mask")
            {
                wl.WrapperUpdateGlobalsForDeviceId(id);
                CHECK(nvvsCommon.throttleIgnoreMask == 1);
            }
        }
    }
}

void TestWhitelist::WrapperUpdateGlobalsForDeviceId(const std::string &deviceId)
{
    UpdateGlobalsForDeviceId(deviceId);
}
