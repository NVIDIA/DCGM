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
#include <DcgmError.h>
#include <ParsingUtility.h>
#include <PluginStrings.h>
#include <catch2/catch_all.hpp>
#include <fstream>
#include <stdexcept>

#include <any>         // It must be included here before the hacky define below
#define private public // Hack to directly manipulate private members
#include <Allowlist.h>

#define VALID_ID   "1194"
#define INVALID_ID "1190"

using namespace DcgmNs::Nvvs;

class TestAllowlist : protected Allowlist
{
public:
    void WrapperUpdateGlobalsForDeviceId(const std::string &deviceId);
    TestAllowlist(const ConfigFileParser_v2 &parser)
        : Allowlist(parser)
    {}
};

static FrameworkConfig fwcfg;

// bool Allowlist::IsAllowlisted(std::string deviceId)
SCENARIO("Allowlist is initialized when constructed", "[.]")
{
    try
    {
        ConfigFileParser_v2 parser("", fwcfg);
        Allowlist wl(parser);
        WHEN("Allowlist is initialized")
        {
            THEN("Allowlist correctly returns status of device IDs")
            {
                CHECK(wl.IsAllowlisted(VALID_ID));
                CHECK(!wl.IsAllowlisted(INVALID_ID));
            }
        }
    }
    catch (std::runtime_error &e)
    {
        CHECK(false);
    }
}

SCENARIO("All valid GPUs are in allowlist", "[.]")
{
    try
    {
        ConfigFileParser_v2 parser("", fwcfg);
        Allowlist wl(parser);
        // Tesla K8
        CHECK(wl.IsAllowlisted("1194"));
        // Tesla K80 ("Stella Duo" Gemini)
        CHECK(wl.IsAllowlisted("102d"));
        // GP100 SKU 201
        CHECK(wl.IsAllowlisted("15fa"));
        // GP100 SKU 200
        CHECK(wl.IsAllowlisted("15fb"));
        // DGX Station GP100
        CHECK(wl.IsAllowlisted("15fc"));
        // Serial Number : PH400-C01-P0203
        CHECK(wl.IsAllowlisted("15f8"));
        // Serial Number : PH400 SKU 202
        CHECK(wl.IsAllowlisted("15f7"));
        // GP100 Bringup - eris-l64-test12*
        CHECK(wl.IsAllowlisted("15ff"));
        // Tesla P100-SXM2-16GB
        CHECK(wl.IsAllowlisted("15f9"));
        // Tesla Stella Solo - Internal only
        CHECK(wl.IsAllowlisted("102f"));
        // Tesla K40d ("Stella" DFF)
        CHECK(wl.IsAllowlisted("102e"));
        // Tesla K40m
        CHECK(wl.IsAllowlisted("1023"));
        // Tesla K40c
        CHECK(wl.IsAllowlisted("1024"));
        // Tesla K40t ("Atlas" TTP) - values copied from Tesla K40c
        CHECK(wl.IsAllowlisted("102a"));
        // Tesla K40s - values copied from Tesla K40c
        CHECK(wl.IsAllowlisted("1029"));
        // Tesla K20X - values copied from Tesla K20xm
        CHECK(wl.IsAllowlisted("101e"));
        // Tesla K20Xm
        CHECK(wl.IsAllowlisted("1021"));
        // Tesla K20c - values copied from Tesla K20xm (except power)
        CHECK(wl.IsAllowlisted("1022"));
        // Tesla K20m - values copied from Tesla K20xm
        CHECK(wl.IsAllowlisted("1028"));
        // Tesla K20X
        CHECK(wl.IsAllowlisted("1020"));
        // Tesla K10
        CHECK(wl.IsAllowlisted("118f"));
        // Quadro M6000 //Not supported officially for NVVS
        CHECK(wl.IsAllowlisted("17f0"));
        // Quadro M5000
        CHECK(wl.IsAllowlisted("13f0"));
        // GeForce GTX TITAN X
        CHECK(wl.IsAllowlisted("17c2"));
        // Tesla M60
        CHECK(wl.IsAllowlisted("13f2"));
        // Tesla M6 (Copied from Tesla M60)
        CHECK(wl.IsAllowlisted("13f3"));
        // Tesla M40
        CHECK(wl.IsAllowlisted("17fd"));
        // Tesla M4
        CHECK(wl.IsAllowlisted("1431"));
        // Mostly copied from Tesla M40
        CHECK(wl.IsAllowlisted("13bd"));
        // Serial Number: PG503-A00-P0447
        CHECK(wl.IsAllowlisted("1dbd"));
        // Tesla GV100 DGX Station - Part number 692-2G500-0201-300
        CHECK(wl.IsAllowlisted("1db2"));
        // Tesla GV100 SXM2-32GB SKU 895
        CHECK(wl.IsAllowlisted("1db1"));
        // Tesla GV100 SXM2-16GB SKU 890
        CHECK(wl.IsAllowlisted("1db0"));
        // Tesla GV100 PCIE-16GB SKU 893
        CHECK(wl.IsAllowlisted("1db4"));
        // Tesla V100-SXM2-32GB-LS - PG503 SKU 203
        CHECK(wl.IsAllowlisted("1db5"));
        // Tesla V100 PCI-E-32GB (Passive) - PG500 SKU 202
        CHECK(wl.IsAllowlisted("1db6"));
        // Tesla P40
        CHECK(wl.IsAllowlisted("1b38"));
        // Tesla P10
        CHECK(wl.IsAllowlisted("1b39"));
        // Tesla P6
        CHECK(wl.IsAllowlisted("1bb4"));
        // Tesla P4
        CHECK(wl.IsAllowlisted("1bb3"));
        // Tesla V100-HS
        CHECK(wl.IsAllowlisted("1db3"));
        // Tesla V100 32GB
        CHECK(wl.IsAllowlisted("1db5"));
        // Tesla V100 DGX-Station 32GB
        CHECK(wl.IsAllowlisted("1db7"));
        // Tesla V100 DGX-Station 32GB
        CHECK(wl.IsAllowlisted("1db8"));
        // Tesla T4 - TU104_PG183_SKU200
        CHECK(wl.IsAllowlisted("1eb8"));
        // Tesla T40 - TU102-895
        CHECK(wl.IsAllowlisted("1e38"));
        // PRIS TS1
        CHECK(wl.IsAllowlisted("2236", "149d"));
        // A10c/noc
        CHECK(wl.IsAllowlisted("2236", "1482"));
        // A800 watercooled
        CHECK(wl.IsAllowlisted("20f3", "179C"));
        // A800
        CHECK(wl.IsAllowlisted("20f3"));
    }
    catch (std::runtime_error &e)
    {
        CHECK(false);
    }
}

SCENARIO("Allowlist provides default values for device", "[.]")
{
    try
    {
        ConfigFileParser_v2 parser("", fwcfg);
        Allowlist wl(parser);
        TestParameters tp;
        GIVEN("valid id, valid test")
        {
            const std::string id       = "102d";
            const std::string testName = SMSTRESS_PLUGIN_NAME;

            // For some amazing reason, when we are getting the test parameters, we
            // free the memory too... so without setting them first, we get a segfault
            tp.AddDouble(SMSTRESS_STR_TARGET_PERF, 0.0);
            tp.AddString(SMSTRESS_STR_USE_DGEMM, "True");
            tp.AddDouble(SMSTRESS_STR_TEMPERATURE_MAX, 0.0);

            wl.GetDefaultsByDeviceId(testName, id, &tp);

            CHECK(tp.GetDouble(SMSTRESS_STR_TARGET_PERF) == 950.0);
            CHECK(tp.GetString(SMSTRESS_STR_USE_DGEMM) == "False");
            CHECK(tp.GetDouble(SMSTRESS_STR_TEMPERATURE_MAX) == 0.0);
        }
    }
    catch (std::runtime_error &e)
    {
        CHECK(false);
    }
}

SCENARIO("UpdateGlobalsForDeviceId sets the appropriate global flags", "[.]")
{
    try
    {
        ConfigFileParser_v2 parser("", fwcfg);
        TestAllowlist wl(parser);

        GIVEN("An ID that requires global change")
        {
            std::string id = "102d";
            WHEN("clocksEventIgnoreMask is blank")
            {
                nvvsCommon.clocksEventIgnoreMask = DCGM_INT64_BLANK;
                THEN("UpdateGlobalsForDeviceId sets clocksEvent mask")
                {
                    wl.WrapperUpdateGlobalsForDeviceId(id);
                    CHECK(nvvsCommon.clocksEventIgnoreMask == MAX_CLOCKS_EVENT_IGNORE_MASK_VALUE);
                }
            }

            WHEN("clocksEventIgnoreMask is not blank")
            {
                nvvsCommon.clocksEventIgnoreMask = 1;
                THEN("UpdateGlobalsForDeviceId does not set clocks event mask")
                {
                    wl.WrapperUpdateGlobalsForDeviceId(id);
                    CHECK(nvvsCommon.clocksEventIgnoreMask == 1);
                }
            }
        }

        GIVEN("An ID that does not require global change")
        {
            std::string id = "0000";

            WHEN("clocksEventIgnoreMask is blank")
            {
                nvvsCommon.clocksEventIgnoreMask = DCGM_INT64_BLANK;
                THEN("UpdateGlobalsForDeviceId does not set clocksEvent mask")
                {
                    wl.WrapperUpdateGlobalsForDeviceId(id);
                    CHECK(nvvsCommon.clocksEventIgnoreMask == DCGM_INT64_BLANK);
                }
            }

            WHEN("clocksEventIgnoreMask is not blank")
            {
                nvvsCommon.clocksEventIgnoreMask = 1;
                THEN("UpdateGlobalsForDeviceId does not set clocksEvent mask")
                {
                    wl.WrapperUpdateGlobalsForDeviceId(id);
                    CHECK(nvvsCommon.clocksEventIgnoreMask == 1);
                }
            }
        }
    }
    catch (std::runtime_error &e)
    {
        CHECK(false);
    }
}

void TestAllowlist::WrapperUpdateGlobalsForDeviceId(const std::string &deviceId)
{
    UpdateGlobalsForDeviceId(deviceId);
}
