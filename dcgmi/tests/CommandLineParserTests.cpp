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

#include <CoutCapture.hpp>
#include <DcgmBuildInfo.hpp>
#include <DcgmStringHelpers.h>
#include <MockDcgmOps.h>
#include <TempFile.hpp>
#include <UniquePtrUtil.h>
#include <dcgm_errors.h>
#include <dcgm_structs.h>

#include <fstream>
#include <vector>

#include <Command.h>
#include <CommandLineParser.h>
#include <NvcmTCLAP.h>
#include <tclap/ArgException.h>

// -----------------------------------------------------------------
namespace
{
std::string const defaultPort       = std::to_string(DCGM_HE_PORT_NUMBER);
std::string const defaultSocketPath = DCGM_DEFAULT_SOCKET_PATH;
std::string const unixSocketPrefix  = DCGM_UNIX_SOCKET_PREFIX;

using ParserFunction = dcgmReturn_t (*)(int, char const *const *, DcgmNs::DcgmOpsAnyPtr);

/**
 * Test adapter that exposes protected command-line parser helpers.
 */
class TestCommandLineParser : public CommandLineParser
{
public:
    using CommandLineParser::CheckGroupIdArgument;
    using CommandLineParser::CheckTestDurationAndTimeout;
    using CommandLineParser::CommandExecutor;
    using CommandLineParser::ConcatenateParameters;
    using CommandLineParser::ExpandRange;
    using CommandLineParser::GetHostListVector;
    using CommandLineParser::HelperProcessWorkloadPowerProfileCommandLine;
    using CommandLineParser::m_commandExecutor;
    using CommandLineParser::NormalizeIpAddress;
    using CommandLineParser::NormalizeUnixSocketPath;
    using CommandLineParser::ProcessAdminCommandLine;
    using CommandLineParser::ProcessAndValidateMnDiagParams;
    using CommandLineParser::ProcessConfigCommandLine;
    using CommandLineParser::ProcessDiagCommandLine;
    using CommandLineParser::ProcessDmonCommandLine;
    using CommandLineParser::ProcessFieldGroupCommandLine;
    using CommandLineParser::ProcessGroupCommandLine;
    using CommandLineParser::ProcessHealthCommandLine;
    using CommandLineParser::ProcessModuleCommandLine;
    using CommandLineParser::ProcessNvlinkCommandLine;
    using CommandLineParser::ProcessPolicyCommandLine;
    using CommandLineParser::ProcessProfileCommandLine;
    using CommandLineParser::ProcessQueryCommandLine;
    using CommandLineParser::ProcessSettingsCommandLine;
    using CommandLineParser::ProcessTopoCommandLine;
    using CommandLineParser::ValidateClocksEventMask;
    using CommandLineParser::ValidateParameters;
};

class ScopedCommandExecutor
{
public:
    ScopedCommandExecutor()
        : m_previousExecutor(TestCommandLineParser::m_commandExecutor)
    {
        TestCommandLineParser::m_commandExecutor = [this](Command &) {
            m_callCount++;
            return DCGM_ST_OK;
        };
    }

    ~ScopedCommandExecutor()
    {
        TestCommandLineParser::m_commandExecutor = m_previousExecutor;
    }

    int CallCount() const
    {
        return m_callCount;
    }

private:
    TestCommandLineParser::CommandExecutor m_previousExecutor;
    int m_callCount = 0;
};

struct CommandLineParserTestFixture
{
    DcgmNs::MockDcgmOps mockDcgmOps;
    DcgmNs::DcgmOpsAnyPtr dcgmOps { &mockDcgmOps };

    void RequireParserThrows(ParserFunction parser, std::vector<char const *> const &argv)
    {
        REQUIRE_THROWS_AS(parser(static_cast<int>(argv.size()), argv.data(), dcgmOps), TCLAP::CmdLineParseException);
    }

    void RequireParserOkAndExecuted(ScopedCommandExecutor const &executeHook,
                                    ParserFunction parser,
                                    std::vector<char const *> const &argv,
                                    int expectedExecuteCalls = 1)
    {
        int const callCount = executeHook.CallCount();
        dcgmReturn_t result = DCGM_ST_GENERIC_ERROR;

        CHECK_NOTHROW(result = parser(static_cast<int>(argv.size()), argv.data(), dcgmOps));
        CHECK(result == DCGM_ST_OK);
        CHECK(executeHook.CallCount() == callCount + expectedExecuteCalls);
    }

    dcgmReturn_t ProcessCommandLine(std::vector<char const *> const &argv)
    {
        return TestCommandLineParser::ProcessCommandLine(static_cast<int>(argv.size()), argv.data(), dcgmOps);
    }
};
} //namespace

// -----------------------------------------------------------------
SCENARIO("CheckTestDurationAndTimeout")
{
    REQUIRE_NOTHROW(TestCommandLineParser::CheckTestDurationAndTimeout("diagnostic.test_duration=200", 300));
    REQUIRE_THROWS_AS(TestCommandLineParser::CheckTestDurationAndTimeout("diagnostic.test_duration=200", 200),
                      TCLAP::CmdLineParseException);
    REQUIRE_THROWS_AS(TestCommandLineParser::CheckTestDurationAndTimeout("diagnostic.test_duration=200", 100),
                      TCLAP::CmdLineParseException);
    REQUIRE_THROWS_AS(TestCommandLineParser::CheckTestDurationAndTimeout(
                          "diagnostic.test_duration=100;pulse_test.test_duration=100", 200),
                      TCLAP::CmdLineParseException);
    REQUIRE_THROWS_AS(TestCommandLineParser::CheckTestDurationAndTimeout(
                          "diagnostic.test_duration=100;pulse_test.test_duration=100", 50),
                      TCLAP::CmdLineParseException);
    REQUIRE_NOTHROW(TestCommandLineParser::CheckTestDurationAndTimeout(
        "diagnostic.test_duration=100;pulse_test.test_duration=100", 300));
    REQUIRE_THROWS_AS(
        TestCommandLineParser::CheckTestDurationAndTimeout(
            "diagnostic.test_duration=100;pulse_test.test_duration=100;targeted_stress.test_duration=100", 300),
        TCLAP::CmdLineParseException);
}

SCENARIO("MnDiag Host List Validation")
{
    SECTION("Valid host list formats")
    {
        // Test for host with default port
        REQUIRE(TestCommandLineParser::GetHostListVector("host1")
                == std::vector<std::string> { "host1:" + defaultPort + "=*" });
        REQUIRE(TestCommandLineParser::GetHostListVector("host1=0,1,2")
                == std::vector<std::string> { "host1:" + defaultPort + "=0,1,2" });
        REQUIRE(TestCommandLineParser::GetHostListVector("host1=0-2")
                == std::vector<std::string> { "host1:" + defaultPort + "=0,1,2" });
        REQUIRE(TestCommandLineParser::GetHostListVector("host1=2-20")
                == std::vector<std::string> { "host1:" + defaultPort
                                              + "=2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20" });
        REQUIRE(TestCommandLineParser::GetHostListVector("host1=0,1,2;host2=3,4,5")
                == std::vector<std::string> { "host1:" + defaultPort + "=0,1,2", "host2:" + defaultPort + "=3,4,5" });
        REQUIRE(TestCommandLineParser::GetHostListVector("host1=0-2;host2=3-5")
                == std::vector<std::string> { "host1:" + defaultPort + "=0,1,2", "host2:" + defaultPort + "=3,4,5" });
        REQUIRE(TestCommandLineParser::GetHostListVector("host1=0,1-3,4")
                == std::vector<std::string> { "host1:" + defaultPort + "=0,1,2,3,4" });
        REQUIRE(TestCommandLineParser::GetHostListVector("host1=0,0-3,4")
                == std::vector<std::string> { "host1:" + defaultPort + "=0,1,2,3,4" });
        REQUIRE(TestCommandLineParser::GetHostListVector("host1=0,3-6,6")
                == std::vector<std::string> { "host1:" + defaultPort + "=0,3,4,5,6" });
        REQUIRE(TestCommandLineParser::GetHostListVector("host1=0,0,4")
                == std::vector<std::string> { "host1:" + defaultPort + "=0,4" });
        REQUIRE(TestCommandLineParser::GetHostListVector("host1=0,1-3,1")
                == std::vector<std::string> { "host1:" + defaultPort + "=0,1,2,3" });

        // Test with host and explicit port specification
        REQUIRE(TestCommandLineParser::GetHostListVector("host1:1234=0,1,2")
                == std::vector<std::string> { "host1:1234=0,1,2" });
        REQUIRE(TestCommandLineParser::GetHostListVector("host1:8888=0-2")
                == std::vector<std::string> { "host1:8888=0,1,2" });
        REQUIRE(TestCommandLineParser::GetHostListVector("host1:5555=0,1,2;host2:6666=3,4,5")
                == std::vector<std::string> { "host1:5555=0,1,2", "host2:6666=3,4,5" });

        // Test with host and mixed port specification
        REQUIRE(TestCommandLineParser::GetHostListVector("host1:1234=0,1,2;host2=3,4,5")
                == std::vector<std::string> { "host1:1234=0,1,2", "host2:" + defaultPort + "=3,4,5" });

        // Test with multiple hosts with different ports
        REQUIRE(TestCommandLineParser::GetHostListVector("host1:1234=0,1;host1:5678=2,3;host3=0")
                == std::vector<std::string> { "host1:1234=0,1", "host1:5678=2,3", "host3:" + defaultPort + "=0" });

        // Test with host and default Unix socket paths
        REQUIRE(TestCommandLineParser::GetHostListVector("host1:" + unixSocketPrefix + "=0,1,2")
                == std::vector<std::string> { "host1:" + unixSocketPrefix + defaultSocketPath + "=0,1,2" });

        // Test with host and mixed Unix socket and TCP hosts
        REQUIRE(TestCommandLineParser::GetHostListVector("host1:" + unixSocketPrefix + "=0,1;host2=2,3")
                == std::vector<std::string> { "host1:" + unixSocketPrefix + defaultSocketPath + "=0,1",
                                              "host2:" + defaultPort + "=2,3" });

        // Test with multiple Unix socket paths
        REQUIRE(TestCommandLineParser::GetHostListVector("host1:" + unixSocketPrefix + "=0,1;host2:" + unixSocketPrefix
                                                         + "=2,3")
                == std::vector<std::string> { "host1:" + unixSocketPrefix + defaultSocketPath + "=0,1",
                                              "host2:" + unixSocketPrefix + defaultSocketPath + "=2,3" });
    }

    SECTION("Invalid host list formats")
    {
        REQUIRE_THROWS_AS(TestCommandLineParser::GetHostListVector("=0,1,2"), TCLAP::CmdLineParseException);
        REQUIRE_THROWS_AS(TestCommandLineParser::GetHostListVector("host1="), TCLAP::CmdLineParseException);
        REQUIRE_THROWS_AS(TestCommandLineParser::GetHostListVector("host1=a,b,c"), TCLAP::CmdLineParseException);
        REQUIRE_THROWS_AS(TestCommandLineParser::GetHostListVector("host1=2-1"), TCLAP::CmdLineParseException);
        REQUIRE_THROWS_AS(TestCommandLineParser::GetHostListVector("host1=0,1;host1=2,3"),
                          TCLAP::CmdLineParseException);
        REQUIRE_THROWS_AS(TestCommandLineParser::GetHostListVector("host1=0,1,2-"), TCLAP::CmdLineParseException);
        REQUIRE_THROWS_AS(TestCommandLineParser::GetHostListVector("host1=-0,1,2"), TCLAP::CmdLineParseException);
        REQUIRE_THROWS_AS(TestCommandLineParser::GetHostListVector("host1=0,1,-2"), TCLAP::CmdLineParseException);
        REQUIRE_THROWS_AS(TestCommandLineParser::GetHostListVector("host1=-0,1,2"), TCLAP::CmdLineParseException);
        REQUIRE_THROWS_AS(TestCommandLineParser::GetHostListVector("host1=0,1,2-"), TCLAP::CmdLineParseException);
        REQUIRE_THROWS_AS(TestCommandLineParser::GetHostListVector("host1=0,-1,2"), TCLAP::CmdLineParseException);
        REQUIRE_THROWS_AS(TestCommandLineParser::GetHostListVector("host1=0,1,-2"), TCLAP::CmdLineParseException);

        // Non numerical charactors
        REQUIRE_THROWS_AS(TestCommandLineParser::GetHostListVector("host1=0,a,-2"), TCLAP::CmdLineParseException);
        REQUIRE_THROWS_AS(TestCommandLineParser::GetHostListVector("host1=0,1a-20"), TCLAP::CmdLineParseException);

        // Very large string case
        std::string largeString(10000, '0');
        REQUIRE_THROWS_AS(TestCommandLineParser::GetHostListVector("host1=1234578" + largeString),
                          TCLAP::CmdLineParseException);

        // Invalid port specification
        REQUIRE_THROWS_AS(TestCommandLineParser::GetHostListVector("host1:=0,1,2"), TCLAP::CmdLineParseException);
        REQUIRE_THROWS_AS(TestCommandLineParser::GetHostListVector("host1:abc=0,1,2"), TCLAP::CmdLineParseException);

        // Invalid port range
        REQUIRE_THROWS_AS(TestCommandLineParser::GetHostListVector("host1:-1=0,1,2"), TCLAP::CmdLineParseException);
        REQUIRE_THROWS_AS(TestCommandLineParser::GetHostListVector("host1:65536=0,1,2"), TCLAP::CmdLineParseException);
        REQUIRE_THROWS_AS(TestCommandLineParser::GetHostListVector("host1:99999=0,1,2"), TCLAP::CmdLineParseException);

        // Duplicate hosts with different ports (should not throw as they're considered different hosts)
        REQUIRE_NOTHROW(TestCommandLineParser::GetHostListVector("host1:1234=0,1;host1:5678=2,3"));

        // Duplicate hosts with same ports
        REQUIRE_THROWS_AS(TestCommandLineParser::GetHostListVector("host1:1234=0,1;host1:1234=2,3"),
                          TCLAP::CmdLineParseException);

        // Invalid hostname containing only colons
        REQUIRE_THROWS_AS(TestCommandLineParser::GetHostListVector(":=0,1,2"), TCLAP::CmdLineParseException);
        REQUIRE_THROWS_AS(TestCommandLineParser::GetHostListVector(":::=0,1,2"), TCLAP::CmdLineParseException);

        // Invalid hostname with leading colons
        REQUIRE_THROWS_AS(TestCommandLineParser::GetHostListVector(":hostname=0,1,2"), TCLAP::CmdLineParseException);

        // Invalid Unix socket path formats
        REQUIRE_THROWS_AS(TestCommandLineParser::GetHostListVector("host1:unix//path=0,1,2"),
                          TCLAP::CmdLineParseException);
        REQUIRE_THROWS_AS(TestCommandLineParser::GetHostListVector("host1:unix:/path=0,1,2"),
                          TCLAP::CmdLineParseException);
        REQUIRE_THROWS_AS(TestCommandLineParser::GetHostListVector("host1:unix://path=0,1,2"),
                          TCLAP::CmdLineParseException);

        // Missing hostname in Unix socket path
        REQUIRE_THROWS_AS(TestCommandLineParser::GetHostListVector(unixSocketPrefix + "/tmp/socket=0,1,2"),
                          TCLAP::CmdLineParseException);

        // Invalid format with no colon before unix:
        REQUIRE_THROWS_AS(TestCommandLineParser::GetHostListVector("hostunix:/tmp/socket=0,1,2"),
                          TCLAP::CmdLineParseException);

        // Invalid format with empty hostname before colon
        REQUIRE_THROWS_AS(TestCommandLineParser::GetHostListVector(":" + unixSocketPrefix + "/tmp/socket=0,1,2"),
                          TCLAP::CmdLineParseException);

        // Invalid format with remainder not starting with unix:/
        REQUIRE_THROWS_AS(TestCommandLineParser::GetHostListVector("host1:unixpath/socket=0,1,2"),
                          TCLAP::CmdLineParseException);
        REQUIRE_THROWS_AS(TestCommandLineParser::GetHostListVector("host1:unix://socket=0,1,2"),
                          TCLAP::CmdLineParseException);

        // Duplicate Unix socket paths
        REQUIRE_THROWS_AS(TestCommandLineParser::GetHostListVector("host1:" + unixSocketPrefix
                                                                   + "/tmp/socket=0,1;host1:" + unixSocketPrefix
                                                                   + "/tmp/socket=2,3"),
                          TCLAP::CmdLineParseException);
    }
}

SCENARIO("MnDiag Parameter Validation")
{
    SECTION("Valid parameter formats")
    {
        REQUIRE_NOTHROW(TestCommandLineParser::ValidateParameters("test.param1=value1"));
        REQUIRE_NOTHROW(TestCommandLineParser::ValidateParameters("test.param1=value1;test.param2=value2"));
        REQUIRE_NOTHROW(TestCommandLineParser::ValidateParameters("test_name.parameter_name=parameter_value"));
    }

    SECTION("Parameter concatenation")
    {
        std::vector<std::string> params = { "test.param1=value1", "test.param2=value2" };
        REQUIRE(TestCommandLineParser::ConcatenateParameters(params) == "test.param1=value1;test.param2=value2");

        params = { "test.passthrough_args=arg1", "test.passthrough_args=arg2" };
        REQUIRE(TestCommandLineParser::ConcatenateParameters(params) == "test.passthrough_args=arg1 arg2");
    }

    SECTION("Test duration validation")
    {
        REQUIRE_NOTHROW(TestCommandLineParser::CheckTestDurationAndTimeout("test.test_duration=50", 100));
        REQUIRE_THROWS_AS(TestCommandLineParser::CheckTestDurationAndTimeout("test.test_duration=100", 50),
                          TCLAP::CmdLineParseException);
    }
}

TEST_CASE("CommandLineParser: ConcatenateParameters")
{
    SECTION("Basic")
    {
        std::vector<std::string> const params {
            "test1.param1=arg1;test1.param2=arg2",
            "test1.param3=arg3",
            "test2.param1=arg1",
        };
        std::string out = TestCommandLineParser::ConcatenateParameters(params);
        REQUIRE(out == "test1.param1=arg1;test1.param2=arg2;test1.param3=arg3;test2.param1=arg1");
    }

    SECTION("Missing '='")
    {
        std::vector<std::string> const params {
            "test1.param1:arg1",
        };
        REQUIRE_THROWS(TestCommandLineParser::ConcatenateParameters(params));
    }

    SECTION("Missing '.' is accepted")
    {
        // The diagnostic binary will reject this if the parameter name isn't a global parameter
        std::vector<std::string> const params {
            "genericMode=true",
            "diagnostic.test_duration=200",
        };
        std::string out = TestCommandLineParser::ConcatenateParameters(params);
        REQUIRE(out == "diagnostic.test_duration=200;genericMode=true");
    }

    SECTION("Unsupported multiple definitions")
    {
        std::vector<std::string> const params {
            "test1.param1=arg1",
            "test1.param1=arg2",
        };
        REQUIRE_THROWS(TestCommandLineParser::ConcatenateParameters(params));
    }

    SECTION("Supported multiple definitions")
    {
        std::vector<std::string> const params {
            "test1.passthrough_args=arg1",
            "test1.passthrough_args=arg2",
            "test1.normal_args=arg3",
            "test2.passthrough_args=arg4",
        };
        std::string out = TestCommandLineParser::ConcatenateParameters(params);
        REQUIRE(out == "test1.normal_args=arg3;test1.passthrough_args=arg1 arg2;test2.passthrough_args=arg4");
    }
}

TEST_CASE("CommandLineParser: ValidateParameters")
{
    SECTION("Basic")
    {
        std::string params = "diagnostic.test_duration=30;memtest.test_duration=10";
        REQUIRE_NOTHROW(TestCommandLineParser::ValidateParameters(params));
    }

    SECTION("Parameter length limit exceeded")
    {
        std::string params = "test1.param1=";
        std::string value(DCGM_MAX_TEST_PARMS_LEN_V2 - params.size(), '6');
        params += value;
        REQUIRE_THROWS(TestCommandLineParser::ValidateParameters(params));
    }
}

TEST_CASE("CommandLineParser: CheckGroupIdArgument")
{
    SECTION("Named built-in groups are accepted")
    {
        REQUIRE(TestCommandLineParser::CheckGroupIdArgument("g") == DCGM_GROUP_ALL_GPUS);
        REQUIRE(TestCommandLineParser::CheckGroupIdArgument("s") == DCGM_GROUP_ALL_NVSWITCHES);
        REQUIRE(TestCommandLineParser::CheckGroupIdArgument("i") == DCGM_GROUP_ALL_INSTANCES);
        REQUIRE(TestCommandLineParser::CheckGroupIdArgument("c") == DCGM_GROUP_ALL_COMPUTE_INSTANCES);
        REQUIRE(TestCommandLineParser::CheckGroupIdArgument("a") == DCGM_GROUP_ALL_ENTITIES);
    }

    SECTION("Numeric group ids are parsed")
    {
        REQUIRE(TestCommandLineParser::CheckGroupIdArgument("0") == 0);
        REQUIRE(TestCommandLineParser::CheckGroupIdArgument("17") == 17);
    }

    SECTION("Unsupported group ids throw")
    {
        REQUIRE_THROWS_AS(TestCommandLineParser::CheckGroupIdArgument("x"), TCLAP::CmdLineParseException);
        REQUIRE_THROWS_AS(TestCommandLineParser::CheckGroupIdArgument("-1"), TCLAP::CmdLineParseException);
    }
}

TEST_CASE("CommandLineParser: ValidateClocksEventMask")
{
    SECTION("Numeric masks accept valid bits and additive combinations")
    {
        REQUIRE_NOTHROW(TestCommandLineParser::ValidateClocksEventMask("0"));
        REQUIRE_NOTHROW(TestCommandLineParser::ValidateClocksEventMask("8"));
        REQUIRE_NOTHROW(TestCommandLineParser::ValidateClocksEventMask("232"));
    }

    SECTION("String masks accept known reason names case-insensitively")
    {
        REQUIRE_NOTHROW(TestCommandLineParser::ValidateClocksEventMask("hw_slowdown"));
        REQUIRE_NOTHROW(
            TestCommandLineParser::ValidateClocksEventMask("HW_SLOWDOWN,sw_thermal,hw_thermal,hw_power_brake"));
    }

    SECTION("Invalid masks throw")
    {
        REQUIRE_THROWS_AS(TestCommandLineParser::ValidateClocksEventMask("1"), TCLAP::CmdLineParseException);
        REQUIRE_THROWS_AS(TestCommandLineParser::ValidateClocksEventMask("8,hw_slowdown"),
                          TCLAP::CmdLineParseException);
        REQUIRE_THROWS_AS(TestCommandLineParser::ValidateClocksEventMask("bad_reason"), TCLAP::CmdLineParseException);
        REQUIRE_THROWS_AS(TestCommandLineParser::ValidateClocksEventMask(std::string(DCGM_CLOCKS_EVENT_MASK_LEN, '8')),
                          TCLAP::CmdLineParseException);
    }
}

TEST_CASE("CommandLineParser: Host normalization helpers")
{
    SECTION("NormalizeIpAddress appends the default port")
    {
        std::string host = "node-a";

        TestCommandLineParser::NormalizeIpAddress(host, host);

        REQUIRE(host == "node-a:" + defaultPort);
    }

    SECTION("NormalizeIpAddress accepts boundary ports")
    {
        std::string host = "node-a:0";
        REQUIRE_NOTHROW(TestCommandLineParser::NormalizeIpAddress(host, host));
        CHECK(host == "node-a:0");

        host = "node-a:65535";
        REQUIRE_NOTHROW(TestCommandLineParser::NormalizeIpAddress(host, host));
        CHECK(host == "node-a:65535");
    }

    SECTION("NormalizeUnixSocketPath appends the default socket when no path is provided")
    {
        std::string host = "node-a:" + unixSocketPrefix;

        TestCommandLineParser::NormalizeUnixSocketPath(host, host);

        REQUIRE(host == "node-a:" + unixSocketPrefix + defaultSocketPath);
    }

    SECTION("NormalizeUnixSocketPath keeps an explicit absolute socket path")
    {
        std::string host = "node-a:" + unixSocketPrefix + "/var/run/dcgm.sock";

        TestCommandLineParser::NormalizeUnixSocketPath(host, host);

        REQUIRE(host == "node-a:" + unixSocketPrefix + "/var/run/dcgm.sock");
    }

    SECTION("NormalizeUnixSocketPath rejects embedded colons in socket paths")
    {
        std::string host = "node-a:" + unixSocketPrefix + "/var/run:dcgm.sock";

        REQUIRE_THROWS_AS(TestCommandLineParser::NormalizeUnixSocketPath(host, host), TCLAP::CmdLineParseException);
    }

    SECTION("NormalizeUnixSocketPath rejects missing host and relative paths")
    {
        std::string missingHost = ":" + unixSocketPrefix + "/var/run/dcgm.sock";
        REQUIRE_THROWS_AS(TestCommandLineParser::NormalizeUnixSocketPath(missingHost, missingHost),
                          TCLAP::CmdLineParseException);

        std::string relativePath = "node-a:" + unixSocketPrefix + "relative.sock";
        REQUIRE_THROWS_AS(TestCommandLineParser::NormalizeUnixSocketPath(relativePath, relativePath),
                          TCLAP::CmdLineParseException);
    }

    SECTION("NormalizeIpAddress rejects malformed host and port values")
    {
        std::string emptyHost;
        REQUIRE_THROWS_AS(TestCommandLineParser::NormalizeIpAddress(emptyHost, emptyHost),
                          TCLAP::CmdLineParseException);

        std::string tooManyColons = "node-a:123:extra";
        REQUIRE_THROWS_AS(TestCommandLineParser::NormalizeIpAddress(tooManyColons, tooManyColons),
                          TCLAP::CmdLineParseException);

        std::string emptyPort = "node-a:";
        REQUIRE_THROWS_AS(TestCommandLineParser::NormalizeIpAddress(emptyPort, emptyPort),
                          TCLAP::CmdLineParseException);

        std::string outOfRangePort = "node-a:18446744073709551616";
        REQUIRE_THROWS_AS(TestCommandLineParser::NormalizeIpAddress(outOfRangePort, outOfRangePort),
                          TCLAP::CmdLineParseException);
    }
}

TEST_CASE("CommandLineParser: HelperProcessWorkloadPowerProfileCommandLine Valid Index")
{
    auto bit = 7, groupId = 2;
    dcgmWorkloadPowerProfile_t mWorkloadPowerProfile {};
    SECTION("Default action - append")
    {
        mWorkloadPowerProfile = TestCommandLineParser::HelperProcessWorkloadPowerProfileCommandLine(bit, 'a', groupId);
        REQUIRE(mWorkloadPowerProfile.action == DCGM_WORKLOAD_PROFILE_ACTION_SET);
    }
    SECTION("action - clear")
    {
        mWorkloadPowerProfile = TestCommandLineParser::HelperProcessWorkloadPowerProfileCommandLine(bit, 'c', groupId);
        REQUIRE(mWorkloadPowerProfile.action == DCGM_WORKLOAD_PROFILE_ACTION_CLEAR);
    }
    SECTION("action - overwrite")
    {
        mWorkloadPowerProfile = TestCommandLineParser::HelperProcessWorkloadPowerProfileCommandLine(bit, 'o', groupId);
        REQUIRE(mWorkloadPowerProfile.action == DCGM_WORKLOAD_PROFILE_ACTION_SET_AND_OVERWRITE);
    }

    REQUIRE(mWorkloadPowerProfile.version == dcgmWorkloadPowerProfile_version1);
    REQUIRE(mWorkloadPowerProfile.groupId == static_cast<unsigned int>(groupId));
    REQUIRE(mWorkloadPowerProfile.profileMask[bit / DCGM_POWER_PROFILE_MASK_BITS_PER_ELEM]
            == static_cast<unsigned int>(1 << (bit % DCGM_POWER_PROFILE_MASK_BITS_PER_ELEM)));
    // Verify that the other elements are 0
    for (int i = 0; i < DCGM_POWER_PROFILE_ARRAY_SIZE; i++)
    {
        if (i != bit / DCGM_POWER_PROFILE_MASK_BITS_PER_ELEM)
        {
            REQUIRE(mWorkloadPowerProfile.profileMask[i] == 0);
        }
    }
}

TEST_CASE("CommandLineParser: HelperProcessWorkloadPowerProfileCommandLine -1 Index")
{
    auto bit = -1, groupId = 2;
    dcgmWorkloadPowerProfile_t mWorkloadPowerProfile {};
    SECTION("Default action - append")
    {
        mWorkloadPowerProfile = TestCommandLineParser::HelperProcessWorkloadPowerProfileCommandLine(bit, 'a', groupId);
        REQUIRE(mWorkloadPowerProfile.action == DCGM_WORKLOAD_PROFILE_ACTION_CLEAR);
    }
    SECTION("action - clear")
    {
        mWorkloadPowerProfile = TestCommandLineParser::HelperProcessWorkloadPowerProfileCommandLine(bit, 'c', groupId);
        REQUIRE(mWorkloadPowerProfile.action == DCGM_WORKLOAD_PROFILE_ACTION_CLEAR);
    }
    SECTION("action - overwrite")
    {
        mWorkloadPowerProfile = TestCommandLineParser::HelperProcessWorkloadPowerProfileCommandLine(bit, 'o', groupId);
        REQUIRE(mWorkloadPowerProfile.action == DCGM_WORKLOAD_PROFILE_ACTION_CLEAR);
    }

    REQUIRE(mWorkloadPowerProfile.version == dcgmWorkloadPowerProfile_version1);
    REQUIRE(mWorkloadPowerProfile.groupId == static_cast<unsigned int>(groupId));
    // Verify that all elements are 0xFFFFFFFF
    for (int i = 0; i < DCGM_POWER_PROFILE_ARRAY_SIZE; i++)
    {
        REQUIRE(mWorkloadPowerProfile.profileMask[i] == 0xFFFFFFFF);
    }
}

TEST_CASE("CommandLineParser: HelperProcessWorkloadPowerProfileCommandLine Invalid Options")
{
    auto groupId = 2;
    SECTION("Invalid action")
    {
        auto bit = 7;
        REQUIRE_THROWS(TestCommandLineParser::HelperProcessWorkloadPowerProfileCommandLine(bit, 'x', groupId));
    }
    SECTION("Invalid bit")
    {
        auto bit = 453;
        REQUIRE_THROWS(TestCommandLineParser::HelperProcessWorkloadPowerProfileCommandLine(bit, 'a', groupId));
    }
}

SCENARIO("MnDiag Command Line Arguments - invalid arguments")
{
    auto drmnd = MakeUniqueZero<dcgmRunMnDiag_v2>();
    std::string hostEngineAddressValue;
    bool hostAddressWasOverridden;
    bool jsonOutput;

    SECTION("Invalid Unix socket path in hostList")
    {
        std::string hostList = "host1:" + unixSocketPrefix + "tmp/socket=0,1,2";
        char const *argv[]   = { "dcgmi", "mndiag", "--hostList", hostList.c_str() };
        int argc             = sizeof(argv) / sizeof(argv[0]);

        REQUIRE_THROWS_AS(TestCommandLineParser::ProcessAndValidateMnDiagParams(
                              argc, argv, *drmnd, hostEngineAddressValue, hostAddressWasOverridden, jsonOutput),
                          TCLAP::CmdLineParseException);
    }

    SECTION("Missing hostname in Unix socket path")
    {
        std::string hostList = unixSocketPrefix + "/tmp/socket=0,1,2";
        char const *argv[]   = { "dcgmi", "mndiag", "--hostList", hostList.c_str() };
        int argc             = sizeof(argv) / sizeof(argv[0]);

        REQUIRE_THROWS_AS(TestCommandLineParser::ProcessAndValidateMnDiagParams(
                              argc, argv, *drmnd, hostEngineAddressValue, hostAddressWasOverridden, jsonOutput),
                          TCLAP::CmdLineParseException);
    }

    SECTION("Invalid run level")
    {
        char const *argv[] = { "dcgmi", "mndiag", "--run", "-1", "--hostList", "host1=0,1,2" };
        int argc           = sizeof(argv) / sizeof(argv[0]);

        REQUIRE_THROWS_AS(TestCommandLineParser::ProcessAndValidateMnDiagParams(
                              argc, argv, *drmnd, hostEngineAddressValue, hostAddressWasOverridden, jsonOutput),
                          TCLAP::CmdLineParseException);
    }

    SECTION("Invalid head node address")
    {
        char const *argv[]
            = { "dcgmi", "mndiag", "--run", "mnubergemm", "--hostEngineAddress", "", "--hostList", "host1=0,1,2" };
        int argc = sizeof(argv) / sizeof(argv[0]);
        REQUIRE_THROWS_AS(TestCommandLineParser::ProcessAndValidateMnDiagParams(
                              argc, argv, *drmnd, hostEngineAddressValue, hostAddressWasOverridden, jsonOutput),
                          TCLAP::CmdLineParseException);
    }

    SECTION("Invalid host list")
    {
        char const *argv[] = { "dcgmi", "mndiag", "--run", "mnubergemm", "--hostList", "" };
        int argc           = sizeof(argv) / sizeof(argv[0]);
        REQUIRE_THROWS_AS(TestCommandLineParser::ProcessAndValidateMnDiagParams(
                              argc, argv, *drmnd, hostEngineAddressValue, hostAddressWasOverridden, jsonOutput),
                          TCLAP::CmdLineParseException);
    }

    SECTION("Invalid parameters")
    {
        char const *argv[] = { "dcgmi",      "mndiag",      "--run",        "mnubergemm",
                               "--hostList", "host1=0,1,2", "--parameters", "invalid_param" };
        int argc           = sizeof(argv) / sizeof(argv[0]);
        REQUIRE_THROWS_AS(TestCommandLineParser::ProcessAndValidateMnDiagParams(
                              argc, argv, *drmnd, hostEngineAddressValue, hostAddressWasOverridden, jsonOutput),
                          TCLAP::CmdLineParseException);
    }

    SECTION("Invalid GPU ID format in hostlist")
    {
        char const *argv[] = { "dcgmi", "mndiag", "--run", "mnubergemm", "--hostList", "localhost=invalid_gpu_id" };
        int argc           = sizeof(argv) / sizeof(argv[0]);
        REQUIRE_THROWS_AS(TestCommandLineParser::ProcessAndValidateMnDiagParams(
                              argc, argv, *drmnd, hostEngineAddressValue, hostAddressWasOverridden, jsonOutput),
                          TCLAP::CmdLineParseException);
    }

    SECTION("Very large hostlist with many hosts (exceeding buffer capacity)")
    {
        // Create a very large hostlist string with 301 hosts, each with 1 GPU
        std::string largeHostlist = "";
        for (int i = 0; i < DCGM_MNDIAG_MAX_NUM_HOSTS + 1; i++)
        {
            if (i > 0)
                largeHostlist += ";";
            largeHostlist += "host" + std::to_string(i) + "=0";
        }

        char const *argv[] = { "dcgmi", "mndiag", "--run", "mnubergemm", "--hostList", largeHostlist.c_str() };
        int argc           = sizeof(argv) / sizeof(argv[0]);

        // This should throw an exception due to the hostlist being too large
        REQUIRE_THROWS_AS(TestCommandLineParser::ProcessAndValidateMnDiagParams(
                              argc, argv, *drmnd, hostEngineAddressValue, hostAddressWasOverridden, jsonOutput),
                          TCLAP::CmdLineParseException);
    }

    SECTION("Too many parameters (exceeding maximum count)")
    {
        // Create a parameters string with more than 100 parameters
        std::string largeParameters = "";
        for (int i = 0; i < DCGM_MAX_TEST_PARMS + 1; i++)
        { // 101 parameters, exceeding the max of 100
            if (i > 0)
                largeParameters += ";";
            largeParameters += "test.param" + std::to_string(i) + "=value" + std::to_string(i);
        }

        char const *argv[] = { "dcgmi",      "mndiag",      "--run",        "mnubergemm",
                               "--hostList", "host1=0,1,2", "--parameters", largeParameters.c_str() };
        int argc           = sizeof(argv) / sizeof(argv[0]);

        // This should throw an exception due to too many parameters
        REQUIRE_THROWS_AS(TestCommandLineParser::ProcessAndValidateMnDiagParams(
                              argc, argv, *drmnd, hostEngineAddressValue, hostAddressWasOverridden, jsonOutput),
                          TCLAP::CmdLineParseException);
    }

    SECTION("Hostlist with single host entry exceeding buffer capacity")
    {
        // Create a hostlist with a single host but with a very long hostname (exceeding 256 chars)
        std::string longHostname = "host";
        // Add characters to make the hostname very long
        for (int i = 0; i < DCGM_MAX_STR_LENGTH + 1; i++)
        { // "host" + 252 chars + "=0" = 259 chars
            longHostname += "x";
        }
        std::string largeHostlist = longHostname + "=0";

        char const *argv[] = { "dcgmi", "mndiag", "--run", "mnubergemm", "--hostList", largeHostlist.c_str() };
        int argc           = sizeof(argv) / sizeof(argv[0]);

        // This should throw an exception due to the hostname being too long
        REQUIRE_THROWS_AS(TestCommandLineParser::ProcessAndValidateMnDiagParams(
                              argc, argv, *drmnd, hostEngineAddressValue, hostAddressWasOverridden, jsonOutput),
                          TCLAP::CmdLineParseException);
    }

    SECTION("Parameter with single entry exceeding buffer capacity")
    {
        // Create a parameter with a very long value (exceeding 256 chars)
        std::string paramName  = "test.param=";
        std::string paramValue = "";
        // Add characters to make the parameter value very long
        for (int i = 0; i < DCGM_MAX_TEST_PARMS_LEN_V2 + 1; i++)
        {
            paramValue += "v";
        }
        std::string largeParameter = paramName + paramValue;

        char const *argv[] = { "dcgmi",      "mndiag",  "--run",        "mnubergemm",
                               "--hostList", "host1=0", "--parameters", largeParameter.c_str() };
        int argc           = sizeof(argv) / sizeof(argv[0]);

        // This should throw an exception due to the parameter being too long
        REQUIRE_THROWS_AS(TestCommandLineParser::ProcessAndValidateMnDiagParams(
                              argc, argv, *drmnd, hostEngineAddressValue, hostAddressWasOverridden, jsonOutput),
                          TCLAP::CmdLineParseException);
    }

    SECTION("Invalid format with no colon before unix:")
    {
        char const *argv[] = { "dcgmi", "mndiag", "--hostList", "hostunix:/tmp/socket=0,1,2" };
        int argc           = sizeof(argv) / sizeof(argv[0]);

        REQUIRE_THROWS_AS(TestCommandLineParser::ProcessAndValidateMnDiagParams(
                              argc, argv, *drmnd, hostEngineAddressValue, hostAddressWasOverridden, jsonOutput),
                          TCLAP::CmdLineParseException);
    }

    SECTION("Invalid format with remainder not starting with unix:/")
    {
        char const *argv[] = { "dcgmi", "mndiag", "--hostList", "host1:unixpath/socket=0,1,2" };
        int argc           = sizeof(argv) / sizeof(argv[0]);

        REQUIRE_THROWS_AS(TestCommandLineParser::ProcessAndValidateMnDiagParams(
                              argc, argv, *drmnd, hostEngineAddressValue, hostAddressWasOverridden, jsonOutput),
                          TCLAP::CmdLineParseException);
    }

    SECTION("Invalid local socket path not starting with /")
    {
        char const *argv[] = { "dcgmi", "mndiag", "--hostList", "host1:unix:/remote/socket:local_socket=0,1,2" };
        int argc           = sizeof(argv) / sizeof(argv[0]);

        REQUIRE_THROWS_AS(TestCommandLineParser::ProcessAndValidateMnDiagParams(
                              argc, argv, *drmnd, hostEngineAddressValue, hostAddressWasOverridden, jsonOutput),
                          TCLAP::CmdLineParseException);
    }
}

TEST_CASE("CommandLineParser: ExpandRange")
{
    SECTION("Valid range expansion")
    {
        REQUIRE(TestCommandLineParser::ExpandRange("0-2") == "0,1,2");
        REQUIRE(TestCommandLineParser::ExpandRange("5-5") == "5");
        REQUIRE(TestCommandLineParser::ExpandRange("3-6") == "3,4,5,6");
    }

    SECTION("Invalid range format")
    {
        REQUIRE_THROWS_AS(TestCommandLineParser::ExpandRange("2-1"), TCLAP::CmdLineParseException);
        REQUIRE_THROWS_AS(TestCommandLineParser::ExpandRange("a-b"), TCLAP::CmdLineParseException);
        REQUIRE_THROWS_AS(TestCommandLineParser::ExpandRange("1-"), TCLAP::CmdLineParseException);
        REQUIRE_THROWS_AS(TestCommandLineParser::ExpandRange("-1"), TCLAP::CmdLineParseException);
        REQUIRE_THROWS_AS(TestCommandLineParser::ExpandRange("2a-26"), TCLAP::CmdLineParseException);

        try
        {
            TestCommandLineParser::ExpandRange("a-b");
            FAIL("expected exception");
        }
        catch (TCLAP::CmdLineParseException const &ex)
        {
            CHECK_THAT(ex.what(), Catch::Matchers::ContainsSubstring("invalid range format"));
        }
    }

    SECTION("Oversize range reports cardinality, not format")
    {
        try
        {
            // DCGM_MAX_NUM_DEVICES is 32; inclusive 0..32 is 33 values
            TestCommandLineParser::ExpandRange("0-32");
            FAIL("expected exception");
        }
        catch (TCLAP::CmdLineParseException const &ex)
        {
            CHECK_THAT(ex.what(), Catch::Matchers::ContainsSubstring("exceeds maximum"));
            CHECK_THAT(ex.what(), !Catch::Matchers::ContainsSubstring("invalid range format"));
        }
    }

    SECTION("Exactly at device max expands")
    {
        // Inclusive 0..31 == DCGM_MAX_NUM_DEVICES
        REQUIRE(TestCommandLineParser::ExpandRange("0-31")
                == "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31");
    }

    SECTION("Single number")
    {
        REQUIRE(TestCommandLineParser::ExpandRange("7") == "7");
    }
}

SCENARIO("MnDiag Command Line Arguments - Valid Cases")
{
    auto drmnd = MakeUniqueZero<dcgmRunMnDiag_v2>();
    std::string hostEngineAddressValue;
    bool hostAddressWasOverridden;
    bool jsonOutput;

    SECTION("Required arguments validation defaults to mnubergemm")
    {
        char const *argv[] = { "dcgmi", "mndiag", "--hostList", "host1=0,1,2" };
        int argc           = sizeof(argv) / sizeof(argv[0]);

        REQUIRE_NOTHROW(TestCommandLineParser::ProcessAndValidateMnDiagParams(
            argc, argv, *drmnd, hostEngineAddressValue, hostAddressWasOverridden, jsonOutput));
        CHECK(std::string_view(drmnd->testName) == "mnubergemm");
    }

    SECTION("Required arguments validation for mnubergemm")
    {
        char const *argv[] = { "dcgmi", "mndiag", "--run", "mnubergemm", "--hostList", "host1=0,1,2" };
        int argc           = sizeof(argv) / sizeof(argv[0]);

        REQUIRE_NOTHROW(TestCommandLineParser::ProcessAndValidateMnDiagParams(
            argc, argv, *drmnd, hostEngineAddressValue, hostAddressWasOverridden, jsonOutput));
        CHECK(std::string_view(drmnd->testName) == "mnubergemm");
    }

    SECTION("Required arguments validation for mnnvbandwidth")
    {
        char const *argv[] = { "dcgmi", "mndiag", "--run", "mnnvbandwidth", "--hostList", "host1=0,1,2" };
        int argc           = sizeof(argv) / sizeof(argv[0]);

        REQUIRE_NOTHROW(TestCommandLineParser::ProcessAndValidateMnDiagParams(
            argc, argv, *drmnd, hostEngineAddressValue, hostAddressWasOverridden, jsonOutput));
        CHECK(std::string_view(drmnd->testName) == "mnnvbandwidth");
    }

    SECTION("Unix socket path in hostList")
    {
        std::string hostList = "host1:" + unixSocketPrefix + defaultSocketPath + "=0,1,2";
        char const *argv[]   = { "dcgmi", "mndiag", "--hostList", hostList.c_str() };
        int argc             = sizeof(argv) / sizeof(argv[0]);

        REQUIRE_NOTHROW(TestCommandLineParser::ProcessAndValidateMnDiagParams(
            argc, argv, *drmnd, hostEngineAddressValue, hostAddressWasOverridden, jsonOutput));
    }

    SECTION("Remote Unix socket path in hostList")
    {
        std::string hostList = "remote:" + unixSocketPrefix + "/remote/socket=0,1,2";
        char const *argv[]   = { "dcgmi", "mndiag", "--hostList", hostList.c_str() };
        int argc             = sizeof(argv) / sizeof(argv[0]);

        REQUIRE_NOTHROW(TestCommandLineParser::ProcessAndValidateMnDiagParams(
            argc, argv, *drmnd, hostEngineAddressValue, hostAddressWasOverridden, jsonOutput));
    }

    SECTION("Mixed Unix socket and TCP hosts in hostList")
    {
        std::string hostList = "host1:" + unixSocketPrefix + defaultSocketPath + "=0,1;host2=2,3";
        char const *argv[]   = { "dcgmi", "mndiag", "--hostList", hostList.c_str() };
        int argc             = sizeof(argv) / sizeof(argv[0]);

        REQUIRE_NOTHROW(TestCommandLineParser::ProcessAndValidateMnDiagParams(
            argc, argv, *drmnd, hostEngineAddressValue, hostAddressWasOverridden, jsonOutput));
    }

    SECTION("Optional parameters with matching test prefix")
    {
        char const *argv[] = { "dcgmi",      "mndiag",      "--run",        "mnubergemm",
                               "--hostList", "host1=0,1,2", "--parameters", "mnubergemm.param=value" };
        int argc           = sizeof(argv) / sizeof(argv[0]);

        REQUIRE_NOTHROW(TestCommandLineParser::ProcessAndValidateMnDiagParams(
            argc, argv, *drmnd, hostEngineAddressValue, hostAddressWasOverridden, jsonOutput));
    }

    SECTION("Open MPI parameters are accepted for every test")
    {
        std::string const testName = GENERATE(as<std::string> {}, "mnubergemm", "nvloom", "mnnvbandwidth");
        CAPTURE(testName);

        char const *argv[] = { "dcgmi",      "mndiag",      "--run",        testName.c_str(),
                               "--hostList", "host1=0,1,2", "--parameters", "openmpi.detect_interfaces=0" };
        int argc           = sizeof(argv) / sizeof(argv[0]);

        REQUIRE_NOTHROW(TestCommandLineParser::ProcessAndValidateMnDiagParams(
            argc, argv, *drmnd, hostEngineAddressValue, hostAddressWasOverridden, jsonOutput));
        CHECK(std::string_view(drmnd->testName) == testName);
        CHECK(std::string_view(drmnd->testParms[0]) == "openmpi.detect_interfaces=0");
    }

    SECTION("Invalid parameter prefix error lists both accepted prefixes")
    {
        char const *argv[] = { "dcgmi",      "mndiag",      "--run",        "mnnvbandwidth",
                               "--hostList", "host1=0,1,2", "--parameters", "openmpii.detect_interface=0" };
        int argc           = sizeof(argv) / sizeof(argv[0]);

        try
        {
            TestCommandLineParser::ProcessAndValidateMnDiagParams(
                argc, argv, *drmnd, hostEngineAddressValue, hostAddressWasOverridden, jsonOutput);
            FAIL("Expected an invalid parameter prefix to be rejected");
        }
        catch (TCLAP::CmdLineParseException const &exception)
        {
            CHECK(exception.error()
                  == "Parameter 'openmpii.detect_interface=0' must use either the 'mnnvbandwidth.' prefix or the "
                     "'openmpi.' prefix");
        }
    }

    SECTION("Parameters for another mndiag test are rejected")
    {
        char const *argv[] = { "dcgmi",      "mndiag",      "--run",        "nvloom",
                               "--hostList", "host1=0,1,2", "--parameters", "mnubergemm.time_to_run=30" };
        int argc           = sizeof(argv) / sizeof(argv[0]);

        REQUIRE_THROWS_AS(TestCommandLineParser::ProcessAndValidateMnDiagParams(
                              argc, argv, *drmnd, hostEngineAddressValue, hostAddressWasOverridden, jsonOutput),
                          TCLAP::CmdLineParseException);
    }

    SECTION("JSON output flag parsing")
    {
        char const *argv[] = { "dcgmi", "mndiag", "--hostList", "host1=0,1,2", "--json" };
        int argc           = sizeof(argv) / sizeof(argv[0]);

        REQUIRE_NOTHROW(TestCommandLineParser::ProcessAndValidateMnDiagParams(
            argc, argv, *drmnd, hostEngineAddressValue, hostAddressWasOverridden, jsonOutput));
        REQUIRE(jsonOutput == true);
    }

    SECTION("JSON output flag not specified defaults to false")
    {
        char const *argv[] = { "dcgmi", "mndiag", "--hostList", "host1=0,1,2" };
        int argc           = sizeof(argv) / sizeof(argv[0]);

        REQUIRE_NOTHROW(TestCommandLineParser::ProcessAndValidateMnDiagParams(
            argc, argv, *drmnd, hostEngineAddressValue, hostAddressWasOverridden, jsonOutput));
        REQUIRE(jsonOutput == false);
    }

    SECTION("Short form of JSON flag (-j)")
    {
        char const *argv[] = { "dcgmi", "mndiag", "--hostList", "host1=0,1,2", "-j" };
        int argc           = sizeof(argv) / sizeof(argv[0]);

        REQUIRE_NOTHROW(TestCommandLineParser::ProcessAndValidateMnDiagParams(
            argc, argv, *drmnd, hostEngineAddressValue, hostAddressWasOverridden, jsonOutput));
        REQUIRE(jsonOutput == true);
    }

    SECTION("Neither --hostList nor --hostfile provided")
    {
        char const *argv[] = { "dcgmi", "mndiag", "--run", "mnubergemm" };
        int argc           = sizeof(argv) / sizeof(argv[0]);

        REQUIRE_THROWS_AS(TestCommandLineParser::ProcessAndValidateMnDiagParams(
                              argc, argv, *drmnd, hostEngineAddressValue, hostAddressWasOverridden, jsonOutput),
                          TCLAP::CmdLineParseException);
    }

    SECTION("Both --hostList and --hostfile provided")
    {
        auto hostFile = TempFile::Random("host1=0,1,2\n", "dcgm-test-hostfile-both-");

        char const *argv[] = { "dcgmi", "mndiag", "--hostList", "host1=0,1,2", "--hostfile", hostFile.Path().c_str() };
        int argc           = sizeof(argv) / sizeof(argv[0]);

        REQUIRE_THROWS_AS(TestCommandLineParser::ProcessAndValidateMnDiagParams(
                              argc, argv, *drmnd, hostEngineAddressValue, hostAddressWasOverridden, jsonOutput),
                          TCLAP::CmdLineParseException);
    }

    SECTION("Non-existent hostfile path")
    {
        char const *argv[] = { "dcgmi", "mndiag", "--hostfile", "/nonexistent/path/to/hostfile.txt" };
        int argc           = sizeof(argv) / sizeof(argv[0]);

        REQUIRE_THROWS_AS(TestCommandLineParser::ProcessAndValidateMnDiagParams(
                              argc, argv, *drmnd, hostEngineAddressValue, hostAddressWasOverridden, jsonOutput),
                          TCLAP::CmdLineParseException);
    }
}

SCENARIO("MnDiag Command Line Arguments - hostfile")
{
    auto drmnd = MakeUniqueZero<dcgmRunMnDiag_v2>();
    std::string hostEngineAddressValue;
    bool hostAddressWasOverridden;
    bool jsonOutput;

    SECTION("Valid hostfile with multiple hosts")
    {
        auto hostFile = TempFile::Random("host1=0,1,2\nhost2=3,4\n", "dcgm-test-hostfile-valid-");

        char const *argv[] = { "dcgmi", "mndiag", "--hostfile", hostFile.Path().c_str() };
        int argc           = sizeof(argv) / sizeof(argv[0]);

        REQUIRE_NOTHROW(TestCommandLineParser::ProcessAndValidateMnDiagParams(
            argc, argv, *drmnd, hostEngineAddressValue, hostAddressWasOverridden, jsonOutput));

        CHECK(std::string_view(drmnd->hostList[0]) != "");
        CHECK(std::string_view(drmnd->hostList[1]) != "");
    }

    SECTION("Hostfile with comments and blank lines")
    {
        auto hostFile = TempFile::Random("# This is a comment\n\n  \nhost1=0,1\n# Another comment\nhost2=2,3\n\n",
                                         "dcgm-test-hostfile-comments-");

        char const *argv[] = { "dcgmi", "mndiag", "--hostfile", hostFile.Path().c_str() };
        int argc           = sizeof(argv) / sizeof(argv[0]);

        REQUIRE_NOTHROW(TestCommandLineParser::ProcessAndValidateMnDiagParams(
            argc, argv, *drmnd, hostEngineAddressValue, hostAddressWasOverridden, jsonOutput));

        CHECK(std::string_view(drmnd->hostList[0]) != "");
        CHECK(std::string_view(drmnd->hostList[1]) != "");
        CHECK(std::string_view(drmnd->hostList[2]) == "");
    }

    SECTION("Empty hostfile path throws")
    {
        char const *argv[] = { "dcgmi", "mndiag", "--hostfile", "" };
        int argc           = sizeof(argv) / sizeof(argv[0]);

        REQUIRE_THROWS_AS(TestCommandLineParser::ProcessAndValidateMnDiagParams(
                              argc, argv, *drmnd, hostEngineAddressValue, hostAddressWasOverridden, jsonOutput),
                          TCLAP::CmdLineParseException);
    }

    SECTION("Empty hostfile")
    {
        auto hostFile = TempFile::Random("", "dcgm-test-hostfile-empty-");

        char const *argv[] = { "dcgmi", "mndiag", "--hostfile", hostFile.Path().c_str() };
        int argc           = sizeof(argv) / sizeof(argv[0]);

        REQUIRE_THROWS_AS(TestCommandLineParser::ProcessAndValidateMnDiagParams(
                              argc, argv, *drmnd, hostEngineAddressValue, hostAddressWasOverridden, jsonOutput),
                          TCLAP::CmdLineParseException);
    }

    SECTION("Hostfile containing only comments")
    {
        auto hostFile = TempFile::Random("# comment 1\n# comment 2\n", "dcgm-test-hostfile-onlycomments-");

        char const *argv[] = { "dcgmi", "mndiag", "--hostfile", hostFile.Path().c_str() };
        int argc           = sizeof(argv) / sizeof(argv[0]);

        REQUIRE_THROWS_AS(TestCommandLineParser::ProcessAndValidateMnDiagParams(
                              argc, argv, *drmnd, hostEngineAddressValue, hostAddressWasOverridden, jsonOutput),
                          TCLAP::CmdLineParseException);
    }

    SECTION("Hostfile with parameters and json flags")
    {
        auto hostFile = TempFile::Random("host1=0,1,2\n", "dcgm-test-hostfile-flags-");

        char const *argv[] = {
            "dcgmi", "mndiag", "--hostfile", hostFile.Path().c_str(), "--json", "--parameters", "mnubergemm.param=value"
        };
        int argc = sizeof(argv) / sizeof(argv[0]);

        REQUIRE_NOTHROW(TestCommandLineParser::ProcessAndValidateMnDiagParams(
            argc, argv, *drmnd, hostEngineAddressValue, hostAddressWasOverridden, jsonOutput));
        CHECK(jsonOutput == true);
    }
}

TEST_CASE_METHOD(CommandLineParserTestFixture, "CommandLineParser: ProcessCommandLine rejects invalid subsystem")
{
    GIVEN("a dcgmi command with an unknown subsystem")
    {
        std::vector<char const *> argv { "dcgmi", "not-a-subsystem" };

        WHEN("the command line is processed")
        {
            CHECK(ProcessCommandLine(argv) == DCGM_ST_BADPARAM);
        }
    }
}

SCENARIO_METHOD(CommandLineParserTestFixture, "CommandLineParser: profile help uses corrected spelling")
{
    GIVEN("a request for profile help")
    {
        std::vector<char const *> argv { "dcgmi", "profile", "--help" };
        CoutCapture capture;

        WHEN("the command line is processed")
        {
            CHECK_THROWS_AS(
                TestCommandLineParser::ProcessProfileCommandLine(static_cast<int>(argv.size()), argv.data(), dcgmOps),
                TCLAP::ExitException);
            std::string const output = capture.str();

            THEN("the entity list description is spelled correctly")
            {
                CHECK(output.find("Comma-separated list of entity IDs") != std::string::npos);
            }
        }
    }
}

TEST_CASE_METHOD(CommandLineParserTestFixture,
                 "CommandLineParser: ProcessQueryCommandLine invalid argument combinations")
{
    GIVEN("query command line validation")
    {
        SECTION("Negative GPU id")
        {
            std::vector<char const *> argv { "dcgmi", "discovery", "--info", "a", "--gpuid", "-1" };

            WHEN("the query command is processed")
            {
                RequireParserThrows(TestCommandLineParser::ProcessQueryCommandLine, argv);
            }
        }

        SECTION("Group and GPU are mutually exclusive")
        {
            std::vector<char const *> argv { "dcgmi", "discovery", "--info", "a", "--group", "1", "--gpuid", "2" };

            WHEN("the query command is processed")
            {
                RequireParserThrows(TestCommandLineParser::ProcessQueryCommandLine, argv);
            }
        }

        SECTION("Group and CPU are mutually exclusive")
        {
            std::vector<char const *> argv { "dcgmi", "discovery", "--info", "a", "--group", "1", "--cpuid", "2" };

            WHEN("the query command is processed")
            {
                RequireParserThrows(TestCommandLineParser::ProcessQueryCommandLine, argv);
            }
        }

        SECTION("Compute hierarchy must be used alone")
        {
            std::vector<char const *> argv { "dcgmi", "discovery", "--compute-hierarchy", "--gpuid", "2" };

            WHEN("the query command is processed")
            {
                RequireParserThrows(TestCommandLineParser::ProcessQueryCommandLine, argv);
            }
        }
    }
}

TEST_CASE_METHOD(CommandLineParserTestFixture, "CommandLineParser: ProcessQueryCommandLine valid execution branches")
{
    ScopedCommandExecutor executeHook;

    GIVEN("discovery command line actions that parse successfully")
    {
        SECTION("List branch")
        {
            std::vector<char const *> argv { "dcgmi", "discovery", "--list", "--all" };
            RequireParserOkAndExecuted(executeHook, TestCommandLineParser::ProcessQueryCommandLine, argv);
        }

        SECTION("Compute hierarchy branch")
        {
            std::vector<char const *> argv { "dcgmi", "discovery", "--compute-hierarchy" };
            RequireParserOkAndExecuted(executeHook, TestCommandLineParser::ProcessQueryCommandLine, argv);
        }

        SECTION("GPU info branch")
        {
            std::vector<char const *> argv { "dcgmi", "discovery", "--info", "aptcw", "--gpuid", "0" };
            RequireParserOkAndExecuted(executeHook, TestCommandLineParser::ProcessQueryCommandLine, argv);
        }

        SECTION("Group info branch")
        {
            std::vector<char const *> argv { "dcgmi", "discovery", "--info", "a", "--group", "0", "--verbose" };
            RequireParserOkAndExecuted(executeHook, TestCommandLineParser::ProcessQueryCommandLine, argv);
        }

        SECTION("CPU info branch")
        {
            std::vector<char const *> argv { "dcgmi", "discovery", "--info", "a", "--cpuid", "0" };
            RequireParserOkAndExecuted(executeHook, TestCommandLineParser::ProcessQueryCommandLine, argv);
        }

        SECTION("Default group info branch")
        {
            std::vector<char const *> argv { "dcgmi", "discovery", "--info", "a" };
            RequireParserOkAndExecuted(executeHook, TestCommandLineParser::ProcessQueryCommandLine, argv);
        }
    }
}

TEST_CASE_METHOD(CommandLineParserTestFixture,
                 "CommandLineParser: ProcessPolicyCommandLine invalid argument combinations")
{
    GIVEN("policy command line validation")
    {
        SECTION("Negative max pages")
        {
            std::vector<char const *> argv { "dcgmi", "policy", "--set", "0,1", "--maxpages", "-1" };

            WHEN("the policy command is processed")
            {
                RequireParserThrows(TestCommandLineParser::ProcessPolicyCommandLine, argv);
            }
        }

        SECTION("Set requires at least one condition")
        {
            std::vector<char const *> argv { "dcgmi", "policy", "--set", "0,1" };

            WHEN("the policy command is processed")
            {
                RequireParserThrows(TestCommandLineParser::ProcessPolicyCommandLine, argv);
            }
        }

        SECTION("Set action and validation must be CSV")
        {
            std::vector<char const *> argv { "dcgmi", "policy", "--set", "01", "--eccerrors" };

            WHEN("the policy command is processed")
            {
                RequireParserThrows(TestCommandLineParser::ProcessPolicyCommandLine, argv);
            }
        }

        SECTION("Set action must be 0 or 1")
        {
            std::vector<char const *> argv { "dcgmi", "policy", "--set", "2,1", "--eccerrors" };

            WHEN("the policy command is processed")
            {
                RequireParserThrows(TestCommandLineParser::ProcessPolicyCommandLine, argv);
            }
        }

        SECTION("Set validation must be between 0 and 3")
        {
            std::vector<char const *> argv { "dcgmi", "policy", "--set", "1,4", "--eccerrors" };

            WHEN("the policy command is processed")
            {
                RequireParserThrows(TestCommandLineParser::ProcessPolicyCommandLine, argv);
            }
        }

        SECTION("Verbose is only valid with get")
        {
            std::vector<char const *> argv { "dcgmi", "policy", "--set", "1,1", "--eccerrors", "--verbose" };

            WHEN("the policy command is processed")
            {
                RequireParserThrows(TestCommandLineParser::ProcessPolicyCommandLine, argv);
            }
        }

        SECTION("Field policy create requires a threshold for threshold operators")
        {
            std::vector<char const *> argv { "dcgmi",  "policy",      "--create",   "--field", "150",
                                             "--name", "temperature", "--operator", ">" };

            WHEN("the policy command is processed")
            {
                RequireParserThrows(TestCommandLineParser::ProcessPolicyCommandLine, argv);
            }
        }

        SECTION("Field policy get with a policy ID requires a field ID")
        {
            std::vector<char const *> argv { "dcgmi", "policy", "--get", "--policy-id", "1" };

            WHEN("the policy command is processed")
            {
                RequireParserThrows(TestCommandLineParser::ProcessPolicyCommandLine, argv);
            }
        }

        SECTION("Field policy modify requires a field ID and a modification")
        {
            std::vector<char const *> missingFieldArgv { "dcgmi", "policy",      "--modify", "--policy-id",
                                                         "1",     "--threshold", "85" };
            std::vector<char const *> fieldOnlyArgv { "dcgmi", "policy",      "--modify", "--field",
                                                      "150",   "--policy-id", "1" };

            WHEN("the policy commands are processed")
            {
                RequireParserThrows(TestCommandLineParser::ProcessPolicyCommandLine, missingFieldArgv);
                RequireParserThrows(TestCommandLineParser::ProcessPolicyCommandLine, fieldOnlyArgv);
            }
        }

        SECTION("Field policy field ID must fit in an unsigned short")
        {
            std::vector<char const *> argv { "dcgmi",       "policy",     "--create", "--field",     "65536", "--name",
                                             "temperature", "--operator", ">",        "--threshold", "80" };

            WHEN("the policy command is processed")
            {
                RequireParserThrows(TestCommandLineParser::ProcessPolicyCommandLine, argv);
            }
        }

        SECTION("Field policy delete requires a field ID and policy ID")
        {
            std::vector<char const *> argv { "dcgmi", "policy", "--delete" };

            WHEN("the policy command is processed")
            {
                RequireParserThrows(TestCommandLineParser::ProcessPolicyCommandLine, argv);
            }
        }

        SECTION("Field policy watch requires a nonzero maximum violation count")
        {
            std::vector<char const *> argv { "dcgmi", "policy", "--watch", "--max-violations", "0" };

            WHEN("the policy command is processed")
            {
                RequireParserThrows(TestCommandLineParser::ProcessPolicyCommandLine, argv);
            }
        }
    }
}

TEST_CASE_METHOD(CommandLineParserTestFixture, "CommandLineParser: ProcessPolicyCommandLine valid execution branches")
{
    ScopedCommandExecutor executeHook;

    GIVEN("policy command line actions that parse successfully")
    {
        SECTION("Get policy branch")
        {
            std::vector<char const *> argv { "dcgmi", "policy", "--get", "--json" };

            RequireParserOkAndExecuted(executeHook, TestCommandLineParser::ProcessPolicyCommandLine, argv);
        }

        SECTION("Clear policy branch")
        {
            std::vector<char const *> argv { "dcgmi", "policy", "--clear" };

            RequireParserOkAndExecuted(executeHook, TestCommandLineParser::ProcessPolicyCommandLine, argv);
        }

        SECTION("Set policy branch with every condition")
        {
            std::vector<char const *> argv { "dcgmi",          "policy",      "--set",      "1,3",
                                             "--eccerrors",    "--pcierrors", "--maxpages", "4",
                                             "--maxtemp",      "85",          "--maxpower", "300",
                                             "--nvlinkerrors", "--xiderrors" };

            RequireParserOkAndExecuted(executeHook, TestCommandLineParser::ProcessPolicyCommandLine, argv);
        }

        SECTION("Create field policy branch")
        {
            std::vector<char const *> argv { "dcgmi",  "policy",      "--create",   "--field",    "150",
                                             "--name", "temperature", "--operator", ">",          "--threshold",
                                             "85",     "--entities",  "gpu:0",      "--channels", "console" };

            RequireParserOkAndExecuted(executeHook, TestCommandLineParser::ProcessPolicyCommandLine, argv);
        }

        SECTION("Field policy query and mutation branches")
        {
            std::vector<std::vector<char const *>> const commands {
                { "dcgmi", "policy", "--list" },
                { "dcgmi", "policy", "--get", "--field", "150" },
                { "dcgmi", "policy", "--modify", "--field", "150", "--policy-id", "1", "--threshold", "90" },
                { "dcgmi", "policy", "--enable", "--field", "150", "--policy-id", "1" },
                { "dcgmi", "policy", "--disable", "--field", "150", "--policy-id", "1" },
                { "dcgmi", "policy", "--delete", "--field", "150", "--policy-id", "1" },
                { "dcgmi", "policy", "--import", "/tmp/policies.yaml" },
                { "dcgmi", "policy", "--watch", "--once" },
            };

            for (auto const &argv : commands)
            {
                RequireParserOkAndExecuted(executeHook, TestCommandLineParser::ProcessPolicyCommandLine, argv);
            }
        }
    }
}

TEST_CASE_METHOD(CommandLineParserTestFixture,
                 "CommandLineParser: ProcessGroupCommandLine invalid argument combinations")
{
    GIVEN("group command line validation")
    {
        SECTION("Negative group id")
        {
            std::vector<char const *> argv { "dcgmi", "group", "--group", "-1", "--info" };

            WHEN("the group command is processed")
            {
                RequireParserThrows(TestCommandLineParser::ProcessGroupCommandLine, argv);
            }
        }

        SECTION("Group id requires an action")
        {
            std::vector<char const *> argv { "dcgmi", "group", "--group", "1" };

            WHEN("the group command is processed")
            {
                RequireParserThrows(TestCommandLineParser::ProcessGroupCommandLine, argv);
            }
        }

        SECTION("Add cannot be combined with info")
        {
            std::vector<char const *> argv { "dcgmi", "group", "--group", "1", "--info", "--add", "gpu:0" };

            WHEN("the group command is processed")
            {
                RequireParserThrows(TestCommandLineParser::ProcessGroupCommandLine, argv);
            }
        }

        SECTION("Remove cannot be combined with info")
        {
            std::vector<char const *> argv { "dcgmi", "group", "--group", "1", "--info", "--remove", "gpu:0" };

            WHEN("the group command is processed")
            {
                RequireParserThrows(TestCommandLineParser::ProcessGroupCommandLine, argv);
            }
        }

        SECTION("Add and remove are mutually exclusive")
        {
            std::vector<char const *> argv { "dcgmi", "group", "--group", "1", "--add", "gpu:0", "--remove", "gpu:1" };

            WHEN("the group command is processed")
            {
                RequireParserThrows(TestCommandLineParser::ProcessGroupCommandLine, argv);
            }
        }

        SECTION("Default GPU and default NvSwitch groups are mutually exclusive")
        {
            std::vector<char const *> argv {
                "dcgmi", "group", "--create", "mixed", "--default", "--defaultnvswitches"
            };

            WHEN("the group command is processed")
            {
                RequireParserThrows(TestCommandLineParser::ProcessGroupCommandLine, argv);
            }
        }
    }
}

TEST_CASE_METHOD(CommandLineParserTestFixture, "CommandLineParser: ProcessGroupCommandLine valid execution branches")
{
    ScopedCommandExecutor executeHook;

    GIVEN("valid group command lines")
    {
        SECTION("list groups")
        {
            std::vector<char const *> argv { "dcgmi", "group", "--list", "--json" };

            RequireParserOkAndExecuted(executeHook, TestCommandLineParser::ProcessGroupCommandLine, argv);
        }

        SECTION("create default GPU group")
        {
            std::vector<char const *> argv { "dcgmi", "group", "--create", "gpus", "--default" };

            RequireParserOkAndExecuted(executeHook, TestCommandLineParser::ProcessGroupCommandLine, argv);
        }

        SECTION("create group with explicit entities")
        {
            std::vector<char const *> argv { "dcgmi", "group", "--create", "mixed", "--add", "gpu:0,cpu:0" };

            RequireParserOkAndExecuted(executeHook, TestCommandLineParser::ProcessGroupCommandLine, argv);
        }

        SECTION("delete group")
        {
            std::vector<char const *> argv { "dcgmi", "group", "--delete", "7" };

            RequireParserOkAndExecuted(executeHook, TestCommandLineParser::ProcessGroupCommandLine, argv);
        }

        SECTION("show group info")
        {
            std::vector<char const *> argv { "dcgmi", "group", "--group", "3", "--info", "--json" };

            RequireParserOkAndExecuted(executeHook, TestCommandLineParser::ProcessGroupCommandLine, argv);
        }

        SECTION("add and remove entities")
        {
            std::vector<char const *> addArgv { "dcgmi", "group", "--group", "3", "--add", "gpu:0" };
            std::vector<char const *> removeArgv { "dcgmi", "group", "--group", "3", "--remove", "gpu:0" };

            RequireParserOkAndExecuted(executeHook, TestCommandLineParser::ProcessGroupCommandLine, addArgv);
            RequireParserOkAndExecuted(executeHook, TestCommandLineParser::ProcessGroupCommandLine, removeArgv);
        }
    }
}

TEST_CASE_METHOD(CommandLineParserTestFixture,
                 "CommandLineParser: ProcessFieldGroupCommandLine invalid argument combinations")
{
    GIVEN("fieldgroup command line validation")
    {
        SECTION("Info requires field group id")
        {
            std::vector<char const *> argv { "dcgmi", "fieldgroup", "--info" };

            WHEN("the fieldgroup command is processed")
            {
                RequireParserThrows(TestCommandLineParser::ProcessFieldGroupCommandLine, argv);
            }
        }

        SECTION("Create requires field IDs")
        {
            std::vector<char const *> argv { "dcgmi", "fieldgroup", "--create", "fields" };

            WHEN("the fieldgroup command is processed")
            {
                RequireParserThrows(TestCommandLineParser::ProcessFieldGroupCommandLine, argv);
            }
        }

        SECTION("Delete requires field group id")
        {
            std::vector<char const *> argv { "dcgmi", "fieldgroup", "--delete" };

            WHEN("the fieldgroup command is processed")
            {
                RequireParserThrows(TestCommandLineParser::ProcessFieldGroupCommandLine, argv);
            }
        }
    }
}

TEST_CASE_METHOD(CommandLineParserTestFixture,
                 "CommandLineParser: ProcessFieldGroupCommandLine valid execution branches")
{
    ScopedCommandExecutor executeHook;

    GIVEN("valid fieldgroup command lines")
    {
        SECTION("list field groups")
        {
            std::vector<char const *> argv { "dcgmi", "fieldgroup", "--list", "--json" };

            RequireParserOkAndExecuted(executeHook, TestCommandLineParser::ProcessFieldGroupCommandLine, argv);
        }

        SECTION("create field group")
        {
            std::vector<char const *> argv { "dcgmi", "fieldgroup", "--create", "temps", "--fieldids", "150,155" };

            RequireParserOkAndExecuted(executeHook, TestCommandLineParser::ProcessFieldGroupCommandLine, argv);
        }

        SECTION("show and delete field group")
        {
            std::vector<char const *> infoArgv { "dcgmi", "fieldgroup", "--info", "--fieldgroup", "2", "--json" };
            std::vector<char const *> deleteArgv { "dcgmi", "fieldgroup", "--delete", "--fieldgroup", "2" };

            RequireParserOkAndExecuted(executeHook, TestCommandLineParser::ProcessFieldGroupCommandLine, infoArgv);
            RequireParserOkAndExecuted(executeHook, TestCommandLineParser::ProcessFieldGroupCommandLine, deleteArgv);
        }
    }
}

TEST_CASE_METHOD(CommandLineParserTestFixture,
                 "CommandLineParser: ProcessConfigCommandLine invalid argument combinations")
{
    GIVEN("config command line validation")
    {
        SECTION("Set requires at least one configuration option")
        {
            std::vector<char const *> argv { "dcgmi", "config", "--set" };

            WHEN("the config command is processed")
            {
                RequireParserThrows(TestCommandLineParser::ProcessConfigCommandLine, argv);
            }
        }

        SECTION("Set rejects malformed application clocks")
        {
            std::vector<char const *> argv { "dcgmi", "config", "--set", "--appclocks", "1000" };

            WHEN("the config command is processed")
            {
                RequireParserThrows(TestCommandLineParser::ProcessConfigCommandLine, argv);
            }
        }

        SECTION("Get rejects set-only options")
        {
            std::vector<char const *> argv { "dcgmi", "config", "--get", "--powerlimit", "250" };

            WHEN("the config command is processed")
            {
                RequireParserThrows(TestCommandLineParser::ProcessConfigCommandLine, argv);
            }
        }

        SECTION("Enforce rejects set-only options")
        {
            std::vector<char const *> argv { "dcgmi", "config", "--enforce", "--compmode", "1" };

            WHEN("the config command is processed")
            {
                RequireParserThrows(TestCommandLineParser::ProcessConfigCommandLine, argv);
            }
        }
    }
}

TEST_CASE_METHOD(CommandLineParserTestFixture, "CommandLineParser: ProcessConfigCommandLine valid execution branches")
{
    ScopedCommandExecutor executeHook;

    GIVEN("valid config command lines")
    {
        SECTION("set scalar config values")
        {
            std::vector<char const *> argv { "dcgmi", "config",      "--set",     "--group",
                                             "3",     "--eccmode",   "1",         "--syncboost",
                                             "0",     "--appclocks", "5000,1500", "--powerlimit",
                                             "250",   "--compmode",  "2" };

            RequireParserOkAndExecuted(executeHook, TestCommandLineParser::ProcessConfigCommandLine, argv);
        }

        SECTION("set workload power profile")
        {
            std::vector<char const *> argv { "dcgmi", "config",
                                             "--set", "--group",
                                             "3",     "--workloadpowerprofile",
                                             "4",     "--workloadpowerprofileaction",
                                             "o" };

            RequireParserOkAndExecuted(executeHook, TestCommandLineParser::ProcessConfigCommandLine, argv, 2);
        }

        SECTION("get and enforce config")
        {
            std::vector<char const *> getArgv { "dcgmi", "config", "--get", "--group", "3", "--verbose", "--json" };
            std::vector<char const *> enforceArgv { "dcgmi", "config", "--enforce", "--group", "3" };

            RequireParserOkAndExecuted(executeHook, TestCommandLineParser::ProcessConfigCommandLine, getArgv);
            RequireParserOkAndExecuted(executeHook, TestCommandLineParser::ProcessConfigCommandLine, enforceArgv);
        }
    }
}

TEST_CASE_METHOD(CommandLineParserTestFixture,
                 "CommandLineParser: ProcessHealthCommandLine invalid argument combinations")
{
    GIVEN("health command line validation")
    {
        SECTION("Invalid group alias is rejected")
        {
            std::vector<char const *> argv { "dcgmi", "health", "--fetch", "--group", "unknown" };

            WHEN("the health command is processed")
            {
                RequireParserThrows(TestCommandLineParser::ProcessHealthCommandLine, argv);
            }
        }

        SECTION("Duplicate set flags are rejected")
        {
            std::vector<char const *> argv { "dcgmi", "health", "--set", "pp" };
            std::vector<char const *> argvImex { "dcgmi", "health", "--set", "ee" };

            WHEN("the health command is processed")
            {
                RequireParserThrows(TestCommandLineParser::ProcessHealthCommandLine, argv);
                RequireParserThrows(TestCommandLineParser::ProcessHealthCommandLine, argvImex);
            }
        }

        SECTION("All-watch flag cannot be combined with specific flags")
        {
            std::vector<char const *> argv { "dcgmi", "health", "--set", "ap" };

            WHEN("the health command is processed")
            {
                RequireParserThrows(TestCommandLineParser::ProcessHealthCommandLine, argv);
            }
        }

        SECTION("Unknown set flags are rejected")
        {
            std::vector<char const *> argv { "dcgmi", "health", "--set", "z" };

            WHEN("the health command is processed")
            {
                RequireParserThrows(TestCommandLineParser::ProcessHealthCommandLine, argv);
            }
        }
    }
}

TEST_CASE_METHOD(CommandLineParserTestFixture, "CommandLineParser: ProcessHealthCommandLine valid execution branches")
{
    ScopedCommandExecutor executeHook;

    GIVEN("valid health command lines")
    {
        SECTION("fetch and check watches")
        {
            std::vector<char const *> fetchArgv { "dcgmi", "health", "--fetch", "--group", "g", "--json" };
            std::vector<char const *> checkArgv { "dcgmi", "health", "--check", "--group", "a", "--json" };

            RequireParserOkAndExecuted(executeHook, TestCommandLineParser::ProcessHealthCommandLine, fetchArgv);
            RequireParserOkAndExecuted(executeHook, TestCommandLineParser::ProcessHealthCommandLine, checkArgv);
        }

        SECTION("clear and set watches")
        {
            std::vector<char const *> clearArgv { "dcgmi", "health", "--clear", "--group", "3" };
            std::vector<char const *> setArgv { "dcgmi", "health",         "--set", "pmitndxe",          "--group",
                                                "3",     "--max-keep-age", "60",    "--update-interval", "10" };
            std::vector<char const *> setAllArgv { "dcgmi", "health", "--set", "a", "--group", "3" };

            RequireParserOkAndExecuted(executeHook, TestCommandLineParser::ProcessHealthCommandLine, clearArgv);
            RequireParserOkAndExecuted(executeHook, TestCommandLineParser::ProcessHealthCommandLine, setArgv);
            RequireParserOkAndExecuted(executeHook, TestCommandLineParser::ProcessHealthCommandLine, setAllArgv);
        }

        SECTION("IMEX flag combinations")
        {
            std::vector<char const *> imexOnly { "dcgmi", "health", "--set", "e", "--group", "3" };
            std::vector<char const *> nvlinkOnly { "dcgmi", "health", "--set", "n", "--group", "3" };
            std::vector<char const *> nvlinkAndImex { "dcgmi", "health", "--set", "ne", "--group", "3" };

            CHECK_NOTHROW(TestCommandLineParser::ProcessHealthCommandLine(
                static_cast<int>(imexOnly.size()), imexOnly.data(), dcgmOps));
            CHECK_NOTHROW(TestCommandLineParser::ProcessHealthCommandLine(
                static_cast<int>(nvlinkOnly.size()), nvlinkOnly.data(), dcgmOps));
            CHECK_NOTHROW(TestCommandLineParser::ProcessHealthCommandLine(
                static_cast<int>(nvlinkAndImex.size()), nvlinkAndImex.data(), dcgmOps));
        }
    }
}

TEST_CASE_METHOD(CommandLineParserTestFixture,
                 "CommandLineParser: ProcessProfileCommandLine invalid argument combinations")
{
    GIVEN("profile command line validation")
    {
        SECTION("Entity and group IDs are mutually exclusive")
        {
            std::vector<char const *> argv { "dcgmi", "profile", "--list", "--entity-id", "0", "--group-id", "1" };

            WHEN("the profile command is processed")
            {
                RequireParserThrows(TestCommandLineParser::ProcessProfileCommandLine, argv);
            }
        }
    }
}

TEST_CASE_METHOD(CommandLineParserTestFixture, "CommandLineParser: valid execution branches for monitoring subsystems")
{
    ScopedCommandExecutor executeHook;

    GIVEN("valid topology, nvlink, dmon, and profile command lines")
    {
        SECTION("topology commands")
        {
            std::vector<char const *> groupArgv { "dcgmi", "topo", "--group", "3", "--json" };
            std::vector<char const *> gpuArgv { "dcgmi", "topo", "--gpuid", "0", "--json" };

            RequireParserOkAndExecuted(executeHook, TestCommandLineParser::ProcessTopoCommandLine, groupArgv);
            RequireParserOkAndExecuted(executeHook, TestCommandLineParser::ProcessTopoCommandLine, gpuArgv);
        }

        SECTION("nvlink commands")
        {
            std::vector<char const *> errorsArgv { "dcgmi", "nvlink", "--errors", "--gpuid", "0", "--json" };
            std::vector<char const *> statusArgv { "dcgmi", "nvlink", "--link-status", "--show-entity-ids" };

            RequireParserOkAndExecuted(executeHook, TestCommandLineParser::ProcessNvlinkCommandLine, errorsArgv);
            RequireParserOkAndExecuted(executeHook, TestCommandLineParser::ProcessNvlinkCommandLine, statusArgv);
        }

        SECTION("device monitor commands")
        {
            std::vector<char const *> listArgv { "dcgmi", "dmon", "--list" };
            std::vector<char const *> fieldsArgv { "dcgmi", "dmon",    "--entity-id", "gpu:0",   "--field-id",
                                                   "150",   "--delay", "1",           "--count", "1" };

            RequireParserOkAndExecuted(executeHook, TestCommandLineParser::ProcessDmonCommandLine, listArgv);
            RequireParserOkAndExecuted(executeHook, TestCommandLineParser::ProcessDmonCommandLine, fieldsArgv);
        }

        SECTION("profile commands")
        {
            std::vector<char const *> listArgv { "dcgmi", "profile", "--list", "--entity-id", "gpu:0", "--json" };
            std::vector<char const *> pauseArgv { "dcgmi", "profile", "--pause" };
            std::vector<char const *> resumeArgv { "dcgmi", "profile", "--resume" };

            RequireParserOkAndExecuted(executeHook, TestCommandLineParser::ProcessProfileCommandLine, listArgv);
            RequireParserOkAndExecuted(executeHook, TestCommandLineParser::ProcessProfileCommandLine, pauseArgv);
            RequireParserOkAndExecuted(executeHook, TestCommandLineParser::ProcessProfileCommandLine, resumeArgv);
        }
    }
}

TEST_CASE_METHOD(CommandLineParserTestFixture,
                 "CommandLineParser: ProcessSettingsCommandLine invalid argument combinations")
{
    GIVEN("settings command line validation")
    {
        SECTION("Attach and detach are mutually exclusive")
        {
            std::vector<char const *> argv { "dcgmi", "set", "--attach-driver", "--detach-driver" };

            WHEN("the settings command is processed")
            {
                RequireParserThrows(TestCommandLineParser::ProcessSettingsCommandLine, argv);
            }
        }

        SECTION("At least one setting action is required")
        {
            std::vector<char const *> argv { "dcgmi", "set" };

            WHEN("the settings command is processed")
            {
                RequireParserThrows(TestCommandLineParser::ProcessSettingsCommandLine, argv);
            }
        }
    }
}

TEST_CASE_METHOD(CommandLineParserTestFixture, "CommandLineParser: valid execution branches for settings and modules")
{
    ScopedCommandExecutor executeHook;

    GIVEN("valid settings and modules command lines")
    {
        SECTION("settings commands")
        {
            std::vector<char const *> severityArgv { "dcgmi", "set",   "--logging-severity", "DEBUG", "--target-logger",
                                                     "BASE",  "--json" };
            std::vector<char const *> attachArgv { "dcgmi", "set", "--attach-driver" };
            std::vector<char const *> detachArgv { "dcgmi", "set", "--detach-driver" };

            RequireParserOkAndExecuted(executeHook, TestCommandLineParser::ProcessSettingsCommandLine, severityArgv);
            RequireParserOkAndExecuted(executeHook, TestCommandLineParser::ProcessSettingsCommandLine, attachArgv);
            RequireParserOkAndExecuted(executeHook, TestCommandLineParser::ProcessSettingsCommandLine, detachArgv);
        }

        SECTION("module commands")
        {
            std::vector<char const *> listArgv { "dcgmi", "modules", "--list", "--json" };
            std::vector<char const *> denylistArgv { "dcgmi", "modules", "--denylist", "config", "--json" };

            RequireParserOkAndExecuted(executeHook, TestCommandLineParser::ProcessModuleCommandLine, listArgv);
            RequireParserOkAndExecuted(executeHook, TestCommandLineParser::ProcessModuleCommandLine, denylistArgv);
        }
    }
}

TEST_CASE_METHOD(CommandLineParserTestFixture,
                 "CommandLineParser: ProcessAdminCommandLine invalid argument combinations")
{
    GIVEN("admin command line validation")
    {
        SECTION("Negative group id")
        {
            std::vector<char const *> argv { "dcgmi", "test", "--introspect", "--group", "-1" };

            WHEN("the admin command is processed")
            {
                RequireParserThrows(TestCommandLineParser::ProcessAdminCommandLine, argv);
            }
        }

        SECTION("GPU and group are mutually exclusive")
        {
            std::vector<char const *> argv { "dcgmi", "test", "--introspect", "--gpuid", "1", "--group", "2" };

            WHEN("the admin command is processed")
            {
                RequireParserThrows(TestCommandLineParser::ProcessAdminCommandLine, argv);
            }
        }

        SECTION("Field is required for introspection")
        {
            std::vector<char const *> argv { "dcgmi", "test", "--introspect" };

            WHEN("the admin command is processed")
            {
                RequireParserThrows(TestCommandLineParser::ProcessAdminCommandLine, argv);
            }
        }

        SECTION("Injection cannot target a group")
        {
            std::vector<char const *> argv { "dcgmi",   "test", "--inject", "--group", "2",
                                             "--field", "100",  "--value",  "1" };

            WHEN("the admin command is processed")
            {
                RequireParserThrows(TestCommandLineParser::ProcessAdminCommandLine, argv);
            }
        }
    }
}

TEST_CASE_METHOD(CommandLineParserTestFixture, "CommandLineParser: ProcessAdminCommandLine valid execution branches")
{
    ScopedCommandExecutor executeHook;

    GIVEN("valid admin command lines")
    {
        SECTION("introspect and inject")
        {
            std::vector<char const *> introspectGpuArgv { "dcgmi", "test",    "--introspect", "--gpuid",
                                                          "0",     "--field", "150" };
            std::vector<char const *> introspectGroupArgv { "dcgmi", "test",    "--introspect", "--group",
                                                            "3",     "--field", "150" };
            std::vector<char const *> injectArgv { "dcgmi", "test",    "--inject", "--gpuid",  "0", "--field",
                                                   "150",   "--value", "42",       "--offset", "2" };

            RequireParserOkAndExecuted(executeHook, TestCommandLineParser::ProcessAdminCommandLine, introspectGpuArgv);
            RequireParserOkAndExecuted(
                executeHook, TestCommandLineParser::ProcessAdminCommandLine, introspectGroupArgv);
            RequireParserOkAndExecuted(executeHook, TestCommandLineParser::ProcessAdminCommandLine, injectArgv);
        }

        SECTION("pause and resume")
        {
            std::vector<char const *> pauseArgv { "dcgmi", "test", "--pause" };
            std::vector<char const *> resumeArgv { "dcgmi", "test", "--resume" };

            RequireParserOkAndExecuted(executeHook, TestCommandLineParser::ProcessAdminCommandLine, pauseArgv);
            RequireParserOkAndExecuted(executeHook, TestCommandLineParser::ProcessAdminCommandLine, resumeArgv);
        }
    }
}

TEST_CASE_METHOD(CommandLineParserTestFixture, "CommandLineParser: ProcessDiagCommandLine valid execution branches")
{
    ScopedCommandExecutor executeHook;

    struct ValidArgsTestCase
    {
        std::string description;
        std::vector<char const *> argv;
    };

    std::vector<ValidArgsTestCase> const testCases = {
        { "verifies --fail-early option",
          { "dcgmi", "diag", "--run", "1", "-i", "0", "--parameters", "diagnostic.test_duration=30", "--fail-early" } },
        { "verifies --check-interval option",
          { "dcgmi",
            "diag",
            "--run",
            "1",
            "-i",
            "0",
            "--parameters",
            "diagnostic.test_duration=30",
            "--fail-early",
            "--check-interval",
            "3" } },
        { "verifies multiple parameters with different test name but same variable name",
          { "dcgmi",
            "diag",
            "--run",
            "1",
            "-i",
            "0",
            "--parameters",
            "memtest.test_duration=10;diagnostic.test_duration=10" } },
        { "verifies that --throttle-mask with HW_SLOWDOWN reason to be ignored",
          { "dcgmi", "diag", "--run", "1", "-i", "0", "--throttle-mask", "HW_SLOWDOWN" } },
        { "verifies that --throttle-mask with HW_THERMAL reason to be ignored",
          { "dcgmi", "diag", "--run", "1", "-i", "0", "--throttle-mask", "HW_THERMAL" } },
        { "verifies that --throttle-mask with HW_POWER_BRAKE reason to be ignored",
          { "dcgmi", "diag", "--run", "1", "-i", "0", "--throttle-mask", "HW_POWER_BRAKE" } },
        { "verifies that --throttle-mask with multiple reasons (HW_SLOWDOWN,SW_THERMAL,HW_POWER_BRAKE) to be ignored",
          { "dcgmi", "diag", "--run", "1", "-i", "0", "--throttle-mask", "HW_SLOWDOWN,SW_THERMAL,HW_POWER_BRAKE" } },
        { "verifies that --throttle-mask with HW_SLOWDOWN (8) and SW_THERMAL (32) reason to be ignored",
          { "dcgmi", "diag", "--run", "1", "-i", "0", "--throttle-mask", "40" } },
        { "verifies that --throttle-mask with SW_THERMAL (32) and HW_THERMAL (64) reason to be ignored",
          { "dcgmi", "diag", "--run", "1", "-i", "0", "--throttle-mask", "96" } },
        { "verifies that --throttle-mask with ALL reasons to be ignored",
          { "dcgmi", "diag", "--run", "1", "-i", "0", "--throttle-mask", "232" } },
        { "verifies that --clocksevent-mask with HW_SLOWDOWN reason to be ignored",
          { "dcgmi", "diag", "--run", "1", "-i", "0", "--clocksevent-mask", "HW_SLOWDOWN" } },
        { "verifies that --clocksevent-mask with SW_THERMAL reason to be ignored",
          { "dcgmi", "diag", "--run", "1", "-i", "0", "--clocksevent-mask", "SW_THERMAL" } },
        { "verifies that --clocksevent-mask with HW_THERMAL reason to be ignored",
          { "dcgmi", "diag", "--run", "1", "-i", "0", "--clocksevent-mask", "HW_THERMAL" } },
        { "verifies that --clocksevent-mask with HW_POWER_BRAKE reason to be ignored",
          { "dcgmi", "diag", "--run", "1", "-i", "0", "--clocksevent-mask", "HW_POWER_BRAKE" } },
        { "verifies that --clocksevent-mask with multiple reasons (SW_THERMAL,HW_THERMAL,HW_SLOWDOWN) to be ignored",
          { "dcgmi", "diag", "--run", "1", "-i", "0", "--clocksevent-mask", "SW_THERMAL,HW_THERMAL,HW_SLOWDOWN" } },
        { "verifies that --clocksevent-mask with HW_SLOWDOWN (8) and SW_THERMAL (32) reason to be ignored",
          { "dcgmi", "diag", "--run", "1", "-i", "0", "--clocksevent-mask", "40" } },
        { "verifies that --clocksevent-mask with SW_THERMAL (32) and HW_THERMAL (64) reason to be ignored",
          { "dcgmi", "diag", "--run", "1", "-i", "0", "--clocksevent-mask", "96" } },
        { "verifies --entity-id option accepts and validates list of GPUs passed in csv format",
          { "dcgmi", "diag", "--run", "1", "--entity-id", "gpu:0,1,2" } },
        { "verifies --entity-id option accepts and validates list of GPUs passed in bracket format",
          { "dcgmi", "diag", "--run", "1", "--entity-id", "gpu:{0}" } },
        { "verifies --entity-id option accepts and validates list of GPUs passed in range format",
          { "dcgmi", "diag", "--run", "1", "--entity-id", "gpu:{0-2}" } },
    };

    for (auto const &testCase : testCases)
    {
        DYNAMIC_SECTION(testCase.description)
        {
            RequireParserOkAndExecuted(executeHook, TestCommandLineParser::ProcessDiagCommandLine, testCase.argv);
        }
    }
}

TEST_CASE_METHOD(CommandLineParserTestFixture, "CommandLineParser: ProcessDiagCommandLine server-validated arguments")
{
    struct ServerArgsTestCase
    {
        std::string description;
        std::vector<char const *> argv;
        dcgmReturn_t expected;
        int expectedServerCalls;
    };

    std::vector<ServerArgsTestCase> const testCases = {
        { "Invalid parameter - misspelled parameter name test_durration",
          { "dcgmi", "diag", "--run", "3", "--parameters", "diagnostic.test_durration=40" },
          DCGM_ST_NVVS_NO_AVAILABLE_TEST,
          1 },
        { "Invalid parameter - misspelled subtest h2d_d2h_singgle_pinned",
          { "dcgmi", "diag", "--run", "3", "--parameters", "pcie.h2d_d2h_singgle_pinned.iterations=4000" },
          DCGM_ST_NVVS_NO_AVAILABLE_TEST,
          1 },
        { "Invalid parameter - misspelled parameter itterations",
          { "dcgmi", "diag", "--run", "3", "--parameters", "pcie.h2d_d2h_single_pinned.itterations=4000" },
          DCGM_ST_NVVS_NO_AVAILABLE_TEST,
          1 },
        { "Invalid parameter - bob.tom",
          { "dcgmi", "diag", "--run", "3", "--parameters", "bob.tom=maybe" },
          DCGM_ST_NVVS_NO_AVAILABLE_TEST,
          1 },
        { "Invalid parameter - truck",
          { "dcgmi", "diag", "--run", "3", "--parameters", "truck=slow" },
          DCGM_ST_NVVS_NO_AVAILABLE_TEST,
          1 },
        { "Invalid parameter - my=life=got=flipped=turned=upside=down",
          { "dcgmi", "diag", "--run", "3", "--parameters", "my=life=got=flipped=turned=upside=down" },
          DCGM_ST_NVVS_NO_AVAILABLE_TEST,
          1 },
        { "Invalid parameter - and.i'd.like.to.take.a=minute=just.sit=right=there",
          { "dcgmi", "diag", "--run", "3", "--parameters", "and.i'd.like.to.take.a=minute=just.sit=right=there" },
          DCGM_ST_NVVS_NO_AVAILABLE_TEST,
          1 },
        { "Invalid parameter - i'll tell you=how.I.became the=prince of .a town called",
          { "dcgmi", "diag", "--run", "3", "--parameters", "i'll tell you=how.I.became the=prince of .a town called" },
          DCGM_ST_NVVS_NO_AVAILABLE_TEST,
          1 },
        { "Make sure -i is a comma-separated list of integers not arbitrary text",
          { "dcgmi", "diag", "--run", "1", "-i", "roshar" },
          DCGM_ST_BADPARAM,
          0 },
        { "Make sure -i is a comma-separated list of integers not letters",
          { "dcgmi", "diag", "--run", "1", "-i", "a,b,c,d,e,f" },
          DCGM_ST_BADPARAM,
          0 },
        { "Test --entity-id option with invalid gpu id (letters)",
          { "dcgmi", "diag", "--run", "1", "--entity-id", "a,b,c,d,e" },
          DCGM_ST_BADPARAM,
          0 },
        { "Test --entity-id option with invalid gpu id (negative numbers)",
          { "dcgmi", "diag", "--run", "1", "--entity-id", "-1,-2,-3,-4,-5" },
          DCGM_ST_BADPARAM,
          0 },
        { "Test --entity-id option with invalid entity type",
          { "dcgmi", "diag", "--run", "1", "--entity-id", "dcgm:0,1" },
          DCGM_ST_BADPARAM,
          0 },
        { "Test --entity-id option with unclosed brace",
          { "dcgmi", "diag", "--run", "1", "--entity-id", "gpu:{0" },
          DCGM_ST_BADPARAM,
          0 },
        { "Test --entity-id with non-GPU and non-CPU entity (core)",
          { "dcgmi", "diag", "--run", "1", "--entity-id", "core:0" },
          DCGM_ST_BADPARAM,
          0 },
        { "Test --entity-id with non-supported wildcard entity type",
          { "dcgmi", "diag", "--run", "1", "--entity-id", "core:*" },
          DCGM_ST_BADPARAM,
          0 },
        { "Test --expectedNumEntities option with empty value after colon",
          { "dcgmi", "diag", "--run", "1", "--expectedNumEntities", "gpu:" },
          DCGM_ST_BADPARAM,
          1 },
        { "Test --expectedNumEntities option with multiple entity types",
          { "dcgmi", "diag", "--run", "1", "--expectedNumEntities", "gpu:0,cpu:1" },
          DCGM_ST_BADPARAM,
          1 },
        { "Test --expectedNumEntities option with negative number",
          { "dcgmi", "diag", "--run", "1", "--expectedNumEntities", "gpu:-1" },
          DCGM_ST_BADPARAM,
          1 },
    };

    for (auto const &testCase : testCases)
    {
        DYNAMIC_SECTION(testCase.description)
        {
            mockDcgmOps.m_actionValidateState.actionValidateReturn = testCase.expected;
            CHECK(TestCommandLineParser::ProcessDiagCommandLine(
                      static_cast<int>(testCase.argv.size()), testCase.argv.data(), dcgmOps)
                  == testCase.expected);
            CHECK(mockDcgmOps.m_actionValidateState.actionValidateCallCount == testCase.expectedServerCalls);
        }
    }
}

TEST_CASE_METHOD(CommandLineParserTestFixture,
                 "CommandLineParser: ProcessDiagCommandLine invalid arguments that throw exceptions")
{
    struct ThrowingArgsTestCase
    {
        std::string description;
        std::vector<char const *> argv;
    };

    std::string const longParam
        = std::string("pcie.h2d_d2h_singgle_pinned.iterations=") + std::string(DCGM_MAX_TEST_PARMS_LEN_V2, '6');
    std::string const longEntityId(DCGM_ENTITY_ID_LIST_LEN + 1, '6');

    std::vector<ThrowingArgsTestCase> const testCases = {
        { "Can't run with a test number that doesn't exist", { "dcgmi", "diag", "--run", "5" } },
        { "Invalid parameter - now this is a story all about how",
          { "dcgmi", "diag", "--run", "3", "--parameters", "now this is a story all about how" } },
        { "Invalid parameter - Bel-Air", { "dcgmi", "diag", "--run", "3", "--parameters", "Bel-Air" } },
        { "Ensure that training is no longer supported", { "dcgmi", "diag", "--train" } },
        { "Verifies --check-interval does not allow negative numbers",
          { "dcgmi",
            "diag",
            "--run",
            "1",
            "--parameters",
            "diagnostic.test_duration=30",
            "--fail-early",
            "--check-interval",
            "-1" } },
        { "Verifies --check-interval does not allow numbers greater than 300",
          { "dcgmi",
            "diag",
            "--run",
            "1",
            "--parameters",
            "diagnostic.test_duration=30",
            "--fail-early",
            "--check-interval",
            "350" } },
        { "Verifies --check-interval requires --fail-early parameter",
          { "dcgmi", "diag", "--run", "1", "--parameters", "diagnostic.test_duration=30", "--check-interval", "10" } },
        { "Verifies that --throttle-mask incorrect reason HW_ZSLOWDOWN does not work",
          { "dcgmi", "diag", "--run", "1", "--throttle-mask", "HW_ZSLOWDOWN" } },
        { "Verifies that --throttle-mask incorrect reason SW_TRHERMAL does not work",
          { "dcgmi", "diag", "--run", "1", "--throttle-mask", "SW_TRHERMAL" } },
        { "Verifies that --throttle-mask incorrect reason HWT_THERMAL does not work",
          { "dcgmi", "diag", "--run", "1", "--throttle-mask", "HWT_THERMAL" } },
        { "Verifies that --throttle-mask incorrect reason HW_POWER_OUTBRAKE does not work",
          { "dcgmi", "diag", "--run", "1", "--throttle-mask", "HW_POWER_OUTBRAKE" } },
        { "Verifies that --throttle-mask does not accept incorrect values for any reasons to be ignored",
          { "dcgmi", "diag", "--run", "1", "--throttle-mask", "-10" } },
        { "Verifies that --clocksevent-mask incorrect reason HW_ZSLOWDOWN does not work",
          { "dcgmi", "diag", "--run", "1", "--clocksevent-mask", "HW_ZSLOWDOWN" } },
        { "Verifies that --clocksevent-mask incorrect reason SW_TRHERMAL does not work",
          { "dcgmi", "diag", "--run", "1", "--clocksevent-mask", "SW_TRHERMAL" } },
        { "Verifies that --clocksevent-mask incorrect reason HWT_THERMAL does not work",
          { "dcgmi", "diag", "--run", "1", "--clocksevent-mask", "HWT_THERMAL" } },
        { "Verifies that --clocksevent-mask incorrect reason HW_POWER_OUTBRAKE does not work",
          { "dcgmi", "diag", "--run", "1", "--clocksevent-mask", "HW_POWER_OUTBRAKE" } },
        { "Verifies that --clocksevent-mask does not accept incorrect values for any reasons to be ignored",
          { "dcgmi", "diag", "--run", "1", "--clocksevent-mask", "-10" } },
        { "Verifies --gpuList option rejects dash-separated list of integers",
          { "dcgmi", "diag", "--run", "1", "--gpuList", "0-1-2-3-4-5" } },
        { "Verifies --gpuList option rejects negative GPU IDs",
          { "dcgmi", "diag", "--run", "1", "--gpuList", "-1,-2,-3,-4,-5" } },
        { "Verifies --gpuList option rejects letters", { "dcgmi", "diag", "--run", "1", "--gpuList", "a,b,c,d,e" } },
        { "Cannot run 0 iterations", { "dcgmi", "diag", "--run", "1", "--iterations", "0" } },
        { "Multiple defined parameters not allowed",
          { "dcgmi",
            "diag",
            "--run",
            "1",
            "--parameters",
            "diagnostic.test_duration=30;diagnostic.test_duration=40" } },
        { "Test case parameter length limit exceeded",
          { "dcgmi", "diag", "--run", "3", "--parameters", longParam.c_str() } },
        { "Test_duration equals the timeout",
          { "dcgmi",
            "diag",
            "--run",
            "diagnostic",
            "--parameters",
            "diagnostic.test_duration=10",
            "--timeout",
            "10" } },
        { "Combined test_duration exceeds the timeout",
          { "dcgmi",
            "diag",
            "--run",
            "diagnostic,targeted_power",
            "--parameters",
            "diagnostic.test_duration=10;targeted_power.test_duration=10",
            "--timeout",
            "15" } },
        { "Test empty --entity-id", { "dcgmi", "diag", "--run", "1", "--entity-id", "" } },
        { "Test too large --entity-id", { "dcgmi", "diag", "--run", "1", "--entity-id", longEntityId.c_str() } },
        { "Test --entity-id and --gpuList cannot be used at the same time",
          { "dcgmi", "diag", "--run", "1", "--entity-id", "0", "--gpuList", "0" } },
        { "Test --entity-id and --group cannot be used at the same time",
          { "dcgmi", "diag", "--run", "1", "--entity-id", "0", "--group", "1" } },
        { "Test --gpuList and --group cannot be used at the same time",
          { "dcgmi", "diag", "--run", "1", "--gpuList", "0", "--group", "1" } },
        { "Test --expectedNumEntities cannot be used with fake gpus",
          { "dcgmi", "diag", "--run", "1", "-f", "0,1", "--expectedNumEntities", "gpu:2" } },
        { "Test --expectedNumEntities cannot be used with gpu list",
          { "dcgmi", "diag", "--run", "1", "--gpuList", "0,1", "--expectedNumEntities", "gpu:2" } },
    };

    for (auto const &testCase : testCases)
    {
        DYNAMIC_SECTION(testCase.description)
        {
            RequireParserThrows(TestCommandLineParser::ProcessDiagCommandLine, testCase.argv);
        }
    }
}
