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

#include <DcgmBuildInfo.hpp>
#include <DcgmStringHelpers.h>
#include <dcgm_errors.h>
#include <dcgm_structs.h>

#include <vector>

#define DCGMI_TESTS
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

using ParserFunction = dcgmReturn_t (*)(int, char const *const *);

class ScopedCommandExecutor
{
public:
    ScopedCommandExecutor()
        : m_previousExecutor(CommandLineParser::m_commandExecutor)
    {
        CommandLineParser::m_commandExecutor = [this](Command &) {
            m_callCount++;
            return DCGM_ST_OK;
        };
    }

    ~ScopedCommandExecutor()
    {
        CommandLineParser::m_commandExecutor = m_previousExecutor;
    }

    int CallCount() const
    {
        return m_callCount;
    }

private:
    CommandLineParser::CommandExecutor m_previousExecutor;
    int m_callCount = 0;
};

void RequireParserThrows(ParserFunction parser, std::vector<char const *> const &argv)
{
    REQUIRE_THROWS_AS(parser(static_cast<int>(argv.size()), argv.data()), TCLAP::CmdLineParseException);
}

void RequireParserOkAndExecuted(ScopedCommandExecutor const &executeHook,
                                ParserFunction parser,
                                std::vector<char const *> const &argv,
                                int expectedExecuteCalls = 1)
{
    int const callCount = executeHook.CallCount();
    dcgmReturn_t result = DCGM_ST_GENERIC_ERROR;

    CHECK_NOTHROW(result = parser(static_cast<int>(argv.size()), argv.data()));
    CHECK(result == DCGM_ST_OK);
    CHECK(executeHook.CallCount() == callCount + expectedExecuteCalls);
}

dcgmReturn_t ProcessCommandLine(std::vector<char const *> const &argv)
{
    return CommandLineParser::ProcessCommandLine(static_cast<int>(argv.size()), argv.data());
}
} //namespace

// -----------------------------------------------------------------
SCENARIO("CheckTestDurationAndTimeout")
{
    REQUIRE_NOTHROW(CommandLineParser::CheckTestDurationAndTimeout("diagnostic.test_duration=200", 300));
    REQUIRE_THROWS_AS(CommandLineParser::CheckTestDurationAndTimeout("diagnostic.test_duration=200", 200),
                      TCLAP::CmdLineParseException);
    REQUIRE_THROWS_AS(CommandLineParser::CheckTestDurationAndTimeout("diagnostic.test_duration=200", 100),
                      TCLAP::CmdLineParseException);
    REQUIRE_THROWS_AS(CommandLineParser::CheckTestDurationAndTimeout(
                          "diagnostic.test_duration=100;pulse_test.test_duration=100", 200),
                      TCLAP::CmdLineParseException);
    REQUIRE_THROWS_AS(
        CommandLineParser::CheckTestDurationAndTimeout("diagnostic.test_duration=100;pulse_test.test_duration=100", 50),
        TCLAP::CmdLineParseException);
    REQUIRE_NOTHROW(CommandLineParser::CheckTestDurationAndTimeout(
        "diagnostic.test_duration=100;pulse_test.test_duration=100", 300));
    REQUIRE_THROWS_AS(
        CommandLineParser::CheckTestDurationAndTimeout(
            "diagnostic.test_duration=100;pulse_test.test_duration=100;targeted_stress.test_duration=100", 300),
        TCLAP::CmdLineParseException);
}

SCENARIO("MnDiag Host List Validation")
{
    SECTION("Valid host list formats")
    {
        // Test for host with default port
        REQUIRE(CommandLineParser::GetHostListVector("host1")
                == std::vector<std::string> { "host1:" + defaultPort + "=*" });
        REQUIRE(CommandLineParser::GetHostListVector("host1=0,1,2")
                == std::vector<std::string> { "host1:" + defaultPort + "=0,1,2" });
        REQUIRE(CommandLineParser::GetHostListVector("host1=0-2")
                == std::vector<std::string> { "host1:" + defaultPort + "=0,1,2" });
        REQUIRE(CommandLineParser::GetHostListVector("host1=2-20")
                == std::vector<std::string> { "host1:" + defaultPort
                                              + "=2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20" });
        REQUIRE(CommandLineParser::GetHostListVector("host1=0,1,2;host2=3,4,5")
                == std::vector<std::string> { "host1:" + defaultPort + "=0,1,2", "host2:" + defaultPort + "=3,4,5" });
        REQUIRE(CommandLineParser::GetHostListVector("host1=0-2;host2=3-5")
                == std::vector<std::string> { "host1:" + defaultPort + "=0,1,2", "host2:" + defaultPort + "=3,4,5" });
        REQUIRE(CommandLineParser::GetHostListVector("host1=0,1-3,4")
                == std::vector<std::string> { "host1:" + defaultPort + "=0,1,2,3,4" });
        REQUIRE(CommandLineParser::GetHostListVector("host1=0,0-3,4")
                == std::vector<std::string> { "host1:" + defaultPort + "=0,1,2,3,4" });
        REQUIRE(CommandLineParser::GetHostListVector("host1=0,3-6,6")
                == std::vector<std::string> { "host1:" + defaultPort + "=0,3,4,5,6" });
        REQUIRE(CommandLineParser::GetHostListVector("host1=0,0,4")
                == std::vector<std::string> { "host1:" + defaultPort + "=0,4" });
        REQUIRE(CommandLineParser::GetHostListVector("host1=0,1-3,1")
                == std::vector<std::string> { "host1:" + defaultPort + "=0,1,2,3" });

        // Test with host and explicit port specification
        REQUIRE(CommandLineParser::GetHostListVector("host1:1234=0,1,2")
                == std::vector<std::string> { "host1:1234=0,1,2" });
        REQUIRE(CommandLineParser::GetHostListVector("host1:8888=0-2")
                == std::vector<std::string> { "host1:8888=0,1,2" });
        REQUIRE(CommandLineParser::GetHostListVector("host1:5555=0,1,2;host2:6666=3,4,5")
                == std::vector<std::string> { "host1:5555=0,1,2", "host2:6666=3,4,5" });

        // Test with host and mixed port specification
        REQUIRE(CommandLineParser::GetHostListVector("host1:1234=0,1,2;host2=3,4,5")
                == std::vector<std::string> { "host1:1234=0,1,2", "host2:" + defaultPort + "=3,4,5" });

        // Test with multiple hosts with different ports
        REQUIRE(CommandLineParser::GetHostListVector("host1:1234=0,1;host1:5678=2,3;host3=0")
                == std::vector<std::string> { "host1:1234=0,1", "host1:5678=2,3", "host3:" + defaultPort + "=0" });

        // Test with host and default Unix socket paths
        REQUIRE(CommandLineParser::GetHostListVector("host1:" + unixSocketPrefix + "=0,1,2")
                == std::vector<std::string> { "host1:" + unixSocketPrefix + defaultSocketPath + "=0,1,2" });

        // Test with host and mixed Unix socket and TCP hosts
        REQUIRE(CommandLineParser::GetHostListVector("host1:" + unixSocketPrefix + "=0,1;host2=2,3")
                == std::vector<std::string> { "host1:" + unixSocketPrefix + defaultSocketPath + "=0,1",
                                              "host2:" + defaultPort + "=2,3" });

        // Test with multiple Unix socket paths
        REQUIRE(CommandLineParser::GetHostListVector("host1:" + unixSocketPrefix + "=0,1;host2:" + unixSocketPrefix
                                                     + "=2,3")
                == std::vector<std::string> { "host1:" + unixSocketPrefix + defaultSocketPath + "=0,1",
                                              "host2:" + unixSocketPrefix + defaultSocketPath + "=2,3" });
    }

    SECTION("Invalid host list formats")
    {
        REQUIRE_THROWS_AS(CommandLineParser::GetHostListVector("=0,1,2"), TCLAP::CmdLineParseException);
        REQUIRE_THROWS_AS(CommandLineParser::GetHostListVector("host1="), TCLAP::CmdLineParseException);
        REQUIRE_THROWS_AS(CommandLineParser::GetHostListVector("host1=a,b,c"), TCLAP::CmdLineParseException);
        REQUIRE_THROWS_AS(CommandLineParser::GetHostListVector("host1=2-1"), TCLAP::CmdLineParseException);
        REQUIRE_THROWS_AS(CommandLineParser::GetHostListVector("host1=0,1;host1=2,3"), TCLAP::CmdLineParseException);
        REQUIRE_THROWS_AS(CommandLineParser::GetHostListVector("host1=0,1,2-"), TCLAP::CmdLineParseException);
        REQUIRE_THROWS_AS(CommandLineParser::GetHostListVector("host1=-0,1,2"), TCLAP::CmdLineParseException);
        REQUIRE_THROWS_AS(CommandLineParser::GetHostListVector("host1=0,1,-2"), TCLAP::CmdLineParseException);
        REQUIRE_THROWS_AS(CommandLineParser::GetHostListVector("host1=-0,1,2"), TCLAP::CmdLineParseException);
        REQUIRE_THROWS_AS(CommandLineParser::GetHostListVector("host1=0,1,2-"), TCLAP::CmdLineParseException);
        REQUIRE_THROWS_AS(CommandLineParser::GetHostListVector("host1=0,-1,2"), TCLAP::CmdLineParseException);
        REQUIRE_THROWS_AS(CommandLineParser::GetHostListVector("host1=0,1,-2"), TCLAP::CmdLineParseException);

        // Non numerical charactors
        REQUIRE_THROWS_AS(CommandLineParser::GetHostListVector("host1=0,a,-2"), TCLAP::CmdLineParseException);
        REQUIRE_THROWS_AS(CommandLineParser::GetHostListVector("host1=0,1a-20"), TCLAP::CmdLineParseException);

        // Very large string case
        std::string largeString(10000, '0');
        REQUIRE_THROWS_AS(CommandLineParser::GetHostListVector("host1=1234578" + largeString),
                          TCLAP::CmdLineParseException);

        // Invalid port specification
        REQUIRE_THROWS_AS(CommandLineParser::GetHostListVector("host1:=0,1,2"), TCLAP::CmdLineParseException);
        REQUIRE_THROWS_AS(CommandLineParser::GetHostListVector("host1:abc=0,1,2"), TCLAP::CmdLineParseException);

        // Invalid port range
        REQUIRE_THROWS_AS(CommandLineParser::GetHostListVector("host1:-1=0,1,2"), TCLAP::CmdLineParseException);
        REQUIRE_THROWS_AS(CommandLineParser::GetHostListVector("host1:65536=0,1,2"), TCLAP::CmdLineParseException);
        REQUIRE_THROWS_AS(CommandLineParser::GetHostListVector("host1:99999=0,1,2"), TCLAP::CmdLineParseException);

        // Duplicate hosts with different ports (should not throw as they're considered different hosts)
        REQUIRE_NOTHROW(CommandLineParser::GetHostListVector("host1:1234=0,1;host1:5678=2,3"));

        // Duplicate hosts with same ports
        REQUIRE_THROWS_AS(CommandLineParser::GetHostListVector("host1:1234=0,1;host1:1234=2,3"),
                          TCLAP::CmdLineParseException);

        // Invalid hostname containing only colons
        REQUIRE_THROWS_AS(CommandLineParser::GetHostListVector(":=0,1,2"), TCLAP::CmdLineParseException);
        REQUIRE_THROWS_AS(CommandLineParser::GetHostListVector(":::=0,1,2"), TCLAP::CmdLineParseException);

        // Invalid hostname with leading colons
        REQUIRE_THROWS_AS(CommandLineParser::GetHostListVector(":hostname=0,1,2"), TCLAP::CmdLineParseException);

        // Invalid Unix socket path formats
        REQUIRE_THROWS_AS(CommandLineParser::GetHostListVector("host1:unix//path=0,1,2"), TCLAP::CmdLineParseException);
        REQUIRE_THROWS_AS(CommandLineParser::GetHostListVector("host1:unix:/path=0,1,2"), TCLAP::CmdLineParseException);
        REQUIRE_THROWS_AS(CommandLineParser::GetHostListVector("host1:unix://path=0,1,2"),
                          TCLAP::CmdLineParseException);

        // Missing hostname in Unix socket path
        REQUIRE_THROWS_AS(CommandLineParser::GetHostListVector(unixSocketPrefix + "/tmp/socket=0,1,2"),
                          TCLAP::CmdLineParseException);

        // Invalid format with no colon before unix:
        REQUIRE_THROWS_AS(CommandLineParser::GetHostListVector("hostunix:/tmp/socket=0,1,2"),
                          TCLAP::CmdLineParseException);

        // Invalid format with empty hostname before colon
        REQUIRE_THROWS_AS(CommandLineParser::GetHostListVector(":" + unixSocketPrefix + "/tmp/socket=0,1,2"),
                          TCLAP::CmdLineParseException);

        // Invalid format with remainder not starting with unix:/
        REQUIRE_THROWS_AS(CommandLineParser::GetHostListVector("host1:unixpath/socket=0,1,2"),
                          TCLAP::CmdLineParseException);
        REQUIRE_THROWS_AS(CommandLineParser::GetHostListVector("host1:unix://socket=0,1,2"),
                          TCLAP::CmdLineParseException);

        // Duplicate Unix socket paths
        REQUIRE_THROWS_AS(CommandLineParser::GetHostListVector("host1:" + unixSocketPrefix + "/tmp/socket=0,1;host1:"
                                                               + unixSocketPrefix + "/tmp/socket=2,3"),
                          TCLAP::CmdLineParseException);
    }
}

SCENARIO("MnDiag Parameter Validation")
{
    SECTION("Valid parameter formats")
    {
        REQUIRE_NOTHROW(CommandLineParser::ValidateParameters("test.param1=value1"));
        REQUIRE_NOTHROW(CommandLineParser::ValidateParameters("test.param1=value1;test.param2=value2"));
        REQUIRE_NOTHROW(CommandLineParser::ValidateParameters("test_name.parameter_name=parameter_value"));
    }

    SECTION("Parameter concatenation")
    {
        std::vector<std::string> params = { "test.param1=value1", "test.param2=value2" };
        REQUIRE(CommandLineParser::ConcatenateParameters(params) == "test.param1=value1;test.param2=value2");

        params = { "test.passthrough_args=arg1", "test.passthrough_args=arg2" };
        REQUIRE(CommandLineParser::ConcatenateParameters(params) == "test.passthrough_args=arg1 arg2");
    }

    SECTION("Test duration validation")
    {
        REQUIRE_NOTHROW(CommandLineParser::CheckTestDurationAndTimeout("test.test_duration=50", 100));
        REQUIRE_THROWS_AS(CommandLineParser::CheckTestDurationAndTimeout("test.test_duration=100", 50),
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
        std::string out = CommandLineParser::ConcatenateParameters(params);
        REQUIRE(out == "test1.param1=arg1;test1.param2=arg2;test1.param3=arg3;test2.param1=arg1");
    }

    SECTION("Missing '='")
    {
        std::vector<std::string> const params {
            "test1.param1:arg1",
        };
        REQUIRE_THROWS(CommandLineParser::ConcatenateParameters(params));
    }

    SECTION("Missing '.' is accepted")
    {
        // The diagnostic binary will reject this if the parameter name isn't a global parameter
        std::vector<std::string> const params {
            "genericMode=true",
            "diagnostic.test_duration=200",
        };
        std::string out = CommandLineParser::ConcatenateParameters(params);
        REQUIRE(out == "diagnostic.test_duration=200;genericMode=true");
    }

    SECTION("Unsupported multiple definitions")
    {
        std::vector<std::string> const params {
            "test1.param1=arg1",
            "test1.param1=arg2",
        };
        REQUIRE_THROWS(CommandLineParser::ConcatenateParameters(params));
    }

    SECTION("Supported multiple definitions")
    {
        std::vector<std::string> const params {
            "test1.passthrough_args=arg1",
            "test1.passthrough_args=arg2",
            "test1.normal_args=arg3",
            "test2.passthrough_args=arg4",
        };
        std::string out = CommandLineParser::ConcatenateParameters(params);
        REQUIRE(out == "test1.normal_args=arg3;test1.passthrough_args=arg1 arg2;test2.passthrough_args=arg4");
    }
}

TEST_CASE("CommandLineParser: ValidateParameters")
{
    SECTION("Basic")
    {
        std::string params = "diagnostic.test_duration=30;memtest.test_duration=10";
        REQUIRE_NOTHROW(CommandLineParser::ValidateParameters(params));
    }

    SECTION("Parameter length limit exceeded")
    {
        std::string params = "test1.param1=";
        std::string value(DCGM_MAX_TEST_PARMS_LEN_V2 - params.size(), '6');
        params += value;
        REQUIRE_THROWS(CommandLineParser::ValidateParameters(params));
    }
}

TEST_CASE("CommandLineParser: CheckGroupIdArgument")
{
    SECTION("Named built-in groups are accepted")
    {
        REQUIRE(CommandLineParser::CheckGroupIdArgument("g") == DCGM_GROUP_ALL_GPUS);
        REQUIRE(CommandLineParser::CheckGroupIdArgument("s") == DCGM_GROUP_ALL_NVSWITCHES);
        REQUIRE(CommandLineParser::CheckGroupIdArgument("i") == DCGM_GROUP_ALL_INSTANCES);
        REQUIRE(CommandLineParser::CheckGroupIdArgument("c") == DCGM_GROUP_ALL_COMPUTE_INSTANCES);
        REQUIRE(CommandLineParser::CheckGroupIdArgument("a") == DCGM_GROUP_ALL_ENTITIES);
    }

    SECTION("Numeric group ids are parsed")
    {
        REQUIRE(CommandLineParser::CheckGroupIdArgument("0") == 0);
        REQUIRE(CommandLineParser::CheckGroupIdArgument("17") == 17);
    }

    SECTION("Unsupported group ids throw")
    {
        REQUIRE_THROWS_AS(CommandLineParser::CheckGroupIdArgument("x"), TCLAP::CmdLineParseException);
        REQUIRE_THROWS_AS(CommandLineParser::CheckGroupIdArgument("-1"), TCLAP::CmdLineParseException);
    }
}

TEST_CASE("CommandLineParser: ValidateClocksEventMask")
{
    SECTION("Numeric masks accept valid bits and additive combinations")
    {
        REQUIRE_NOTHROW(CommandLineParser::ValidateClocksEventMask("0"));
        REQUIRE_NOTHROW(CommandLineParser::ValidateClocksEventMask("8"));
        REQUIRE_NOTHROW(CommandLineParser::ValidateClocksEventMask("232"));
    }

    SECTION("String masks accept known reason names case-insensitively")
    {
        REQUIRE_NOTHROW(CommandLineParser::ValidateClocksEventMask("hw_slowdown"));
        REQUIRE_NOTHROW(CommandLineParser::ValidateClocksEventMask("HW_SLOWDOWN,sw_thermal,hw_thermal,hw_power_brake"));
    }

    SECTION("Invalid masks throw")
    {
        REQUIRE_THROWS_AS(CommandLineParser::ValidateClocksEventMask("1"), TCLAP::CmdLineParseException);
        REQUIRE_THROWS_AS(CommandLineParser::ValidateClocksEventMask("8,hw_slowdown"), TCLAP::CmdLineParseException);
        REQUIRE_THROWS_AS(CommandLineParser::ValidateClocksEventMask("bad_reason"), TCLAP::CmdLineParseException);
        REQUIRE_THROWS_AS(CommandLineParser::ValidateClocksEventMask(std::string(DCGM_CLOCKS_EVENT_MASK_LEN, '8')),
                          TCLAP::CmdLineParseException);
    }
}

TEST_CASE("CommandLineParser: Host normalization helpers")
{
    SECTION("NormalizeIpAddress appends the default port")
    {
        std::string host = "node-a";

        CommandLineParser::NormalizeIpAddress(host, host);

        REQUIRE(host == "node-a:" + defaultPort);
    }

    SECTION("NormalizeIpAddress accepts boundary ports")
    {
        std::string host = "node-a:0";
        REQUIRE_NOTHROW(CommandLineParser::NormalizeIpAddress(host, host));
        CHECK(host == "node-a:0");

        host = "node-a:65535";
        REQUIRE_NOTHROW(CommandLineParser::NormalizeIpAddress(host, host));
        CHECK(host == "node-a:65535");
    }

    SECTION("NormalizeUnixSocketPath appends the default socket when no path is provided")
    {
        std::string host = "node-a:" + unixSocketPrefix;

        CommandLineParser::NormalizeUnixSocketPath(host, host);

        REQUIRE(host == "node-a:" + unixSocketPrefix + defaultSocketPath);
    }

    SECTION("NormalizeUnixSocketPath keeps an explicit absolute socket path")
    {
        std::string host = "node-a:" + unixSocketPrefix + "/var/run/dcgm.sock";

        CommandLineParser::NormalizeUnixSocketPath(host, host);

        REQUIRE(host == "node-a:" + unixSocketPrefix + "/var/run/dcgm.sock");
    }

    SECTION("NormalizeUnixSocketPath rejects embedded colons in socket paths")
    {
        std::string host = "node-a:" + unixSocketPrefix + "/var/run:dcgm.sock";

        REQUIRE_THROWS_AS(CommandLineParser::NormalizeUnixSocketPath(host, host), TCLAP::CmdLineParseException);
    }

    SECTION("NormalizeUnixSocketPath rejects missing host and relative paths")
    {
        std::string missingHost = ":" + unixSocketPrefix + "/var/run/dcgm.sock";
        REQUIRE_THROWS_AS(CommandLineParser::NormalizeUnixSocketPath(missingHost, missingHost),
                          TCLAP::CmdLineParseException);

        std::string relativePath = "node-a:" + unixSocketPrefix + "relative.sock";
        REQUIRE_THROWS_AS(CommandLineParser::NormalizeUnixSocketPath(relativePath, relativePath),
                          TCLAP::CmdLineParseException);
    }

    SECTION("NormalizeIpAddress rejects malformed host and port values")
    {
        std::string emptyHost;
        REQUIRE_THROWS_AS(CommandLineParser::NormalizeIpAddress(emptyHost, emptyHost), TCLAP::CmdLineParseException);

        std::string tooManyColons = "node-a:123:extra";
        REQUIRE_THROWS_AS(CommandLineParser::NormalizeIpAddress(tooManyColons, tooManyColons),
                          TCLAP::CmdLineParseException);

        std::string emptyPort = "node-a:";
        REQUIRE_THROWS_AS(CommandLineParser::NormalizeIpAddress(emptyPort, emptyPort), TCLAP::CmdLineParseException);

        std::string outOfRangePort = "node-a:18446744073709551616";
        REQUIRE_THROWS_AS(CommandLineParser::NormalizeIpAddress(outOfRangePort, outOfRangePort),
                          TCLAP::CmdLineParseException);
    }
}

TEST_CASE("CommandLineParser: HelperProcessWorkloadPowerProfileCommandLine Valid Index")
{
    auto bit = 7, groupId = 2;
    dcgmWorkloadPowerProfile_t mWorkloadPowerProfile {};
    SECTION("Default action - append")
    {
        mWorkloadPowerProfile = CommandLineParser::HelperProcessWorkloadPowerProfileCommandLine(bit, 'a', groupId);
        REQUIRE(mWorkloadPowerProfile.action == DCGM_WORKLOAD_PROFILE_ACTION_SET);
    }
    SECTION("action - clear")
    {
        mWorkloadPowerProfile = CommandLineParser::HelperProcessWorkloadPowerProfileCommandLine(bit, 'c', groupId);
        REQUIRE(mWorkloadPowerProfile.action == DCGM_WORKLOAD_PROFILE_ACTION_CLEAR);
    }
    SECTION("action - overwrite")
    {
        mWorkloadPowerProfile = CommandLineParser::HelperProcessWorkloadPowerProfileCommandLine(bit, 'o', groupId);
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
        mWorkloadPowerProfile = CommandLineParser::HelperProcessWorkloadPowerProfileCommandLine(bit, 'a', groupId);
        REQUIRE(mWorkloadPowerProfile.action == DCGM_WORKLOAD_PROFILE_ACTION_CLEAR);
    }
    SECTION("action - clear")
    {
        mWorkloadPowerProfile = CommandLineParser::HelperProcessWorkloadPowerProfileCommandLine(bit, 'c', groupId);
        REQUIRE(mWorkloadPowerProfile.action == DCGM_WORKLOAD_PROFILE_ACTION_CLEAR);
    }
    SECTION("action - overwrite")
    {
        mWorkloadPowerProfile = CommandLineParser::HelperProcessWorkloadPowerProfileCommandLine(bit, 'o', groupId);
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
        REQUIRE_THROWS(CommandLineParser::HelperProcessWorkloadPowerProfileCommandLine(bit, 'x', groupId));
    }
    SECTION("Invalid bit")
    {
        auto bit = 453;
        REQUIRE_THROWS(CommandLineParser::HelperProcessWorkloadPowerProfileCommandLine(bit, 'a', groupId));
    }
}

SCENARIO("MnDiag Command Line Arguments - invalid arguments")
{
    dcgmRunMnDiag_v1 drmnd = {};
    std::string hostEngineAddressValue;
    bool hostAddressWasOverridden;
    bool jsonOutput;

    SECTION("Invalid Unix socket path in hostList")
    {
        std::string hostList = "host1:" + unixSocketPrefix + "tmp/socket=0,1,2";
        char const *argv[]   = { "dcgmi", "mndiag", "--hostList", hostList.c_str() };
        int argc             = sizeof(argv) / sizeof(argv[0]);

        REQUIRE_THROWS_AS(CommandLineParser::ProcessAndValidateMnDiagParams(
                              argc, argv, drmnd, hostEngineAddressValue, hostAddressWasOverridden, jsonOutput),
                          TCLAP::CmdLineParseException);
    }

    SECTION("Missing hostname in Unix socket path")
    {
        std::string hostList = unixSocketPrefix + "/tmp/socket=0,1,2";
        char const *argv[]   = { "dcgmi", "mndiag", "--hostList", hostList.c_str() };
        int argc             = sizeof(argv) / sizeof(argv[0]);

        REQUIRE_THROWS_AS(CommandLineParser::ProcessAndValidateMnDiagParams(
                              argc, argv, drmnd, hostEngineAddressValue, hostAddressWasOverridden, jsonOutput),
                          TCLAP::CmdLineParseException);
    }

    SECTION("Invalid run level")
    {
        char const *argv[] = { "dcgmi", "mndiag", "--run", "-1", "--hostList", "host1=0,1,2" };
        int argc           = sizeof(argv) / sizeof(argv[0]);

        REQUIRE_THROWS_AS(CommandLineParser::ProcessAndValidateMnDiagParams(
                              argc, argv, drmnd, hostEngineAddressValue, hostAddressWasOverridden, jsonOutput),
                          TCLAP::CmdLineParseException);
    }

    SECTION("Invalid head node address")
    {
        char const *argv[]
            = { "dcgmi", "mndiag", "--run", "mnubergemm", "--hostEngineAddress", "", "--hostList", "host1=0,1,2" };
        int argc = sizeof(argv) / sizeof(argv[0]);
        REQUIRE_THROWS_AS(CommandLineParser::ProcessAndValidateMnDiagParams(
                              argc, argv, drmnd, hostEngineAddressValue, hostAddressWasOverridden, jsonOutput),
                          TCLAP::CmdLineParseException);
    }

    SECTION("Invalid host list")
    {
        char const *argv[] = { "dcgmi", "mndiag", "--run", "mnubergemm", "--hostList", "" };
        int argc           = sizeof(argv) / sizeof(argv[0]);
        REQUIRE_THROWS_AS(CommandLineParser::ProcessAndValidateMnDiagParams(
                              argc, argv, drmnd, hostEngineAddressValue, hostAddressWasOverridden, jsonOutput),
                          TCLAP::CmdLineParseException);
    }

    SECTION("Invalid parameters")
    {
        char const *argv[] = { "dcgmi",      "mndiag",      "--run",        "mnubergemm",
                               "--hostList", "host1=0,1,2", "--parameters", "invalid_param" };
        int argc           = sizeof(argv) / sizeof(argv[0]);
        REQUIRE_THROWS_AS(CommandLineParser::ProcessAndValidateMnDiagParams(
                              argc, argv, drmnd, hostEngineAddressValue, hostAddressWasOverridden, jsonOutput),
                          TCLAP::CmdLineParseException);
    }

    SECTION("Invalid GPU ID format in hostlist")
    {
        char const *argv[] = { "dcgmi", "mndiag", "--run", "mnubergemm", "--hostList", "localhost=invalid_gpu_id" };
        int argc           = sizeof(argv) / sizeof(argv[0]);
        REQUIRE_THROWS_AS(CommandLineParser::ProcessAndValidateMnDiagParams(
                              argc, argv, drmnd, hostEngineAddressValue, hostAddressWasOverridden, jsonOutput),
                          TCLAP::CmdLineParseException);
    }

    SECTION("Very large hostlist with many hosts (exceeding buffer capacity)")
    {
        // Create a very large hostlist string with 73 hosts, each with 1 GPU
        std::string largeHostlist = "";
        for (int i = 0; i < DCGM_MAX_NUM_HOSTS + 1; i++)
        {
            if (i > 0)
                largeHostlist += ";";
            largeHostlist += "host" + std::to_string(i) + "=0";
        }

        char const *argv[] = { "dcgmi", "mndiag", "--run", "mnubergemm", "--hostList", largeHostlist.c_str() };
        int argc           = sizeof(argv) / sizeof(argv[0]);

        // This should throw an exception due to the hostlist being too large
        REQUIRE_THROWS_AS(CommandLineParser::ProcessAndValidateMnDiagParams(
                              argc, argv, drmnd, hostEngineAddressValue, hostAddressWasOverridden, jsonOutput),
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
        REQUIRE_THROWS_AS(CommandLineParser::ProcessAndValidateMnDiagParams(
                              argc, argv, drmnd, hostEngineAddressValue, hostAddressWasOverridden, jsonOutput),
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
        REQUIRE_THROWS_AS(CommandLineParser::ProcessAndValidateMnDiagParams(
                              argc, argv, drmnd, hostEngineAddressValue, hostAddressWasOverridden, jsonOutput),
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
        REQUIRE_THROWS_AS(CommandLineParser::ProcessAndValidateMnDiagParams(
                              argc, argv, drmnd, hostEngineAddressValue, hostAddressWasOverridden, jsonOutput),
                          TCLAP::CmdLineParseException);
    }

    SECTION("Invalid format with no colon before unix:")
    {
        char const *argv[] = { "dcgmi", "mndiag", "--hostList", "hostunix:/tmp/socket=0,1,2" };
        int argc           = sizeof(argv) / sizeof(argv[0]);

        REQUIRE_THROWS_AS(CommandLineParser::ProcessAndValidateMnDiagParams(
                              argc, argv, drmnd, hostEngineAddressValue, hostAddressWasOverridden, jsonOutput),
                          TCLAP::CmdLineParseException);
    }

    SECTION("Invalid format with remainder not starting with unix:/")
    {
        char const *argv[] = { "dcgmi", "mndiag", "--hostList", "host1:unixpath/socket=0,1,2" };
        int argc           = sizeof(argv) / sizeof(argv[0]);

        REQUIRE_THROWS_AS(CommandLineParser::ProcessAndValidateMnDiagParams(
                              argc, argv, drmnd, hostEngineAddressValue, hostAddressWasOverridden, jsonOutput),
                          TCLAP::CmdLineParseException);
    }

    SECTION("Invalid local socket path not starting with /")
    {
        char const *argv[] = { "dcgmi", "mndiag", "--hostList", "host1:unix:/remote/socket:local_socket=0,1,2" };
        int argc           = sizeof(argv) / sizeof(argv[0]);

        REQUIRE_THROWS_AS(CommandLineParser::ProcessAndValidateMnDiagParams(
                              argc, argv, drmnd, hostEngineAddressValue, hostAddressWasOverridden, jsonOutput),
                          TCLAP::CmdLineParseException);
    }
}

TEST_CASE("CommandLineParser: ExpandRange")
{
    SECTION("Valid range expansion")
    {
        REQUIRE(CommandLineParser::ExpandRange("0-2") == "0,1,2");
        REQUIRE(CommandLineParser::ExpandRange("5-5") == "5");
        REQUIRE(CommandLineParser::ExpandRange("3-6") == "3,4,5,6");
    }

    SECTION("Invalid range format")
    {
        REQUIRE_THROWS_AS(CommandLineParser::ExpandRange("2-1"), TCLAP::CmdLineParseException);
        REQUIRE_THROWS_AS(CommandLineParser::ExpandRange("a-b"), TCLAP::CmdLineParseException);
        REQUIRE_THROWS_AS(CommandLineParser::ExpandRange("1-"), TCLAP::CmdLineParseException);
        REQUIRE_THROWS_AS(CommandLineParser::ExpandRange("-1"), TCLAP::CmdLineParseException);
        REQUIRE_THROWS_AS(CommandLineParser::ExpandRange("2a-26"), TCLAP::CmdLineParseException);
    }

    SECTION("Single number")
    {
        REQUIRE(CommandLineParser::ExpandRange("7") == "7");
    }
}

SCENARIO("MnDiag Command Line Arguments - Valid Cases")
{
    dcgmRunMnDiag_v1 drmnd = {};
    std::string hostEngineAddressValue;
    bool hostAddressWasOverridden;
    bool jsonOutput;

    SECTION("Required arguments validation")
    {
        char const *argv[] = { "dcgmi", "mndiag", "--hostList", "host1=0,1,2" };
        int argc           = sizeof(argv) / sizeof(argv[0]);

        REQUIRE_NOTHROW(CommandLineParser::ProcessAndValidateMnDiagParams(
            argc, argv, drmnd, hostEngineAddressValue, hostAddressWasOverridden, jsonOutput));
    }

    SECTION("Unix socket path in hostList")
    {
        std::string hostList = "host1:" + unixSocketPrefix + defaultSocketPath + "=0,1,2";
        char const *argv[]   = { "dcgmi", "mndiag", "--hostList", hostList.c_str() };
        int argc             = sizeof(argv) / sizeof(argv[0]);

        REQUIRE_NOTHROW(CommandLineParser::ProcessAndValidateMnDiagParams(
            argc, argv, drmnd, hostEngineAddressValue, hostAddressWasOverridden, jsonOutput));
    }

    SECTION("Remote Unix socket path in hostList")
    {
        std::string hostList = "remote:" + unixSocketPrefix + "/remote/socket=0,1,2";
        char const *argv[]   = { "dcgmi", "mndiag", "--hostList", hostList.c_str() };
        int argc             = sizeof(argv) / sizeof(argv[0]);

        REQUIRE_NOTHROW(CommandLineParser::ProcessAndValidateMnDiagParams(
            argc, argv, drmnd, hostEngineAddressValue, hostAddressWasOverridden, jsonOutput));
    }

    SECTION("Mixed Unix socket and TCP hosts in hostList")
    {
        std::string hostList = "host1:" + unixSocketPrefix + defaultSocketPath + "=0,1;host2=2,3";
        char const *argv[]   = { "dcgmi", "mndiag", "--hostList", hostList.c_str() };
        int argc             = sizeof(argv) / sizeof(argv[0]);

        REQUIRE_NOTHROW(CommandLineParser::ProcessAndValidateMnDiagParams(
            argc, argv, drmnd, hostEngineAddressValue, hostAddressWasOverridden, jsonOutput));
    }

    SECTION("Optional parameters with debug and config")
    {
        char const *argv[] = { "dcgmi",      "mndiag",      "--run",        "mnubergemm",
                               "--hostList", "host1=0,1,2", "--parameters", "test.param=value" };
        int argc           = sizeof(argv) / sizeof(argv[0]);

        REQUIRE_NOTHROW(CommandLineParser::ProcessAndValidateMnDiagParams(
            argc, argv, drmnd, hostEngineAddressValue, hostAddressWasOverridden, jsonOutput));
    }

    SECTION("JSON output flag parsing")
    {
        char const *argv[] = { "dcgmi", "mndiag", "--hostList", "host1=0,1,2", "--json" };
        int argc           = sizeof(argv) / sizeof(argv[0]);

        REQUIRE_NOTHROW(CommandLineParser::ProcessAndValidateMnDiagParams(
            argc, argv, drmnd, hostEngineAddressValue, hostAddressWasOverridden, jsonOutput));
        REQUIRE(jsonOutput == true);
    }

    SECTION("JSON output flag not specified defaults to false")
    {
        char const *argv[] = { "dcgmi", "mndiag", "--hostList", "host1=0,1,2" };
        int argc           = sizeof(argv) / sizeof(argv[0]);

        REQUIRE_NOTHROW(CommandLineParser::ProcessAndValidateMnDiagParams(
            argc, argv, drmnd, hostEngineAddressValue, hostAddressWasOverridden, jsonOutput));
        REQUIRE(jsonOutput == false);
    }

    SECTION("Short form of JSON flag (-j)")
    {
        char const *argv[] = { "dcgmi", "mndiag", "--hostList", "host1=0,1,2", "-j" };
        int argc           = sizeof(argv) / sizeof(argv[0]);

        REQUIRE_NOTHROW(CommandLineParser::ProcessAndValidateMnDiagParams(
            argc, argv, drmnd, hostEngineAddressValue, hostAddressWasOverridden, jsonOutput));
        REQUIRE(jsonOutput == true);
    }
}

TEST_CASE("CommandLineParser: ProcessCommandLine rejects invalid subsystem")
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

TEST_CASE("CommandLineParser: ProcessQueryCommandLine invalid argument combinations")
{
    GIVEN("query command line validation")
    {
        SECTION("Negative GPU id")
        {
            std::vector<char const *> argv { "dcgmi", "discovery", "--info", "a", "--gpuid", "-1" };

            WHEN("the query command is processed")
            {
                RequireParserThrows(CommandLineParser::ProcessQueryCommandLine, argv);
            }
        }

        SECTION("Group and GPU are mutually exclusive")
        {
            std::vector<char const *> argv { "dcgmi", "discovery", "--info", "a", "--group", "1", "--gpuid", "2" };

            WHEN("the query command is processed")
            {
                RequireParserThrows(CommandLineParser::ProcessQueryCommandLine, argv);
            }
        }

        SECTION("Group and CPU are mutually exclusive")
        {
            std::vector<char const *> argv { "dcgmi", "discovery", "--info", "a", "--group", "1", "--cpuid", "2" };

            WHEN("the query command is processed")
            {
                RequireParserThrows(CommandLineParser::ProcessQueryCommandLine, argv);
            }
        }

        SECTION("Compute hierarchy must be used alone")
        {
            std::vector<char const *> argv { "dcgmi", "discovery", "--compute-hierarchy", "--gpuid", "2" };

            WHEN("the query command is processed")
            {
                RequireParserThrows(CommandLineParser::ProcessQueryCommandLine, argv);
            }
        }
    }
}

TEST_CASE("CommandLineParser: ProcessQueryCommandLine valid execution branches")
{
    ScopedCommandExecutor executeHook;

    GIVEN("discovery command line actions that parse successfully")
    {
        SECTION("List branch")
        {
            std::vector<char const *> argv { "dcgmi", "discovery", "--list", "--all" };
            RequireParserOkAndExecuted(executeHook, CommandLineParser::ProcessQueryCommandLine, argv);
        }

        SECTION("Compute hierarchy branch")
        {
            std::vector<char const *> argv { "dcgmi", "discovery", "--compute-hierarchy" };
            RequireParserOkAndExecuted(executeHook, CommandLineParser::ProcessQueryCommandLine, argv);
        }

        SECTION("GPU info branch")
        {
            std::vector<char const *> argv { "dcgmi", "discovery", "--info", "aptcw", "--gpuid", "0" };
            RequireParserOkAndExecuted(executeHook, CommandLineParser::ProcessQueryCommandLine, argv);
        }

        SECTION("Group info branch")
        {
            std::vector<char const *> argv { "dcgmi", "discovery", "--info", "a", "--group", "0", "--verbose" };
            RequireParserOkAndExecuted(executeHook, CommandLineParser::ProcessQueryCommandLine, argv);
        }

        SECTION("CPU info branch")
        {
            std::vector<char const *> argv { "dcgmi", "discovery", "--info", "a", "--cpuid", "0" };
            RequireParserOkAndExecuted(executeHook, CommandLineParser::ProcessQueryCommandLine, argv);
        }

        SECTION("Default group info branch")
        {
            std::vector<char const *> argv { "dcgmi", "discovery", "--info", "a" };
            RequireParserOkAndExecuted(executeHook, CommandLineParser::ProcessQueryCommandLine, argv);
        }
    }
}

TEST_CASE("CommandLineParser: ProcessPolicyCommandLine invalid argument combinations")
{
    GIVEN("policy command line validation")
    {
        SECTION("Negative max pages")
        {
            std::vector<char const *> argv { "dcgmi", "policy", "--set", "0,1", "--maxpages", "-1" };

            WHEN("the policy command is processed")
            {
                RequireParserThrows(CommandLineParser::ProcessPolicyCommandLine, argv);
            }
        }

        SECTION("Set requires at least one condition")
        {
            std::vector<char const *> argv { "dcgmi", "policy", "--set", "0,1" };

            WHEN("the policy command is processed")
            {
                RequireParserThrows(CommandLineParser::ProcessPolicyCommandLine, argv);
            }
        }

        SECTION("Set action and validation must be CSV")
        {
            std::vector<char const *> argv { "dcgmi", "policy", "--set", "01", "--eccerrors" };

            WHEN("the policy command is processed")
            {
                RequireParserThrows(CommandLineParser::ProcessPolicyCommandLine, argv);
            }
        }

        SECTION("Set action must be 0 or 1")
        {
            std::vector<char const *> argv { "dcgmi", "policy", "--set", "2,1", "--eccerrors" };

            WHEN("the policy command is processed")
            {
                RequireParserThrows(CommandLineParser::ProcessPolicyCommandLine, argv);
            }
        }

        SECTION("Set validation must be between 0 and 3")
        {
            std::vector<char const *> argv { "dcgmi", "policy", "--set", "1,4", "--eccerrors" };

            WHEN("the policy command is processed")
            {
                RequireParserThrows(CommandLineParser::ProcessPolicyCommandLine, argv);
            }
        }

        SECTION("Verbose is only valid with get")
        {
            std::vector<char const *> argv { "dcgmi", "policy", "--set", "1,1", "--eccerrors", "--verbose" };

            WHEN("the policy command is processed")
            {
                RequireParserThrows(CommandLineParser::ProcessPolicyCommandLine, argv);
            }
        }
    }
}

TEST_CASE("CommandLineParser: ProcessPolicyCommandLine valid execution branches")
{
    ScopedCommandExecutor executeHook;

    GIVEN("policy command line actions that parse successfully")
    {
        SECTION("Get policy branch")
        {
            std::vector<char const *> argv { "dcgmi", "policy", "--get", "--json" };

            RequireParserOkAndExecuted(executeHook, CommandLineParser::ProcessPolicyCommandLine, argv);
        }

        SECTION("Clear policy branch")
        {
            std::vector<char const *> argv { "dcgmi", "policy", "--clear" };

            RequireParserOkAndExecuted(executeHook, CommandLineParser::ProcessPolicyCommandLine, argv);
        }

        SECTION("Set policy branch with every condition")
        {
            std::vector<char const *> argv { "dcgmi",          "policy",      "--set",      "1,3",
                                             "--eccerrors",    "--pcierrors", "--maxpages", "4",
                                             "--maxtemp",      "85",          "--maxpower", "300",
                                             "--nvlinkerrors", "--xiderrors" };

            RequireParserOkAndExecuted(executeHook, CommandLineParser::ProcessPolicyCommandLine, argv);
        }
    }
}

TEST_CASE("CommandLineParser: ProcessGroupCommandLine invalid argument combinations")
{
    GIVEN("group command line validation")
    {
        SECTION("Negative group id")
        {
            std::vector<char const *> argv { "dcgmi", "group", "--group", "-1", "--info" };

            WHEN("the group command is processed")
            {
                RequireParserThrows(CommandLineParser::ProcessGroupCommandLine, argv);
            }
        }

        SECTION("Group id requires an action")
        {
            std::vector<char const *> argv { "dcgmi", "group", "--group", "1" };

            WHEN("the group command is processed")
            {
                RequireParserThrows(CommandLineParser::ProcessGroupCommandLine, argv);
            }
        }

        SECTION("Add cannot be combined with info")
        {
            std::vector<char const *> argv { "dcgmi", "group", "--group", "1", "--info", "--add", "gpu:0" };

            WHEN("the group command is processed")
            {
                RequireParserThrows(CommandLineParser::ProcessGroupCommandLine, argv);
            }
        }

        SECTION("Remove cannot be combined with info")
        {
            std::vector<char const *> argv { "dcgmi", "group", "--group", "1", "--info", "--remove", "gpu:0" };

            WHEN("the group command is processed")
            {
                RequireParserThrows(CommandLineParser::ProcessGroupCommandLine, argv);
            }
        }

        SECTION("Add and remove are mutually exclusive")
        {
            std::vector<char const *> argv { "dcgmi", "group", "--group", "1", "--add", "gpu:0", "--remove", "gpu:1" };

            WHEN("the group command is processed")
            {
                RequireParserThrows(CommandLineParser::ProcessGroupCommandLine, argv);
            }
        }

        SECTION("Default GPU and default NvSwitch groups are mutually exclusive")
        {
            std::vector<char const *> argv {
                "dcgmi", "group", "--create", "mixed", "--default", "--defaultnvswitches"
            };

            WHEN("the group command is processed")
            {
                RequireParserThrows(CommandLineParser::ProcessGroupCommandLine, argv);
            }
        }
    }
}

TEST_CASE("CommandLineParser: ProcessGroupCommandLine valid execution branches")
{
    ScopedCommandExecutor executeHook;

    GIVEN("valid group command lines")
    {
        SECTION("list groups")
        {
            std::vector<char const *> argv { "dcgmi", "group", "--list", "--json" };

            RequireParserOkAndExecuted(executeHook, CommandLineParser::ProcessGroupCommandLine, argv);
        }

        SECTION("create default GPU group")
        {
            std::vector<char const *> argv { "dcgmi", "group", "--create", "gpus", "--default" };

            RequireParserOkAndExecuted(executeHook, CommandLineParser::ProcessGroupCommandLine, argv);
        }

        SECTION("create group with explicit entities")
        {
            std::vector<char const *> argv { "dcgmi", "group", "--create", "mixed", "--add", "gpu:0,cpu:0" };

            RequireParserOkAndExecuted(executeHook, CommandLineParser::ProcessGroupCommandLine, argv);
        }

        SECTION("delete group")
        {
            std::vector<char const *> argv { "dcgmi", "group", "--delete", "7" };

            RequireParserOkAndExecuted(executeHook, CommandLineParser::ProcessGroupCommandLine, argv);
        }

        SECTION("show group info")
        {
            std::vector<char const *> argv { "dcgmi", "group", "--group", "3", "--info", "--json" };

            RequireParserOkAndExecuted(executeHook, CommandLineParser::ProcessGroupCommandLine, argv);
        }

        SECTION("add and remove entities")
        {
            std::vector<char const *> addArgv { "dcgmi", "group", "--group", "3", "--add", "gpu:0" };
            std::vector<char const *> removeArgv { "dcgmi", "group", "--group", "3", "--remove", "gpu:0" };

            RequireParserOkAndExecuted(executeHook, CommandLineParser::ProcessGroupCommandLine, addArgv);
            RequireParserOkAndExecuted(executeHook, CommandLineParser::ProcessGroupCommandLine, removeArgv);
        }
    }
}

TEST_CASE("CommandLineParser: ProcessFieldGroupCommandLine invalid argument combinations")
{
    GIVEN("fieldgroup command line validation")
    {
        SECTION("Info requires field group id")
        {
            std::vector<char const *> argv { "dcgmi", "fieldgroup", "--info" };

            WHEN("the fieldgroup command is processed")
            {
                RequireParserThrows(CommandLineParser::ProcessFieldGroupCommandLine, argv);
            }
        }

        SECTION("Create requires field IDs")
        {
            std::vector<char const *> argv { "dcgmi", "fieldgroup", "--create", "fields" };

            WHEN("the fieldgroup command is processed")
            {
                RequireParserThrows(CommandLineParser::ProcessFieldGroupCommandLine, argv);
            }
        }

        SECTION("Delete requires field group id")
        {
            std::vector<char const *> argv { "dcgmi", "fieldgroup", "--delete" };

            WHEN("the fieldgroup command is processed")
            {
                RequireParserThrows(CommandLineParser::ProcessFieldGroupCommandLine, argv);
            }
        }
    }
}

TEST_CASE("CommandLineParser: ProcessFieldGroupCommandLine valid execution branches")
{
    ScopedCommandExecutor executeHook;

    GIVEN("valid fieldgroup command lines")
    {
        SECTION("list field groups")
        {
            std::vector<char const *> argv { "dcgmi", "fieldgroup", "--list", "--json" };

            RequireParserOkAndExecuted(executeHook, CommandLineParser::ProcessFieldGroupCommandLine, argv);
        }

        SECTION("create field group")
        {
            std::vector<char const *> argv { "dcgmi", "fieldgroup", "--create", "temps", "--fieldids", "150,155" };

            RequireParserOkAndExecuted(executeHook, CommandLineParser::ProcessFieldGroupCommandLine, argv);
        }

        SECTION("show and delete field group")
        {
            std::vector<char const *> infoArgv { "dcgmi", "fieldgroup", "--info", "--fieldgroup", "2", "--json" };
            std::vector<char const *> deleteArgv { "dcgmi", "fieldgroup", "--delete", "--fieldgroup", "2" };

            RequireParserOkAndExecuted(executeHook, CommandLineParser::ProcessFieldGroupCommandLine, infoArgv);
            RequireParserOkAndExecuted(executeHook, CommandLineParser::ProcessFieldGroupCommandLine, deleteArgv);
        }
    }
}

TEST_CASE("CommandLineParser: ProcessConfigCommandLine invalid argument combinations")
{
    GIVEN("config command line validation")
    {
        SECTION("Set requires at least one configuration option")
        {
            std::vector<char const *> argv { "dcgmi", "config", "--set" };

            WHEN("the config command is processed")
            {
                RequireParserThrows(CommandLineParser::ProcessConfigCommandLine, argv);
            }
        }

        SECTION("Set rejects malformed application clocks")
        {
            std::vector<char const *> argv { "dcgmi", "config", "--set", "--appclocks", "1000" };

            WHEN("the config command is processed")
            {
                RequireParserThrows(CommandLineParser::ProcessConfigCommandLine, argv);
            }
        }

        SECTION("Get rejects set-only options")
        {
            std::vector<char const *> argv { "dcgmi", "config", "--get", "--powerlimit", "250" };

            WHEN("the config command is processed")
            {
                RequireParserThrows(CommandLineParser::ProcessConfigCommandLine, argv);
            }
        }

        SECTION("Enforce rejects set-only options")
        {
            std::vector<char const *> argv { "dcgmi", "config", "--enforce", "--compmode", "1" };

            WHEN("the config command is processed")
            {
                RequireParserThrows(CommandLineParser::ProcessConfigCommandLine, argv);
            }
        }
    }
}

TEST_CASE("CommandLineParser: ProcessConfigCommandLine valid execution branches")
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

            RequireParserOkAndExecuted(executeHook, CommandLineParser::ProcessConfigCommandLine, argv);
        }

        SECTION("set workload power profile")
        {
            std::vector<char const *> argv { "dcgmi", "config",
                                             "--set", "--group",
                                             "3",     "--workloadpowerprofile",
                                             "4",     "--workloadpowerprofileaction",
                                             "o" };

            RequireParserOkAndExecuted(executeHook, CommandLineParser::ProcessConfigCommandLine, argv, 2);
        }

        SECTION("get and enforce config")
        {
            std::vector<char const *> getArgv { "dcgmi", "config", "--get", "--group", "3", "--verbose", "--json" };
            std::vector<char const *> enforceArgv { "dcgmi", "config", "--enforce", "--group", "3" };

            RequireParserOkAndExecuted(executeHook, CommandLineParser::ProcessConfigCommandLine, getArgv);
            RequireParserOkAndExecuted(executeHook, CommandLineParser::ProcessConfigCommandLine, enforceArgv);
        }
    }
}

TEST_CASE("CommandLineParser: ProcessHealthCommandLine invalid argument combinations")
{
    GIVEN("health command line validation")
    {
        SECTION("Invalid group alias is rejected")
        {
            std::vector<char const *> argv { "dcgmi", "health", "--fetch", "--group", "unknown" };

            WHEN("the health command is processed")
            {
                RequireParserThrows(CommandLineParser::ProcessHealthCommandLine, argv);
            }
        }

        SECTION("Duplicate set flags are rejected")
        {
            std::vector<char const *> argv { "dcgmi", "health", "--set", "pp" };

            WHEN("the health command is processed")
            {
                RequireParserThrows(CommandLineParser::ProcessHealthCommandLine, argv);
            }
        }

        SECTION("All-watch flag cannot be combined with specific flags")
        {
            std::vector<char const *> argv { "dcgmi", "health", "--set", "ap" };

            WHEN("the health command is processed")
            {
                RequireParserThrows(CommandLineParser::ProcessHealthCommandLine, argv);
            }
        }

        SECTION("Unknown set flags are rejected")
        {
            std::vector<char const *> argv { "dcgmi", "health", "--set", "z" };

            WHEN("the health command is processed")
            {
                RequireParserThrows(CommandLineParser::ProcessHealthCommandLine, argv);
            }
        }
    }
}

TEST_CASE("CommandLineParser: ProcessHealthCommandLine valid execution branches")
{
    ScopedCommandExecutor executeHook;

    GIVEN("valid health command lines")
    {
        SECTION("fetch and check watches")
        {
            std::vector<char const *> fetchArgv { "dcgmi", "health", "--fetch", "--group", "g", "--json" };
            std::vector<char const *> checkArgv { "dcgmi", "health", "--check", "--group", "a", "--json" };

            RequireParserOkAndExecuted(executeHook, CommandLineParser::ProcessHealthCommandLine, fetchArgv);
            RequireParserOkAndExecuted(executeHook, CommandLineParser::ProcessHealthCommandLine, checkArgv);
        }

        SECTION("clear and set watches")
        {
            std::vector<char const *> clearArgv { "dcgmi", "health", "--clear", "--group", "3" };
            std::vector<char const *> setArgv { "dcgmi", "health",         "--set", "pmitndx",           "--group",
                                                "3",     "--max-keep-age", "60",    "--update-interval", "10" };
            std::vector<char const *> setAllArgv { "dcgmi", "health", "--set", "a", "--group", "3" };

            RequireParserOkAndExecuted(executeHook, CommandLineParser::ProcessHealthCommandLine, clearArgv);
            RequireParserOkAndExecuted(executeHook, CommandLineParser::ProcessHealthCommandLine, setArgv);
            RequireParserOkAndExecuted(executeHook, CommandLineParser::ProcessHealthCommandLine, setAllArgv);
        }
    }
}

TEST_CASE("CommandLineParser: ProcessProfileCommandLine invalid argument combinations")
{
    GIVEN("profile command line validation")
    {
        SECTION("Entity and group IDs are mutually exclusive")
        {
            std::vector<char const *> argv { "dcgmi", "profile", "--list", "--entity-id", "0", "--group-id", "1" };

            WHEN("the profile command is processed")
            {
                RequireParserThrows(CommandLineParser::ProcessProfileCommandLine, argv);
            }
        }
    }
}

TEST_CASE("CommandLineParser: valid execution branches for monitoring subsystems")
{
    ScopedCommandExecutor executeHook;

    GIVEN("valid topology, nvlink, dmon, and profile command lines")
    {
        SECTION("topology commands")
        {
            std::vector<char const *> groupArgv { "dcgmi", "topo", "--group", "3", "--json" };
            std::vector<char const *> gpuArgv { "dcgmi", "topo", "--gpuid", "0", "--json" };

            RequireParserOkAndExecuted(executeHook, CommandLineParser::ProcessTopoCommandLine, groupArgv);
            RequireParserOkAndExecuted(executeHook, CommandLineParser::ProcessTopoCommandLine, gpuArgv);
        }

        SECTION("nvlink commands")
        {
            std::vector<char const *> errorsArgv { "dcgmi", "nvlink", "--errors", "--gpuid", "0", "--json" };
            std::vector<char const *> statusArgv { "dcgmi", "nvlink", "--link-status", "--show-entity-ids" };

            RequireParserOkAndExecuted(executeHook, CommandLineParser::ProcessNvlinkCommandLine, errorsArgv);
            RequireParserOkAndExecuted(executeHook, CommandLineParser::ProcessNvlinkCommandLine, statusArgv);
        }

        SECTION("device monitor commands")
        {
            std::vector<char const *> listArgv { "dcgmi", "dmon", "--list" };
            std::vector<char const *> fieldsArgv { "dcgmi", "dmon",    "--entity-id", "gpu:0",   "--field-id",
                                                   "150",   "--delay", "1",           "--count", "1" };

            RequireParserOkAndExecuted(executeHook, CommandLineParser::ProcessDmonCommandLine, listArgv);
            RequireParserOkAndExecuted(executeHook, CommandLineParser::ProcessDmonCommandLine, fieldsArgv);
        }

        SECTION("profile commands")
        {
            std::vector<char const *> listArgv { "dcgmi", "profile", "--list", "--entity-id", "gpu:0", "--json" };
            std::vector<char const *> pauseArgv { "dcgmi", "profile", "--pause" };
            std::vector<char const *> resumeArgv { "dcgmi", "profile", "--resume" };

            RequireParserOkAndExecuted(executeHook, CommandLineParser::ProcessProfileCommandLine, listArgv);
            RequireParserOkAndExecuted(executeHook, CommandLineParser::ProcessProfileCommandLine, pauseArgv);
            RequireParserOkAndExecuted(executeHook, CommandLineParser::ProcessProfileCommandLine, resumeArgv);
        }
    }
}

TEST_CASE("CommandLineParser: ProcessSettingsCommandLine invalid argument combinations")
{
    GIVEN("settings command line validation")
    {
        SECTION("Attach and detach are mutually exclusive")
        {
            std::vector<char const *> argv { "dcgmi", "set", "--attach-driver", "--detach-driver" };

            WHEN("the settings command is processed")
            {
                RequireParserThrows(CommandLineParser::ProcessSettingsCommandLine, argv);
            }
        }

        SECTION("At least one setting action is required")
        {
            std::vector<char const *> argv { "dcgmi", "set" };

            WHEN("the settings command is processed")
            {
                RequireParserThrows(CommandLineParser::ProcessSettingsCommandLine, argv);
            }
        }
    }
}

TEST_CASE("CommandLineParser: valid execution branches for settings and modules")
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

            RequireParserOkAndExecuted(executeHook, CommandLineParser::ProcessSettingsCommandLine, severityArgv);
            RequireParserOkAndExecuted(executeHook, CommandLineParser::ProcessSettingsCommandLine, attachArgv);
            RequireParserOkAndExecuted(executeHook, CommandLineParser::ProcessSettingsCommandLine, detachArgv);
        }

        SECTION("module commands")
        {
            std::vector<char const *> listArgv { "dcgmi", "modules", "--list", "--json" };
            std::vector<char const *> denylistArgv { "dcgmi", "modules", "--denylist", "profiling", "--json" };

            RequireParserOkAndExecuted(executeHook, CommandLineParser::ProcessModuleCommandLine, listArgv);
            RequireParserOkAndExecuted(executeHook, CommandLineParser::ProcessModuleCommandLine, denylistArgv);
        }
    }
}

TEST_CASE("CommandLineParser: ProcessAdminCommandLine invalid argument combinations")
{
    GIVEN("admin command line validation")
    {
        SECTION("Negative group id")
        {
            std::vector<char const *> argv { "dcgmi", "test", "--introspect", "--group", "-1" };

            WHEN("the admin command is processed")
            {
                RequireParserThrows(CommandLineParser::ProcessAdminCommandLine, argv);
            }
        }

        SECTION("GPU and group are mutually exclusive")
        {
            std::vector<char const *> argv { "dcgmi", "test", "--introspect", "--gpuid", "1", "--group", "2" };

            WHEN("the admin command is processed")
            {
                RequireParserThrows(CommandLineParser::ProcessAdminCommandLine, argv);
            }
        }

        SECTION("Field is required for introspection")
        {
            std::vector<char const *> argv { "dcgmi", "test", "--introspect" };

            WHEN("the admin command is processed")
            {
                RequireParserThrows(CommandLineParser::ProcessAdminCommandLine, argv);
            }
        }

        SECTION("Injection cannot target a group")
        {
            std::vector<char const *> argv { "dcgmi",   "test", "--inject", "--group", "2",
                                             "--field", "100",  "--value",  "1" };

            WHEN("the admin command is processed")
            {
                RequireParserThrows(CommandLineParser::ProcessAdminCommandLine, argv);
            }
        }
    }
}

TEST_CASE("CommandLineParser: ProcessAdminCommandLine valid execution branches")
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

            RequireParserOkAndExecuted(executeHook, CommandLineParser::ProcessAdminCommandLine, introspectGpuArgv);
            RequireParserOkAndExecuted(executeHook, CommandLineParser::ProcessAdminCommandLine, introspectGroupArgv);
            RequireParserOkAndExecuted(executeHook, CommandLineParser::ProcessAdminCommandLine, injectArgv);
        }

        SECTION("pause and resume")
        {
            std::vector<char const *> pauseArgv { "dcgmi", "test", "--pause" };
            std::vector<char const *> resumeArgv { "dcgmi", "test", "--resume" };

            RequireParserOkAndExecuted(executeHook, CommandLineParser::ProcessAdminCommandLine, pauseArgv);
            RequireParserOkAndExecuted(executeHook, CommandLineParser::ProcessAdminCommandLine, resumeArgv);
        }
    }
}

