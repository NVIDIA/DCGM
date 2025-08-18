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
#include <catch2/catch_all.hpp>

#include <DcgmBuildInfo.hpp>
#include <DcgmStringHelpers.h>
#include <dcgm_errors.h>
#include <dcgm_structs.h>

#define DCGMI_TESTS
#include <CommandLineParser.h>
#include <NvcmTCLAP.h>
#include <tclap/ArgException.h>

// -----------------------------------------------------------------
namespace
{
std::string const defaultPort       = std::to_string(DCGM_HE_PORT_NUMBER);
std::string const defaultSocketPath = DCGM_DEFAULT_SOCKET_PATH;
std::string const unixSocketPrefix  = DCGM_UNIX_SOCKET_PREFIX;
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