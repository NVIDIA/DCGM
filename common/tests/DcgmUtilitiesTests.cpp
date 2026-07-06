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

#include <DcgmStringHelpers.h>
#include <DcgmUtilities.h>

#include <DcgmException.hpp>
#include <Defer.hpp>
#include <chrono>
#include <csignal>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <unordered_set>

#include <catch2/catch_all.hpp>

TEST_CASE("Utils: GetMaxAge")
{
    using namespace std::chrono;
    using namespace std::chrono_literals;

    auto maxAge = DcgmNs::Utils::GetMaxAge(100ms, 10s, 10);
    REQUIRE(maxAge == 10s);

    maxAge = DcgmNs::Utils::GetMaxAge(100ms, 10s, 10, 2);
    REQUIRE(maxAge == 10s);

    maxAge = DcgmNs::Utils::GetMaxAge(0ms, 10s, 10);
    REQUIRE(maxAge == 10s);

    maxAge = DcgmNs::Utils::GetMaxAge(0ms, 0s, 0);
    REQUIRE(maxAge == 1s);

    maxAge = DcgmNs::Utils::GetMaxAge(10ms, 0s, 10);
    REQUIRE(maxAge == 1s);

    maxAge = DcgmNs::Utils::GetMaxAge(10ms, 400ms, 0);
    REQUIRE(maxAge == 1s);

    maxAge = DcgmNs::Utils::GetMaxAge(10ms, 400ms, 10);
    REQUIRE(maxAge == 1s);
}

TEST_CASE("Utils: EraseIf (map)")
{
    auto container = std::map<std::string, int> { { "hello", 1 }, { "world", 2 } };
    auto removed   = DcgmNs::Utils::EraseIf(container, [](auto const &pair) { return pair.first == "hello"; });

    REQUIRE(removed == 1);
    REQUIRE(container.begin()->second == 2);
}

TEST_CASE("Utils: EraseIf (set)")
{
    auto container = std::set<std::string> { { "hello" }, { "world" } };
    auto removed   = DcgmNs::Utils::EraseIf(container, [](auto const &str) { return str == "hello"; });

    REQUIRE(removed == 1);
    REQUIRE(*container.begin() == "world");
}

TEST_CASE("Utils: EraseIf (unordered_map)")
{
    auto container = std::unordered_map<std::string, int> { { "hello", 1 }, { "world", 2 } };
    auto removed   = DcgmNs::Utils::EraseIf(container, [](auto const &pair) { return pair.first == "hello"; });

    REQUIRE(removed == 1);
    REQUIRE(container.begin()->second == 2);
}

TEST_CASE("Utils: EraseIf (unordered_set)")
{
    auto container = std::unordered_set<std::string> { { "hello" }, { "world" } };
    auto removed   = DcgmNs::Utils::EraseIf(container, [](auto const &str) { return str == "hello"; });

    REQUIRE(removed == 1);
    REQUIRE(*container.begin() == "world");
}

TEST_CASE("Utils: EraseIf (vector)")
{
    auto container = std::vector<std::string> { "hello", "world" };
    auto removed   = DcgmNs::Utils::EraseIf(container, [](auto const &str) { return str == "hello"; });

    REQUIRE(removed == 1);
    REQUIRE(*container.begin() == "world");
}
TEST_CASE("DcgmException")
{
    try
    {
        REQUIRE(DcgmNs::Utils::NvmlReturnToDcgmReturn(NVML_ERROR_NO_PERMISSION) == DCGM_ST_NO_PERMISSION);
        throw DcgmNs::DcgmException(DcgmNs::Utils::NvmlReturnToDcgmReturn(NVML_ERROR_NO_PERMISSION));
    }
    catch (DcgmNs::DcgmException const &ex)
    {
        REQUIRE(ex.what() != nullptr);
    }

    try
    {
        throw DcgmNs::DcgmException(dcgmReturn_t(1));
    }
    catch (DcgmNs::DcgmException const &ex)
    {
        REQUIRE(ex.what() == nullptr);
    }
}

TEST_CASE("Dcgmi Config: Bitmask helper")
{
    unsigned int mask[DCGM_POWER_PROFILE_ARRAY_SIZE] = {};

    SECTION("multiple bits")
    {
        for (unsigned int i = 0; i < DCGM_POWER_PROFILE_ARRAY_SIZE; i++)
        {
            mask[i] |= (1 << 10);
            mask[i] |= (1 << 20);
            mask[i] |= (1 << 30);
        }

        auto result = DcgmNs::Utils::HelperDisplayPowerBitmask(mask);
        REQUIRE(result == "10,20,30,42,52,62,74,84,94,106,116,126,138,148,158,170,180,190,202,212,222,234,244,254");
    }

    SECTION("empty")
    {
        std::ranges::fill_n(mask, std::size(mask), DCGM_INT32_BLANK);

        auto result = DcgmNs::Utils::HelperDisplayPowerBitmask(mask);
        REQUIRE(result == "Not Specified");
    }
}

namespace
{
/* Some systems are configured to use non-local sources for passwd. The queries are probably
 * undesirible in those cases, and can also result in test failure.
 */
bool PasswdIsFilesOnly()
{
    std::ifstream nsswitch("/etc/nsswitch.conf");
    if (!nsswitch)
        return true; // If we can't read it, assume local only
    std::string line;
    while (std::getline(nsswitch, line))
    {
        auto hash = line.find('#');
        if (hash != std::string::npos)
            line = line.substr(0, hash);
        auto pos = line.find("passwd:");
        if (pos != std::string::npos)
        {
            std::string rest = line.substr(pos + 7);
            // Remove leading/trailing whitespace
            rest.erase(0, rest.find_first_not_of(" \t"));
            rest.erase(rest.find_last_not_of(" \t") + 1);
            return rest == "files";
        }
    }
    return true; // Assume local only if not found
}
} // namespace

TEST_CASE("GetUserCredentials")
{
    if (!PasswdIsFilesOnly())
    {
        SKIP("Skipping test: /etc/nsswitch.conf passwd is not 'files' only");
    }

    SECTION("nullptr username handling")
    {
        auto result = GetUserCredentials(nullptr);
        REQUIRE_FALSE(result.has_value());
    }

    SECTION("non-existent user handling")
    {
        auto result = GetUserCredentials("non_existent_user_12345");
        REQUIRE_FALSE(result.has_value());
    }

    SECTION("valid user handling")
    {
        auto result = GetUserCredentials("nobody");
        REQUIRE(result.has_value());
        // Note: We don't check specific uid/gid as they might vary by system
        REQUIRE(result->uid != 0); // nobody should never be root
        REQUIRE(result->gid != 0);
    }
}

TEST_CASE("RunCmdAndGetOutput")
{
    class TestRunCmdHelper : public DcgmNs::Utils::RunCmdHelper
    {
    public:
        std::vector<std::string> GetTokenizedArgs(std::string const &cmd) const
        {
            return dcgmTokenizeString(cmd, " ");
        }

        using DcgmNs::Utils::RunCmdHelper::RunCmdAndGetOutput;
    };

    TestRunCmdHelper helper;

    SECTION("Basic command parsing")
    {
        auto tokens = helper.GetTokenizedArgs("ls -la /tmp");
        REQUIRE(tokens.size() == 3);
        REQUIRE(tokens[0] == "ls");
        REQUIRE(tokens[1] == "-la");
        REQUIRE(tokens[2] == "/tmp");
    }

    SECTION("Unquoted spaces are condensed")
    {
        SKIP("Current implementation does not condense unquoted spaces");
        auto tokens = helper.GetTokenizedArgs("ls  -la   /tmp");
        REQUIRE(tokens.size() == 5);
        REQUIRE(tokens[0] == "ls");
        REQUIRE(tokens[1] == "");
        REQUIRE(tokens[2] == "-la");
        REQUIRE(tokens[3] == "");
        REQUIRE(tokens[4] == "/tmp");
    }

    SECTION("Quoted spaces are retained")
    {
        SKIP("Current implementation does not retain quoted spaces");
        auto tokens = helper.GetTokenizedArgs("echo \"Hello World\"");
        REQUIRE(tokens.size() == 2);
        REQUIRE(tokens[0] == "echo");
        REQUIRE(tokens[1] == "\"Hello World\"");
        // Note that "World\"" is not included - showing the issue with space splitting
    }

    SECTION("Command with many arguments")
    {
        auto tokens = helper.GetTokenizedArgs("command arg1 arg2 arg3 arg4 arg5");
        REQUIRE(tokens.size() == 6);
        REQUIRE(tokens[0] == "command");
        REQUIRE(tokens[5] == "arg5");
    }
}

TEST_CASE("Utils: RunCmdAndGetOutputWithTimeout")
{
    SECTION("successful command captures stdout")
    {
        std::string output;

        auto result
            = DcgmNs::Utils::RunCmdAndGetOutputWithTimeout("/bin/echo dcgm", output, std::chrono::seconds { 1 });

        REQUIRE(result == DCGM_ST_OK);
        REQUIRE(output == "dcgm\n");
    }

    SECTION("non-zero exit is reported as initialization error")
    {
        std::string output;

        auto result = DcgmNs::Utils::RunCmdAndGetOutputWithTimeout("/bin/false", output, std::chrono::seconds { 1 });

        REQUIRE(result == DCGM_ST_INIT_ERROR);
    }

    SECTION("slow command returns timeout")
    {
        std::string output;

        auto result
            = DcgmNs::Utils::RunCmdAndGetOutputWithTimeout("/bin/sleep 1", output, std::chrono::milliseconds { 1 });

        REQUIRE(result == DCGM_ST_TIMEOUT);
    }
}

TEST_CASE("Utils: RunCmdHelper wrappers")
{
    DcgmNs::Utils::RunCmdHelper helper;
    std::string output;

    SECTION("GIVEN helper wrapper WHEN command succeeds without timeout THEN output is returned")
    {
        REQUIRE(helper.RunCmdAndGetOutput("/bin/echo wrapper", output) == DCGM_ST_OK);
        CHECK(output == "wrapper\n");
    }

    SECTION("GIVEN helper wrapper WHEN command succeeds with timeout THEN output is returned")
    {
        REQUIRE(helper.RunCmdAndGetOutputWithTimeout("/bin/echo timed", output, std::chrono::seconds { 1 })
                == DCGM_ST_OK);
        CHECK(output == "timed\n");
    }
}

TEST_CASE("Utils: FileHandle ownership")
{
    SECTION("Release abandons ownership")
    {
        int fds[2];
        REQUIRE(pipe(fds) == 0);

        DcgmNs::Utils::FileHandle readFd(fds[0]);
        DcgmNs::Utils::FileHandle writeFd(fds[1]);

        int rawReadFd = readFd.Release();
        REQUIRE(rawReadFd == fds[0]);
        CHECK(readFd.Get() == -1);

        REQUIRE(close(rawReadFd) == 0);
    }

    SECTION("Move assignment swaps ownership")
    {
        int firstPipe[2];
        int secondPipe[2];
        REQUIRE(pipe(firstPipe) == 0);
        REQUIRE(pipe(secondPipe) == 0);

        DcgmNs::Utils::FileHandle firstRead(firstPipe[0]);
        DcgmNs::Utils::FileHandle firstWrite(firstPipe[1]);
        DcgmNs::Utils::FileHandle secondRead(secondPipe[0]);
        DcgmNs::Utils::FileHandle secondWrite(secondPipe[1]);

        int originalFirstRead  = firstRead.Get();
        int originalSecondRead = secondRead.Get();

        firstRead = std::move(secondRead);

        CHECK(firstRead.Get() == originalSecondRead);
        CHECK(secondRead.Get() == originalFirstRead);
    }

    SECTION("GIVEN invalid handle WHEN reading or writing THEN EBADF is reported")
    {
        DcgmNs::Utils::FileHandle handle;
        std::array<std::byte, 4> buffer {};

        CHECK(handle.ReadExact(buffer.data(), buffer.size()) == -1);
        CHECK(handle.GetErrno() == EBADF);
        CHECK(handle.Write("x", 1) == -1);
        CHECK(handle.GetErrno() == EBADF);
    }

    SECTION("GIVEN pipe-backed handles WHEN reading and writing THEN exact data is transferred")
    {
        int fds[2];
        REQUIRE(pipe(fds) == 0);
        DcgmNs::Utils::FileHandle readFd(fds[0]);
        DcgmNs::Utils::FileHandle writeFd(fds[1]);
        std::array<std::byte, 4> buffer {};

        REQUIRE(writeFd.Write("dcgm", 4) == 4);
        REQUIRE(readFd.ReadExact(buffer.data(), buffer.size()) == 4);
        CHECK(std::string(reinterpret_cast<char *>(buffer.data()), buffer.size()) == "dcgm");
    }

    SECTION("GIVEN EOF before requested bytes WHEN reading exact data THEN zero is returned")
    {
        int fds[2];
        REQUIRE(pipe(fds) == 0);
        DcgmNs::Utils::FileHandle readFd(fds[0]);
        DcgmNs::Utils::FileHandle writeFd(fds[1]);
        std::array<std::byte, 4> buffer {};

        writeFd = DcgmNs::Utils::FileHandle {};

        CHECK(readFd.ReadExact(buffer.data(), buffer.size()) == 0);
    }
}

TEST_CASE("Utils: PipePair helpers")
{
    SECTION("GIVEN blocking pipe pair WHEN endpoints are borrowed and given up THEN ownership moves")
    {
        auto pipePair = DcgmNs::Utils::PipePair::Create(DcgmNs::Utils::PipePair::BlockingType::Blocking);
        REQUIRE(pipePair != nullptr);
        REQUIRE(pipePair->BorrowSender().Get() >= 0);
        REQUIRE(pipePair->BorrowReceiver().Get() >= 0);

        auto sender   = pipePair->GiveupSender();
        auto receiver = pipePair->GiveupReceiver();

        CHECK(sender.Get() >= 0);
        CHECK(receiver.Get() >= 0);
        CHECK(pipePair->BorrowSender().Get() == -1);
        CHECK(pipePair->BorrowReceiver().Get() == -1);
    }

    SECTION("GIVEN nonblocking pipe pair WHEN receiver is empty THEN read reports EAGAIN")
    {
        auto pipePair = DcgmNs::Utils::PipePair::Create(DcgmNs::Utils::PipePair::BlockingType::NonBlocking);
        REQUIRE(pipePair != nullptr);
        std::array<std::byte, 1> buffer {};
        auto receiver = pipePair->GiveupReceiver();

        CHECK(receiver.ReadExact(buffer.data(), buffer.size()) == -1);
        CHECK(receiver.GetErrno() == EAGAIN);

        pipePair->CloseSender();
        CHECK(pipePair->BorrowSender().Get() == -1);
        CHECK(pipePair->BorrowReceiver().Get() == -1);
    }
}

TEST_CASE("Utils: dynamic loading and low-level IO")
{
    SECTION("GIVEN missing symbol WHEN loading function THEN nullptr is returned")
    {
        void *handle = dlopen(nullptr, RTLD_LAZY);
        REQUIRE(handle != nullptr);

        CHECK(DcgmNs::Utils::LoadFunction(handle, "dcgm_missing_symbol_for_test", "self") == nullptr);
    }

    SECTION("GIVEN writable pipe WHEN WriteAll is called THEN all bytes are written")
    {
        int fds[2];
        REQUIRE(pipe(fds) == 0);
        DcgmNs::Utils::FileHandle readFd(fds[0]);
        DcgmNs::Utils::FileHandle writeFd(fds[1]);
        std::array<char, 5> buffer {};

        REQUIRE(DcgmNs::Utils::WriteAll(writeFd.Get(), "hello", 5) == 0);
        REQUIRE(read(readFd.Get(), buffer.data(), buffer.size()) == static_cast<ssize_t>(buffer.size()));
        CHECK(std::string(buffer.data(), buffer.size()) == "hello");
    }
}

TEST_CASE("Utils: BER and log path helpers")
{
    SECTION("GIVEN BER values WHEN parsed and checked THEN threshold rules are applied")
    {
        CHECK_FALSE(DcgmNs::Utils::IsEffectiveBerThresholdExceeded(15, 255));
        CHECK_FALSE(DcgmNs::Utils::IsEffectiveBerThresholdExceeded(1, 12));
        CHECK(DcgmNs::Utils::IsEffectiveBerThresholdExceeded(2, 12));
        CHECK_FALSE(DcgmNs::Utils::IsEffectiveBerThresholdExceeded(10, 13));
        CHECK(DcgmNs::Utils::IsEffectiveBerThresholdExceeded(11, 13));
        CHECK(DcgmNs::Utils::IsEffectiveBerThresholdExceeded(1, 11));

        auto const [mantissa, exponent, parsed] = DcgmNs::Utils::NvmlBerParser(0);
        CHECK(mantissa == 0);
        CHECK(exponent == 0);
        CHECK(parsed == 0.0);
    }
}

TEST_CASE("Utils: FindExecutable", "[DcgmUtilities]")
{
    // Create a temporary directory and executable for testing
    std::filesystem::path tempDir = std::filesystem::current_path() / "tmp";
    std::filesystem::create_directories(tempDir);
    std::filesystem::path testExecutable = tempDir / "test_exe";
    std::ofstream { testExecutable }.close();
    std::filesystem::permissions(
        testExecutable, std::filesystem::perms::owner_exec, std::filesystem::perm_options::add);

    struct TestCase
    {
        std::string name;
        std::vector<std::string> searchPaths;
        bool expectSuccess;
        std::string expectedPath;
        std::string expectedDir;
        dcgmReturn_t expectedError;
    };

    std::vector<TestCase> testCases
        = { { "Executable found in search paths",
              { tempDir.string() },
              true,
              testExecutable.string(),
              tempDir.string(),
              DCGM_ST_OK },
            { "Executable not found returns error", { "/nonexistent/path" }, false, "", "", DCGM_ST_NO_DATA },
            { "Multiple search paths - found in second path",
              { "/nonexistent", tempDir.string() },
              true,
              testExecutable.string(),
              tempDir.string(),
              DCGM_ST_OK } };

    for (const auto &tc : testCases)
    {
        SECTION(tc.name)
        {
            std::string executableDir;
            auto result = DcgmNs::Utils::FindExecutable("test_exe", tc.searchPaths, executableDir);

            REQUIRE(result.has_value() == tc.expectSuccess);
            if (result.has_value())
            {
                REQUIRE(result.value() == tc.expectedPath);
                REQUIRE(executableDir == tc.expectedDir);
            }
            else
            {
                REQUIRE(result.error() == tc.expectedError);
            }
        }
    }

    // Cleanup
    std::filesystem::remove_all(tempDir);
}

TEST_CASE("Utils: IsProcessRunning", "[DcgmUtilities]")
{
    SECTION("Invalid PIDs return false")
    {
        REQUIRE_FALSE(DcgmNs::Utils::IsProcessRunning(-1));
        REQUIRE_FALSE(DcgmNs::Utils::IsProcessRunning(0));
    }

    SECTION("Non-existent PID returns false")
    {
        REQUIRE_FALSE(DcgmNs::Utils::IsProcessRunning(999999));
    }

    SECTION("Current process is running")
    {
        REQUIRE(DcgmNs::Utils::IsProcessRunning(getpid()));
    }
}

TEST_CASE("Utils: TerminateProcess", "[DcgmUtilities]")
{
    SECTION("Invalid PID is a no-op")
    {
        auto result
            = DcgmNs::Utils::TerminateProcess(-1, [](pid_t) { return true; }, 1, std::chrono::milliseconds { 10 });
        REQUIRE(result == DCGM_ST_OK);
    }

    SECTION("Already-dead process succeeds immediately")
    {
        auto result
            = DcgmNs::Utils::TerminateProcess(42, [](pid_t) { return false; }, 4, std::chrono::milliseconds { 10 });
        REQUIRE(result == DCGM_ST_OK);
    }

    SECTION("Returns DCGM_ST_CHILD_NOT_KILLED when process stays alive")
    {
        auto result
            = DcgmNs::Utils::TerminateProcess(42, [](pid_t) { return true; }, 2, std::chrono::milliseconds { 1 });
        REQUIRE(result == DCGM_ST_CHILD_NOT_KILLED);
    }

    SECTION("Process that dies after first signal")
    {
        int callCount = 0;
        auto result   = DcgmNs::Utils::TerminateProcess(
            42,
            [&callCount](pid_t) {
                ++callCount;
                return callCount <= 1;
            },
            4,
            std::chrono::milliseconds { 1 });
        REQUIRE(result == DCGM_ST_OK);
        // 1: loop-check (alive) → send SIGTERM, 2: loop-check (dead) → exit, 3: post-loop check
        REQUIRE(callCount == 3);
    }
}

TEST_CASE("Utils: KillAndReapChild with forked process", "[DcgmUtilities]")
{
    pid_t child = fork();
    REQUIRE(child >= 0);

    if (child == 0)
    {
        pause();
        _exit(0);
    }

    REQUIRE(kill(child, 0) == 0);
    DcgmNs::Utils::KillAndReapChild(child, 4, std::chrono::milliseconds { 50 });

    int status;
    REQUIRE(waitpid(child, &status, WNOHANG) <= 0);
}

TEST_CASE("Utils: StopProcess", "[DcgmUtilities]")
{
    SECTION("Non-existent process returns DCGM_ST_INSTANCE_NOT_FOUND")
    {
        auto result = DcgmNs::Utils::StopProcess(999999, 1, std::chrono::milliseconds { 10 });
        REQUIRE(result == DCGM_ST_INSTANCE_NOT_FOUND);
    }

    SECTION("Invalid PIDs return DCGM_ST_INSTANCE_NOT_FOUND")
    {
        REQUIRE(DcgmNs::Utils::StopProcess(-1, 1, std::chrono::milliseconds { 10 }) == DCGM_ST_INSTANCE_NOT_FOUND);
        REQUIRE(DcgmNs::Utils::StopProcess(0, 1, std::chrono::milliseconds { 10 }) == DCGM_ST_INSTANCE_NOT_FOUND);
    }
}
