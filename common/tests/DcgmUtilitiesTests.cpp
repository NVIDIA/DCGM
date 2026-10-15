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
#include <ScopedFd.hpp>
#include <TempDir.hpp>
#include <TempExecutable.hpp>
#include <UserCredentials.hpp>
#include <chrono>
#include <csignal>
#include <dlfcn.h>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <unordered_set>
#include <vector>

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

    SECTION("sample duration wins after slack")
    {
        CHECK(DcgmNs::Utils::GetMaxAge(250ms, 1s, 8, 0.5) == 3s);
    }

    SECTION("negative sample count is normalized to the minimum retention")
    {
        CHECK(DcgmNs::Utils::GetMaxAge(500ms, 0ms, -4, 0.1) == 1s);
    }

    SECTION("negative slack reduces the sample duration before normalization")
    {
        CHECK(DcgmNs::Utils::GetMaxAge(500ms, 0ms, 4, -0.5) == 1s);
    }

    SECTION("explicit max age wins over computed sample duration")
    {
        CHECK(DcgmNs::Utils::GetMaxAge(250ms, 5s, 8, 0.5) == 5s);
    }
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

TEST_CASE("NvmlReturnToDcgmReturn: FUNCTION_NOT_FOUND maps to NOT_SUPPORTED")
{
    REQUIRE(DcgmNs::Utils::NvmlReturnToDcgmReturn(NVML_ERROR_FUNCTION_NOT_FOUND) == DCGM_ST_NOT_SUPPORTED);
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

        ScopedFd rawReadFd(readFd.Release());
        REQUIRE(rawReadFd.Get() == fds[0]);
        CHECK(readFd.Get() == -1);
    }

    SECTION("Move constructor transfers fd and errno")
    {
        int fds[2];
        REQUIRE(pipe(fds) == 0);

        DcgmNs::Utils::FileHandle source(fds[0]);
        DcgmNs::Utils::FileHandle writeEnd(fds[1]);

        REQUIRE(source.Write("x", 1) == -1);
        REQUIRE(source.GetErrno() == EBADF);

        int const originalFd                  = source.Get();
        DcgmNs::Utils::FileHandle destination = std::move(source);

        CHECK(destination.Get() == originalFd);
        CHECK(destination.GetErrno() == EBADF);
        CHECK(source.Get() == -1);
        CHECK(source.GetErrno() == 0);
    }

    SECTION("Move assignment closes prior fd and transfers ownership and errno")
    {
        int firstPipe[2];
        int secondPipe[2];
        REQUIRE(pipe(firstPipe) == 0);
        REQUIRE(pipe(secondPipe) == 0);

        DcgmNs::Utils::FileHandle firstRead(firstPipe[0]);
        DcgmNs::Utils::FileHandle firstWrite(firstPipe[1]);
        DcgmNs::Utils::FileHandle secondRead(secondPipe[0]);
        DcgmNs::Utils::FileHandle secondWrite(secondPipe[1]);

        REQUIRE(firstRead.GetErrno() == 0);
        REQUIRE(secondRead.Write("x", 1) == -1);
        REQUIRE(secondRead.GetErrno() == EBADF);

        int const originalFirstRead  = firstRead.Get();
        int const originalSecondRead = secondRead.Get();

        firstRead = std::move(secondRead);

        CHECK(firstRead.Get() == originalSecondRead);
        CHECK(firstRead.GetErrno() == EBADF);
        CHECK(secondRead.Get() == -1);
        CHECK(secondRead.GetErrno() == 0);
        CHECK((fcntl(originalFirstRead, F_GETFD) == -1 && errno == EBADF));
        CHECK(fcntl(originalSecondRead, F_GETFD) != -1);
    }

    SECTION("GIVEN invalid handle WHEN reading or writing THEN EBADF is reported")
    {
        DcgmNs::Utils::FileHandle handle;
        std::array<std::byte, 4> buffer {};

        CHECK(handle.ReadExact(buffer.data(), buffer.size()) == -1);
        CHECK(handle.GetErrno() == EBADF);
        CHECK(handle.Write("x", 1) == -1);
        CHECK(handle.GetErrno() == EBADF);
        CHECK(handle.WriteExact("x", 1) == -1);
        CHECK(handle.GetErrno() == EBADF);
    }

    SECTION("GIVEN pipe-backed handles WHEN reading and writing THEN exact data is transferred")
    {
        int fds[2];
        REQUIRE(pipe(fds) == 0);
        DcgmNs::Utils::FileHandle readFd(fds[0]);
        DcgmNs::Utils::FileHandle writeFd(fds[1]);
        std::array<std::byte, 4> buffer {};

        REQUIRE(writeFd.WriteExact("dcgm", 4) == 4);
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
    SECTION("GIVEN a process handle WHEN loading a known exported symbol THEN a callable pointer is returned")
    {
        void *handle = dlopen(nullptr, RTLD_LAZY);
        REQUIRE(handle != nullptr);
        DcgmNs::Defer closeHandle([&] { dlclose(handle); });

        using GetPidFn = pid_t (*)();
        auto *symbol   = DcgmNs::Utils::LoadFunction(handle, "getpid", "self");

        REQUIRE(symbol != nullptr);
        auto getPid = reinterpret_cast<GetPidFn>(symbol);
        CHECK(getPid() == getpid());
    }

    SECTION("GIVEN missing symbol WHEN loading function THEN nullptr is returned")
    {
        void *handle = dlopen(nullptr, RTLD_LAZY);
        REQUIRE(handle != nullptr);
        DcgmNs::Defer closeHandle([&] { dlclose(handle); });

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
    SECTION("raw zero decodes to zero")
    {
        auto const [mantissa, exponent, parsed] = DcgmNs::Utils::NvmlBerParser(0);
        CHECK(mantissa == 0);
        CHECK(exponent == 0);
        CHECK(parsed == 0.0);
    }

    SECTION("packed mantissa and exponent are decoded")
    {
        // (5 << 8) | 15 -> mantissa 5, exponent 15 -> 5e-15
        auto const [mantissa, exponent, parsed] = DcgmNs::Utils::NvmlBerParser((5 << 8) | 15);
        CHECK(mantissa == 5);
        CHECK(exponent == 15);
        CHECK_THAT(parsed, Catch::Matchers::WithinRel(5e-15, 1e-12));
    }

    SECTION("GIVEN an encoded BER value WHEN parsed THEN mantissa exponent and value are decoded")
    {
        auto const encodedBer = static_cast<int64_t>((3 << NVML_NVLINK_BER_MANTISSA_SHIFT) | 4);

        auto const [mantissa, exponent, parsed] = DcgmNs::Utils::NvmlBerParser(encodedBer);

        CHECK(mantissa == 3);
        CHECK(exponent == 4);
        CHECK_THAT(parsed, Catch::Matchers::WithinAbs(0.0003, 0.00000001));
    }

    SECTION("GIVEN root checker helpers WHEN queried THEN they reflect the process effective user")
    {
        DcgmNs::Utils::RunningUserChecker checker;

        CHECK(DcgmNs::Utils::IsRunningAsRoot() == (geteuid() == 0));
        CHECK(checker.IsRoot() == DcgmNs::Utils::IsRunningAsRoot());
    }
}
TEST_CASE("Utils: FindExecutable")
{
    auto tempDir        = TempDir::Random("dcgm-find-executable-");
    auto testExecutable = TempExecutable::Fixed(tempDir.Path() / "test_exe");

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
              { tempDir.Path().string() },
              true,
              testExecutable.Path().string(),
              tempDir.Path().string(),
              DCGM_ST_OK },
            { "Executable not found returns error", { "/nonexistent/path" }, false, "", "", DCGM_ST_NO_DATA },
            { "Multiple search paths - found in second path",
              { "/nonexistent", tempDir.Path().string() },
              true,
              testExecutable.Path().string(),
              tempDir.Path().string(),
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

    SECTION("Empty search paths leave the output directory unchanged")
    {
        std::string executableDir = "unchanged";

        auto result = DcgmNs::Utils::FindExecutable("test_exe", {}, executableDir);

        REQUIRE_FALSE(result.has_value());
        CHECK(result.error() == DCGM_ST_NO_DATA);
        CHECK(executableDir == "unchanged");
    }
}

TEST_CASE("Utils: IsProcessRunning")
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

TEST_CASE("Utils: TerminateProcess")
{
    auto const testPid = static_cast<pid_t>(42);
    std::vector<int> sentSignals;
    auto sendSignal = [&sentSignals, testPid](pid_t pid, int signalNumber) {
        REQUIRE(pid == testPid);
        sentSignals.push_back(signalNumber);
    };
    auto makeCallbacks = [&sendSignal](DcgmNs::Utils::LivenessCheckFn isAlive) {
        return DcgmNs::Utils::TerminateProcessCallbacks {
            .isAlive    = isAlive,
            .sendSignal = sendSignal,
        };
    };

    SECTION("Invalid PID is a no-op")
    {
        auto result = DcgmNs::Utils::TerminateProcess(
            -1, makeCallbacks([](pid_t) { return true; }), 1, std::chrono::milliseconds { 10 });
        REQUIRE(result == DCGM_ST_OK);
        REQUIRE(sentSignals.empty());
    }

    SECTION("Already-dead process succeeds immediately")
    {
        auto result = DcgmNs::Utils::TerminateProcess(
            testPid, makeCallbacks([](pid_t) { return false; }), 4, std::chrono::milliseconds { 10 });
        REQUIRE(result == DCGM_ST_OK);
        REQUIRE(sentSignals.empty());
    }

    SECTION("Missing callbacks return DCGM_ST_BADPARAM")
    {
        auto missingIsAlive = DcgmNs::Utils::TerminateProcessCallbacks {
            .isAlive    = nullptr,
            .sendSignal = sendSignal,
        };
        CHECK(DcgmNs::Utils::TerminateProcess(testPid, missingIsAlive, 1, std::chrono::milliseconds { 1 })
              == DCGM_ST_BADPARAM);
        CHECK(sentSignals.empty());

        auto missingSendSignal = DcgmNs::Utils::TerminateProcessCallbacks {
            .isAlive = [](pid_t) { return true; },
            .sendSignal = nullptr,
        };
        CHECK(DcgmNs::Utils::TerminateProcess(testPid, missingSendSignal, 1, std::chrono::milliseconds { 1 })
              == DCGM_ST_BADPARAM);
        CHECK(sentSignals.empty());
    }

    SECTION("Returns DCGM_ST_CHILD_NOT_KILLED when process stays alive")
    {
        auto result = DcgmNs::Utils::TerminateProcess(
            testPid, makeCallbacks([](pid_t) { return true; }), 2, std::chrono::milliseconds { 1 });
        REQUIRE(result == DCGM_ST_CHILD_NOT_KILLED);
        REQUIRE(sentSignals == std::vector { SIGTERM, SIGKILL });
    }

    SECTION("Process that dies after first signal")
    {
        int callCount = 0;
        auto result   = DcgmNs::Utils::TerminateProcess(testPid,
                                                        makeCallbacks([&callCount](pid_t) {
                                                          ++callCount;
                                                          return callCount <= 1;
                                                        }),
                                                        4,
                                                        std::chrono::milliseconds { 1 });
        REQUIRE(result == DCGM_ST_OK);
        // 1: loop-check (alive) -> send SIGTERM, 2: loop-check (dead) -> exit, 3: post-loop check
        REQUIRE(callCount == 3);
        REQUIRE(sentSignals == std::vector { SIGTERM });
    }

    SECTION("No attempts returns child-not-killed when the liveness callback still reports alive")
    {
        int callCount = 0;
        // PID is arbitrary: liveness is supplied by the injected predicate and maxAttempts==0
        // means no signal is ever sent, so no real process is touched.
        auto result = DcgmNs::Utils::TerminateProcess(42,
                                                      makeCallbacks([&callCount](pid_t) {
                                                          ++callCount;
                                                          return true;
                                                      }),
                                                      0,
                                                      std::chrono::milliseconds { 1 });

        CHECK(result == DCGM_ST_CHILD_NOT_KILLED);
        CHECK(callCount == 1);
    }
}

TEST_CASE("Utils: KillAndReapChild with forked process")
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

    int status   = 0;
    errno        = 0;
    int const rc = waitpid(child, &status, WNOHANG);
    REQUIRE(rc == -1);
    REQUIRE(errno == ECHILD);
}

TEST_CASE("Utils: StopProcess")
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

SCENARIO("Utils: maps NVML errors")
{
    GIVEN("representative NVML return codes")
    {
        struct Case
        {
            nvmlReturn_t nvml;
            dcgmReturn_t dcgm;
        };
        std::array<Case, 16> const cases { {
            { NVML_SUCCESS, DCGM_ST_OK },
            { NVML_ERROR_NOT_SUPPORTED, DCGM_ST_NOT_SUPPORTED },
            { NVML_ERROR_FUNCTION_NOT_FOUND, DCGM_ST_NOT_SUPPORTED },
            { NVML_ERROR_NO_PERMISSION, DCGM_ST_NO_PERMISSION },
            { NVML_ERROR_NOT_FOUND, DCGM_ST_NO_DATA },
            { NVML_ERROR_TIMEOUT, DCGM_ST_TIMEOUT },
            { NVML_ERROR_GPU_IS_LOST, DCGM_ST_GPU_IS_LOST },
            { NVML_ERROR_RESET_REQUIRED, DCGM_ST_RESET_REQUIRED },
            { NVML_ERROR_INVALID_ARGUMENT, DCGM_ST_BADPARAM },
            { NVML_ERROR_INSUFFICIENT_RESOURCES, DCGM_ST_INSUFFICIENT_RESOURCES },
            { NVML_ERROR_UNINITIALIZED, DCGM_ST_NVML_NOT_LOADED },
            { NVML_ERROR_IRQ_ISSUE, DCGM_ST_NVML_ERROR },
            { NVML_ERROR_LIBRARY_NOT_FOUND, DCGM_ST_NVML_ERROR },
            { NVML_ERROR_DRIVER_NOT_LOADED, DCGM_ST_NVML_ERROR },
            { NVML_ERROR_INSUFFICIENT_SIZE, DCGM_ST_NVML_ERROR },
            { NVML_ERROR_UNKNOWN, DCGM_ST_NVML_ERROR },
        } };

        WHEN("the NVML values are translated")
        {
            THEN("each public mapping remains stable")
            {
                for (auto const &tc : cases)
                {
                    CAPTURE(tc.nvml);
                    CHECK(DcgmNs::Utils::NvmlReturnToDcgmReturn(tc.nvml) == tc.dcgm);
                }
            }
        }
    }
}

SCENARIO("Utils: ChangeUser rejects an unknown privilege policy")
{
    GIVEN("an unexpected privilege policy value")
    {
        UserCredentials credentials { getegid(), geteuid(), {} };

        WHEN("ChangeUser is called")
        {
            auto result = ChangeUser(static_cast<ChangeUserPolicy>(999), credentials);

            THEN("it fails without changing the effective user or group")
            {
                CHECK_FALSE(result.has_value());
                CHECK(geteuid() == credentials.uid);
                CHECK(getegid() == credentials.gid);
            }
        }
    }
}

SCENARIO("Utils: container helpers notify only removed elements")
{
    GIVEN("a map with removable and retained entries")
    {
        std::map<std::string, int> values { { "drop-a", 1 }, { "keep", 2 }, { "drop-b", 3 } };
        std::vector<std::string> notified;

        WHEN("EraseAndNotifyIf removes matching entries")
        {
            auto const removed = DcgmNs::Utils::EraseAndNotifyIf(
                values,
                [](auto const &entry) { return entry.first.starts_with("drop"); },
                [&notified](auto const &entry) { notified.push_back(entry.first); });

            THEN("only removed entries are reported to the callback")
            {
                CHECK(removed == 2);
                CHECK(values.size() == 1);
                REQUIRE(values.contains("keep"));
                CHECK(values.at("keep") == 2);
                CHECK_THAT(notified, Catch::Matchers::UnorderedEquals(std::vector<std::string> { "drop-a", "drop-b" }));
            }
        }
    }

    GIVEN("a set with no matching entries")
    {
        std::set<int> values { 1, 2, 3 };
        int callbackCount = 0;

        WHEN("EraseAndNotifyIf has nothing to erase")
        {
            auto const removed = DcgmNs::Utils::EraseAndNotifyIf(
                values, [](int value) { return value > 10; }, [&callbackCount](int) { ++callbackCount; });

            THEN("the container and callback state remain unchanged")
            {
                CHECK(removed == 0);
                CHECK(values == std::set<int> { 1, 2, 3 });
                CHECK(callbackCount == 0);
            }
        }
    }
}

SCENARIO("Utils: low-level pipe reads expose EOF and moved-handle behavior")
{
    GIVEN("an invalid file handle")
    {
        fmt::memory_buffer output;

        WHEN("ReadProcessOutput tries to monitor it")
        {
            int const result = DcgmNs::Utils::ReadProcessOutput(output, DcgmNs::Utils::FileHandle {});

            THEN("the descriptor error is returned and output remains empty")
            {
                CHECK(result == EBADF);
                CHECK(output.size() == 0);
            }
        }
    }

    GIVEN("a pipe with a closed writer")
    {
        int fds[2];
        REQUIRE(pipe(fds) == 0);
        DcgmNs::Utils::FileHandle readFd(fds[0]);
        DcgmNs::Utils::FileHandle writeFd(fds[1]);
        writeFd = DcgmNs::Utils::FileHandle {};
        fmt::memory_buffer output;

        WHEN("ReadProcessOutput consumes the receiver")
        {
            int const result = DcgmNs::Utils::ReadProcessOutput(output, std::move(readFd));

            THEN("EOF is reported as successful empty output")
            {
                CHECK(result == 0);
                CHECK(output.size() == 0);
                CHECK(readFd.Get() == -1);
            }
        }
    }

    GIVEN("a pipe with a short payload")
    {
        int fds[2];
        REQUIRE(pipe(fds) == 0);
        DcgmNs::Utils::FileHandle readFd(fds[0]);
        DcgmNs::Utils::FileHandle writeFd(fds[1]);
        REQUIRE(writeFd.Write("abc", 3) == 3);
        writeFd = DcgmNs::Utils::FileHandle {};
        fmt::memory_buffer output;

        WHEN("ReadProcessOutput drains the receiver")
        {
            int const result = DcgmNs::Utils::ReadProcessOutput(output, std::move(readFd));

            THEN("the exact bytes are appended to the output buffer")
            {
                CHECK(result == 0);
                CHECK(fmt::to_string(output) == "abc");
                CHECK(readFd.Get() == -1);
            }
        }
    }
}


SCENARIO("Utils: ForkAndExecCommand exposes child output and failure paths")
{
    GIVEN("a missing output handle")
    {
        WHEN("ForkAndExecCommand is asked to run a valid program")
        {
            auto const pid = DcgmNs::Utils::ForkAndExecCommand(
                { "/bin/true" }, nullptr, nullptr, nullptr, false, nullptr, nullptr);

            THEN("the contract rejects the invalid caller-owned output pointer")
            {
                CHECK(pid == -1);
            }
        }
    }

    GIVEN("a child process that writes separate stdout and stderr streams")
    {
        DcgmNs::Utils::FileHandle stdoutFd;
        DcgmNs::Utils::FileHandle stderrFd;
        pid_t child = DcgmNs::Utils::ForkAndExecCommand({ "/bin/sh", "-c", "printf dcgm-out; printf dcgm-err >&2" },
                                                        nullptr,
                                                        &stdoutFd,
                                                        &stderrFd,
                                                        false,
                                                        nullptr,
                                                        nullptr);
        REQUIRE(child > 0);
        DcgmNs::Defer cleanup([&] {
            if (child > 0)
            {
                DcgmNs::Utils::KillAndReapChild(child, 4, std::chrono::milliseconds { 50 });
            }
        });

        WHEN("the parent drains both pipes")
        {
            fmt::memory_buffer stdoutOutput;
            fmt::memory_buffer stderrOutput;

            auto const stdoutResult = DcgmNs::Utils::ReadProcessOutput(stdoutOutput, std::move(stdoutFd));
            auto const stderrResult = DcgmNs::Utils::ReadProcessOutput(stderrOutput, std::move(stderrFd));
            int status              = 0;
            REQUIRE(waitpid(child, &status, 0) == child);
            child = -1;

            THEN("stdout and stderr are preserved independently")
            {
                CHECK(stdoutResult == 0);
                CHECK(stderrResult == 0);
                CHECK(WIFEXITED(status));
                CHECK(WEXITSTATUS(status) == 0);
                CHECK(fmt::to_string(stdoutOutput) == "dcgm-out");
                CHECK(fmt::to_string(stderrOutput) == "dcgm-err");
            }
        }
    }

    GIVEN("a child process that reads stdin")
    {
        DcgmNs::Utils::FileHandle stdinFd;
        DcgmNs::Utils::FileHandle stdoutFd;
        pid_t child
            = DcgmNs::Utils::ForkAndExecCommand({ "/bin/cat" }, &stdinFd, &stdoutFd, nullptr, true, nullptr, nullptr);
        REQUIRE(child > 0);
        DcgmNs::Defer cleanup([&] {
            if (child > 0)
            {
                DcgmNs::Utils::KillAndReapChild(child, 4, std::chrono::milliseconds { 50 });
            }
        });

        WHEN("the parent writes input and closes stdin")
        {
            REQUIRE(stdinFd.Write("from-parent", 11) == 11);
            stdinFd = DcgmNs::Utils::FileHandle {};

            fmt::memory_buffer output;
            auto const readResult = DcgmNs::Utils::ReadProcessOutput(output, std::move(stdoutFd));
            int status            = 0;
            REQUIRE(waitpid(child, &status, 0) == child);
            child = -1;

            THEN("the child receives exactly the provided input")
            {
                CHECK(readResult == 0);
                CHECK(WIFEXITED(status));
                CHECK(WEXITSTATUS(status) == 0);
                CHECK(fmt::to_string(output) == "from-parent");
            }
        }
    }

    GIVEN("a child process with stderr redirected into stdout")
    {
        DcgmNs::Utils::FileHandle stdoutFd;
        DcgmNs::Utils::FileHandle stderrFd;
        pid_t child = DcgmNs::Utils::ForkAndExecCommand({ "/bin/sh", "-c", "printf joined-out; printf joined-err >&2" },
                                                        nullptr,
                                                        &stdoutFd,
                                                        &stderrFd,
                                                        true,
                                                        nullptr,
                                                        nullptr);
        REQUIRE(child > 0);
        DcgmNs::Defer cleanup([&] {
            if (child > 0)
            {
                DcgmNs::Utils::KillAndReapChild(child, 4, std::chrono::milliseconds { 50 });
            }
        });

        WHEN("the parent drains stdout")
        {
            fmt::memory_buffer output;

            auto const readResult = DcgmNs::Utils::ReadProcessOutput(output, std::move(stdoutFd));
            int status            = 0;
            REQUIRE(waitpid(child, &status, 0) == child);
            child = -1;

            THEN("both streams are available from the stdout pipe")
            {
                CHECK(readResult == 0);
                CHECK(stderrFd.Get() == -1);
                CHECK(WIFEXITED(status));
                CHECK(WEXITSTATUS(status) == 0);
                CHECK(fmt::to_string(output) == "joined-outjoined-err");
            }
        }
    }

    GIVEN("an executable path that cannot be launched")
    {
        DcgmNs::Utils::FileHandle stdoutFd;
        DcgmNs::Utils::FileHandle stderrFd;
        pid_t child = DcgmNs::Utils::ForkAndExecCommand(
            { "/no/such/dcgm-test-executable" }, nullptr, &stdoutFd, &stderrFd, false, nullptr, nullptr);
        REQUIRE(child > 0);
        DcgmNs::Defer cleanup([&] {
            if (child > 0)
            {
                DcgmNs::Utils::KillAndReapChild(child, 4, std::chrono::milliseconds { 50 });
            }
        });

        WHEN("the parent drains stderr")
        {
            fmt::memory_buffer stderrOutput;

            auto const stderrResult = DcgmNs::Utils::ReadProcessOutput(stderrOutput, std::move(stderrFd));
            int status              = 0;
            REQUIRE(waitpid(child, &status, 0) == child);
            child = -1;

            THEN("the child reports the exec failure")
            {
                CHECK(stderrResult == 0);
                CHECK(WIFEXITED(status));
                CHECK(WEXITSTATUS(status) != 0);
                CHECK_THAT(fmt::to_string(stderrOutput), Catch::Matchers::ContainsSubstring("Could not exec"));
            }
        }
    }

    GIVEN("a non-existent service account")
    {
        if (!PasswdIsFilesOnly())
        {
            SKIP("Skipping service account failure test: /etc/nsswitch.conf passwd is not 'files' only");
        }

        DcgmNs::Utils::FileHandle stdoutFd;
        DcgmNs::Utils::FileHandle stderrFd;
        pid_t child = DcgmNs::Utils::ForkAndExecCommand({ "/bin/true" },
                                                        nullptr,
                                                        &stdoutFd,
                                                        &stderrFd,
                                                        false,
                                                        "dcgm_missing_service_account_for_unit_test",
                                                        nullptr);
        REQUIRE(child > 0);
        DcgmNs::Defer cleanup([&] {
            if (child > 0)
            {
                DcgmNs::Utils::KillAndReapChild(child, 4, std::chrono::milliseconds { 50 });
            }
        });

        WHEN("the parent drains stderr")
        {
            fmt::memory_buffer stderrOutput;

            auto const stderrResult = DcgmNs::Utils::ReadProcessOutput(stderrOutput, std::move(stderrFd));
            int status              = 0;
            REQUIRE(waitpid(child, &status, 0) == child);
            child = -1;

            THEN("the child reports the missing credentials before exec")
            {
                CHECK(stderrResult == 0);
                CHECK(WIFEXITED(status));
                CHECK(WEXITSTATUS(status) != 0);
                CHECK_THAT(fmt::to_string(stderrOutput),
                           Catch::Matchers::ContainsSubstring("Unable to find credentials"));
            }
        }
    }
}
