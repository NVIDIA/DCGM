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

#include "DcgmChildProcessManager.hpp"
#include "mocks/MockChildProcess.hpp"
#include <DcgmStringHelpers.h>
#include <mock/FileHandleMock.h>

#include <catch2/catch_all.hpp>
#include <fmt/format.h>

static ChildProcessFactory MockChildProcessFactory = []() {
    return std::make_unique<MockChildProcess>();
};

static bool ValidateParams(dcgmChildProcessParams_t const &params, MockChildProcess::SpawnParams const &spawnParams)
{
    if (spawnParams.executable != params.executable)
    {
        log_error("Executable mismatch: {} != {}", spawnParams.executable.string(), params.executable);
        return false;
    }

    for (size_t i = 0; i < params.numArgs; i++)
    {
        if (spawnParams.args[i] != params.args[i])
        {
            log_error("Argument mismatch: {} != {}", spawnParams.args[i], params.args[i]);
            return false;
        }
    }

    for (size_t i = 0; i < params.numEnv; i++)
    {
        auto keyValue = DcgmNs::Split(params.env[i], '=');
        auto it       = spawnParams.env.find(std::string(keyValue[0]));
        if (it == spawnParams.env.end())
        {
            log_error("Environment variable mismatch: {} not found", params.env[i]);
            return false;
        }
        if (it->second != keyValue[1])
        {
            log_error("Environment variable mismatch: {} != {}", it->second, keyValue[1]);
            return false;
        }
    }

    if (params.userName)
    {
        if (spawnParams.userName.value_or("") != params.userName)
        {
            log_error("User name mismatch: {} != {}", spawnParams.userName.value_or(""), params.userName);
            return false;
        }
    }

    if (params.dataChannelFd > 2)
    {
        if (spawnParams.dataChannelFd.value_or(-1) != params.dataChannelFd)
        {
            log_error("Data channel file descriptor mismatch: {} != {}",
                      spawnParams.dataChannelFd.value_or(-1),
                      params.dataChannelFd);
            return false;
        }
    }

    return true;
}

TEST_CASE("ChildProcessManager")
{
    // Create manager with our mock factory
    DcgmChildProcessManager manager(MockChildProcessFactory);

    dcgmChildProcessParams_t params;
    params.version       = dcgmChildProcessParams_version1;
    params.executable    = "/random/executable/that/does/not/exist";
    char const *args[]   = { "0.01" };
    params.args          = args;
    params.numArgs       = 1;
    params.env           = nullptr;
    params.numEnv        = 0;
    params.dataChannelFd = -1;
    params.userName      = nullptr;
    MockChildProcess::Reset();

    DcgmLoggingInit("test_log.txt", DcgmLoggingSeverityVerbose, DcgmLoggingSeverityVerbose);
    RouteLogToBaseLogger(SYSLOG_LOGGER);

    SECTION("Invalid handle")
    {
        log_info("Invalid handle");
        // Use a random handle that is not valid
        ChildProcessHandle_t handle = 342;
        dcgmChildProcessStatus_t status;
        status.version = dcgmChildProcessStatus_version;
        REQUIRE(DCGM_ST_BADPARAM == manager.GetStatus(handle, status));
        REQUIRE(DCGM_ST_BADPARAM == manager.Wait(handle));
        REQUIRE(DCGM_ST_BADPARAM == manager.Destroy(handle));
        REQUIRE(DCGM_ST_BADPARAM == manager.Stop(handle));
        int fd = -1;
        REQUIRE(DCGM_ST_BADPARAM == manager.GetStdErrHandle(handle, fd));
        REQUIRE(DCGM_ST_BADPARAM == manager.GetStdOutHandle(handle, fd));
        REQUIRE(DCGM_ST_BADPARAM == manager.GetDataChannelHandle(handle, fd));
    }

    SECTION("Spawn")
    {
        SECTION("Spawn with only executable")
        {
            log_info("Spawn with only executable");
            int expectedPid = 1234;
            MockChildProcess::SetPid(expectedPid);
            ChildProcessHandle_t handle = INVALID_CHILD_PROCESS_HANDLE;
            int pid                     = 0;
            params.numArgs              = 0;
            params.args                 = nullptr;
            REQUIRE(DCGM_ST_OK == manager.Spawn(params, handle, pid));
            MockChildProcess::CloseAllPipes();
            CHECK(handle != INVALID_CHILD_PROCESS_HANDLE);
            CHECK(pid == expectedPid);
            auto spawnParams = MockChildProcess::GetSpawnParams();
            CHECK(ValidateParams(params, spawnParams));
        }
        SECTION("Spawn with all params")
        {
            log_info("Spawn with all params");
            int expectedPid      = 1234;
            char const *env[]    = { "TEST_ENV=test", "testenv2=test2" };
            params.env           = env;
            params.numEnv        = 2;
            char const *args[]   = { "arg1", "arg2", "arg3" };
            params.args          = args;
            params.numArgs       = 3;
            params.userName      = "testuser";
            params.dataChannelFd = 3;
            MockChildProcess::SetPid(expectedPid);
            ChildProcessHandle_t handle = INVALID_CHILD_PROCESS_HANDLE;
            int pid                     = 0;
            REQUIRE(DCGM_ST_OK == manager.Spawn(params, handle, pid));
            MockChildProcess::CloseAllPipes();
            CHECK(handle != INVALID_CHILD_PROCESS_HANDLE);
            CHECK(pid == expectedPid);
            auto spawnParams = MockChildProcess::GetSpawnParams();
            CHECK(ValidateParams(params, spawnParams));
        }
        SECTION("Spawn generates new handle for each call")
        {
            log_info("Spawn generates new handle for each call");
            ChildProcessHandle_t handle[10] = { INVALID_CHILD_PROCESS_HANDLE },
                                 lastHandle = INVALID_CHILD_PROCESS_HANDLE;
            int expectedPid = 1234, pid = 0;
            for (int i = 0; i < 10; i++)
            {
                MockChildProcess::SetPid(expectedPid++);
                REQUIRE(DCGM_ST_OK == manager.Spawn(params, handle[i], pid));
                CHECK(handle[i] == lastHandle + 1);
                lastHandle = handle[i];
                MockChildProcess::CloseAllPipes();
            }
        }
        SECTION("Spawn returns error when a PID is not available(Run failed)")
        {
            log_info("Spawn returns error when a PID is not available(Run failed)");
            ChildProcessHandle_t handle = INVALID_CHILD_PROCESS_HANDLE;
            int pid                     = 0;
            REQUIRE(DCGM_ST_CHILD_SPAWN_FAILED == manager.Spawn(params, handle, pid));
            CHECK(handle == INVALID_CHILD_PROCESS_HANDLE);
            CHECK(pid == 0);
        }
        SECTION("Spawn returns error when env variable format does not follow key=value format")
        {
            log_info("Spawn returns error when env variable format does not follow key=value format");
            char const *env[]           = { "INVALID_ENV" };
            params.env                  = env;
            params.numEnv               = 1;
            ChildProcessHandle_t handle = INVALID_CHILD_PROCESS_HANDLE;
            int pid                     = 0;
            int expectedPid             = 1234;
            MockChildProcess::SetPid(expectedPid);
            REQUIRE(DCGM_ST_BADPARAM == manager.Spawn(params, handle, pid));
            CHECK(handle == INVALID_CHILD_PROCESS_HANDLE);
            CHECK(pid == 0);
        }
    }

    SECTION("GetStatus")
    {
        log_info("GetStatus");
        ChildProcessHandle_t handle = INVALID_CHILD_PROCESS_HANDLE;
        int expectedPid             = 1234;
        MockChildProcess::SetPid(expectedPid);
        int pid = 0;
        REQUIRE(DCGM_ST_OK == manager.Spawn(params, handle, pid));
        MockChildProcess::CloseAllPipes();
        REQUIRE(handle != INVALID_CHILD_PROCESS_HANDLE);
        REQUIRE(pid == expectedPid);

        dcgmChildProcessStatus_t status1;
        status1.version = dcgmChildProcessStatus_version;
        REQUIRE(DCGM_ST_OK == manager.GetStatus(handle, status1));
        CHECK(status1.running == 1);
        // The remaining fields are not set until the process exits
        CHECK(status1.exitCode == 0);
        CHECK(status1.receivedSignal == 0);
        CHECK(status1.receivedSignalNumber == 0);

        SECTION("Simulate normal process exit with exit code 0")
        {
            // Simulate the process exiting
            MockChildProcess::SimulateExit(0);

            // Get the status of the process
            dcgmChildProcessStatus_t status2;
            status2.version = dcgmChildProcessStatus_version;
            REQUIRE(DCGM_ST_OK == manager.GetStatus(handle, status2));
            CHECK(status2.running == 0);
            CHECK(status2.exitCode == 0);
            CHECK(status2.receivedSignal == 0);
            CHECK(status2.receivedSignalNumber == 0);
        }
        SECTION("Simulate process exit with a signal")
        {
            // Simulate the process exiting with a signal
            MockChildProcess::SimulateExit(2, SIGKILL);

            // Get the status of the process
            dcgmChildProcessStatus_t status3;
            status3.version = dcgmChildProcessStatus_version;
            REQUIRE(DCGM_ST_OK == manager.GetStatus(handle, status3));
            CHECK(status3.running == 0);
            CHECK(status3.exitCode == 2);
            CHECK(status3.receivedSignal == 1);
            CHECK(status3.receivedSignalNumber == SIGKILL);
        }
    }
    SECTION("Wait forwards timeout to child process")
    {
        log_info("Wait forwards timeout to child process");
        ChildProcessHandle_t handle = INVALID_CHILD_PROCESS_HANDLE;
        int pid                     = 0;
        int expectedPid             = 1234;
        MockChildProcess::SetPid(expectedPid);
        REQUIRE(DCGM_ST_OK == manager.Spawn(params, handle, pid));
        MockChildProcess::CloseAllPipes();
        REQUIRE(handle != INVALID_CHILD_PROCESS_HANDLE);
        REQUIRE(pid == expectedPid);

        SECTION("Wait with timeout = 0")
        {
            // Wait should just forward the timeout to the child process.
            REQUIRE(DCGM_ST_OK == manager.Wait(handle, 0));
            auto waitTimeout = MockChildProcess::GetWaitTimeout();
            CHECK(waitTimeout.has_value());
            CHECK(waitTimeout.value() == std::chrono::milliseconds(0));
        }
        SECTION("Wait with timeout > 0")
        {
            // Wait should just forward the timeout to the child process.
            std::chrono::seconds timeout(10);
            REQUIRE(DCGM_ST_OK == manager.Wait(handle, timeout.count()));
            auto waitTimeout = MockChildProcess::GetWaitTimeout();
            CHECK(waitTimeout.has_value());
            CHECK(waitTimeout.value() == timeout); // Timeout will be in milliseconds
        }
        SECTION("Wait with no timeout")
        {
            // Wait should just forward the timeout to the child process.
            REQUIRE(DCGM_ST_OK == manager.Wait(handle, -1));
            auto waitTimeout = MockChildProcess::GetWaitTimeout();
            CHECK(!waitTimeout.has_value());
        }
    }
    SECTION("Stop forwards force to child process")
    {
        SECTION("Force = false")
        {
            log_info("Stop, force = false");
            ChildProcessHandle_t handle = INVALID_CHILD_PROCESS_HANDLE;
            int pid                     = 0;
            int expectedPid             = 1234;
            MockChildProcess::SetPid(expectedPid);
            REQUIRE(DCGM_ST_OK == manager.Spawn(params, handle, pid));
            MockChildProcess::CloseAllPipes();
            REQUIRE(handle != INVALID_CHILD_PROCESS_HANDLE);
            REQUIRE(pid == expectedPid);
            REQUIRE(DCGM_ST_OK == manager.Stop(handle, false));

            // The mock process will exit with a SIGTERM when force is false
            dcgmChildProcessStatus_t status;
            status.version = dcgmChildProcessStatus_version;
            REQUIRE(DCGM_ST_OK == manager.GetStatus(handle, status));
            CHECK(!status.running);
            CHECK(status.receivedSignal == 1);
            CHECK(status.receivedSignalNumber == SIGTERM);
        }
        SECTION("Force = true")
        {
            log_info("Stop, force = true");
            ChildProcessHandle_t handle = INVALID_CHILD_PROCESS_HANDLE;
            int pid                     = 0;
            int expectedPid             = 1234;
            MockChildProcess::SetPid(expectedPid);
            REQUIRE(DCGM_ST_OK == manager.Spawn(params, handle, pid));
            MockChildProcess::CloseAllPipes();
            REQUIRE(handle != INVALID_CHILD_PROCESS_HANDLE);
            REQUIRE(pid == expectedPid);
            REQUIRE(DCGM_ST_OK == manager.Stop(handle, true));

            // The mock process will exit with a SIGKILL when force is true
            dcgmChildProcessStatus_t status;
            status.version = dcgmChildProcessStatus_version;
            REQUIRE(DCGM_ST_OK == manager.GetStatus(handle, status));
            CHECK(!status.running);
            CHECK(status.receivedSignal == 1);
            CHECK(status.receivedSignalNumber == SIGKILL);
        }
    }
    SECTION("Destroy")
    {
        log_info("Destroy");
        ChildProcessHandle_t handle = INVALID_CHILD_PROCESS_HANDLE;
        int pid                     = 0;
        int expectedPid             = 1234;
        MockChildProcess::SetPid(expectedPid);
        REQUIRE(DCGM_ST_OK == manager.Spawn(params, handle, pid));
        MockChildProcess::CloseAllPipes();
        REQUIRE(handle != INVALID_CHILD_PROCESS_HANDLE);
        REQUIRE(pid == expectedPid);
        REQUIRE(DCGM_ST_OK == manager.Destroy(handle));
        // Handle should no longer be valid
        dcgmChildProcessStatus_t status;
        status.version = dcgmChildProcessStatus_version1;
        CHECK(DCGM_ST_BADPARAM == manager.GetStatus(handle, status));
        CHECK(DCGM_ST_BADPARAM == manager.Wait(handle));
        CHECK(DCGM_ST_BADPARAM == manager.Stop(handle));
        int fd = -1;
        CHECK(DCGM_ST_BADPARAM == manager.GetStdErrHandle(handle, fd));
        CHECK(DCGM_ST_BADPARAM == manager.GetStdOutHandle(handle, fd));
        CHECK(DCGM_ST_BADPARAM == manager.GetDataChannelHandle(handle, fd));
    }

    SECTION("GetDataChannelHandle returns error when child process does not have a data channel")
    {
        log_info("GetDataChannelHandle returns error when child process does not have a data channel");
        ChildProcessHandle_t handle = INVALID_CHILD_PROCESS_HANDLE;
        int pid                     = 0;
        int expectedPid             = 1234;
        MockChildProcess::SetPid(expectedPid);
        REQUIRE(DCGM_ST_OK == manager.Spawn(params, handle, pid));
        MockChildProcess::CloseAllPipes();
        REQUIRE(handle != INVALID_CHILD_PROCESS_HANDLE);
        REQUIRE(pid == expectedPid);
        int fd = -1;
        REQUIRE(DCGM_ST_BADPARAM == manager.GetDataChannelHandle(handle, fd));
    }

    SECTION("Pipe Handles")
    {
        struct PipeTestParams
        {
            std::string name;
            dcgmReturn_t (DcgmChildProcessManager::*getHandleFunc)(ChildProcessHandle_t, int &);
            void (*writeFunc)(std::string const &);
        };

        auto pipeTestParams = GENERATE(
            PipeTestParams {
                "GetStdErrHandle", &DcgmChildProcessManager::GetStdErrHandle, &MockChildProcess::WriteToStderr },
            PipeTestParams {
                "GetStdOutHandle", &DcgmChildProcessManager::GetStdOutHandle, &MockChildProcess::WriteToStdout },
            PipeTestParams { "GetDataChannelHandle",
                             &DcgmChildProcessManager::GetDataChannelHandle,
                             &MockChildProcess::WriteToDataChannel });

        SECTION(pipeTestParams.name)
        {
            log_info("{}", pipeTestParams.name);
            ChildProcessHandle_t handle = INVALID_CHILD_PROCESS_HANDLE;
            int pid                     = 0;
            int expectedPid             = 1234;

            // For data channel, we need to specify a data channel fd in params
            if (pipeTestParams.name == "GetDataChannelHandle")
            {
                params.dataChannelFd = 3; // Use a valid fd > 2
            }

            MockChildProcess::SetPid(expectedPid);
            REQUIRE(DCGM_ST_OK == manager.Spawn(params, handle, pid));
            REQUIRE(handle != INVALID_CHILD_PROCESS_HANDLE);
            REQUIRE(pid == expectedPid);
            int pipeFd = -1;
            REQUIRE(DCGM_ST_OK == (manager.*pipeTestParams.getHandleFunc)(handle, pipeFd));
            CHECK(pipeFd >= 0);

            // Generate a buffer with 1kb of random data
            std::vector<char> writeBuffer(1024);
            std::vector<char> readBuffer;
            for (int i = 0; i < 1024; i++)
            {
                writeBuffer[i] = static_cast<char>(rand() % 256);
            }

            SECTION("Write data all at once")
            {
                // Create a thread to write the buffer to the pipe
                std::thread writer([&]() {
                    (*pipeTestParams.writeFunc)(std::string(writeBuffer.begin(), writeBuffer.end()));
                    MockChildProcess::CloseAllPipes();
                });
                // Create another thread to read the buffer from the file descriptor
                std::thread reader([&]() {
                    char buffer[1024];
                    ssize_t bytesRead = 0;
                    while ((bytesRead = read(pipeFd, buffer, sizeof(buffer))) > 0)
                    {
                        readBuffer.insert(readBuffer.end(), buffer, buffer + bytesRead);
                    }
                });
                writer.join();
                reader.join();
                close(pipeFd);
                // Account for the new line added at the end of the writeBuffer
                if (pipeTestParams.name != "GetDataChannelHandle")
                {
                    writeBuffer.push_back('\n');
                }
                CHECK(readBuffer.size() == writeBuffer.size());
                CHECK(readBuffer == writeBuffer);
            }
            SECTION("Write data in chunks")
            {
                // Write buffer with newlines added after each chunk
                std::vector<char> writeBufferWithNewlines;
                // Create a thread to write the buffer to the pipe
                std::thread writer([&]() {
                    size_t dataChunkSize = 128;
                    for (size_t i = 0; i < writeBuffer.size(); i += dataChunkSize)
                    {
                        // Calculate proper chunk size for last chunk
                        size_t remainingBytes = writeBuffer.size() - i;
                        size_t chunkSize      = std::min(dataChunkSize, remainingBytes);

                        // Add a newline after each chunk for stdout and stderr
                        writeBufferWithNewlines.insert(writeBufferWithNewlines.end(),
                                                       writeBuffer.begin() + i,
                                                       writeBuffer.begin() + i + chunkSize);
                        if (pipeTestParams.name != "GetDataChannelHandle")
                        {
                            writeBufferWithNewlines.push_back('\n');
                        }
                        (*pipeTestParams.writeFunc)(
                            std::string(writeBuffer.begin() + i, writeBuffer.begin() + i + chunkSize));
                        std::this_thread::sleep_for(std::chrono::microseconds(100));
                    }
                    MockChildProcess::CloseAllPipes();
                });
                // Create another thread to read the buffer from the file descriptor
                std::thread reader([&]() {
                    char buffer[1024];
                    ssize_t bytesRead = 0;
                    while ((bytesRead = read(pipeFd, buffer, sizeof(buffer))) > 0)
                    {
                        readBuffer.insert(readBuffer.end(), buffer, buffer + bytesRead);
                    }
                });
                writer.join();
                reader.join();
                close(pipeFd);
                CHECK(readBuffer.size() == writeBufferWithNewlines.size());
                CHECK(readBuffer == writeBufferWithNewlines);
            }

            // Reset dataChannelFd for other tests
            if (pipeTestParams.name == "GetDataChannelHandle")
            {
                params.dataChannelFd = -1;
            }
        }
    }
}

TEST_CASE("WriteToFd")
{
    std::vector<std::byte> buffer(255);
    for (int i = 0; i < 255; i++)
    {
        buffer[i] = static_cast<std::byte>(i);
    }

    SECTION("normal case")
    {
        FileHandleMock fileHandleMock;

        auto stopToken = std::stop_token();
        auto ret       = WriteToFd(buffer.data(), buffer.size(), fileHandleMock, stopToken);
        REQUIRE(ret == DCGM_ST_OK);
        REQUIRE(fileHandleMock.GetWriteBuffer() == buffer);
    }

    SECTION("EINTR")
    {
        FileHandleMock fileHandleMock;
        fileHandleMock.AddSizeToTriggerEINTR(8);
        fileHandleMock.AddSizeToTriggerEINTR(64);

        auto stopToken = std::stop_token();
        auto ret       = WriteToFd(buffer.data(), buffer.size(), fileHandleMock, stopToken);
        REQUIRE(ret == DCGM_ST_OK);
        REQUIRE(fileHandleMock.GetWriteBuffer() == buffer);
    }
}

TEST_CASE("DcgmChildProcessManager Reset")
{
    // Create manager with our mock factory
    DcgmChildProcessManager manager(MockChildProcessFactory);

    dcgmChildProcessParams_t params;
    params.version       = dcgmChildProcessParams_version1;
    params.executable    = "/random/executable/that/does/not/exist";
    char const *args[]   = { "0.01" };
    params.args          = args;
    params.numArgs       = 1;
    params.env           = nullptr;
    params.numEnv        = 0;
    params.dataChannelFd = -1;
    params.userName      = nullptr;

    DcgmLoggingInit("test_log.txt", DcgmLoggingSeverityVerbose, DcgmLoggingSeverityVerbose);
    RouteLogToBaseLogger(SYSLOG_LOGGER);

    SECTION("Reset with no processes")
    {
        log_info("Reset with no processes");
        MockChildProcess::Reset();
        // Reset should succeed even with no processes
        REQUIRE(DCGM_ST_OK == manager.Reset());
    }

    SECTION("Reset with single process")
    {
        log_info("Reset with single process");
        MockChildProcess::Reset();

        ChildProcessHandle_t handle = INVALID_CHILD_PROCESS_HANDLE;
        int pid                     = 0;
        int expectedPid             = 1234;
        MockChildProcess::SetPid(expectedPid);
        REQUIRE(DCGM_ST_OK == manager.Spawn(params, handle, pid));
        MockChildProcess::CloseAllPipes();
        REQUIRE(handle != INVALID_CHILD_PROCESS_HANDLE);
        REQUIRE(pid == expectedPid);

        // Reset should kill the process and clear all processes
        REQUIRE(DCGM_ST_OK == manager.Reset());

        // Process handle should no longer be valid after reset
        dcgmChildProcessStatus_t status;
        status.version = dcgmChildProcessStatus_version;
        CHECK(DCGM_ST_BADPARAM == manager.GetStatus(handle, status));
        CHECK(DCGM_ST_BADPARAM == manager.Wait(handle));
        CHECK(DCGM_ST_BADPARAM == manager.Stop(handle));
        int fd = -1;
        CHECK(DCGM_ST_BADPARAM == manager.GetStdErrHandle(handle, fd));
        CHECK(DCGM_ST_BADPARAM == manager.GetStdOutHandle(handle, fd));
        CHECK(DCGM_ST_BADPARAM == manager.GetDataChannelHandle(handle, fd));
    }

    SECTION("Reset with multiple processes")
    {
        log_info("Reset with multiple processes");
        MockChildProcess::Reset();

        std::vector<ChildProcessHandle_t> handles(5);
        int expectedPid = 1000;

        // Spawn multiple processes
        for (size_t i = 0; i < handles.size(); i++)
        {
            int pid = 0;
            MockChildProcess::SetPid(expectedPid + static_cast<int>(i));
            REQUIRE(DCGM_ST_OK == manager.Spawn(params, handles[i], pid));
            MockChildProcess::CloseAllPipes();
            REQUIRE(handles[i] != INVALID_CHILD_PROCESS_HANDLE);
            REQUIRE(pid == expectedPid + static_cast<int>(i));
        }

        // Reset should kill all processes and return OK
        REQUIRE(DCGM_ST_OK == manager.Reset());

        // All process handles should no longer be valid
        for (auto handle : handles)
        {
            dcgmChildProcessStatus_t status;
            status.version = dcgmChildProcessStatus_version;
            CHECK(DCGM_ST_BADPARAM == manager.GetStatus(handle, status));
            CHECK(DCGM_ST_BADPARAM == manager.Wait(handle));
            CHECK(DCGM_ST_BADPARAM == manager.Stop(handle));
            int fd = -1;
            CHECK(DCGM_ST_BADPARAM == manager.GetStdErrHandle(handle, fd));
            CHECK(DCGM_ST_BADPARAM == manager.GetStdOutHandle(handle, fd));
            CHECK(DCGM_ST_BADPARAM == manager.GetDataChannelHandle(handle, fd));
        }
    }

    SECTION("Reset resets handle counter")
    {
        log_info("Reset resets handle counter");
        MockChildProcess::Reset();

        ChildProcessHandle_t handle1 = INVALID_CHILD_PROCESS_HANDLE;
        int pid                      = 0;
        int expectedPid              = 1234;

        // Spawn a process - should get handle 1
        MockChildProcess::SetPid(expectedPid);
        REQUIRE(DCGM_ST_OK == manager.Spawn(params, handle1, pid));
        MockChildProcess::CloseAllPipes();
        REQUIRE(handle1 != INVALID_CHILD_PROCESS_HANDLE);
        CHECK(handle1 == 1); // Should be 1 for fresh manager

        // Reset the manager - this invalidates handle1
        REQUIRE(DCGM_ST_OK == manager.Reset());

        // After reset, the old handle should be invalid (no processes exist)
        dcgmChildProcessStatus_t status;
        status.version = dcgmChildProcessStatus_version;
        CHECK(DCGM_ST_BADPARAM == manager.GetStatus(handle1, status));

        // Spawn another process - should get handle 1 again (counter was reset)
        ChildProcessHandle_t handle2 = INVALID_CHILD_PROCESS_HANDLE;
        MockChildProcess::SetPid(expectedPid + 1);
        REQUIRE(DCGM_ST_OK == manager.Spawn(params, handle2, pid));
        MockChildProcess::CloseAllPipes();

        // The new handle should be 1 (counter was reset)
        CHECK(handle2 == 1);

        // Now handle2 (which is 1) should be valid, confirming the counter reset worked
        status.version = dcgmChildProcessStatus_version;
        CHECK(DCGM_ST_OK == manager.GetStatus(handle2, status));
    }

    SECTION("Reset with processes having data channels")
    {
        log_info("Reset with processes having data channels");
        MockChildProcess::Reset();

        ChildProcessHandle_t handle = INVALID_CHILD_PROCESS_HANDLE;
        int pid                     = 0;
        int expectedPid             = 1234;

        // Set up params with data channel
        params.dataChannelFd = 3;
        MockChildProcess::SetPid(expectedPid);
        REQUIRE(DCGM_ST_OK == manager.Spawn(params, handle, pid));
        MockChildProcess::CloseAllPipes();
        REQUIRE(handle != INVALID_CHILD_PROCESS_HANDLE);
        REQUIRE(pid == expectedPid);

        // Verify data channel is accessible before reset
        int dataFd = -1;
        REQUIRE(DCGM_ST_OK == manager.GetDataChannelHandle(handle, dataFd));
        CHECK(dataFd >= 0);

        // Reset should succeed
        REQUIRE(DCGM_ST_OK == manager.Reset());

        // Process should no longer be accessible
        dcgmChildProcessStatus_t status;
        status.version = dcgmChildProcessStatus_version;
        CHECK(DCGM_ST_BADPARAM == manager.GetStatus(handle, status));

        // Reset dataChannelFd for other tests
        params.dataChannelFd = -1;
    }

    SECTION("Multiple resets should work")
    {
        log_info("Multiple resets should work");
        MockChildProcess::Reset();

        // First reset on empty manager
        REQUIRE(DCGM_ST_OK == manager.Reset());

        // Add a process
        ChildProcessHandle_t handle = INVALID_CHILD_PROCESS_HANDLE;
        int pid                     = 0;
        int expectedPid             = 1234;
        MockChildProcess::SetPid(expectedPid);
        REQUIRE(DCGM_ST_OK == manager.Spawn(params, handle, pid));
        MockChildProcess::CloseAllPipes();

        // Second reset
        REQUIRE(DCGM_ST_OK == manager.Reset());

        // Third reset on empty manager again
        REQUIRE(DCGM_ST_OK == manager.Reset());
    }
}