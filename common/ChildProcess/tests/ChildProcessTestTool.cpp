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

#include <fcntl.h>
#include <unistd.h>

#include <functional>
#include <iostream>
#include <string>
#include <unordered_map>

bool WriteAll(int fd, void *buf, size_t count)
{
    size_t totalWritten = 0;
    ssize_t lastWritten = 0;
    while (totalWritten < count)
    {
        lastWritten = write(fd, static_cast<char const *>(buf) + totalWritten, count - totalWritten);
        if (lastWritten < 0)
        {
            if (errno == EINTR || errno == EAGAIN)
            {
                continue;
            }
            return false;
        }

        totalWritten += lastWritten;
    }
    return true;
}

int Stdout(int /* argc */, char *argv[])
{
    std::cout << argv[0] << std::endl;
    return 0;
}

int DelayedStdout(int /* argc */, char *argv[])
{
    usleep(20000);
    std::cout << argv[0] << std::endl;
    return 0;
}

int Stderr(int /* argc */, char *argv[])
{
    std::cerr << argv[0] << std::endl;
    return 1;
}

int DelayedStderr(int /* argc */, char *argv[])
{
    usleep(20000);
    std::cerr << argv[0] << std::endl;
    return 1;
}

int Env(int /* argc */, char *argv[])
{
    auto *envVal = std::getenv(argv[0]);
    if (!envVal)
    {
        std::cout << "env not found!" << std::endl;
        return 1;
    }

    std::cout << std::string(envVal) << std::endl;
    return 0;
}

int FdChannel(int /* argc */, char *argv[])
{
    int fdChannel = atoi(argv[0]);

    if (-1 == fcntl(fdChannel, F_GETFD))
    {
        std::cerr << "Unable to get file descriptor for response channel " << argv[0] << std::endl;
        return 1;
    }

    std::string msg = argv[1];
    auto length     = static_cast<std::uint32_t>(msg.size());

    if (!WriteAll(fdChannel, &length, sizeof(length)))
    {
        std::cerr << "Unable to write to fd " << fdChannel << "." << std::endl;
        return 1;
    }

    if (!WriteAll(fdChannel, msg.data(), msg.size()))
    {
        std::cerr << "Unable to write to fd " << fdChannel << "." << std::endl;
        return 1;
    }

    return 0;
}

int Sleep(int /* argc */, char *argv[])
{
    int seconds = atoi(argv[0]);
    sleep(seconds);
    std::cout << "Sleep [" << seconds << "] seconds completed." << std::endl;
    return 0;
}

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        std::cerr << "Too few arguments." << std::endl;
        return 1;
    }

    std::unordered_map<std::string, std::pair<int, std::function<int(int, char *[])>>> handlers {
        { "stdout", { 1, Stdout } },
        { "stderr", { 1, Stderr } },
        { "env", { 1, Env } },
        { "fd-channel", { 2, FdChannel } },
        { "sleep", { 1, Sleep } },
        { "delayedStdout", { 1, DelayedStdout } },
        { "delayedStderr", { 1, DelayedStderr } },
    };

    if (!handlers.contains(argv[1]))
    {
        std::cout << "Unknown action." << std::endl;
        return 1;
    }

    auto &[expectedArgc, handler] = handlers.at(argv[1]);
    if (argc < 2 + expectedArgc)
    {
        std::cerr << "Too few arguments." << std::endl;
        return 1;
    }

    return handler(argc - 2, argv + 2);
}
