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
#include <catch2/catch.hpp>
#include <cstring>
#include <sys/stat.h>

#include <TestFramework.h>

class WrapperTestFramework : protected TestFramework
{
public:
    WrapperTestFramework(bool jsonOutput, std::unique_ptr<GpuSet> &gpuSet);
    std::string WrapperGetPluginDir();
    std::string WrapperGetPluginBaseDir();
    std::string WrapperGetPluginDirExtension() const;
};

WrapperTestFramework::WrapperTestFramework(bool jsonOutput, std::unique_ptr<GpuSet> &gpuSet)
    : TestFramework(jsonOutput, gpuSet.get())
{}

std::string WrapperTestFramework::WrapperGetPluginDir()
{
    return GetPluginDir();
}
std::string WrapperTestFramework::WrapperGetPluginBaseDir()
{
    return GetPluginBaseDir();
}
std::string WrapperTestFramework::WrapperGetPluginDirExtension() const
{
    return GetPluginDirExtension();
}

/*
 * This function's behaviour mirrors Linux's /proc/<pid>/exec
 * It follows symlinks and returns the location of the executable
 */
std::string getThisExecsLocation()
{
    char szTmp[64];
    char buf[1024] = { 0 };
    snprintf(szTmp, sizeof(szTmp), "/proc/%d/exe", getpid());
    auto const bytes = readlink(szTmp, buf, sizeof(buf) - 1);
    if (bytes >= 0)
    {
        buf[bytes] = '\0';
    }
    else
    {
        throw std::runtime_error("Test error. Expected bytes >= 0");
    }
    std::string path { buf };
    std::string parentDir = path.substr(0, path.find_last_of("/"));
    return parentDir;
}

SCENARIO("GetPluginBaseDir returns plugin directory relative to current process's location")
{
    const std::string myLocation   = getThisExecsLocation();
    const std::string pluginDir    = myLocation + "/plugins";
    std::unique_ptr<GpuSet> gpuSet = std::make_unique<GpuSet>();
    WrapperTestFramework tf(true, gpuSet);

    rmdir(pluginDir.c_str());
    CHECK_THROWS(tf.WrapperGetPluginBaseDir());

    int st = mkdir(pluginDir.c_str(), 770);
    if (st != 0)
    {
        // This has fail at this point. The checks below are only to aid in debugging the failure
        CHECK("" == pluginDir);
        CHECK("" == std::string { strerror(errno) });
    }
    CHECK(tf.WrapperGetPluginBaseDir() == pluginDir);
}
