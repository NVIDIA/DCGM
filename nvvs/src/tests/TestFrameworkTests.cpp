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
#include <TestFramework.h>

#include <catch2/catch_all.hpp>
#include <cstring>
#include <sys/stat.h>

class WrapperTestFramework : public TestFramework
{
public:
    WrapperTestFramework(std::vector<std::unique_ptr<EntitySet>> &entitySet);
    std::string WrapperGetPluginUsingDriverDir();
    std::string WrapperGetPluginBaseDir();
    std::string WrapperGetPluginCudaDirExtension() const;
    std::string WrapperGetPluginCudalessDir();
};

WrapperTestFramework::WrapperTestFramework(std::vector<std::unique_ptr<EntitySet>> &entitySet)
    : TestFramework(entitySet)
{}

std::string WrapperTestFramework::WrapperGetPluginUsingDriverDir()
{
    return GetPluginUsingDriverDir().value_or("");
}
std::string WrapperTestFramework::WrapperGetPluginBaseDir()
{
    return GetPluginBaseDir();
}
std::string WrapperTestFramework::WrapperGetPluginCudaDirExtension() const
{
    return GetPluginCudaDirExtension().value_or("");
}
std::string WrapperTestFramework::WrapperGetPluginCudalessDir()
{
    return GetPluginCudalessDir();
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
    std::string const pluginDir = getThisExecsLocation() + "/plugins";
    INFO("pluginDir: " << pluginDir);

    std::vector<std::unique_ptr<EntitySet>> entitySet;
    WrapperTestFramework tf(entitySet);

    // TODO: DCGM-6030
    //
    // Execution of a unit test should not have observable side effects on the
    // state of the process or test system. This is a file system race with
    // other unit tests executing concurrently. However, the semantics of the
    // GetPluginBaseDir member function of the TestFramework class leave little
    // alternative.
    //
    // At time of writing the TestFramework::GetPluginBaseDir eagerly verifies
    // the existence of each subdirectory in the search path. This is a
    // time-of-check/time-of-use race condition. In addition, it renders the
    // function impure and imposes contraints on test scheduling.
    if (rmdir(pluginDir.c_str()) != 0 && errno != ENOENT)
    {
        FAIL("rmdir failed: " << strerror(errno));
    }

    CHECK_THROWS(tf.WrapperGetPluginBaseDir());

    if (mkdir(pluginDir.c_str(), static_cast<mode_t>(0770)) != 0 && errno != EEXIST)
    {
        FAIL("mkdir failed: " << strerror(errno));
    }

    CHECK(tf.WrapperGetPluginBaseDir() == pluginDir);
}

SCENARIO("GetPluginCudalessDir returns cudaless directory in plugin directory")
{
    std::string const pluginDir = getThisExecsLocation() + "/plugins";
    INFO("pluginDir: " << pluginDir);

    if (mkdir(pluginDir.c_str(), static_cast<mode_t>(0770)) != 0 && errno != EEXIST)
    {
        FAIL("mkdir failed: " << strerror(errno));
    }

    std::vector<std::unique_ptr<EntitySet>> entitySet;
    WrapperTestFramework tf(entitySet);

    std::string const reference = pluginDir + "/cudaless/";

    CHECK(tf.WrapperGetPluginCudalessDir() == reference);
}
