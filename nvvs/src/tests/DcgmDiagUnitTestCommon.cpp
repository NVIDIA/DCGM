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
#include "DcgmDiagUnitTestCommon.h"
#include <stdexcept>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

char ourTempDir[] = "/tmp/dcgm-diag-test-XXXXXX";

std::string createTmpFile(const char *prefix)
{
    std::string pathStr = std::string(ourTempDir) + "/" + prefix + ".XXXXXX";

    // According to man 2 umask, there is no way to read the current umask
    // through the API without also setting it. It can be read through
    // /proc/[pid]/status, but that is likely excessive and is only supported on
    // Linux 4.7+. So we silence the warning here, and we chmod the file to 600
    // below as we don't want to set the umask here without being able to revert
    // it to its original value
    // coverity[secure_temp]
    int ret = mkstemp(&pathStr[0]);
    if (ret == -1)
    {
        throw std::runtime_error("Could not create temp file" + pathStr);
    }
    ret = chmod(&pathStr[0], 0600);
    if (ret == -1)
    {
        throw std::runtime_error("Could not chmod temp file" + pathStr);
    }
    close(ret);
    return pathStr;
}

NvvsTests::NvvsTests()
{
    createTmpDir();
}

NvvsTests::~NvvsTests()
{
    deleteTmpDir();
}

int NvvsTests::createTmpDir()
{
    char *ret = mkdtemp(ourTempDir);
    if (ret == NULL)
    {
        return 1;
    }

    return 0;
}

int NvvsTests::deleteTmpDir()
{
    std::string cmd = std::string("rm -rf ") + ourTempDir;
    system(cmd.c_str());
    return 0;
}
