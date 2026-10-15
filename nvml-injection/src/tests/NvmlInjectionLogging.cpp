/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
#include "../NvmlLogging.h"
#include "../NvmlLogging_internal.h"
#include "NvmlLoggerDependencyTestFiles.h"

#include "dcgm_nvml.h"
#include <catch2/catch_all.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cstdlib>
#include <cstring>
#include <filesystem>

#include <array>
#include <cstdio>
#include <iostream>
#include <string>

#include <TempFile.hpp>


TEST_CASE("NvmlInjection Logger")
{
    static auto tmpLogFile = TempFile::Random("", "nvml-injection-tmp.log");
    static auto tmpLogPath = tmpLogFile.Path().string();

    NvmlLoggerDependency<NVML_LOGGER_DEPENDENCY_TYPE> &dependency
        = NvmlLogger<NVML_LOGGER_DEPENDENCY_TYPE>::Log(nullptr, 0, nullptr);

    dependency.remove(NvmlInjectionBackupLogFile);
    dependency.remove(NvmlInjectionTmpLogFile);
    dependency.remove(tmpLogPath.c_str());
    dependency.remove(BAD_LOG_FILE);

    SECTION("NvmlInjection No Logging")
    {
        dependency.UnSetEnv(NvmlInjectionLogEnvVar);
        NvmlLogger<NVML_LOGGER_DEPENDENCY_TYPE>::Log(nullptr, 0, nullptr); /* reset logger target */
        NVML_LOG_ERR("%s", "Test Log.");
        NvmlLogger<NVML_LOGGER_DEPENDENCY_TYPE>::Log(nullptr, 0, nullptr); /* reset logger target */

        CHECK(!dependency.exists(NvmlInjectionBackupLogFile));
        CHECK(!dependency.exists(tmpLogPath.c_str()));
    }

    SECTION("NvmlInjection Specified Logging")
    {
        dependency.SetEnv(NvmlInjectionLogEnvVar, tmpLogPath.c_str());

        /*
         * We don't reset here because we want the implied open on logging
         * to actually open the file. Previous resets would just close any
         * existing file, but would not open one for lack of environment
         * variable specifying one. Here, Open(), called by the call to log
         * would open the log file implicitly. So, we want to check that that
         * happens without a call to reset that would implicitly open it.
         *
         * NvmlLogger<NVML_LOGGER_DEPENDENCY_TYPE>::Log(nullptr, 0, nullptr);
         */

        NVML_LOG_ERR("%s", "Test Log.");
        dependency.UnSetEnv(NvmlInjectionLogEnvVar);
        NvmlLogger<NVML_LOGGER_DEPENDENCY_TYPE>::Log(nullptr, 0, nullptr); /* reset logger target */

        CHECK(!dependency.exists(NvmlInjectionBackupLogFile));
        CHECK(!dependency.exists(NvmlInjectionTmpLogFile));
        CHECK(dependency.exists(tmpLogPath.c_str()));
    }

    SECTION("NvmlInjection Backup Logging")
    {
        dependency.SetEnv(NvmlInjectionLogEnvVar, BAD_LOG_FILE);
        NvmlLogger<NVML_LOGGER_DEPENDENCY_TYPE>::Log(nullptr, 0, nullptr); /* reset logger target */
        NVML_LOG_ERR("%s", "Test Log.");
        dependency.UnSetEnv(NvmlInjectionLogEnvVar);
        NvmlLogger<NVML_LOGGER_DEPENDENCY_TYPE>::Log(nullptr, 0, nullptr); /* reset logger target */

        CHECK((dependency.exists(NvmlInjectionBackupLogFile) || dependency.exists(NvmlInjectionTmpLogFile)));
        CHECK(!dependency.exists(tmpLogPath.c_str()));
        CHECK(!dependency.exists(BAD_LOG_FILE));
    }
}
