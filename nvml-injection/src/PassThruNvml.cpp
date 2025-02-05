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
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wanalyzer-malloc-leak"

#include <PassThruNvml.h>

#pragma GCC diagnostic pop

#include <cassert>

std::unique_ptr<PassThruNvml> PassThruNvml::m_passThruInstance;

bool PassThruNvml::LoadNvmlLibrary()
{
    static const char *nvmlLibPaths[]
        = { "/opt/cross/x86_64-linux-gnu/usr/local/cuda-11.8/targets/x86_64-linux/lib/stubs/",
            "/usr/lib/x86_64-linux-gnu",
            "/usr/lib",
            "/usr/lib64",
            "/usr/local/lib/",
            nullptr };

    for (unsigned int i = 0; nvmlLibPaths[i] != nullptr; i++)
    {
        dlerror(); // Clear any old errors
        m_nvmlLib = dlopen(nvmlLibPaths[i], RTLD_LAZY);
        if (m_nvmlLib != nullptr)
        {
            m_lastError.clear();
            return true;
        }
        m_lastError = dlerror();
    }

    return false;
}

PassThruNvml *PassThruNvml::Init()
{
    if (m_passThruInstance == nullptr)
    {
        m_passThruInstance = std::unique_ptr<PassThruNvml>(new PassThruNvml {});
    }

    return m_passThruInstance.get();
}

bool PassThruNvml::IsLoaded(const std::string &funcname) const
{
    return m_loadedFuncs.contains(funcname);
}

bool PassThruNvml::LoadFunction(const std::string &funcname)
{
    dlerror(); // Clear any old errors
    void *f = dlsym(m_nvmlLib, funcname.c_str());
    if (f == nullptr)
    {
        m_lastError = dlerror();
        return false;
    }

    m_lastError.clear();
    m_loadedFuncs[funcname] = f;
    return true;
}

void *PassThruNvml::GetFunction(const std::string &funcname)
{
    return m_loadedFuncs[funcname];
}

std::string PassThruNvml::GetLastError() const
{
    return m_lastError;
}

PassThruNvml *PassThruNvml::GetInstance()
{
    assert(m_passThruInstance != nullptr); // We should have been initialized by now
    return m_passThruInstance.get();
}
