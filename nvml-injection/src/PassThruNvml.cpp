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
#include <PassThruNvml.h>

PassThruNvml *PassThruNvml::m_passThruInstance = nullptr;

PassThruNvml::PassThruNvml()
    : m_loadedFuncs()
    , m_nvmlLib()
    , m_lastError()
{
    m_passThruInstance = this;
}

bool PassThruNvml::LoadNvmlLibrary()
{
    static const char *nvmlLibPaths[]
        = { "/opt/cross/x86_64-linux-gnu/usr/local/cuda-11.4/targets/x86_64-linux/lib/stubs/"
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
        else
        {
            m_lastError = dlerror();
        }
    }

    return true;
}

PassThruNvml *PassThruNvml::Init()
{
    if (m_passThruInstance == nullptr)
    {
        m_passThruInstance = new PassThruNvml();
    }

    return m_passThruInstance;
}

bool PassThruNvml::IsLoaded(const std::string &funcname) const
{
    return (m_loadedFuncs.find(funcname) != m_loadedFuncs.end());
}

bool PassThruNvml::LoadFunction(const std::string &funcname)
{
    dlerror(); // Clear any old errors
    void *f = dlsym(m_nvmlLib, funcname.c_str());
    if (f != nullptr)
    {
        m_lastError.clear();
        m_loadedFuncs[funcname] = f;
        return true;
    }
    else
    {
        m_lastError = dlerror();
    }

    return false;
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
    return m_passThruInstance;
}
