//
// Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.
//

#include "nvsdm_loader.hpp"
#include <dcgm_structs.h>
#include <dlfcn.h>
#include <cstdlib>

#include <stdio.h>
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>

#include <dlfcn.h>

#include <mutex>
#include <memory>

#include "utils.h"

#ifdef _DEBUG
#define TRACE_FUNC() \
    printf("%s\n", __PRETTY_FUNCTION__)
#define PRINT_DEBUG(...) \
    printf("DEBUG: " __VA_ARGS__)
#else
#define TRACE_FUNC()
#define PRINT_DEBUG(...)
#endif

#define PRINT_FORCE(...) \
    fprintf(stdout, __VA_ARGS__)

struct LibHandle
{
    std::string m_name;
    std::mutex m_mutex;
    std::unique_ptr <void, int(*)(void *)> m_handle;

    static void *dlopen(char const *name)
    {
        TRACE_FUNC();
        PRINT_DEBUG("dlopen name = %s\n", name);
        if (name == NULL)
        {
            return NULL;
        }
        dlerror();
        void *handle = ::dlopen(name, RTLD_LAZY);
        return handle;
    }

    static int dlclose(void *handle)
    {
        TRACE_FUNC();
        if (handle)
        {
            return ::dlclose(handle);
        }
        return 0;
    }

    LibHandle(std::string&& name) : m_name(std::move(name)), m_handle(LibHandle::dlopen(NULL), &LibHandle::dlclose) {}

    void *getHandle()
    {
        TRACE_FUNC();
        std::lock_guard <std::mutex> guard(m_mutex);
        if (m_handle == nullptr)
        {
            m_handle = std::unique_ptr<void, int(*)(void *)>(LibHandle::dlopen(m_name.c_str()),
                                                             &LibHandle::dlclose);
        }
        return m_handle.get();
    }
};

static LibHandle s_handle("libnvsdm.so.1");

template <typename T> struct LoadFunc
{
    T m_func;
    std::string m_name;
    std::mutex m_mutex;

    LoadFunc(std::string&& name) : m_name(std::move(name)) {}

    nvsdmRet_t load()
    {
        std::lock_guard <std::mutex> guard(m_mutex);
        nvsdmRet_t ret = NVSDM_SUCCESS;
        if (m_func == nullptr)
        {
            dlerror();
            void *handle = s_handle.getHandle();
            if (handle == nullptr)
            {
                ret = NVSDM_ERROR_LIBRARY_LOAD;
            }
            else
            {
                m_func = (T)dlsym(s_handle.getHandle(), m_name.c_str());
                if (m_func == nullptr)
                {
                    ret = NVSDM_ERROR_FUNCTION_NOT_FOUND;
                }
            }
        }
        return ret;
    }
};

#define NVSDM_FUNCTION(retType, name, argTypes, ...) \
    typedef retType (*name ## _func_t) argTypes; \
    static LoadFunc<name ## _func_t> s_loadFunc ## name(#name); \
    retType name argTypes { \
        if (s_loadFunc ## name.m_func == NULL) { \
            nvsdmRet_t _ret = s_loadFunc ## name.load(); \
            if (_ret != NVSDM_SUCCESS) { \
                return _ret; \
            } \
        } \
        return s_loadFunc ## name.m_func(__VA_ARGS__); \
    }

#include "nvsdm_entry_points.h"

/*
 * The error string function is compiled in directly
 * instead of loaded from the NVSDM dynamic lib because
 * it needs to be callable regardless of whether or
 * not the NVSDM library has been installed on the
 * target system.
 */
extern "C"
char const *nvsdmGetErrorString(nvsdmRet_t ret)
{
    return nvsdm::utils::getErrorString(ret);
}

// DCGM portion

const char *NVSDM_LIBNAME = "libnvsdm.so.1";

namespace DcgmNs {
dcgmReturn_t dcgmLoadNvsdm() {
    static void *nvsdmHandle = nullptr;
    if (nvsdmHandle) {
        return DCGM_ST_ALREADY_INITIALIZED;
    }

    s_handle.m_name = NVSDM_LIBNAME;
    nvsdmHandle = s_handle.getHandle();
    if (!nvsdmHandle) {
        return DCGM_ST_LIBRARY_NOT_FOUND;
    }

    return DCGM_ST_OK;
}
}
