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
#pragma once

#include <dlfcn.h>
#include <memory>
#include <string>
#include <unordered_map>


class PassThruNvml final
{
public:
    /*****************************************************************************/
    static PassThruNvml *Init();

    /*****************************************************************************/
    /*
     * Checks if a function with the specified name is loaded
     *
     * @param funcname (in) - the name of the function we are checking
     * @return true if there is a function loaded with that name, false otherwise
     */
    bool IsLoaded(const std::string &funcname) const;

    /*****************************************************************************/
    /*
     * Loads a function with the specified name
     *
     * @param funcname (in) - the name of the function we are attempting to dynamically load
     * @return true if the function was found, false otherwise
     */
    bool LoadFunction(const std::string &funcname);

    /*****************************************************************************/
    /*
     * Returns the most recent error that occurred
     *
     * @return a string representation of the most recent error
     */
    std::string GetLastError() const;

    /*****************************************************************************/
    void *GetFunction(const std::string &funcname);

    /*****************************************************************************/
    static PassThruNvml *GetInstance();

private:
    std::unordered_map<std::string, void *> m_loadedFuncs;
    void *m_nvmlLib = nullptr;
    std::string m_lastError;

    static std::unique_ptr<PassThruNvml> m_passThruInstance;

    /*****************************************************************************/
    PassThruNvml() = default;

    /*****************************************************************************/
    /*
     * Dynamically loads the NVML library
     *
     * @return false if the library couldn't be found, true otherwise
     */
    bool LoadNvmlLibrary();
};
