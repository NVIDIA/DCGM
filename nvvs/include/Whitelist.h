/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
#ifndef _NVVS_NVVS_Whitelist_H
#define _NVVS_NVVS_Whitelist_H

#include "Gpu.h"
#include "NvvsCommon.h"
#include "TestParameters.h"
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

class Whitelist
{
    /***************************PUBLIC***********************************/
public:
    Whitelist();
    ~Whitelist();

    // getters
    bool isWhitelisted(std::string deviceId);
    void getDefaultsByDeviceId(const std::string &testName, const std::string &deviceId, TestParameters *tp);

    /****************************************************************/
    /*
     * Adjust the whitelist values once GPUs have been read from DCGM.
     * This must be done separate from the constructor because the
     * whitelist must be bootstrapped in order to generate the list of
     * supported GPUs
     */
    void postProcessWhitelist(std::vector<Gpu *> &gpus);

protected:
    /****************************************************************/
    /*
     * Updates global (test-agnostic) configuration (e.g. throttle mask) for the given device if it is whitelisted.
     */
    void UpdateGlobalsForDeviceId(const std::string &deviceId);

    /***************************PRIVATE**********************************/
private:
    void FillMap();

    /* Per-hardware whitelist parameters database keyed by deviceId and
     * then plugin name */
    std::map<std::string, std::map<std::string, TestParameters *>> m_featureDb;

    /* Set of hardware deviceIds which require global configuration changes. */
    std::set<std::string> m_globalChanges;
};


#endif // _NVVS_NVVS_Whitelist_H
