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
#include <stdexcept>

#include "Allowlist.h"
#include "ParsingUtility.h"
#include "PluginStrings.h"
#include <dcgm_fields.h>
#include <unordered_set>

using namespace DcgmNs::Nvvs;

/*****************************************************************************/
Allowlist::Allowlist(const ConfigFileParser_v2 &configFileParser)
    : m_configFileParser(configFileParser)
{
    // initialize map of maps
    FillMap();
}

/*****************************************************************************/
Allowlist::~Allowlist()
{
    std::map<std::string, std::map<std::string, TestParameters *>>::iterator outerIt;
    std::map<std::string, TestParameters *>::iterator innerIter;

    for (outerIt = m_featureDb.begin(); outerIt != m_featureDb.end(); outerIt++)
    {
        for (innerIter = (*outerIt).second.begin(); innerIter != (*outerIt).second.end(); innerIter++)
        {
            TestParameters *tp;

            tp = innerIter->second;

            if (tp)
            {
                delete (tp);
                tp = 0;
            }
        }

        outerIt->second.clear();
    }

    m_featureDb.clear();
}

/*****************************************************************************/
bool Allowlist::IsAllowlisted(const std::string deviceId, const std::string ssid)
{
    std::map<std::string, std::map<std::string, TestParameters *>>::const_iterator outerIt
        = m_featureDb.find(deviceId + ssid);
    std::set<std::string>::const_iterator globalChangesIt = m_globalChanges.find(deviceId + ssid);

    if (outerIt == m_featureDb.end() && globalChangesIt == m_globalChanges.end())
    {
        DCGM_LOG_INFO << "DeviceId " << deviceId.c_str() << " is NOT allowlisted";
        return false;
    }
    else
    {
        DCGM_LOG_INFO << "DeviceId " << deviceId.c_str() << " is allowlisted";
        return true;
    }
}

/*****************************************************************************/
void Allowlist::GetDefaultsByDeviceId(const std::string &testName, const std::string &deviceId, TestParameters *tp)
{
    int st;
    TestParameters *testDeviceTp = m_featureDb[deviceId][testName];
    bool requiresGlobalChanges   = m_globalChanges.find(deviceId) != m_globalChanges.end();
    if (testDeviceTp == nullptr)
    {
        // WAR: Replace any '_' in testName with a ' ' to make sure it really isn't found (DCGM-1774)
        std::string replaced(testName);
        std::replace(replaced.begin(), replaced.end(), '_', ' ');
        testDeviceTp = m_featureDb[deviceId][replaced];
    }

    if (!testDeviceTp && !requiresGlobalChanges)
    {
        log_info("No allowlist overrides found for deviceId {} test {}", deviceId.c_str(), testName.c_str());
        return; /* No overrides configured for this subtest */
    }

    if (testDeviceTp)
    {
        st = tp->OverrideFrom(testDeviceTp);
        if (st)
        {
            throw std::runtime_error("Unable to override bootstrapped test values with device ones.");
        }
    }

    UpdateGlobalsForDeviceId(deviceId);
}

/*****************************************************************************/
void Allowlist::PostProcessAllowlist(std::vector<Gpu *> &gpus)
{
    std::vector<Gpu *>::iterator gpuIter;
    unsigned int minMemClockSeen = DCGM_INT32_BLANK;
    unsigned int memClock;

    static const char *TESLA_V100_ID                 = "1db1";
    const unsigned int TESLA_V100_BASE_SKU_MEM_CLOCK = 877;

    /* Record the minimum memory clock seen for Tesla V100 (1db1).
       It's possible that we have a mix of V100 recovery SKUs and healthy SKUs.
       We'll just use the threshold for the recovery SKU since the healthy SKUs
       will easily exceed that threshold
     */
    for (gpuIter = gpus.begin(); gpuIter != gpus.end(); ++gpuIter)
    {
        if ((*gpuIter)->getDevicePciDeviceId() != TESLA_V100_ID)
        {
            continue;
        }

        memClock = (*gpuIter)->getMaxMemoryClock();
        if (DCGM_INT32_IS_BLANK(memClock))
        {
            continue;
        }

        if (DCGM_INT32_IS_BLANK(minMemClockSeen) || memClock < minMemClockSeen)
        {
            minMemClockSeen = memClock;
        }
    }

    if (DCGM_INT32_IS_BLANK(minMemClockSeen))
    {
        return; /* No 1db1 SKUs. Nothing to do */
    }
    if (minMemClockSeen == TESLA_V100_BASE_SKU_MEM_CLOCK)
    {
        return; /* All SKUs are the base SKU that has a 877 memory clock */
    }

    double ratio = (double)minMemClockSeen / (double)TESLA_V100_BASE_SKU_MEM_CLOCK;

    double existingValue   = m_featureDb[TESLA_V100_ID][MEMBW_PLUGIN_NAME]->GetDouble(MEMBW_STR_MINIMUM_BANDWIDTH);
    double discountedValue = ratio * existingValue;
    m_featureDb[TESLA_V100_ID][MEMBW_PLUGIN_NAME]->SetDouble(MEMBW_STR_MINIMUM_BANDWIDTH, discountedValue);
    log_debug("Updated allowlist for minimium_bandwidth from {} -> {} due to memory clock {}.",
              existingValue,
              discountedValue,
              minMemClockSeen);
}

/*****************************************************************************/
static bool IsBoolParam(std::string const &param)
{
    static std::unordered_set<std::string> const boolParams = {
        "use_dgemv",
        "use_dgemm",
        "use_doubles",
        "l1_is_allowed",
    };
    return boolParams.find(param) != boolParams.end();
}

/*****************************************************************************/
static bool IsStringParam(std::string const &param)
{
    static std::unordered_set<std::string> const stringParams = { PS_IGNORE_ERROR_CODES };
    return stringParams.contains(param);
}

/*****************************************************************************/
void Allowlist::FillMap()
{
    std::string id;
    TestParameters *tp;

    auto skuMap = m_configFileParser.GetSkus();

    try
    {
        for (auto const &pair : skuMap)
        {
            id       = pair.first;
            auto sku = pair.second;

            DCGM_LOG_VERBOSE << "Filling diag config for SKU ID " << id;

            for (auto pluginIt = sku.begin(); pluginIt != sku.end(); pluginIt++)
            {
                tp                           = new TestParameters();
                const std::string pluginName = pluginIt->first.Scalar();
                auto plugin                  = pluginIt->second;
                bool isAllowed               = true;

                for (auto testIt = plugin.begin(); testIt != plugin.end(); testIt++)
                {
                    const std::string testName = testIt->first.Scalar();
                    auto paramOrSubtest        = testIt->second;
                    DCGM_LOG_VERBOSE << "SKU " << id << " plugin " << pluginName << " param/subtest " << testName;
                    if (paramOrSubtest.IsMap())
                    {
                        for (auto subtestIt = paramOrSubtest.begin(); subtestIt != paramOrSubtest.end(); subtestIt++)
                        {
                            const std::string subtestName = subtestIt->first.Scalar();
                            auto const subtestParam       = subtestIt->second;
                            DCGM_LOG_VERBOSE << "Reading " << testName << "." << subtestName
                                             << " as a double. Scalar: " << subtestParam.Scalar();
                            tp->AddSubTestDouble(testName, subtestName, subtestParam.as<double>());
                        }
                    }
                    else
                    {
                        if (testName == "is_allowed")
                        {
                            DCGM_LOG_VERBOSE << "Reading " << testName
                                             << " as a bool. Scalar: " << paramOrSubtest.Scalar();
                            isAllowed = paramOrSubtest.as<bool>();
                        }
                        else if (IsBoolParam(testName))
                        {
                            DCGM_LOG_VERBOSE << "Reading " << testName
                                             << " as a bool. Scalar: " << paramOrSubtest.Scalar();
                            bool paramValue = paramOrSubtest.as<bool>();
                            tp->AddString(testName, paramValue ? "True" : "False");
                        }
                        else if (IsStringParam(testName))
                        {
                            DCGM_LOG_VERBOSE << "Reading " << testName
                                             << " as a string. String: " << paramOrSubtest.as<std::string>();
                            tp->AddString(testName, paramOrSubtest.as<std::string>());
                        }
                        else
                        {
                            DCGM_LOG_VERBOSE << "Reading " << testName
                                             << " as a double. Scalar: " << paramOrSubtest.Scalar();
                            tp->AddDouble(testName, paramOrSubtest.as<double>());
                        }
                    }
                }
                tp->AddString("is_allowed", isAllowed ? "True" : "False");
                m_featureDb[id][pluginName] = tp;
                tp                          = 0;
            }
        }
    }
    catch (const std::exception &e)
    {
        if (tp)
        {
            delete tp;
            tp = 0;
        }
        throw;
    }
}

void Allowlist::UpdateGlobalsForDeviceId(const std::string &deviceId)
{
    /* NOTE: As this function is updated we need to update testing/python/test_utils.py's is_clocks_event_masked()
     * in order to avoid false positive test failures */

    // Tesla K80 ("Stella Duo" Gemini) and Tesla T4 - TU104_PG183_SKU200 or RTX 6000/8000 passive
    if (deviceId == "102d" || deviceId == "1eb8" || deviceId == "1e78")
    {
        // K80 expects some clocks event issues - see bug2454355/DCGM-865 for details
        if (nvvsCommon.clocksEventIgnoreMask == DCGM_INT64_BLANK)
        {
            // Override the clocks event mask only if the user has not specified a mask
            nvvsCommon.clocksEventIgnoreMask = MAX_CLOCKS_EVENT_IGNORE_MASK_VALUE;
        }
        return;
    }
    // V100S experiences SW clocks event
    else if (deviceId == "1df6")
    {
        if (nvvsCommon.clocksEventIgnoreMask == DCGM_INT64_BLANK)
        {
            nvvsCommon.clocksEventIgnoreMask = DCGM_CLOCKS_EVENT_REASON_SW_THERMAL;
        }
    }
    // RTX 6000 experiences HW clocks event
    else if (deviceId == "1e30")
    {
        if (nvvsCommon.clocksEventIgnoreMask == DCGM_INT64_BLANK)
        {
            nvvsCommon.clocksEventIgnoreMask = DCGM_CLOCKS_EVENT_REASON_HW_SLOWDOWN
                                               | DCGM_CLOCKS_EVENT_REASON_HW_THERMAL
                                               | DCGM_CLOCKS_EVENT_REASON_SW_THERMAL;
        }
    }
}
