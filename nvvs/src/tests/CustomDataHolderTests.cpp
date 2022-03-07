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

#include <unordered_set>
#include <yaml-cpp/yaml.h>

#include "DcgmDiagUnitTestCommon.h"
#include <CustomStatHolder.h>

std::string investiture("investiture");
std::string breaths("breaths");
std::string bridgemen("bridgemen");
std::string lost("lost");

SCENARIO("unsigned int CustomStatHolder::GpuIdToJsonStatsIndex(unsigned int gpuId)")
{
    CustomStatHolder cdh;

    // This will always return 0 until it's initialized
    for (unsigned int i = 0; i < 1000; i++)
    {
        CHECK(cdh.GpuIdToJsonStatsIndex(i) == 0);
    }

    std::vector<unsigned int> gpus;
    gpus.push_back(0);
    gpus.push_back(2);
    gpus.push_back(4);

    cdh.InitGpus(gpus);
    CHECK(cdh.GpuIdToJsonStatsIndex(0) == 0);
    CHECK(cdh.GpuIdToJsonStatsIndex(2) == 1);
    CHECK(cdh.GpuIdToJsonStatsIndex(4) == 2);

    gpus.clear();
    for (unsigned int i = 0; i < 16; i++)
    {
        gpus.push_back(i);
    }

    cdh.InitGpus(gpus);
    for (unsigned int i = 0; i < 16; i++)
    {
        CHECK(cdh.GpuIdToJsonStatsIndex(i) == i);
    }
}

void get_vector_from_custom_stats(CustomStatHolder &cdh, std::vector<dcgmDiagCustomStats_t> &statsList)
{
    std::unique_ptr<dcgmDiagCustomStats_t> stats = std::make_unique<dcgmDiagCustomStats_t>();
    do
    {
        cdh.PopulateCustomStats(*stats);
        statsList.push_back(*stats);
    } while (stats->moreStats);
}

void check_vector_for_completeness(std::vector<dcgmDiagCustomStats_t> &statsList,
                                   std::unordered_set<double> investSet,
                                   std::unordered_set<long long> breathsSet,
                                   std::unordered_set<long long> bridgemenSet)
{
    // Make sure we find everything in the list
    unsigned int wrongTypeCount = 0;

    for (auto &&stats : statsList)
    {
        for (unsigned int i = 0; i < stats.numStats; i++)
        {
            switch (stats.stats[i].type)
            {
                case DCGM_CUSTOM_STAT_TYPE_GPU:
                    if (investiture == stats.stats[i].statName)
                    {
                        CHECK(stats.stats[i].gpuId == 0);
                        for (unsigned int valueIndex = 0; valueIndex < stats.stats[i].numValues; valueIndex++)
                        {
                            CHECK(stats.stats[i].values[valueIndex].type == DcgmPluginParamFloat);
                            CHECK(stats.stats[i].values[valueIndex].value.dbl >= 0.0);
                            CHECK(stats.stats[i].values[valueIndex].value.dbl < 100.0);
                            investSet.erase(stats.stats[i].values[valueIndex].value.dbl);
                        }
                    }
                    else
                    {
                        CHECK(breaths == stats.stats[i].statName); // this is the only other valid option
                        CHECK(stats.stats[i].gpuId == 0);
                        for (unsigned int valueIndex = 0; valueIndex < stats.stats[i].numValues; valueIndex++)
                        {
                            CHECK(stats.stats[i].values[valueIndex].type == DcgmPluginParamInt);
                            CHECK(stats.stats[i].values[valueIndex].value.i >= 0);
                            CHECK(stats.stats[i].values[valueIndex].value.i < 400);
                            breathsSet.erase(stats.stats[i].values[valueIndex].value.i);
                        }
                    }
                    break;

                case DCGM_CUSTOM_STAT_TYPE_GROUPED:
                    CHECK(bridgemen == stats.stats[i].category);
                    CHECK(lost == stats.stats[i].statName);
                    for (unsigned int valueIndex = 0; valueIndex < stats.stats[i].numValues; valueIndex++)
                    {
                        CHECK(stats.stats[i].values[valueIndex].type == DcgmPluginParamInt);
                        CHECK(stats.stats[i].values[valueIndex].value.i >= 0);
                        CHECK(stats.stats[i].values[valueIndex].value.i < 400);
                        bridgemenSet.erase(stats.stats[i].values[valueIndex].value.i);
                    }

                    break;

                case DCGM_CUSTOM_STAT_TYPE_SINGLE:
                    wrongTypeCount++;
                    break;
            }
        }
    }

    CHECK(investSet.size() == 0);
    CHECK(breathsSet.size() == 0);
    CHECK(bridgemenSet.size() == 0);
}

TEST_CASE("CustomStatHolder : Storing Values and Outputting in structs / JSON")
{
    CustomStatHolder cdh;
    std::vector<unsigned int> gpus;
    gpus.push_back(0);
    gpus.push_back(1);
    cdh.InitGpus(gpus);

    std::unordered_set<double> investSet;
    std::unordered_set<long long> breathsSet;
    std::unordered_set<long long> bridgemenSet;

    for (double i = 0.0; i < 100.0; i += 0.25)
    {
        cdh.SetGpuStat(0, investiture, i);
        investSet.insert(i);
    }

    for (long long i = 0; i < 400; i++)
    {
        cdh.SetGpuStat(0, breaths, i);
        cdh.SetGroupedStat(bridgemen, lost, i);
        breathsSet.insert(i);
        bridgemenSet.insert(i);
    }

    std::vector<dcgmDiagCustomStats_t> statsList;
    get_vector_from_custom_stats(cdh, statsList);

    check_vector_for_completeness(statsList, investSet, breathsSet, bridgemenSet);
    CustomStatHolder holder;
    holder.AddDiagStats(statsList);
    statsList.clear();
    get_vector_from_custom_stats(holder, statsList);
    check_vector_for_completeness(statsList, investSet, breathsSet, bridgemenSet);

    Json::Value jv;
    cdh.AddCustomData(jv);

    for (unsigned int i = 0; i < 400; i++)
    {
        double d = 0.25 * i;
        CHECK(jv[GPUS][0][investiture][i]["value"].asDouble() == d);
        CHECK(jv[GPUS][0][breaths][i]["value"].asInt64() == i);
        CHECK(jv[bridgemen][lost][i]["value"].asInt64() == i);
    }
}
