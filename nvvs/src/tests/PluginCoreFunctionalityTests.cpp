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
#include <vector>

#include <PluginCoreFunctionality.h>

extern errorType_t standardErrorFields[];
extern unsigned short standardInfoFields[];

TEST_CASE("PluginCoreFunctionalityTests: build field list")
{
    std::vector<unsigned short> fieldIds;
    std::vector<unsigned short> additionalFields;
    PluginCoreFunctionality core;
    core.PopulateFieldIds(additionalFields, fieldIds);

    for (unsigned int i = 0; standardErrorFields[i].fieldId != 0; i++)
    {
        CHECK(std::find(fieldIds.begin(), fieldIds.end(), standardErrorFields[i].fieldId) != fieldIds.end());
    }

    for (unsigned int i = 0; standardInfoFields[i] != 0; i++)
    {
        CHECK(std::find(fieldIds.begin(), fieldIds.end(), standardInfoFields[i]) != fieldIds.end());
    }

    // Make sure duplicates are never inserted
    fieldIds.clear();
    for (unsigned int i = 0; standardErrorFields[i].fieldId != 0; i++)
    {
        additionalFields.push_back(standardErrorFields[i].fieldId);
    }

    core.PopulateFieldIds(additionalFields, fieldIds);

    for (unsigned int i = 0; standardErrorFields[i].fieldId != 0; i++)
    {
        auto it = std::find(fieldIds.begin(), fieldIds.end(), standardErrorFields[i].fieldId);
        CHECK(it != fieldIds.end());
        // coverity[increment_iterator] # Intentionally moving past the end
        it++; // move past the found element
        CHECK(std::find(it, fieldIds.end(), standardErrorFields[i].fieldId) == fieldIds.end());
    }
}
