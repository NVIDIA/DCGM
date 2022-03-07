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
#include <DcgmGpuInstance.h>
#include <catch2/catch.hpp>

TEST_CASE("GpuInstance: DriveGpuInstanceName")
{
    std::string name = DcgmGpuInstance::DeriveGpuInstanceName("4c.4g.8gb");
    REQUIRE(name == "4g.8gb"); // strip 'Xc.'

    std::string same = DcgmGpuInstance::DeriveGpuInstanceName(name);
    REQUIRE(name == same); // don't change it when we don't need to strip 'Xc.'

    name = DcgmGpuInstance::DeriveGpuInstanceName("24c.4g.8gb");
    REQUIRE(name == "4g.8gb"); // strip 'Xc.' when it's multiple digits as well

    // Make sure these don't crash and don't change the string
    std::string cases[] = { "iron eyes", "tom.bombadil", "ged" };
    for (auto const &c : cases)
    {
        name = DcgmGpuInstance::DeriveGpuInstanceName(c);
        REQUIRE(name == c);
    }
}
