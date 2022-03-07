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

#include <PluginInterface.h>
#include <Software.h>
#include <dcgm_structs.h>

TEST_CASE("Software: CountDevEntry")
{
    dcgmHandle_t handle = (dcgmHandle_t)0;
    Software s(handle, nullptr);

    bool valid = s.CountDevEntry("nvidia0");
    CHECK(valid == true);
    valid = s.CountDevEntry("nvidia16");
    CHECK(valid == true);
    valid = s.CountDevEntry("nvidia6");
    CHECK(valid == true);

    valid = s.CountDevEntry("nvidiatom");
    CHECK(valid == false);
    valid = s.CountDevEntry("nvidiactl");
    CHECK(valid == false);
    valid = s.CountDevEntry("nvidia-uvm");
    CHECK(valid == false);
    valid = s.CountDevEntry("nvidia-modeset");
    CHECK(valid == false);
    valid = s.CountDevEntry("nvidia-nvswitch");
    CHECK(valid == false);
    valid = s.CountDevEntry("nvidia-caps");
    CHECK(valid == false);
}
