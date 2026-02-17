# Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# test the policy manager for DCGM

import dcgm_agent
import test_utils
import dcgm_structs


def test_dcgm_pwr_profile_ids():
    """
    Verify that every valid power-profile ID maps to a unique string.
    """

    profile_names = set()

    for profile_id in range(0, dcgm_structs.DCGM_POWER_PROFILE_MAX):
        name = dcgm_agent.dcgmPowerProfileIdToName(profile_id)
        assert name.startswith(
            "POWER_PROFILE_"), f"Power Profile ID {profile_id} {name} does not start with POWER_PROFILE_"
        assert name not in profile_names, f"Power Profile ID {profile_id} {name} is a duplicate."
        profile_names.add(name)


def test_dcgm_pwr_profile_ids_out_of_range():
    """
    Verify that mapping an out of range power profile raises
    dcgm_structs.DCGM_ST_BADPARAM.
    """
    with test_utils.assert_raises(dcgm_structs.dcgmExceptionClass(dcgm_structs.DCGM_ST_BADPARAM)):
        dcgm_agent.dcgmPowerProfileIdToName(
            dcgm_structs.DCGM_POWER_PROFILE_MAX)
