# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import dcgm_agent_internal
import dcgm_fields
import dcgm_agent

def helper_nvsdm_pause_resume(handle):
    flags = 0
    switchIds = dcgm_agent.dcgmGetEntityGroupEntities(handle, dcgm_fields.DCGM_FE_SWITCH, flags)
    ibCxIds = dcgm_agent.dcgmGetEntityGroupEntities(handle, dcgm_fields.DCGM_FE_CONNECTX, flags)
    portIds = dcgm_agent.dcgmGetEntityGroupEntities(handle, dcgm_fields.DCGM_FE_LINK, flags)

    # Exceptions are checked inside the `dcgmPauseTelemetryForDiag` and `dcgmResumeTelemetryForDiag` functions.
    dcgm_agent_internal.dcgmPauseTelemetryForDiag(handle)
    dcgm_agent_internal.dcgmResumeTelemetryForDiag(handle)

    newSwitchIds = dcgm_agent.dcgmGetEntityGroupEntities(handle, dcgm_fields.DCGM_FE_SWITCH, flags)
    newIbCxIds = dcgm_agent.dcgmGetEntityGroupEntities(handle, dcgm_fields.DCGM_FE_CONNECTX, flags)
    newPortIds = dcgm_agent.dcgmGetEntityGroupEntities(handle, dcgm_fields.DCGM_FE_LINK, flags)

    def assertIdListIsTheSame(old, new, name):
        assert len(old) == len(new), f"The length of {name} does not match: old length is [{len(old)}], new length is [{len(new)}]."
        for i in range(len(old)):
            assert old[i] == new[i], f"The index [{i}] of {name} does not match: old: [{old[i]}], new: [{new[i]}]"

    assertIdListIsTheSame(switchIds, newSwitchIds, "Switch")
    assertIdListIsTheSame(ibCxIds, newIbCxIds, "CX")
    assertIdListIsTheSame(portIds, newPortIds, "Port")