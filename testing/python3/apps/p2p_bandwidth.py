# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
from . import app_runner
import os
import utils

# it could take up to 360 seconds on dgx-2
P2P_BANDWIDTH_TIMEOUT_SECS = 360

class RunP2Pbandwidth(app_runner.AppRunner):
    """ Runs the p2pb_bandwidth binary to generate traffic between Gpus using nvswitch """

    paths = {
            "Linux_64bit": "./apps/p2p_bandwidth/p2p_bandwidth",
            "Linux_ppc64le": "./apps/p2p_bandwidth/p2p_bandwidth",
            "Linux_aarch64": "./apps/p2p_bandwidth/p2p_bandwidth",
            }

    def __init__(self, args):
        path = os.path.join(utils.script_dir, RunP2Pbandwidth.paths[utils.platform_identifier])
        super(RunP2Pbandwidth, self).__init__(path, args)

    def start(self):
        """
        Runs the p2p_bandwidth test on available Gpus
        Raises Exception if it does not work
        """

        super(RunP2Pbandwidth, self).start(timeout=P2P_BANDWIDTH_TIMEOUT_SECS)
        self.stdout_readtillmatch(lambda x: x.find("test PASSED") != -1)

    def __str__(self):
        return "RunP2Pbandwidth on all supported devices " + super(RunP2Pbandwidth, self).__str__()
