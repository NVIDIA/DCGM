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
import logger
import os
import utils
import test_utils

class RunNVpex2(app_runner.AppRunner):
    """ Runs the nvpex2 app to inject errors during nvswitch testing """
    
    paths = {
            "Linux_64bit": "./apps/nvpex2/nvpex2",
            "Linux_ppc64le": "./apps/nvpex2/nvpex2",
            "Linux_aarch64": "./apps/nvpex2/nvpex2",
            }

    def __init__(self, args=None):
        path = os.path.join(utils.script_dir, RunNVpex2.paths[utils.platform_identifier])
        super(RunNVpex2, self).__init__(path, args)

    def start(self):
        """
        Runs the nvpex2 command
        """
        super(RunNVpex2, self).start(timeout=10)

    def __str__(self):
        return "RunNVpex2 on all supported devices " + super(RunNVpex2, self).__str__()
