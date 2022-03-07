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
import os
import re

from . import app_runner
import dcgm_structs
import dcgm_agent_internal
import test_utils
import utils

class LspciApp(app_runner.AppRunner):
    """
    Run lspci
    """
    
    paths = {
            "Linux_32bit": "./lspci/Linux-x86/",
            "Linux_64bit": "./lspci/Linux-x86_64/",
            "Linux_ppc64le": "./lspci/Linux-ppc64le/",
            "Linux_aarch64": "./lspci/Linux-aarch64/",
            }
    
    def __init__(self, busId, flags):
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), LspciApp.paths[utils.platform_identifier])
        exepath = path + "/sbin/lspci"
        self.processes = None
        args = ["-s", busId, "-i", path + "/share/pci.ids"]
        for flag in flags:
            args.append(flag)
        super(LspciApp, self).__init__(exepath, args)

