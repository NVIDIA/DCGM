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

class XidApp(app_runner.AppRunner):
    paths = {
            "Linux_32bit": "./apps/xid/xid_32bit",
            "Linux_64bit": "./apps/xid/xid_64bit",
            "Linux_ppc64le": "./apps/xid/xid_ppc64le",
            "Windows_64bit": "./apps/xid/xid_64bit.exe"
            }
    def __init__(self, device):
        self.device = device
        path = os.path.join(utils.script_dir, XidApp.paths[utils.platform_identifier])
        super(XidApp, self).__init__(path, ["-i", device.busId], cwd=os.path.dirname(path))

    def start(self, timeout=app_runner.default_timeout):
        """
        Blocks till XID has been delivered

        Raises exception with EOFError if XID application cannot start.
        """
        super(XidApp, self).start(timeout)
        
        # if matching line is not found then EOFError exception is risen
        self.stdout_readtillmatch(lambda x: x == "All done. Finishing.")

    def __str__(self):
        return "XidApp on device " + str(self.device) + " with " + super(XidApp, self).__str__()
