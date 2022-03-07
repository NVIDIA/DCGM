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

class CudaCtxCreateApp(app_runner.AppRunner):
    """
    Creates a cuda context on a single device and waits for a return char to terminate.

    """
    paths = {
            "Linux_32bit": "./apps/cuda_ctx_create/cuda_ctx_create_32bit",
            "Linux_64bit": "./apps/cuda_ctx_create/cuda_ctx_create_64bit",
            "Linux_ppc64le": "./apps/cuda_ctx_create/cuda_ctx_create_ppc64le",
            "Linux_aarch64": "./apps/cuda_ctx_create/cuda_ctx_create_aarch64",
            "Windows_64bit": "./apps/cuda_ctx_create/cuda_ctx_create_64bit.exe"
            }
    def __init__(self, device):
        self.device = device
        path = os.path.join(utils.script_dir, CudaCtxCreateApp.paths[utils.platform_identifier])
        super(CudaCtxCreateApp, self).__init__(path, ["-i", device.busId, "--getchar"], cwd=os.path.dirname(path))

    def start(self, timeout=app_runner.default_timeout):
        """
        Blocks till cuda ctx is really created

        Raises exception EOFError if ctx application cannot start
        """
        super(CudaCtxCreateApp, self).start(timeout)

        # if matching line is not found then EOFError exception is risen
        self.stdout_readtillmatch(lambda x: x == "Context created")

    def __str__(self):
        return "CudaCtxCreateApp on device " + str(self.device) + " with " + super(CudaCtxCreateApp, self).__str__()

class CudaCtxCreateAdvancedApp(app_runner.AppRunner):
    """
    More universal version of CudaCtxCreateApp which provides access to:
      - creating multiple contexts
      - launching kernels (that use quite a bit of power)
      - allocate additional memory
    See apps/cuda_ctx_create/cuda_ctx_create_32bit -h for more details.

    """

    paths = {
            "Linux_32bit": "./apps/cuda_ctx_create/cuda_ctx_create_32bit",
            "Linux_64bit": "./apps/cuda_ctx_create/cuda_ctx_create_64bit",
            "Linux_ppc64le": "./apps/cuda_ctx_create/cuda_ctx_create_ppc64le",
            "Linux_aarch64": "./apps/cuda_ctx_create/cuda_ctx_create_aarch64",
            "Windows_64bit": "./apps/cuda_ctx_create/cuda_ctx_create_64bit.exe"
            }
    def __init__(self, args, env=None):
        path = os.path.join(utils.script_dir, CudaCtxCreateApp.paths[utils.platform_identifier])
        super(CudaCtxCreateAdvancedApp, self).__init__(path, args, cwd=os.path.dirname(path), env=env)

    def __str__(self):
        return "CudaCtxCreateAdvancedApp with " + super(CudaCtxCreateAdvancedApp, self).__str__()
