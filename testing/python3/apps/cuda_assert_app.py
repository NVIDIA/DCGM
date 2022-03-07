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
import test_utils

class RunCudaAssert(app_runner.AppRunner):
    """ Class to assert a Cuda Kernel and generate a XID 43 Error """
    
    paths = {
            "Linux_64bit": "./apps/cuda_ctx_create/cuda_assert_64bit",
            "Linux_ppc64le": "./apps/cuda_ctx_create/cuda_assert_ppc64le",
            "Linux_aarch64": "./apps/cuda_ctx_create/cuda_assert_aarch64",
            }

    def __init__(self, args, env=None):
        path = os.path.join(utils.script_dir, RunCudaAssert.paths[utils.platform_identifier])
        super(RunCudaAssert, self).__init__(path, args, cwd=os.path.dirname(path), env=env)

    def start(self, timeout=app_runner.default_timeout):
        """
        Blocks till cuda ctx is really created
        Raises Exception if assert does not work
        """

        super(RunCudaAssert, self).start(timeout)

        with test_utils.assert_raises(EOFError):
            # if matching line is not found then EOFError exception is risen
            self.stdout_readtillmatch(lambda x: x == "Assertion `false` failed")

    def __str__(self):
        return "RunCudaAssert on device " + super(RunCudaAssert, self).__str__()
