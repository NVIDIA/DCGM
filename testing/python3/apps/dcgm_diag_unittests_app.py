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
import string
from . import app_runner
import utils
import test_utils
import logger
import option_parser

class TestDcgmDiagUnittestsApp(app_runner.AppRunner):
    paths = {
            "Linux_32bit": "./apps/x86/testdiag",
            "Linux_64bit": "./apps/amd64/testdiag",
            "Linux_ppc64le": "./apps/ppc64le/testdiag",
            "Linux_aarch64": "./apps/aarch64/testdiag",
            "Windows_64bit": "./apps/amd64/testdiag.exe"
            }

    def __init__(self, args=None):
        path = TestDcgmDiagUnittestsApp.paths[utils.platform_identifier]
        super(TestDcgmDiagUnittestsApp, self).__init__(path, args)
