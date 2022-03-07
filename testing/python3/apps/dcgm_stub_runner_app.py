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
import datetime
import subprocess

class DcgmStubRunnerApp(app_runner.AppRunner):
    # Including future supported architectures
    paths = {
            "Linux_32bit": "./apps/x86/stub_library_test",
            "Linux_64bit": "./apps/amd64/stub_library_test",
            "Linux_ppc64le": "./apps/ppc64le/stub_library_test",
            "Linux_aarch64": "./apps/aarch64/stub_library_test",
            "Windows_64bit": "./apps/amd64/stub_library_test.exe"
            }
    forbidden_strings = [
            # None of this error codes should be ever printed by nv-hostengine
            "Unknown Error",
            "Uninitialized",
            "Invalid Argument",
            "(null)", # e.g. from printing %s from null ptr
            ]
    def __init__(self, args=None):
        path = DcgmStubRunnerApp.paths[utils.platform_identifier]
        self.stub = None
        self.output_filename = None
        super(DcgmStubRunnerApp, self).__init__(path, args)
        
        if not test_utils.noLogging:
            self.nvml_trace_fname = os.path.join(logger.log_dir, "app_%03d_nvml_trace.log" % (self.process_nb))
            self.env["__NVML_DBG_FILE"] = self.nvml_trace_fname
            self.env["__NVML_DBG_LVL"] = test_utils.loggingLevel

            self.dcgm_trace_fname = os.path.join(logger.log_dir, "app_%03d_dcgm_trace.log" % (self.process_nb))
            self.env["__DCGM_DBG_FILE"] = self.dcgm_trace_fname
            self.env["__DCGM_DBG_LVL"] = test_utils.loggingLevel
        else:
            self.nvml_trace_fname = None
            self.dcgm_trace_fname = None

   
    def _process_finish(self, stdout_buf, stderr_buf):
        super(DcgmStubRunnerApp, self)._process_finish(stdout_buf, stderr_buf)
       
        # Skip this part if --no-logging option is used
        if logger.log_dir is None:
            return
    
        # Verify that stub_library_test doesn't print any strings that should never be printed
        stdout = "\n".join(self.stdout_lines)
        for forbidden_text in DcgmStubRunnerApp.forbidden_strings:
            assert stdout.find(forbidden_text) == -1, "stub_library_test printed \"%s\", this should never happen!" % forbidden_text

    def __str__(self):
        return "stub_library_test" + super(DcgmStubRunnerApp, self).__str__()
    
    def stdout(self):
        stdout = "\n".join(self.stdout_lines)
        return stdout
    
