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

class TestDcgmUnittestsApp(app_runner.AppRunner):
    # Including future supported architectures
    paths = {
            "Linux_32bit": "./apps/x86/testdcgmunittests",
            "Linux_64bit": "./apps/amd64/testdcgmunittests",
            "Linux_ppc64le": "./apps/ppc64le/testdcgmunittests",
            "Linux_aarch64": "./apps/aarch64/testdcgmunittests",
            "Windows_64bit": "./apps/amd64/testdcgmunittests.exe"
            }
    forbidden_strings = [
            # None of this error codes should be ever printed by testnvcmunittests
            "Unknown Error",
            "Uninitialized",
            "Invalid Argument",
            "Already Initialized",
            "Insufficient Size",
            "Driver Not Loaded",
            "Timeout",
            "DCGM Shared Library Not Found",
            "Function Not Found",
            "(null)", # e.g. from printing %s from null ptr
            ]
    def __init__(self, args=None):
        path = TestDcgmUnittestsApp.paths[utils.platform_identifier]
        self.nv_hostengine = None
        self.output_filename = None
        super(TestDcgmUnittestsApp, self).__init__(path, args)
        
        if not test_utils.noLogging:
            self.trace_fname = os.path.join(logger.log_dir, "app_%03d_trace.log" % (self.process_nb))
            self.env["__DCGM_DBG_FILE"] = self.trace_fname
            self.env["__DCGM_DBG_LVL"] = test_utils.loggingLevel
        else:
            self.trace_fname = None
   
    def _process_finish(self, stdout_buf, stderr_buf):
        super(TestDcgmUnittestsApp, self)._process_finish(stdout_buf, stderr_buf)
       
        # Skip this part if --no-logging option is used
        if logger.log_dir is None:
            return

        # Verify that nv_hostengine doesn't print any strings that should never be printed on a working system
        stdout = "\n".join(self.stdout_lines)
        for forbidden_text in TestDcgmUnittestsApp.forbidden_strings:
            assert stdout.find(forbidden_text) == -1, "testdcgmunittests printed \"%s\", this should never happen!" % forbidden_text

        if self.retvalue() != 0:
            stderr = "\n".join(self.stderr_lines)
            logger.warning("testdcgmunittests returned %d" % self.retvalue())
            logger.warning("stdout:\n%s\n" % stdout)
            logger.warning("stderr:\n%s\n" % stderr)

    def __str__(self):
        return "nv_hostengine" + super(TestDcgmUnittestsApp, self).__str__()
