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
import os
import string

from . import app_runner
import dcgm_structs
import dcgm_agent_internal
import utils
import test_utils
import logger
import option_parser
from . import performance_stats

class NvidiaSmiApp(app_runner.AppRunner):
    # TODO add option to also run just compiled nvidia-smi
    paths = {
            "Linux_32bit": "nvidia-smi", # it should be in the path
            "Linux_64bit": "nvidia-smi", # it should be in the path
            "Linux_aarch64": "nvidia-smi", # it should be in the path
            "Windows_64bit": os.path.join(os.getenv("ProgramFiles", "C:/Program Files"), "NVIDIA Corporation/NVSMI/nvidia-smi.exe")
            }
    forbidden_strings = [
            # None of this error codes should be ever printed by nvidia-smi
            "Unknown Error",
            "Uninitialized",
            "Invalid Argument",
            "Already Initialized",
            "Insufficient Size",
            "Insufficient External Power",
            "Driver Not Loaded",
            "Timeout",
            "Interrupt Request Issue",
            "NVML Shared Library Not Found",
            "Function Not Found",
            "Corrupted infoROM",
            "ERR!", # from non-verbose output
            "(null)", # e.g. from printing %s from null ptr
            ]
    def __init__(self, args=None):
        path = NvidiaSmiApp.paths[utils.platform_identifier]
        self.output_filename = None
        super(NvidiaSmiApp, self).__init__(path, args)
        
        if not test_utils.noLogging:
            self.trace_fname = os.path.join(logger.log_dir, "app_%03d_trace.log" % (self.process_nb))
            self.env["__NVML_DBG_FILE"] = self.trace_fname
            self.env["__NVML_DBG_LVL"] = test_utils.loggingLevel
        else:
            self.trace_fname = ""

    def append_switch_filename(self, filename=None):
        """
        Appends [-f | --filename] switch to args.
        If filename is None than filename is generated automatically

        """
        # Skip this part if --no-logging option is used
        if logger.log_dir is None:
            return None

        if filename is None:
            filename = os.path.join(logger.log_dir, "app_%03d_filename_output.txt" % (self.process_nb))

        self.args.extend(["-f", filename])
        self.output_filename = filename

        return filename
    
    def _process_finish(self, stdout_buf, stderr_buf):
        super(NvidiaSmiApp, self)._process_finish(stdout_buf, stderr_buf)
       
        # Skip this part if --no-logging option is used
        if logger.log_dir is None:
            return
         
        # TODO, debug builds can print to stderr.  We can check for release build here
        #assert self.stderr_lines == [], "nvidia-smi printed something to stderr. It shouldn't ever do that!"

        # Verify that nvidia smi doesn't print any strings that should never be printed on a working system
        stdout = "\n".join(self.stdout_lines)
        for forbidden_text in NvidiaSmiApp.forbidden_strings:
            assert stdout.find(forbidden_text) == -1, "nvidia-smi printed \"%s\", this should never happen!" % forbidden_text

    def __str__(self):
        return "nvidia-smi" + super(NvidiaSmiApp, self).__str__()

