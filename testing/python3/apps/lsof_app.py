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
import logger

class LsofApp(app_runner.AppRunner):
    """
    Allows to query processes that have some file open (e.g. device node)

    """

    paths = {
            "Linux_32bit": "lsof",
            "Linux_64bit": "lsof",
            "Linux_ppc64le": "lsof",
            "Linux_aarch64": "lsof",
            }
    def __init__(self, fname):
        path = LsofApp.paths[utils.platform_identifier]
        self.processes = None
        self.fname = fname
        super(LsofApp, self).__init__(path, ["-F", "-V", fname])
    
    def start(self, timeout=app_runner.default_timeout):
        # try to run as root, otherwise the list of processes might be incomplete
        # (e.g. it won't report processes running by other users)
        with test_utils.tryRunAsRoot():
            super(LsofApp, self).start(timeout)

    def _process_finish(self, stdout_buf, stderr_buf):
        super(LsofApp, self)._process_finish(stdout_buf, stderr_buf)

        if self.retvalue() == 1 and self.stdout_lines and self.stdout_lines[0].startswith("lsof: no file use located: "):
            # lsof returns with error code 1 and prints message when no processes opened the file 
            self.validate()
            self._retvalue = 0 # Fake success
            self.processes = []
            return

        if self.retvalue() == 1 and self.stderr_lines and self.stderr_lines[0].endswith("No such file or directory"):
            # lsof returns with error code 1 and prints message when target file doesn't exist
            self.validate()
            self._retvalue = 0 # Fake success
            self.processes = [] # no file, no processes using it
            return

        if self.retvalue() != 0:
            #Print out stdout and stderr so we can see it in eris
            logger.warning("lsof with args %s had a retval of %s. stderr:" % (self.args, str(self._retvalue)))
            if self.stderr_lines:
                logger.warning(str(self.stderr_lines))
            logger.warning("stdout:")
            if self.stdout_lines:
                logger.warning(str(self.stdout_lines))

        assert self._retvalue == 0, "Failed to read processes that have the file opened. Read process log for more details"
        assert len(self.stdout_lines) > 0, "Behavior of lsof changed. Returned 0 return code but stdout is empty"

        self.processes = []
        if utils.is_esx_hypervisor_system():
            # ESX lsof ignores input args and outputs data in its own format
            lsofre = re.compile("^(\d+)\s+(\S+)\s+(\S+)\s+(-?\d+)\s+(.*)")
            for line in self.stdout_lines[2:]:
                (pid, pname, fdtype, fd, fpath) = lsofre.match(line).groups()
                if fpath == self.fname:
                    self.processes.append([int(pid), pname])
        else:
            last_value = [None, None]
            for line in self.stdout_lines:
                if not line:
                    continue # skip empty lines
                tag = line[0]
                content = line[1:]
                if tag == "p":
                    last_value = [int(content), None]
                    self.processes.append(last_value)
                elif tag == "c":
                    last_value[1] = content

    def get_processes(self, ignore_pids=None, ignore_names=None):
        """
        Returns list of processes (list of pairs (pid, process name)).
        By default it filters out current process from the list.

        ignore_pids - by default contains current pid
        ignore_names - by default empty (but could be used e.g. to easily filter "Xorg")
        """

        ignore_pids = ignore_pids or [os.getpid()]
        ignore_names = ignore_names or []

        if self.processes is None:
            self.run()
        result = self.processes[:]

        if ignore_pids:
            result = [p for p in result if p[0] not in ignore_pids]
        if ignore_names:
            result = [p for p in result if p[1] not in ignore_names]

        return result
