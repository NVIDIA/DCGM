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
import time
import datetime
import subprocess
from posix import wait
from subprocess import CalledProcessError

from . import app_runner
import option_parser
import logger
import utils
import test_utils

default_timeout = 10.0 # 10s

class NvHostEngineApp(app_runner.AppRunner):
    # Including future supported architectures
    paths = {
            "Linux_32bit": "./apps/x86/nv-hostengine",
            "Linux_64bit": "./apps/amd64/nv-hostengine",
            "Linux_ppc64le": "./apps/ppc64le/nv-hostengine",
            "Linux_aarch64": "./apps/aarch64/nv-hostengine",
            "Windows_64bit": "./apps/amd64/nv-hostengine.exe"
            }
    forbidden_strings = [
            # None of this error codes should be ever printed by nv-hostengine
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
    supported_profile_tools = ['callgrind', 'massif']
    
    def __init__(self, args=None, profile_dir=None):
        '''
        args: special args to execute nv-hostengine with
        profile_dir: output directory to create which will contain 
                     profiling files if profiling is enabled.
        '''
        path = NvHostEngineApp.paths[utils.platform_identifier]
        self.hostengine_executable = path
        self.nv_hostengine = None
        self._pid = None
        self._retvalue = None
        if test_utils.noLogging:
            self._pidFilename = os.path.join(os.getcwd(), 'nv-hostengine.pid')
        else:
            self._pidFilename = os.path.join(logger.log_dir, 'nv-hostengine.pid')
        
        if option_parser.options.profile:
            self._check_valgrind_installed()
            self.output_dir = self._create_output_dir(option_parser.options.profile, profile_dir)
            
            args, path = self._create_profile_command(args, path, option_parser.options.profile)
            logger.info('profiling output files available under %s' % utils.shorten_path(self.output_dir, 3))

        #Make sure we're writing to a local .pid file in case we're running as non-root
        pidArgs = []
        if args is not None and '--pid' in args:
            if self._pidFilename not in args: # When we retry, --pid self._pidFilename is already there. Don't fail in that case
                raise Exception("Custom --pid parameter is not supported at this time. You must update _terminate_hostengine() as well. args: " + str(args))
        else:
            pidArgs = ['--pid', self._pidFilename]

        if args is None:
            args = pidArgs
        else:
            args.extend(pidArgs)

        super(NvHostEngineApp, self).__init__(path, args)
        
        if not test_utils.noLogging and not option_parser.options.profile:
            self.dcgm_trace_fname = os.path.join(logger.log_dir, "app_%03d_dcgm_trace.log" % (self.process_nb))
            self.env["__DCGM_DBG_FILE"] = self.dcgm_trace_fname
            self.env["__DCGM_DBG_LVL"] = test_utils.loggingLevel
        else:
            self.dcgm_trace_fname = None

        #logger.error("env: %s" % (str(self.env)))

    def _check_valgrind_installed(self):
        output = subprocess.check_output('which valgrind', shell=True).strip()
        if output == '':
            raise Exception('Valgrind must be installed in order to run profiling. ' +
                            '"which valgrind" could not find it.')

    def _create_output_dir(self, profile_tool, profile_dir=None):
        ''' Create and return the output directory for the callgrind files '''
        base_output_dir = os.path.join(logger.log_dir, profile_tool)
        utils.create_dir(base_output_dir)
        
        if profile_dir is not None:
            output_dir = os.path.join(base_output_dir, profile_dir)
        else:
            # if no name specified, store in a folder for the current datetime, including microseconds
            dir_name = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%dT%H:%M:%S.%f')
            output_dir = os.path.join(base_output_dir, dir_name)

        utils.create_dir(output_dir)
        return output_dir
   
    def _create_profile_command(self, args, path, valgrind_tool):
        ''' 
        Return the proper (args, path) to initialize the AppRunner with in order 
        to run the hostengine under callgrind
        '''
        if valgrind_tool not in self.supported_profile_tools:
            raise Exception('%s is not a supported tool for profiling' % valgrind_tool)
        
        common_args = [
            # logfile is needed instead of printing to stderr since AppRunner.terminate tries to gather all stderr 
            # but we are only able to REALLY terminate the process in self.terminate.  This means stderr will 
            # not close as expected and the program will deadlock
            '--tool=%s' % valgrind_tool,
            '--log-file=%s' % os.path.join(self.output_dir, '%s.log.%%p' % valgrind_tool)
        ]
        tool_args = []
        tool_log_file = os.path.join(self.output_dir, valgrind_tool + '.out.%p')
        
        if valgrind_tool == 'callgrind':
            
            tool_args = [
                '--separate-threads=yes', 
                '--dump-instr=yes',         # allow to look at profiling data at machine instr lvl instead of src lines
                '--collect-jumps=yes',      # conditional jump info is collected
                '--collect-systime=yes',    # collect system call times
                '--collect-bus=yes',        # collect atomic instr. calls.  Useful for finding excessive locking
                '--cache-sim=yes',          # collect memory and cache miss/hit info
                '--callgrind-out-file=%s' % tool_log_file,
            ]
            
        elif valgrind_tool == 'massif':
            
            tool_args = [
                '--stacks=yes', # include stack information
                '--massif-out-file=%s' % tool_log_file,
            ]
            
        args = common_args + tool_args + [path] + (args or [])
        path = 'valgrind'
        
        return args, path
   
    def _process_finish(self, stdout_buf, stderr_buf):
        super(NvHostEngineApp, self)._process_finish(stdout_buf, stderr_buf)

        if logger.log_dir is None:
            return

        # Verify that nv_hostengine doesn't print any strings that should never be printed on a working system
        stdout = "\n".join(self.stdout_lines)
        for forbidden_text in NvHostEngineApp.forbidden_strings:
            assert stdout.find(forbidden_text) == -1, "nv_hostengine printed \"%s\", this should never happen!" % forbidden_text

    def __str__(self):
        return "nv_hostengine" + super(NvHostEngineApp, self).__str__()
    
    def start(self, timeout=default_timeout):
        # if an existing hostengine is running, stop it
        self._kill_hostengine(self._getpid())

        #Don't timeout nv-hostengine for now. We already call it from 
        #RunStandaloneHostEngine, which will start and stop the host engine between tests
        timeout = None 
        
        super(NvHostEngineApp, self).start(timeout=timeout)
        
        # get and cache the pid
        waitTime = 5.0
        start = time.time()
        while time.time() - start < waitTime:
            self._pid = self._getpid()
            if self._pid is not None:
                break
            time.sleep(0.050)

        if self._pid is None:
            retValue = super(NvHostEngineApp, self).poll() # Use super method to check status of subprocess object
            if retValue is None:
                # Hostengine did not start up correctly - terminate it so that we clean up subprocess object
                # This prevents multiple zombie processes from being created.
                self.terminate()
            self.validate() # Prevent unecessary failure messages due to not validated processes
            logger.error("Could not start nvhostengine. Output from the failed launch (if any) follows.")
            # log whatever output we have available
            for line in self.stdout_lines:
                logger.info(line)
            for line in self.stderr_lines:
                logger.error(line)
            raise RuntimeError('Failed to start hostengine app')
            
    def _kill_hostengine(self, pid):
        if pid is None:
            return
        
        self._terminate_hostengine()
        utils.wait_for_pid_to_die(pid)
            
    def _getpid_old(self):
        # assuming that only one hostengine exists we do a pgrep for it 
        # we have to specify --P=1 (init process) or else we will also get the PID of 
        # the pgrep shell command.  Use -P instead of --parent because some versions of pgrep only have -P
        try:
            pid = subprocess.check_output('pgrep -P 1 -f "%s"' % os.path.basename(self.hostengine_executable), 
                                        stderr=subprocess.PIPE,
                                        shell=True).strip()

            # verify only one hostengine exists
            pids = pid.split()
            if len(pids) > 1:
                logger.warning('Multiple hostengine pids found: "%s".  Using the last one and hoping for the best.' % pid)

            return int(pids[len(pids) - 1])
        except CalledProcessError:
            return None

    def _getpid(self):
        #Try to read the PID file for the host engine
        if not os.path.isfile(self._pidFilename):
            logger.debug("Pid file %s not found" % self._pidFilename)
            return None

        with open(self._pidFilename) as fp:
            lines = fp.readlines()
            if len(lines) == 0:
                return self._getpid_old()

            pidStr = lines[0].strip()
        #logger.error("pidStr %s" % pidStr)

        procPath = "/proc/" + pidStr + "/"
        #logger.error("exists? %s : %s" % (procPath, str(os.path.exists(procPath))))
        if not os.path.exists(procPath):
            logger.debug("Found pid file %s with pid %s but /proc/%s did not exist" % (self._pidFilename, pidStr, pidStr))
            return None

        return int(pidStr)
    
    def getpid(self):
        return self._pid
    
    def poll(self):
        # terminated via this apprunner
        if self._retvalue is not None:
            return self._retvalue
        
        # still running
        elif self._pid == self._getpid():
            return None
        
        # it was terminated or killed in some other way
        else:
            return 1
    
    def terminate(self):
        """
        Forcfully terminates the host engine daemon and return the app's error code/string.
        
        """
        super(NvHostEngineApp, self).terminate()
        
        if option_parser.options.profile:
            self._remove_useless_profiling_files(option_parser.options.profile)

        self._kill_hostengine(self._pid)
        
        self._retvalue = 0
        return self._retvalue

    def _terminate_hostengine(self):
        try:
            subprocess.check_output([self.hostengine_executable, '--term', '--pid', self._pidFilename],
                                    stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            # We do not want to propogate this error because it will obscure the actual reason why a test fails
            # in some cases when an assertion fails for a test
            logger.error("Failed to terminate host engine! (PID %s)" % self._getpid())
            logger.error("Command: '%s' returned non-zero exit status %s.\nOutput:%s"
                         % (e.cmd, e.returncode, e.output))
            # Log information about any running hostengine processes for better debugging info when failures occur
            test_utils.check_for_running_hostengine_and_log_details(False)

    def _remove_useless_profiling_files(self, profiling_tool):
        ''' 
        Remove any callgrind files that are not useful.
        This happens since starting the nv-hostengine executable creates a new process
        so the initial starting process also is profiled by the profiling tool
        '''
        def is_profiling_file(file):
            return (profiling_tool + '.out.') in file
        
        def profiling_file_is_useful(file):
            if str(self._pid) in file:
                return True
            return False
        
        for file in os.listdir(self.output_dir):
            if is_profiling_file(file) and not profiling_file_is_useful(file):
                logger.debug('deleting useless profiling file "%s"' % file)
                os.remove(os.path.join(self.output_dir, file))
