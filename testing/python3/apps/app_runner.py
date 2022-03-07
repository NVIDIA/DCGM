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
import subprocess
import os
import threading
import string
import datetime
import signal

import logger
import option_parser
import utils
import test_utils

default_timeout = 10.0 # 10s

class AppRunner(object):
    """
    Class for running command line applications. It handles timeouts, logging and reading stdout/stderr.
    Stdout and stderr of an application is also stored in log output dir in files process_<NB>_stdout/stderr.txt

    If application finished with non 0 error code you need to mark it with .validate() function. Otherwise testing
    framework will fail a subtest. AppRunner is also "validated" when .terminate() is called.

    You can access all lines read so far (or when the application terminates all lines printed) from attributes
    .stdout_lines
    .stderr_lines

    You can see how long the application ran for +/- some minimal overhead (must run() for time to be accurate:
    .runTime

    # Sample usage
    app = AppRunner("nvidia-smi", ["-l", "1"])
    app.run(timeout=2.5)
    print "\n".join(app.stdout_lines)

    Notes: AppRunner works very closely with test_utils SubTest environment. SubTest at the end of the test
           checks that all applications finished successfully and kills applications that didn't finish by
           the end of the test.
    """
    RETVALUE_TERMINATED = "Terminated"
    RETVALUE_TIMEOUT = "Terminated - Timeout"

    _processes = []               # Contains list of all processes running in the background
    _processes_not_validated = [] # Contains list of processes that finished with non 0 error code 
                                  #     and were not marked as validated
    _process_nb = 0

    def __init__(self, executable, args=None, cwd=None, env=None):
        self.executable = executable
        if args is None:
            args = []
        self.args = args
        self.cwd = cwd
        if env is None:
            env = dict()
        self.env = env

        self._timer = None              # to implement timeout
        self._subprocess = None
        self._retvalue = None           # stored return code or string when the app was terminated
        self._lock = threading.Lock()   # to implement thread safe timeout/terminate
        self.stdout_lines = []          # buff that stores all app's output
        self.stderr_lines = []
        self._logfile_stdout = None
        self._logfile_stderr = None
        self._is_validated = False
        self._info_message = False
        
        self.process_nb = AppRunner._process_nb
        AppRunner._process_nb += 1

    def run(self, timeout=default_timeout):
        """
        Run the application and wait for it to finish. 
        Returns the app's error code/string
        """
        self.start(timeout)
        return self.wait()
    
    def start(self, timeout=default_timeout):
        """
        Begin executing the application.
        The application may block if stdout/stderr buffers become full.
        This should be followed by self.terminate() or self.wait() to finish execution.
        Execution will be forcefully terminated if the timeout expires.
        If timeout is None, then this app will never timeout.
        """
        assert self._subprocess is None

        logger.debug("Starting " + str(self))
        
        env = self._create_subprocess_env()
        if utils.is_linux():
            if os.path.exists(self.executable):
                # On linux, for binaries inside the package (not just commands in the path) test that they have +x
                # e.g. if package is extracted on windows and copied to Linux, the +x privileges will be lost
                assert os.access(self.executable, os.X_OK), "Application binary %s is not executable! Make sure that the testing archive has been correctly extracted." % (self.executable)
        self.startTime = datetime.datetime.now()
        self._subprocess = subprocess.Popen(
                [self.executable] + self.args, 
                stdin=None, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                cwd=self.cwd,
                env=env)
        AppRunner._processes.append(self) # keep track of running processe
        # Start timeout if we want one
        self._timer = None
        if timeout is not None:
            self._timer = threading.Timer(timeout, self._trigger_timeout)
            self._timer.start()

        if not test_utils.noLogging:
            def args_to_fname(args):
                # crop each argument to 16 characters and make sure the output string is no longer than 50 chars
                # Long file names are hard to read (hard to find the extension of the file)
                # Also python sometimes complains about file names being too long.
                #   IOError: [Errno 36] File name too long
                return "_".join([utils.string_to_valid_file_name(x)[:16] for x in self.args])[:50]
            shortname = os.path.basename(self.executable) + "_" + args_to_fname(self.args)
            stdout_fname = os.path.relpath(os.path.join(
                logger.log_dir, "app_%03d_%s_stdout.txt" % (self.process_nb, shortname)))
            stderr_fname = os.path.relpath(os.path.join(
                logger.log_dir, "app_%03d_%s_stderr.txt" % (self.process_nb, shortname)))
            # If the app fails, this message will get printed. If it succeeds it'll get popped in _process_finish
            self._info_message = logger.info("Starting %s...\nstdout in %s\nstderr in %s" % (
                str(self)[:64], # cut the string to make it more readable
                stdout_fname, stderr_fname), defer=True)
            self._logfile_stdout = open(stdout_fname, "w", encoding='utf-8')
            self._logfile_stderr = open(stderr_fname, "w", encoding='utf-8')

    def _process_finish(self, stdout_buf, stderr_buf):
        """
        Logs return code/string and reads the remaining stdout/stderr.

        """
        logger.debug("Application %s returned with status: %s" % (self.executable, self._retvalue))
        self.runTime = datetime.datetime.now() - self.startTime

        self._split_and_log_lines(stdout_buf, self.stdout_lines, self._logfile_stdout)
        self._split_and_log_lines(stderr_buf, self.stderr_lines, self._logfile_stderr)

        if self._logfile_stdout:
            self._logfile_stdout.close()
        if self._logfile_stderr:
            self._logfile_stderr.close()
        AppRunner._processes.remove(self)
        if self._retvalue != 0 and self._retvalue != AppRunner.RETVALUE_TERMINATED:
            AppRunner._processes_not_validated.append(self)
        else:
            self._is_validated = True
            logger.pop_defered(self._info_message)

    def wait(self):
        """
        Wait for application to finish and return the app's error code/string

        """
        if self._retvalue is not None:
            return self._retvalue

        logger.debug("Waiting for application %s, pid %d to finish" % (str(self), self._subprocess.pid))
        stdout_buf, stderr_buf = self._subprocess.communicate()
        if self._timer is not None:
            self._timer.cancel()

        with self._lock:                   # set ._retvalue in thread safe way. Make sure it wasn't set by timeout already
            if self._retvalue is None:
                self._retvalue = self._subprocess.returncode
                self._process_finish(stdout_buf.decode('utf-8'), stderr_buf.decode('utf-8'))

        return self._retvalue

    def poll(self):
        if self._retvalue is None:
            self._retvalue = self._subprocess.poll()
            if self._retvalue is not None:
                stdout_buf = self._read_all_remaining(self._subprocess.stdout)
                stderr_buf = self._read_all_remaining(self._subprocess.stderr)
                self._process_finish(stdout_buf, stderr_buf)
                
        return self._retvalue

    def _trigger_timeout(self):
        """
        Function called by timeout routine. Kills the app in a thread safe way.

        """
        logger.warning("App %s with pid %d has timed out. Killing it." % (self.executable, self.getpid()))
        with self._lock: # set ._retvalue in thread safe way. Make sure that app wasn't terminated already
            if self._retvalue is not None:
                return self._retvalue

            self._subprocess.kill()
            stdout_buf = self._read_all_remaining(self._subprocess.stdout)
            stderr_buf = self._read_all_remaining(self._subprocess.stderr)
            self._retvalue = AppRunner.RETVALUE_TIMEOUT
            self._process_finish(stdout_buf, stderr_buf)

            return self._retvalue

    def _create_subprocess_env(self):
        ''' Merge additional env with current env '''
        env = os.environ.copy()
        for key in self.env:
            env[key] = self.env[key]
        return env

    def validate(self):
        """
        Marks the process that finished with error code as validated - the error was either expected or handled by the caller
        If process finished with error but wasn't validated one of the subtest will fail.

        """
        assert self.retvalue() != None, "This function shouldn't be called when process is still running"

        if self._is_validated:
            return
        self._is_validated = True
        self._processes_not_validated.remove(self)
        logger.pop_defered(self._info_message)

    def terminate(self):
        """
        Forcfully terminates the application and return the app's error code/string.

        """
        with self._lock: # set ._retvalue in thread safe way. Make sure that app didn't timeout
            if self._retvalue is not None:
                return self._retvalue

            if self._timer is not None:
                self._timer.cancel()
            self._subprocess.kill()
            
            stdout_buf = self._read_all_remaining(self._subprocess.stdout)
            stderr_buf = self._read_all_remaining(self._subprocess.stderr)
            self._retvalue = AppRunner.RETVALUE_TERMINATED
            self._process_finish(stdout_buf, stderr_buf)
            
            return self._retvalue

    def signal(self, signal):
        """
        Send a signal to the process
        """
        self._subprocess.send_signal(signal)

    def _read_all_remaining(self, stream):
        """
        Return a string representing the entire remaining contents of the specified stream
        This will block if the stream does not end
        Should only be called on a terminated process
        """
        out_buf = ""

        while True:
            rawline = stream.readline().decode('utf-8')
            if rawline == "":
                break
            else:
                out_buf += rawline

        return out_buf

    def _split_and_log_lines(self, input_string, buff, log_file):
        """
        Splits string into lines, removes '\\n's, and appends to buffer & log file

        """
        lines = input_string.splitlines()

        for i in range(len(lines)):
            lines[i] = lines[i].rstrip("\n\r")
            if log_file:
                log_file.write(lines[i])
                log_file.write("\n")
            buff.append(lines[i])

    def stdout_readtillmatch(self, match_fn):
        """
        Blocking function that reads input until function match_fn(line : str) returns True.
        If match_fn didn't match anything function raises EOFError exception
        """
        logger.debug("stdout_readtillmatch called", caller_depth=1)

        while True:
            rawline = self._subprocess.stdout.readline().decode("utf-8")
            if rawline == "":
                break
            else:
                rawline = rawline.rstrip("\n\r")
                line = rawline
                if self._logfile_stdout:
                    self._logfile_stdout.write(line)
                    self._logfile_stdout.write("\n")
                self.stdout_lines.append(line)

            if match_fn(rawline) is True:
                return 
        raise EOFError("Process finished and requested match wasn't found")

    def retvalue(self):
        """
        Returns code/string if application finished or None otherwise.

        """
        if self._subprocess.poll() is not None:
            self.wait()
        return self._retvalue

    def getpid(self):
        """
        Returns the pid of the process

        """
        
        return self._subprocess.pid

    def __str__(self):
        return ("AppRunner #%d: %s %s (cwd: %s; env: %s)" %
                (self.process_nb, self.executable, " ".join(self.args), self.cwd, self.env))
    def __repr__(self):
        return str(self)

    @classmethod
    def clean_all(cls):
        """
        Terminate all processes that were created using this class and makes sure that all processes that were spawned were validated.

        """
        import test_utils
        def log_output(message, process):
            """
            Prints last 10 lines of stdout and stderr for faster lookup
            """
            logger.info("%s: %s" % (message, process))
            
            numLinesToPrint = 100
            #Print more lines for ERIS since this is all we'll have to go by
            if option_parser.options.dvssc_testing or option_parser.options.eris:
                numLinesToPrint = 500
            
            logger.info("Last %d lines of stdout" % numLinesToPrint)
            with logger.IndentBlock():
                for line in process.stdout_lines[-numLinesToPrint:]:
                    logger.info(line)
            logger.info("Last %d lines of stderr" % numLinesToPrint)
            with logger.IndentBlock():
                for line in process.stderr_lines[-numLinesToPrint:]:
                    logger.info(line)

        with test_utils.SubTest("not terminated processes", quiet=True):
            assert AppRunner._processes == [], "Some processes were not terminated by previous test: " + str(AppRunner._processes)
        for process in AppRunner._processes[:]:
            log_output("Unterminated process", process)
            process.terminate()
        with test_utils.SubTest("not validated processes", quiet=True):
            for process in AppRunner._processes_not_validated:
                log_output("Process returned %s ret code" % process.retvalue(), process)
            assert AppRunner._processes_not_validated == [], "Some processes failed and were not validated by previous test: " + str(AppRunner._processes_not_validated)
        AppRunner._processes_not_validated = []
