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
import sys
import os
import shutil
import datetime
import threading
import trace
import zipfile
import string
import re
import libs_3rdparty.colorama as colorama
import dcgm_structs
import dcgm_agent_internal
import option_parser
import utils
import test_utils

log_dir = None
have_removed_log_dir = False #Have we removed the log dir from previous runs?
default_log_dir = '_out_runLogs'
_log_file = None
_log_file_counter = None
summary_filename = "terminal_main.py_stdout.txt"
_summary_file = None
log_archive_filename = "all_results.zip"
dcgm_trace_log_filename = None
nvml_trace_log_filename = None
_indent_lvl = 0
_coloring_enabled = True #coloring is applied even there is no file descriptor connected to tty like device
_message_levels = (FATAL, ERROR, INFO, WARNING, DEBUG) = list(range(5))
messages_level_counts = [0] * len(_message_levels)
level_names = ("FATAL", "ERROR", "INFO", "WARNING", "DEBUG")

stdout_loglevel = WARNING

def caller_function_details(depth=1):
    """
    Returns tuple with details of function up the call stack
    returns (file_name, func_name, file_nb)

    """
    import inspect
    # Get up the stack of functions
    func = inspect.currentframe().f_back
    for i in range(depth):
        func = func.f_back
    func = func.f_code

    return (os.path.relpath(func.co_filename), func.co_name, func.co_firstlineno)

def addtrace_logging(module, filter_fns=lambda name, fn: True):
    '''
    Find all functions in module and add logging before and after each call.
    '''
    from functools import wraps
    import inspect

    for name in dir(module):
        if name.startswith("_"):
            continue

        fn = getattr(module, name)
        if not inspect.isfunction(fn):
            continue

        if not filter_fns(name, fn):
            continue

        def genfunc(fn):
            @wraps(fn)
            def tmpfn(*args, **kwargs):
                debug("Call %s(args: %s kwargs: %s)" % (fn.__name__, list(zip(inspect.getfullargspec(fn).args, args)), kwargs), caller_depth=1)
                try:
                    res = fn(*args, **kwargs)
                    debug("Call %s returned: %s" % (fn.__name__, res), caller_depth=1)
                    return res
                except Exception as e:
                    debug("Call %s raised: %s" % (fn.__name__, e), caller_depth=1)
                    raise
            return tmpfn
        setattr(module, name, genfunc(fn))

def setup_environment():
    global _log_file
    global _summary_file
    global log_dir
    global dcgm_trace_log_filename
    global nvml_trace_log_filename
    global have_removed_log_dir
    curr_log_dir = None

    # Users can specify a non-default logging base path via the command line
    if option_parser.options.log_dir:
        assert os.path.exists(option_parser.options.log_dir)
        curr_log_dir = os.path.normpath(os.path.join(option_parser.options.log_dir, default_log_dir))
    else:
        curr_log_dir = os.path.join(os.getcwd(), default_log_dir)
    log_dir = os.path.realpath(curr_log_dir)

    dcgm_trace_log_filename = os.path.join(log_dir, "dcgm_trace.log")
    nvml_trace_log_filename = os.path.join(log_dir, "nvml_trace.log")

    #We clean up the log dir from previous runs once per test run in order to prevent the constant accumulation of logs
    if not have_removed_log_dir:
        have_removed_log_dir = True #We set this boolean here so we don't incorrectly remove this directory if this function is called again
        if os.path.isdir(log_dir):
            shutil.rmtree(log_dir, ignore_errors=True)

    if not test_utils.noLogging:
        try:
            os.mkdir(log_dir)
        except OSError:
            pass
        os.chmod(log_dir, 0o777) # so that non-root tests could write to this directory
        _log_file = open(os.path.join(log_dir, "log.txt"), "a")
        _summary_file = open(os.path.join(log_dir, summary_filename), "a")

        #Log both NVML and DCGM so we can the underlying NVML calls for the DCGM data
        os.putenv('__NVML_DBG_FILE', nvml_trace_log_filename)
        os.putenv('__NVML_DBG_LVL', test_utils.loggingLevel)
        os.putenv('__NVML_DBG_APPEND', '1')
        os.putenv('__DCGM_DBG_FILE', dcgm_trace_log_filename)
        os.putenv('__DCGM_DBG_LVL', test_utils.loggingLevel)
        os.putenv('__DCGM_DBG_APPEND', '1')
    
        # create the file upfront ant set proper chmod
        # so that non-root tests could dcgmInit and write to this log
        with open(nvml_trace_log_filename, "a"):
            pass
        os.chmod(nvml_trace_log_filename, 0o777)
        with open(dcgm_trace_log_filename, "a"):
            pass
        os.chmod(dcgm_trace_log_filename, 0o777) 

        addtrace_logging(dcgm_structs, 
                # Don't attach trace logging to some functions
                lambda name, fn: name not in ["dcgmErrorString", "dcgmStructToFriendlyObject"])
    else:
        #Not logging. Clear the environmental variables
        envVars = ['__NVML_DBG_FILE', '__NVML_DBG_LVL', '__NVML_DBG_APPEND', 
                   '__DCGM_DBG_FILE', '__DCGM_DBG_LVL', '__DCGM_DBG_APPEND']
        for envVar in envVars:
            if envVar in os.environ:
                del(os.environ[envVar])

    global _coloring_enabled
    if sys.stdout.isatty():
        colorama.init()
        _coloring_enabled = True
    else:
        _coloring_enabled = False

    if not test_utils.SubTest.FAILED:
        info("Package version information:")
        with IndentBlock():
            try:
                version_file = open(os.path.join(utils.script_dir, "data/version.txt"), "wr+")
                info("".join(version_file.readlines()))
                version_file.close()
            except IOError:
                warning("No build version information")
        
    if os.path.exists(log_archive_filename):
        info("Removing old %s" % log_archive_filename)
        try:
            os.remove(log_archive_filename)
        except IOError:
            pass

def close():
    """
    Closes all the debug file streams and archives all logs 
    into single zip file logger.log_archive_filename
    
    """
    
    debug("Storing all logs in " + log_archive_filename)

    if _coloring_enabled:
        colorama.deinit()

    if log_dir is not None and os.path.isdir(log_dir):
        try:
            zip = zipfile.ZipFile(log_archive_filename, 'w', compression=zipfile.ZIP_DEFLATED)
            for root, dirs, files in os.walk(log_dir):
                for file in files:
                        zip.write(os.path.join(root, file))
            zip.close()
            os.chmod(log_archive_filename, 0o777)
        except Exception:
            pass

def run_with_coverage(fn):
    """
    Runs the function (that shouldn't take any arguments!) with coverage tool.
    Stores the results in a log and returns the results.

    """
    
    coverage_trace = trace.Trace(trace=0, ignoredirs=[sys.prefix, sys.exec_prefix])
    try:
        coverage_trace.runfunc(lambda ignore1, ignore2: fn(), [], {})
    finally:
        results = coverage_trace.results()
        coverdir = os.path.join(log_dir, "pycoverage")
        if not os.path.exists(coverdir):
            os.mkdir(coverdir)
        os.chmod(coverdir, 0o777) # so that non-root tests could write to this directory
        results.write_results(show_missing=False, coverdir=coverdir)
        return results

_defered_lines = []

log_lock = threading.Lock()
_log_id = 0
def log(level, msg, caller_depth=0, defer=False):
    def apply_coloring(level, line):
        if option_parser.options.eris:
            return line
        if not _coloring_enabled:
            return line
        if level == FATAL or level == ERROR:
            return colorama.Fore.RED + line + colorama.Fore.RESET
        elif level == WARNING:
            return colorama.Fore.YELLOW + line + colorama.Fore.RESET
        coloring = [
                ("SKIPPED", colorama.Fore.CYAN),
                ("Reason: .*", colorama.Fore.CYAN),
                ("FAILED", colorama.Fore.RED),
                ("DEBUG", colorama.Fore.MAGENTA),
                ("FAILURE_LOGGED", colorama.Fore.CYAN),
                ]
        for (what, color) in coloring:
            coloring_match = re.match("(.*)(%s)(.*)" % what, line)
            if coloring_match:
                groups = coloring_match.groups()
                line = "".join([groups[0], color, groups[1], colorama.Fore.RESET, groups[2]])
        return line

    global _log_id
    with log_lock:
        _log_id += 1
        messages_level_counts[level] += 1

        if option_parser.options.eris:
            indent = ""
        else:
            indent = "    " * _indent_lvl

        timestamp = datetime.datetime.now()

        if not defer and level <= stdout_loglevel:
            # will be printing this message to stdout so we need to flush all deferred lines first
            for (log_id, defered_level, line) in _defered_lines:
                print(apply_coloring(defered_level, line))
                if _summary_file and (not test_utils.noLogging):
                    _summary_file.write(line)
                    _summary_file.write("\n")
            del _defered_lines[:]

        for msg in msg.splitlines():
            if level <= stdout_loglevel:
                if level != INFO:
                    level_name = level_names[level] + ": "
                else:
                    level_name = ""
                line = "%s%s%s" % (indent, level_name, msg)
                if defer:
                    _defered_lines.append((_log_id, level, line))
                else:
                    print(apply_coloring(level, line))
                    if _summary_file and (not test_utils.noLogging):
                        _summary_file.write(line)
                        _summary_file.write("\n")

            if _log_file and (not test_utils.noLogging):
                _log_file.write("%-8s: [%s] %s%s\n" % (level_names[level], timestamp, indent, msg))

                try:
                    fndetails = caller_function_details(caller_depth + 1)
                    #dcgm_agent_internal.dcgmTraceLogPrintLine("<testing %s> [%s - %s:%s:%d] %s%s" %
                    #     (level_names[level], timestamp, fndetails[0], fndetails[1], fndetails[2], indent, msg))
                except dcgm_structs.DCGMError:
                    pass

                # Every 100 messages force OS to write to log file to HD in case OS kernel panics
                if _log_id % 100 == 0:
                    os.fsync(_log_file)

    if level == FATAL:
        close()
        os._exit(1)

    return _log_id

def pop_defered(log_id):
    """
    Removes the message from deferred lines buffer and returns True. Removed log_id must be the last log_id on the list.
    If the log_id is not found returns False.

    Note: Messages added to defered log need to be removed in reverse order that they were added (like unrolling stack).
    """
    result = False
    while _defered_lines and _defered_lines[-1][0] == log_id:
        _defered_lines.pop()
        result = True
    return result

def fatal(msg="\n", caller_depth = 0, defer=False):
    """
    Calls sys.exit at the end
    """
    return log(FATAL, msg, caller_depth + 1, defer)

def error(msg="\n", caller_depth = 0, defer=False):
    return log(ERROR, msg, caller_depth + 1, defer)

def info(msg="\n", caller_depth = 0, defer=False):
    return log(INFO, msg, caller_depth + 1, defer)

def warning(msg="\n", caller_depth = 0, defer=False):
    return log(WARNING, msg, caller_depth + 1, defer)

def debug(msg="\n", caller_depth = 0, defer=False):
    return log(DEBUG, msg, caller_depth + 1, defer)

def indent_icrement(val=1):
    global _indent_lvl
    _indent_lvl += val

def indent_decrement(val=1):
    global _indent_lvl
    _indent_lvl -= val

class IndentBlock(object):
    def __init__(self, val=1):
        self._old_indent = _indent_lvl
        self._val = val

    def __enter__(self):
        indent_icrement(self._val)

    def __exit__(self, exception_type, exception, trace):
        indent_decrement(self._val)

# Sample usage
if __name__ == "__main__":
    option_parser.parse_options()
    setup_environment()

    info("This gets printed to stdout")
    debug("This by default gets printed only to debug log in %s dir" % (log_dir))
    log_id1 = info("This message is deferred. It can be removed from stdout but not from debug log", defer=True)
    debug("Even when one prints to debug log, deferred line can be removed")
    assert pop_defered(log_id1) == True

    log_id2 = info("This message is also deferred. But will get printed as soon as error message requested", defer=True)
    with IndentBlock(): # indent this message
        error("This causes the deferred message to be printed")
    assert pop_defered(log_id2) == False

    close()

