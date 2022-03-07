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
from functools import wraps
import inspect
import os
import platform
import string
import traceback
from collections import namedtuple
import stat
import time
import apps
import re
import ctypes
import sys
from shutil import which as find_executable

from progress_printer import *
import logger
import option_parser
import utils
import apps
import dcgm_agent
import dcgm_structs
import dcgm_agent_internal
import dcgm_structs_internal
import dcgm_fields
import DcgmDiag
import dcgmvalue
import pydcgm
import version

from dcgm_structs import DCGM_ST_INIT_ERROR, dcgmExceptionClass
import nvidia_smi_utils
import errno
import shlex
import xml.etree.ElementTree as ET
import subprocess
from subprocess import Popen, check_call, CalledProcessError, PIPE

test_directory = 'tests'
noLogging = True
noLoggingBackup = noLogging
reRunning = False
loggingLevel = "DEBUG" #Level to use for logging. These come from DcgmLogging.h

def check_output(*args, **kwargs):
    kwargs['universal_newlines'] = True
    return subprocess.check_output(*args, **kwargs)

def set_tests_directory(testDir):
    '''
    Set the directory where test .py files will be looked for (ex: 'tests' for DCGM)
    '''
    global test_directory

    test_directory = testDir

def is_nvswitch_detected():
    """ Tries to detect if nvswitch is present """

    try:
        lsPciOutput = check_output("lspci | grep -i nvidia", shell=True)
    except CalledProcessError as e:
        logger.info("lspci did not successfully execute. Ignoring {_e}".format(_e=str(e)))
        return True

    # pylint: disable=unsupported-membership-test
    if "Bridge: NVIDIA Corporation Device" in lsPciOutput:
        return True
    else:
        return False

def is_hostengine_running():
    """ Helper function to detect if there is an existing host engine running """

    processList = check_output(["ps", "-ef"])
    # pylint: disable=unsupported-membership-test
    if "nv-hostengine" in processList:
        return True
    else:
        return False

def check_for_running_hostengine_and_log_details(quiet):
    """ 
    Helper function to check if there is an existing hostengine running. 
    Logs entries (level INFO) from `ps -ef` output which correspond to running hostengine processes.
    If no hostengine process is found, logs "No hostengine process found" (level INFO)

    Returns True if a running host engine was found, and False otherwise.
    """

    header = "*************** List of nv-hostengine processes ***************"
    ps_output = check_output(["ps", "-ef"])
    processes_list = ps_output.split("\n")
    process_ids = []
    for process in processes_list:
        if "nv-hostengine" in process:
            if header != None:
                if not quiet:
                    logger.info(header)
                header = None
            if not quiet:
                logger.info(process)
            fields = process.split(' ')
            if len(fields) > 1 and fields[1]:
                process_ids.append(fields[1])
    
    if header is None:
        if not quiet:
            logger.info("*************** End list of nv-hostengine processes ***************")
    elif not quiet: 
        logger.info("No hostengine process found")

    return process_ids

def run_p2p_bandwidth_app(args):

    """ Helper function to run the p2p_bandwidth test """

    p2p_app = apps.RunP2Pbandwidth(args)
    p2p_app.start()
    pid = p2p_app.getpid()
    ret = p2p_app.wait()
    p2p_app.validate()

    logger.info("The p2p_bandwidth pid is %s" % pid)
    return ret, p2p_app.stdout_lines, p2p_app.stderr_lines

def run_nvpex2_app(args):

    """ Helper function to run the nvpex2 app for error injection """

    nvpex2_app = apps.RunNVpex2(args)
    nvpex2_app.start()
    pid = nvpex2_app.getpid()
    ret = nvpex2_app.wait()
    nvpex2_app.validate()

    logger.info("The nvpex2 pid is %s" % pid)
    return ret, nvpex2_app.stdout_lines, nvpex2_app.stderr_lines

def get_cuda_visible_devices_str(handle, entityGroupId, entityId):
    entities = []
    fieldIds = [ dcgm_fields.DCGM_FI_DEV_CUDA_VISIBLE_DEVICES_STR ]
    entities.append(dcgm_structs.c_dcgmGroupEntityPair_t(entityGroupId, entityId))
    flags = dcgm_structs.DCGM_FV_FLAG_LIVE_DATA
    fieldValues = dcgm_agent.dcgmEntitiesGetLatestValues(handle, entities, fieldIds, flags)
    return fieldValues[0].value.str

def get_cuda_driver_version(handle, gpuId):
    '''
    Returns cuda driver version value as a tuple of integes like [10, 1] (10.1)
    '''
    entities = []
    fieldIds = [ dcgm_fields.DCGM_FI_CUDA_DRIVER_VERSION ]
    entities.append(dcgm_structs.c_dcgmGroupEntityPair_t(dcgm_fields.DCGM_FE_GPU, gpuId))
    flags = dcgm_structs.DCGM_FV_FLAG_LIVE_DATA
    fieldValues = dcgm_agent.dcgmEntitiesGetLatestValues(handle, entities, fieldIds, flags)
    majorVersion = fieldValues[0].value.i64 / 1000
    minorVersion = (fieldValues[0].value.i64 - majorVersion * 1000) / 10
    return [majorVersion, minorVersion]

def cuda_visible_devices_required(handle, gpuId):
    # We need to have a cuda_visible_devices value if this GPU has any MIG entities
    hierarchy = dcgm_agent.dcgmGetGpuInstanceHierarchy(handle)
    for i in range(0, hierarchy.count):
        entity = hierarchy.entityList[i]
        if entity.entity.entityGroupId == dcgm_fields.DCGM_FE_GPU_I:
            if entity.parent.entityId == gpuId:
                return True

    return False


def get_cuda_visible_devices_env(handle, gpuId):
    env = {}
    # If we have no MIG entities, then don't bother passing a CUDA_VISIBLE_DEVICES value - it is error prone
    if cuda_visible_devices_required(handle, gpuId):
        if 'REQUESTED_CUDA_VISIBLE_DEVICES' not in os.environ:
            skip_test("This test relies on a requested CUDA_VISIBLE_DEVICES value, but none was found. Please set it in the environment before running")
        else:
            env['CUDA_VISIBLE_DEVICES'] = os.environ['REQUESTED_CUDA_VISIBLE_DEVICES']
    return env

class FilePrivilegesReduced(object):
    def __init__(self, devnode):
        self.devnode = devnode

    def __enter__(self):
        if not self.devnode: # For ease of programming, support case when devnode is None
            return           # See for_all_device_nodes for more context

        with RunAsRoot(reload_driver=False):
            self.old_st_mode = st_mode = os.stat(self.devnode).st_mode

            self.new_st_mode = st_mode & ~utils.stat_everyone_read_write
            logger.debug("setting %s chmod to %s" % (self.devnode, bin(self.new_st_mode)))

            os.chmod(self.devnode, self.new_st_mode)

    def __exit__(self, exception_type, exception, trace):
        if not self.devnode:
            return

        with RunAsRoot(reload_driver=False):
            current_st_mode = os.stat(self.devnode).st_mode
            if current_st_mode != self.new_st_mode:
                logger.warning("Some other entity changed permission of %s from requested %s to %s" %
                        (self.devnode, self.new_st_mode, current_st_mode))
            logger.debug("restoring %s chmod to %s" % (self.devnode, bin(self.old_st_mode)))
            os.chmod(self.devnode, self.old_st_mode) # restore

def run_as_root_and_non_root():
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwds):
            with SubTest("As root", quiet=True):
                RunAsRoot.is_supported(skip_if_not_supported=True)
                with RunAsRoot():
                    fn(*args, **kwds)

            with SubTest("As non-root", quiet=True):
                RunAsNonRoot.is_supported(skip_if_not_supported=True)
                with RunAsNonRoot():
                    fn(*args, **kwds)
        return wrapper
    return decorator

def run_only_as_root():
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwds):
            RunAsRoot.is_supported(skip_if_not_supported=True)
            with RunAsRoot():
                fn(*args, **kwds)
        return wrapper
    return decorator

def run_only_as_non_root():
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwds):
            RunAsNonRoot.is_supported(skip_if_not_supported=True)
            with RunAsNonRoot():
                fn(*args, **kwds)

        return wrapper
    return decorator

def run_only_on_windows():
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwds):
            if utils.is_windows():
                result = fn(*args, **kwds)
            else:
                skip_test("This test is to run only on Windows.")
        return wrapper
    return decorator


def run_only_on_x86():
    """
    Run only on x86 based machines
    """
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwds):
            if utils.platform_identifier in ["Linux_32bit", "Linux_64bit", "Windows_64bit"]:
                result = fn(*args, **kwds)
            else:
                skip_test("This test is to run only on x86 platform")
        return wrapper
    return decorator

def run_only_on_ppc():
    """
    Run only on ppc Platform
    """
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwds):
            if utils.platform_identifier in ["Linux_ppc64le"]:
                result = fn(*args, **kwds)
            else:
                skip_test("This test is to run only on ppc64le platform")
        return wrapper
    return decorator

def run_only_on_linux():
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwds):
            if utils.is_linux():
                result = fn(*args, **kwds)
            else:
                skip_test("This test is to run only on Linux")
        return wrapper
    return decorator

def run_only_on_bare_metal():
    """
    Run only on bare metal systems
    """
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwds):
            if utils.is_bare_metal_system():
                result = fn(*args, **kwds)
            else:
                skip_test("This test is only supported on bare metal systems")
        return wrapper
    return decorator

def run_first():
    """
    Forces get_test_content to move this test at the top of the list.

    Note: can coexist with run_last. Test is just duplicated.

    """
    def decorator(fn):
        fn.run_first = True
        return fn
    return decorator

def run_last():
    """
    Forces get_test_content to move this test at the bottom of the list

    Note: can coexist with run_first. Test is just duplicated.

    """
    def decorator(fn):
        fn.run_last = True
        return fn
    return decorator

def needs_cuda():
    """
    Skips the test on platforms that don't support CUDA (e.g. VMkernel).

    """
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwds):
            if utils.is_cuda_supported_system():
                result = fn(*args, **kwds)
            else:
                skip_test("This test requires CUDA which is not supported on this platform")
        return wrapper
    return decorator

def is_xorg_running():
    if utils.is_windows():
        return False

    try:
        processes = apps.LsofApp("/dev/nvidiactl").get_processes()
    except OSError as xxx_todo_changeme:
        errno.ENOENT = xxx_todo_changeme
        return False

    for (pid, pname) in processes:
        if pname == "Xorg":
            return True
    return False

def is_driver_in_use():
    """
    Returns True if testing2 is the only process keeping the driver loaded.

    Note: doesn't take Persistence Mode into account!
    """
    # !!! Keep in sync with run_only_if_driver_unused decorator !!!
    if utils.is_windows():
        return True

    if is_xorg_running():
        return True

    processes = apps.LsofApp("/dev/nvidiactl").get_processes()
    if processes:
        return True

    return False


def run_only_if_driver_unused():
    """
    Skips the test if driver is in use (e.g. some other application, except for current testing framework)
    is using nvidia driver.

    """
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwds):
            # !!! Keep in sync with is_driver_in_use function !!!
            if utils.is_windows():
                skip_test("Can't run this test when other processes are using the GPU. (This can run only on Linux)")

            if is_xorg_running():
                skip_test("Can't run this test when X server is running.")

            processes = apps.LsofApp("/dev/nvidiactl").get_processes()
            if processes:
                skip_test("Can't run this test when other processes (%s) are using the GPU." % processes)

            result = fn(*args, **kwds)
        return wrapper
    return decorator


class assert_raises(object):
    def __init__(self, expected_exception):
        assert not (expected_exception is None), "expected_exception can't be None"

        self.expected_exception = expected_exception

    def __enter__(self):
        pass

    def __exit__(self, exception_type, exception, trace):
        if isinstance(exception, KeyboardInterrupt):
            return False
        #If we weren't expecting a connection exception and we get one, pass it up the stack rather than the assertion exception
        notConnectedClass = dcgm_structs.dcgmExceptionClass(dcgm_structs.DCGM_ST_CONNECTION_NOT_VALID)
        if (not self.expected_exception == notConnectedClass) and isinstance(exception, notConnectedClass):
           return False

        assert not exception is None, \
            "This code block didn't return ANY exception (expected %s exception)" % self.expected_exception.__name__

        assert isinstance(exception, self.expected_exception), \
                "Expected that this code block will return exception of type %s but it returned exception of type " \
                "%s instead:\n %s" % \
                (
                        self.expected_exception.__name__,
                        exception_type.__name__,
                        " ".join(traceback.format_exception(exception_type, exception, trace))
                )
        return isinstance(exception, self.expected_exception)

def get_test_content():
    '''
    Searches for all modules with name "test*" and all functions with name "test*" in each module.

    Returns list of pairs [(module, test functions in module), ...]

    '''
    # Get all test names
    test_module_files = utils.find_files(os.path.join(utils.script_dir, test_directory), mask = "test*.py", recurse=True)
    test_module_names = [os.path.splitext(os.path.relpath(fname, utils.script_dir))[0].replace(os.path.sep, ".") for fname in test_module_files]

    test_module_names.sort()
    # see help(__import__) for info on __import__ fromlist parameter
    test_modules = [__import__(name,
        fromlist=("non-empty list has a side effect of import loading the module.submodule instead of module"))
        for name in test_module_names]
    def test_functions_in_module(module):
        attributes = dir(module)
        attributes.sort()
        for attr_name in attributes:
            if not attr_name.startswith("test"):
                continue
            attr = getattr(module, attr_name)
            if not inspect.isfunction(attr):
                continue
            if option_parser.options.filter_tests:
                if option_parser.options.filter_tests.search(module.__name__ + "." + attr_name) is None:
                    # Skip tests that don't match provided filter test regex
                    continue
            yield attr
    test_content = [(module, list(test_functions_in_module(module))) for module in test_modules]

    # split into 3 groups (some tests might show in two groups)
    # split into run_first, normal and run_last
    filter_run_first = lambda x: hasattr(x, "run_first") and x.run_first
    filter_run_last = lambda x: hasattr(x, "run_last") and x.run_last
    filter_run_normal = lambda x: not filter_run_first(x) and not filter_run_last(x)

    test_content_first = [(module, list(filter(filter_run_first, test_funcs))) for (module, test_funcs) in test_content]
    test_content_normal = [(module, list(filter(filter_run_normal, test_funcs))) for (module, test_funcs) in test_content]
    test_content_last = [(module, list(filter(filter_run_last, test_funcs))) for (module, test_funcs) in test_content]
    test_content = test_content_first + test_content_normal + test_content_last

    # return modules with at least one test function
    test_content = [x for x in test_content if x[1] != []]
    return test_content

class TestSkipped(Exception):
    pass

def skip_test(reason):
    raise TestSkipped(reason)

def skip_test_notsupported(feature_name):
    raise TestSkipped("Test runs only on devices that don't support %s." % feature_name)

def skip_test_supported(feature_name):
    raise TestSkipped("Test runs only on devices that support %s." % feature_name)

class _RunAsUser(object):
    """
    Switches euid, egid and groups to target_user and later restores the old settings.

    """
    def __init__(self, target_user, reload_driver):
        self._target_user = target_user
        self._reload_driver = reload_driver

        if utils.is_linux():
            ids = utils.get_user_idinfo(target_user)
            self._target_uid = ids.uid
            self._target_gid = ids.gid
            self._orig_uid = None
            self._orig_gid = None
            self._orig_user = None
        else:
            # on non-linux switching user is not supported
            assert (self._target_user == "root") == utils.is_root()


    def __enter__(self):
        if utils.is_linux():
            self._orig_uid = os.geteuid()
            self._orig_gid = os.getegid()
            self._orig_user = utils.get_name_by_uid(self._orig_uid)

            if self._target_user == self._orig_user:
                return # Nothing to do

            logger.debug("Switching current user from %s (uid %d gid %d) to %s (uid %d gid %d)" %
                    (self._orig_user, self._orig_uid, self._orig_gid,
                     self._target_user, self._target_uid, self._target_gid))
            logger.debug("Groups before: %s" % os.getgroups())

            if os.geteuid() == 0:
                # initgroups can be called only while effective user is root
                # before seteuid effective user is root
                os.initgroups(self._target_user, self._target_gid)
                os.setegid(self._target_gid)

            os.seteuid(self._target_uid)

            if os.geteuid() == 0:
                os.initgroups(self._target_user, self._target_gid)
                os.setegid(self._target_gid)

            logger.debug("Groups after: %s" % os.getgroups())


    def __exit__(self, exception_type, exception, trace):
        if utils.is_linux():
            if self._target_user == self._orig_user:
                return # Nothing to do

            logger.debug("Switching back current user from %s (uid %d gid %d) to %s (uid %d gid %d)" %
                    (self._target_user, self._target_uid, self._target_gid,
                    self._orig_user, self._orig_uid, self._orig_gid))
            logger.debug("Groups before: %s" % os.getgroups())

            if os.geteuid() == 0:
                os.initgroups(self._orig_user, self._orig_gid)
                os.setegid(self._orig_gid)

            os.seteuid(self._orig_uid)

            if os.geteuid() == 0:
                os.initgroups(self._orig_user, self._orig_gid)
                os.setegid(self._orig_gid)

            logger.debug("Groups after: %s" % os.getgroups())


class RunAsNonRoot(_RunAsUser):
    """
    Switches euid to option_parser.options.non_root_user.

    """
    def __init__(self, reload_driver=True):
        non_root_user = option_parser.options.non_root_user
        if not non_root_user and utils.is_linux() and not utils.is_root():
            non_root_user = utils.get_name_by_uid(os.getuid())
        super(RunAsNonRoot, self).__init__(non_root_user, reload_driver)

    @classmethod
    def is_supported(cls, skip_if_not_supported=False):
        if not utils.is_root():
            return True # if current user is non-root then running as non-root is supported

        if not utils.is_linux():
            if skip_if_not_supported:
                skip_test("Changing user mid way is only supported on Linux")
            return False

        if not option_parser.options.non_root_user:
            if skip_if_not_supported:
                skip_test("Please run as non-root or as root with --non-root-user flag")
            return False

        return True

class RunAsRoot(_RunAsUser):
    """
    Switches euid to root (possible only if real uid is root) useful e.g. when euid is non-root.

    """
    def __init__(self, reload_driver=True):
        super(RunAsRoot, self).__init__("root", reload_driver)

    @classmethod
    def is_supported(cls, skip_if_not_supported=False):
        if utils.is_root():
            return True # if current user is root then running as root is supported

        if not utils.is_linux():
            if skip_if_not_supported:
                skip_test("Changing user mid way is only supported on Linux")
            return False

        if not utils.is_real_user_root():
            if skip_if_not_supported:
                skip_test("Run as root user.")
            return False

        return True

def tryRunAsNonRoot():
    if RunAsNonRoot.is_supported():
        return RunAsNonRoot()
    return _DoNothingBlock()

def tryRunAsRoot():
    if RunAsRoot.is_supported():
        return RunAsRoot()
    return _DoNothingBlock()

class SubTest(object):
    _stack = [None]
    _log = []
    SUCCESS,SKIPPED,FAILED,FAILURE_LOGGED,NOT_CONNECTED = ("SUCCESS", "SKIPPED", "FAILED", "FAILURE_LOGGED", "NOT_CONNECTED")
    ResultDetailsRaw = namedtuple("ResultDetailsRaw", "exception_type, exception, trace")

    def __init__(self, name, quiet=False, supress_errors=True, disconnect_is_failure=True):
        """
        Set quiet to True if you want the test to be removed from the logs if it succeeded.
        Useful when test is minor and you don't want to clobber the output with minor tests.

        """
        self.name = name
        self.result = None
        self.result_details = None
        self.result_details_raw = None
        self.parent = None
        self.depth = None
        self.subtests = []
        self.quiet = quiet
        self.stats = dict([(SubTest.SUCCESS, 0), (SubTest.SKIPPED, 0), (SubTest.FAILED, 0), (SubTest.FAILURE_LOGGED, 0), (SubTest.NOT_CONNECTED, 0)])
        self.supress_errors = supress_errors
        self.disconnect_is_failure = disconnect_is_failure

    def __enter__(self):
        self.parent = SubTest._stack[-1]
        self.depth = len(SubTest._stack)
        SubTest._stack.append(self)
        SubTest._log.append(self)
        if self.parent:
            self.parent.subtests.append(self)

        # pylint: disable=undefined-variable
        progress_printer.subtest_start(self)

        # returns the current subtest
        return self


    def __exit__(self, exception_type, exception, trace):
        SubTest._stack.pop()
        for subtest in self.subtests:
            self.stats[SubTest.SUCCESS] += subtest.stats[SubTest.SUCCESS]
            self.stats[SubTest.SKIPPED] += subtest.stats[SubTest.SKIPPED]
            self.stats[SubTest.FAILED] += subtest.stats[SubTest.FAILED]
            self.stats[SubTest.FAILURE_LOGGED] += subtest.stats[SubTest.FAILURE_LOGGED]
        if exception is None:
            self.result = SubTest.SUCCESS
        elif isinstance(exception, TestSkipped):
            self.result = SubTest.SKIPPED
        elif isinstance(exception, KeyboardInterrupt):
            self.result = SubTest.SKIPPED
        elif isinstance(exception, dcgm_structs.dcgmExceptionClass(dcgm_structs.DCGM_ST_CONNECTION_NOT_VALID)):
            if self.disconnect_is_failure:
                self.result = SubTest.FAILED
            else:
                self.result = SubTest.NOT_CONNECTED
        elif reRunning == True:
            self.result = SubTest.FAILURE_LOGGED
        else:
            self.result = SubTest.FAILED

        self.result_details = " ".join(traceback.format_exception(exception_type, exception, trace))
        self.result_details_raw = SubTest.ResultDetailsRaw(exception_type, exception, trace)
        self.stats[self.result] += 1
        if self.quiet and self.result == SubTest.SUCCESS and self.subtests == []:
            SubTest._log.remove(self)
            if self.parent:
                self.parent.subtests.remove(self)

        # pylint: disable=undefined-variable
        progress_printer.subtest_finish(self)

        # terminate on KeyboardInterrupt exceptions
        if isinstance(exception, KeyboardInterrupt):
            return False

        if self.result == SubTest.FAILED and option_parser.options.break_at_failure:
            try:
                import debugging
                debugging.break_after_exception()
            except ImportError:
                logger.warning("Unable to find Python Debugging Module - \"-b\" option is unavailable")

        return self.supress_errors

    def __str__(self):
        # traverse the entire path from node to parent
        # to retrieve all the names of the subtests
        path_to_parent = [self]
        while path_to_parent[-1].parent:
            path_to_parent.append(path_to_parent[-1].parent)
        path_to_parent.reverse()
        return "Test %s - %s" % ("::".join([s.name for s in path_to_parent]), self.result)

    @staticmethod
    def get_all_subtests():
        return SubTest._log

class _IgnoreExceptions(object):
    def __init__(self, dontignore=None):
        """
        dontignore = optional argument, list of exception types that shouldn't be ignored

        """
        self.dontignore = dontignore

    def __enter__(self):
        pass

    def __exit__(self, exception_type, exception, trace):
        if isinstance(exception, KeyboardInterrupt):
            return False
        if self.dontignore:
            for ex in self.dontignore:
                if isinstance(exception, ex):
                    return False
        return True

class ExceptionAsWarning(object):
    """
    Block wrapper used to "mark" known issues as warnings.

    As reason pass a string with explanation (e.g. describe that issue is tracked in a bug X).

    """
    def __init__(self, reason):
        self.reason = reason

    def __enter__(self):
        pass

    def __exit__(self, exception_type, exception, trace):
        if isinstance(exception, KeyboardInterrupt):
            return False
        if isinstance(exception, TestSkipped):
            return False

        if exception:
            logger.warning("Exception treated as warning: %s\nOriginal issue: %s" % (self.reason, str(exception)))
            logger.debug(" ".join(traceback.format_exception(exception_type, exception, trace)))

        return True

class _DoNothingBlock(object):
    """
    Class that can be used in "with" statement that has completely NO effect.
    Used as a fall back if some other class is not supported.

    """
    def __enter__(self):
        pass
    def __exit__(self, exception_type, exception, trace):
        pass

class RestoreDefaultEnvironment(object):
    """
    Class that should be used in "with" clause. It stores some values before the block executes and
    then restores the state to know predefined state after the block finishes (even if block returned with exceptions)

    """

    def __init__(self):
        pass

    def __enter__(self):
        return

    def __exit__(self, exception_type, exception, trace):
        logger.debug("Restoring default environment - START")

        # Restore env variables
        RestoreDefaultEnvironment.restore_env()

        # Turn off all processes started by the test
        apps.AppRunner.clean_all()

        #    with _IgnoreExceptions():

        logger.debug("Restoring default environment - END")

    @classmethod
    def restore(cls):
        """
        Restores environmental variables and NVML state to predefined default state.
        e.g.
            all device settings pending == current
            persistence mode on

        """
        with RestoreDefaultEnvironment():
            pass

    @classmethod
    def restore_dev_node_permissions(cls):
        if not utils.is_linux():
            return # nothing to do

        # make sure that current user can access /dev/nvidiactl, /dev/nvidia-uvm and the [0-9] nodes
        with tryRunAsRoot():
            for fname in utils.find_files("/dev/", "nvidia*"):
                st_mode = os.stat(fname).st_mode
                if st_mode & utils.stat_everyone_read_write != utils.stat_everyone_read_write:
                    try:
                        logger.warning("Device node %s permissions (%s) are not set as read/write for everyone (%s)."
                                       " Framework will try to fix that" % (fname, bin(st_mode), bin(utils.stat_everyone_read_write)))
                        os.chmod(fname, st_mode | utils.stat_everyone_read_write)
                    except OSError:
                        logger.warning("Failed to change permission of %s. This might cause some failures down the line" % fname)

    @classmethod
    def restore_env(cls):
        unset_envs = ['CUDA_VISIBLE_DEVICES', ]
        for env in unset_envs:
            if os.getenv(env) is not None:
                if env == 'CUDA_VISIBLE_DEVICES':
                    # Save this for use in tests
                    os.environ['REQUESTED_CUDA_VISIBLE_DEVICES'] = os.environ['CUDA_VISIBLE_DEVICES']
                    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
                logger.warning("%s env is set (value: %s) and is about to be unset." % (env, os.getenv(env)))
                os.unsetenv(env)
                del os.environ[env]

        warn_envs = []
        for env in warn_envs:
            if os.getenv(env) is not None:
                logger.warning("%s is set (value: %s)" % (env, os.getenv(env)))

        return True

knownWordDict = None
def _loadWordList():
    global knownWordDict
    if knownWordDict is None:
        with open('./data/wordlist', 'r') as f:
            knownWordDict = dict((s.strip().lower(), True) for s in f.readlines())

def check_spelling(text):
    _loadWordList()
    global knownWordDict
    # split into words, remove special characters
    text = text.translate(None, '0123456789%*$[]()<>\"\'|')
    tokens = re.split(' |\t|\n|-|_|/|:|\.|=|\?|\!|,', text)
    words = [ s.strip().lower() for s in tokens ]
    unknownWords = []
    for word in words:
        if word not in knownWordDict:
            unknownWords.append(word)
    assert 0 == len(unknownWords), "Unknown words: " + str(unknownWords)

def _busIdRemoveDomain(busId):
    return ":".join(string.split(busId, ':')[-2:])


class RunCudaAppInBackGround:
    """
    This class is used as part of "with" clause. It creates a CUDA app leading to GPU utilization and
    memory usage. Starts the app for the specified time.

    Usage:
        with RunCudaAppInBackGround(busId, timeInMilliSeconds):
            # Code to run when the app is running
        # Cuda app is terminated
    """

    def __init__(self, busId, timeToRun):
        '''
        Initializes cuda context
        '''

        #self.busId = _busIdRemoveDomain(busId)
        self.busId = busId
        self.timeToRun = timeToRun
        #self.app = apps.CudaCtxCreateAdvancedApp(["--ctxCreate", self.busId, "--busyGpu", self.busId, timeToRun, "--getchar"])
        self.app = apps.CudaCtxCreateAdvancedApp(["--ctxCreate", self.busId, "--busyGpu", self.busId, self.timeToRun])

    def __enter__(self):
        '''
        Runs the CUDA app for the specified amount of time
        '''

        ## Start the app and change the default timeout (in secs)
        self.app.start(timeout=apps.default_timeout + float(self.timeToRun)/1000.0)
        self.app.stdout_readtillmatch(lambda s: s.find("Calling cuInit") != -1)

    def __exit__(self, exception_type, exception, trace):
        '''
        Wait for completion of CUDA app
        '''
        self.app.wait()
        self.app.validate()

"""
Helper functions for setting/getting connection mode. These are needed for other helpers to know
if we are in embedded/remote mode
"""
DCGM_CONNECT_MODE_UNKNOWN  = 0 #Not connected
DCGM_CONNECT_MODE_EMBEDDED = 1 #Connected to an embedded host engine
DCGM_CONNECT_MODE_REMOTE   = 2 #Connected to a remote host engine. Note that this doesn't guarantee a tcp connection, just that the HE process is running

def set_connect_mode(connectMode):
    global dcgm_connect_mode
    dcgm_connect_mode = connectMode

def get_connect_mode():
    global dcgm_connect_mode
    return dcgm_connect_mode

class RunEmbeddedHostEngine:
    """
    This class is used as part of a "with" clause to start and stop an embedded host engine
    """
    def __init__(self, opmode=dcgm_structs.DCGM_OPERATION_MODE_AUTO, startTcpServer=False):
        self.hostEngineStarted = False
        self.opmode = opmode
        self.handle = None
        self.startTcpServer = startTcpServer

        if option_parser.options.use_running_hostengine:
            skip_test("Skipping embedded test due to option --use-running-hostengine")

    def __enter__(self):
        dcgm_agent.dcgmInit() #Will throw an exception on error
        self.handle = dcgm_agent.dcgmStartEmbedded(self.opmode)
        logger.info("embedded host engine started")
        self.hostEngineStarted = True
        if self.startTcpServer:
            dcgm_agent_internal.dcgmServerRun(5555, '127.0.0.1', 1)
            self.handle = dcgm_agent.dcgmConnect('127.0.0.1:5555')
            logger.info("Started TCP server")
        set_connect_mode(DCGM_CONNECT_MODE_EMBEDDED)
        return self.handle

    def __exit__(self, exception_type, exception, trace):
        if self.hostEngineStarted:
            logger.info("Stopping embedded host engine")
            try:
                dcgm_agent.dcgmShutdown()
            except dcgm_structs.dcgmExceptionClass(dcgm_structs.DCGM_ST_UNINITIALIZED):
                logger.info("embedded host engine was already stopped")
            self.hostEngineStarted = False
        else:
            logger.info("Skipping dcgmEngineShutdown. Host engine was not running")
        set_connect_mode(DCGM_CONNECT_MODE_UNKNOWN)

def run_with_embedded_host_engine(opmode=dcgm_structs.DCGM_OPERATION_MODE_AUTO, startTcpServer=False):
    """
    Run this test with an embedded host engine. This will start the host engine before the test
    and stop the host engine after the test
    """
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwds):
            with RunEmbeddedHostEngine(opmode=opmode, startTcpServer=startTcpServer) as handle:
                kwds['handle'] = handle
                fn(*args, **kwds)
            return
        return wrapper
    return decorator

class RunStandaloneHostEngine:
    """
    This class is used as part of a "with" clause to start and stop an standalone host engine process
    """

    _nvswitches_detected = None

    def __init__(self, timeout=15, heArgs = None, profile_dir=None): #DCGM_HE_PORT_NUMBER
        self.hostEngineStarted = False
        self.timeout = timeout

        if option_parser.options.use_running_hostengine:
            self.nvhost_engine = None
        elif heArgs is None:
            self.nvhost_engine = apps.NvHostEngineApp(profile_dir=profile_dir)
        else:
            self.nvhost_engine = apps.NvHostEngineApp(heArgs, profile_dir=profile_dir)

    def __enter__(self):
        if self.nvhost_engine is not None:
            self.nvhost_engine.start(self.timeout)
            assert self.nvhost_engine.getpid() != None, "start hostengine failed"
            logger.info("standalone host engine started with pid %d" % self.nvhost_engine.getpid())
            self.hostEngineStarted = True
            set_connect_mode(DCGM_CONNECT_MODE_REMOTE)
            return self.nvhost_engine

    def __exit__(self, exception_type, exception, trace):
        if self.nvhost_engine is not None:
            if self.hostEngineStarted:
                if self.nvhost_engine.poll() is None:
                    logger.info("Stopping standalone host engine")
                    self.nvhost_engine.terminate()
                self.nvhost_engine.validate()
                self.hostEngineStarted = False
            else:
                logger.info("Skipping standalone host engine terminate. Host engine was not running")
        set_connect_mode(DCGM_CONNECT_MODE_UNKNOWN)

def run_with_standalone_host_engine(timeout=15, heArgs=None, passAppAsArg=False):
    """
    Run this test with the standalone host engine.  This will start the host engine process before the test
    and stop the host engine process after the test
    """
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwds):
            with RunStandaloneHostEngine(timeout, heArgs, profile_dir=fn.__name__) as hostengineApp:
                # pass the hostengine app to the test function in case they want to interact with it
                if passAppAsArg:
                    kwds['hostengineApp'] = hostengineApp
                fn(*args, **kwds)
            return
        return wrapper
    return decorator

class RunClientInitShutdown:
    """
    This class is used as part of a "with" clause to initialize and shutdown the client API
    """
    def __init__(self, pIpAddr = "127.0.0.1", persistAfterDisconnect=False):
        self.clientAPIStarted = False
        self.dcgm_handle = None
        self.ipAddress = pIpAddr
        self.persistAfterDisconnect = persistAfterDisconnect

    def __enter__(self):
        connectParams = dcgm_structs.c_dcgmConnectV2Params_v1()
        if self.persistAfterDisconnect:
            connectParams.persistAfterDisconnect = 1
        else:
            connectParams.persistAfterDisconnect = 0

        dcgm_agent.dcgmInit()
        for attempt in range(3):
            try:
                self.dcgm_handle = dcgm_agent.dcgmConnect_v2(self.ipAddress, connectParams)
            except dcgm_structs.dcgmExceptionClass(dcgm_structs.DCGM_ST_CONNECTION_NOT_VALID):
                pass
            else:
                break

        if not self.dcgm_handle:
            raise Exception('failed connection to dcgm hostengine')

        self.clientAPIStarted = True
        return self.dcgm_handle

    def __exit__(self, exception_type, exception, trace):
        if self.clientAPIStarted:
            try:
                dcgm_agent.dcgmShutdown()
            except dcgmExceptionClass(DCGM_ST_INIT_ERROR):
                logger.info("Client API is already shut down")
            self.clientAPIStarted = False

def run_with_initialized_client(ipAddress = "127.0.0.1"):
    """
    Run test having called client init and then call shutdown on exit
    """
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwds):
            with RunClientInitShutdown(ipAddress) as handle:
                kwds['handle'] = handle
                fn(*args, **kwds)
            return
        return wrapper
    return decorator

def get_live_gpu_ids(handle):
    """
    Get the gpu ids of live GPUs on the system. This works in embedded or remote mode
    """
    gpuIdList = []
    dcgmHandle = pydcgm.DcgmHandle(handle=handle)
    dcgmSystem = dcgmHandle.GetSystem()
    gpuIdList = dcgmSystem.discovery.GetAllSupportedGpuIds()
    return gpuIdList

def get_live_gpu_count(handle):
    return len(get_live_gpu_ids(handle))

def run_only_with_live_gpus():
    """
    Only run this test if live gpus are present on the system
    """
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwds):
            if 'handle' in kwds:
                gpuIds = get_live_gpu_ids(kwds['handle'])
            else:
                raise Exception("Not connected to remote or embedded host engine. Use appropriate decorator")

            if len(gpuIds) < 1:
                logger.warning("Skipping test that requires live GPUs. None were found")
            else:
                kwds['gpuIds'] = gpuIds
                fn(*args, **kwds)
            return
        return wrapper
    return decorator

def run_with_injection_gpus(gpuCount=1):
    """
    Run this test with injection-only GPUs x gpuCount
    """
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwds):
            if 'handle' not in kwds:
                raise Exception("Not connected to remote or embedded host engine. Use approriate decorator")

            numGpus = len(dcgm_agent.dcgmGetAllDevices(kwds['handle']))
            if numGpus + gpuCount >= dcgm_structs.DCGM_MAX_NUM_DEVICES:
                skip_test("unable to add fake Gpu with more than %d gpus" % dcgm_structs.DCGM_MAX_NUM_DEVICES)
            cfe = dcgm_structs_internal.c_dcgmCreateFakeEntities_v2()
            cfe.numToCreate = gpuCount
            for i in range(0, gpuCount):
                cfe.entityList[i].entity.entityGroupId = dcgm_fields.DCGM_FE_GPU
            updated = dcgm_agent_internal.dcgmCreateFakeEntities(kwds['handle'], cfe)
            gpuIds = []
            for i in range(0, updated.numToCreate):
                gpuIds.append(updated.entityList[i].entity.entityId)
            kwds['gpuIds'] = gpuIds
            fn(*args, **kwds)
            return
        return wrapper
    return decorator

def run_with_injection_gpu_instances(totalInstances=1):
    """
    Run this test with injection-only <totalInstances> GPU instances 

    This does not inject hierarchy now but should in the future
    """
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwds):
            if 'handle' not in kwds:
                raise Exception("Not connected to remote or embedded host engine. Use approriate decorator")

            if 'gpuIds' not in kwds:
                raise Exception("Injected instances require GPU IDs")

            gpuIds = kwds['gpuIds']

            numGpus = len(dcgm_agent.dcgmGetAllDevices(kwds['handle']))
            cfe = dcgm_structs_internal.c_dcgmCreateFakeEntities_v2()
            cfe.numToCreate = totalInstances
            for i in range(0, totalInstances):
                cfe.entityList[i].entity.entityGroupId = dcgm_fields.DCGM_FE_GPU_I
                # Set the parent to the first GPu in the test
                cfe.entityList[i].parent.entityGroupId = dcgm_fields.DCGM_FE_GPU
                cfe.entityList[i].parent.entityId = gpuIds[0]
            updated = dcgm_agent_internal.dcgmCreateFakeEntities(kwds['handle'], cfe)
            instanceIds = []
            for i in range(0, updated.numToCreate):
                instanceIds.append(updated.entityList[i].entity.entityId)
            kwds['instanceIds'] = instanceIds
            fn(*args, **kwds)
            return
        return wrapper
    return decorator

def run_with_injection_gpu_compute_instances(totalCIs=1):
    """
    Run this test with <totalCIs> compute instances

    This does not inject hierarchy now but should in the future
    """
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwds):
            if 'handle' not in kwds:
                raise Exception("Not connected to remote or embedded host engine. Use approriate decorator")

            if 'instanceIds' not in kwds:
                raise Exception("Injected CIs require instance IDs")
            instanceIds = kwds['instanceIds']

            numGpus = len(dcgm_agent.dcgmGetAllDevices(kwds['handle']))
            cfe = dcgm_structs_internal.c_dcgmCreateFakeEntities_v2()
            cfe.numToCreate = totalCIs
            instanceIndex = 0
            numInstances = len(instanceIds)
            for i in range(0, totalCIs):
                cfe.entityList[i].entity.entityGroupId = dcgm_fields.DCGM_FE_GPU_CI
                # Set the parents so that this compute instance has a parent that is part of the test
                cfe.entityList[i].parent.entityGroupId = dcgm_fields.DCGM_FE_GPU_I
                cfe.entityList[i].parent.entityId = instanceIds[instanceIndex]

                #Make sure we increment the instanceIndex correctly: same if there's just one and wrap the increment
                #if there are more compute instances than instances
                if numInstances > 1:
                    instanceIndex = instanceIndex + 1
                    if numInstances < totalCIs and instanceIndex == numInstances:
                        instanceIndex = 0
            updated = dcgm_agent_internal.dcgmCreateFakeEntities(kwds['handle'], cfe)
            ciIds = []
            for i in range(0, updated.numToCreate):
                ciIds.append(updated.entityList[i].entity.entityId)
            kwds['ciIds'] = ciIds
            fn(*args, **kwds)
            return
        return wrapper
    return decorator

# Long timeout by default, but the app is terminated once the test has concluded
def run_with_cuda_app(timeout=10000):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwds):
            if 'handle' not in kwds:
                raise Exception("Not connected to remote or embedded host engine. Use appropriate decorator")

            if 'gpuIds' not in kwds or len(kwds['gpuIds']) < 1:
                raise Exception("cuda_app requires GPU ID")

            handle = kwds['handle']
            gpuId = kwds['gpuIds'][0]
            updateFreq = 1000000
            maxKeepAge = 3600.0 #one hour
            maxKeepEntries = 0 #no limit
            fieldId = dcgm_fields.DCGM_FI_DEV_PCI_BUSID
            dcgm_agent_internal.dcgmWatchFieldValue(handle, gpuId, fieldId, updateFreq, maxKeepAge, maxKeepEntries)
            dcgm_agent.dcgmUpdateAllFields(handle, 1)
            values = dcgm_agent_internal.dcgmGetLatestValuesForFields(handle, gpuId, [fieldId,])
            busId = values[0].value.str

            params = [ "--ctxCreate", busId,
                       "--busyGpu", busId, str(timeout),
                       "--ctxDestroy", busId ]

            app = apps.CudaCtxCreateAdvancedApp(params, env=get_cuda_visible_devices_env(handle, gpuId))
            app.start(timeout * 2)
            kwds['cudaApp'] = app
            fn(*args, **kwds)
            app.terminate()
            return
        return wrapper
    return decorator

def get_live_nvswitch_ids(handle):
    """
    Get the entityIds of live NvSwitches on the system. This works in embedded or remote mode
    """
    entityIdList = []
    try:
        dcgmHandle = pydcgm.DcgmHandle(handle=handle)
        dcgmSystem = dcgmHandle.GetSystem()
        entityIdList = dcgmSystem.discovery.GetEntityGroupEntities(dcgm_fields.DCGM_FE_SWITCH, True)
    except dcgm_structs.DCGMError as e:
        raise Exception("Not connected to remote or embedded host engine. Use appropriate decorator")
    return entityIdList

def get_live_nvswitch_count(handle):
    return len(get_live_nvswitch_ids(handle))

def run_only_with_live_nvswitches():
    """
    Only run this test if live nvswitches are present on the system
    """
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwds):
            if 'handle' in kwds:
                entityIdList = get_live_nvswitch_ids(kwds['handle'])
            else:
                raise Exception("Not connected to remote or embedded host engine. Use appropriate decorator")

            if len(entityIdList) < 1:
                logger.warning("Skipping test that requires live GPUs. None were found")
            else:
                kwds['switchIds'] = entityIdList
                fn(*args, **kwds)
            return
        return wrapper
    return decorator

def run_with_injection_nvswitches(switchCount=1):
    """
    Run this test with injection-only NvSwitches x switchCount
    """
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwds):
            if 'handle' not in kwds:
                raise Exception("Not connected to remote or embedded host engine. Use approriate decorator")

            numActiveSwitches = len(dcgm_agent.dcgmGetEntityGroupEntities(kwds['handle'], dcgm_fields.DCGM_FE_SWITCH, 0))
            if numActiveSwitches + switchCount >= dcgm_structs.DCGM_MAX_NUM_SWITCHES:
                skip_test("unable to add fake NvSwitch with more than %d NvSwitches" % dcgm_structs.DCGM_MAX_NUM_SWITCHES)
            cfe = dcgm_structs_internal.c_dcgmCreateFakeEntities_v2()
            cfe.numToCreate = switchCount
            for i in range(0, switchCount):
                cfe.entityList[i].entity.entityGroupId = dcgm_fields.DCGM_FE_SWITCH
            updated = dcgm_agent_internal.dcgmCreateFakeEntities(kwds['handle'], cfe)
            switchIds = []
            for i in range(0, updated.numToCreate):
                switchIds.append(updated.entityList[i].entity.entityId)
            kwds['switchIds'] = switchIds
            fn(*args, **kwds)
            return
        return wrapper
    return decorator

def run_with_introspection_enabled(runIntervalMs=10):
    """
    Run this test with metadata gathering enabled ("Metadata" API calls can be made).
    The run interval for the metadata manager is artificially fast since this helps in testing.
    """
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            if 'handle' not in kwargs:
                raise Exception("Not connected to remote or embedded host engine. Use approriate decorator")
            handle = pydcgm.DcgmHandle(handle=kwargs['handle'])
            system = handle.GetSystem()

            system.introspect.state.toggle(dcgm_structs.DCGM_INTROSPECT_STATE.ENABLED)
            dcgm_agent_internal.dcgmMetadataStateSetRunInterval(kwargs['handle'], runIntervalMs)
            return fn(*args, **kwargs)
        return wrapper
    return decorator

def watch_all_fields(handle,
                     gpuIds,
                     updateFreq=1000, # 1ms
                     maxKeepAge=86400.0,
                     maxKeepEntries=1000,
                     startTimestamp=0):
    '''
    Watch every field in DCGM and return a list of the fields that are watched.
    This also calls to make sure that the watched fields are updated at least once
    before returning.
    '''
    watchedFields = set()

    mig_enabled = is_mig_mode_enabled()

    for gpuId in gpuIds:
        for fieldId in range(1, dcgm_fields.DCGM_FI_MAX_FIELDS):
            # Accounting cannot be enabled for MIG mode: CUDANVML-153
            if mig_enabled and fieldId == dcgm_fields.DCGM_FI_DEV_ACCOUNTING_DATA:
                continue

            # can't tell ahead of time which field Ids are valid from the python API so we must try/except watching
            try:
                dcgm_agent_internal.dcgmWatchFieldValue(handle,
                                                        gpuId=gpuId,
                                                        fieldId=fieldId,
                                                        updateFreq=updateFreq,
                                                        maxKeepAge=maxKeepAge,
                                                        maxKeepEntries=maxKeepEntries)
            except dcgm_structs.dcgmExceptionClass(dcgm_structs.DCGM_ST_REQUIRES_ROOT):
                pass
            except dcgm_structs.dcgmExceptionClass(dcgm_structs.DCGM_ST_BADPARAM):
                pass
            else:
                watchedFields.add(fieldId)

    dcgm_agent.dcgmUpdateAllFields(handle, True)

    return watchedFields

def set_logging_state(enableLogging):
    '''
    Helper function to enable or disable logging. Call restore_logging_state() to
    undo this call
    '''
    global noLogging, noLoggingBackup

    noLoggingDesired = not enableLogging

    #is our logging state already what we wnat?
    if noLoggingDesired == noLogging:
        return

    noLogging = noLoggingDesired
    logger.setup_environment()

def restore_logging_state():
    #Restore the state of logging to what it was before set_logging_state()
    global noLogging, noLoggingBackup

    if noLogging == noLoggingBackup:
        return

    noLogging = noLoggingBackup
    logger.setup_environment()

def run_subtest(subtestFn, *args, **kwargs):
    #List that contains failings test to re-run with logging enabled
    global noLogging
    global reRunning

    #Work around a race condition where the test framework can't connect to
    #the host engine right away. See bug 200417787 for details.
    maxDisconnectedRetries = 3

    for retryCount in range(maxDisconnectedRetries+1):
        if retryCount > 0:
            logger.info("Retrying test %s time %d/%d due to not being connected to the host engine. War for bug 200417787" %
                        (subtestFn.__name__, retryCount, maxDisconnectedRetries))

        disconnect_is_failure = False
        if retryCount == maxDisconnectedRetries:
            disconnect_is_failure = True #Fail if disconnected on the last retry
        with SubTest("%s" % (subtestFn.__name__), disconnect_is_failure=disconnect_is_failure) as subtest:
            subtestFn(*args, **kwargs)

        if subtest.result != SubTest.NOT_CONNECTED:
            break #Passed/failed for another reason. Break out of the loop

    if subtest.result == SubTest.FAILED:
        #Running failing tests with logging enabled
        set_logging_state(True)
        reRunning = True

        logger.warning("Re-running failing test \"%s\" with logging enabled" % subtest.name)
        with SubTest("%s" % (subtestFn.__name__)) as subtest:
            subtestFn(*args, **kwargs)

        restore_logging_state()
        reRunning = False


def group_gpu_ids_by_sku(handle, gpuIds):
    '''
    Return a list of lists where the 2nd level list is each gpuId that is the same sku as each other

    Example [[gpu0, gpu1], [gpu2, gpu3]]
    In the above example, gpu0 and gpu1 are the same sku, and gpu2 and gpu3 are the same sku
    '''
    skuGpuLists = {}

    for gpuId in gpuIds:
        deviceAttrib = dcgm_agent.dcgmGetDeviceAttributes(handle, gpuId)
        pciDeviceId = deviceAttrib.identifiers.pciDeviceId

        if pciDeviceId in skuGpuLists:
            skuGpuLists[pciDeviceId].append(gpuId)
        else:
            skuGpuLists[pciDeviceId] = [gpuId, ]

    retList = []
    for k in list(skuGpuLists.keys()):
        retList.append(skuGpuLists[k])

    #logger.info("skuGpuLists: %s, retList %s" % (str(skuGpuLists), str(retList)))
    return retList

def for_all_same_sku_gpus():
    '''
    Run a test multiple times, passing a list of gpuIds that are the same SKU each time

    This decorator must come after a decorator that provides a list of gpuIds like run_only_with_live_gpus
    '''
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwds):
            gpuGroupList = group_gpu_ids_by_sku(kwds['handle'], kwds['gpuIds'])
            for i, gpuIdList in enumerate(gpuGroupList):
                with SubTest("GPU group %d. gpuIds: %s" % (i, str(gpuIdList))):
                    kwds2 = kwds
                    kwds2['gpuIds'] = gpuIdList
                    fn(*args, **kwds2)
            return

        return wrapper
    return decorator

def set_max_power_limit(handle, gpuIds):
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetEmptyGroup("test1")

    for gpuId in gpuIds:

        ## Add first GPU to the group
        groupObj.AddGpu(gpuId)

        ## Get Min and Max Power limit on the group
        attributes = systemObj.discovery.GetGpuAttributes(gpuId)

        ## Verify that power is supported on the GPUs in the group
        if dcgmvalue.DCGM_INT32_IS_BLANK(attributes.powerLimits.maxPowerLimit):
            skip_test("Needs Power limit to be supported on the GPU")

        ##Get the max Power Limit for the GPU
        maxPowerLimit = attributes.powerLimits.maxPowerLimit

        config_values = dcgm_structs.c_dcgmDeviceConfig_v1()
        config_values.mEccMode = dcgmvalue.DCGM_INT32_BLANK
        config_values.mPerfState.syncBoost = dcgmvalue.DCGM_INT32_BLANK
        config_values.mPerfState.targetClocks.memClock =  dcgmvalue.DCGM_INT32_BLANK
        config_values.mPerfState.targetClocks.smClock = dcgmvalue.DCGM_INT32_BLANK
        config_values.mComputeMode = dcgmvalue.DCGM_INT32_BLANK
        config_values.mPowerLimit.type = dcgm_structs.DCGM_CONFIG_POWER_CAP_INDIVIDUAL
        config_values.mPowerLimit.val = maxPowerLimit

        ##Set the max Power Limit for the group
        groupObj.config.Set(config_values)

        ##Remove the GPU from the group
        groupObj.RemoveGpu(gpuId)

    groupObj.Delete()

def run_with_max_power_limit_set():
    '''
    Sets the power limit of all the GPUs in the list to the max Power Limit.
    '''
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwds):
            set_max_power_limit(kwds['handle'], kwds['gpuIds'])
            fn(*args, **kwds)
            return

        return wrapper
    return decorator

def log_gpu_information(handle):
    '''
    Log information about the GPUs that DCGM is going to run against

    Returns: Number of DCGM-supported GPUs in the system
    '''
    dcgmHandle = pydcgm.DcgmHandle(handle=handle)
    dcgmSystem = dcgmHandle.GetSystem()
    allGpuIds = dcgmSystem.discovery.GetAllGpuIds()
    allDcgmGpuIds = dcgmSystem.discovery.GetAllSupportedGpuIds()
    logger.info("All GPU IDs: %s" % str(allGpuIds))
    logger.info("DCGM-Supported GPU IDs: %s" % str(allDcgmGpuIds))
    logger.info("GPU Info:")
    for gpuId in allGpuIds:
        gpuAttrib = dcgmSystem.discovery.GetGpuAttributes(gpuId)
        logger.info("gpuId %d, name %s, pciBusId %s" % (gpuId, gpuAttrib.identifiers.deviceName, gpuAttrib.identifiers.pciBusId))

    log_instance_information(dcgmHandle.handle)

    return len(allDcgmGpuIds)

def log_instance_information(handle):
    hierarchy = dcgm_agent.dcgmGetGpuInstanceHierarchy(handle)

    # See Query.cpp::TopologicalSort for sorting details
    sorted(hierarchy.entityList, key=lambda id: (id.info.nvmlGpuIndex, id.info.nvmlInstanceId, id.entity.entityGroupId, id.info.nvmlComputeInstanceId))
    for i in range(0, hierarchy.count):
        entity = hierarchy.entityList[i];
        entityId = entity.entity.entityId;
        if entity.entity.entityGroupId == dcgm_fields.DCGM_FE_GPU_I:
            logger.info("GPU[%d] GI[%d] entityId %d" % (entity.info.nvmlGpuIndex, entity.info.nvmlInstanceId, entityId))
        if entity.entity.entityGroupId == dcgm_fields.DCGM_FE_GPU_CI:
            logger.info("GPU[%d] GI[%d] CI[%d] entityId %d" % (entity.info.nvmlGpuIndex, entity.info.nvmlInstanceId, entity.info.nvmlComputeInstanceId, entityId))


def are_all_gpus_dcgm_supported(handle=None):
    # type: (pydcgm.DcgmHandle) -> (bool, list[int])
    """
    Determines if there are DCGM Supported GPUs
    :param handle: DCGM handle or None
    :return: Tuple of bool and list of ids. If all GPUs are supported then result is (True, [list of GPU ids]),
             otherwise that is (False, None)
    """
    dcgmHandle = pydcgm.DcgmHandle(handle=handle)
    dcgmSystem = dcgmHandle.GetSystem()
    allGpuIds = dcgmSystem.discovery.GetAllGpuIds()
    allDcgmGpuIds = dcgmSystem.discovery.GetAllSupportedGpuIds()

    if allGpuIds != allDcgmGpuIds:
        return False, None
    else:
        return True, allDcgmGpuIds


def run_only_with_all_supported_gpus():
    """
    This decorator skips a test if allGpus != supportedGpus.
    This decorator provides gpuIds list of live GPUs to the wrapped function
    """
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwds):
            (all_gpus_supported, gpu_ids) = are_all_gpus_dcgm_supported(kwds.get('handle', None))
            if not all_gpus_supported:
                skip_test("Unsupported GPU(s) detected, skipping test")
            else:
                if len(gpu_ids) < 1:
                    logger.warning("Skipping test that requires live GPUs. None were found")
                else:
                    kwds['gpuIds'] = gpu_ids
                    fn(*args, **kwds)
                return
        return wrapper
    return decorator


def get_device_names(gpu_ids, handle=None):
    dcgm_handle = pydcgm.DcgmHandle(handle=handle)
    dcgm_system = dcgm_handle.GetSystem()
    for gpuId in gpu_ids:
        attributes = dcgm_system.discovery.GetGpuAttributes(gpuId)
        yield (str(attributes.identifiers.deviceName).lower(), gpuId)


def skip_blacklisted_gpus(blacklist=None):
    """
    This decorator gets gpuIds list and excludes GPUs which names are blacklisted
    :type blacklist: [string]
    :return: decorated function
    """
    if blacklist is None:
        blacklist = {}
    else:
        blacklist = {b.lower() for b in blacklist}

    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            if (blacklist is not None) and ('gpuIds' in kwargs):
                gpu_ids = kwargs['gpuIds']
                passed_ids = []
                for gpuName, gpuId in get_device_names(gpu_ids=gpu_ids, handle=kwargs.get('handle', None)):
                    if gpuName not in blacklist:
                        passed_ids.append(gpuId)
                    else:
                        logger.info(
                            "GPU %s (id: %d) was blacklisted from participating in the test." % (gpuName, gpuId))
                kwargs['gpuIds'] = passed_ids

            fn(*args, **kwargs)
            return

        return wrapper

    return decorator

def run_with_developer_mode(msg="Use developer mode to enable this test."):
    """
    Run test only when developer mode is set.
    """
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwds):
            if not option_parser.options.developer_mode:
                skip_test(msg)
            fn(*args, **kwds)
            return
        return wrapper
    return decorator

def are_any_nvlinks_down(handle):
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()

    #Will throw an exception on API error
    linkStatus = systemObj.discovery.GetNvLinkLinkStatus()

    #Further sanity checks
    for i in range(linkStatus.numGpus):
        for j in range(dcgm_structs.DCGM_NVLINK_MAX_LINKS_PER_GPU):
            ls = linkStatus.gpus[i].linkState[j]
            if ls == dcgm_structs.DcgmNvLinkLinkStateDown:
                return True

    for i in range(linkStatus.numNvSwitches):
        for j in range(dcgm_structs.DCGM_NVLINK_MAX_LINKS_PER_NVSWITCH):
            ls = linkStatus.nvSwitches[i].linkState[j]
            if ls == dcgm_structs.DcgmNvLinkLinkStateDown:
                return True

    return False


def skip_test_if_any_nvlinks_down(handle):
    if are_any_nvlinks_down(handle):
        skip_test("Skipping test due to a NvLink being down")


def is_nvidia_fabricmanager_running():
    """
    Return True if nvidia-fabricmanager service is running on the system
    """
    cmd = 'systemctl status nvidia-fabricmanager'
    p = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)
    out_buf, _ = p.communicate()
    out = out_buf.decode('utf-8')
    if "running" in out.rstrip():
        return True
    else:
        return False

def is_framework_compatible():
    """
    Checks whether the Test Framework is using the expected build version DCGM
    """

    #initialize the DCGM library globally ONCE
    try:
        dcgm_structs._dcgmInit(utils.get_testing_framework_library_path())
    except dcgm_structs.dcgmExceptionClass(dcgm_structs.DCGM_ST_LIBRARY_NOT_FOUND):
        print("DCGM Library hasn't been found in the system, is the DCGM package correctly installed?", file=sys.stderr)
        sys.exit(1)

    rawVersionInfo = dcgm_agent.dcgmVersionInfo()
    versionInfo = {}
    for kv in str.split(rawVersionInfo.rawBuildInfoString, ';'):
        if (kv):
            key, value = str.split(kv, ':', 1)
            versionInfo[key] = value

    def validate_build_info(key, expected):
        if key not in versionInfo:
            logger.warning("Key %s was expected but not provided in the build info" % key)
            return False
        if versionInfo[key] != expected:
            logger.warning(
                "{key} equals to '{value}', but the test framework was built with {expected} instead".format(
                    key=key,
                    value=versionInfo[key],
                    expected=expected))
            return False
        return True

    #      version : DCGM Version.<br>
    #      arch    : Target DCGM Architecture.<br>
    #      buildid : Build ID. Usually a sequential number.<br>
    #      commit  : Commit ID (Usually a git commit hash).<br>
    #      branch  : Branch (Usually a git branch that was used for the build).<br>
    #      builddate : Date of the build.<br>
    #      buildplatform   : Platform where the build was made.<br>
    #      buildtype : Build Type (Debug, Release, etc.)

    # DCGM_VERSION
    # TARGET_ARCH
    # BUILD_ID
    # BUILD_DATE
    # GIT_COMMIT
    # GIT_BRANCH
    # BUILD_PLATFORM

    if not validate_build_info("version", version.DCGM_VERSION):
        return False
    if not validate_build_info("commit", version.GIT_COMMIT):
        return False
    if not validate_build_info("arch", version.TARGET_ARCH):
        return False
    if not validate_build_info("buildtype", version.BUILD_TYPE):
        return False

    return True


def is_test_environment_sane():
    """
    Checks whether the SUT (system under test) has any obvious issues
    before allowing the test framework to run
    """

    print("\n########### VERIFYING DCGM TEST ENVIRONMENT  ###########\n")

    ############## INFOROM CORRUPTION ##############
    nvsmiObj = nvidia_smi_utils.NvidiaSmiJob()
    inforomCorruption = nvsmiObj.CheckInforom()
    if inforomCorruption:
        logger.warning("Corrupted Inforom Detected, exiting framework...\n")
        return False

    ############## PAGE RETIREMENT ##############
    pageRetirementBad = nvsmiObj.CheckPageRetirementErrors()
    if pageRetirementBad:
        logger.warning("Page Retirement issues have been detected, exiting framework...\n")
        return False

    return True

def run_with_persistence_mode_on():
    """
    Run this test with persistence mode on. This function runs "nvidia-smi -pm 0"
    before the test is run
    """
    def decorator(fn):
        cmd = ['nvidia-smi', '-pm', '1']
        @wraps(fn)
        def wrapper(*args, **kwds):
            fnull = open(os.devnull, 'w')
            if not find_executable(cmd[0]):
                logger.info("Could not find executable " + cmd[0] + ". Skipping enabling persistence mode")
            elif not utils.is_root():
                logger.info("Not running as root. Skipping enabling persistence mode")
            else:
                logger.info("Enabling persistence mode")
                p = Popen(cmd, stdout=fnull, stderr=fnull, close_fds=True)
                p.wait()
            fn(*args, **kwds)
            return
        return wrapper
    return decorator

def get_device_id(handle, gpuId):
    attrs = dcgm_agent.dcgmGetDeviceAttributes(handle, gpuId)
    return attrs.identifiers.pciDeviceId >> 16

def is_throttling_masked_by_nvvs(handle, gpuId, throttle_type):
    deviceId = get_device_id(handle, gpuId)
    if deviceId == 0x102d or deviceId == 0x1eb8:
        return True
    elif deviceId == 0x1df6:
        return throttle_type == dcgm_fields.DCGM_CLOCKS_THROTTLE_REASON_SW_THERMAL
    elif deviceId == 0x1e30:
        ignored = [ dcgm_fields.DCGM_CLOCKS_THROTTLE_REASON_HW_SLOWDOWN,
                    dcgm_fields.DCGM_CLOCKS_THROTTLE_REASON_HW_THERMAL,
                    dcgm_fields.DCGM_CLOCKS_THROTTLE_REASON_SW_THERMAL ]
        return throttle_type in ignored

    return False

def is_mig_mode_enabled():
    cmd = 'nvidia-smi mig -lgi'
    p = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE, universal_newlines=True)
    out, err = p.communicate()
    # Return false if we detect MIG=off or that we are running on an older
    # driver that does not support MIG
    if out.find("No MIG-enabled devices found.") != -1 or out.find("Invalid combination of input arguments.") != -1:
        return False
    else:
        return True

def run_only_if_mig_is_disabled():
    '''
    Decorator to skip tests that are unsupported or not yet supported with MIG mode enabled
    '''
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwds):
            if is_mig_mode_enabled():
                skip_test("This test is not yet supported while MIG mode is enabled.")
            else:
                result = fn(*args, **kwds)
        return wrapper
    return decorator

def run_only_if_mig_is_enabled():
    '''
    Decorator to skip tests that only matter if MIG mode is enabled
    '''
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwds):
            if not is_mig_mode_enabled():
                skip_test("this test does nothing if MIG mode is not enabled")
            else:
                result = fn(*args, **kwds)
        return wrapper
    return decorator

def is_mig_incompatible_failure(failure_msg):
    mig_incompatible_str = "MIG configuration is incompatible with the diagnostic because it prevents access to the entire GPU."
    pos = failure_msg.find(mig_incompatible_str)
    # Return true if this is the end of the error string
    return pos != -1 and pos + len(mig_incompatible_str) == len(failure_msg) - 1

def diag_execute_wrapper(dd, handle):
    try:
        response = dd.Execute(handle)
        return response
    except dcgm_structs.DCGMError as e:
        if is_mig_incompatible_failure(str(e)):
            skip_test("Skipping this test because MIG is configured incompatibly (preventing access to the whole GPU)")
        else:
            raise e

def action_validate_wrapper(runDiagInfo, handle, runDiagVersion=dcgm_structs.dcgmRunDiag_version7):
    try:
        response = dcgm_agent.dcgmActionValidate_v2(handle, runDiagInfo, runDiagVersion)
        return response
    except dcgm_structs.DCGMError as e:
        if is_mig_incompatible_failure(str(e)):
            skip_test("Skipping this test because MIG is configured incompatibly (preventing access to the whole GPU)")
        else:
            raise e

def run_only_if_checking_libraries():
    '''
    Decorator to only run a test if we're verifying the modules
    '''
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwds):
            if option_parser.options.no_library_check:
                skip_test("The test framework has been run with --no-library-check, skipping this test.")
            else:
                result = fn(*args, **kwds)
        return wrapper
    return decorator

def run_with_logging_on():
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwds):
            set_logging_state(True)
            fn(*args, **kwds)
            restore_logging_state()
            return
        return wrapper
    return decorator

def set_nvvs_bin_path():
    '''
    Helper to make sure that NVVS_BIN_PATH is set so that dcgmi diag will actually run
    '''
    dirOfThisFile = os.path.dirname(os.path.realpath(__file__))
    nvvsDir = os.path.join(dirOfThisFile, 'apps/nvvs')

    if (not os.path.isdir(nvvsDir)) or (not os.path.isfile(nvvsDir + '/nvvs')):
        logger.warning("NVVS is missing from the test framework install. Hopefully it's installed. Looked in " + nvvsDir)
    else:
        logger.debug("NVVS directory: %s" % nvvsDir)
        os.environ['NVVS_BIN_PATH'] = nvvsDir #The env variable parser in DcgmDiagManager is only the directory
