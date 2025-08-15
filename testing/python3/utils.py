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
## {{{ http://code.activestate.com/recipes/577479/ (r1)
from collections import namedtuple
from functools import wraps
import ctypes
import os
import sys
import fnmatch
import logger
import platform
import subprocess
import string
import stat
import getpass
import signal
import socket
try:
    from distro import linux_distribution
except ImportError:
    try:
        from platform import linux_distribution
    except ImportError:
        print("Please install the distro package")
        raise

import option_parser
from subprocess import check_output

_CacheInfo = namedtuple("CacheInfo", "hits misses maxsize currsize")

def cache():
    """Memorizing cache decorator.

    Arguments to the cached function must be hashable.

    View the cache statistics named tuple (hits, misses maxsize, size) with
    f.cache_info().  Clear the cache and statistics with f.cache_clear().

    """

    def decorating_function(user_function,
                tuple=tuple, sorted=sorted, len=len, KeyError=KeyError):

        cache = dict()
        hits_misses = [0, 0]
        kwd_mark = object()             # separates positional and keyword args

        @wraps(user_function)
        def wrapper(*args, **kwds):
            key = args
            if kwds:
                key += (kwd_mark,) + tuple(sorted(kwds.items()))
            try:
                result = cache[key]
                hits_misses[0] += 1
            except KeyError:
                result = user_function(*args, **kwds)
                cache[key] = result
                hits_misses[1] += 1
            return result

        def cache_info():
            """Report cache statistics"""
            return _CacheInfo(hits_misses[0], hits_misses[1], None, len(cache))

        def cache_clear():
            """Clear the cache and cache statistics"""
            cache.clear()
            hits_misses = [0, 0]

        wrapper.cache_info = cache_info
        wrapper.cache_clear = cache_clear
        return wrapper

    return decorating_function


# ----- Example ----------------------------------------------------------------

if __name__ == '__main__':

    @cache()
    def fib(n):
        if n < 2:
            return 1
        return fib(n-1) + fib(n-2)

    from random import shuffle
    inputs = list(range(30))
    shuffle(inputs)
    results = sorted(fib(n) for n in inputs)
    print(results)
    print((fib.cache_info()))
        
    expected_output = '''[1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 
         233, 377, 610, 987, 1597, 2584, 4181, 6765, 10946, 17711, 28657, 
         46368, 75025, 121393, 196418, 317811, 514229, 832040]
         CacheInfo(hits=56, misses=30, maxsize=None, currsize=30)
    '''
## end of http://code.activestate.com/recipes/577479/ }}}

# Log an exception message before raising it.
def logException(msg):
    logger.error("Exception. " + msg);
    raise Exception(msg)

def is_root():
    if is_linux():
        return os.geteuid() == 0
    else:
        return ctypes.windll.shell32.IsUserAnAdmin()

def is_real_user_root():
    """
    Effective user can be changed. But real user is always the same and can't be changed

    """
    if is_linux():
        return os.getuid() == 0
    else:
        # Windows can't change user so implementation is the same as in is_root()
        return ctypes.windll.shell32.IsUserAnAdmin()


_UserInfo = namedtuple("UserInfo", "uid, gid, name")
@cache()
def get_user_idinfo(username):
    from pwd import getpwnam
    info = getpwnam(username)
    return _UserInfo(info.pw_uid, info.pw_gid, info.pw_name)

@cache()
def get_name_by_uid(uid):
    from pwd import getpwuid
    return getpwuid(uid).pw_name

script_dir = os.path.realpath(sys.path[0])

def find_files(path = script_dir, mask = "*", skipdirs=None, recurse=True, test_file_name="", skip_file_name=""):
    skipdirs = skipdirs or []
    #Recurse subdirectories?
    if recurse:
        for root, dirnames, filenames in os.walk(path):
            if skipdirs is not None:
                [dirnames.remove(skip) for skip in skipdirs if skip in dirnames]  # don't visit directories in skipdirs list
            for filename in fnmatch.filter(filenames, mask):
                if test_file_name != "":
                    if test_file_name==filename:
                        yield os.path.abspath(os.path.join(root, filename))
                else:
                    if filename != skip_file_name:
                        yield os.path.abspath(os.path.join(root, filename))
    else:
        #Just list files inside "path"
        filenames = os.listdir(path)
        for filename in fnmatch.filter(filenames, mask):
            if test_file_name != "":
                if test_file_name==filename:
                    yield os.path.abspath(os.path.join(root, filename))
            else:
                if filename != skip_file_name:
                    yield os.path.abspath(os.path.join(root, filename))

def which(name):
    """
    Returns True if command line application is in the PATH.
    """
    if is_linux():
        return 0 == os.system('which "%s" 1> /dev/null 2> /dev/null' % name)
    else:
        # TODO on windows os.system pops (for a brief moment) cmd console
        # this function should be reimplemented so that it wouldn't happen
        return 0 == os.system('where "%s" &> NUL' % name)

"""
stores string representing current platform
expected returned values
"Linux_64bit"
"Linux_32bit"
"Windows_64bit"
"Linux_aarch64"

treats VMkernel platform as Linux
"""

current_os = platform.system()
if current_os == "VMkernel":
    current_os = "Linux" # Treat VMkernel as normal Linux.

def is_windows(os=current_os):
    return os == "Windows"
def is_linux(os=current_os):
    return os  == "Linux"

def is_cuda_supported_system():
    # CUDA is supported everywhere except in virtualization environments
    return is_bare_metal_system()

def is_healthmon_supported_system():
    return is_linux() and is_cuda_supported_system()

def is_esx_hypervisor_system():
    return platform.system() == "VMkernel"

def is_microsoft_hyper_v():

    try:
        dmi = check_output(["which", "dmidecode"])
    except Exception:
        pass
        return False

    if is_root():        
        if os.path.isfile(dmi.strip()):    
            systemType = check_output(["dmidecode", "-s", "system-product-name"])
            if systemType.strip() == "Virtual Machine":
                return True
        else:
            return False
    else:
        return False

# Util method to check if QEMU VM is running
# DGX-2 VM uses QEMU
def is_qemu_vm():
    """
    Returns True if QEMU VM is running on the system()
    """

    cmd = 'lshw -c system | grep QEMU | wc -l'
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()

    if int(out) > 0:
        return True
    else:
        return False

def is_bare_metal_system():
    if is_esx_hypervisor_system():
        return False
    elif is_linux() and linux_distribution()[0] == "XenServer":
        return False
    elif is_linux() and is_microsoft_hyper_v():
        return False
    elif is_linux() and is_qemu_vm():
        return False
    else:
        return True

def is_64bit():
    if os.name == 'nt':
        if platform.uname()[4] == 'AMD64':
            return True
    return platform.architecture()[0] == "64bit"
     
def is_32bit():
    if os.name == 'nt':
        if platform.uname()[4] == 'x86':
            return True            
    return platform.architecture()[0] == "32bit"
    
def is_system_64bit():
    return platform.machine() in ["x86_64", "AMD64"]

# 32-bit Python on 64-bit Windows reports incorrect architecture, therefore not using platform.architecture() directly
platform_identifier = current_os + "_" + ("64bit" if is_64bit() else "32bit")
if platform.machine() == "aarch64":
    platform_identifier = "Linux_aarch64"
assert platform_identifier in ["Linux_32bit", "Linux_64bit", "Windows_64bit", "Linux_aarch64"], "Result %s is not of expected platform" % platform_identifier

valid_file_name_characters = "-_.() " + string.ascii_letters + string.digits
def string_to_valid_file_name(s):
    """
    Replaces invalid characters from string and replaces with dot '.'

    """
    result = []
    for ch in s:
        if ch in valid_file_name_characters:
            result.append(ch)
        else:
            result.append(".")
    return "".join(result)

def gen_diff(left, right):
    import difflib
    s = difflib.SequenceMatcher(None, left, right)
    for tag, i1, i2, j1, j2 in s.get_opcodes():
        if tag == "equal":
            for k in range(i1,i2):
                l = left[k]
                r = right[k-i1+j1]
                yield (" ", l, r) 
        elif tag == "insert":
            for k in range(j1,j2):
                r = right[k]
                yield ("+", "", r) 
        elif tag == "delete":
            for k in range(i1,i2):
                l = left[k]
                yield ("-", l, "")
        elif tag == "replace":
            for k in range(i1,i2):
                l = left[k]
                r = right[k-i1+j1]
                # difflib combines blocks for some reason and returns "replace" tag
                # for lines that are the same. Let's fix that
                if l == r: 
                    yield (" ", l ,r)
                else:
                    yield ("|", l ,r)

def plural_s(val):
    """
    returns "s" if val > 1 or "" otherwise.
    Can be used in strings to have proper plural form.

    """
    if val > 1:
        return "s"
    return ""

def chunks(l, n):
    """
    returns list of list of length n.
    E.g. chunks([1, 2, 3, 4, 5], 2) returns [[1, 2],  [3, 4], [5]]
    """
    return [l[i:i+n] for i in range(0, len(l), n)]

def format_dev_sub_dev_id(pciIdPair):
    """
    pciIdPair (int pci device id, int pci sub device id or None)
    """
    if pciIdPair[1] is None:
        return "(0x%08X, None)" % pciIdPair[0]
    return "(0x%08X, 0x%08X)" % pciIdPair
        
# permission of:      other                         owner                         group
stat_everyone_read_write = stat.S_IROTH | stat.S_IWOTH | stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP

# Exit if the current (effective) user can't create a file in the base test directory
def verify_file_permissions(user):

    # Not a complete check, but enough to verify absolute path permission issues
    try:
        filename = os.path.join(os.path.realpath(script_dir), "test.txt")
        f = open(filename, "w")
        f.write("Permission test")
        f.close()
        os.remove(filename)
    except:
        print("Please run the test framework under a less restrictive directory, with RW access to the full path.")
        print("The user '%s' doesn't have sufficient permissions here." % user)
        sys.exit(1)

# Exit if either the current user, or specified non-root user appear to lack sufficient
# file system permissions for the test framework
def verify_user_file_permissions():
    import test_utils

    # Check current user
    verify_file_permissions(getpass.getuser())

    # Check non-root user, if specified
    user = option_parser.options.non_root_user
    if user:
        try:
            get_user_idinfo(user)
        except KeyError:
            print("User '%s' doesn't exist" % user)
            sys.exit(1)
        with test_utils.RunAsNonRoot(reload_driver=False):
            verify_file_permissions(user)

# Exit if 'localhost' does not resolve to 127.0.0.1
def verify_localhost_dns():
    try:
        host_ip = socket.gethostbyname("localhost")
    except:
        print("Unable to resolve localhost")
        sys.exit(1)
    if host_ip != "127.0.0.1":
        print("localhost does not resolve to 127.0.0.1")
        sys.exit(1)

## Util method to check if the mps server is running in the background
def is_mps_server_running():
    """
    Returns True if MPS server is running on the system
    """

    if is_linux():
        ## MPS server is only supported on Linux.
        cmd = 'ps -aux | grep nvidia-cuda | tr -s " " | cut -d " " -f 11'
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()
        if b"nvidia-cuda-mps-server" in out.rstrip():
            return True
        elif b"nvidia-cuda-mps-control" in out.rstrip():
            return True
        else:
            return False
    else:
        ## MPS server is not supported on Windows. Return False for Windows
        return False

def shorten_path(path, shorten_to_levels=2):
    '''
    Given a path, return a path of only the last "shorten_to_levels" levels.
    For example, shorten_path('a/b/c/d/e', 2) => "d/e"
    '''
    path = os.path.normpath(path)
    shortened_paths = path.split(os.sep)[-shorten_to_levels:]
    return os.path.join(*shortened_paths)

def create_dir(path):
    '''
    Create the full directory structure specified by path.  If the directory cannot be created 
    due to permission issues or because part of the path already exists and is not a directory
    then an OSError is raised.
    '''
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise

def wait_for_pid_to_die(pid):
    '''This function returns once the pid no longer exists'''
    while True:
        try:
            os.kill(pid, 0)
        except OSError:
            break

def verify_dcgm_service_not_active():
    cmd = 'systemctl is-active --quiet dcgm'
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    if p.returncode == 0:
        logException("Tests cannot run because the DCGM service is active")

    cmd = 'systemctl is-active --quiet nvidia-dcgm'
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    if p.returncode == 0:
        logException("Tests cannot run because the Nvidia DCGM service is active")

def verify_nvidia_fabricmanager_service_active_if_needed():
    cmd = "find /dev -regextype egrep -regex '/dev/nvidia-nvswitch[0-9]+'"
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out_buf, _ = p.communicate()
    out = out_buf.decode('utf-8')
    if not "nvswitch" in out.rstrip():
        return

    # Fabricmanager must be running if there are nvswitch devices.
    cmd = 'systemctl status nvidia-fabricmanager'
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out_buf, _ = p.communicate()
    out = out_buf.decode('utf-8')
    if not "running" in out.rstrip():
        logException("Tests cannot run because the Nvidia Fabricmanager service is not active on systems with nvswitches")

    # Check for version mismatch
    cmd = "/usr/bin/dmesg | /usr/bin/grep -F 'nvidia-nvswitch: Version mismatch' | /usr/bin/tail -1"
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out_buf, _ = p.communicate()
    if out_buf:
        out_buf = out_buf.decode('utf-8')
        errmsg = f'Tests cannot run because dmesg contains this problem: "{out_buf}".'
        if option_parser.options.ignore_dmesg_checks:
            logger.warning(f'{errmsg} Resolve the problem and retry before filing a bug report. (Ignored by user request).')
        else:
            logException(f'{errmsg} Resolve the problem before proceeding.')

def checkDmesgForProblems():
    if not option_parser.options.dvssc_testing:
        ignore_reason = 'Ignored as \'--dvssc-testing\' is not set'
    if option_parser.options.ignore_dmesg_checks:
        ignore_reason = 'Ignored by user request'
    ignore_problems = option_parser.options.ignore_dmesg_checks

    # Lines matching these patterns only emit warnings.
    # If a pattern is a refinement of an 'abort_on_...' pattern, it will prevent the abort.
    warn_on_these = [
        r'NVRM: nvAssertFailed: Assertion failed: 0 @ g_kernel_gsp_nvoc.h',
        r'NVRM: .*Possible bad register read',
    ]

    # Lines matching these patterns only abort when --dvssc-testing is set (unless ignored).
    abort_on_these_if_dvssc = [
        r'NVRM: A GPU crash dump has been created',
        r'NVRM: .*GPU recovery action changed.*Reboot Required',
        r'NVRM: .*GPU has fallen off the bus',
    ]

    # Lines matching these patterns always abort unless ignored.
    abort_on_these = [
        r'NVRM: .*Assertion failed:',
    ]

    def check_display_limit():
        if display_limit <= 0:
            logger.info("Maximum problems displayed. Further potential problems may be present and not displayed here.")

    # Read up to `history_len` lines from dmesg. This could use a time-based search or starting with the most recent driver load.
    history_len = 25000
    cmd = f"/usr/bin/dmesg | /usr/bin/tail -{int(history_len)}"
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out_buf, _ = p.communicate(timeout=15)
    if not out_buf:
        logger.debug('No output from dmesg')
        return

    # Build the regexes used to find problems to warn or abort on
    import re
    if option_parser.options.dvssc_testing:
        warn_pattern = re.compile('|'.join(warn_on_these))
        abort_pattern = re.compile('|'.join(abort_on_these_if_dvssc + abort_on_these))
    else:
        warn_pattern = re.compile('|'.join(warn_on_these + abort_on_these_if_dvssc))
        abort_pattern = re.compile('|'.join(abort_on_these))

    # Don't spam output on a messy system. Print the first few messages, then stop. Abort still means abort.
    display_limit = 5

    for line in out_buf.decode('utf-8').split('\n'):
        abort_match = abort_pattern.search(line)
        if abort_match:
            if not warn_pattern.search(abort_match.string):
                if ignore_problems:
                    if display_limit > 0:
                        logger.warning(f'Potential problem found in dmesg: "{abort_match.string}". ({ignore_reason}).')
                        display_limit -= 1
                        check_display_limit()
                else:
                    logException(f'Tests cannot run because dmesg contains this problem: \'{abort_match.string}\'. ' \
                                 'Resolve the problem before proceeding.')
            elif display_limit > 0:
                logger.warning(f'Potential problem found in dmesg: "{abort_match.string}". (Ignored as it is believed to be non-fatal).')
                display_limit -= 1
                check_display_limit()
        else:
            warn_match = warn_pattern.search(line)
            if warn_match and display_limit > 0:
                logger.warning(f'Potential problem found in dmesg: "{warn_match.string}".')
                display_limit -= 1
                check_display_limit()

def checkProcesses():
    process_list = \
    [
        'nv-hostengine',
        'nvvs'
    ]

    for procname in process_list:
        if option_parser.options.use_running_hostengine and procname == 'nv-hostengine':
            continue
        cmd = f'/usr/bin/pidof {procname}'
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out_buf, _ = p.communicate(timeout=5)
        if out_buf:
            out_buf = out_buf.decode('utf-8')
            errmsg = f'Tests cannot run because one or more potentially conflicting process \'{procname}\' is running with pid(s): {out_buf}.\n'
            if option_parser.options.ignore_conflicting_procs:
                logger.warning(f'{errmsg} Remove the conflicting processes and retry before filing a bug report. (Ignored by user request).')
            else:
                logException(f'{errmsg} Remove the conflicting processes before proceeding.')

def checkSystemLoad():
    try:
        nproc = os.cpu_count()
        warnAt = 0.85 * nproc
        failAt = 1.0 * nproc
    except:
        logger.warning("Unable to get system cpu count, host CPU load not checked")
        return

    try:
        load1, load5, _ = os.getloadavg()
        if max(load1, load5) > failAt:
            errmsg = f'Tests cannot run because load average exceeds {failAt}.'
            if option_parser.options.ignore_loadavg_checks:
                logger.warning(f'{errmsg} Terminate other disruptive processes and retry before filing a bug report. (Ignored by user request).')
            else:
                logException(f'{errmsg} Terminate other disruptive processes before proceeding.')
        elif max(load1, load5) > warnAt:
            errmsg = f'Load average exceeds {warnAt}. Terminate other disruptive processes and retry before filing a bug report.'
            logger.warning(errmsg)

    except OSError:
        logger.warning('Unable to retrieve load average, host CPU load not checked')
        return

def checkMemoryPressure():
    failAtMem = 0.95
    warnAtMem = 0.90
    failAtSwap = 1.00
    warnAtSwap = 0.50

    def normalizeToKb(val, units):
        if not units:
            if val == 0:
                return val
            else:
                return val / 1024.0
        if units == 'kB':
            return val
        elif units == 'MB':
            return val * (1024.0 ** 2)
        elif units == 'GB':
            return val * (1024.0 ** 3)
        elif units == 'TB':
            return val * (1024.0 ** 4)
        elif units == 'PB':
            return val * (1024.0 ** 5)
        else:
            raise ValueError(f'Unhandled unit type \"{units}\"')

    def collectNormalizedVal(line):
        _, valWithUnits = line.split(':', 1)
        valWithUnits = valWithUnits.strip()
        try:
            val, units = valWithUnits.split(' ', 1)
            return normalizeToKb(int(val), units)
        except ValueError:
            return normalizeToKb(int(val), "")

    memTotal = memAvail = swapTotal = swapFree = None

    try:
        with open('/proc/meminfo', 'r') as f:
            meminfo = f.readlines()
            for line in meminfo:
                if 'MemTotal:' in line:
                    memTotal = collectNormalizedVal(line)
                elif 'MemAvailable:' in line:
                    memAvail = collectNormalizedVal(line)
                elif 'SwapTotal:' in line:
                    swapTotal = collectNormalizedVal(line)
                elif 'SwapFree:' in line:
                    swapFree = collectNormalizedVal(line)
                if memTotal and memAvail and swapTotal and swapFree:
                    break
    except IOError as e:
        logger.warning(f'Unable to read \'/proc/meminfo\': {e}: skipping memory and swap utilization check')
        return

    memUsed = (memTotal - memAvail) / memTotal
    if memUsed > failAtMem:
        msg = f"Tests cannot run because memory available memory is {round(memUsed * 100.0),1}%."
        if option_parser.options.ignore_mem_checks:
            logger.warning(f'{msg} Resolve memory pressure and retry before filing a bug report. (Ignored by user request).')
        else:
            logException(f'{msg} Resolve memory pressure before proceeding.')
    if memUsed > warnAtMem:
        logger.Warning(f'Available memory is {round(memUsed * 100.0, 1)}%. Resolve memory pressure and ' \
                       'retry before filing a bug report.')

    if swapTotal > 0:
        swapUsed = (swapTotal - swapFree) / swapTotal
        if swapUsed > failAtSwap:
            msg = f'Tests cannot run because available swap is {round(swapUsed * 100.0, 1)}%.'
            if option_parser.options.ignore_mem_checks:
                logger.warning(f'{msg} Resolve memory pressure and retry before filing a bug report. (Ignored by user request).')
            else:
                logException(f'{msg} Resolve memory pressure before proceeding.')
        if swapUsed > warnAtSwap:
            logger.Warning(f'Available swap is {round(swapUsed * 100.0, 1)}%. Resolve memory pressure and ' \
                           'retry before filing a bug report.')

def verifyEcosystem():
    if option_parser.options.filter_tests:
        logger.warning("filter_tests has been selected, skipping preflight checks.")
        return
    checkSystemLoad()
    checkMemoryPressure()
    checkProcesses()
    checkDmesgForProblems()

def find_process_using_hostengine_port():
    cmd = 'lsof -i -P -n | grep -Fw 5555'
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out_buf, err_buf = p.communicate()
    out = out_buf.decode('utf-8')
    err = err_buf.decode('utf-8')
    if len(out) == 0:
        # Nothing is using that port, we are good
        return None
    
    he = "nv-hostengine"
    process_name = None
    
    lines = out.split('\n')
    for line in lines:
        info = line.split()
        if len(info) < 2:
            continue

        if info[-1] != '(LISTEN)':
            continue # not listening, ignore

        process_name = info[0]
        pid = int(info[1])

        if he.find(info[0]) == 0:
            # an nv-hostengine process is found
            return pid

    if process_name:
        # We have some other process using port 5555
        msg = "Process %s with pid %d is listening to port 5555. Cannot run tests." % (process_name, pid)
        logException(msg)

'''
Attempt to clean up zombie or accidentally left-open processes using port 5555
'''
def verify_hostengine_port_is_usable():
    pid = find_process_using_hostengine_port()
    if not pid:
        # no hostengine process, move on with life
        return
    
    verify_dcgm_service_not_active()
    os.kill(pid, signal.SIGKILL)

    return 


def get_testing_framework_library_path():
    # type: () -> string
    """
    Returns platform dependent path to the dcgm libraries.
    To use in testing framework for :func:`_dcgmInit` calls
    :return: String with relative path to the libdcgm*.so libraries
    """

    def _get_arch_string():
        # type: () -> string
        if platform.machine() in ["x86_64", "AMD64"]:
            return 'amd64'
        elif platform.machine().lower().startswith('aarch64'):
            return 'aarch64'

        logException("Unsupported arch: %s" % platform.machine())

    return './apps/%s/' % _get_arch_string()

"""
    Makes sure we can locate nvvs and dcgmi. If not, exit with an error.
    If so, set NVVS_BIN_PATH appropriately and return the path to dcgmi
"""
def verify_binary_locations():
    nvvs_location = "%s/apps/nvvs/nvvs" % os.getcwd()
    if not os.path.isfile(nvvs_location):
        print(("nvvs is NOT installed in: %s\n" % nvvs_location))
        sys.exit(1)

    # This will instruct the embedded hostengine to find the nvvs binary here
    nvvs_location = nvvs_location[:-5]
    print(("Setting NVVS_BIN_PATH to %s" % nvvs_location))
    os.environ["NVVS_BIN_PATH"] = nvvs_location

    dcgmi_location = os.path.join(get_testing_framework_library_path(), "dcgmi")

    if not os.path.isfile(dcgmi_location):
        print(("dcgmi is NOT installed in: %s\n" % dcgmi_location))
        sys.exit(1)

    dcgmi_prefix = dcgmi_location.strip("/dcgmi")

    return dcgmi_prefix