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
from subprocess import PIPE, Popen

###############################################################################################
#
# Utility function to execute bash commands from this script
# Prints the stdout to screen and returns a return code from the shell
#
###############################################################################################
def executeBashCmd(cmd, prnt):
    """
    Executes a shell command as a separated process, return stdout, stderr and returncode
    """
    ret_line = ''

    print("executeCmd: \"%s\"" % str(cmd))
    try:
        result = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True)
        ret_line = result.stdout.readline().decode('utf-8')
        while True:
            line = result.stdout.readline().decode('utf-8')
            if prnt:
                print(line.strip('\n'))
            if not line:
                break
            sys.stdout.flush()
        (stdout_buf, stderr_buf) = result.communicate()
        stdout = stdout_buf.decode('utf-8')
        stderr = stderr_buf.decode('utf-8')

        if stdout:
            print(stdout)#pass
        if stderr:
            print(stderr)
    except Exception as msg:
        print("Failed with: %s" % msg)

    return result.returncode, ret_line.strip()

###############################################################################################
#
# This function removes cleans up all the dependent libraries
# It also uninstalls any existing installation of datacenter gpu manager
# Returns success in the end.
#
###############################################################################################
def removeDependencies(prnt):
    #Remove existing installation files and binaries
    ret = executeBashCmd("echo {0} | sudo pip uninstall pandas".format('y'), prnt)
    print(("sudo pip uninstall pandas returned: ", ret[0]))
    #if module is not installed, this returns 1, so check for both values
    if (ret[0] in [0, 1]):
        ret = executeBashCmd("echo {0} | sudo pip uninstall wget".format('y'), prnt)
        print(("sudo pip uninstall wget returned: ", ret[0]))

    if (ret[0] in [0, 1]):
        ret = executeBashCmd("echo {0} | sudo pip uninstall xlwt".format('y'), prnt)
        print(("sudo pip uninstall xlwt returned: ", ret[0]))

    if (ret[0] in [0, 1]):
        ret = executeBashCmd("echo {0} | sudo pip uninstall xlrd".format('y'), prnt)
        print(("sudo pip uninstall xlrd returned: ", ret[0]))

    if ret[0] in [0, 1]:
        print("\nRemoveDependencies returning 0")
        return 0

    print(("\nReturning: ", ret[0]))
    return ret[0]


def installDependencies(prnt):
    ret = 0
    #Install all dependent libraries
    ret = executeBashCmd("sudo pip install pandas", prnt)
    if ret[0] == 0:
        ret = executeBashCmd("sudo pip install wget", prnt)

    if ret[0] == 0:
        ret = executeBashCmd("sudo pip install xlwt", prnt)

    if ret[0] == 0:
        ret = executeBashCmd("sudo pip install xlrd", prnt)

    print("InstallDependencies returning: ", ret[0])
    return ret[0]
