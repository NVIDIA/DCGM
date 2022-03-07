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
import argparse
import util
import sys
#Copy the dcgm data in csv file

def main(cmdArgs):
    metrics = cmdArgs.metrics
    time = cmdArgs.time
    gpuid_list = cmdArgs.gpuid_list
    download_bin = cmdArgs.download_bin

    ret = util.removeDependencies(True)
    if ret == 0:
        ret = util.installDependencies(True)

    if ret == 0:
        if download_bin:
            cmd = '{executable} run_validate_dcgm.py -m {0} -t {1} -d -i {2}'\
                    .format(metrics, time, gpuid_list, executable=sys.executable)
        else:
            cmd = '{executable} run_validate_dcgm.py -m {0} -t {1} -i {2}'\
                    .format(metrics, time, gpuid_list, executable=sys.executable)
        ret = util.executeBashCmd(cmd, True)

    print("\nTests are done, removing dependencies")

    ret = util.removeDependencies(False)

    print("\n All Done")

def parseCommandLine():
    parser = argparse.ArgumentParser(description="Validation of dcgm metrics")
    parser.add_argument("-m", "--metrics", required=True, help="Metrics to be validated \
            E.g. \"1009\", etc")
    parser.add_argument("-i", "--gpuid_list", required=False, default='0', \
            help="comma separated gpu id list starting from 0, eg \"0,1,2\"")
    parser.add_argument("-t", "--time", required=True, help="time in seconds")
    parser.add_argument("-d", "--download_bin", action='store_true', required=False, default=False,\
            help="If specified, download new binaries")

    args = parser.parse_args()
    return args

if __name__ == "__main__":

    # Parsing command line options
    cmdArgs = parseCommandLine()

    main(cmdArgs)
