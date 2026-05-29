#!/bin/bash
#
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
#

############################################################################################################
# This script attemps to gather all of the relevant DCGM logs after a failure is reported by the Diagnostic.
#
# 
############################################################################################################

function usage() {
    echo "Script for gathering debug information after a DCGM diagnostic failure.
Usage: ${0} [options] [-- [any additional cmake arguments]]
    Where options are:
        -d --dcgm-home-dir            : Full path to the DCGM home directory.
        -l --dcgm-log-dir             : Full path to nv-hostengine's log directory, if different from DCGM home dir.
        -f --fabric-manager-home-dir  : Generated build dumps coverage information
        -h --help                     : This information

    Default values:
        -d /var/log/nvidia-dcgm
        -l /var/log/nvidia-dcgm
        -f /var/log

    If nv-hostengine (DCGM's daemon) was started using the service scripts, then DCGM's home directory was set 
    to /var/log/nvidia-dcgm, which is the default value used for this script.

    Example: ${0} -d /var/log/custom-dcgm-home-dir"
}

set -euxo pipefail

# Create a temporary directory
dateExt=`date +%Y-%m-%dT%H%M%S%z`
tmpname="dcgm${dateExt}"
tmpdir="/tmp/${tmpname}"

mkdir $tmpdir
cd $tmpdir


dcgmHomeDir="/var/log/nvidia-dcgm"
dcgmLogDir="${dcgmHomeDir}"
fmLogDir="/var/log"

LONGOPTS=dcgm-home-dir:,dcgm-log-dir,fabric-manager-log-dir:,help
SHORTOPTS=d:l:f:h

! PARSED=$(getopt --options=${SHORTOPTS} --longoptions=${LONGOPTS} --name "${0}" -- "$@")
if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
    echo "Failed to parse arguments"
    exit 1
fi

eval set -- "${PARSED}"

while true; do
    case "${1}" in
        -d|--dcgm-home-dir)
            dcgmHomeDir=("${2}")
            shift 2
            ;;
        -l|--dcgmLogDir)
            dcgmLogDir=("${2}")
            shift 2
            ;;
        -f|--fabric-manager-log-dir)
            fmLogDir=("${2}")
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "Wrong arguments"
            exit 1
            ;;
    esac
done

# Gather all stats files, plugin files, etc.
cp "${dcgmHomeDir}"/* .

# Gather log files if sent to a different directory
if [[ "${dcgmLogDir}" != "${dcgmHomeDir}" ]]; then
    cp "${dcgmLogDir}"/* .
fi

# Gather the fabric manager log files
if ls "${fmLogDir}"/fabric* 1>/dev/null 2>&1; then
    cp "${fmLogDir}"/fabric* .
else
    echo "No FM Logs found in ${fmLogDir}" > noFMLogs.txt
fi

# Bundle it up nicely
cd /tmp
tarname=dcgmLogs${dateExt}.tgz
tar -czf ${tarname} ${tmpname}
echo "Please attach /tmp/${tarname} as debugging data to any support or bug report."
