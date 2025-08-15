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

set -o pipefail -o errexit -o nounset

source /etc/os-release

case $ID in
  azurelinux)
    DCGM_MNDIAG_MPIRUN_PATH=/usr/lib/openmpi/bin/mpirun
    DCGM_MNDIAG_LD_LIBRARY_PATH=/usr/lib/openmpi/lib
    ;;
  debian)
    DCGM_MNDIAG_MPIRUN_PATH=/usr/bin/mpirun
    DCGM_MNDIAG_LD_LIBRARY_PATH=
    ;;
  amzn|fedora|rhel|rocky)
    DCGM_MNDIAG_MPIRUN_PATH=/usr/lib64/openmpi/bin/mpirun
    DCGM_MNDIAG_LD_LIBRARY_PATH=/usr/lib64/openmpi/lib
    ;;
  sles|suse)
    DCGM_MNDIAG_MPIRUN_PATH=/usr/lib64/mpi/gcc/openmpi4/bin/mpirun
    DCGM_MNDIAG_LD_LIBRARY_PATH=/usr/lib64/mpi/gcc/openmpi4/lib64
    ;;
  ubuntu)
    if [[ -d /usr/mpi/gcc/openmpi-4.1.7rc1/ ]];
    then
      DCGM_MNDIAG_MPIRUN_PATH=/usr/mpi/gcc/openmpi-4.1.7rc1/bin/mpirun
      DCGM_MNDIAG_LD_LIBRARY_PATH=/usr/mpi/gcc/openmpi-4.1.7rc1/lib
    else
      DCGM_MNDIAG_MPIRUN_PATH=/usr/bin/mpirun
      DCGM_MNDIAG_LD_LIBRARY_PATH=
    fi
    ;;
  *)
    echo "Unsupported distribution" 1>&2
    exit 1
    ;;
esac
    
mkdir --parents /run/systemd/system/nvidia-dcgm.service.d
cat <<EOF > /run/systemd/system/nvidia-dcgm.service.d/50-mndiag.conf
[Service]
Environment="DCGM_MNDIAG_MPIRUN_PATH=$DCGM_MNDIAG_MPIRUN_PATH"
Environment="LD_LIBRARY_PATH=$DCGM_MNDIAG_LD_LIBRARY_PATH"
EOF
