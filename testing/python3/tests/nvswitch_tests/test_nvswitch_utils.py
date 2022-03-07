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
import apps
import logger
import subprocess
import test_utils
import time

# p2p_bandwidth app argument lists
P2P_TEST_LIST = ["-l"]
MEMCPY_DTOD_WRITE_CE_BANDWIDTH = ["-t", "Memcpy_DtoD_Write_CE_Bandwidth"]
MEMCPY_DTOD_READ_CE_BANDWIDTH = ["-t", "Memcpy_DtoD_Read_CE_Bandwidth"]

def is_nvidia_docker_running():
    """
    Return True if nvidia-docker service is running on the system
    """
    cmd = 'systemctl status nvidia-docker'
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out_buf, err_buf = p.communicate()
    out = out_buf.decode('utf-8')
    err = err_buf.decode('utf-8')
    if "running" in out.rstrip():
        return True
    else:
        return False

def get_nvswitch_pci_bdf():
    """ Get nvswitch PCI BDFs """
    bdf = []
    try:
        lsPciOutputBuf = subprocess.check_output("lspci | grep -i nvidia | grep -i bridge", shell=True)
        lsPciOutput = lsPciOutputBuf.decode('utf-8')
    except subprocess.CalledProcessError as e:
        logger.error(e.message)
        return bdf

    nvswitches = lsPciOutput.split('\n')
    for i in range(len(nvswitches)):
        dev = nvswitches[i].split()
        if len(dev) > 0:
            bdf.append(dev[0])

    return bdf

def get_gpu_pci_bdf():
    """ Get GPU PCI BDFs """
    bdf = []
    try:
        lsPciOutputBuf = subprocess.check_output("lspci | grep -i nvidia | grep -i '3d controller'", shell=True)
        lsPciOutput = lsPciOutputBuf.decode('utf-8')
    except subprocess.CalledProcessError as e:
        logger.error(e.message)
        return bdf

    gpus = lsPciOutput.split('\n')
    for i in range(len(gpus)):
        dev = gpus[i].split()
        if len(dev) > 0:
            bdf.append(dev[0])

    return bdf

def is_dgx_2_full_topology():
    """
    Return true if detect all nvswitches and GPUs on two base boards or one base board
    """
    switch_bdf = get_nvswitch_pci_bdf()
    gpu_bdf = get_gpu_pci_bdf()

    if len(switch_bdf) == 12 and len(gpu_bdf) == 16:
        return True
    elif len(switch_bdf) == 6 and len(gpu_bdf) == 8:
        return True
    else:
        return False
