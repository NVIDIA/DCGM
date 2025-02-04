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

import os

def test_ptx_version():
    cu_dir = []
    ptx_files = []

    # get the dcgm directory as root directory
    root_directory = os.getcwd() + "/dcgm"
    
    # search for all directories which have .cu file
    for root, _, files in os.walk(root_directory):
        if "_out" not in root and "testing" not in root:
            if any(file.endswith('.cu') for file in files):
                cu_dir.append("dcgm/" + os.path.relpath(root, root_directory))

    # get the .ptx files
    for directory in cu_dir:
        for file_name in os.listdir(directory):
            if file_name.endswith('.ptx'):
                ptx_files.append(directory+"/"+file_name)
    
    # checking .ptx files
    for file in ptx_files:
        with open(file, 'r') as f:
            lines = f.readlines()
        lines = [line.replace('\n', '').replace('\t', '') for line in lines]

        # get line which has .target
        target_present = 0
        for l in lines:
            if ".target" in l: 
                target_present = 1
                break
        
        # .target not present
        assert target_present, "File Name:" + file + " | .target line not present"

        if "dcgmproftester" in file.lower():    
            # check if dcgmproftester
            # check if sm_70 is present
            assert "sm_70" in l, "File Name:" + file + " | sm_70 not present. Present value:" + l.split(" ")[-1]
        else:
            # if dcgm plugin
            # check if sm_30 is present
            assert "sm_30" in l, "File Name:" + file + " | sm_30 not present. Present value:" + l.split(" ")[-1]
                    