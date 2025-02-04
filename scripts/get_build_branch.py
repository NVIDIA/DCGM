#!/usr/bin/env python3

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

import re
import subprocess
import sys

def get_branch_names(commit):
    process = subprocess.run(["git", "branch", "-a", "--no-abbrev",
                              "--contains", commit, "--format",
                              "%(refname:lstrip=2)"], text=True,
                             capture_output=True)
    out = process.stdout
    status = process.returncode

    if status != 0:
        raise Exception("git branch returned an error")

    return out.split('\n')

def is_detached_head(branch):
    return bool(re.match(r'^\(HEAD detached at [a-z0-9]+\)', branch))

def is_mr_branch(branch):
    return bool(re.match(r'^merge[-_]requests/', branch))

def trim_branch_name(branch):
    return re.sub(r'^(origin|remotes)/', '', branch)

def negate(fn):
    return lambda x: not fn(x)

def pick_release_branch(branches):
    found = filter(lambda branch: re.match(r'^rel_dcgm_\d+_\d+', branch), branches)
    return next(found, None)

def pick_main_branch(branches):
    found = filter(lambda branch: re.match(r'^master|main$', branch), branches)
    return next(found, None)

def main():
    commit = sys.argv[1]
    branches = list(
      map(trim_branch_name,
      filter(negate(is_detached_head),
      filter(negate(is_mr_branch),
      get_branch_names(commit)))))

    release_branch = pick_release_branch(branches)
    main_branch = pick_main_branch(branches)

    out = ""

    if release_branch:
        out = release_branch
    elif main_branch:
        out = main_branch
    elif branches:
        out = branches[0]

    print(out, end="")

if __name__ == '__main__':
    main()
