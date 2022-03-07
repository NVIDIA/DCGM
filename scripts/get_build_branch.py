#!/usr/bin/env python3

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

    return out.split()

def ignore_mr_branches(branches):
    return filter(lambda branch: not re.match(r'^merge[-_]requests/', branch), branches)

def trim_branch_names(branches):
    return map(lambda branch: re.sub(r'^(origin|remotes)/', '', branch), branches)

def pick_release_branch(branches):
    found = filter(lambda branch: re.match(r'^rel_dcgm_\d+_\d+', branch), branches)
    return next(found, None)

def pick_main_branch(branches):
    found = filter(lambda branch: re.match(r'^master|main$', branch), branches)
    return next(found, None)

def main():
    commit = sys.argv[1]
    b1 = get_branch_names(commit)
    b2 = ignore_mr_branches(b1)
    branches = list(trim_branch_names(b2))

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
