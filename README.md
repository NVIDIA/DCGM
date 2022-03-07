# NVIDIA Data Center GPU Manager

[![GitHub license](https://img.shields.io/github/license/NVIDIA/dcgm?style=flat-square)](https://raw.githubusercontent.com/NVIDIA/dcgm/master/LICENSE)
=======================

Data Center GPU Manager (DCGM) is a daemon that allows users to monitor NVIDIA
data-center GPUs. You can find out more about DCGM by visiting [DCGM's official
page](https://developer.nvidia.com/dcgm)

![dcgm](https://developer.nvidia.com/sites/default/files/akamai/datacenter/dcgm-icon.png)

## Introduction

NVIDIA Data Center GPU Manager (DCGM) is a suite of tools for managing and monitoring NVIDIA datacenter GPUs in cluster environments. It includes active health monitoring, comprehensive diagnostics, system alerts and governance policies including power and clock management. It can be used standalone by infrastructure teams and easily integrates into cluster management tools, resource scheduling and monitoring products from NVIDIA partners.

DCGM simplifies GPU administration in the data center, improves resource reliability and uptime, automates administrative tasks, and helps drive overall infrastructure efficiency. DCGM supports Linux operating systems on x86_64, Arm and POWER (ppc64le) platforms. The installer packages include libraries, binaries, NVIDIA Validation Suite (NVVS) and source examples for using the API (C, Python and Go).

DCGM integrates into the Kubernetes ecosystem by allowing users to gather GPU telemetry using [dcgm-exporter](https://github.com/NVIDIA/gpu-monitoring-tools).

More information is available on [DCGM's official page](https://developer.nvidia.com/dcgm)

## Quickstart

DCGM installer packages are available on the CUDA network repository and DCGM can be easily installed using Linux package managers. 

### Ubuntu LTS

**Set up the CUDA network repository meta-data, GPG key:**

```bash
$ wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin \
    && sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600 \
    && sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub \
    && sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
```

**Install DCGM**

```bash
$ sudo apt-get update \
    && sudo apt-get install -y datacenter-gpu-manager
```

### Red Hat

**Set up the CUDA network repository meta-data, GPG key:**

```bash
$ sudo dnf config-manager \
    --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo
```

**Install DCGM**

```bash
$ sudo dnf clean expire-cache \
    && sudo dnf install -y datacenter-gpu-manager
```

### Start the DCGM service

```bash
$ sudo systemctl --now enable nvidia-dcgm
```

## Product Documentation

For information on platform support, getting started and using DCGM APIs, visit the official documentation [repository](https://docs.nvidia.com/datacenter/dcgm/latest).

## Building DCGM

Once this repo is cloned, DCGM can be built by:

- creating the build image
- using the build image to generate a DCGM build

### Why a Docker build image

The Docker build image provides two benefits

- easier builds as the environment is controlled
- a repeatable environment which creates reproducible builds

New dependencies can be added by adding a script in the “scripts” directory
similar to the existing scripts.

As DCGM needs to support some older Linux distributions on various CPU
architectures, the image provide custom builds of GCC compilers that produce
binaries which depend on older versions of the GLibc libraries. The DCGM build
image will also contain all 3rd party libraries that are precompiled using those
custom GCC builds.

### Prerequisites

In order to create the build image and to then generate a DCGM build, you will
need to have the following installed and configured:

* git and git-lfs (can be skipped if downloading a snapshot archive)
* realpath, basename, nproc (coreutils deb package)
* gawk (via awk binary, may not work on MacOS without installing a gawk port)
* getopt (util-linux deb package)
* recent version of Docker

The build.sh script was tested in Linux, Windows (WSL2) and MacOS, though MacOS
may need some minor changes in the script (like `s/awk/gawk/`) as MacOS is not an
officially supported development environment.

### Creating the build image

The build image is stored in `./dcgmbuild`.

The image can be built by:

- ensuring Docker is installed and running
- navigating to `./dcgmbuild`
- running `./build.sh`

Note that if your user does not have permission to access the Docker socket, you
will need to run sudo ./build.sh

The build process may take several hours to create the image as the image is
building 3 versions of GCC toolset for all supported platforms. Once the image
has been built, it can be reused to build DCGM.

### Generating a DCGM build
Once the build image is created, you can use the run `build.sh` to produce builds. A simple debian build of release (non-debug) code for an x86_64 system can be made with: 

`./build.sh -r --deb`

The rpm will be placed in `_out/Linux-amd64-release/datacenter-gpu-manager_2.1.4_amd64.deb`; it can now be installed as needed. The script includes options for building just the binaries (default), tarballs (--packages), or RPM (--rpm) as well. A complete list of options can been seen using `./build.sh -h`.

### Running the Test Framework
DCGM includes an extensive test suite that can be run on any system with one or more supported GPUs. After successfully building DCGM, a `datacenter-gpu-manager-tests` package is created alongside the normal DCGM package. There are multiple ways to run the tests but the most straightforward steps are the following:
1. Install or extract the datacenter-gpu-manager-tests package
2. Navigate to `usr/share/dcgm_tests`
3. Execute `run_tests.sh`

Notes:
- The location of the tests depends on the type of DCGM package. If the installed package was a `.deb` or `.rpm` file then the location is `/usr/share/dcgm_tests`. If the package was `.tar.gz` then the location is relative to where it was uncompresssed.
- The test suite utilizes DCGM's python bindings. Python version 2 is the only supported version at this time.
- Running the tests as root is not required. However, some tests that require root permissions will not execute. For maximum test coverage running the tests as root is recommended.
- The entire test suite can take anywhere from 10 to >60 minutes depending on the speed of the machine, the number of gpus and the presence of additional NVIDIA hardware such as NVSwitches and NVLinks. On most systems the average time is ~30 minutes.
- Please do note file bug reports based on test failures. While great effort is made to ensure the tests are stable and resilient transient failures will occur from time to time.

## Reporting An Issue

Issues in DCGM can be reported by opening an [issue](https://github.com/NVIDIA/datacenter-gpu-manager/issues) in Github. Please include in reporting an issue:

- A description of the problem.
- Steps to reproduce the issue.
- If the issue cannot be reproduced, then please provide as much information as possible about conditions on the system that bring it about, how often it happens, and any other discernible patterns around when the bug occurs.
- Relevant configuration information, potentially including things like: whether or not this is in a container, VM, or bare metal environment, GPU SKU, number of GPUs on the system, operating system, driver version, GPU power settings, etc.
- DCGM logging for the error.
    - For `nv-hostengine`, you can start it with `-f /tmp/hostengine.log --log-level ERROR` to generate a log file with all error messages in `/tmp/hostengine.log`.
    - For the diagnostic, you can add `--debugLogFile /tmp/diag.log -d ERROR` to your command line in order to generate `/tmp/diag.log` with all error messages.
    - For any additional information about the command line, please see the documentation.
- The output of `nvidia-smi` and `nvidia-smi -q`.
- Full output of `dcgmi -v`.

The following template may be helpful:
```Environment: baremetal/docker container/VM
GPU SKU(s):
OS:
DRIVER:
GPU power settings (nvidia-smi -q -d POWER):
CPU(s):
RAM:
Topology (nvidia-smi topo -m):
```

### Reporting Security Issues

We ask that all community members and users of DCGM follow the standard Nvidia process for reporting security vulnerabilities. This process is documented at the [NVIDIA Product Security](https://www.nvidia.com/en-us/security/) website. 
Following the process will result in any needed CVE being created as well as appropriate notifications being communicated 
to the entire DCGM community.

Please refer to the policies listed there to answer questions related to reporting security issues.

## Tagging DCGM Releases

DCGM releases will be tagged once the release is finalized. The last commit will be the one that sets the release version, and we will then tag the releases. Releases tags will be the release version prepended with a v. For example, `v2.0.13`.

## License 

The source code for DCGM in this repository is licensed under [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0). Binary installer packages for DCGM are available for download from the [product page](https://developer.nvidia.com/dcgm) and are licensed under the [NVIDIA DCGM SLA](https://developer.download.nvidia.com/compute/DCGM/docs/NVIDIA_DCGM_EULA_Jan_2021.pdf).

## Additional Topics

- [NVIDIA DCGM Product Webpage](https://developer.nvidia.com/dcgm)
- [Contributing to DCGM](docs/contributing.md)
- [Coding Best Practices](docs/coding_best_practices.md)
- [Support and End-of-Life Information](docs/support_EOL.md)
