# Docker image with DCGM build environment

## Structure

### dockerfiles
Dockerfiles describing how to generate container images related to dcgmbuild

### cmake
Toolchain files for cross-compiling sources

### crosstool-ng
Configuration files for crosstool-ng, a tool used to generate cross-compilers
and toolsets for x86_64/aarch64

### scripts/host
Contains scripts to build libraries and tools that the build host needs

### scripts/target
Contains scripts to build 3rd party libraries for target architectures using
cross-compilers
