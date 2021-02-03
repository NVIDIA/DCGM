# Docker image with DCGM build environment

## Structure
### patches
Contains patches for crosstool-ng packages

### scripts
Contains scripts to build 3rd party libraries for target architectures using cross-compilers

### scripts_host
Contains scripts to build libraries and tools that build host needs

### crosstool-ng config files
Crosstool-ng is used to generate cross-compilers and toolsets for x86_64/ppc64le