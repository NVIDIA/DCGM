# CMake toolchain definition for powerpc64le-linux-gnu target

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR ppc64le)

set(CMAKE_SYSROOT /opt/cross/powerpc64le-linux-gnu/sysroot)
set(CMAKE_FIND_ROOT_PATH /opt/cross/powerpc64le-linux-gnu)

set(CMAKE_C_COMPILER /opt/cross/bin/powerpc64le-linux-gnu-gcc)
set(CMAKE_C_COMPILER_TARGET powerpc64le-linux-gnu)

set(CMAKE_CXX_COMPILER /opt/cross/bin/powerpc64le-linux-gnu-g++)
set(CMAKE_CXX_COMPILER_TARGET powerpc64le-linux-gnu)

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM BOTH)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

set(CMAKE_SYSTEM_PREFIX_PATH /opt/cross;/opt/cross/powerpc64le-linux-gnu)

set(CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES
    /opt/cross/bin/../lib/gcc/powerpc64le-linux-gnu/9.4.0/../../../../powerpc64le-linux-gnu/include/c++/9.4.0
    /opt/cross/bin/../lib/gcc/powerpc64le-linux-gnu/9.4.0/../../../../powerpc64le-linux-gnu/include/c++/9.4.0/powerpc64le-linux-gnu
    /opt/cross/bin/../lib/gcc/powerpc64le-linux-gnu/9.4.0/../../../../powerpc64le-linux-gnu/include/c++/9.4.0/backward
    /opt/cross/bin/../lib/gcc/powerpc64le-linux-gnu/9.4.0/include
    /opt/cross/bin/../lib/gcc/powerpc64le-linux-gnu/9.4.0/include-fixed
    /opt/cross/bin/../lib/gcc/powerpc64le-linux-gnu/9.4.0/../../../../powerpc64le-linux-gnu/include
    /opt/cross/bin/../powerpc64le-linux-gnu/sysroot/usr/include
)

set(CMAKE_C_STANDARD_INCLUDE_DIRECTORIES
    /opt/cross/bin/../lib/gcc/powerpc64le-linux-gnu/9.4.0/../../../../powerpc64le-linux-gnu/include/c++/9.4.0
    /opt/cross/bin/../lib/gcc/powerpc64le-linux-gnu/9.4.0/../../../../powerpc64le-linux-gnu/include/c++/9.4.0/powerpc64le-linux-gnu
    /opt/cross/bin/../lib/gcc/powerpc64le-linux-gnu/9.4.0/../../../../powerpc64le-linux-gnu/include/c++/9.4.0/backward
    /opt/cross/bin/../lib/gcc/powerpc64le-linux-gnu/9.4.0/include
    /opt/cross/bin/../lib/gcc/powerpc64le-linux-gnu/9.4.0/include-fixed
    /opt/cross/bin/../lib/gcc/powerpc64le-linux-gnu/9.4.0/../../../../powerpc64le-linux-gnu/include
    /opt/cross/bin/../powerpc64le-linux-gnu/sysroot/usr/include
)
