# CMake toolchain definition for aarch64-linux-gnu target

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(CMAKE_SYSROOT /opt/cross/aarch64-linux-gnu/sysroot)
set(CMAKE_FIND_ROOT_PATH /opt/cross/aarch64-linux-gnu)

set(CMAKE_C_COMPILER /opt/cross/bin/aarch64-linux-gnu-gcc)
set(CMAKE_C_COMPILER_TARGET aarch64-linux-gnu)

set(CMAKE_CXX_COMPILER /opt/cross/bin/aarch64-linux-gnu-g++)
set(CMAKE_CXX_COMPILER_TARGET aarch64-linux-gnu)

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM BOTH)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

set(CMAKE_SYSTEM_PREFIX_PATH /opt/cross;/opt/cross/aarch64-linux-gnu)

set(CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES 
    /opt/cross/bin/../lib/gcc/aarch64-linux-gnu/9.4.0/../../../../aarch64-linux-gnu/include/c++/9.4.0
    /opt/cross/bin/../lib/gcc/aarch64-linux-gnu/9.4.0/../../../../aarch64-linux-gnu/include/c++/9.4.0/aarch64-linux-gnu
    /opt/cross/bin/../lib/gcc/aarch64-linux-gnu/9.4.0/../../../../aarch64-linux-gnu/include/c++/9.4.0/backward
    /opt/cross/bin/../lib/gcc/aarch64-linux-gnu/9.4.0/include
    /opt/cross/bin/../lib/gcc/aarch64-linux-gnu/9.4.0/include-fixed
    /opt/cross/bin/../lib/gcc/aarch64-linux-gnu/9.4.0/../../../../aarch64-linux-gnu/include
    /opt/cross/bin/../aarch64-linux-gnu/sysroot/usr/include
)

set(CMAKE_C_STANDARD_INCLUDE_DIRECTORIES
    /opt/cross/bin/../lib/gcc/aarch64-linux-gnu/9.4.0/../../../../aarch64-linux-gnu/include/c++/9.4.0
    /opt/cross/bin/../lib/gcc/aarch64-linux-gnu/9.4.0/../../../../aarch64-linux-gnu/include/c++/9.4.0/aarch64-linux-gnu
    /opt/cross/bin/../lib/gcc/aarch64-linux-gnu/9.4.0/../../../../aarch64-linux-gnu/include/c++/9.4.0/backward
    /opt/cross/bin/../lib/gcc/aarch64-linux-gnu/9.4.0/include
    /opt/cross/bin/../lib/gcc/aarch64-linux-gnu/9.4.0/include-fixed
    /opt/cross/bin/../lib/gcc/aarch64-linux-gnu/9.4.0/../../../../aarch64-linux-gnu/include
    /opt/cross/bin/../aarch64-linux-gnu/sysroot/usr/include
)
