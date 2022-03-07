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
import ctypes
import dcgm_structs
import test_utils

## Device structures
class struct_c_CUdevice(ctypes.Structure):
    pass # opaque handle
c_CUdevice = ctypes.POINTER(struct_c_CUdevice)

# constants
CUDA_SUCCESS = 0
CU_DEVICE_ATTRIBUTE_PCI_BUS_ID = 33
CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID = 34
CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID = 50

_cudaLib = None
def _loadCuda():
    global _cudaLib
    if _cudaLib is None:
        _cudaLib = ctypes.CDLL("libcuda.so.1")
        cuInitFn = getattr(_cudaLib, "cuInit")
        assert CUDA_SUCCESS == cuInitFn(ctypes.c_uint(0))

def _unloadCuda():
    global _cudaLib
    _cudaLib = None

def cuDeviceGetCount():
    global _cudaLib
    _loadCuda()
    cuDeviceGetCountFn = getattr(_cudaLib, "cuDeviceGetCount")
    c_count = ctypes.c_uint(0)
    assert CUDA_SUCCESS == cuDeviceGetCountFn(ctypes.byref(c_count))
    _unloadCuda()
    return c_count.value

def cuDeviceGet(idx):
    global _cudaLib
    _loadCuda()
    cuDeviceGetFn = getattr(_cudaLib, "cuDeviceGet")
    c_dev = c_CUdevice()
    assert CUDA_SUCCESS == cuDeviceGetFn(ctypes.byref(c_dev), ctypes.c_uint(idx))
    _unloadCuda()
    return c_dev

def cuDeviceGetBusId(c_dev):
    global _cudaLib
    _loadCuda()
    cuDeviceGetAttributeFn = getattr(_cudaLib, "cuDeviceGetAttribute")
    c_domain = ctypes.c_uint()
    c_bus = ctypes.c_uint()
    c_device = ctypes.c_uint()
    assert CUDA_SUCCESS == cuDeviceGetAttributeFn(ctypes.byref(c_domain),
        ctypes.c_uint(CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID), c_dev)
    assert CUDA_SUCCESS == cuDeviceGetAttributeFn(ctypes.byref(c_bus),
        ctypes.c_uint(CU_DEVICE_ATTRIBUTE_PCI_BUS_ID), c_dev)
    assert CUDA_SUCCESS == cuDeviceGetAttributeFn(ctypes.byref(c_device),
        ctypes.c_uint(CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID), c_dev)
    _unloadCuda()
    return "%04x:%02x:%02x.0" % (c_domain.value, c_bus.value, c_device.value)

