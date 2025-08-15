# NVML Injection

This project is an injectable NVML that can be used for testing purposes.

[[_TOC_]]

## Library

The idea of NVML Injection is simply replaced the NVML functions by our fake implementations. Fake implementations use internal maps to store the return values of each function. We base on environment variable `NVML_INJECTION_MODE` to determine the target NVML shared library. When `NVML_INJECTION_MODE` is set, we will `dlopen` `libnvml_injection.so` which provides all the fake implementations. In addition, we also have environment variable `NVML_YAML_FILE` to indicate the YAML file that stores all the NVML functions returns of system in captured timeframe.

### Key

In YAML file format, we introduce key concept. The key of NVML functions is basically the suffix of function name. For example, the key of`nvmlDeviceGetTotalEccErrors` is `TotalEccErrors`. The reason behinds this is to support setter and getter. For example, as the key of `nvmlDeviceGetFanSpeed` and `nvmlDeviceSetFanSpeed` is the same, the setter can affect the getter.

### Extra Keys

If the NVML function has more than one input. We use extra keys to represent it. The type of extra key is `InjectionArgument`. It is an union of all existing NVML structures and basic types. For example, the second parameter of `nvmlDeviceGetFanSpeed_v2` is `fan` which will use `InjectionArgument` with `INJECTION_UINT` type as key in our internal map.

### Example of YAML Section

```
APIRestriction:
  0:
    FunctionReturn: 3
  1:
    FunctionReturn: 0
    ReturnValue: 0
```

In this example, `APIRestriction` is the key and 0, 1 are the extra keys. Each pair shows the function return (`nvmlReturn_t`) and the return value.

### Usage

To run diag in NVML injection sandbox:

```sh
env NVML_INJECTION_MODE=True NVML_YAML_FILE=<path-to-yaml-file> dcgmi diag -r software
```

To run host engine in NVML injection sandbox:

```sh
env NVML_INJECTION_MODE=True NVML_YAML_FILE=<path-to-yaml-file> ./apps/amd64/nv-hostengine -n -f -
```

Please note that the YAML file path depends on your system.

## generate_nvml_stubs.py

This script is used to auto-generate nvml YAML parser as well as fake implementation of nvml functions.

### Prerequisites

This script depends on `libclang.so`. We need to prepare it. The following provides steps to build llvm 18.1.2. For more information, Please see the [official llvm page](https://clang.llvm.org/get_started.html).

```sh
mkdir llvm && cd llvm
wget https://github.com/llvm/llvm-project/archive/refs/tags/llvmorg-18.1.2.tar.gz
tar zxvf llvmorg-18.1.2.tar.gz
cd llvm-project-llvmorg-18.1.2
mkdir build && cd build
cmake -DLLVM_ENABLE_PROJECTS=clang -DCMAKE_BUILD_TYPE=Release -G "Unix Makefiles" ../llvm
make
```

### Usage

When `sdk/nvidia/nvml/entry_points.h` or `sdk/nvidia/nvml/nvml.h` updated. Please re-run `generate_nvml_stubs.py` to ensure we can inject corresponding functions. As `nvml.h` is not a well-consistency header. It may have chance that you need to (slightly) modify `generate_nvml_stubs.py` to support new added functions, defines, etc.

From the dcgm/ directory:

```sh
env PYTHONPATH=$PYTHONPATH:/llvm/llvm-project-llvmorg-18.1.2/clang/bindings/python/ CLANG_LIBRARY_PATH=/llvm/llvm-project-llvmorg-18.1.2/build/lib/ python nvml-injection/scripts/generate_nvml_stubs.py
```

Please note that, the llvm library and binding path depend on your system and setup.

## nvml_api_recorder.py

This script is used to capture the current NVML state of system. It executes all NVML functions to retrieve all the return values and store them into a YAML file.

### Usage

It will automatically capture the environment and save YAML file to `_out_runLogs` if test failed.

If you want to capture it on purpose. You can use this command.

```sh
python3 main.py --capture-nvml-environment-to=<file_name>
```

## dcgm_nvml.py

Shipped PyNVML in our tests package. So that we can run NVML injection tests in as many platform as possible.

Basically, it is a mirror of https://pypi.org/project/nvidia-ml-py/. We just add Lint Skiper in the beginning of the copied file.

## Process for Adding New Functions

In this section, we will go through the process to show what to do to enable the NVML injection for new added functions.

1. Add new functions and definitions into `nvml.h` and `entry_points.h`

2. Run `generate_nvml_stubs.py`

   * In dcgm root folder
   * `python nvml-injection/scripts/generate_nvml_stubs.py`
     * Please see generate_nvml_stubs.py section for more information.

3. Update `nvml_api_recorder.py` so that it can capture the value from new added functions

   * Basically we need to write the python serializer to store return value into YAML file.

     * Take `nvmlDeviceGetPciInfo` as examples. From PyNVML, we know it returns `nvmlPciInfo_t()` which includes the following fields.

       ```python
       class nvmlPciInfo_t(_PrintableStructure):
           _fields_ = [
               # Moved to the new busId location below
               ('busIdLegacy', c_char * NVML_DEVICE_PCI_BUS_ID_BUFFER_V2_SIZE),
               ('domain', c_uint),
               ('bus', c_uint),
               ('device', c_uint),
               ('pciDeviceId', c_uint),

               # Added in 2.285
               ('pciSubSystemId', c_uint),
               # New busId replaced the long deprecated and reserved fields with a
               # field of the same size in 9.0
               ('busId', c_char * NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE),
           ]

       def nvmlDeviceGetPciInfo_v3(handle):
           c_info = nvmlPciInfo_t()
           fn = _nvmlGetFunctionPointer("nvmlDeviceGetPciInfo_v3")
           ret = fn(handle, byref(c_info))
           _nvmlCheckReturn(ret)
           return c_info
       ```

       As it only has device handler as input, we can add a new entry in `_nvml_device_attr_funcs`.

       ```python
       NVMLSimpleFunc("nvmlDeviceGetPciInfo", pci_info_parser),
       ```

       Finally, we need to write down structure YAML serializer `pci_info_parser`.

       ```python
       def pci_info_parser(value):
           return {
               "busIdLegacy": value.busIdLegacy,
               "domain": value.domain,
               "bus": value.bus,
               "device": value.device,
               "pciDeviceId": value.pciDeviceId,
               "pciSubSystemId": value.pciSubSystemId,
               "busId": value.busId,
           }
       ```

       Please note that if the field name is not the same as in C++ structure, please use C++ field name as the serialized key.

     * If to be added function uses reference to return its value. For example, `nvmlDeviceGetGpuFabricInfo`. We can write down the wrapper function in `nvml_api_recorder.py` to use return value for output parameters.

       ```python
       def nvmlDeviceGetGpuFabricInfo(device):
           import dcgm_nvml as pynvml
           c_fabricInfo = c_nvmlGpuFabricInfo_t()
           pynvml.nvmlDeviceGetGpuFabricInfo(device, byref(c_fabricInfo))
           return c_fabricInfo
       ```

       After this, we can follow the same way mentioned to add YAML serializer.

       ```python
       NVMLSimpleFunc("nvmlDeviceGetGpuFabricInfo", fabric_info_parser),
       ```

       ```python
       def fabric_info_parser(value):
           return {
               "clusterUuid": value.clusterUuid,
               "status": value.status,
               "cliqueId": value.partitionId,
               "state": value.state,
           }
       ```

       We should use `cliqueId` instead of `partitionId` as in C++ the field name is `cliqueId`.

     * If the function return basic type, for example, `nvmlDeviceGetComputeMode`, we can directly use `basic_type_value_parser` as YAML serializer.

     * If the function needs more than one inputs, please add entry in `_nvml_device_extra_key_attr_funcs`. It accepts another parameter to indicate the extra input. Take `nvmlDeviceGetClockInfo` as example. We will add an entry in `_nvml_device_extra_key_attr_funcs`.

       ```python
       NVMLExtraKeyFunc("nvmlDeviceGetClockInfo", range(NVML_CLOCK_COUNT), basic_type_value_parser),
       ```

       It has an extra element `range(NVML_CLOCK_COUNT)` which indicates the all possibility of second parameter (i.e., 0, 1, 2, 3).

   * We can skip updating `nvml_api_recorder.py` if PyNVML has not supported new added functions yet (i.e., we cannot find the function in PyNVML).

4. Update YAML file to include new key

   * If you are unable to access the machine that can produce informatiuon you want, you can also modify the YAML file directly to indicate the return values and use it for testing your code.

5. Run program with `NVML_INJECTION_MODE` and `NVML_YAML_FILE` for testing

## Add Tests Using NVML Injection

### Mock NVML APIs

As an overview for NVML injection. Its goal is to let us mock the return value of NVML calls. The mocked value can be either loading from the NVML injection YAML file (e.g., `run_with_injection_nvml_using_specific_sku`), or from those that we "inject" using the `dcgmInjectNvmlDevice` API. After loading the NVML injection YAML file, the NVML APIs will return values specified in this YAML file. Once we inject values using `dcgmInjectNvmlDevice`, the returned values will be those from the injected configuration. This means, manual injection has higher priority than the YAML file.

To correctly set up the `dcgmInjectNvmlDevice`, we need to refer to the `InjectionArgument.h` to see the appropriate value type and value name for the mocking target.
For example, we can see the type (i.e., `INJECTION_NVLINKERRORCOUNTER`) and value name (i.e., `NvLinkErrorCounter`) of `nvmlNvLinkErrorCounter_t` in the segment.

```c++
InjectionArgument(nvmlNvLinkErrorCounter_t NvLinkErrorCounter)
    : m_type(INJECTION_NVLINKERRORCOUNTER)
{
    memset(&m_value, 0, sizeof(m_value));
    m_value.NvLinkErrorCounter = NvLinkErrorCounter;
}
```

With the above information, we then can mock `nvmlDeviceGetNvLinkErrorCounter` by the following snippet

```python
def mock_nvlink_error_counter(handle, gpuId, linkId, counterType, nvmlRet, value):
    # To set the return value, since this function returns `unsigned long long *counterValue`,
    # we can refer to InjectionArgument.h to know that its type is INJECTION_ULONG_LONG and the value name is ULongLong.
    injectedRet = nvml_injection.c_injectNvmlRet_t()
    injectedRet.nvmlRet = nvmlRet
    injectedRet.values[0].type = nvml_injection_structs.c_injectionArgType_t.INJECTION_ULONG_LONG
    injectedRet.values[0].value.ULongLong = value
    injectedRet.valueCount = 1

    # Set the expected extra keys (parameters) for this function.
    # Since nvmlDeviceGetNvLinkErrorCounter has two parameters: unsigned int link and nvmlNvLinkErrorCounter_t counter.
    # We need to create an array of size 2 for these extra keys and configure their types and expected arguments.
    # Refer to InjectionArgument.h for the type and name details.
    extraKeysType = nvml_injection_structs.c_injectNvmlVal_t * 2
    extraKeys = extraKeysType()
    extraKeys[0].type = nvml_injection_structs.c_injectionArgType_t.INJECTION_UINT
    extraKeys[0].value.UInt = linkId
    extraKeys[1].type = nvml_injection_structs.c_injectionArgType_t.INJECTION_NVLINKERRORCOUNTER
    extraKeys[1].value.NvLinkErrorCounter = counterType

    # Call the dcgmInjectNvmlDevice function with the target GPU and the key "NvLinkErrorCounter" along with the parameters and the given output.
    # After calling dcgmInjectNvmlDevice,
    # the return value of the newly called nvmlDeviceGetNvLinkErrorCounter with the provided gpuId, link, and counter will be specified by injectedRet.
    ret = dcgm_agent_internal.dcgmInjectNvmlDevice(handle, gpuId, "NvLinkErrorCounter", extraKeys, 2, injectedRet)
    assert (ret == dcgm_structs.DCGM_ST_OK)
```

Please note that:

* calling `dcgmInjectNvmlDevice` multiple times on the same API will override the injected value. We need to mock and test each field individually.
* The injection target should not include the Get prefix. i.e., use `NvLinkErrorCounter` rather than `GetNvLinkErrorCounter`. This enhances flexibility when calling a setter NVML API in our codebase. For example, as the target (i.e., `FanSpeed`) of `nvmlDeviceGetFanSpeed` and `nvmlDeviceSetFanSpeed` is the same, the setter can affect the getter.

### Inject NVML Fields (`nvmlDeviceGetFieldValues`)

To inject a value into a specific field of `nvmlDeviceGetFieldValues`, we can use the `inject_nvml_value` function from `testing/python3/dcgm_field_injection_helpers.py`. This function automatically converts the DCGM field to an NVML field and injects the value accordingly.

For example, the following code snippet injects the value 20 into `NVML_FI_DEV_REMAPPED_FAILURE` for `gpuIds[0]`.

```python
    fieldId = dcgm_fields.DCGM_FI_DEV_ROW_REMAP_FAILURE
    injected_value = 20 # random non-zero number
    inject_nvml_value(handle, gpuIds[0], fieldId, injected_value, 0)
```

### More Examples

Please refer `testing/python3/tests/test_nvml_injection.py`.
