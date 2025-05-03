#!/usr/bin/env python3
#
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
#

import argparse
import sys
import os
import re
import clang.cindex
from clang.cindex import Config
import inspect
import logging

# Regular expression for pulling out the different pieces of the nvml entry points
# It will match something in the form of:
# funcname, tsapiFuncname, (argument list), "(argument type matching)", arg1[, arg2, ...])
# We place funcname, (argument list), and arg1[, arg2, ...] into groups for use later
preg = re.compile(r"(nvml\w+),[^)]+(\([^)]+\)),\s+\"[^\"]+\",\s+([^)]+)")

MAX_NVML_ARGS = 20
INJECTION_ARG_COUNT_STR = 'InjectionArgCount'
NVML_RET = 'nvmlReturn_t'

# Globals
g_key_to_function = {}

# Generated file names
STUB_PATH = 'nvml-injection/src/nvml_generated_stubs.cpp'
INJECTION_ARGUMENT_HEADER = 'InjectionArgument.h'
INJECTION_STRUCTS_NAME = 'nvml_injection_structs.h'
INJECTION_STRUCTS_PY_NAME = 'nvml_injection_structs.py'
INJECTION_STRUCTS_PATH = 'nvml-injection/include/%s' % INJECTION_STRUCTS_NAME
INJECTION_ARGUMENT_PATH = 'nvml-injection/include/%s' % INJECTION_ARGUMENT_HEADER
INJECTION_CPP_PATH = 'nvml-injection/src/InjectionArgument.cpp'
NVML_RETURN_DESERIALIZER_CPP_PATH = 'nvml-injection/src/NvmlReturnDeserializer.cpp'
NVML_RETURN_DESERIALIZER_HEADER_PATH = 'nvml-injection/include/NvmlReturnDeserializer.h'
KEY_LIST_PATH = 'nvml-injection/src/InjectionKeys.cpp'
KEY_LIST_HEADER_PATH = 'nvml-injection/include/InjectionKeys.h'
LINUX_DEFS_PATH = 'nvml-injection/src/nvml-injection.linux_defs'

skip_functions = [ 'nvmlGetBlacklistDeviceCount', 'nvmlGetBlacklistDeviceInfoByIndex' ]

uint_aliases = [
    'nvmlBusType_t',
    'nvmlVgpuTypeId_t',
    'nvmlVgpuInstance_t',
    'nvmlBusType_t',
    'nvmlDeviceArchitecture_t',
    'nvmlPowerSource_t',
    'nvmlAffinityScope_t',
]

def get_version(funcname):
    if funcname and funcname[-3:-1] == '_v':
        return int(funcname[-1])

    return 0

class AllFunctionTypes(object):
    def __init__(self):
        self.all_func_declarations = []
        self.all_argument_type_strs = []
        self.funcname_to_func_type = {}
        self.arg_types_to_func_type = {}

    def AddFunctionType(self, funcname, funcinfo):
        arg_type_str = funcinfo.GetArgumentTypesAsString()
        if arg_type_str not in self.all_argument_type_strs:
            func_declaration = "typedef nvmlReturn_t (*%s_f)%s;" % (funcname, funcinfo.GetArgumentList())
            func_type = "%s_f" % funcname
            self.all_func_declarations.append(func_declaration)
            self.arg_types_to_func_type[arg_type_str] = func_type
            self.funcname_to_func_type[funcname] = func_type
            self.all_argument_type_strs.append(arg_type_str)
        else:
            self.funcname_to_func_type[funcname] = self.arg_types_to_func_type[arg_type_str]

    def GetAllFunctionDeclarations(self):
        return self.all_func_declarations

    def GetFunctionType(self, funcname):
        return self.funcname_to_func_type[funcname]

class AllFunctions(object):
    def __init__(self):
        self.func_dict = {}
        self.versioned_funcs = {}

    def AddFunction(self, funcinfo):
        if funcinfo.GetName() not in skip_functions:
            funcname = funcinfo.GetName()
            self.func_dict[funcname] = funcinfo

            version = get_version(funcname)
            if version > 0:
                self.versioned_funcs[funcname] = version

    def GetFunctionDict(self):
        return self.func_dict

    def RemoveEarlierVersions(self):
        for funcname in self.versioned_funcs:
            version = self.versioned_funcs[funcname]
            without_version = funcname[:-3]
            for i in range(1, version):
                try:
                    if i == 1:
                        del self.func_dict[without_version]
                    else:
                        to_remove = "%s_v%d" % (without_version, i)
                        del self.func_dict[to_remove]
                except KeyError:
                    pass

        self.versioned_funcs = {}


class FunctionInfo(object):
    def __init__(self, funcname, arg_list, arg_names):
        self.funcname  = funcname.strip()
        self.arg_list  = self.CleanArgList(arg_list)
        self.arg_names = arg_names
        self.arg_types = get_argument_types_from_argument_list(arg_list)

    def CleanArgList(self, arg_list):
        # Make sure '*' is always 'T *' and not 'T* ' for our formatter
        tokens = arg_list.split('*')
        new_list = ''
        for token in tokens:
            if not new_list:
                new_list = token
            else:
                if token[0] == ' ':
                    new_list = new_list + '*%s' % token[1:]
                else:
                    new_list = new_list + '*%s' % token

            if token[-1] != ' ':
                new_list += ' '

        return new_list

    def GetName(self):
        return self.funcname

    def GetArgumentList(self):
        return self.arg_list

    def GetArgumentNames(self):
        return self.arg_names

    def GetArgumentTypes(self):
        return self.arg_types

    def GetArgumentTypesAsString(self):
        type_str = ''
        for arg_type in self.arg_types:
            if type_str == '':
                type_str = str(arg_type)
            else:
                type_str = type_str + ",%s" % str(arg_type)
        return type_str


def get_true_arg_type(arg_type):
    if is_pointer_type(arg_type):
        if arg_type[:-2] in uint_aliases:
            return 'unsigned int *'
    elif arg_type in uint_aliases:
        return 'unsigned int'

    return arg_type

def remove_extra_spaces(text):
    while text.find('  ') != -1:
        text = text.replace('  ', ' ')
    return text

def get_function_signature(entry_point, first):
    # Remove all line breaks, remove the extra whitespace on the ends, and then get
    # get rid of the parenthesis around the string
    # We are left with something in the form of:
    # funcname, tsapiFuncname, (argument list), "(argument type matching)", arg1, arg2, ...)
    entry_point = entry_point.replace('\n', ' ').strip()[1:-1]
    m = preg.search(entry_point)
    if m:
        return remove_extra_spaces(m.group(1)), remove_extra_spaces(m.group(2)), remove_extra_spaces(m.group(3))
    else:
        if entry_point == "include \"nvml.h":
            pass
        # Ignore errors on the first token because it is everything from before the first entry point
        elif not first:
            print("no match found in entry point = '%s'" % entry_point)
        return None, None, None

def print_body_line(line, file, extra_indent):
    indent = "    "
    for _ in range(0, extra_indent):
        indent += "    "
    file.write("%s%s\n" % (indent, line))

def add_function_type(function_type_dict, funcname, arg_list, function_types):
    func_declaration = "typedef nvmlReturn_t (*%s_f)%s;" % (funcname, arg_list)
    function_types.append(func_declaration)

keyPrefixes = [
    'nvmlDeviceGetHandleBy',
    'nvmlDeviceGet',
    'nvmlSystemGet',
    'nvmlDeviceSet',
    'nvmlUnitGet',
    'nvmlUnitSet',
    'nvmlVgpuTypeGet',
    'nvmlVgpuInstanceGet',
    'nvmlVgpuInstanceSet',
    'nvmlGet',
    'nvmlSet',
    'nvmlGpuInstanceGet',
    'nvmlComputeInstanceGet',
    'nvmlDeviceClear',
    'nvmlDeviceFreeze',
    'nvmlDeviceModify',
    'nvmlDeviceQuery',
    'nvmlDeviceCreate',
    'nvmlDeviceReset',
    'nvmlDeviceIs',
    'nvmlDevice',
]

Setter = [
    'nvmlDeviceSet',
    'nvmlUnitSet',
    'nvmlVgpuInstanceSet',
    'nvmlSet',
    'nvmlDeviceClear',
    'nvmlDeviceFreeze',
    'nvmlDeviceModify',
    'nvmlDeviceCreate',
    'nvmlDeviceReset',
    'nvmlDeviceRemove',
    'nvmlGpuInstanceDestroy',
    'nvmlComputeInstanceDestroy',
    'nvmlGpuInstanceCreateComputeInstance',
]

HandlerGetter = [
    'nvmlDeviceGetHandleBy',
    'nvmlDeviceGetDeviceHandleFromMigDeviceHandle',
    'nvmlDeviceGetGpuInstanceById',
    'nvmlDeviceGetMigDeviceHandleByIndex',
    'nvmlGpuInstanceGetComputeInstanceById',
]

gpmPrefix = 'nvmlGpm'

def get_basic_type():
    return {
        "int",
        "unsigned int",
        "char",
        "unsigned char",
        "long",
        "long long",
        "unsigned long",
        "unsigned long long",
        "short",
        "unsigned short",
        "double",
    }

def is_basic_type(type_name):
    return type_name in get_basic_type() or type_name == "unsigned"

def is_basic_ptr_type(type_name):
    basic = {
        "int *",
        "unsigned *",
        "unsigned int *",
        "char *",
        "unsigned char *",
        "long *",
        "long long *",
        "unsigned long *",
        "unsigned long long *",
        "short *",
        "unsigned short *",
        "double *",
    }
    return type_name in basic

def remove_ptr_if_any(type_name):
    if type_name[-1] == '*':
        return type_name[:-1].strip()
    return type_name

def is_enum_type(type_name, all_enum):
    return type_name in all_enum

def get_suffix_if_match(funcname, prefix):
    if funcname[:len(prefix)] == prefix:
        key = funcname[len(prefix):]
        return key

    return None

def get_function_info_from_name(funcname):
    key = None
    version = 1

    for prefix in keyPrefixes:
        key = get_suffix_if_match(funcname, prefix)
        if key:
            break

    if not key:
        key = get_suffix_if_match(funcname, gpmPrefix)
        if key:
            if key[-3:] == 'Get':
                key = key[:-3]

    # Check for version at the end
    if key:
        if key[-3:-1] == '_v':
            version = int(key[-1])
            key = key[:-3]

        if key in g_key_to_function:
            func_list = "%s, %s" % (g_key_to_function[key], funcname)
            g_key_to_function[key] = func_list
        else:
            g_key_to_function[key] = funcname

    return key, version

def print_generator_source_info(out_file, indent):
    frame = inspect.currentframe().f_back
    function_name = frame.f_code.co_name
    if indent == 0:
        out_file.write(f"// The following snippet is generated from {function_name}\n")
    else:
        print_body_line(f"// The following snippet is generated from {function_name}", out_file, indent - 1)

def check_and_write_get_string_body(stub_file, key, arg_types, arg_names):
    if is_pointer(arg_types[0]):
        return False

    # Only handle the following two cases
    # 1. Direct string output (e.g. nvmlDeviceGetVbiosVersion)
    # 2. Has another extra key and direct string output (e.g. nvmlDeviceGetInforomVersion)
    if len(arg_types) != 3 and len(arg_types) != 4:
        return False

    if len(arg_types) == 3:
        if arg_types[1] != CHAR:
            return False
        if arg_types[2] != UINT and arg_types[2] != UINT_PTR:
            return False

    if len(arg_types) == 4:
        if arg_types[2] != CHAR:
            return False
        if arg_types[3] != UINT and arg_types[3] != UINT_PTR:
            return False

    # InjectionNvml::GetString will return a std::string associated with two keys
    print_generator_source_info(stub_file, 2)
    print_body_line("InjectionArgument arg(%s);" % arg_names[0], stub_file, 1)
    if len(arg_types) == 3:
        print_body_line("auto [tmpNvmlRet, buf] = injectedNvml->GetString(arg, \"%s\");" % (key), stub_file, 1)
    if len(arg_types) == 4:
        print_body_line("auto [tmpNvmlRet, buf] = injectedNvml->GetString(arg, \"%s\", InjectionArgument(%s));" % (key, arg_names[1]), stub_file, 1)

    print_body_line("if (tmpNvmlRet != NVML_SUCCESS)", stub_file, 1)
    print_body_line("{", stub_file, 1)
    print_body_line("return tmpNvmlRet;", stub_file, 2)
    print_body_line("}", stub_file, 1)

    if len(arg_types) == 3:
        if arg_types[2] == UINT:
            print_body_line("snprintf(%s, %s, \"%s\", buf.c_str());" % (arg_names[1], arg_names[2], '%s'), stub_file, 1)
        elif arg_types[2] == UINT_PTR:
            print_body_line("snprintf(%s, *%s, \"%s\", buf.c_str());" % (arg_names[1], arg_names[2], '%s'), stub_file, 1)
    if len(arg_types) == 4:
        if arg_types[3] == UINT:
            print_body_line("snprintf(%s, %s, \"%s\", buf.c_str());" % (arg_names[2], arg_names[3], '%s'), stub_file, 1)
        elif arg_types[3] == UINT_PTR:
            print_body_line("snprintf(%s, *%s, \"%s\", buf.c_str());" % (arg_names[2], arg_names[3], '%s'), stub_file, 1)

    return True

def is_pointer(arg_type):
    if arg_type[-1] == '*':
        return True

    return False

CONST_CHAR = 'const char *'
CHAR = 'char *'
NVML_DEVICE = 'nvmlDevice_t'
NVML_DEVICE_PTR = 'nvmlDevice_t *'
UINT_PTR = 'unsigned int *'
CLOCKTYPE = 'nvmlClockType_t'
UINT = 'unsigned int'
UINT_PTR = 'unsigned int *'
VGPUTYPEID = 'nvmlVgpuTypeId_t'
UNIT = 'nvmlUnit_t'
VGPU_INSTANCE = 'nvmlVgpuInstance_t'

def print_ungenerated_function(funcname, arg_types):
    arg_type_string = ''
    for arg_type in arg_types:
        if len(arg_type_string):
            arg_type_string = arg_type_string + ',%s' % arg_type
        else:
            arg_type_string = arg_type
    #print("Not generated: %s with (%d) arg_types: %s" % (funcname, len(arg_types), arg_type_string))

def write_auto_generate_c_file_header(out_file):
    copyright_notice ='''/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */\n
'''
    auto_generated_notice = '/*\n * NOTE: This code is auto-generated by generate_nvml_stubs.py\n * DO NOT EDIT MANUALLY\n */\n\n\n'
    out_file.write(copyright_notice)
    out_file.write(auto_generated_notice)

def write_auto_generate_py_file_header(out_file):
    copyright_notice ='''#
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
#\n
'''
    auto_generated_notice = '# NOTE: This code is auto-generated by generate_nvml_stubs.py\n# DO NOT EDIT MANUALLY\n\n\n'
    out_file.write(copyright_notice)
    out_file.write(auto_generated_notice)

def generate_getter_functions(stub_file, funcname, arg_list, arg_types, arg_names, justifyLen):
    generated = True
    key, version = get_function_info_from_name(funcname)

    if funcname == "nvmlDeviceGetFieldValues":
        print_generator_source_info(stub_file, 2)
        print_body_line("if (%s == nullptr)" % arg_names[2], stub_file, 1)
        print_body_line("{", stub_file, 1)
        print_body_line("return NVML_ERROR_INVALID_ARGUMENT;", stub_file, 2)
        print_body_line("}\n", stub_file, 1)
        print_body_line("injectedNvml->GetFieldValues(%s, %s, %s);" % (arg_names[0], arg_names[1], arg_names[2]), stub_file, 1)
    elif len(arg_types) == 2 and arg_types[1] == NVML_DEVICE_PTR:
        # Device handler getter (e.g. nvmlDeviceGetHandleByIndex)
        print_generator_source_info(stub_file, 2)
        print_body_line("InjectionArgument identifier(%s);" % arg_names[0], stub_file, 1)
        print_body_line("*%s = injectedNvml->GetNvmlDevice(identifier, \"%s\");" % (arg_names[1], key), stub_file, 1)
    elif check_and_write_get_string_body(stub_file, key, arg_types, arg_names):
        pass
    elif len(arg_types) == 1 and is_pointer(arg_types[0]):
        # Global attribute getter (e.g. nvmlSystemGetCudaDriverVersion)
        print_generator_source_info(stub_file, 2)
        print_body_line("InjectionArgument arg(%s);" % arg_names[0], stub_file, 1)
        print_body_line("arg.SetValueFrom(injectedNvml->ObjectlessGet(\"%s\"));" % (key), stub_file, 1)
    elif len(arg_types) == 2 and arg_types[0] == CHAR and arg_types[1] == UINT:
        # Global string attribute getter (e.g. nvmlSystemGetDriverVersion)
        print_generator_source_info(stub_file, 2)
        lhand = "std::string str"
        print_body_line("%s = injectedNvml->ObjectlessGet(\"%s\").AsString();" % (lhand.ljust(justifyLen), key), stub_file, 1)
        print_body_line("snprintf(%s, %s, \"%s\", str.c_str());" % (arg_names[0], arg_names[1], "%s"), stub_file, 1)
    else:
        print_ungenerated_function(funcname, arg_types)
        generated = False


    if generated:
        print_body_line('return NVML_SUCCESS;', stub_file, 1)

    return generated

def is_getter(funcname):
    if funcname.find("Get") != -1:
        return True

    return False

def is_setter(funcname):
    if funcname.find("Set") != -1:
        return True

    return False

def generate_setter_functions(stub_file, funcname, arg_list, arg_types, arg_names):
    generated = True
    key, version = get_function_info_from_name(funcname)

    if len(arg_types) == 2 and arg_types[0] == NVML_DEVICE:
        # Device setter (e.g. nvmlDeviceSetAutoBoostedClocksEnabled)
        print_generator_source_info(stub_file, 2)
        print_body_line("InjectionArgument value(%s);" % (arg_names[1].strip()), stub_file, 1)
        print_body_line("return injectedNvml->DeviceSet(%s, \"%s\", {}, NvmlFuncReturn(NVML_SUCCESS, value));" % (arg_names[0], key), stub_file, 1)
    elif len(arg_types) == 3 and arg_types[0] == NVML_DEVICE:
        if funcname == 'nvmlDeviceSetFanSpeed_v2' or funcname == 'nvmlDeviceSetTemperatureThreshold':
            print_generator_source_info(stub_file, 2)
            print_body_line("InjectionArgument extraKey(%s);" % arg_names[1], stub_file, 1)
            print_body_line("InjectionArgument value(%s);" % arg_names[2], stub_file, 1)
            print_body_line("return injectedNvml->DeviceSet(%s, \"%s\", {extraKey}, NvmlFuncReturn(NVML_SUCCESS, value));" % (arg_names[0], key), stub_file, 1)
        else:
            # Device setter with two values (e.g nvmlDeviceSetGpuLockedClocks)
            print_generator_source_info(stub_file, 2)
            print_body_line("std::vector<InjectionArgument> preparedValues;", stub_file, 1)
            print_body_line("preparedValues.push_back(InjectionArgument(%s));" % arg_names[1].strip(), stub_file, 1)
            print_body_line("preparedValues.push_back(InjectionArgument(%s));" % arg_names[2].strip(), stub_file, 1)
            print_body_line("CompoundValue cv(preparedValues);", stub_file, 1)
            print_body_line("return injectedNvml->DeviceSet(%s, \"%s\", {}, NvmlFuncReturn(NVML_SUCCESS, cv));" % (arg_names[0], key), stub_file, 1)
    else:
        generated = False
        print_ungenerated_function(funcname, arg_types)

    return generated

cant_generate = [
    'nvmlDeviceGetVgpuMetadata',
    'nvmlDeviceSetDriverModel', # Windows only
    'nvmlDeviceSetMigMode', # Too unique - requires specific checks
]
def generate_injection_function(stub_file, funcname, arg_list, arg_types, arg_names, justifyLen):
    if funcname in cant_generate:
        return False

    generated = False

    if is_getter(funcname):
        generated = generate_getter_functions(stub_file, funcname, arg_list, arg_types, arg_names, justifyLen)
    elif is_setter(funcname):
        generated = generate_setter_functions(stub_file, funcname, arg_list, arg_types, arg_names)
    else:
        print_ungenerated_function(funcname, arg_types)

    return generated

def write_function_definition_start(fileHandle, funcname, arg_list):
    first_part = "%s %s" % (NVML_RET, funcname)
    line = "%s%s" % (first_part, arg_list)
    line = remove_extra_spaces(line).strip()
    if len(line) <= 120:
        fileHandle.write("%s\n{\n" % line)
    else:
        tokens = arg_list.split(',')
        count = len(tokens)
        fileHandle.write("%s%s,\n" % (first_part, tokens[0]))
        if count > 2:
            index = 1
            while index < count - 1:
                fileHandle.write("%s%s,\n" % (" ".ljust(len(first_part)), tokens[index]))
                index = index + 1
        fileHandle.write("%s %s\n{\n" % (" ".ljust(len(first_part)), tokens[-1].strip()))

def write_function(stub_file, funcinfo, all_functypes):
    funcname = funcinfo.GetName()
    arg_list = funcinfo.GetArgumentList()
    arg_names = funcinfo.GetArgumentNames()
    arg_types = funcinfo.GetArgumentTypes()
    key, version = get_function_info_from_name(funcname)

    generated = False
    write_function_definition_start(stub_file, funcname, arg_list)

    # Write the body
    print_generator_source_info(stub_file, 1)
    print_body_line("if (GLOBAL_PASS_THROUGH_MODE)", stub_file, 0)
    print_body_line("{", stub_file, 0)
    print_body_line("auto PassThruNvml = PassThruNvml::GetInstance();", stub_file, 1)
    print_body_line("if (PassThruNvml->IsLoaded(__func__) == false)", stub_file, 1)
    print_body_line("{", stub_file, 1)
    print_body_line("PassThruNvml->LoadFunction(__func__);", stub_file, 2)
    print_body_line("}", stub_file, 1)
    print_body_line("return NVML_ERROR_NOT_SUPPORTED;", stub_file, 1)
    print_body_line("}", stub_file, 0)
    print_body_line("else", stub_file, 0)
    print_body_line("{", stub_file, 0)

    unstripped_arguments = arg_names.split(",")
    arguments = []
    for arg in unstripped_arguments:
        arguments.append(arg.strip())

    if len(arg_types) != len(arguments):
        print(f'failed to generate {funcname} due to unexpected args')
        return False

    print_generator_source_info(stub_file, 2)
    start = "auto *injectedNvml"
    print_body_line("%s = InjectedNvml::GetInstance();" % start, stub_file, 1)
    print_body_line("if (!injectedNvml)", stub_file, 1)
    print_body_line("{", stub_file, 1)
    print_body_line("return NVML_ERROR_UNINITIALIZED;", stub_file, 2)
    print_body_line("}", stub_file, 1)
    print_body_line("injectedNvml->AddFuncCallCount(\"%s\");" % funcname, stub_file, 1)
    if generate_injection_function(stub_file, funcname, arg_list, arg_types, arguments, len(start)):
        generated = True
    else:
        # General case, we put all non-pointer args into args and treat all pointer args as values
        print_generator_source_info(stub_file, 2)
        print_body_line("std::vector<InjectionArgument> args;", stub_file, 1)
        print_body_line("std::vector<InjectionArgument> preparedValues;", stub_file, 1)
        for i in range(len(arguments)):
            argument = arguments[i]
            if is_pointer_type(arg_types[i]):
                print_body_line("preparedValues.push_back(InjectionArgument(%s));" % argument.strip(), stub_file, 1)
            else:
                if arg_types[i] != "unsigned int" or not arg_is_count(argument):
                    # We should skip the buffer length as it is used for buffer validation not the actual function arg
                    # e.g. in nvmlDeviceGetInforomImageVersion, we should not push length in args
                    print_body_line("args.push_back(InjectionArgument(%s));" % argument.strip(), stub_file, 1)

        stub_file.write("\n")
        print_body_line("if (injectedNvml->IsGetter(__func__))", stub_file, 1)
        print_body_line("{", stub_file, 1)
        print_body_line(f"return injectedNvml->GetWrapper(__func__, \"{key}\", args, preparedValues);", stub_file, 2)
        print_body_line("}", stub_file, 1)
        print_body_line("else", stub_file, 1)
        print_body_line("{", stub_file, 1)
        print_body_line(f"return injectedNvml->SetWrapper(__func__, \"{key}\", args, preparedValues);", stub_file, 2)
        print_body_line("}", stub_file, 1)
    print_body_line("}", stub_file, 0)
    print_body_line("return NVML_SUCCESS;", stub_file, 0)

    # Write the end of the function
    stub_file.write("}\n\n")
    return generated

def write_stub_file_header(stub_file):
    write_auto_generate_c_file_header(stub_file)
    stub_file.write('#pragma GCC diagnostic push\n#pragma GCC diagnostic ignored "-Wunused-parameter"\n\n')
    stub_file.write('// clang-format off\n')
    stub_file.write("#include \"InjectedNvml.h\"\n")
    stub_file.write("#include \"nvml.h\"\n")
    stub_file.write("#include \"PassThruNvml.h\"\n\n")
    stub_file.write("#ifdef __cplusplus\n")
    stub_file.write("extern \"C\"\n{\n#endif\n\n")
    stub_file.write("bool GLOBAL_PASS_THROUGH_MODE = false;\n\n")

def get_argument_types_from_argument_list(arg_list):
    argument_types = []
    arg_list = arg_list.strip()
    if arg_list[0] == '(':
        arg_list = arg_list[1:]
    if arg_list[-1] == ')':
        arg_list = arg_list[:-1]
    arguments = arg_list.split(',')

    for argument in arguments:
        words = argument.strip().split(' ')
        arg_type = words[0]
        if len(words) == 2:
            arg_name = words[1].strip()[0]
        else:
            for i in range(1, len(words)-1):
                arg_type += ' %s' % words[i]
            arg_name = words[len(words)-1]

        if arg_name[0] == '*':
            arg_type += ' *'
        elif arg_type[-1] == '*' and arg_type[-2] != ' ':
            arg_type = arg_type[:-1] + ' *'

        argument_types.append(arg_type)

    return argument_types

def get_duplicate_types():
    return {
        # nvmlProcessInfo_v2_t and nvmlProcessInfo_t are the same
        "nvmlProcessInfo_t",
        # unsigned and unsigned int are the same
        "unsigned",
    }

def build_argument_type_list(arg_list, all_argument_types):
    argument_types = get_argument_types_from_argument_list(arg_list)
    duplicate_types = get_duplicate_types()

    for arg_type in argument_types:
        check_type = get_true_arg_type(arg_type)
        # skip nvmlProcessInfo_t as nvmlProcessInfo_v2_t and nvmlProcessInfo_t are the same
        if remove_ptr_if_any(check_type) in duplicate_types:
            continue
        if check_type not in all_argument_types:
            all_argument_types.append(check_type)
        # also build non-pointer version in InjectionArgument for easier use.
        # but skip const char and const nvmlGpuInstancePlacement_t as const member in union is strange
        if remove_ptr_if_any(check_type) not in all_argument_types and\
                remove_ptr_if_any(check_type) != "const char" and remove_ptr_if_any(check_type) != "const nvmlGpuInstancePlacement_t":
            all_argument_types.append(remove_ptr_if_any(check_type))

    return argument_types

def is_pointer_type(arg_type):
    return arg_type[-2:] == ' *'

def is_nvml_enum(arg_type):
    return arg_type[:4] == 'nvml'

def ends_with_t(arg_type):
    return arg_type[-2:] == '_t'

def get_enum_name(arg_type, const=False):
    ret = ''
    ptr_ret = ''
    prefix = 'INJECTION_'
    if const:
        prefix += 'CONST_'

    if is_nvml_enum(arg_type):
        if ends_with_t(arg_type):
            ret = '%s%s' % (prefix, arg_type[4:-2].upper())
        else:
            ret = '%s%s' % (prefix, arg_type[4:].upper())
    else:
        words = arg_type.strip().split(' ')
        if len(words) == 1:
            ret = '%s%s' % (prefix, arg_type.upper())
        else:
            ret = prefix
            for word in words:
                if word == 'unsigned':
                    ret += 'U'
                else:
                    ret += '%s_' % word.upper()
            ret = ret[:-1]

    ptr_ret = ret + "_PTR"
    return ret, ptr_ret

def print_memcpy(fileHandle, indentLevel, destName, srcName, destIsPtr, srcIsPtr):
    src = ''
    dst = ''
    sizeof = ''

    if destIsPtr:
        dst = 'm_value.%s' % destName
    else:
        dst = '&m_value.%s' % destName

    if srcIsPtr:
        src = 'other.m_value.%s' % srcName
    else:
        src = '&other.m_value.%s' % srcName

    sizeof = f'sizeof(*{dst})'
    if srcIsPtr and destIsPtr:
        sizeof = f'sizeof(*{dst}) * (other.m_isArray ? other.m_arrLen : 1)'

    line = 'memcpy(%s, %s, %s);' % (dst, src, sizeof)
    print_body_line(line, fileHandle, indentLevel)

def write_string_case_entry(injectionCpp):
    print_generator_source_info(injectionCpp, 2)
    print_body_line('case INJECTION_STRING:', injectionCpp, 1)
    print_body_line('{', injectionCpp, 1)
    print_body_line('if (other.m_type == INJECTION_STRING)', injectionCpp, 2)
    print_body_line('{', injectionCpp, 2)
    print_body_line('this->m_str = other.m_str;', injectionCpp, 3)
    print_body_line('set         = true;', injectionCpp, 3)
    print_body_line('}', injectionCpp, 2)
    print_body_line('else if (other.m_type == INJECTION_CHAR_PTR && other.m_value.Str != nullptr)', injectionCpp, 2)
    print_body_line('{', injectionCpp, 2)
    print_body_line('this->m_str = other.m_value.Str;', injectionCpp, 3)
    print_body_line('set         = true;', injectionCpp, 3)
    print_body_line('}', injectionCpp, 2)
    print_body_line('break;', injectionCpp, 2)
    print_body_line('}', injectionCpp, 1)

def write_set_value_from_case_entry(injectionCpp, struct_name, all_enum):
    variable_name, variable_name_ptr = get_struct_variable_name(struct_name)
    enum_name, enum_nam_ptr = get_enum_name(struct_name)
    int_variable_name, int_variable_name_ptr = get_struct_variable_name('int')
    uint_variable_name, uint_variable_name_ptr = get_struct_variable_name('unsigned int')

    print_generator_source_info(injectionCpp, 2)
    print_body_line(f'case {enum_name}:', injectionCpp, 1)
    print_body_line('{', injectionCpp, 1)
    print_body_line('if (other.m_type == %s)' % enum_name, injectionCpp, 2)
    print_body_line('{', injectionCpp, 2)
    print_memcpy(injectionCpp, 3, variable_name, variable_name, False, False)
    print_body_line('set = true;', injectionCpp, 3)
    print_body_line('}', injectionCpp, 2)
    # Add setting a non-pointer from a pointer to the same arg
    print_generator_source_info(injectionCpp, 3)
    print_body_line('else if (other.m_type == %s)' % enum_nam_ptr, injectionCpp, 2)
    print_body_line('{', injectionCpp, 2)
    print_memcpy(injectionCpp, 3, variable_name, variable_name_ptr, False, True)
    print_body_line('set = true;', injectionCpp, 3)
    print_body_line('}', injectionCpp, 2)
    if enum_name == 'INJECTION_UINT':
        # Add method to convert int to uint
        print_generator_source_info(injectionCpp, 3)
        print_body_line(f'else if (other.m_type == INJECTION_INT && other.m_value.{int_variable_name} > 0)', injectionCpp, 2)
        print_body_line('{', injectionCpp, 2)
        print_body_line(f'this->m_value.{uint_variable_name} = other.m_value.{int_variable_name};', injectionCpp, 3)
        print_body_line('set = true;', injectionCpp, 3)
        print_body_line('}', injectionCpp, 2)
        print_body_line(f'else if (other.m_type == INJECTION_INT_PTR && *other.m_value.{int_variable_name_ptr} > 0)', injectionCpp, 2)
        print_body_line('{', injectionCpp, 2)
        print_body_line(f'this->m_value.{uint_variable_name} = *other.m_value.{int_variable_name_ptr};', injectionCpp, 3)
        print_body_line('set = true;', injectionCpp, 3)
        print_body_line('}', injectionCpp, 2)
    elif enum_name == 'INJECTION_INT':
        # Add method to convert uint to int
        print_generator_source_info(injectionCpp, 3)
        print_body_line(f'else if (other.m_type == INJECTION_UINT && other.m_value.{uint_variable_name} <= INT_MAX)', injectionCpp, 2)
        print_body_line('{', injectionCpp, 2)
        print_body_line(f'this->m_value.{int_variable_name} = other.m_value.{uint_variable_name};', injectionCpp, 3)
        print_body_line('set = true;', injectionCpp, 3)
        print_body_line('}', injectionCpp, 2)
        print_body_line(f'else if (other.m_type == INJECTION_UINT_PTR && *other.m_value.{uint_variable_name_ptr} <= INT_MAX)', injectionCpp, 2)
        print_body_line('{', injectionCpp, 2)
        print_body_line(f'this->m_value.{int_variable_name} = *other.m_value.{uint_variable_name_ptr};', injectionCpp, 3)
        print_body_line('set = true;', injectionCpp, 3)
        print_body_line('}', injectionCpp, 2)
    print_body_line('break;', injectionCpp, 2)
    print_body_line('}', injectionCpp, 1)

    print_body_line(f'case {enum_nam_ptr}:', injectionCpp, 1)
    print_body_line('{', injectionCpp, 1)
    print_body_line('if (other.m_type == %s)' % enum_nam_ptr, injectionCpp, 2)
    print_body_line('{', injectionCpp, 2)
    print_memcpy(injectionCpp, 3, variable_name_ptr, variable_name_ptr, True, True)
    print_body_line('set = true;', injectionCpp, 3)
    print_body_line('}', injectionCpp, 2)
    # Add setting a pointer from a non-pointer to the same arg
    print_generator_source_info(injectionCpp, 3)
    print_body_line('else if (other.m_type == %s)' % enum_name, injectionCpp, 2)
    print_body_line('{', injectionCpp, 2)
    print_memcpy(injectionCpp, 3, variable_name_ptr, variable_name, True, False)
    print_body_line('set = true;', injectionCpp, 3)
    print_body_line('}', injectionCpp, 2)
    if enum_nam_ptr == 'INJECTION_UINT_PTR':
        # Add method to convert int to uint pointer
        print_generator_source_info(injectionCpp, 3)
        print_body_line(f'else if (other.m_type == INJECTION_INT && other.m_value.{int_variable_name} > 0)', injectionCpp, 2)
        print_body_line('{', injectionCpp, 2)
        print_body_line(f'*this->m_value.{uint_variable_name_ptr} = other.m_value.{int_variable_name};', injectionCpp, 3)
        print_body_line('set = true;', injectionCpp, 3)
        print_body_line('}', injectionCpp, 2)
        print_body_line(f'else if (other.m_type == INJECTION_INT_PTR && *other.m_value.{int_variable_name_ptr} > 0)', injectionCpp, 2)
        print_body_line('{', injectionCpp, 2)
        print_body_line(f'*this->m_value.{uint_variable_name_ptr} = *other.m_value.{int_variable_name_ptr};', injectionCpp, 3)
        print_body_line('set = true;', injectionCpp, 3)
        print_body_line('}', injectionCpp, 2)
    elif enum_nam_ptr == 'INJECTION_INT_PTR':
        # Add method to convert uint to int pointer
        print_generator_source_info(injectionCpp, 3)
        print_body_line(f'else if (other.m_type == INJECTION_UINT && other.m_value.{uint_variable_name} <= INT_MAX)', injectionCpp, 2)
        print_body_line('{', injectionCpp, 2)
        print_body_line(f'*this->m_value.{int_variable_name_ptr} = other.m_value.{uint_variable_name};', injectionCpp, 3)
        print_body_line('set = true;', injectionCpp, 3)
        print_body_line('}', injectionCpp, 2)
        print_body_line(f'else if (other.m_type == INJECTION_UINT_PTR && *other.m_value.{uint_variable_name_ptr} <= INT_MAX)', injectionCpp, 2)
        print_body_line('{', injectionCpp, 2)
        print_body_line(f'*this->m_value.{int_variable_name_ptr} = *other.m_value.{uint_variable_name_ptr};', injectionCpp, 3)
        print_body_line('set = true;', injectionCpp, 3)
        print_body_line('}', injectionCpp, 2)
    print_body_line('break;', injectionCpp, 2)
    print_body_line('}', injectionCpp, 1)

def get_all_type_name_arr(struct_with_member, all_enum):
    all_type_name_arr = list(struct_with_member.keys())
    all_type_name_arr += get_basic_type()
    all_type_name_arr += all_enum
    all_type_name_arr += get_handlers()
    all_type_name_arr += get_not_well_defined_type()
    all_type_name_arr.sort()
    return all_type_name_arr

def write_injection_free_function(injectionCpp, struct_with_member, all_enum):
    injectionCpp.write('// InjectionArgument may be copied for serveral cases.\n')
    injectionCpp.write('// But we keep the ownership of pointer in nvml injection lib.\n')
    injectionCpp.write('// We free it when we shutdown the injection lib.\n')
    print_generator_source_info(injectionCpp, 0)

    all_type_name_arr = get_all_type_name_arr(struct_with_member, all_enum)
    injectionCpp.write('InjectionArgument::~InjectionArgument()\n{\n')
    print_body_line('switch (this->m_type)', injectionCpp, 0)
    print_body_line('{', injectionCpp, 0)
    for type_name in all_type_name_arr:
        variable_name, variable_name_ptr = get_struct_variable_name(type_name)
        enum_name, enum_name_ptr = get_enum_name(type_name)
        print_body_line('case %s:' % enum_name_ptr, injectionCpp, 1)
        print_body_line('{', injectionCpp, 1)
        print_body_line(f'if (m_inHeap && m_value.{variable_name_ptr})', injectionCpp, 2)
        print_body_line('{', injectionCpp, 2)
        print_body_line(f'free(m_value.{variable_name_ptr});', injectionCpp, 3)
        print_body_line('}', injectionCpp, 2)
        print_body_line('break;', injectionCpp, 2)
        print_body_line('}', injectionCpp, 1)
    print_body_line('default:', injectionCpp, 1)
    print_body_line('break;', injectionCpp, 2)
    print_body_line('}', injectionCpp, 0)
    injectionCpp.write('}\n\n')

def struct_compare_func_name(struct_name):
    return f"{struct_name}Compare"

def write_struct_compare_declare(out_file, struct_with_member, all_enum):
    print_generator_source_info(out_file, 0)
    for struct_name, _ in struct_with_member.items():
        if struct_name == "nvmlGpuThermalSettings_t" or struct_name == "nvmlGpuDynamicPstatesInfo_t" or struct_name == "nvmlGpmMetric_t" or struct_name == "nvmlGpmMetricsGet_t":
            continue
        func_name = struct_compare_func_name(struct_name)
        out_file.write(f"int {func_name}(const {struct_name} &a, const {struct_name} &b);\n")

def write_struct_compare_definition(out_file, struct_with_member, all_enum):
    for struct_name, member_list in struct_with_member.items():
        func_name = struct_compare_func_name(struct_name)
        if struct_name == "nvmlGpuThermalSettings_t" or struct_name == "nvmlGpuDynamicPstatesInfo_t" or struct_name == "nvmlGpmMetric_t" or struct_name == "nvmlGpmMetricsGet_t":
            continue
        print_generator_source_info(out_file, 0)
        out_file.write(f"int {func_name}(const {struct_name} &a, const {struct_name} &b)\n")
        out_file.write("{\n")
        for member_type, member_name in member_list:
            if member_type == "nvmlVgpuSchedulerParams_t" or member_type == "nvmlVgpuSchedulerSetParams_t":
                print_body_line(f'NVML_LOG_ERR("{member_type} conatins union, and cannot compare now. May cause problems...");', out_file, 0)
            elif is_basic_type(member_type) or is_enum_type(member_type, all_enum) or 'nvmlDevice_t' == member_type or 'nvmlGpuInstance_t' == member_type:
                # basic type directly compares the value
                print_generator_source_info(out_file, 1)
                print_body_line(f"if (a.{member_name} != b.{member_name})", out_file, 0)
                print_body_line("{", out_file, 0)
                print_body_line(f"return a.{member_name} < b.{member_name} ? -1 : 1;", out_file, 1)
                print_body_line("}", out_file, 0)
            elif 'unsigned char[' in member_type:
                # byte-array type uses memcmp
                print_generator_source_info(out_file, 1)
                print_body_line(f"if (auto ret = memcmp(a.{member_name}, b.{member_name}, sizeof(a.{member_name})); ret)", out_file, 0)
                print_body_line("{", out_file, 0)
                print_body_line("return ret;", out_file, 1)
                print_body_line("}", out_file, 0)
            elif 'char[' in member_type:
                # c-string type uses strcmp
                print_generator_source_info(out_file, 1)
                print_body_line(f"if (auto ret = strcmp(a.{member_name}, b.{member_name}); ret)", out_file, 0)
                print_body_line("{", out_file, 0)
                print_body_line("return ret;", out_file, 1)
                print_body_line("}", out_file, 0)
            elif 'nvmlValue_t' == member_type:
                # nvmlValue_t type uses memcmp
                print_generator_source_info(out_file, 1)
                print_body_line(f"if (auto ret = memcmp(&a.{member_name}, &b.{member_name}, sizeof(a.{member_name})); ret)", out_file, 0)
                print_body_line("{", out_file, 0)
                print_body_line("return ret;", out_file, 1)
                print_body_line("}", out_file, 0)
            elif '[' in member_type:
                # for general array, compare every element
                print_generator_source_info(out_file, 1)
                pure_type = member_type.split("[")[0]
                print_body_line(f"for (unsigned int i = 0; i < sizeof(a.{member_name}) / sizeof(a.{member_name}[0]); ++i)", out_file, 0)
                print_body_line("{", out_file, 0)
                if is_basic_type(pure_type):
                    # basic type directly compares the value
                    print_body_line(f"if (a.{member_name}[i] != b.{member_name}[i])", out_file, 1)
                    print_body_line("{", out_file, 1)
                    print_body_line(f"return a.{member_name}[i] < b.{member_name}[i] ? -1 : 1;", out_file, 2)
                    print_body_line("}", out_file, 1)
                else:
                    # call the appropriate compare function to compare if the array elemenet is another struct
                    compare_func_name = struct_compare_func_name(pure_type)
                    print_body_line(f"if (auto ret = {compare_func_name}(a.{member_name}[i], b.{member_name}[i]); ret)", out_file, 1)
                    print_body_line("{", out_file, 1)
                    print_body_line(f"return ret;", out_file, 2)
                    print_body_line("}", out_file, 1)
                print_body_line("}", out_file, 0)
            elif is_basic_ptr_type(member_type):
                print_generator_source_info(out_file, 1)
                print_body_line(f"if (*a.{member_name} != *b.{member_name})", out_file, 0)
                print_body_line("{", out_file, 0)
                print_body_line(f"return *a.{member_name} < *b.{member_name};", out_file, 1)
                print_body_line("}", out_file, 0)
            elif is_pointer_type(member_type):
                # call the appropriate compare function to compare for member type is another struct
                compare_func_name = struct_compare_func_name(remove_ptr_if_any(member_type))
                print_generator_source_info(out_file, 1)
                print_body_line(f"if (auto ret = {compare_func_name}(*a.{member_name}, *b.{member_name}); ret)", out_file, 0)
                print_body_line("{", out_file, 0)
                print_body_line(f"return ret;", out_file, 1)
                print_body_line("}", out_file, 0)
            else:
                # call the appropriate compare function to compare for member type is another struct
                compare_func_name = struct_compare_func_name(member_type)
                print_generator_source_info(out_file, 1)
                print_body_line(f"if (auto ret = {compare_func_name}(a.{member_name}, b.{member_name}); ret)", out_file, 0)
                print_body_line("{", out_file, 0)
                print_body_line(f"return ret;", out_file, 1)
                print_body_line("}", out_file, 0)
        print_body_line("return 0;", out_file, 0)
        out_file.write("}\n\n")

def write_injection_argument_compare(out_file, struct_with_member, all_enum):
    print_generator_source_info(out_file, 0)
    out_file.write('int InjectionArgument::Compare(const InjectionArgument &other) const\n')
    out_file.write('{\n')
    print_body_line('if (m_type < other.m_type)', out_file, 0)
    print_body_line('{', out_file, 0)
    print_body_line('return -1;', out_file, 1)
    print_body_line('}', out_file, 0)
    print_body_line('else if (m_type > other.m_type)', out_file, 0)
    print_body_line('{', out_file, 0)
    print_body_line('return 1;', out_file, 1)
    print_body_line('}', out_file, 0)
    print_body_line('else', out_file, 0)
    print_body_line('{', out_file, 0)
    print_body_line('if (m_type == INJECTION_STRING)', out_file, 1)
    print_body_line('{', out_file, 1)
    print_body_line('if (m_str < other.m_str)', out_file, 2)
    print_body_line('{', out_file, 2)
    print_body_line('return -1;', out_file, 3)
    print_body_line('}', out_file, 2)
    print_body_line('else if (m_str > other.m_str)', out_file, 2)
    print_body_line('{', out_file, 2)
    print_body_line('return 1;', out_file, 3)
    print_body_line('}', out_file, 2)
    print_body_line('else', out_file, 2)
    print_body_line('{', out_file, 2)
    print_body_line('return 0;', out_file, 3)
    print_body_line('}', out_file, 2)
    print_body_line('}', out_file, 1)
    print_body_line('else if (m_type == INJECTION_CONST_CHAR_PTR)', out_file, 1)
    print_body_line('{', out_file, 1)
    print_body_line('return strcmp(m_value.ConstStr, other.m_value.ConstStr);', out_file, 2)
    print_body_line('}', out_file, 1)
    print_body_line('else', out_file, 1)
    print_body_line('{', out_file, 1)
    print_body_line('switch (m_type)', out_file, 2)
    print_body_line('{', out_file, 2)

    all_type_name_arr = get_all_type_name_arr(struct_with_member, all_enum)

    for type_name in all_type_name_arr:
        variable_name, variable_name_ptr = get_struct_variable_name(type_name)
        enum_name, enum_name_ptr = get_enum_name(type_name)

        # pointer case
        print_body_line('case %s:' % enum_name_ptr, out_file, 3)
        print_body_line('{', out_file, 3)
        if enum_name_ptr == 'INJECTION_CHAR_PTR' or enum_name_ptr == 'INJECTION_CONST_CHAR_PTR':
            # c-string uses strcmp
            print_body_line('return strcmp(m_value.%s, other.m_value.%s);' % (variable_name_ptr, variable_name_ptr), out_file, 4)
        elif is_basic_type(type_name) or is_enum_type(remove_ptr_if_any(type_name), all_enum):
            # pointer type should take whether it is array or not in account
            print_generator_source_info(out_file, 5)
            print_body_line('if (!m_isArray)', out_file, 4)
            print_body_line('{', out_file, 4)
            print_body_line('if (*m_value.%s == *other.m_value.%s)' % (variable_name_ptr, variable_name_ptr), out_file, 5)
            print_body_line('{', out_file, 5)
            print_body_line('return 0;', out_file, 6)
            print_body_line('}', out_file, 5)
            print_body_line(f'return *m_value.{variable_name_ptr} < *other.m_value.{variable_name_ptr} ? -1 : 1;', out_file, 5)
            print_body_line('}', out_file, 4)
            print_body_line('for (unsigned i = 0; i < m_arrLen; ++i)', out_file, 4)
            print_body_line('{', out_file, 4)
            print_body_line('if (m_value.%s[i] == other.m_value.%s[i])' % (variable_name_ptr, variable_name_ptr), out_file, 5)
            print_body_line('{', out_file, 5)
            print_body_line('continue;', out_file, 6)
            print_body_line('}', out_file, 5)
            print_body_line(f'return m_value.{variable_name_ptr}[i] < other.m_value.{variable_name_ptr}[i] ? -1 : 1;', out_file, 5)
            print_body_line('}', out_file, 4)
            print_body_line(f'return 0;', out_file, 4)
        elif remove_ptr_if_any(type_name) == "nvmlDevice_t" or remove_ptr_if_any(type_name) == "nvmlComputeInstance_t" or remove_ptr_if_any(type_name) == "nvmlEventSet_t" or remove_ptr_if_any(type_name) == "nvmlGpmMetricsGet_t" or remove_ptr_if_any(type_name) == "nvmlGpmSample_t" or remove_ptr_if_any(type_name) == "nvmlGpuDynamicPstatesInfo_t" or remove_ptr_if_any(type_name) == "nvmlGpuInstance_t" or remove_ptr_if_any(type_name) == "nvmlGpuThermalSettings_t" or remove_ptr_if_any(type_name) == "nvmlUnit_t" or type_name == "nvmlGpmMetric_t":
            # Listed types are not well-defined or may have inner struct defined, use memcmp for them
            print_generator_source_info(out_file, 5)
            print_body_line('unsigned size = m_isArray ? m_arrLen : 1;', out_file, 4)
            print_body_line(f'return memcmp(m_value.{variable_name_ptr}, other.m_value.{variable_name_ptr}, size * sizeof(*m_value.{variable_name_ptr}));', out_file, 4)
        else:
            # Otherwise use generated compare function to check
            print_generator_source_info(out_file, 5)
            compare_func = struct_compare_func_name(remove_ptr_if_any(type_name))
            if compare_func.startswith('const '):
                compare_func = compare_func[6:]
            print_body_line('if (!m_isArray)', out_file, 4)
            print_body_line('{', out_file, 4)
            print_body_line(f'return {compare_func}(*m_value.%s, *other.m_value.%s);' % (variable_name_ptr, variable_name_ptr), out_file, 5)
            print_body_line('}', out_file, 4)
            print_body_line('for (unsigned i = 0; i < m_arrLen; ++i)', out_file, 4)
            print_body_line('{', out_file, 4)
            print_body_line(f'if (auto ret = {compare_func}(*m_value.{variable_name_ptr}, *other.m_value.{variable_name_ptr}); ret)', out_file, 5)
            print_body_line('{', out_file, 5)
            print_body_line('return ret;', out_file, 6)
            print_body_line('}', out_file, 5)
            print_body_line('}', out_file, 4)
            print_body_line(f'return 0;', out_file, 4)
        print_body_line('break; // NOT REACHED', out_file, 4)
        print_body_line('}', out_file, 3)

        # non-pointer case
        print_body_line('case %s:' % enum_name, out_file, 3)
        print_body_line('{', out_file, 3)
        if is_basic_type(type_name) or is_enum_type(type_name, all_enum) or type_name == "nvmlDevice_t" or type_name == "nvmlComputeInstance_t":
            # Basic type directly compares
            # Note: nvmlDevice_t and nvmlComputeInstance_t are number in our implementation
            print_generator_source_info(out_file, 5)
            print_body_line('if (m_value.%s < other.m_value.%s)'  % (variable_name, variable_name), out_file, 4)
            print_body_line('{', out_file, 4)
            print_body_line('return -1;', out_file, 5)
            print_body_line('}', out_file, 4)
            print_body_line('else if (m_value.%s > other.m_value.%s)' % (variable_name, variable_name), out_file, 4)
            print_body_line('{', out_file, 4)
            print_body_line('return 1;', out_file, 5)
            print_body_line('}', out_file, 4)
            print_body_line('else', out_file, 4)
            print_body_line('{', out_file, 4)
            print_body_line('return 0;', out_file, 5)
            print_body_line('}', out_file, 4)
        elif type_name == "nvmlEventSet_t" or type_name == "nvmlGpmMetricsGet_t" or type_name == "nvmlGpmSample_t" or type_name == "nvmlGpuDynamicPstatesInfo_t" or type_name == "nvmlGpuInstance_t" or type_name == "nvmlGpuThermalSettings_t" or type_name == "nvmlUnit_t" or type_name == "nvmlGpmMetric_t":
            # Listed types are not well-defined or may have inner struct defined, use memcmp for them
            print_generator_source_info(out_file, 5)
            print_body_line(f'return memcmp(&m_value.{variable_name}, &other.m_value.{variable_name}, sizeof(m_value.{variable_name}));', out_file, 4)
        else:
            # Otherwise use generated compare function to check
            print_generator_source_info(out_file, 5)
            compare_func = struct_compare_func_name(remove_ptr_if_any(type_name))
            if compare_func.startswith('const '):
                compare_func = compare_func[6:]
            print_body_line(f'return {compare_func}(m_value.%s, other.m_value.%s);' % (variable_name, variable_name), out_file, 4)
        print_body_line('break; // NOT REACHED', out_file, 4)
        print_body_line('}', out_file, 3)
    print_body_line('default:', out_file, 3)
    print_body_line('break;', out_file, 4)
    print_body_line('}', out_file, 2)
    print_body_line('}', out_file, 1)
    print_body_line('}', out_file, 0)
    print_body_line('return 1;', out_file, 0)
    out_file.write('}\n\n')

def write_injection_argument_cpp(output_dir, struct_with_member, all_enum):
    injectionCppPath = '%s/%s' % (output_dir, INJECTION_CPP_PATH)
    with open(injectionCppPath, 'w') as injectionCpp:
        write_auto_generate_c_file_header(injectionCpp)
        # some compare functions may not be used
        injectionCpp.write('#pragma GCC diagnostic ignored "-Wunused-function"\n')
        injectionCpp.write('// clang-format off\n')
        injectionCpp.write('#include <%s>\n' % INJECTION_ARGUMENT_HEADER)
        injectionCpp.write('#include "NvmlLogging.h"\n')
        injectionCpp.write('#include <limits.h>\n')
        injectionCpp.write('#include <cstring>\n\n\n')

        print_generator_source_info(injectionCpp, 0)
        injectionCpp.write('namespace\n{\n\n')
        write_struct_compare_declare(injectionCpp, struct_with_member, all_enum)
        injectionCpp.write('\n')
        write_struct_compare_definition(injectionCpp, struct_with_member, all_enum)
        injectionCpp.write('}\n\n')

        print_generator_source_info(injectionCpp, 0)
        injectionCpp.write('nvmlReturn_t InjectionArgument::SetValueFrom(const InjectionArgument &other)\n{\n')
        print_body_line('bool set = false;\n', injectionCpp, 0)
        print_body_line('if (other.IsEmpty())', injectionCpp, 0)
        print_body_line('{', injectionCpp, 0)
        print_body_line('return NVML_ERROR_NOT_FOUND;', injectionCpp, 1)
        print_body_line('}', injectionCpp, 0)
        print_body_line('switch (this->m_type)', injectionCpp, 0)
        print_body_line('{', injectionCpp, 0)

        all_type_name_arr = get_all_type_name_arr(struct_with_member, all_enum)
        for struct_name in all_type_name_arr:
            write_set_value_from_case_entry(injectionCpp, struct_name, all_enum)

        write_string_case_entry(injectionCpp)

        print_body_line('default:', injectionCpp, 1)
        print_body_line('break;', injectionCpp, 2)

        print_body_line('}', injectionCpp, 0)
        print_body_line('if (set)', injectionCpp, 0)
        print_body_line('{', injectionCpp, 0)
        print_body_line('return NVML_SUCCESS;', injectionCpp, 1)
        print_body_line('}', injectionCpp, 0)
        print_body_line('else', injectionCpp, 0)
        print_body_line('{', injectionCpp, 0)
        print_body_line('return NVML_ERROR_INVALID_ARGUMENT;', injectionCpp, 1)
        print_body_line('}', injectionCpp, 0)
        injectionCpp.write('}\n\n')

        write_injection_argument_compare(injectionCpp, struct_with_member, all_enum)
        write_injection_free_function(injectionCpp, struct_with_member, all_enum)

    logging.info(f"{INJECTION_CPP_PATH} generated")

c_type_mapping = {
    'char': 'c_char',
    'unsigned char': 'c_char',
    'wchar_t': 'c_wchar',
    'char *': 'c_char_p',
    'wchar_t *': 'c_wchar_p',
    'int': 'c_int',
    'unsigned int': 'c_uint',
    'unsigned': 'c_uint',
    'long': 'c_long',
    'unsigned long': 'c_ulong',
    'short': 'c_short',
    'unsigned short': 'c_ushort',
    'long long': 'c_longlong',
    'unsigned long long': 'c_ulonglong',
    'float': 'c_float',
    'double': 'c_double',
    'long double': 'c_longdouble',
    'void *': 'c_void_p',
    'size_t': 'c_size_t',
    'int8_t': 'c_int8',
    'uint8_t': 'c_uint8',
    'int16_t': 'c_int16',
    'uint16_t': 'c_uint16',
    'int32_t': 'c_int32',
    'uint32_t': 'c_uint32',
    'int64_t': 'c_int64',
    'uint64_t': 'c_uint64',
}

def is_pynvml_missing_struct(struct_name):
    struct_name_with_c_prefix = "c_" + struct_name
    # To enable injecting function returns from NVML functions in Python code, we rely on the definitions provided by PyNVML.
    # The following structures are not yet defined in PyNVML. To avoid import failures from nvml_injection_structs.py,
    # we're temporarily skipping their generation. We'll revisit this when updating PyNVML and consider removing them from this list or
    # adapting to its well-defined forms.
    missing = {
        "c_nvmlPciInfoExt_t",
        "c_nvmlNvLinkUtilizationControl_t",
        "c_nvmlCoolerInfo_t",
        "c_nvmlClkMonFaultInfo_t",
        "c_nvmlClkMonStatus_t",
        "c_nvmlFanSpeedInfo_t",
        "c_nvmlDevicePerfModes_t",
        "c_nvmlDeviceCurrentClockFreqs_t",
        "c_nvmlProcessesUtilizationInfo_t",
        "c_nvmlVgpuHeterogeneousMode_t",
        "c_nvmlVgpuPlacementId_t",
        "c_nvmlVgpuPlacementList_t",
        "c_nvmlVgpuTypeBar1Info_t",
        "c_nvmlVgpuInstancesUtilizationInfo_t",
        "c_nvmlVgpuProcessesUtilizationInfo_t",
        "c_nvmlEncoderSessionInfo_t",
        "c_nvmlFBCSessionInfo_t",
        "c_nvmlSystemConfComputeSettings_t",
        "c_nvmlConfComputeSetKeyRotationThresholdInfo_v1_t",
        "c_nvmlConfComputeGetKeyRotationThresholdInfo_v1_t",
        "c_nvmlSystemDriverBranchInfo_t",
        "c_nvmlTemperature_t",
        "c_nvmlGpuInstanceProfileInfo_v3_t",
        "c_nvmlComputeInstanceProfileInfo_v3_t",
        "c_nvmlDeviceCapabilities_t",
        "c_nvmlMask255_t",
        "c_nvmlWorkloadPowerProfileInfo_t",
        "c_nvmlWorkloadPowerProfileProfilesInfo_t",
        "c_nvmlWorkloadPowerProfileCurrentProfiles_t",
        "c_nvmlWorkloadPowerProfileRequestedProfiles_t",
        "c_nvmlMarginTemperature_t",
        "c_nvmlVgpuRuntimeState_t",
    }
    return struct_name_with_c_prefix in missing

def get_struct_variable_name(struct_name, const=False):
    no_ptr_struct_name = remove_ptr_if_any(struct_name)
    if is_basic_type(no_ptr_struct_name):
        if struct_name == 'char':
            if const:
                return 'ConstChar', 'ConstStr'
            else:
                return 'Char', 'Str'
        token = struct_name.split(" ")
        ret = ''
        for t in token:
            if t == "unsigned":
                ret += "U"
            else:
                ret += t.capitalize()
        return ret, ret+"Ptr"
    ret = struct_name
    if ret.startswith('nvml'):
        ret = struct_name[4:]
    if ret.endswith('_t'):
        ret = ret[:-2]
    if const:
        return "Const"+ret, "Const"+ret+"Ptr"
    else:
        return ret, ret+"Ptr"

def get_handlers():
    return [NVML_DEVICE, "nvmlGpuInstance_t", "nvmlComputeInstance_t", "nvmlUnit_t"]

def get_const_type():
    return ["nvmlGpuInstancePlacement_t", "char"]

def get_not_well_defined_type():
    return ["nvmlGpmSample_t", "nvmlEventSet_t"]

def write_nvml_injection_struct_py(output_dir, struct_with_member, all_enum):
    out_file_path = f"testing/python3/{INJECTION_STRUCTS_PY_NAME}"
    handlers = get_handlers()

    with open(out_file_path, 'w') as out_file:
        write_auto_generate_py_file_header(out_file)
        out_file.write("# pylint: skip-file\n")
        out_file.write("# skip linting for generated file\n")
        out_file.write("global nvml_injection_usable\n")
        out_file.write("try:\n")
        print_body_line("from dcgm_nvml import *", out_file, 0)
        print_body_line("from ctypes import *\n", out_file, 0)

        # PyNVML does not define c_nvmlProcessInfo_v1_t. We define it here
        print_body_line('class c_nvmlProcessInfo_v1_t(Structure):', out_file, 0)
        print_body_line("_fields_ = [", out_file, 1)
        print_body_line("(\"pid\", c_uint),", out_file, 2)
        print_body_line("(\"usedGpuMemory\", c_ulonglong),", out_file, 2)
        print_body_line("]\n", out_file, 1)

        # PyNVML does not define c_nvmlPlatformInfo_v2_t. We define it here
        print_body_line('class c_nvmlPlatformInfo_v2_t(Structure):', out_file, 0)
        print_body_line("_fields_ = [", out_file, 1)
        print_body_line("(\"version\", c_uint),", out_file, 2)
        print_body_line("(\"ibGuid\", c_char * 16),", out_file, 2)
        print_body_line("(\"chassisSerialNumber\", c_char * 16),", out_file, 2)
        print_body_line("(\"slotNumber\", c_char),", out_file, 2)
        print_body_line("(\"trayIndex\", c_char),", out_file, 2)
        print_body_line("(\"hostId\", c_char),", out_file, 2)
        print_body_line("(\"peerType\", c_char),", out_file, 2)
        print_body_line("(\"moduleId\", c_char),", out_file, 2)
        print_body_line("]\n", out_file, 1)

        # PyNVML's c_nvmlGpuFabricInfo_t is wrong. We define correct version here.
        print_body_line('class c_nvmlGpuFabricInfo_t_dcgm_ver(Structure):', out_file, 0)
        print_body_line("_fields_ = [", out_file, 1)
        print_body_line("(\"clusterUuid\", c_ubyte * 16),", out_file, 2)
        print_body_line("(\"status\", c_uint),", out_file, 2)
        print_body_line("(\"cliqueId\", c_uint32),", out_file, 2)
        print_body_line("(\"state\", c_ubyte),", out_file, 2)
        print_body_line("]\n", out_file, 1)

        # PyNVML's c_nvmlGpuFabricInfoV_t is wrong. We define correct version here.

        print_body_line('class c_nvmlGpuFabricInfoV_t_dcgm_ver(Structure):', out_file, 0)
        print_body_line("_fields_ = [", out_file, 1)
        print_body_line("(\"version\", c_uint),", out_file, 2)
        print_body_line("(\"clusterUuid\", c_char * 16),", out_file, 2)
        print_body_line("(\"status\", c_uint),", out_file, 2)
        print_body_line("(\"cliqueId\", c_uint),", out_file, 2)
        print_body_line("(\"state\", c_ubyte),", out_file, 2)
        print_body_line("(\"healthMask\", c_uint),", out_file, 2)
        print_body_line("]\n", out_file, 1)

        # PyNVML does not define c_nvmlPlatformInfo_t yet. Define separately here.
        print_body_line('class c_nvmlPlatformInfo_v1_t(Structure):', out_file, 0)
        print_body_line("_fields_ = [", out_file, 1)
        print_body_line("(\"version\", c_uint),", out_file, 2)
        print_body_line("(\"ibGuid\", c_char * 16),", out_file, 2)
        print_body_line("(\"rackGuid\", c_char * 16),", out_file, 2)
        print_body_line("(\"chassisPhysicalSlotNumber\", c_char),", out_file, 2)
        print_body_line("(\"computeSlotIndex\", c_char),", out_file, 2)
        print_body_line("(\"nodeIndex\", c_char),", out_file, 2)
        print_body_line("(\"peerType\", c_char),", out_file, 2)
        print_body_line("(\"moduleId\", c_char),", out_file, 2)
        print_body_line("]\n", out_file, 1)

        print_body_line('class c_simpleValue_t(Union):', out_file, 0)
        print_body_line("_fields_ = [", out_file, 1)
        for struct_name, _ in struct_with_member.items():
            variable_name, variable_name_ptr = get_struct_variable_name(struct_name)
            print_body_line(f"(\"{variable_name_ptr}\", c_void_p),", out_file, 2)
            if is_pynvml_missing_struct(struct_name):
                # we don't know the structure declarition now, comment out and wait for PyNVML updates.
                print_body_line(f"# (\"{variable_name}\", c_{struct_name}),", out_file, 2)
            elif struct_name == "nvmlDeviceAttributes_t" or struct_name == "nvmlRowRemapperHistogramValues_t":
                # nvmlDeviceAttributes_t in PyNVML is c_nvmlDeviceAttributes
                # nvmlRowRemapperHistogramValues_t in PyNVML is nvmlRowRemapperHistogramValues
                print_body_line(f"(\"{variable_name}\", c_{struct_name[:-2]}),", out_file, 2)
            elif variable_name == "nvmlFBCSessionInfo_t":
                # nvmlFBCSessionInfo_t in PyNVML is c_nvmlFBCSession_t
                print_body_line(f"(\"{variable_name}\", c_nvmlFBCSession_t),", out_file, 2)
            elif variable_name == "nvmlEncoderSessionInfo_t":
                # nvmlEncoderSessionInfo_t in PyNVML is c_nvmlEncoderSession_t
                print_body_line(f"(\"{variable_name}\", c_nvmlEncoderSession_t),", out_file, 2)
            elif variable_name == "nvmlNvLinkUtilizationControl_t" or struct_name == "nvmlPciInfo_t":
                # nvmlNvLinkUtilizationControl_t in PyNVML is nvmlNvLinkUtilizationControl_t
                # nvmlPciInfo_t in PyNVML is nvmlPciInfo_t
                print_body_line(f"(\"{variable_name}\", {struct_name}),", out_file, 2)
            elif struct_name == "nvmlGpuFabricInfo_t":
                # nvmlGpuFabricInfo_t is wrong in PyNVML, use our own definition
                print_body_line(f"(\"{variable_name}\", c_{struct_name}_dcgm_ver),", out_file, 2)
            elif struct_name == "nvmlGpuFabricInfoV_t":
                # nvmlGpuFabricInfoV_t is wrong in PyNVML, use our own definition
                print_body_line(f"(\"{variable_name}\", c_{struct_name}_dcgm_ver),", out_file, 2)
            elif struct_name == "nvmlPlatformInfo_t":
                # nvmlPlatformInfo_t is not defined in PyNVML, add definition
                print_body_line(f"(\"{variable_name}\", c_nvmlPlatformInfo_v2_t),", out_file, 2)
            elif struct_name == "nvmlPlatformInfo_v1_t":
                # nvmlPlatformInfo_t is not defined in PyNVML, add definition
                print_body_line(f"(\"{variable_name}\", c_nvmlPlatformInfo_v1_t),", out_file, 2)
            elif struct_name == "nvmlEccSramErrorStatus_t":
                # nvmlEccSramErrorStatus_t in PyNVML is c_nvmlEccSreamErrorStatus_v1_t
                print_body_line(f"(\"{variable_name}\", c_nvmlEccSramErrorStatus_v1_t),", out_file, 2)
            else:
                # other nvml structures will add c_ as prefix in PyNVML
                # e.g. nvmlBAR1Memory_t => c_nvmlBAR1Memory_t
                print_body_line(f"(\"{variable_name}\", c_{struct_name}),", out_file, 2)
        basic_types = [basic_type for basic_type in get_basic_type()]
        basic_types.sort()
        for basic in basic_types:
            variable_name, variable_name_ptr = get_struct_variable_name(basic)
            print_body_line(f"(\"{variable_name_ptr}\", c_void_p),", out_file, 2)
            # replace correct ctype from mapping for basic type
            print_body_line(f"(\"{variable_name}\", {c_type_mapping.get(basic)}),", out_file, 2)
        for enum in all_enum:
            variable_name, variable_name_ptr = get_struct_variable_name(enum)
            # use int to represent enum type
            print_body_line(f"(\"{variable_name}\", c_int),", out_file, 2)
            print_body_line(f"(\"{variable_name_ptr}\", c_void_p),", out_file, 2)

        for handler in handlers:
            handler_name, handler_name_ptr = get_struct_variable_name(handler)
            print_body_line(f"(\"{handler_name}\", c_void_p),", out_file, 2)
            print_body_line(f"(\"{handler_name_ptr}\", c_void_p),", out_file, 2)

        not_well_defined_types = get_not_well_defined_type()
        for t in not_well_defined_types:
            name, name_ptr = get_struct_variable_name(t)
            print_body_line(f"(\"{name}\", c_void_p),", out_file, 2)
            print_body_line(f"(\"{name_ptr}\", c_void_p),", out_file, 2)
        print_body_line("]\n", out_file, 1)

        print_body_line('class c_injectionArgType_t(c_int):', out_file, 0)
        index = 0
        all_type_name_arr = get_all_type_name_arr(struct_with_member, all_enum)
        for t in all_type_name_arr:
            enum_name, enum_nam_ptr = get_enum_name(t)
            print_body_line(f"{enum_name} = {index}", out_file, 1)
            index += 1
            print_body_line(f"{enum_nam_ptr} = {index}", out_file, 1)
            index += 1

        print_body_line(f"INJECTION_STRING = {index}", out_file, 1)
        index += 1
        const_types = get_const_type()
        for const_type in const_types:
            handler_enum_name, handler_enum_name_ptr = get_enum_name(const_type, True)
            print_body_line(f"{handler_enum_name} = {index}", out_file, 1)
            index += 1
            print_body_line(f"{handler_enum_name_ptr} = {index}", out_file, 1)
            index += 1
        out_file.write("\n")
        print_body_line('class c_injectNvmlVal_t(Structure):', out_file, 0)
        print_body_line("_fields_ = [", out_file, 1)
        print_body_line("(\"value\", c_simpleValue_t),", out_file, 2)
        print_body_line("(\"type\", c_injectionArgType_t),", out_file, 2)
        print_body_line("]\n", out_file, 1)

        print_body_line("nvml_injection_usable = True\n", out_file, 0)

        out_file.write("except ModuleNotFoundError:\n")
        print_body_line("print (\"No dcgm_nvml is currently present.\")", out_file, 0)
        print_body_line("nvml_injection_usable = False", out_file, 0)
        out_file.write("except NameError as e:\n")
        print_body_line("print (f\"dcgm_nvml is probably an older version. It is missing a definition for {str(e)}.\")", out_file, 0)
        print_body_line("nvml_injection_usable = False", out_file, 0)

    logging.info(f"{INJECTION_STRUCTS_PY_NAME} genereted")

def write_injection_structs_header(struct_with_member, output_dir, all_enum):
    injection_structs_path = "%s/%s" % (output_dir, INJECTION_STRUCTS_PATH)

    with open(injection_structs_path, 'w') as injectionStructs:
        write_auto_generate_c_file_header(injectionStructs)
        injectionStructs.write('// clang-format off\n')
        injectionStructs.write('#pragma once\n\n')
        injectionStructs.write('#include <nvml.h>\n')

        all_type_name_arr = get_all_type_name_arr(struct_with_member, all_enum)
        # write the union for simple value types
        print_generator_source_info(injectionStructs, 0)
        injectionStructs.write('typedef union\n{\n')
        for t in all_type_name_arr:
            variable_name, variable_name_ptr = get_struct_variable_name(t)
            print_body_line('%s *%s;' % (t, variable_name_ptr), injectionStructs, 0)
            print_body_line('%s %s;' % (t, variable_name), injectionStructs, 0)
        const_types = get_const_type()
        for const_type in const_types:
            name, name_ptr = get_struct_variable_name(const_type, True)
            print_body_line('const %s *%s;' % (const_type, name_ptr), injectionStructs, 0)

        injectionStructs.write('} simpleValue_t;\n\n')

        # write the enum for the types
        injectionStructs.write('typedef enum injectionArg_enum\n{\n')
        index = 0
        for t in all_type_name_arr:
            enum_name, enum_nam_ptr = get_enum_name(t)
            print_body_line("%s = %d," % (enum_name, index), injectionStructs, 0)
            index += 1
            print_body_line("%s = %d," % (enum_nam_ptr, index), injectionStructs, 0)
            index += 1

        print_body_line('%s = %d,' % ("INJECTION_STRING", index), injectionStructs, 0)
        index += 1

        const_types = get_const_type()
        for const_type in const_types:
            handler_enum_name, handler_enum_name_ptr = get_enum_name(const_type, True)
            print_body_line('%s = %d,' % (handler_enum_name, index), injectionStructs, 0)
            index += 1
            print_body_line('%s = %d,' % (handler_enum_name_ptr, index), injectionStructs, 0)
            index += 1
        print_body_line(INJECTION_ARG_COUNT_STR, injectionStructs, 0)
        injectionStructs.write('} injectionArgType_t;\n\n')

        injectionStructs.write('typedef struct\n{\n')
        print_body_line('simpleValue_t value;', injectionStructs, 0)
        print_body_line('injectionArgType_t type;', injectionStructs, 0)
        injectionStructs.write('} injectNvmlVal_t;\n\n')

    logging.info(f"{INJECTION_STRUCTS_PATH} genereted")

    return

def write_injection_argument_deep_copy_function(out_file, struct_with_member, all_enum):
    print_generator_source_info(out_file, 1)
    print_body_line('// Some type of holding member live in heap. Do deep copy so that our destructor can work well.', out_file, 0)
    print_body_line('void DeepCopy(const InjectionArgument &other)', out_file, 0)
    print_body_line('{', out_file, 0)
    print_body_line('m_type = other.m_type;', out_file, 1)
    print_body_line('m_value = other.m_value;', out_file, 1)
    print_body_line('m_str = other.m_str;', out_file, 1)
    print_body_line('m_isArray = other.m_isArray;', out_file, 1)
    print_body_line('m_arrLen = other.m_arrLen;', out_file, 1)
    print_body_line('m_inHeap = other.m_inHeap;', out_file, 1)
    print_body_line('if (!m_inHeap)', out_file, 1)
    print_body_line('{', out_file, 1)
    print_body_line('return;', out_file, 2)
    print_body_line('}', out_file, 1)
    print_body_line('switch (m_type)', out_file, 1)
    print_body_line('{', out_file, 1)
    all_type_name_arr = get_all_type_name_arr(struct_with_member, all_enum)
    for struct_name in all_type_name_arr:
        _, variable_name_ptr = get_struct_variable_name(struct_name)
        _, enum_nam_ptr = get_enum_name(struct_name)
        print_body_line(f'case {enum_nam_ptr}:', out_file, 2)
        print_body_line('{', out_file, 2)
        print_body_line('unsigned int allocateNum = m_isArray ? m_arrLen : 1;', out_file, 3)
        print_body_line(f'm_value.{variable_name_ptr} = static_cast<{struct_name} *>(malloc(allocateNum * sizeof(*other.m_value.{variable_name_ptr})));', out_file, 3)
        print_body_line(f'if (m_value.{variable_name_ptr} != nullptr)', out_file, 3)
        print_body_line('{', out_file, 3)
        print_body_line(f'std::memcpy(m_value.{variable_name_ptr}, other.m_value.{variable_name_ptr}, allocateNum * sizeof(*other.m_value.{variable_name_ptr}));', out_file, 4)
        print_body_line('}', out_file, 3)
        print_body_line('break;', out_file, 3)
        print_body_line('}', out_file, 2)
    print_body_line('default:', out_file, 2)
    print_body_line('break; // NOT Reached', out_file, 3)
    print_body_line('}', out_file, 1)
    print_body_line('}\n', out_file, 0)

def write_injection_argument_copy_constructor(out_file):
    print_generator_source_info(out_file, 1)
    print_body_line('InjectionArgument &operator=(const InjectionArgument &other)', out_file, 0)
    print_body_line('{', out_file, 0)
    print_body_line('DeepCopy(other);', out_file, 1)
    print_body_line('return *this;', out_file, 1)
    print_body_line('}\n', out_file, 0)

    print_generator_source_info(out_file, 1)
    print_body_line('InjectionArgument(const InjectionArgument &other)', out_file, 0)
    print_body_line('{', out_file, 0)
    print_body_line('DeepCopy(other);', out_file, 1)
    print_body_line('}\n', out_file, 0)


def write_injection_argument_header(output_dir, struct_with_member, all_enum):
    injectionHeaderPath = '%s/%s' % (output_dir, INJECTION_ARGUMENT_PATH)

    with open(injectionHeaderPath, 'w') as injectionHeader:
        write_auto_generate_c_file_header(injectionHeader)

        injectionHeader.write('// clang-format off\n')
        injectionHeader.write('#pragma once\n\n')
        injectionHeader.write('#include <cstring>\n')
        injectionHeader.write('#include <nvml.h>\n')
        injectionHeader.write('#include <string>\n\n')
        injectionHeader.write('#include "%s"\n\n' % INJECTION_STRUCTS_NAME)


        print_generator_source_info(injectionHeader, 0)
        injectionHeader.write('class InjectionArgument\n{\nprivate:\n')
        print_body_line('injectionArgType_t m_type;', injectionHeader, 0)
        print_body_line('simpleValue_t m_value;', injectionHeader, 0)
        print_body_line('std::string m_str;\n', injectionHeader, 0)
        print_body_line('bool m_isArray = false;\n', injectionHeader, 0)
        print_body_line('unsigned m_arrLen = 0;\n', injectionHeader, 0)
        print_body_line('bool m_inHeap = false;\n', injectionHeader, 0)
        write_injection_argument_deep_copy_function(injectionHeader, struct_with_member, all_enum)
        injectionHeader.write('public:\n')
        print_body_line('InjectionArgument()', injectionHeader, 0)
        print_body_line(': m_type(%s)' % INJECTION_ARG_COUNT_STR, injectionHeader, 1)
        print_body_line('{', injectionHeader, 0)
        print_body_line('Clear();', injectionHeader, 1)
        print_body_line('}\n', injectionHeader, 0)
        print_body_line('InjectionArgument(const injectNvmlVal_t &value)', injectionHeader, 0)
        print_body_line(': m_type(value.type)', injectionHeader, 1)
        print_body_line(', m_value(value.value)', injectionHeader, 1)
        print_body_line('{}\n', injectionHeader, 0)
        print_body_line('/**', injectionHeader, 0)
        print_body_line(' * SetValueFrom - Sets this injection argument based other\'s value', injectionHeader, 0)
        print_body_line(' * @param other - the InjectionArgument whose value we flexibly copy if possible.', injectionHeader, 0)
        print_body_line(' *', injectionHeader, 0)
        print_body_line(' * @return 0 if we could set from other\'s value, 1 if incompatible', injectionHeader, 0)
        print_body_line(' **/', injectionHeader, 0)
        print_body_line('nvmlReturn_t SetValueFrom(const InjectionArgument &other);\n', injectionHeader, 0)
        print_body_line('injectionArgType_t GetType() const', injectionHeader, 0)
        print_body_line('{', injectionHeader, 0)
        print_body_line('return m_type;', injectionHeader, 1)
        print_body_line('}\n', injectionHeader, 0)
        print_body_line('simpleValue_t GetSimpleValue() const', injectionHeader, 0)
        print_body_line('{', injectionHeader, 0)
        print_body_line('return m_value;', injectionHeader, 1)
        print_body_line('}\n', injectionHeader, 0)
        print_body_line('void Clear()', injectionHeader, 0)
        print_body_line('{', injectionHeader, 0)
        print_body_line('memset(&m_value, 0, sizeof(m_value));', injectionHeader, 1)
        print_body_line('}\n', injectionHeader, 0)
        print_body_line('int Compare(const InjectionArgument &other) const;\n', injectionHeader, 0)
        print_body_line('bool operator<(const InjectionArgument &other) const', injectionHeader, 0)
        print_body_line('{', injectionHeader, 0)
        print_body_line('return this->Compare(other) == -1;', injectionHeader, 1)
        print_body_line('}\n', injectionHeader, 0)
        print_body_line('bool operator==(const InjectionArgument &other) const', injectionHeader, 0)
        print_body_line('{', injectionHeader, 0)
        print_body_line('return this->Compare(other) == 0;', injectionHeader, 1)
        print_body_line('}\n', injectionHeader, 0)
        print_body_line('bool IsEmpty() const', injectionHeader, 0)
        print_body_line('{', injectionHeader, 0)
        print_body_line('return m_type == %s;' % INJECTION_ARG_COUNT_STR, injectionHeader, 1)
        print_body_line('}\n', injectionHeader, 0)

        all_type_name_arr = get_all_type_name_arr(struct_with_member, all_enum)
        for struct_name in all_type_name_arr:
            # Write constructor
            variable_name, variable_name_ptr = get_struct_variable_name(struct_name)
            enum_name, enum_nam_ptr = get_enum_name(struct_name)

            if struct_name not in all_enum or all_enum[struct_name] == ENUM_TYPE_PURE_ENUM:
                # for case like `typedef unsigned int nvmlFanControlPolicy_t;`
                # we cannot use the following way to create constructor. Since it will re-define constructors of unsigned int...
                print_generator_source_info(injectionHeader, 1)
                print_body_line('InjectionArgument(%s *%s, bool inHeap = false)' % (struct_name, variable_name_ptr), injectionHeader, 0)
                print_body_line(': m_type(%s), m_inHeap(inHeap)' % (enum_nam_ptr), injectionHeader, 1)
                print_body_line('{', injectionHeader, 0)
                print_body_line('memset(&m_value, 0, sizeof(m_value));', injectionHeader, 1)
                print_body_line('m_value.%s = %s;' % (variable_name_ptr, variable_name_ptr), injectionHeader, 1)
                print_body_line('}', injectionHeader, 0)

                print_body_line('InjectionArgument(%s %s)' % (struct_name, variable_name), injectionHeader, 0)
                print_body_line(': m_type(%s)' % (enum_name), injectionHeader, 1)
                print_body_line('{', injectionHeader, 0)
                print_body_line('memset(&m_value, 0, sizeof(m_value));', injectionHeader, 1)
                print_body_line('m_value.%s = %s;' % (variable_name, variable_name), injectionHeader, 1)
                print_body_line('}', injectionHeader, 0)

                # Write array constructor
                print_body_line('InjectionArgument(%s *%s, unsigned int arrLen, bool inHeap = false)' % (struct_name, variable_name_ptr), injectionHeader, 0)
                print_body_line(': m_type(%s), m_isArray(true), m_arrLen(arrLen), m_inHeap(inHeap)' % (enum_nam_ptr), injectionHeader, 1)
                print_body_line('{', injectionHeader, 0)
                print_body_line('memset(&m_value, 0, sizeof(m_value));', injectionHeader, 1)
                print_body_line('m_value.%s = %s;' % (variable_name_ptr, variable_name_ptr), injectionHeader, 1)
                print_body_line('}\n', injectionHeader, 0)

                # Write As* function
                print_body_line('%s *As%s() const' % (struct_name, variable_name_ptr), injectionHeader, 0)
                print_body_line('{', injectionHeader, 0)
                print_body_line('return m_value.%s;' % variable_name_ptr, injectionHeader, 1)
                print_body_line('}\n', injectionHeader, 0)

                print_body_line('%s const &As%s() const' % (struct_name, variable_name), injectionHeader, 0)
                print_body_line('{', injectionHeader, 0)
                print_body_line('return m_value.%s;' % variable_name, injectionHeader, 1)
                print_body_line('}\n', injectionHeader, 0)
            else:
                # for case like `typedef unsigned int nvmlFanControlPolicy_t;`
                # for case like `typedef unsigned char nvmlPowerScopeType_t;`
                underly_variable_name, underly_variable_name_ptr = "", ""
                if all_enum[struct_name] == ENUM_TYPE_UNSIGNED_INT:
                    underly_variable_name, underly_variable_name_ptr = get_struct_variable_name('unsigned int')
                else:
                    underly_variable_name, underly_variable_name_ptr = get_struct_variable_name('unsigned char')
                # Write As* function for the case like `typedef unsigned int nvmlFanControlPolicy_t;`
                print_body_line('%s *As%s() const' % (struct_name, variable_name_ptr), injectionHeader, 0)
                print_body_line('{', injectionHeader, 0)
                print_body_line(f'return static_cast<{struct_name} *>(m_value.{underly_variable_name_ptr});', injectionHeader, 1)
                print_body_line('}\n', injectionHeader, 0)

                print_body_line('%s As%s() const' % (struct_name, variable_name), injectionHeader, 0)
                print_body_line('{', injectionHeader, 0)
                print_body_line(f'return static_cast<{struct_name}>(m_value.{underly_variable_name});', injectionHeader, 1)
                print_body_line('}\n', injectionHeader, 0)

        const_types = get_const_type()
        for const_type in const_types:
            variable_name, variable_name_ptr = get_struct_variable_name(const_type, True)
            enum_name, enum_nam_ptr = get_enum_name(const_type, True)
            print_generator_source_info(injectionHeader, 1)
            print_body_line('InjectionArgument(const %s *%s, bool inHeap = false)' % (const_type, variable_name_ptr), injectionHeader, 0)
            print_body_line(': m_type(%s), m_inHeap(inHeap)' % (enum_nam_ptr), injectionHeader, 1)
            print_body_line('{', injectionHeader, 0)
            print_body_line('memset(&m_value, 0, sizeof(m_value));', injectionHeader, 1)
            print_body_line('m_value.%s = %s;' % (variable_name_ptr, variable_name_ptr), injectionHeader, 1)
            print_body_line('}', injectionHeader, 0)

            print_body_line('const %s *As%s() const' % (const_type, variable_name_ptr), injectionHeader, 0)
            print_body_line('{', injectionHeader, 0)
            print_body_line(f'return m_value.{variable_name_ptr};', injectionHeader, 1)
            print_body_line('}\n', injectionHeader, 0)

        write_injection_argument_copy_constructor(injectionHeader)

        # Write destructor
        print_body_line('~InjectionArgument();\n', injectionHeader, 0)

        print_body_line('InjectionArgument(const std::string &val)', injectionHeader, 0)
        print_body_line(': m_type(INJECTION_STRING)', injectionHeader, 1)
        print_body_line(', m_str(val)', injectionHeader, 1)
        print_body_line('{', injectionHeader, 0)
        print_body_line('memset(&m_value, 0, sizeof(m_value));', injectionHeader, 1)
        print_body_line('}', injectionHeader, 0)
        print_body_line('std::string AsString() const', injectionHeader, 0)
        print_body_line('{', injectionHeader, 0)
        print_body_line('switch (m_type)', injectionHeader, 1)
        print_body_line('{', injectionHeader, 1)
        print_body_line('case INJECTION_STRING:', injectionHeader, 2)
        print_body_line('{', injectionHeader, 2)
        print_body_line('return m_str;', injectionHeader, 3)
        print_body_line('}', injectionHeader, 2)
        print_body_line('break;', injectionHeader, 3)
        print_body_line('case INJECTION_CHAR_PTR:', injectionHeader, 2)
        print_body_line('{', injectionHeader, 2)
        print_body_line('if (m_value.Str != nullptr)', injectionHeader, 3)
        print_body_line('{', injectionHeader, 3)
        print_body_line('return std::string(m_value.Str);', injectionHeader, 4)
        print_body_line('}', injectionHeader, 3)
        print_body_line('break;', injectionHeader, 3)
        print_body_line('}', injectionHeader, 2)
        print_body_line('case INJECTION_CONST_CHAR_PTR:', injectionHeader, 2)
        print_body_line('{', injectionHeader, 2)
        print_body_line('if (m_value.ConstStr != nullptr)', injectionHeader, 3)
        print_body_line('{', injectionHeader, 3)
        print_body_line('return std::string(m_value.ConstStr);', injectionHeader, 4)
        print_body_line('}', injectionHeader, 3)
        print_body_line('break;', injectionHeader, 3)
        print_body_line('}', injectionHeader, 2)
        print_body_line('default:', injectionHeader, 2)
        print_body_line('break;', injectionHeader, 3)
        print_body_line('}', injectionHeader, 1)
        print_body_line('return "";', injectionHeader, 1)
        print_body_line('}', injectionHeader, 0)

        injectionHeader.write('};\n')

    logging.info(f"{INJECTION_ARGUMENT_PATH} genereted")

def write_key_file(output_dir):
    key_file_path = "%s/%s" % (output_dir, KEY_LIST_PATH)
    with open(key_file_path, 'w') as key_file:
        write_auto_generate_c_file_header(key_file)
        key_file.write("// clang-format off\n")
        for key in g_key_to_function:
            key_file.write("const char *INJECTION_%s_KEY = \"%s\"; // Function name(s): %s\n" % (key.upper(), key, g_key_to_function[key]))

    logging.info(f"{KEY_LIST_PATH} genereted")

    key_header_path = "%s/%s" % (output_dir, KEY_LIST_HEADER_PATH)
    with open(key_header_path, 'w') as key_header:
        write_auto_generate_c_file_header(key_header)
        key_header.write('// clang-format off\n\n')
        for key in g_key_to_function:
            key_header.write("extern const char *INJECTION_%s_KEY;\n" % (key.upper()))

    logging.info(f"{KEY_LIST_HEADER_PATH} genereted")

def write_linux_defs(output_dir, func_dict):
    linux_defs_path = '%s/%s' % (output_dir, LINUX_DEFS_PATH)
    manually_written_functions = [
        'nvmlInit_v2',
        'nvmlErrorString',
        'nvmlShutdown',
        'nvmlCreateDevice',
        'nvmlDeviceInject',
        'nvmlDeviceInjectForFollowingCalls',
        'nvmlDeviceReset',
        'nvmlDeviceInjectFieldValue',
        'nvmlGetFuncCallCount',
        'nvmlResetFuncCallCount',
        'nvmlRemoveGpu',
        'nvmlRestoreGpu',
    ]

    with open(linux_defs_path, 'w') as linux_defs_file:
        linux_defs_file.write('{\n    global:\n')
        for funcname in manually_written_functions:
            print_body_line('%s;' % funcname, linux_defs_file, 1)
        for funcname in func_dict:
            print_body_line('%s;' % funcname, linux_defs_file, 1)

        print_body_line('extern "C++" {', linux_defs_file, 1)
        print_body_line('_ZTI*;', linux_defs_file, 2)
        print_body_line('_ZTS*;', linux_defs_file, 2)
        print_body_line('};\n', linux_defs_file, 1)
        print_body_line('local:', linux_defs_file, 0)
        print_body_line('*;', linux_defs_file, 1)
        print_body_line('extern "C++" {', linux_defs_file, 1)
        print_body_line('*;', linux_defs_file, 2)
        print_body_line('};', linux_defs_file, 1)
        linux_defs_file.write('};')

    logging.info(f"{LINUX_DEFS_PATH} genereted")

def get_entry_points_all_functions(entry_points_contents):
    all_functions = AllFunctions()
    entry_points = entry_points_contents.split('NVML_ENTRY_POINT')
    first = True
    for entry_point in entry_points:
        funcname, arg_list, arg_names = get_function_signature(entry_point, first)
        first = False
        if funcname and arg_list:
            fi = FunctionInfo(funcname, arg_list, arg_names)
            all_functions.AddFunction(fi)
    return all_functions

def parse_entry_points_contents(contents, output_dir):
    function_dict = {}
    all_argument_types = []
    all_functions = AllFunctions()
    all_functypes = AllFunctionTypes()

    entry_points = contents.split('NVML_ENTRY_POINT')
    total_funcs = 0
    auto_generated = 0
    not_generated = []

    outputStubPath = '%s/%s' % (output_dir, STUB_PATH)

    with open(outputStubPath, 'w') as stub_file:
        write_stub_file_header(stub_file)

        first = True
        for entry_point in entry_points:
            funcname, arg_list, arg_names = get_function_signature(entry_point, first)
            first = False
            if funcname and arg_list:
                fi = FunctionInfo(funcname, arg_list, arg_names)
                all_functions.AddFunction(fi)

        #all_functions.RemoveEarlierVersions()

        for funcname in all_functions.func_dict:
            funcinfo = all_functions.func_dict[funcname]
            all_functypes.AddFunctionType(funcname, funcinfo)

        for funcname in all_functions.func_dict:
            funcinfo = all_functions.func_dict[funcname]
            build_argument_type_list(funcinfo.GetArgumentList(), all_argument_types)
            if write_function(stub_file, funcinfo, all_functypes):
                auto_generated = auto_generated + 1
            else:
                not_generated.append(funcname)
            total_funcs = total_funcs + 1
            function_dict[funcname] = arg_list

        stub_file.write("#ifdef __cplusplus\n}\n")
        stub_file.write("#endif\n")
        stub_file.write("\n#pragma GCC diagnostic pop\n\n")
        stub_file.write('// END nvml_generated_stubs')

    logging.info(f"{STUB_PATH} genereted")

    write_key_file(output_dir)
    write_linux_defs(output_dir, all_functions.func_dict)

def parse_entry_points(inputPath, output_dir):
    with open(inputPath, 'r') as entryFile:
        contents = entryFile.read()
        parse_entry_points_contents(contents, output_dir)


def type_name_to_parser_name(type_name):
    key = type_name[4:-2]
    return f"{key}Parser"

def lowercase_first_letter(s):
    if s:
        return s[:1].lower() + s[1:]
    else:
        return ''

def separate_struct_by_generable(struct_with_member, all_enum):
    cannot_write_deserializer = get_cannot_write_deserializer_struct(struct_with_member)
    struct_cannot_gen_parsers = {}
    # The following types are not used in entry_point.h. In this case, InjectionArgument.h does not have relevent constructor.
    struct_cannot_gen_parsers["nvmlBridgeChipInfo_t"] = True
    struct_cannot_gen_parsers["nvmlClkMonFaultInfo_t"] = True
    struct_cannot_gen_parsers["nvmlVgpuLicenseExpiry_t"] = True
    struct_cannot_gen_parsers["nvmlGridLicenseExpiry_t"] = True
    struct_cannot_gen_parsers["nvmlUnitFanInfo_t"] = True
    struct_cannot_gen_parsers["nvmlConfComputeSystemCaps_t"] = True
    struct_cannot_gen_parsers["nvmlConfComputeMemSizeInfo_t"] = True
    struct_cannot_gen_parsers["nvmlConfComputeGpuCertificate_t"] = True
    struct_cannot_gen_parsers["nvmlConfComputeGpuAttestationReport_t"] = True
    struct_cannot_gen_parsers["nvmlComputeInstancePlacement_t"] = True
    struct_cannot_gen_parsers["nvmlGridLicensableFeature_t"] = True
    struct_cannot_gen_parsers["nvmlClkMonStatus_t"] = True

    # The following types are not defiend well
    struct_cannot_gen_parsers["nvmlGpmSample_t"] = True
    struct_cannot_gen_parsers["nvmlDevice_t"] = True
    struct_can_gen_parsers = {}
    for struct_name, _ in cannot_write_deserializer.items():
        struct_cannot_gen_parsers[struct_name] = True
    for struct_name, member_list in struct_with_member.items():
        if struct_name in struct_cannot_gen_parsers:
            continue
        can = True
        for member_type, name in member_list:
            pure_type = member_type.split("[")[0]
            if is_basic_type(member_type) or "char[" in member_type:
                continue
            if is_enum_type(member_type, all_enum):
                continue
            if name == "reserved":
                continue
            if pure_type not in cannot_write_deserializer:
                continue
            can = False
            break
        if can:
            struct_can_gen_parsers[struct_name] = True
        else:
            struct_cannot_gen_parsers[struct_name] = True
    return struct_can_gen_parsers, struct_cannot_gen_parsers

def func_is_setter(func_name):
    for setter in Setter:
        if len(func_name) < len(setter):
            continue
        if func_name[0:len(setter)] == setter:
            return True
    return False

def func_is_handler_getter(func_name):
    for getter in HandlerGetter:
        if len(func_name) < len(getter):
            continue
        if func_name[0:len(getter)] == getter:
            return True
    return False

def arg_is_count(arg):
    return 'count' == arg or 'SetSize' in arg or 'sessionCount' == arg or 'pageCount' == arg or 'vgpuCount' == arg or 'infoCount' == arg

def try_to_write_device_handler(out_file, func_name, parameters, struct_cannot_gen_parsers, all_enum):
    # We try to store the latest version of struct. So that we parse nvmlDeviceGetMemoryInfo_v2 automatically.
    # For nvmlDeviceGetMemoryInfo, we hand write a GetWrapper to copy the v2 value to v1 struct
    if func_name == "nvmlDeviceGetMemoryInfo":
        return False
    if len(parameters) < 1 or parameters[0].type != "nvmlDevice_t":
        return False
    key, version = get_function_info_from_name(func_name)
    if version > 1:
        parser_name = f"{key}_v{version}Parser"
    else:
        parser_name = f"{key}Parser"
    if len(parameters) == 1:
        print_generator_source_info(out_file, 2)
        print_body_line(f'{{"{key}", NvmlReturnParser}},', out_file, 1)
        return True
    if len(parameters) == 2:
        if remove_ptr_if_any(parameters[1].type) in struct_cannot_gen_parsers:
            return False
        if is_basic_type(parameters[1].type) or is_basic_ptr_type(parameters[1].type):
            print_generator_source_info(out_file, 2)
            print_body_line(f'{{"{key}", BasicTypeParser<{remove_ptr_if_any(parameters[1].type)}, {remove_ptr_if_any(parameters[1].type)}>}},', out_file, 1)
            return True
        if is_enum_type(remove_ptr_if_any(parameters[1].type), all_enum):
            print_generator_source_info(out_file, 2)
            print_body_line(f'{{"{key}", BasicTypeParser<int, {remove_ptr_if_any(parameters[1].type)}>}},', out_file, 1)
            return True
        print_generator_source_info(out_file, 2)
        print_body_line(f'{{"{key}", {type_name_to_parser_name(remove_ptr_if_any(parameters[1].type))}}},', out_file, 1)
        return True
    if len(parameters) == 3:
        if not is_pointer_type(parameters[1].type) and not arg_is_count(parameters[1].name):
            return False
        if parameters[1].type == "char *" and parameters[2].type == "unsigned int":
            # e.g. nvmlDeviceGetVbiosVersion
            print_generator_source_info(out_file, 2)
            print_body_line(f'{{"{key}", BasicTypeParser<std::string, std::string>}},', out_file, 1)
            return True
        if is_basic_ptr_type(parameters[1].type) and is_basic_ptr_type(parameters[2].type):
            # e.g. nvmlDeviceGetGspFirmwareMode
            print_generator_source_info(out_file, 2)
            print_body_line(f'{{"{key}", {parser_name}}},', out_file, 1)
            return True
        if remove_ptr_if_any(parameters[1].type) in all_enum and remove_ptr_if_any(parameters[2].type) in all_enum:
            # e.g. nvmlDeviceGetGpuOperationMode
            print_generator_source_info(out_file, 2)
            print_body_line(f'{{"{key}", {parser_name}}},', out_file, 1)
            return True
        if arg_is_count(parameters[1].name) and (parameters[1].type == "unsigned int *" or parameters[1].type == "unsigned int"):
            # e.g. nvmlDeviceGetFBCSessions
            print_generator_source_info(out_file, 2)
            print_body_line(f'{{"{key}", {parser_name}}},', out_file, 1)
            return True
    if len(parameters) == 4:
        all_basic_type = is_basic_ptr_type(parameters[1].type) and is_basic_ptr_type(parameters[2].type) and is_basic_ptr_type(parameters[3].type)
        all_enum_type = is_enum_type(remove_ptr_if_any(parameters[1].type), all_enum) and\
                is_enum_type(remove_ptr_if_any(parameters[2].type), all_enum) and is_enum_type(remove_ptr_if_any(parameters[3].type), all_enum)
        if not all_basic_type and not all_enum_type:
            return False
        print_generator_source_info(out_file, 2)
        print_body_line(f'{{"{key}", {key}Parser}},', out_file, 1)
        return True
    return False

def try_to_write_gpu_instance_handler(out_file, func_name, parameters, struct_cannot_gen_parsers, all_enum):
    if len(parameters) < 1 or parameters[0].type != "nvmlGpuInstance_t":
        return False
    key, version = get_function_info_from_name(func_name)
    if version != 1:
        key = f"{key}_v{version}"
    if len(parameters) == 2:
        if remove_ptr_if_any(parameters[1].type) in struct_cannot_gen_parsers:
            return False
        if is_basic_type(parameters[1].type) or is_basic_ptr_type(parameters[1].type):
            print_generator_source_info(out_file, 2)
            print_body_line(f'{{"{key}", BasicTypeParser<{remove_ptr_if_any(parameters[1].type)}, {remove_ptr_if_any(parameters[1].type)}>}},', out_file, 1)
            return True
        if is_enum_type(remove_ptr_if_any(parameters[1].type), all_enum):
            print_generator_source_info(out_file, 2)
            print_body_line(f'{{"{key}", BasicTypeParser<int, {remove_ptr_if_any(parameters[1].type)}>}},', out_file, 1)
            return True
        print_generator_source_info(out_file, 2)
        print_body_line(f'{{"{key}", {type_name_to_parser_name(remove_ptr_if_any(parameters[1].type))}}},', out_file, 1)
        return True
    return False

def try_to_write_compute_instance_handler(out_file, func_name, parameters, struct_cannot_gen_parsers, all_enum):
    if len(parameters) < 1 or parameters[0].type != "nvmlComputeInstance_t":
        return False
    key, version = get_function_info_from_name(func_name)
    if version != 1:
        key = f"{key}_v{version}"
    if len(parameters) == 2:
        if remove_ptr_if_any(parameters[1].type) in struct_cannot_gen_parsers:
            return False
        if is_basic_type(parameters[1].type) or is_basic_ptr_type(parameters[1].type):
            print_generator_source_info(out_file, 2)
            print_body_line(f'{{"{key}", BasicTypeParser<{remove_ptr_if_any(parameters[1].type)}, {remove_ptr_if_any(parameters[1].type)}>}},', out_file, 1)
            return True
        if is_enum_type(remove_ptr_if_any(parameters[1].type), all_enum):
            print_generator_source_info(out_file, 2)
            print_body_line(f'{{"{key}", BasicTypeParser<int, {remove_ptr_if_any(parameters[1].type)}>}},', out_file, 1)
            return True
        print_generator_source_info(out_file, 2)
        print_body_line(f'{{"{key}", {type_name_to_parser_name(remove_ptr_if_any(parameters[1].type))}}},', out_file, 1)
        return True
    return False

def try_to_write_vgpu_type_handler(out_file, func_name, parameters, struct_cannot_gen_parsers, all_enum):
    if len(parameters) < 1 or parameters[0].type != "nvmlVgpuTypeId_t":
        return False
    key, version = get_function_info_from_name(func_name)
    if version != 1:
        key = f"{key}_v{version}"
    parser_name = f"{key}Parser"
    if len(parameters) == 2:
        if remove_ptr_if_any(parameters[1].type) in struct_cannot_gen_parsers:
            return False
        if is_basic_type(parameters[1].type) or is_basic_ptr_type(parameters[1].type):
            # e.g. nvmlVgpuTypeGetFramebufferSize
            print_generator_source_info(out_file, 2)
            print_body_line(f'{{"{key}", BasicTypeParser<{remove_ptr_if_any(parameters[1].type)}, {remove_ptr_if_any(parameters[1].type)}>}},', out_file, 1)
            return True
        if is_enum_type(remove_ptr_if_any(parameters[1].type), all_enum):
            print_generator_source_info(out_file, 2)
            print_body_line(f'{{"{key}", BasicTypeParser<int, {remove_ptr_if_any(parameters[1].type)}>}},', out_file, 1)
            return True
        print_generator_source_info(out_file, 2)
        print_body_line(f'{{"{key}", {type_name_to_parser_name(remove_ptr_if_any(parameters[1].type))}}},', out_file, 1)
        return True
    if len(parameters) == 3:
        if parameters[1].type == "char *" and parameters[2].type == "unsigned int":
            # e.g. nvmlVgpuTypeGetLicense
            print_generator_source_info(out_file, 2)
            print_body_line(f'{{"{key}", BasicTypeParser<std::string, std::string>}},', out_file, 1)
            return True
        if parameters[1].type == "char *" and parameters[2].type == "unsigned int *":
            # e.g. nvmlVgpuTypeGetName
            print_generator_source_info(out_file, 2)
            print_body_line(f'{{"{key}", BasicTypeParser<std::string, std::string>}},', out_file, 1)
            return True
        if is_basic_ptr_type(parameters[1].type) and is_basic_ptr_type(parameters[2].type):
            # e.g. nvmlVgpuTypeGetDeviceID
            print_generator_source_info(out_file, 2)
            print_body_line(f'{{"{key}", {parser_name}}},', out_file, 1)
            return True
    if len(parameters) == 4 and func_name == "nvmlVgpuTypeGetResolution":
        # this function in PyNVML only has one input...
        # hardcode here as it is a special case which does not have extra key but with 4 parameters
        print_generator_source_info(out_file, 2)
        print_body_line(f'{{"{key}", {parser_name}}},', out_file, 1)
        return True
    return False

def try_to_write_vgpu_type_extra_key_handler(out_file, func_name, parameters, struct_cannot_gen_parsers, all_enum):
    if func_name == "nvmlVgpuTypeGetResolution":
        # this function in PyNVML only has one input...
        return False
    key, version = get_function_info_from_name(func_name)
    if version != 1:
        key = f"{key}_v{version}"
    if len(parameters) < 3:
        return False
    if parameters[0].type != "nvmlVgpuTypeId_t":
        return False
    if arg_is_count(parameters[1].name):
        # array or string length is not extra key
        return False
    key2_parser = None
    value_parser = None
    if is_enum_type(parameters[1].type, all_enum):
        key2_parser = f"BasicKeyTypeParser<int, {parameters[1].type}>"
    elif is_basic_type(parameters[1].type):
        key2_parser = f"BasicKeyTypeParser<{parameters[1].type}, {parameters[1].type}>"
    if len(parameters) == 3:
        if is_basic_ptr_type(parameters[2].type):
            # e.g. nvmlVgpuTypeGetCapabilities
            value_parser = f"BasicTypeParser<{remove_ptr_if_any(parameters[2].type)}, {remove_ptr_if_any(parameters[2].type)}>"
    if len(parameters) == 4:
        if is_basic_ptr_type(parameters[2].type) and is_basic_ptr_type(parameters[3].type):
            value_parser = f"{key}Parser"
    if key2_parser is None or value_parser is None:
        return False
    print_generator_source_info(out_file, 2)
    print_body_line(f'{{"{key}", {{{key2_parser}, {value_parser}}}}},', out_file, 1)
    return False

def try_to_write_vgpu_instance_handler(out_file, func_name, parameters, struct_cannot_gen_parsers, all_enum):
    if len(parameters) < 1 or parameters[0].type != "nvmlVgpuInstance_t":
        return False
    key, _ = get_function_info_from_name(func_name)
    parser_name = f"{key}Parser"
    if len(parameters) == 2:
        if remove_ptr_if_any(parameters[1].type) in struct_cannot_gen_parsers:
            return False
        if is_basic_type(parameters[1].type) or is_basic_ptr_type(parameters[1].type):
            # e.g. nvmlVgpuInstanceGetFbUsage
            print_generator_source_info(out_file, 2)
            print_body_line(f'{{"{key}", BasicTypeParser<{remove_ptr_if_any(parameters[1].type)}, {remove_ptr_if_any(parameters[1].type)}>}},', out_file, 1)
            return True
        if is_enum_type(remove_ptr_if_any(parameters[1].type), all_enum):
            # e.g. nvmlVgpuInstanceGetType
            print_generator_source_info(out_file, 2)
            print_body_line(f'{{"{key}", BasicTypeParser<int, {remove_ptr_if_any(parameters[1].type)}>}},', out_file, 1)
            return True
        print_generator_source_info(out_file, 2)
        print_body_line(f'{{"{key}", {type_name_to_parser_name(remove_ptr_if_any(parameters[1].type))}}},', out_file, 1)
        return True
    if len(parameters) == 3:
        if parameters[1].type == "char *" and parameters[2].type == "unsigned int":
            # e.g. nvmlVgpuInstanceGetUUID
            print_generator_source_info(out_file, 2)
            print_body_line(f'{{"{key}", BasicTypeParser<std::string, std::string>}},', out_file, 1)
            return True
        if parameters[1].type == "char *" and parameters[2].type == "unsigned int *":
            # e.g. nvmlVgpuInstanceGetGpuPciId
            print_generator_source_info(out_file, 2)
            print_body_line(f'{{"{key}", BasicTypeParser<std::string, std::string>}},', out_file, 1)
            return True
        if is_basic_ptr_type(parameters[1].type) and is_basic_ptr_type(parameters[2].type):
            print_generator_source_info(out_file, 2)
            print_body_line(f'{{"{key}", {parser_name}}},', out_file, 1)
            return True
        if is_basic_type(parameters[1].type) and parameters[2].type not in struct_cannot_gen_parsers:
            # e.g. nvmlVgpuInstanceGetAccountingStats
            print_generator_source_info(out_file, 2)
            print_body_line(f'{{"{key}", {parser_name}}},', out_file, 1)
            return True
        if arg_is_count(parameters[1].name) and (parameters[1].type == "unsigned int *" or parameters[1].type == "unsigned int"):
            # e.g. nvmlVgpuInstanceGetEncoderSessions
            print_generator_source_info(out_file, 2)
            print_body_line(f'{{"{key}", {parser_name}}},', out_file, 1)
            return True
    if len(parameters) == 4:
        if is_basic_ptr_type(parameters[1].type) and is_basic_ptr_type(parameters[2].type) and is_basic_ptr_type(parameters[3].type):
            # e.g. nvmlVgpuInstanceGetEncoderStats
            print_generator_source_info(out_file, 2)
            print_body_line(f'{{"{key}", {parser_name}}},', out_file, 1)
            return True
    return False

def try_to_write_device_extra_key_handler(out_file, func_name, parameters, struct_cannot_gen_parsers, all_enum):
    key, _ = get_function_info_from_name(func_name)
    if len(parameters) < 3:
        return False
    if parameters[0].type != "nvmlDevice_t":
        return False
    if len(parameters) == 3:
        if arg_is_count(parameters[1].name):
            return False
        key2_parser = None
        value_parser = None
        if is_enum_type(parameters[1].type, all_enum):
            # e.g. nvmlDeviceGetClockInfo
            key2_parser = f"BasicKeyTypeParser<int, {parameters[1].type}>"
        elif is_basic_type(parameters[1].type):
            # e.g. nvmlDeviceGetGpuInstanceProfileInfo
            key2_parser = f"BasicKeyTypeParser<{parameters[1].type}, {parameters[1].type}>"
        if is_basic_ptr_type(parameters[2].type):
            # e.g. nvmlDeviceGetClockInfo
            value_parser = f"BasicTypeParser<{remove_ptr_if_any(parameters[2].type)}, {remove_ptr_if_any(parameters[2].type)}>"
        elif is_enum_type(remove_ptr_if_any(parameters[2].type), all_enum):
            # e.g. nvmlDeviceGetAPIRestriction
            value_parser = f"BasicTypeParser<int, {remove_ptr_if_any(parameters[2].type)}>"
        elif remove_ptr_if_any(parameters[2].type) not in struct_cannot_gen_parsers:
            # e.g. nvmlDeviceGetThermalSettings
            value_parser = type_name_to_parser_name(remove_ptr_if_any(parameters[2].type))
        if key2_parser is None or value_parser is None:
            return False
        print_generator_source_info(out_file, 2)
        print_body_line(f'{{"{key}", {{{key2_parser}, {value_parser}}}}},', out_file, 1)
        return True
    if len(parameters) == 4:
        key2_parser = None
        value_parser = None
        if is_enum_type(parameters[1].type, all_enum):
            # e.g. nvmlDeviceGetInforomVersion
            key2_parser = f"BasicKeyTypeParser<int, {parameters[1].type}>"
        if is_basic_type(parameters[1].type):
            # e.g. nvmlDeviceGetMemoryAffinity
            key2_parser = f"BasicKeyTypeParser<{parameters[1].type}, {parameters[1].type}>"
        if parameters[2].type == "char *" and parameters[3].type == "unsigned int":
            # e.g. nvmlDeviceGetInforomVersion
            value_parser = "BasicTypeParser<std::string, std::string>"
        if is_enum_type(parameters[3].type, all_enum) and arg_is_count(parameters[1].name):
            # array type with extra key in the last (e.g. nvmlDeviceGetCpuAffinityWithinScope)
            key2_parser = f"BasicKeyTypeParser<int, {parameters[3].type}>"
            if is_basic_ptr_type(parameters[2].type):
                value_parser = f"{key}Parser"
        if parameters[2].type == "unsigned int *" and is_basic_ptr_type(parameters[3].type):
            # e.g. nvmlDeviceGetSupportedGraphicsClocks
            value_parser = f"{key}Parser"
        if key2_parser is None or value_parser is None:
            return False
        print_generator_source_info(out_file, 2)
        print_body_line(f'{{"{key}", {{{key2_parser}, {value_parser}}}}},', out_file, 1)
        return True
    return False

def try_to_write_device_three_keys_handler(out_file, func_name, parameters, struct_cannot_gen_parsers, all_enum):
    key, version = get_function_info_from_name(func_name)
    if version != 1:
        key = f"{key}_v{version}"
    if len(parameters) < 4:
        return False
    if parameters[0].type != "nvmlDevice_t":
        return False
    key1_parser = None
    key2_parser = None
    value1_parser = None
    if is_enum_type(parameters[1].type, all_enum):
        # e.g. nvmlDeviceGetDetailedEccErrors
        key1_parser = f"BasicKeyTypeParser<int, {parameters[1].type}>"
    elif is_basic_type(parameters[1].type):
        # e.g. nvmlDeviceGetNvLinkCapability
        key1_parser = f"BasicKeyTypeParser<{parameters[1].type}, {parameters[1].type}>"
    if is_enum_type(parameters[2].type, all_enum):
        # e.g. nvmlDeviceGetDetailedEccErrors
        key2_parser = f"BasicKeyTypeParser<int, {parameters[2].type}>"
    elif is_basic_type(parameters[2].type):
        # e.g. nvmlDeviceGetNvLinkUtilizationCounter
        key2_parser = f"BasicKeyTypeParser<{parameters[2].type}, {parameters[2].type}>"
    if is_basic_ptr_type(parameters[3].type):
        # e.g. nvmlDeviceGetNvLinkCapability
        value1_parser = f"BasicTypeParser<{remove_ptr_if_any(parameters[3].type)}, {remove_ptr_if_any(parameters[3].type)}>"
    elif remove_ptr_if_any(parameters[3].type) not in struct_cannot_gen_parsers:
        # e.g. nvmlDeviceGetDetailedEccErrors
        value1_parser = type_name_to_parser_name(remove_ptr_if_any(parameters[3].type))
    if len(parameters) == 4:
        if key1_parser is None or key2_parser is None or value1_parser is None:
            return False
        print_generator_source_info(out_file, 2)
        print_body_line(f'{{"{key}", {{{key1_parser}, {key2_parser}, {value1_parser}}}}},', out_file, 1)
        return True
    if len(parameters) == 5:
        can_parse_value2 = False
        if is_basic_ptr_type(parameters[4].type):
            # e.g. nvmlDeviceGetNvLinkUtilizationCounter
            can_parse_value2 = True
        elif remove_ptr_if_any(parameters[4].type) not in struct_cannot_gen_parsers:
            can_parse_value2 = True
        if key1_parser is None or key2_parser is None or value1_parser is None or not can_parse_value2:
            return False
        value_parser = f"{key}Parser"
        print_generator_source_info(out_file, 2)
        print_body_line(f'{{"{key}", {{{key1_parser}, {key2_parser}, {value_parser}}}}},', out_file, 1)
        return True
    return False

def try_to_write_gpu_instance_extra_key_handler(out_file, func_name, parameters, struct_cannot_gen_parsers, all_enum):
    key, version = get_function_info_from_name(func_name)
    if version != 1:
        key = f"{key}_v{version}"
    if len(parameters) < 3:
        return False
    if parameters[0].type != "nvmlGpuInstance_t":
        return False
    if len(parameters) == 3:
        if arg_is_count(parameters[1].name):
            return False
        key2_parser = None
        value_parser = None
        if is_enum_type(parameters[1].type, all_enum):
            key2_parser = f"BasicKeyTypeParser<int, {parameters[1].type}>"
        elif is_basic_type(parameters[1].type):
            # e.g. nvmlGpuInstanceGetComputeInstanceRemainingCapacity
            key2_parser = f"BasicKeyTypeParser<{parameters[1].type}, {parameters[1].type}>"
        if is_basic_ptr_type(parameters[2].type):
            # e.g. nvmlGpuInstanceGetComputeInstanceRemainingCapacity
            value_parser = f"BasicTypeParser<{remove_ptr_if_any(parameters[2].type)}, {remove_ptr_if_any(parameters[2].type)}>"
        if key2_parser is None or value_parser is None:
            return False
        print_generator_source_info(out_file, 2)
        print_body_line(f'{{"{key}", {{{key2_parser}, {value_parser}}}}},', out_file, 1)
        return True
    return False

def try_to_write_gpu_instance_three_keys_handler(out_file, func_name, parameters, struct_cannot_gen_parsers, all_enum):
    key, version = get_function_info_from_name(func_name)
    if version != 1:
        key = f"{key}_v{version}"
    if len(parameters) < 4:
        return False
    if parameters[0].type != "nvmlGpuInstance_t":
        return False
    if len(parameters) == 4:
        key1_parser = None
        key2_parser = None
        value_parser = None
        if is_enum_type(parameters[1].type, all_enum):
            key1_parser = f"BasicKeyTypeParser<int, {parameters[1].type}>"
        elif is_basic_type(parameters[1].type):
            # e.g. nvmlGpuInstanceGetComputeInstanceProfileInfo
            key1_parser = f"BasicKeyTypeParser<{parameters[1].type}, {parameters[1].type}>"
        if is_enum_type(parameters[2].type, all_enum):
            key2_parser = f"BasicKeyTypeParser<int, {parameters[2].type}>"
        elif is_basic_type(parameters[2].type):
            # e.g. nvmlGpuInstanceGetComputeInstanceProfileInfo
            key2_parser = f"BasicKeyTypeParser<{parameters[2].type}, {parameters[2].type}>"
        if is_basic_ptr_type(parameters[3].type):
            value_parser = f"BasicTypeParser<{remove_ptr_if_any(parameters[3].type)}, {remove_ptr_if_any(parameters[3].type)}>"
        elif remove_ptr_if_any(parameters[3].type) not in struct_cannot_gen_parsers:
            # e.g. nvmlGpuInstanceGetComputeInstanceProfileInfo
            value_parser = type_name_to_parser_name(remove_ptr_if_any(parameters[3].type))
        if key1_parser is None or key2_parser is None or value_parser is None:
            return False
        print_generator_source_info(out_file, 2)
        print_body_line(f'{{"{key}", {{{key1_parser}, {key2_parser}, {value_parser}}}}},', out_file, 1)
        return True
    return False

def try_to_write_general_handler(out_file, func_name, parameters, struct_cannot_gen_parsers, all_enum):
    key, version = get_function_info_from_name(func_name)
    if key is None:
        return False
    if version != 1:
        key = f"{key}_v{version}"
    if len(parameters) == 1:
        if is_basic_ptr_type(parameters[0].type):
            # e.g. nvmlSystemGetCudaDriverVersion
            print_generator_source_info(out_file, 2)
            print_body_line(f'{{"{key}", BasicTypeParser<{remove_ptr_if_any(parameters[0].type)}, {remove_ptr_if_any(parameters[0].type)}>}},', out_file, 1)
            return True
        if remove_ptr_if_any(parameters[0].type) in all_enum:
            print_generator_source_info(out_file, 2)
            print_body_line(f'{{"{key}", BasicTypeParser<int, {remove_ptr_if_any(parameters[0].type)}>}},', out_file, 1)
            return True
        if remove_ptr_if_any(parameters[0].type) not in struct_cannot_gen_parsers:
            # e.g. nvmlSystemGetConfComputeState
            print_generator_source_info(out_file, 2)
            print_body_line(f'{{"{key}", {type_name_to_parser_name(remove_ptr_if_any(parameters[0].type))}}},', out_file, 1)
            return True
        return False
    if len(parameters) == 2:
        if parameters[0].type != "char *" or parameters[1].type != "unsigned int":
            return False
        # e.g. nvmlSystemGetNVMLVersion
        print_generator_source_info(out_file, 2)
        print_body_line(f'{{"{key}", BasicTypeParser<std::string, std::string>}},', out_file, 1)
        return True
    return False

def struct_name_to_deserializer_name(struct_name):
    return f"{struct_name}Deserializer"

def get_cannot_write_deserializer_struct(struct_with_member):
    cannot_write_deserializer_struct = {}
    cannot_write_deserializer_struct["nvmlValue_t"] = True
    cannot_write_deserializer_struct["nvmlDevice_t"] = True
    for struct_name, member_list in struct_with_member.items():
        can = True
        for member_type, _ in member_list:
            if "unnamed struct" in member_type:
                cannot_write_deserializer_struct[struct_name] = True
                continue
            if "nvmlDevice_t" == member_type:
                can = False
                break
            if "nvmlValue_t" == member_type:
                can = False
                break
        if not can:
            cannot_write_deserializer_struct[struct_name] = True
    # Do it again as some of the member may be unable to generate
    for struct_name, member_list in struct_with_member.items():
        for member_type, _ in member_list:
            pure_type = member_type.split("[")[0]
            if pure_type in cannot_write_deserializer_struct:
                cannot_write_deserializer_struct[struct_name] = True
    return cannot_write_deserializer_struct

def write_deserializer_declare(out_file, struct_with_member, cannot_write_deserializer_struct):
    print_generator_source_info(out_file, 0)
    for struct_name, _ in struct_with_member.items():
        if struct_name in cannot_write_deserializer_struct:
            continue
        func_name = struct_name_to_deserializer_name(struct_name)
        out_file.write(f"{struct_name} *{func_name}(const YAML::Node &node);\n")
    out_file.write("\n")

def write_deserializer_definition(out_file, struct_with_member, all_enum, cannot_write_deserializer_struct):
    for struct_name, member_list in struct_with_member.items():
        if struct_name in cannot_write_deserializer_struct:
            continue
        func_name = struct_name_to_deserializer_name(struct_name)
        print_generator_source_info(out_file, 0)
        out_file.write(f"{struct_name} *{func_name}(const YAML::Node &node)\n{{\n")
        variable_name = lowercase_first_letter(struct_name[4:-2])
        print_body_line(f"auto *{variable_name} = reinterpret_cast<{struct_name} *>(malloc(sizeof({struct_name})));", out_file, 0)
        print_body_line(f"if ({variable_name} == nullptr)", out_file, 0)
        print_body_line('{', out_file, 0)
        print_body_line('return nullptr;', out_file, 1)
        print_body_line('}', out_file, 0)
        print_body_line(f'memset({variable_name}, 0, sizeof(*{variable_name}));', out_file, 0)
        tmp_cnt = 0
        for member_type, name in member_list:
            if struct_name == "nvmlAccountingStats_t" and name == "reserved":
                continue
            if struct_name == "nvmlVgpuMetadata_t" and (name == "reserved" or name == "opaqueData"):
                continue
            if struct_name == "nvmlVgpuPgpuMetadata_t" and (name == "reserved" or name == "opaqueData"):
                continue
            # PyNVML does not return usedGpuCcProtectedMemory in nvmlProcessInfo_t
            if struct_name == "nvmlProcessInfo_t" and name == "usedGpuCcProtectedMemory":
                continue
            print_body_line(f'if (node["{name}"])', out_file, 0)
            print_body_line('{', out_file, 0)
            if is_basic_type(member_type):
                # basic type uses build-in yaml-cpp convertion method
                print_generator_source_info(out_file, 2)
                print_body_line(f'{variable_name}->{name} = node["{name}"].as<{member_type}>();', out_file, 1)
            elif is_basic_ptr_type(member_type):
                # Basic pointer types can involve arrays with a count stored in a separate struct member.
                type_name = remove_ptr_if_any(member_type)
                print_generator_source_info(out_file, 2)
                tmp_varialbe = f"tmp{tmp_cnt}"
                tmp_cnt += 1
                print_body_line(f'auto const {tmp_varialbe} = node["{name}"].as<std::vector<{type_name}>>();', out_file, 1)
                print_body_line(f'{variable_name}->{name} = reinterpret_cast<{type_name} *>(malloc(sizeof({type_name}) * {tmp_varialbe}.size()));', out_file, 1)
                print_body_line(f'for (unsigned int ii = 0; ii < {tmp_varialbe}.size(); ++ii)', out_file, 1)
                print_body_line('{', out_file, 1)
                print_body_line(f'{variable_name}->{name}[ii] = {tmp_varialbe}[ii];', out_file, 2)
                print_body_line('}', out_file, 1)
            elif is_enum_type(member_type, all_enum):
                # enum type uses build-in yaml-cpp convertion method to covert to int and cast to enum type
                print_generator_source_info(out_file, 2)
                print_body_line(f'{variable_name}->{name} = static_cast<{member_type}>(node["{name}"].as<int>());', out_file, 1)
            elif "char[" in member_type:
                # c-string
                print_generator_source_info(out_file, 2)
                print_body_line(f'auto {name} = node["{name}"].as<std::string>();', out_file, 1)
                if (name == "clusterUuid") and ((struct_name == "nvmlGpuFabricInfo_t") or (struct_name == "nvmlGpuFabricInfoV_t")):
                    print_body_line(f'NvmlInjectionUuid uuid;', out_file, 1)
                    print_body_line(f'NvmlUuidParse({name}, uuid);', out_file, 1)
                    print_body_line(f'std::memcpy(&{variable_name}->{name}, uuid, sizeof({variable_name}->{name}));', out_file, 1)
                else:
                    print_body_line(f'std::memcpy(&{variable_name}->{name}, {name}.data(), sizeof({variable_name}->{name}));', out_file, 1)
            elif "[" in member_type:
                # custom type array
                # for each element, we call corresponding deserializer and memcpy the result to our struct
                type_name = member_type.split("[")[0]
                print_generator_source_info(out_file, 2)
                if is_basic_type(type_name):
                    tmp_varialbe = f"tmp{tmp_cnt}"
                    tmp_cnt += 1
                    print_body_line(f'auto const {tmp_varialbe} = node["{name}"].as<std::vector<{type_name}>>();', out_file, 1)
                    print_body_line(f'for (unsigned int ii = 0; ii < {tmp_varialbe}.size(); ++ii)', out_file, 1)
                    print_body_line('{', out_file, 1)
                    print_body_line(f'{variable_name}->{name}[ii] = {tmp_varialbe}[ii];', out_file, 2)
                    print_body_line('}', out_file, 1)
                else:
                    print_body_line('int idx = 0;', out_file, 1)
                    print_body_line(f'int size = std::min(node["{name}"].size(), sizeof({variable_name}->{name}) / sizeof({type_name}));', out_file, 1)
                    print_body_line(f'for (YAML::const_iterator it = node["{name}"].begin(); it != node["{name}"].end(); ++it)', out_file, 1)
                    print_body_line('{', out_file, 1)
                    deserialzier_name = struct_name_to_deserializer_name(type_name)
                    print_body_line(f'auto *tmp = {deserialzier_name}(*it);', out_file, 2)
                    print_body_line('if (!tmp)', out_file, 2)
                    print_body_line('{', out_file, 2)
                    print_body_line(f'free({variable_name});', out_file, 3)
                    print_body_line('return nullptr;', out_file, 3)
                    print_body_line('}', out_file, 2)
                    print_body_line('if (idx >= size)', out_file, 2)
                    print_body_line('{', out_file, 2)
                    print_body_line('break;', out_file, 3)
                    print_body_line('}', out_file, 2)
                    print_body_line(f'std::memcpy(&{variable_name}->{name}[idx++], tmp, sizeof({type_name}));', out_file, 2)
                    print_body_line('free(tmp);', out_file, 2)
                    print_body_line('}', out_file, 1)
            elif is_pointer_type(member_type):
                type_name = remove_ptr_if_any(member_type)
                deserialzier_name = struct_name_to_deserializer_name(type_name)
                tmp_varialbe = f"tmp{tmp_cnt}"
                tmp_cnt += 1
                print_generator_source_info(out_file, 2)
                if type_name in cannot_write_deserializer_struct or type_name not in struct_with_member:
                    print_body_line(f'NVML_LOG_ERR("Skipping loading {name} for struct {struct_name}");', out_file, 1)
                else:
                    print_body_line(f'auto *{tmp_varialbe} = {deserialzier_name}(node["{name}"]);', out_file, 1)
                    print_body_line(f'if (!{tmp_varialbe})', out_file, 1)
                    print_body_line('{', out_file, 1)
                    print_body_line(f'free({variable_name});', out_file, 2)
                    print_body_line('return nullptr;', out_file, 2)
                    print_body_line('}', out_file, 1)
                    print_body_line(f'std::memcpy({variable_name}->{name}, {tmp_varialbe}, sizeof({type_name}));', out_file, 1)
                    print_body_line(f'free({tmp_varialbe});', out_file, 1)
            else:
                # custom type
                # we call corresponding deserializer and memcpy the result to our struct
                deserialzier_name = struct_name_to_deserializer_name(member_type)
                tmp_varialbe = f"tmp{tmp_cnt}"
                tmp_cnt += 1
                print_generator_source_info(out_file, 2)
                if member_type in cannot_write_deserializer_struct or member_type not in struct_with_member:
                    print_body_line(f'NVML_LOG_ERR("Skipping loading {name} for struct {struct_name}");', out_file, 1)
                else:
                    print_body_line(f'auto *{tmp_varialbe} = {deserialzier_name}(node["{name}"]);', out_file, 1)
                    print_body_line(f'if (!{tmp_varialbe})', out_file, 1)
                    print_body_line('{', out_file, 1)
                    print_body_line(f'free({variable_name});', out_file, 2)
                    print_body_line('return nullptr;', out_file, 2)
                    print_body_line('}', out_file, 1)
                    print_body_line(f'std::memcpy(&{variable_name}->{name}, {tmp_varialbe}, sizeof({member_type}));', out_file, 1)
                    print_body_line(f'free({tmp_varialbe});', out_file, 1)
            print_body_line('}', out_file, 0)
            # PyNVML will return function not found in nvmlDeviceGetMemoryInfo_v2
            if struct_name == "nvmlMemory_v2_t" and (name == "version" or name == "reserved"):
                continue
            print_body_line('else', out_file, 0)
            print_body_line('{', out_file, 0)
            print_body_line(f'NVML_LOG_ERR("missing {name} for struct {struct_name}");', out_file, 1)
            print_body_line('}', out_file, 0)
        print_body_line(f'return {variable_name};', out_file, 0)
        out_file.write("}\n\n")

def write_basic_type_parser(out_file):
    print_generator_source_info(out_file, 0)
    out_file.write("template <typename underlyType, typename resultType>\n")
    out_file.write("std::optional<NvmlFuncReturn> BasicTypeParser(const YAML::Node &node)\n{\n")
    print_body_line("if (!node || !node[\"FunctionReturn\"])", out_file, 0)
    print_body_line('{', out_file, 0)
    print_body_line('return NvmlFuncReturn(NVML_ERROR_UNKNOWN);', out_file, 1)
    print_body_line('}', out_file, 0)
    print_body_line('auto ret = static_cast<nvmlReturn_t>(node[\"FunctionReturn\"].as<int>(NVML_ERROR_UNKNOWN));', out_file, 0)
    print_body_line("if (!node[\"ReturnValue\"])", out_file, 0)
    print_body_line('{', out_file, 0)
    print_body_line('return NvmlFuncReturn(ret);', out_file, 1)
    print_body_line('}', out_file, 0)
    print_body_line('auto value = static_cast<resultType>(node["ReturnValue"].as<underlyType>());', out_file, 0)
    print_body_line('return NvmlFuncReturn(ret, std::move(value));', out_file, 0)
    out_file.write("}\n\n")

def write_basic_key_type_parser(out_file):
    print_generator_source_info(out_file, 0)
    out_file.write("template <typename underlyType, typename resultType>\n")
    out_file.write("std::optional<InjectionArgument> BasicKeyTypeParser(const YAML::Node &node)\n{\n")
    print_body_line("if (!node)", out_file, 0)
    print_body_line('{', out_file, 0)
    print_body_line('return std::nullopt;', out_file, 1)
    print_body_line('}', out_file, 0)
    print_body_line('return static_cast<resultType>(node.as<underlyType>());', out_file, 0)
    out_file.write("}\n\n")

# some functions only have return without attribute (e.g. nvmlDeviceValidateInforom)
def write_nvml_return_parser(out_file):
    print_generator_source_info(out_file, 0)
    out_file.write("std::optional<NvmlFuncReturn> NvmlReturnParser(const YAML::Node &node)\n{\n")
    print_body_line("if (!node || !node[\"FunctionReturn\"])", out_file, 0)
    print_body_line('{', out_file, 0)
    print_body_line('return NvmlFuncReturn(NVML_ERROR_UNKNOWN);', out_file, 1)
    print_body_line('}', out_file, 0)
    print_body_line('auto ret = static_cast<nvmlReturn_t>(node[\"FunctionReturn\"].as<int>(NVML_ERROR_UNKNOWN));', out_file, 0)
    print_body_line('return NvmlFuncReturn(ret);', out_file, 0)
    out_file.write("}\n\n")

def write_known_struct_parser(out_file, struct_with_member, struct_cannot_gen_parsers, all_enum, written_parser):
    for struct_name, _ in struct_with_member.items():
        if struct_name in struct_cannot_gen_parsers:
            logging.debug(f"Unable to generate struct parser of [{struct_name}], you may write your own.")
            continue
        variable_name = lowercase_first_letter(struct_name[4:-2])
        parser_name = type_name_to_parser_name(struct_name)
        if parser_name in written_parser:
            continue
        print_generator_source_info(out_file, 0)
        out_file.write(f"std::optional<NvmlFuncReturn> {parser_name}(const YAML::Node &node)\n{{\n")
        print_body_line("if (!node || !node[\"FunctionReturn\"])", out_file, 0)
        print_body_line('{', out_file, 0)
        print_body_line('return NvmlFuncReturn(NVML_ERROR_UNKNOWN);', out_file, 1)
        print_body_line('}', out_file, 0)
        print_body_line('auto ret = static_cast<nvmlReturn_t>(node[\"FunctionReturn\"].as<int>(NVML_ERROR_UNKNOWN));', out_file, 0)
        print_body_line("if (!node[\"ReturnValue\"])", out_file, 0)
        print_body_line('{', out_file, 0)
        print_body_line('return NvmlFuncReturn(ret);', out_file, 1)
        print_body_line('}', out_file, 0)
        derserializer_name = struct_name_to_deserializer_name(struct_name)
        print_body_line(f"auto *{variable_name} = {derserializer_name}(node[\"ReturnValue\"]);", out_file, 0)
        print_body_line(f"if ({variable_name} == nullptr)", out_file, 0)
        print_body_line('{', out_file, 0)
        print_body_line('return std::nullopt;', out_file, 1)
        print_body_line('}', out_file, 0)
        print_body_line(f'return NvmlFuncReturn(ret, {{{variable_name}, true}});', out_file, 0)
        out_file.write("}\n\n")
        written_parser[parser_name] = True

def write_array_funcs_parser(out_file, functions, all_enum, written_parser, cannot_write_deserializer_struct):
    for func_name, parameters in functions.items():
        if len(parameters) < 3:
            continue
        key, version = get_function_info_from_name(func_name)
        if version != 1:
            key = f"{key}_v{version}"
        parser_name = f"{key}Parser"
        if parser_name in written_parser:
            continue
        can_write = False
        # Only handle the following two cases
        # 1. Direct array output (e.g. nvmlDeviceGetCpuAffinity)
        # 2. Has another extra key and direct array output (e.g. nvmlDeviceGetCpuAffinityWithinScope)
        if len(parameters) == 3:
            can_write = (arg_is_count(parameters[1].name) and parameters[1].type == "unsigned int" and remove_ptr_if_any(parameters[2].type) not in cannot_write_deserializer_struct)
            target_idx = len(parameters) - 1
        elif len(parameters) == 4:
            if (is_basic_type(parameters[1].type) or is_enum_type(parameters[1].type, all_enum)) and\
                  (arg_is_count(parameters[2].name) and (parameters[2].type == "unsigned int" or parameters[2].type == "unsigned int *")):
                can_write = True
                target_idx = len(parameters) - 1
            if (arg_is_count(parameters[1].name) and parameters[1].type == "unsigned int"):
                can_write = True
                target_idx = len(parameters) - 2
        if not can_write:
            continue
        print_generator_source_info(out_file, 0)
        out_file.write(f"std::optional<NvmlFuncReturn> {parser_name}(const YAML::Node &node)\n{{\n")
        print_body_line("if (!node || !node[\"FunctionReturn\"])", out_file, 0)
        print_body_line('{', out_file, 0)
        print_body_line('return NvmlFuncReturn(NVML_ERROR_UNKNOWN);', out_file, 1)
        print_body_line('}', out_file, 0)
        print_body_line('auto ret = static_cast<nvmlReturn_t>(node[\"FunctionReturn\"].as<int>(NVML_ERROR_UNKNOWN));', out_file, 0)
        print_body_line("if (!node[\"ReturnValue\"])", out_file, 0)
        print_body_line('{', out_file, 0)
        print_body_line('return NvmlFuncReturn(ret);', out_file, 1)
        print_body_line('}', out_file, 0)
        print_body_line('auto size = static_cast<unsigned int>(node[\"ReturnValue\"].size());', out_file, 0)
        print_body_line('unsigned int idx = 0;', out_file, 0)
        print_body_line(f'auto *valPtr = reinterpret_cast<{parameters[target_idx].type}>(malloc(sizeof({remove_ptr_if_any(parameters[target_idx].type)}) * size));', out_file, 0)
        print_body_line('for (YAML::const_iterator it = node[\"ReturnValue\"].begin(); it != node[\"ReturnValue\"].end(); ++it)', out_file, 0)
        print_body_line('{', out_file, 0)
        if is_basic_ptr_type(parameters[target_idx].type):
            # basic type uses build-in yaml-cpp convertion
            print_generator_source_info(out_file, 2)
            print_body_line(f'valPtr[idx++] = it->as<{remove_ptr_if_any(parameters[target_idx].type)}>();', out_file, 1)
        elif is_enum_type(remove_ptr_if_any(parameters[target_idx].type, all_enum)):
            # enum type uses build-in yaml-cpp int convertion and cast to corresponding enum
            print_generator_source_info(out_file, 2)
            print_body_line(f'valPtr[idx++] = static_cast<{remove_ptr_if_any(parameters[target_idx].type)}>(it->as<int>());', out_file, 1)
        else:
            # struct type calls deserializer and memcpy to our struct
            print_generator_source_info(out_file, 2)
            deserializer_name = struct_name_to_deserializer_name(remove_ptr_if_any(parameters[target_idx].type))
            print_body_line(f'auto *tmp = {deserializer_name}(*it);', out_file, 1)
            print_body_line('if (!tmp)', out_file, 1)
            print_body_line('{', out_file, 1)
            print_body_line('free(valPtr);', out_file, 2)
            print_body_line('return std::nullopt;', out_file, 2)
            print_body_line('}', out_file, 1)
            print_body_line(f'std::memcpy(&valPtr[idx++], tmp, sizeof({remove_ptr_if_any(parameters[target_idx].type)}));', out_file, 1)
            print_body_line('free(tmp);', out_file, 1)
        print_body_line('}', out_file, 0)
        if parameters[2].type == "unsigned int *":
            # if array size is also an output
            print_generator_source_info(out_file, 2)
            print_body_line('std::vector<InjectionArgument> args;', out_file, 0)
            print_body_line('args.emplace_back(idx);', out_file, 0)
            print_body_line('args.emplace_back(valPtr, idx, true);', out_file, 0)
            print_body_line('return NvmlFuncReturn(ret, args);', out_file, 0)
        else:
            print_generator_source_info(out_file, 2)
            print_body_line('return NvmlFuncReturn(ret, {valPtr, size, true});', out_file, 0)
        out_file.write("}\n\n")
        written_parser[parser_name] = True

def write_two_attrs_funcs_parser(out_file, functions, all_enum, written_parser, cannot_write_deserializer_struct):
    for func_name, parameters in functions.items():
        if len(parameters) < 3:
            continue
        key, version = get_function_info_from_name(func_name)
        if version != 1:
            key = f"{key}_v{version}"
        parser_name = f"{key}Parser"
        if parser_name in written_parser:
            continue
        pointer_cnt = 0
        for i in range(len(parameters) - 1, -1, -1):
            if is_pointer_type(parameters[i].type):
                pointer_cnt += 1
            else:
                break
        if pointer_cnt != 2:
            continue
        first_point_idx = len(parameters) - 2
        second_point_idx = len(parameters) - 1
        can_write = (arg_is_count(parameters[first_point_idx].name) and parameters[first_point_idx].type == "unsigned int *" and remove_ptr_if_any(parameters[second_point_idx].type) not in cannot_write_deserializer_struct) or\
                (is_basic_ptr_type(parameters[first_point_idx].type) and is_basic_ptr_type(parameters[second_point_idx].type)) or\
                (remove_ptr_if_any(parameters[first_point_idx].type) in all_enum and remove_ptr_if_any(parameters[second_point_idx].type) in all_enum)
        if not can_write:
            continue
        print_generator_source_info(out_file, 0)
        out_file.write(f"std::optional<NvmlFuncReturn> {parser_name}(const YAML::Node &node)\n{{\n")
        print_body_line("if (!node || !node[\"FunctionReturn\"])", out_file, 0)
        print_body_line('{', out_file, 0)
        print_body_line('return NvmlFuncReturn(NVML_ERROR_UNKNOWN);', out_file, 1)
        print_body_line('}', out_file, 0)
        print_body_line('auto ret = static_cast<nvmlReturn_t>(node[\"FunctionReturn\"].as<int>(NVML_ERROR_UNKNOWN));', out_file, 0)
        print_body_line("if (!node[\"ReturnValue\"])", out_file, 0)
        print_body_line('{', out_file, 0)
        print_body_line('return NvmlFuncReturn(ret);', out_file, 1)
        print_body_line('}', out_file, 0)
        print_body_line('std::vector<InjectionArgument> args;', out_file, 0)
        if arg_is_count(parameters[first_point_idx].name):
            # array type
            print_generator_source_info(out_file, 1)
            print_body_line('auto size = static_cast<unsigned int>(node[\"ReturnValue\"].size());', out_file, 0)
            print_body_line('int idx = 0;', out_file, 0)
            print_body_line(f'auto *valPtr = reinterpret_cast<{parameters[second_point_idx].type}>(malloc(sizeof({remove_ptr_if_any(parameters[second_point_idx].type)}) * size));', out_file, 0)
            print_body_line('for (YAML::const_iterator it = node[\"ReturnValue\"].begin(); it != node[\"ReturnValue\"].end(); ++it)', out_file, 0)
            print_body_line('{', out_file, 0)
            if is_basic_ptr_type(parameters[second_point_idx].type):
                # basic type uses build-in yaml-cpp convertion
                print_generator_source_info(out_file, 1)
                print_body_line(f'valPtr[idx++] = it->as<{remove_ptr_if_any(parameters[second_point_idx].type)}>();', out_file, 1)
            elif is_enum_type(remove_ptr_if_any(parameters[second_point_idx].type), all_enum):
                # enum type uses build-in yaml-cpp int convertion and cast to corresponding enum
                print_generator_source_info(out_file, 1)
                print_body_line(f'valPtr[idx++] = static_cast<{remove_ptr_if_any(parameters[second_point_idx].type)}>(it->as<int>());', out_file, 1)
            else:
                # struct type calls deserializer and memcpy to our struct
                print_generator_source_info(out_file, 1)
                deserializer_name = struct_name_to_deserializer_name(remove_ptr_if_any(parameters[second_point_idx].type))
                print_body_line(f'auto *tmp = {deserializer_name}(*it);', out_file, 1)
                print_body_line('if (!tmp)', out_file, 1)
                print_body_line('{', out_file, 1)
                print_body_line('free(valPtr);', out_file, 2)
                print_body_line('return std::nullopt;', out_file, 2)
                print_body_line('}', out_file, 1)
                print_body_line(f'std::memcpy(&valPtr[idx++], tmp, sizeof({remove_ptr_if_any(parameters[second_point_idx].type)}));', out_file, 1)
                print_body_line('free(tmp);', out_file, 1)
            print_body_line('}', out_file, 0)
            print_body_line('args.emplace_back(size);', out_file, 0)
            print_body_line('args.emplace_back(valPtr, size, true);', out_file, 0)
        else:
            # actual two attributes
            print_generator_source_info(out_file, 1)
            for i in range(first_point_idx, second_point_idx + 1):
                if is_basic_ptr_type(parameters[i].type):
                    # basic type uses build-in yaml-cpp convertion
                    print_body_line(f'args.emplace_back(node[\"ReturnValue\"]["{parameters[i].name}"].as<{remove_ptr_if_any(parameters[i].type)}>());', out_file, 0)
                else:
                    # enum type uses build-in yaml-cpp int convertion and cast to corresponding enum
                    print_body_line(f'args.emplace_back(static_cast<{remove_ptr_if_any(parameters[i].type)}>(node[\"ReturnValue\"]["{parameters[i].name}"].as<int>()));', out_file, 0)
        print_body_line('return NvmlFuncReturn(ret, std::move(args));', out_file, 0)
        out_file.write("}\n\n")
        written_parser[parser_name] = True

def write_three_attrs_funcs_parser(out_file, functions, all_enum, written_parser):
    for func_name, parameters in functions.items():
        if len(parameters) != 4:
            continue
        key, version = get_function_info_from_name(func_name)
        if version != 1:
            key = f"{key}_v{version}"
        parser_name = f"{key}Parser"
        if parser_name in written_parser:
            continue
        all_basic_type = is_basic_ptr_type(parameters[1].type) and is_basic_ptr_type(parameters[2].type) and is_basic_ptr_type(parameters[3].type)
        all_enum_type = is_enum_type(remove_ptr_if_any(parameters[1].type), all_enum) and\
                is_enum_type(remove_ptr_if_any(parameters[2].type), all_enum) and is_enum_type(remove_ptr_if_any(parameters[3].type), all_enum)
        can_write = all_basic_type or all_enum_type
        if not can_write:
            continue
        # e.g. nvmlDeviceGetEncoderStats
        print_generator_source_info(out_file, 0)
        out_file.write(f"std::optional<NvmlFuncReturn> {parser_name}(const YAML::Node &node)\n{{\n")
        print_body_line("if (!node || !node[\"FunctionReturn\"])", out_file, 0)
        print_body_line('{', out_file, 0)
        print_body_line('return NvmlFuncReturn(NVML_ERROR_UNKNOWN);', out_file, 1)
        print_body_line('}', out_file, 0)
        print_body_line('auto ret = static_cast<nvmlReturn_t>(node[\"FunctionReturn\"].as<int>(NVML_ERROR_UNKNOWN));', out_file, 0)
        print_body_line("if (!node[\"ReturnValue\"])", out_file, 0)
        print_body_line('{', out_file, 0)
        print_body_line('return NvmlFuncReturn(ret);', out_file, 1)
        print_body_line('}', out_file, 0)
        print_body_line('std::vector<InjectionArgument> args;', out_file, 0)
        for i in range(1, 4):
            if is_basic_ptr_type(parameters[i].type):
                print_body_line(f'args.emplace_back(node[\"ReturnValue\"]["{parameters[i].name}"].as<{remove_ptr_if_any(parameters[i].type)}>());', out_file, 0)
            else:
                print_body_line(f'args.emplace_back(static_cast<{remove_ptr_if_any(parameters[i].type)}>(node[\"ReturnValue\"]["{parameters[i].name}"].as<int>()));', out_file, 0)
        print_body_line('return NvmlFuncReturn(ret, std::move(args));', out_file, 0)
        out_file.write("}\n\n")
        written_parser[parser_name] = True

def write_handle_method(out_file):
    handle_methods = [
        ("DeviceHandle", "m_deviceHandlers"),
        ("GpuInstanceHandle", "m_gpuInstanceHandlers"),
        ("ComputeInstanceHandle", "m_computeInstanceHandlers"),
        ("VgpuTypeHandle", "m_vgpuTypeHandlers"),
        ("VgpuInstanceHandle", "m_vgpuInstanceHandlers"),
        ("GeneralHandle", "m_generalHandlers"),
    ]
    for func_name, handler_map in handle_methods:
        print_generator_source_info(out_file, 0)
        out_file.write(f'std::optional<NvmlFuncReturn> NvmlReturnDeserializer::{func_name}(const std::string &key, const YAML::Node &node)\n{{\n')
        print_body_line(f'if (!{handler_map}.contains(key))', out_file, 0)
        print_body_line('{', out_file, 0)
        print_body_line('return std::nullopt;', out_file, 1)
        print_body_line('}', out_file, 0)
        print_body_line(f'return {handler_map}[key](node);', out_file, 0)
        out_file.write('}\n\n')

def write_extra_key_handle_method(out_file):
    handle_methods = [
        ("DeviceExtraKeyHandle", "m_deviceExtraKeyHandlers"),
        ("GpuInstanceExtraKeyHandle", "m_gpuInstanceExtraKeyHandlers"),
        ("VgpuTypeExtraKeyHandle", "m_vgpuTypeExtraKeyHandlers"),
    ]
    for func_name, handler_map in handle_methods:
        print_generator_source_info(out_file, 0)
        out_file.write(f'std::optional<std::vector<std::tuple<InjectionArgument, NvmlFuncReturn>>> NvmlReturnDeserializer::{func_name}(const std::string &key, const YAML::Node &node)\n{{\n')
        print_body_line(f'if (!{handler_map}.contains(key))', out_file, 0)
        print_body_line('{', out_file, 0)
        print_body_line('return std::nullopt;', out_file, 1)
        print_body_line('}', out_file, 0)
        print_body_line('std::vector<std::tuple<InjectionArgument, NvmlFuncReturn>> ret;', out_file, 0)
        print_body_line(f'auto &[keyParser, valueParser] = {handler_map}[key];', out_file, 0)
        print_body_line('for (YAML::const_iterator it = node.begin(); it != node.end(); ++it)', out_file, 0)
        print_body_line('{', out_file, 0)
        print_body_line('YAML::Node innerKey = it->first;', out_file, 1)
        print_body_line('YAML::Node innerValue = it->second;', out_file, 1)
        print_body_line('auto key2Opt = keyParser(innerKey);', out_file, 1)
        print_body_line('if (!key2Opt)', out_file, 1)
        print_body_line('{', out_file, 1)
        print_body_line('return std::nullopt;', out_file, 2)
        print_body_line('}', out_file, 1)
        print_body_line('auto valOpt = valueParser(innerValue);', out_file, 1)
        print_body_line('if (!valOpt)', out_file, 1)
        print_body_line('{', out_file, 1)
        print_body_line('return std::nullopt;', out_file, 2)
        print_body_line('}', out_file, 1)
        print_body_line('ret.emplace_back(key2Opt.value(), valOpt.value());', out_file, 1)
        print_body_line('}', out_file, 0)
        print_body_line('return ret;', out_file, 0)
        out_file.write('}\n\n')

def write_three_key_handle_method(out_file):
    handle_methods = [
        ("DeviceThreeKeysHandle", "m_deviceThreeKeysHandlers"),
        ("GpuInstanceThreeKeysHandle", "m_gpuInstanceThreeKeysHandlers"),
    ]
    for func_name, handler_map in handle_methods:
        print_generator_source_info(out_file, 0)
        out_file.write(f'std::optional<std::vector<std::tuple<InjectionArgument, InjectionArgument, NvmlFuncReturn>>> NvmlReturnDeserializer::{func_name}(const std::string &key, const YAML::Node &node)\n{{\n')
        print_body_line(f'if (!{handler_map}.contains(key))', out_file, 0)
        print_body_line('{', out_file, 0)
        print_body_line('return std::nullopt;', out_file, 1)
        print_body_line('}', out_file, 0)
        print_body_line('std::vector<std::tuple<InjectionArgument, InjectionArgument, NvmlFuncReturn>> ret;', out_file, 0)
        print_body_line(f'auto &[key1Parser, key2Parser, valueParser] = {handler_map}[key];', out_file, 0)
        print_body_line('for (YAML::const_iterator it = node.begin(); it != node.end(); ++it)', out_file, 0)
        print_body_line('{', out_file, 0)
        print_body_line('YAML::Node innerKey = it->first;', out_file, 1)
        print_body_line('YAML::Node innerValue = it->second;', out_file, 1)
        print_body_line('auto key1Opt = key1Parser(innerKey);', out_file, 1)
        print_body_line('if (!key1Opt)', out_file, 1)
        print_body_line('{', out_file, 1)
        print_body_line('return std::nullopt;', out_file, 2)
        print_body_line('}', out_file, 1)
        print_body_line('for (YAML::const_iterator layerThreeIt = innerValue.begin(); layerThreeIt != innerValue.end(); ++layerThreeIt)', out_file, 1)
        print_body_line('{', out_file, 1)
        print_body_line('YAML::Node layerThreeKey = layerThreeIt->first;', out_file, 2)
        print_body_line('YAML::Node layerThreeValue = layerThreeIt->second;', out_file, 2)
        print_body_line('auto key2Opt = key2Parser(layerThreeKey);', out_file, 2)
        print_body_line('if (!key2Opt)', out_file, 2)
        print_body_line('{', out_file, 2)
        print_body_line('return std::nullopt;', out_file, 3)
        print_body_line('}', out_file, 2)
        print_body_line('auto valOpt = valueParser(layerThreeValue);', out_file, 2)
        print_body_line('if (!valOpt)', out_file, 2)
        print_body_line('{', out_file, 2)
        print_body_line('return std::nullopt;', out_file, 3)
        print_body_line('}', out_file, 2)
        print_body_line('ret.emplace_back(key1Opt.value(), key2Opt.value(), valOpt.value());', out_file, 2)
        print_body_line('}', out_file, 1)
        print_body_line('}', out_file, 0)
        print_body_line('return ret;', out_file, 0)
        out_file.write('}\n\n')

def get_skipped_funcs():
    skip_funcs = {}
    skip_funcs["nvmlInit_v2"] = True
    skip_funcs["nvmlInitWithFlags"] = True
    skip_funcs["nvmlShutdown"] = True
    skip_funcs["nvmlErrorString"] = True
    skip_funcs["nvmlSystemGetProcessName"] = True
    skip_funcs["nvmlUnitGetHandleByIndex"] = True
    skip_funcs["nvmlUnitGetUnitInfo"] = True
    skip_funcs["nvmlUnitGetLedState"] = True
    skip_funcs["nvmlUnitGetPsuInfo"] = True
    skip_funcs["nvmlUnitGetTemperature"] = True
    skip_funcs["nvmlUnitGetFanSpeedInfo"] = True
    skip_funcs["nvmlUnitGetDevices"] = True
    skip_funcs["nvmlSystemGetHicVersion"] = True
    skip_funcs["nvmlEventSetCreate"] = True
    skip_funcs["nvmlEventSetWait_v2"] = True
    skip_funcs["nvmlEventSetFree"] = True
    skip_funcs["nvmlDeviceRegisterEvents"] = True
    skip_funcs["nvmlGpmSampleFree"] = True
    skip_funcs["nvmlGpmSampleAlloc"] = True
    skip_funcs["nvmlVgpuInstanceClearAccountingPids"] = True
    # special functions handled by hand-written parser
    skip_funcs["nvmlDeviceGetActiveVgpus"] = True
    skip_funcs["nvmlDeviceGetTopologyCommonAncestor"] = True
    skip_funcs["nvmlDeviceGetTopologyNearestGpus"] = True
    skip_funcs["nvmlDeviceGetGpuInstances"] = True
    skip_funcs["nvmlGpuInstanceGetInfo"] = True
    skip_funcs["nvmlGpuInstanceGetComputeInstances"] = True
    skip_funcs["nvmlDeviceOnSameBoard"] = True
    skip_funcs["nvmlComputeInstanceGetInfo_v2"] = True
    skip_funcs["nvmlDeviceGetRemappedRows"] = True
    skip_funcs["nvmlDeviceGetMemoryErrorCounter"] = True
    skip_funcs["nvmlVgpuInstanceGetVmID"] = True
    skip_funcs["nvmlDeviceGetFieldValues"] = True
    skip_funcs["nvmlDeviceGetMemoryInfo"] = True
    skip_funcs["nvmlDeviceGetVgpuProcessUtilization"] = True
    skip_funcs["nvmlDeviceGetVgpuUtilization"] = True
    skip_funcs["nvmlDeviceGetProcessUtilization"] = True
    # not used
    skip_funcs["nvmlDeviceGetNvLinkUtilizationControl"] = True
    skip_funcs["nvmlSystemGetHicVersion"] = True
    skip_funcs["nvmlSystemGetTopologyGpuSet"] = True
    skip_funcs["nvmlDeviceGetP2PStatus"] = True
    skip_funcs["nvmlGetVgpuCompatibility"] = True

    skip_funcs["nvmlDeviceGetPgpuMetadataString"] = True
    skip_funcs["nvmlDeviceGetGspFirmwareVersion"] = True
    skip_funcs["nvmlDeviceGetGspFirmwareMode"] = True
    skip_funcs["nvmlDeviceQueryDrainState"] = True
    skip_funcs["nvmlDeviceDiscoverGpus"] = True
    skip_funcs["nvmlGetBlacklistDeviceCount"] = True
    skip_funcs["nvmlGetBlacklistDeviceInfoByIndex"] = True
    skip_funcs["nvmlGetVgpuVersion"] = True
    skip_funcs["nvmlDeviceGetGpuInstancePossiblePlacements"] = True
    skip_funcs["nvmlDeviceGetGpuInstancePossiblePlacements_v2"] = True
    skip_funcs["nvmlDeviceGetDynamicPstatesInfo"] = True
    skip_funcs["nvmlDeviceGetThermalSettings"] = True
    skip_funcs["nvmlDeviceGetMinMaxClockOfPState"] = True
    skip_funcs["nvmlDeviceGetSupportedPerformanceStates"] = True
    skip_funcs["nvmlDeviceGetMinMaxFanSpeed"] = True
    skip_funcs["nvmlDeviceGetGpcClkMinMaxVfOffset"] = True
    skip_funcs["nvmlDeviceGetMemClkMinMaxVfOffset"] = True
    skip_funcs["nvmlDeviceGetSamples"] = True
    return skip_funcs

def nvml_return_deserializer_cpp_writer(out_dir, struct_with_member, all_enum, functions):
    _, struct_cannot_gen_parsers = separate_struct_by_generable(struct_with_member, all_enum)
    handled_funcs = {}

    with open(f"{out_dir}/{NVML_RETURN_DESERIALIZER_CPP_PATH}", "w") as out_file:
        write_auto_generate_c_file_header(out_file)

        out_file.write('// clang-format off\n')
        # Some structs parser are generated but not used in this time
        out_file.write('#pragma GCC diagnostic ignored "-Wunused-function"\n')
        out_file.write('#include "NvmlReturnDeserializer.h"\n\n')
        out_file.write('#include <functional>\n')
        out_file.write('#include <optional>\n')
        out_file.write('#include <unordered_map>\n\n')
        out_file.write('#include "nvml.h"\n')
        out_file.write('#include <yaml-cpp/yaml.h>\n')
        out_file.write('#include <yaml-cpp/node/node.h>\n\n')
        out_file.write('#include "NvmlLogging.h"\n')
        out_file.write('#include "NvmlFuncReturn.h"\n')
        out_file.write('#include "NvmlInjectionUtil.h"\n\n')

        out_file.write('namespace {\n\n')

        cannot_write_deserializer_struct = get_cannot_write_deserializer_struct(struct_with_member)
        write_deserializer_declare(out_file, struct_with_member, cannot_write_deserializer_struct)
        write_deserializer_definition(out_file, struct_with_member, all_enum, cannot_write_deserializer_struct)
        write_basic_type_parser(out_file)
        write_basic_key_type_parser(out_file)
        write_nvml_return_parser(out_file)
        written_parser = {}
        write_known_struct_parser(out_file, struct_with_member, struct_cannot_gen_parsers, all_enum, written_parser)
        write_array_funcs_parser(out_file, functions, all_enum, written_parser, cannot_write_deserializer_struct)
        write_two_attrs_funcs_parser(out_file, functions, all_enum, written_parser, cannot_write_deserializer_struct)
        write_three_attrs_funcs_parser(out_file, functions, all_enum, written_parser)

        out_file.write('}\n\n')

        print_generator_source_info(out_file, 0)
        out_file.write('NvmlReturnDeserializer::NvmlReturnDeserializer()\n{\n')
        print_generator_source_info(out_file, 1)
        print_body_line('m_deviceHandlers = {', out_file, 0)
        for func_name, parameters in functions.items():
            if try_to_write_device_handler(out_file, func_name, parameters, struct_cannot_gen_parsers, all_enum):
                handled_funcs[func_name] = True
                continue
        print_body_line('};', out_file, 0)

        print_generator_source_info(out_file, 1)
        print_body_line('m_deviceExtraKeyHandlers = {', out_file, 0)
        for func_name, parameters in functions.items():
            if try_to_write_device_extra_key_handler(out_file, func_name, parameters, struct_cannot_gen_parsers, all_enum):
                handled_funcs[func_name] = True
                continue
        print_body_line('};', out_file, 0)

        print_generator_source_info(out_file, 1)
        print_body_line('m_deviceThreeKeysHandlers = {', out_file, 0)
        for func_name, parameters in functions.items():
            if try_to_write_device_three_keys_handler(out_file, func_name, parameters, struct_cannot_gen_parsers, all_enum):
                handled_funcs[func_name] = True
                continue
        print_body_line('};', out_file, 0)

        print_generator_source_info(out_file, 1)
        print_body_line('m_gpuInstanceHandlers = {', out_file, 0)
        for func_name, parameters in functions.items():
            if try_to_write_gpu_instance_handler(out_file, func_name, parameters, struct_cannot_gen_parsers, all_enum):
                handled_funcs[func_name] = True
                continue
        print_body_line('};', out_file, 0)

        print_generator_source_info(out_file, 1)
        print_body_line('m_gpuInstanceExtraKeyHandlers = {', out_file, 0)
        for func_name, parameters in functions.items():
            if try_to_write_gpu_instance_extra_key_handler(out_file, func_name, parameters, struct_cannot_gen_parsers, all_enum):
                handled_funcs[func_name] = True
                continue
        print_body_line('};', out_file, 0)

        print_generator_source_info(out_file, 1)
        print_body_line('m_gpuInstanceThreeKeysHandlers = {', out_file, 0)
        for func_name, parameters in functions.items():
            if try_to_write_gpu_instance_three_keys_handler(out_file, func_name, parameters, struct_cannot_gen_parsers, all_enum):
                handled_funcs[func_name] = True
                continue
        print_body_line('};', out_file, 0)

        print_generator_source_info(out_file, 1)
        print_body_line('m_computeInstanceHandlers = {', out_file, 0)
        for func_name, parameters in functions.items():
            if try_to_write_compute_instance_handler(out_file, func_name, parameters, struct_cannot_gen_parsers, all_enum):
                handled_funcs[func_name] = True
                continue
        print_body_line('};', out_file, 0)

        print_generator_source_info(out_file, 1)
        print_body_line('m_vgpuTypeHandlers = {', out_file, 0)
        for func_name, parameters in functions.items():
            if try_to_write_vgpu_type_handler(out_file, func_name, parameters, struct_cannot_gen_parsers, all_enum):
                handled_funcs[func_name] = True
                continue
        print_body_line('};', out_file, 0)

        print_generator_source_info(out_file, 1)
        print_body_line('m_vgpuTypeExtraKeyHandlers = {', out_file, 0)
        for func_name, parameters in functions.items():
            if try_to_write_vgpu_type_extra_key_handler(out_file, func_name, parameters, struct_cannot_gen_parsers, all_enum):
                handled_funcs[func_name] = True
                continue
        print_body_line('};', out_file, 0)

        print_generator_source_info(out_file, 1)
        print_body_line('m_vgpuInstanceHandlers = {', out_file, 0)
        for func_name, parameters in functions.items():
            if try_to_write_vgpu_instance_handler(out_file, func_name, parameters, struct_cannot_gen_parsers, all_enum):
                handled_funcs[func_name] = True
                continue
        print_body_line('};', out_file, 0)

        print_generator_source_info(out_file, 1)
        print_body_line('m_generalHandlers = {', out_file, 0)
        for func_name, parameters in functions.items():
            if try_to_write_general_handler(out_file, func_name, parameters, struct_cannot_gen_parsers, all_enum):
                handled_funcs[func_name] = True
                continue
        print_body_line('};', out_file, 0)

        out_file.write('}\n\n')

        write_handle_method(out_file)
        write_extra_key_handle_method(out_file)
        write_three_key_handle_method(out_file)

        generated_func_count = len(handled_funcs)
        all_funcs_count = len(functions)
        logging.debug(f"I was able to generate YAML parser for {generated_func_count} out of {all_funcs_count}")
        logging.debug("The result of the following functions are unable to parse:")
        for func_name, _ in functions.items():
            if func_name in handled_funcs:
                continue
            logging.debug(func_name)
    logging.info(f"{NVML_RETURN_DESERIALIZER_CPP_PATH} genereted")

def nvml_return_deserializer_header_writer(out_dir):
    with open(f"{out_dir}/{NVML_RETURN_DESERIALIZER_HEADER_PATH}", "w") as out_file:
        write_auto_generate_c_file_header(out_file)

        out_file.write('// clang-format off\n')
        out_file.write('#pragma once\n\n')
        out_file.write('#include <functional>\n')
        out_file.write('#include <optional>\n')
        out_file.write('#include <string>\n')
        out_file.write('#include <tuple>\n')
        out_file.write('#include <vector>\n\n')
        out_file.write('#include <unordered_map>\n\n')
        out_file.write('#include "NvmlFuncReturn.h"\n\n')
        out_file.write('#include <yaml-cpp/node/node.h>\n\n')

        print_generator_source_info(out_file, 0)
        out_file.write('class NvmlReturnDeserializer\n{\n')
        out_file.write('private:\n')
        print_body_line('std::unordered_map<std::string, std::function<std::optional<NvmlFuncReturn>(const YAML::Node &)>> m_deviceHandlers;', out_file, 0)
        print_body_line('std::unordered_map<std::string, std::tuple<std::function<std::optional<InjectionArgument>(const YAML::Node &)>, std::function<std::optional<NvmlFuncReturn>(const YAML::Node &)>>> m_deviceExtraKeyHandlers;\n', out_file, 0)
        print_body_line('std::unordered_map<std::string, std::tuple<std::function<std::optional<InjectionArgument>(const YAML::Node &)>, std::function<std::optional<InjectionArgument>(const YAML::Node &)>, std::function<std::optional<NvmlFuncReturn>(const YAML::Node &)>>> m_deviceThreeKeysHandlers;\n', out_file, 0)
        print_body_line('std::unordered_map<std::string, std::function<std::optional<NvmlFuncReturn>(const YAML::Node &)>> m_gpuInstanceHandlers;', out_file, 0)
        print_body_line('std::unordered_map<std::string, std::function<std::optional<NvmlFuncReturn>(const YAML::Node &)>> m_computeInstanceHandlers;', out_file, 0)
        print_body_line('std::unordered_map<std::string, std::tuple<std::function<std::optional<InjectionArgument>(const YAML::Node &)>, std::function<std::optional<NvmlFuncReturn>(const YAML::Node &)>>> m_gpuInstanceExtraKeyHandlers;\n', out_file, 0)
        print_body_line('std::unordered_map<std::string, std::tuple<std::function<std::optional<InjectionArgument>(const YAML::Node &)>, std::function<std::optional<InjectionArgument>(const YAML::Node &)>, std::function<std::optional<NvmlFuncReturn>(const YAML::Node &)>>> m_gpuInstanceThreeKeysHandlers;\n', out_file, 0)
        print_body_line('std::unordered_map<std::string, std::function<std::optional<NvmlFuncReturn>(const YAML::Node &)>> m_vgpuTypeHandlers;', out_file, 0)
        print_body_line('std::unordered_map<std::string, std::tuple<std::function<std::optional<InjectionArgument>(const YAML::Node &)>, std::function<std::optional<NvmlFuncReturn>(const YAML::Node &)>>> m_vgpuTypeExtraKeyHandlers;\n', out_file, 0)
        print_body_line('std::unordered_map<std::string, std::function<std::optional<NvmlFuncReturn>(const YAML::Node &)>> m_vgpuInstanceHandlers;', out_file, 0)
        print_body_line('std::unordered_map<std::string, std::function<std::optional<NvmlFuncReturn>(const YAML::Node &)>> m_generalHandlers;', out_file, 0)
        out_file.write('public:\n')
        print_body_line('NvmlReturnDeserializer();', out_file, 0)
        print_body_line('std::optional<NvmlFuncReturn> DeviceHandle(const std::string &key, const YAML::Node &node);', out_file, 0)
        print_body_line('std::optional<NvmlFuncReturn> GpuInstanceHandle(const std::string &key, const YAML::Node &node);', out_file, 0)
        print_body_line('std::optional<NvmlFuncReturn> ComputeInstanceHandle(const std::string &key, const YAML::Node &node);', out_file, 0)
        print_body_line('std::optional<NvmlFuncReturn> VgpuTypeHandle(const std::string &key, const YAML::Node &node);', out_file, 0)
        print_body_line('std::optional<NvmlFuncReturn> VgpuInstanceHandle(const std::string &key, const YAML::Node &node);', out_file, 0)
        print_body_line('std::optional<NvmlFuncReturn> GeneralHandle(const std::string &key, const YAML::Node &node);', out_file, 0)
        print_body_line('std::optional<std::vector<std::tuple<InjectionArgument, NvmlFuncReturn>>> DeviceExtraKeyHandle(const std::string &key, const YAML::Node &node);', out_file, 0)
        print_body_line('std::optional<std::vector<std::tuple<InjectionArgument, InjectionArgument, NvmlFuncReturn>>> DeviceThreeKeysHandle(const std::string &key, const YAML::Node &node);', out_file, 0)
        print_body_line('std::optional<std::vector<std::tuple<InjectionArgument, NvmlFuncReturn>>> GpuInstanceExtraKeyHandle(const std::string &key, const YAML::Node &node);', out_file, 0)
        print_body_line('std::optional<std::vector<std::tuple<InjectionArgument, NvmlFuncReturn>>> VgpuTypeExtraKeyHandle(const std::string &key, const YAML::Node &node);', out_file, 0)
        print_body_line('std::optional<std::vector<std::tuple<InjectionArgument, InjectionArgument, NvmlFuncReturn>>> GpuInstanceThreeKeysHandle(const std::string &key, const YAML::Node &node);', out_file, 0)
        out_file.write('};\n')

    logging.info(f"{NVML_RETURN_DESERIALIZER_HEADER_PATH} genereted")

def extract_struct(cursor):
    fields = []
    for field in cursor.type.get_fields():
        fields.append((field.type.spelling, field.spelling))
    return cursor.spelling, fields

class Arg:
    def __init__(self, name, type):
        self.name = name
        self.type = type

ENUM_TYPE_UNSIGNED_INT = 0
ENUM_TYPE_UNSIGNED_CHAR = 1
ENUM_TYPE_PURE_ENUM = 2

def parse_nvml_define_from_header_file(entryPointsHeaderFile, nvmlHeaderFile):
    skip_funcs = get_skipped_funcs()
    with open(entryPointsHeaderFile, 'r') as entryFile:
        contents = entryFile.read()
        used_functions = get_entry_points_all_functions(contents)
    index = clang.cindex.Index.create()
    tu = index.parse(nvmlHeaderFile)
    typedef_name_mapping = {}
    struct_with_member = {}
    functions = {}
    """
    The key to this dictionary represents an enumeration type.

    If the value associated with a particular key is `ENUM_TYPE_UNSIGNED_INT`, it indicates
    that this enumeration was defined as a typedef over a primitive type (unsigned int).

    If the value associated with a particular key is `ENUM_TYPE_UNSIGNED_CHAR`, it indicates
    that this enumeration was defined as a typedef over a primitive type (unsigned char).

    If the value associated with a particular key is `ENUM_TYPE_PURE_ENUM`, it indicates
    that this actually an enumeration.
    """
    all_enum = {}
    for child in tu.cursor.get_children():
        if child.kind != clang.cindex.CursorKind.TYPEDEF_DECL:
            continue
        if child.underlying_typedef_type.spelling.startswith("unsigned int"):
            all_enum[child.spelling] = ENUM_TYPE_UNSIGNED_INT
            continue
        if child.underlying_typedef_type.spelling.startswith("unsigned char"):
            all_enum[child.spelling] = ENUM_TYPE_UNSIGNED_CHAR
            continue
        if child.underlying_typedef_type.spelling.startswith("struct "):
            typedef_name_mapping[child.underlying_typedef_type.spelling[7:]] = child.spelling
            continue
        if child.underlying_typedef_type.spelling.startswith("enum "):
            all_enum[child.spelling] = ENUM_TYPE_PURE_ENUM
            continue
        if child.underlying_typedef_type.spelling.startswith("nvml"):
            typedef_name_mapping[child.underlying_typedef_type.spelling] = child.spelling
            continue

    for child in tu.cursor.get_children():
        if child.kind == clang.cindex.CursorKind.FUNCTION_DECL:
            func_name = child.spelling
            if func_name not in used_functions.func_dict:
                continue
            if func_is_setter(func_name) or func_is_handler_getter(func_name) or func_name in skip_funcs:
                continue
            parameters = []
            for c in child.get_children():
                if c.kind == clang.cindex.CursorKind.PARM_DECL:
                    parameters.append(Arg(c.spelling, c.type.spelling))
            functions[func_name] = parameters

    for child in tu.cursor.get_children():
        if child.kind != clang.cindex.CursorKind.STRUCT_DECL:
            continue
        name, fields = extract_struct(child)
        if name not in typedef_name_mapping:
            continue
        struct_with_member[typedef_name_mapping[name]] = fields

    return struct_with_member, all_enum, functions

def nvml_return_deserializer_writer(outputDir, struct_with_member, all_enum, functions):
    nvml_return_deserializer_cpp_writer(outputDir, struct_with_member, all_enum, functions)
    nvml_return_deserializer_header_writer(outputDir)

def injection_argument_writer(output_dir, struct_with_member, all_enum):
    write_injection_argument_header(output_dir, struct_with_member, all_enum)
    write_injection_argument_cpp(output_dir, struct_with_member, all_enum)

def injection_structs_writer(output_dir, struct_with_member, all_enum):
    write_injection_structs_header(struct_with_member, output_dir, all_enum)
    write_nvml_injection_struct_py(output_dir, struct_with_member, all_enum)

def is_in_dcgm_root():
    return os.path.exists("./dcgmlib") and os.path.exists("./dcgmi") and os.path.exists("./nvml-injection")

def main():
    description = "Generates source files containing stubs needed for NVML injection.\n" \
                  "Example:\n" \
                  "\t# nvml-injection/scripts/generate_nvml_stubs.py -i sdk/nvidia/nvml/entry_points.h -o nvml-injection"
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=description)
    parser.add_argument('-i', '--input-file', default='sdk/nvidia/nvml/entry_points.h', dest='inputPath')
    parser.add_argument('-n', '--nvml-header', default='sdk/nvidia/nvml/nvml.h', dest='nvmlHeaderPath')
    parser.add_argument('-o', '--output-dir', default='.', dest='outputDir')
    parser.add_argument('--verbose', action="store_true")
    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(message)s")
    else:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
    if "CLANG_LIBRARY_PATH" in os.environ:
        Config.set_library_path(os.environ["CLANG_LIBRARY_PATH"])

    if not is_in_dcgm_root():
        logging.error("Please run this script from the top-level DCGM directory.")
        exit(1)

    struct_with_member, all_enum, functions = parse_nvml_define_from_header_file(args.inputPath, args.nvmlHeaderPath)
    nvml_return_deserializer_writer(args.outputDir, struct_with_member, all_enum, functions)
    injection_argument_writer(args.outputDir, struct_with_member, all_enum)
    injection_structs_writer(args.outputDir, struct_with_member, all_enum)
    parse_entry_points(args.inputPath, args.outputDir)

if __name__ == '__main__':
    main()
