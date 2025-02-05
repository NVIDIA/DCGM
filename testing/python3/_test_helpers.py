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
from functools import wraps
# pylint: disable=no-name-in-module
from test_utils import skip_test
import logger

MOCK_MSG = "mock not installed. Please run \"pip install mock\""

try:
    import mock
    MOCK_INSTALLED = True
except ImportError:
    logger.warning(MOCK_MSG)
    MOCK_INSTALLED = False

def skip_test_if_no_mock():
    '''
    Returns a decorator for functions. The decorator skips
    the test in the provided function if mock is not installed
    '''
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwds):
            if MOCK_INSTALLED:
                fn(*args, **kwds)
            else:
                skip_test(MOCK_MSG)
        return wrapper
    return decorator

# Do not use this class directly
class _MaybeMock:
    def __call__(self, *args, **kwds):
        return skip_test_if_no_mock()

    def __getattr__(self, attr):
        if (MOCK_INSTALLED):
            return getattr(mock, attr)
        return self

def KeywordizeLastArgument(keywordName):
    '''
    This takes the last passed positional argument and calls the decorated
    function with it as a keyword parameter of the name specified. This helps
    with the positional arguments inserted by mock.patch (and, by extension,
    maybemock.patch) to work with amortized decorators.
    '''
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            kwargs[keywordName] = args[-1]
            args = args[:-1]
            fn(*args, **kwargs)
        return wrapper
    return decorator
    
maybemock = _MaybeMock()

DCGM_NVML_NOT_EXIST_MSG = "dcgm_nvml not presented."
DCGM_NVML_VER_MSG = "dcgm_nvml exists, but missing some definitions. Please update it to the latest version https://pypi.org/project/nvidia-ml-py/#description"

try:
    import dcgm_nvml
    DCGM_NVML_PRESENTED = True
except ImportError:
    logger.warning(MOCK_MSG)
    DCGM_NVML_PRESENTED = False

import nvml_injection_structs
if not nvml_injection_structs.nvml_injection_usable:
    logger.warning(DCGM_NVML_VER_MSG)
    DCGM_NVML_PRESENTED = False

def skip_test_if_no_dcgm_nvml():
    '''
    Returns a decorator for functions. The decorator skips
    the test in the provided function if dcgm_nvml is not presented
    '''
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwds):
            if DCGM_NVML_PRESENTED:
                fn(*args, **kwds)
            else:
                skip_test(DCGM_NVML_NOT_EXIST_MSG)
        return wrapper
    return decorator

# Do not use this class directly
class _MaybeDcgmNVML:
    def __call__(self, *args, **kwds):
        return skip_test_if_no_dcgm_nvml()

    def __getattr__(self, attr):
        if (DCGM_NVML_PRESENTED):
            return getattr(dcgm_nvml, attr)
        return self

maybe_dcgm_nvml = _MaybeDcgmNVML()
