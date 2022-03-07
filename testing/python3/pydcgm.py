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

def _python_version_check():
    import sys
    python_version = sys.version.split(None, 1)[0]
    if python_version < '3':
        print('[ERROR] Detected Python version {}. These bindings are for Python 3.5+. Please load the Python 2 bindings found at /usr/local/dcgm/bindings'.format(python_version))
        sys.exit(1)
_python_version_check()

#Bring classes into this namespace
from DcgmHandle import *
from DcgmGroup import *
from DcgmStatus import *
from DcgmSystem import *
from DcgmFieldGroup import *

import os
if '__DCGM_TESTING_FRAMEWORK_ACTIVE' in os.environ and os.environ['__DCGM_TESTING_FRAMEWORK_ACTIVE'] == '1':
    import utils
    import dcgm_structs
    dcgm_structs._dcgmInit(utils.get_testing_framework_library_path())

'''
Define a unique exception type we will return so that callers can distinguish our exceptions from python standard ones
'''
class DcgmException(Exception):
    pass
