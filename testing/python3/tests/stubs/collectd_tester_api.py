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

################################################################################
##### Spoof collectd for our testing framework
################################################################################
import collectd_tester_globals
import dcgm_collectd_plugin

################################################################################
def register_config(func_ptr, name=None):
    collectd_tester_globals.gvars['config'] = func_ptr

################################################################################
def register_init(func_ptr):
    collectd_tester_globals.gvars['init'] = func_ptr

################################################################################
def register_read(func_ptr, interval=10):
    collectd_tester_globals.gvars['read'] = func_ptr
    collectd_tester_globals.interval = interval

################################################################################
def register_shutdown(func_ptr):
    collectd_tester_globals.gvars['shutdown'] = func_ptr

################################################################################
def info(msg):
    print(msg)

################################################################################
def error(msg):
    print(msg)

################################################################################
def debug(msg):
    pass

################################################################################
class Values:

    ############################################################################
    def __init__(self, **kwargs):
        # dcgm_collectd references these, so we'll reference them as well
        self.plugin = ''
        self.plugin_instance = ''

    ############################################################################
    def dispatch(self, **kwargs):
        if 'out' not in collectd_tester_globals.gvars:
            collectd_tester_globals.gvars['out'] = {}

        if 'type_instance' in kwargs and 'type' in kwargs and 'values' in kwargs:
            gpuId = kwargs['type_instance']
            fieldTag = kwargs['type']
            oneVal = kwargs['values'][0]

            if gpuId not in collectd_tester_globals.gvars['out']:
                collectd_tester_globals.gvars['out'][gpuId] = {}

            # Put this in a global dictionary for later inspection
            collectd_tester_globals.gvars['out'][gpuId][fieldTag] = oneVal

