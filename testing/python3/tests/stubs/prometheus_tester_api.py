# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
##### Spoof prometheus for our testing framework
################################################################################

import prometheus_tester_globals

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
class Value:
    def __init__(self, id, value = None):
        self.id = id
        self.value = value

    def set(self, value):
        self.value = value

    def get(self):
        return self.value

################################################################################
class Labels:
    def __init__(self):
        self.values = {}

################################################################################
class Gauge:
    def __init__(self, name, documentation, fields=[]):
        self.name = name
        self.documentation = documentation

        if not 'fields' in prometheus_tester_globals.gvars:
            prometheus_tester_globals.gvars['fields'] = {}
            
        prometheus_tester_globals.gvars['fields'][name] = Labels()

    def labels(self, id, uniqueId):
        prometheus_tester_globals.gvars['fields'][self.name].values[uniqueId] = Value(id)

        return prometheus_tester_globals.gvars['fields'][self.name].values[uniqueId]

def start_http_server(port):
    return
