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
from common.dcgm_client_main import main
from DcgmJsonReader import DcgmJsonReader
from socket import socket, AF_INET, SOCK_DGRAM

# Displayed to the user
TELEGRAF_NAME = 'Telegraf'
DEFAULT_TELEGRAF_PORT = 8094

# Telegraf Configuration
# ======================
#
# In order for Telegraf to understand the format of the data sent by this
# module, it needs to be configured with the input plugin below
#
# If you modify the list of published fields, you will need to add non-numeric
# ones as tag_keys for Telegraf to store them
#
# [[inputs.socket_listener]]
#   name_override = "dcgm"
#   service_address = "udp://:8094"
#   data_format = "json"
#   tag_keys = [
#     "compute_pids",
#     "driver_version",
#     "gpu_uuid",
#     "nvml_version",
#     "process_name",
#     "xid_errors"
#   ]

class DcgmTelegraf(DcgmJsonReader):
    ###########################################################################
    def __init__(self, publish_hostname, publish_port, **kwargs):
        self.m_sock = socket(AF_INET, SOCK_DGRAM)
        self.m_dest = (publish_hostname, publish_port)
        super(DcgmTelegraf, self).__init__(**kwargs)

    ###########################################################################
    def SendToTelegraf(self, payload):
        self.m_sock.sendto(payload, self.m_dest)

    ###########################################################################
    def CustomJsonHandler(self, outJson):
        self.SendToTelegraf(outJson)

if __name__ == '__main__': # pragma: no cover
    main(DcgmTelegraf, TELEGRAF_NAME, DEFAULT_TELEGRAF_PORT, add_target_host=True)
