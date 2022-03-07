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
FLUENTD_NAME = 'Fluentd'
DEFAULT_FLUENTD_PORT = 24225

# Fluentd Configuration
# =====================
# In order to use this client, Fluentd needs to accept json over udp.
# The default port is 24225

class DcgmFluentd(DcgmJsonReader):
    ###########################################################################
    def __init__(self, publish_hostname, publish_port, **kwargs):
        self.m_sock = socket(AF_INET, SOCK_DGRAM)
        self.m_dest = (publish_hostname, publish_port)
        super(DcgmFluentd, self).__init__(**kwargs)

    ###########################################################################
    def SendToFluentd(self, payload):
        self.m_sock.sendto(payload, self.m_dest)

    ###########################################################################
    def CustomJsonHandler(self, outJson):
        self.SendToFluentd(outJson)

if __name__ == '__main__': # pragma: no cover
    main(DcgmFluentd, FLUENTD_NAME, DEFAULT_FLUENTD_PORT, add_target_host=True)
