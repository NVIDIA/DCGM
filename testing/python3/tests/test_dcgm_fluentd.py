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
from socket import socket, AF_INET, SOCK_DGRAM

from common.Struct import Struct
from dcgm_fluentd import DcgmFluentd

def test_send_to_fluentd():
    # Can't create a proper closure in Python, so we create an object which acts
    # as a closure
    namespace = Struct(message=None, dest=None)

    def mysendto(_message, _dest):
        namespace.message = _message
        namespace.dest = _dest

    mysock = Struct(sendto=mysendto)

    dr = DcgmFluentd('FAKE_HOST', 101010)

    # Assert that we are sending over UDP
    assert dr.m_sock.family == AF_INET
    assert dr.m_sock.type == SOCK_DGRAM

    dr.m_sock = mysock

    dr.SendToFluentd('message')

    assert(namespace.message == 'message')
    assert(namespace.dest == ('FAKE_HOST', 101010))

def test_custom_json_handler():
    namespace = Struct(arg=None)

    def MySendToFluentd(json):
        namespace.arg = json # pylint: disable=no-member

    dr = DcgmFluentd('FAKE_HOST', 101010)
    dr.SendToFluentd = MySendToFluentd

    dr.CustomJsonHandler('value')
    assert namespace.arg == 'value' # pylint: disable=no-member
