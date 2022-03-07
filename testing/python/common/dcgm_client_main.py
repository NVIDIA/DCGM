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
from time import sleep
import dcgm_client_cli_parser as cli
import signal

###############################################################################
def exit_handler(signum, frame):
    # The Prometheus client does something smarter but more complex
    # Here we just exit
    exit()

###############################################################################
def initialize_signal_handlers():
    signal.signal(signal.SIGINT, exit_handler)
    signal.signal(signal.SIGTERM, exit_handler)


###############################################################################
def main(DRConstructor, name, default_port, add_target_host=False):
    '''
    This main function should work for most DCGM clients. It creates a
    DcgmReader object using DRConstructor and enters a loop that queries DCGM
    for data

    Arguments
    ---------
    DRConstructor:   A constructor for a DcgmReader. The constructor must
                     accept the following keyword arguments:
                         - hostname: DCGM hostname
                         - publish_port: port on which the data is published
                     In some cases, the constructor will also need to accept:
                         - publish_hostname: hostname the data is published to
                         - field_ids: field ids to query and publish
    name:            The name of the client. This is displayed to the user
    default_port:    Default port to publish to

    Keyword arguments
    -----------------
    add_target_host: Boolean that indicates whether this client accepts a
                     publish hostname

    '''

    initialize_signal_handlers()
    settings = cli.parse_command_line(
        name,
        default_port,
        add_target_host=add_target_host,
    )

    # Create a dictionary for the arguments because field_ids might not be
    # provided (if it's None) when we want to use the default in DcgmReader
    dr_args = {
        'hostname': settings['dcgm_hostname'],
        'publish_port': settings['publish_port'],
    }

    # publish_hostname is only available if we add the target_host parameter
    if add_target_host:
        dr_args['publish_hostname'] = settings['publish_hostname']

    if settings['field_ids']:
        dr_args['fieldIds'] = settings['field_ids']

    dr = DRConstructor(**dr_args)

    try:
        while True:
            dr.Process()
            sleep(settings['interval'])
    except KeyboardInterrupt:
        print('Caught CTRL-C. Exiting')
