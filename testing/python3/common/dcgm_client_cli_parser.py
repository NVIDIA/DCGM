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
from os import environ
import argparse
import logging
import sys

###############################################################################
def create_parser(
        publish_port = 8000,
        interval = 10,
        name = 'the monitoring tool', # Replace with 'prometheus', 'telegraf', etc.
        field_ids = None,
        log_file = None,
        log_level = 'CRITICAL',
        dcgm_hostname = environ.get('DCGM_HOSTNAME') or 'localhost',
):
    '''
    Create a parser that defaults to sane parameters.

    The default parameters can be overridden through keyword arguments.

    Note: if DCGM_HOSTNAME is set as an environment variable, it is used as
    the default instead of localhost
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--publish-port', dest='publish_port', type=int, default=publish_port,
                        help='TCP port that the client should publish to. Default={}.'.format(publish_port))
    parser.add_argument('-i', '--interval', dest='interval', type=int, default=interval,
                        help='How often the client should retrieve new values from DCGM in seconds. Default={}.'.format(interval))
    parser.add_argument('-f', '--field-ids', dest='field_ids', type=str, default=field_ids,
                        help='Comma-separated list of field IDs that should be retrieved from DCGM. '+
                             'The full list of available field IDs can be obtained from dcgm_fields.h, dcgm_fields.py, '+
                             'or running \'dcgmi dmon -l\'.')
    parser.add_argument('--log-file', dest='logfile', type=str, default=log_file,
                        help='A path to a log file for recording what information is being sent to {}'.format(name))
    parser.add_argument('--log-level', dest='loglevel', type=str, default=log_level,
                        help='Specify a log level to use for logging.\n\tCRITICAL (0) - log only critical errors that drastically affect execution' +
                             '\n\tERROR (1) - Log any error in execution\n\tWARNING (2) - Log all warnings and errors that occur' +
                             '\n\tINFO (3) - Log informational messages about program execution in addition to warnings and errors' +
                             '\n\tDEBUG (4) - Log debugging information in addition to all information about execution' +

                             '\nDefault: {}'.format(log_level))

    group = parser.add_mutually_exclusive_group()
    group.add_argument('-n', '--hostname', dest='hostname', type=str, default=dcgm_hostname,
                       help='IP/hostname where the client should query DCGM for values. Default={} (all interfaces).' .format(dcgm_hostname))
    group.add_argument('-e', '--embedded', dest='embedded', action='store_true',
                       help='Launch DCGM from within this process instead of connecting to nv-hostengine.')

    return parser

def add_custom_argument(parser, *args, **kwargs):
    parser.add_argument(*args, **kwargs)

###############################################################################
def add_target_host_argument(name, parser, default_target='localhost'):
    parser.add_argument('-t', '--publish-hostname', dest='publish_hostname',
                        type=str, default=default_target,
                        help='The hostname at which the client will publish the readings to {}'.format(name))

###############################################################################
def run_parser(parser):
    '''
    Run a parser created using create_parser
    '''
    return parser.parse_args()

###############################################################################
def get_field_ids(args):
    # This indicates the user supplied a string, so we should override the
    # default
    if isinstance(args.field_ids, str):
        tokens = args.field_ids.split(",")
        field_ids = [int(token) for token in tokens]
        return field_ids
    # The default object should already be an array of ints. Just return it
    else:
        return args.field_ids

###############################################################################
def get_log_level(args):
    levelStr = args.loglevel.upper()
    if levelStr == '0' or levelStr == 'CRITICAL':
        numeric_log_level = logging.CRITICAL
    elif levelStr == '1' or levelStr == 'ERROR':
        numeric_log_level = logging.ERROR
    elif levelStr == '2' or levelStr == 'WARNING':
        numeric_log_level = logging.WARNING
    elif levelStr == '3' or levelStr == 'INFO':
        numeric_log_level = logging.INFO
    elif levelStr == '4' or levelStr == 'DEBUG':
        numeric_log_level = logging.DEBUG
    else:
        print("Could not understand the specified --log-level '%s'" % (args.loglevel))
        args.print_help()
        sys.exit(2)
    return numeric_log_level

###############################################################################
def parse_command_line(name, default_port, add_target_host=False):
    # Fields we accept raw from the CLI
    FIELDS_AS_IS = ['publish_port', 'interval', 'logfile', 'publish_hostname']

    parser = create_parser(
        name=name,
        publish_port=default_port,
    )

    if add_target_host:
        add_target_host_argument(name, parser)

    args = run_parser(parser)
    field_ids = get_field_ids(args)
    log_level = get_log_level(args)

    args_as_dict = vars(args)
    settings = {i: args_as_dict[i] for i in FIELDS_AS_IS}
    settings['dcgm_hostname'] = None if args.embedded else args.hostname
    settings['field_ids'] = field_ids
    settings['log_level'] = log_level

    return settings
