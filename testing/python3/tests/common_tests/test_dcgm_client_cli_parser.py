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
from _test_helpers import maybemock, skip_test_if_no_mock
from common.Struct import Struct
import logging

from common import dcgm_client_cli_parser as cli

def get_mock_call_name(call):
    return call[0]
def get_mock_call_args(call):
    return call[1]
def get_mock_call_kwargs(call):
    return call[2]

def helper_check_argument_added(call_list, short_param=None, long_param=None, dest=None, type=None):
    calls_with_short_param = list(filter(
        lambda call: get_mock_call_name(call) == 'add_argument' and
                     len(get_mock_call_args(call)) == 2,
        call_list,
    ))

    calls_without_short_param = list(filter(
        lambda call: get_mock_call_name(call) == 'add_argument' and
                     len(get_mock_call_args(call)) == 1,
        call_list,
    ))

    if short_param:
        filtered = list(filter(
            lambda call: get_mock_call_args(call)[0] == short_param and
                         get_mock_call_args(call)[1] == long_param and
                         get_mock_call_kwargs(call)['dest'] == dest,
            calls_with_short_param,
        ))
    else:
        filtered = list(filter(
            lambda call: get_mock_call_args(call)[0] == long_param and
                         get_mock_call_kwargs(call)['dest'] == dest,
            calls_without_short_param,
        ))

    # Check we have found at least one match
    if len(filtered) == 0:
        return False

    # Check the type is correct if it has been provided
    if type and type != get_mock_call_kwargs(filtered[0])['type']:
        return False

    # Check we have found exactly one match
    return len(filtered) == 1

def helper_check_mutually_exclusive_group_added():
    pass

# autospec tells mock to return objects that have the same interface
@maybemock.patch('argparse.ArgumentParser', autospec=True)
def test_create_parser(MockArgumentParser):
    result = cli.create_parser()
    mock_calls = result.mock_calls # pylint: disable=no-member

    assert helper_check_argument_added(mock_calls, '-p', '--publish-port',
                                    'publish_port', type=int)
    assert helper_check_argument_added(mock_calls, '-i', '--interval',
                                    dest='interval', type=int)
    assert helper_check_argument_added(mock_calls, '-f', '--field-ids',
                                    dest='field_ids', type=str)
    assert helper_check_argument_added(mock_calls, long_param='--log-file',
                                    dest='logfile', type=str)
    assert helper_check_argument_added(mock_calls, long_param='--log-level',
                                    dest='loglevel', type=str)
    # TODO mutually-exclusive group tests

@maybemock.patch('argparse.ArgumentParser', autospec=True)
def test_add_target_host_argument(MockArgumentParser):
        parser = MockArgumentParser()
        cli.add_target_host_argument('name', parser)
        mock_calls = parser.mock_calls # pylint: disable=no-member

        assert helper_check_argument_added(mock_calls, '-t', '--publish-hostname',
                                        dest='publish_hostname', type=str)

@skip_test_if_no_mock()
def test_run_parser():
    parser = maybemock.Mock()
    cli.run_parser(parser)
    parser.parse_args.assert_called()

def test_get_field_ids():
    assert cli.get_field_ids(Struct(field_ids="1,2,3")) == [1,2,3]
    assert cli.get_field_ids(Struct(field_ids=[1,2,3])) == [1,2,3]

@maybemock.patch('sys.exit')
def test_get_log_level(mock_exit):
    mock_help = maybemock.Mock()
    assert cli.get_log_level(Struct(loglevel='0')) == logging.CRITICAL
    assert cli.get_log_level(Struct(loglevel='1')) == logging.ERROR
    assert cli.get_log_level(Struct(loglevel='2')) == logging.WARNING
    assert cli.get_log_level(Struct(loglevel='3')) == logging.INFO
    assert cli.get_log_level(Struct(loglevel='4')) == logging.DEBUG
    assert cli.get_log_level(Struct(loglevel='critical')) == logging.CRITICAL
    assert cli.get_log_level(Struct(loglevel='error'))    == logging.ERROR
    assert cli.get_log_level(Struct(loglevel='warning'))  == logging.WARNING
    assert cli.get_log_level(Struct(loglevel='info'))     == logging.INFO
    assert cli.get_log_level(Struct(loglevel='debug'))    == logging.DEBUG

    mock_exit.assert_not_called()
    try: # It raises an exception because it tries to return an undeclared var
        cli.get_log_level(Struct(loglevel='wrong', print_help=mock_help))
    except:
        pass

    mock_exit.assert_called()
    mock_help.assert_called()

def test_parse_command_line():
    # TODO maybe add a test here. This function will be a pain to test
    pass
