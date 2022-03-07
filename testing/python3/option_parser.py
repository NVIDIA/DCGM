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
from optparse import OptionParser, OptionGroup
import apps
import utils
import sys
import test_utils
import logger
import string
import re

def parse_options():
    """
    Parses command line options but doesn't perform error checking of them.
    For the entire process to be completed run later_parse next (after logging is initalized).

    """
    global options
    global args
    parser = OptionParser()
    
    parser.add_option(
            "--test-info",
            dest="test_info",
            action="store_true",
            help="Prints to list of all tests available in the testing framework (can be combined with --verbose)"
            )
    parser.add_option(
            "--no-process-check",
            dest="no_process_check",
            action="store_true",
            default=False,
            help="Does not check if GPUs are being used by other processes"
            )
    parser.add_option(
            "--no-env-check",
            dest="no_env_check",
            action="store_true",
            default=False,
            help="Skips test environment checks for debugging"
            )
    parser.add_option(
            "-v", "--verbose",
            dest="verbose",
            action="store_true",
            help="Prints additional information to stdout"
            )
    parser.add_option(
            "--no-dcgm-trace-patching",
            dest="no_dcgm_trace_patching",
            action="store_true",
            help="Disables trace log patching with information from data/dcgm_decode_db.txt. Use when target ncm version doesn't match exactly data/version.txt CL"
            )
    parser.add_option(
            "--burn",
            dest="burn",
            action="store_true",
            help="Runs a single iteration of the burn_in_stress.py test for sanity check"
            )
    parser.add_option(
            "--eris",
            dest="eris",
            action="store_true",
            help="Prints additional Eris-formatted summary of results"
            )
    parser.add_option(
            "--log-dir",
            dest="log_dir",
            help="Creates all logging files in the specified directory"
            )
    parser.add_option(
            "--dev-mode",
            dest="developer_mode",
            action="store_true",
            help="Run the test framework in developer mode. This mode runs additional tests "+
                 "that should only be run by DCGM developers, either due to intermittency "+
                 "or heavy reliance on environmental conditions."
            )
    parser.add_option(
            "--no-lint",
            dest="lint",
            action="store_false",
            default=True, 
            help="[deprecated] noop preserved to avoid breaking script invocations"

            )
    parser.add_option(
            "-c", "--clear-lint-artifacts",
            dest="clear_lint_artifacts",
            action="store_true",
            default=False,
            help="Delete any files in your development environment that are a product of " +
                 "running the test linter.  This will cause test linting to re-lint ALL python " +
                 "files instead of just those that have changed since their last successful lint.")
    parser.add_option(
            "--profile",
            dest="profile",
            choices=apps.nv_hostengine_app.NvHostEngineApp.supported_profile_tools,
            help="Can be one of: %s " % apps.nv_hostengine_app.NvHostEngineApp.supported_profile_tools +
                 "Turns on profiling of nv-hostengine while tests are running.  " +
                 "This only works for tests that run a standalone hostengine (not embedded).  " +
                 "Valgrind must also be installed.  The selected tool will profile nv-hostengine and " +
                 "generate files for each test that runs.  These files can then be examined using " +
                 "other tools (like KCachegrind for callgrind files).  " +
                 "The tests will output the directory where these files can be found."
            )

    parser.add_option(
            "--use-running-hostengine",
            dest="use_running_hostengine",
            action="store_true",
            default=False,
            help="Can be used to run the test framework against a remote host engine that is already running on the system." +
                 "This option is useful for debugging the stand-alone host engine, which can be started separately inside of" +
                  "valgrind or gdb. This will skip embedded-only tests due to the host engine already being running."
            )
    parser.add_option(
            "--coverage",
            dest="coverage",
            action="store_true",
            default=False,
            help="Informs the framework that this is a coverage build of DCGM and we want to aggregate coverage numbers for the files"
            )
    parser.add_option(
            "--dvssc-testing",
            dest="dvssc_testing",
            action="store_true",
            help="Tests run in DVS-SC"
            )

    test_group = OptionGroup(parser, "Testing modifiers")
    test_group.add_option(
            "-d", "--device",
            dest="device",
            help="Run only on target device (DEVICE_NVML_ID) + global tests"
            )
    test_group.add_option(
            "-f", "--filter-tests",
            dest="filter_tests",
            help="Runs module.test_fn tests that match provided regular expression"
            )
    test_group.add_option(
            "-u", "--non-root-user", 
            dest="non_root_user",
            help="User name that can be used to run tests that should be run as non-root." +
                 " Without this option some test will not be run. Can be used ONLY when current user is root." +
                 " (Linux only)")
    parser.add_option_group(test_group)

    debug_group = OptionGroup(parser, "Debugging options")
    debug_group.add_option(
            "--debug",
            action="store_true",
            help="Disables some of the features as to not interfere with debugging session"
            )
    debug_group.add_option(
            "-b", "--break-at-failure",
            dest="break_at_failure",
            action="store_true",
            help="Start debugger if test fails"
            )
    debug_group.add_option(
            "--force-logging",
            dest="force_logging",
            action="store_true",
            help="Force logging even for actions that have logging disabled by default"
            )
    parser.add_option(
            "--no-library-check",
            dest="no_library_check",
            action="store_true",
            default=False,
            help="Skips the test which verifies that all modules are present."
            )
    parser.add_option_group(debug_group)
    (options, args) = parser.parse_args()

    if options.debug:
        logger.stdout_loglevel = logger.DEBUG

    # by default some actions shouldn't generate any log
    if options.test_info:
        test_utils.noLogging = True

    # unless force log is specified
    if options.force_logging:
        test_utils.noLogging = False
    
    #Change the backup value as well
    test_utils.noLoggingBackup = test_utils.noLogging

    #Use a different logging level for ERIS as we log to the console
    if options.eris:
        test_utils.loggingLevel = "WARNING"
    else:
        test_utils.loggingLevel = "DEBUG"

class OptionParserStub():
    def __init__(self):
        self.profile = False
        self.eris = False
        self.break_at_failure = False
        self.force_logging = False
        self.test_info = False
        self.non_root_user = None
        self.lint = False
        self.clear_lint_artifacts = False
        self.burn = False
        self.log_dir = None
        self.verbose = False
        self.no_dcgm_trace_patching = True
        self.use_running_hostengine = False
        self.no_process_check = False
        self.developer_mode = False
        self.no_env_check = False
        self.coverage = ''
        self.dvssc_testing = False


def initialize_as_stub():
    """
    Initialize the values of this library as a stub module. This is so we can call DCGM framework classes
    from outside the DCGM framework
    """
    global options
    options = OptionParserStub()


def validate():
    """
    Should be run after logging is enabled.

    """
    logger.debug("Running script with argv: " + str(sys.argv))
    logger.debug("Parsed options: " + str(options))
    logger.debug("Unparsed args: " + str(args))
    
    if args:
        logger.fatal("Unrecognized command line arguments: " + " ".join(args))

    if options.non_root_user:
        try:
            utils.get_user_idinfo(options.non_root_user)
        except KeyError:
            logger.fatal("User '%s' doesn't exist" % options.non_root_user)

    if options.non_root_user and not utils.is_root():
        logger.fatal("[-u | --non-root-user] flags are invalid when current user is not root")

    if options.break_at_failure:
        options.debug = True

    if options.filter_tests:
        options.filter_tests = re.compile(options.filter_tests)
    
    logger.debug("Preprocessed options: " + str(options))
