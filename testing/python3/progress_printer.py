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
import test_utils
import logger
import option_parser
import datetime

_erisTestNumber = 0

class ProgressPrinter(object):
    def subtest_start(self, subtest):
        pass

    def subtest_finish(self, subtest):
        pass

class DefaultProgressPrinter(ProgressPrinter):
    def subtest_start(self, subtest):
        global _erisTestNumber
        # defer the quiet tests. If they don't fail there's no need to print their name
        # but print right away all non quiet tests

        if option_parser.options.eris:
            if subtest.depth == 3:
                self.childrenTest = ""
                _erisTestNumber += 1
                subtest._DefaultProgressPrinter_header_log_id = logger.info("&&&& RUNNING %s - %d" % (subtest.name, _erisTestNumber), defer=subtest.quiet)
            else:
                subtest._DefaultProgressPrinter_header_log_id = logger.info("", defer=subtest.quiet)
        else:
           subtest._DefaultProgressPrinter_header_log_id = logger.info("- Test %s" % (subtest.name), defer=subtest.quiet)

        logger.indent_icrement()

        if subtest.name.startswith("test_") and not subtest.name.endswith("restore state"):
            logger.info("Test %s start time: %s" % (subtest.name, datetime.datetime.now()))

    def subtest_finish(self, subtest):
        global _erisTestNumber
        
        if subtest.name.startswith("test_") and not subtest.name.endswith("restore state"):
            logger.info("Test %s end time: %s" % (subtest.name, datetime.datetime.now()))
        
        if subtest.result == test_utils.SubTest.FAILED and test_utils.reRunning == True:
            subtest.result = test_utils.SubTest.FAILURE_LOGGED
            logger.error(subtest.result_details)
        if subtest.result == test_utils.SubTest.FAILED:
            logger.error(subtest.result_details)

        logger.indent_decrement()
        logger.pop_defered(subtest._DefaultProgressPrinter_header_log_id)

        if subtest.result == test_utils.SubTest.SKIPPED:
            with logger.IndentBlock():
                logger.info("SKIPPED: " + str(subtest.result_details_raw.exception))
        elif subtest.result != test_utils.SubTest.SUCCESS:
            logger.info("<< %s" % (subtest))

        if option_parser.options.eris:
            # Validating results of subtest with depth bigger than 3 
            if subtest.depth > 3 and not subtest.name.endswith("restore state"):
                 if subtest.result == test_utils.SubTest.FAILED:
                    self.childrenTest = "F"

            if subtest.depth == 3 and subtest.name.startswith("test_") and not subtest.name.endswith("restore state"):
                if subtest.result == test_utils.SubTest.SKIPPED:
                    logger.info("&&&& WAIVED %s - %d" % (subtest.name, _erisTestNumber))
                elif subtest.result == test_utils.SubTest.SUCCESS and not self.childrenTest == "F":
                    logger.info("&&&& PASSED %s - %d" % (subtest.name, _erisTestNumber))
                elif subtest.result == test_utils.SubTest.FAILURE_LOGGED:
                    logger.info("&&&& FAILURE_LOGGED %s - %d" % (subtest.name, _erisTestNumber))
                elif subtest.result == test_utils.SubTest.FAILED or self.childrenTest == "F":
                    logger.info("&&&& FAILED %s - %d" % (subtest.name, _erisTestNumber))
                elif subtest.result == test_utils.SubTest.NOT_CONNECTED:
                    logger.info("&&&& RETRY %s - %d" % (subtest.name, _erisTestNumber))

progress_printer = DefaultProgressPrinter()
