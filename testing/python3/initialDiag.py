# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

# This is the initial diagnostic, which runs the software and context
# create tests. If these fail, this system is unprepared to run other
# tests correctly.

from functools import wraps

import pydcgm
import DcgmDiag
import dcgm_structs
import logger
import option_parser
import test_utils

class DcgmInitialDiagError(Exception):
    def __init__(self, msg):
        super().__init__(msg)

def handleMigFailure(err):
    # Returns `true`, and logs the failure if the exception is a MIG related
    # failure, `false` otherwise.

    # migFailures entries must match MIG failures that can be reported by
    # NVVS, see NvidiaValidationSuite.cpp

    import re
    migFailuresRe = re.compile('|'.join([
        r'CUDA does not support enumerating GPUs',
        r'MIG configuration is incompatible with the diagnostic',
        r'MIG is enabled, but no compute instances are configured',
        r'failed to set the environment variable CUDA_VISIBLE_DEVICES',
    ]))

    if migFailuresRe.search(str(err)):
        logger.warning(f'Problem executing initial diagnostic: {str(err)}. (Ignored because MIG is enabled.)')
        return True

    # Caller should forward the exception
    return False

def runDiagForGpuGroup(handle, gpuIds):
    # Run diag for the specified set of gpuIds. Check for errors.
    try:
        # 'context_create' currently implies 'software'
        dd = DcgmDiag.DcgmDiag(testNamesStr='context_create', gpuIds=gpuIds)
        dcgmHandle = pydcgm.DcgmHandle(handle=handle)
        response = dd.Execute(dcgmHandle.handle)
    except Exception as err:
        if option_parser.options.ignore_init_diag:
            logger.warning(f'Problem executing initial diagnostic: {str(err)}. (Ignored by user request).')
            return
        raise DcgmInitialDiagError(f'Problem executing initial diagnostic: {str(err)}')

    failingTests = []
    sysErrs = sum(1 for err in filter(lambda cur: cur.testId == dcgm_structs.DCGM_DIAG_RESPONSE_SYSTEM_ERROR,
                                    response.errors[:min(response.numErrors, dcgm_structs.DCGM_DIAG_RESPONSE_ERRORS_MAX)]))

    # Ignore non-passing results (SKIP, NOT_RUN) that aren't FAIL.
    for testId in range(min(response.numTests, dcgm_structs.DCGM_DIAG_RESPONSE_TESTS_MAX)):
        if response.tests[testId].result == dcgm_structs.DCGM_DIAG_RESULT_FAIL:
            failingTests.append(response.tests[testId].name)

    for error in response.errors[:min(response.numErrors, dcgm_structs.DCGM_DIAG_RESPONSE_ERRORS_MAX)]:
        if error.testId == dcgm_structs.DCGM_DIAG_RESPONSE_SYSTEM_ERROR:
            logger.error(f'System error: {error.msg}')
        else:
            logger.error(f'{response.tests[error.testId].name}: {error.msg}')

    if sysErrs:
        msg = f'Tests cannot run because a system error occurred while running initial diag.'
        if option_parser.options.ignore_init_diag:
            logger.warning(f'{msg} Run diag, address all identified issues, and retry before filing a bug report. (Ignored by user request).')
            return
        else:
            msg = f'{msg}  Run diag and address all identified issues before proceeding.'
            raise DcgmInitialDiagError(msg)

    if failingTests:
        msg = f'Tests cannot run because the following diagnostics failed: {failingTests}.'
        if option_parser.options.ignore_init_diag:
            logger.warning(f'{msg} Run diag, address all identified issues, and retry before filing a bug report. (Ignored by user request).')
            return
        else:
            msg = f'{msg}  Run diag and address all identified issues before proceeding.'
            raise DcgmInitialDiagError(msg)

def runInitialDiag(handle):
    # Run the initial diagnostic. On failure, raise DcgmInitialDiagError.
    # Assumes persistence mode was enabled by run_tests()
    if option_parser.options.filter_tests:
        test_utils.skip_test('Skipping initial diagnostic because filter-tests is specified.')

    try:
        gpuIds = test_utils.get_live_gpu_ids(handle)
    except dcgm_structs.DCGMError_NvmlNotLoaded:
        gpuIds = []
    if not gpuIds:
        test_utils.skip_test('Skipping initial diagnostic because no live GPUs were found.')

    migEnabled = test_utils.is_mig_mode_enabled()
    if migEnabled:
        logger.debug('Performing initial diagnostic sequentially because MIG is enabled.')
        gpuGroups = [ [gpuId] for gpuId in gpuIds ]
    else:
        gpuGroups = test_utils.group_gpu_ids_by_sku(handle, gpuIds)

    for gpuGroup in gpuGroups:
        if gpuGroup:
            try:
                runDiagForGpuGroup(handle, gpuGroup)
            except DcgmInitialDiagError as e:
                if migEnabled and handleMigFailure(e):
                    pass
                else:
                    raise
