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
import dcgm_structs
import dcgm_structs_internal
import dcgm_agent
import dcgm_agent_internal
import logger
import test_utils
import dcgm_fields
import pydcgm
from dcgm_structs import dcgmExceptionClass

import time

def get_usec_since_1970():
    sec = time.time()
    return int(sec * 1000000.0)


def helper_verify_fv_equal(fv1, fv2):
    '''
    Helper function to verify that fv1 == fv2 with useful errors if they are not equal. An
    assertion is thrown from this if they are not equal
    '''
    #assert fv1.version == fv2.version #Don't check version. We may be comparing injected values to retrieved ones
    assert fv1.fieldId == fv2.fieldId, "%d != %d" % (fv1.value.fieldId, fv2.value.fieldId)
    assert fv1.status == fv2.status, "%d != %d" % (fv1.value.status, fv2.value.status)
    assert fv1.fieldType == fv2.fieldType, "%d != %d" % (fv1.value.fieldType, fv2.value.fieldType)
    assert fv1.ts == fv2.ts, "%d != %d" % (fv1.value.ts, fv2.value.ts)
    assert fv1.value.i64 == fv2.value.i64, "%d != %d" % (fv1.value.i64, fv2.value.i64)

@test_utils.run_with_embedded_host_engine()
@test_utils.run_with_injection_gpus(1)
def test_dcgm_injection_agent(handle, gpuIds):
    """
    Verifies that injection works with the agent host engine
    """
    gpuId = gpuIds[0]

    #Make a base value that is good for starters
    fvGood = dcgm_structs_internal.c_dcgmInjectFieldValue_v1()
    fvGood.version = dcgm_structs_internal.dcgmInjectFieldValue_version1
    fvGood.fieldId = dcgm_fields.DCGM_FI_DEV_ECC_CURRENT
    fvGood.status = 0
    fvGood.fieldType = ord(dcgm_fields.DCGM_FT_INT64)
    fvGood.ts = get_usec_since_1970()
    fvGood.value.i64 = 1

    fieldInfoBefore = dcgm_agent_internal.dcgmGetCacheManagerFieldInfo(handle, gpuId, fvGood.fieldId)
    countBefore = fieldInfoBefore.numSamples

    #This will throw an exception if it fails
    dcgm_agent_internal.dcgmInjectFieldValue(handle, gpuId, fvGood)

    fieldInfoAfter = dcgm_agent_internal.dcgmGetCacheManagerFieldInfo(handle, gpuId, fvGood.fieldId)
    countAfter = fieldInfoAfter.numSamples

    assert countAfter > countBefore, "Expected countAfter %d > countBefore %d after injection" % (countAfter, countBefore)

    #Fetch the value we just inserted and verify its attributes are the same
    fvFetched = dcgm_agent_internal.dcgmGetLatestValuesForFields(handle, gpuId, [fvGood.fieldId, ])[0]
    helper_verify_fv_equal(fvFetched, fvGood)

    #Should be able to insert a null timestamp. The agent will just use "now"
    fvAlsoGood = fvGood
    fvAlsoGood.ts = 0
    #This will thrown an exception if it fails
    dcgm_agent_internal.dcgmInjectFieldValue(handle, gpuId, fvAlsoGood)

    #Now make some attributes bad and expect an error
    fvBad = fvGood
    fvBad.fieldType = ord(dcgm_fields.DCGM_FT_DOUBLE)
    with test_utils.assert_raises(dcgmExceptionClass(dcgm_structs.DCGM_ST_BADPARAM)):
        dcgm_agent_internal.dcgmInjectFieldValue(handle, gpuId, fvBad)

    fvGood.fieldType = ord(dcgm_fields.DCGM_FT_INT64)
    """ TODO: DCGM-2130 - Restore this test when protobuf is removed
    #Now make some attributes bad and expect an error
    fvBad = fvGood
    fvBad.version = 0
    with test_utils.assert_raises(dcgmExceptionClass(dcgm_structs.DCGM_ST_VER_MISMATCH)):
        dcgm_agent_internal.dcgmInjectFieldValue(handle, gpuId, fvBad)

    fvGood.version = dcgm_structs_internal.dcgmInjectFieldValue_version1
    """
    fvBad = fvGood
    fvBad.fieldId = dcgm_fields.DCGM_FI_MAX_FIELDS + 100
    with test_utils.assert_raises(dcgmExceptionClass(dcgm_structs.DCGM_ST_BADPARAM)):
        dcgm_agent_internal.dcgmInjectFieldValue(handle, gpuId, fvBad)

@test_utils.run_with_standalone_host_engine()
@test_utils.run_with_initialized_client()
@test_utils.run_with_injection_gpus(1)
def test_dcgm_injection_remote(handle, gpuIds):
    """
    Verifies that injection works with the remote host engine
    """
    gpuId = gpuIds[0]

    #Make a base value that is good for starters
    fvGood = dcgm_structs_internal.c_dcgmInjectFieldValue_v1()
    fvGood.version = dcgm_structs_internal.dcgmInjectFieldValue_version1
    fvGood.fieldId = dcgm_fields.DCGM_FI_DEV_ECC_CURRENT
    fvGood.status = 0
    fvGood.fieldType = ord(dcgm_fields.DCGM_FT_INT64)
    fvGood.ts = get_usec_since_1970()
    fvGood.value.i64 = 1

    fieldInfoBefore = dcgm_agent_internal.dcgmGetCacheManagerFieldInfo(handle, gpuId, fvGood.fieldId)
    countBefore = fieldInfoBefore.numSamples

    #This will throw an exception if it fails
    dcgm_agent_internal.dcgmInjectFieldValue(handle, gpuId, fvGood)

    fieldInfoAfter = dcgm_agent_internal.dcgmGetCacheManagerFieldInfo(handle, gpuId, fvGood.fieldId)
    countAfter = fieldInfoAfter.numSamples

    assert countAfter > countBefore, "Expected countAfter %d > countBefore %d after injection" % (countAfter, countBefore)

    #Fetch the value we just inserted and verify its attributes are the same
    fvFetched = dcgm_agent_internal.dcgmGetLatestValuesForFields(handle, gpuId, [fvGood.fieldId, ])[0]
    helper_verify_fv_equal(fvFetched, fvGood)

    #Should be able to insert a null timestamp. The agent will just use "now"
    fvAlsoGood = fvGood
    fvAlsoGood.ts = 0
    #This will thrown an exception if it fails
    dcgm_agent_internal.dcgmInjectFieldValue(handle, gpuId, fvAlsoGood)

    #Now make some attributes bad and expect an error
    fvBad = fvGood
    fvBad.fieldType = ord(dcgm_fields.DCGM_FT_DOUBLE)
    with test_utils.assert_raises(dcgmExceptionClass(dcgm_structs.DCGM_ST_BADPARAM)):
        dcgm_agent_internal.dcgmInjectFieldValue(handle, gpuId, fvBad)

    fvGood.fieldType = ord(dcgm_fields.DCGM_FT_INT64)
    """ TODO: DCGM-2130 - Restore this test when protobuf is removed
    #Now make some attributes bad and expect an error
    fvBad = fvGood
    fvBad.version = 0
    with test_utils.assert_raises(dcgmExceptionClass(dcgm_structs.DCGM_ST_VER_MISMATCH)):
        dcgm_agent_internal.dcgmInjectFieldValue(handle, gpuId, fvBad)

    fvGood.version = dcgm_structs_internal.dcgmInjectFieldValue_version1
    """
    fvBad = fvGood
    fvBad.fieldId = dcgm_fields.DCGM_FI_MAX_FIELDS + 100
    with test_utils.assert_raises(dcgmExceptionClass(dcgm_structs.DCGM_ST_BADPARAM)):
        dcgm_agent_internal.dcgmInjectFieldValue(handle, gpuId, fvBad)

def helper_verify_multi_values(fieldValues, order, injectedValues):
    """
    Helper to verify that a returned list of values is internally consistent
    The 'values' parameter is expected to be the result of a
    dcgm*GetMultipleValuesForField request

    This function will assert if there is a problem detected
    """
    assert order == dcgm_structs.DCGM_ORDER_ASCENDING or order == dcgm_structs.DCGM_ORDER_DESCENDING, "Invalid order %d" % order

    if len(fieldValues) < 2:
        return

    if order == dcgm_structs.DCGM_ORDER_DESCENDING:
        injectedValues = list(reversed(injectedValues))

    Nerrors = 0

    for i, fv in enumerate(fieldValues):
        injectedFv = injectedValues[i]

        if fv.ts == 0:
            logger.error("Null timestamp at index %d" % i)
            Nerrors += 1
        if fv.ts != injectedFv.ts:
            logger.error("Timestamp mismatch at index %d: read %d. injected %d" % (i, fv.ts, injectedFv.ts))
            Nerrors += 1
        if fv.value.i64 != injectedFv.value.i64:
            logger.error("Value mismatch at index %d: read %d. injected %d" % (i, fv.value.i64, injectedFv.value.i64))
            Nerrors += 1

        #Don't compare against previous until we are > 1
        if i < 1:
            continue

        fvPrev = fieldValues[i-1]

        if order == dcgm_structs.DCGM_ORDER_ASCENDING:
            if fv.ts <= fvPrev.ts:
                logger.error("Out of order ASC timestamp at index %d, fv.ts %d, fvPrev.ts %d" % (i, fv.ts, fvPrev.ts))
                Nerrors += 1
        else: #Descending
            if fv.ts >= fvPrev.ts:
                logger.error("Out of order DESC timestamp at index %d, fv.ts %d, fvPrev.ts %d" % (i, fv.ts, fvPrev.ts))
                Nerrors += 1

    assert Nerrors < 1, "Comparison errors occurred"

@test_utils.run_with_embedded_host_engine()
@test_utils.run_with_injection_gpus(1)
def test_dcgm_injection_multi_fetch_agent(handle, gpuIds):
    """
    Verify that multi-fetches work with the agent
    """
    gpuId = gpuIds[0]
    NinjectValues = 10

    firstTs = get_usec_since_1970()
    lastTs = 0
    injectedValues = []

    #Make a base value that is good for starters


    #Inject the values we're going to fetch
    for i in range(NinjectValues):

        fvGood = dcgm_structs_internal.c_dcgmInjectFieldValue_v1()
        fvGood.version = dcgm_structs_internal.dcgmInjectFieldValue_version1
        fvGood.fieldId = dcgm_fields.DCGM_FI_DEV_ECC_PENDING
        fvGood.status = 0
        fvGood.fieldType = ord(dcgm_fields.DCGM_FT_INT64)
        fvGood.ts = firstTs + i
        fvGood.value.i64 = 1 + i

        #This will throw an exception if it fails
        dcgm_agent_internal.dcgmInjectFieldValue(handle, gpuId, fvGood)

        injectedValues.append(fvGood)

    #Fetch in forward order with no timestamp. Verify
    startTs = 0
    endTs = 0
    maxCount = 2 * NinjectValues #Pick a bigger number so we can verify only NinjectValues come back
    order = dcgm_structs.DCGM_ORDER_ASCENDING
    fvFetched = dcgm_agent_internal.dcgmGetMultipleValuesForField(handle, gpuId, fvGood.fieldId, maxCount, startTs, endTs, order)

    assert len(fvFetched) == NinjectValues, "Expected %d rows. Got %d" % (NinjectValues, len(fvFetched))
    helper_verify_multi_values(fvFetched, order, injectedValues)

    #Now do the same fetch with descending values
    startTs = 0
    endTs = 0
    maxCount = 2 * NinjectValues #Pick a bigger number so we can verify only NinjectValues come back
    order = dcgm_structs.DCGM_ORDER_DESCENDING
    fvFetched = dcgm_agent_internal.dcgmGetMultipleValuesForField(handle, gpuId, fvGood.fieldId, maxCount, startTs, endTs, order)

    assert len(fvFetched) == NinjectValues, "Expected %d rows. Got %d" % (NinjectValues, len(fvFetched))
    helper_verify_multi_values(fvFetched, order, injectedValues)

@test_utils.run_with_standalone_host_engine()
@test_utils.run_with_initialized_client()
@test_utils.run_with_injection_gpus(1)
def test_dcgm_injection_multi_fetch_remote(handle, gpuIds):
    """
    Verify that multi-fetches work with the agent
    """

    gpuId = gpuIds[0]
    NinjectValues = 10

    firstTs = get_usec_since_1970()
    lastTs = 0
    injectedValues = []

    #Inject the values we're going to fetch
    for i in range(NinjectValues):
        fvGood = dcgm_structs_internal.c_dcgmInjectFieldValue_v1()
        fvGood.version = dcgm_structs_internal.dcgmInjectFieldValue_version1
        fvGood.fieldId = dcgm_fields.DCGM_FI_DEV_ECC_PENDING
        fvGood.status = 0
        fvGood.fieldType = ord(dcgm_fields.DCGM_FT_INT64)
        fvGood.ts = firstTs + i
        fvGood.value.i64 = 1 + i

        #This will throw an exception if it fails
        dcgm_agent_internal.dcgmInjectFieldValue(handle, gpuId, fvGood)

        injectedValues.append(fvGood)

    lastTs = fvGood.ts

    #Fetch in forward order with no timestamp. Verify
    startTs = 0
    endTs = 0
    maxCount = 2 * NinjectValues #Pick a bigger number so we can verify only NinjectValues come back
    order = dcgm_structs.DCGM_ORDER_ASCENDING
    fvFetched = dcgm_agent_internal.dcgmGetMultipleValuesForField(handle, gpuId, fvGood.fieldId, maxCount, startTs, endTs, order)

    assert len(fvFetched) == NinjectValues, "Expected %d rows. Got %d" % (NinjectValues, len(fvFetched))
    helper_verify_multi_values(fvFetched, order, injectedValues)

    #Now do the same fetch with descending values
    startTs = 0
    endTs = 0
    maxCount = 2 * NinjectValues #Pick a bigger number so we can verify only NinjectValues come back
    order = dcgm_structs.DCGM_ORDER_DESCENDING
    fvFetched = dcgm_agent_internal.dcgmGetMultipleValuesForField(handle, gpuId, fvGood.fieldId, maxCount, startTs, endTs, order)

    assert len(fvFetched) == NinjectValues, "Expected %d rows. Got %d" % (NinjectValues, len(fvFetched))
    helper_verify_multi_values(fvFetched, order, injectedValues)

def helper_test_dcgm_injection_summaries(handle, gpuIds):

    gpuId = gpuIds[0]
    
    # Watch the field we're inserting into
    dcgm_agent_internal.dcgmWatchFieldValue(handle, gpuId, dcgm_fields.DCGM_FI_DEV_ECC_SBE_AGG_TOTAL, 1, 3600.0,
                                            10000)

    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()

    #Make a base value that is good for starters
    field = dcgm_structs_internal.c_dcgmInjectFieldValue_v1()
    field.version = dcgm_structs_internal.dcgmInjectFieldValue_version1
    field.fieldId = dcgm_fields.DCGM_FI_DEV_ECC_SBE_AGG_TOTAL
    field.status = 0
    field.fieldType = ord(dcgm_fields.DCGM_FT_INT64)

    baseTime = get_usec_since_1970()

    for i in range(0, 10):
        field.ts = baseTime + i
        field.value.i64 = i
        ret = dcgm_agent_internal.dcgmInjectFieldValue(handle, gpuId, field)
        assert (ret == dcgm_structs.DCGM_ST_OK)
    
    time.sleep(1)

    systemObj.UpdateAllFields(1)

    tmpMask = dcgm_structs.DCGM_SUMMARY_MIN | dcgm_structs.DCGM_SUMMARY_MAX 
    tmpMask = tmpMask | dcgm_structs.DCGM_SUMMARY_AVG | dcgm_structs.DCGM_SUMMARY_DIFF
    # Pass baseTime for the start to get nothing from the first query
    with test_utils.assert_raises(dcgm_structs.dcgmExceptionClass(dcgm_structs.DCGM_ST_NO_DATA)):
        request = dcgm_agent.dcgmGetFieldSummary(handle, dcgm_fields.DCGM_FI_DEV_ECC_SBE_AGG_TOTAL,
                                                 dcgm_fields.DCGM_FE_GPU, gpuId, tmpMask, baseTime - 60,
                                                 baseTime - 30)

    # Now adjust the time so we get values
    request = dcgm_agent.dcgmGetFieldSummary(handle, dcgm_fields.DCGM_FI_DEV_ECC_SBE_AGG_TOTAL,
                                             dcgm_fields.DCGM_FE_GPU, gpuId, tmpMask, 0, 0)
    assert (request.response.values[0].i64 == 0)
    assert (request.response.values[1].i64 == 9)
    assert (request.response.values[2].i64 == 4)
    assert (request.response.values[3].i64 == 9)

@test_utils.run_with_embedded_host_engine()
@test_utils.run_with_injection_gpus(1)
def test_dcgm_injection_summaries_embedded(handle, gpuIds):
    """
    Verifies that inject works and we can get summaries of that data
    """
    helper_test_dcgm_injection_summaries(handle, gpuIds)




