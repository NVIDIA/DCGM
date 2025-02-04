/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <catch2/catch_all.hpp>

#include <DcgmError.h>
#include <dcgm_errors.h>
#include <iostream>

extern dcgm_error_meta_t dcgmErrorMeta[DCGM_FR_ERROR_SENTINEL];

TEST_CASE("dcgm_errors: check initialization")
{
    for (unsigned int i = 0; i < DCGM_FR_ERROR_SENTINEL; i++)
    {
        const auto errorMeta = dcgmGetErrorMeta(static_cast<dcgmError_t>(i));
        CHECK(errorMeta->errorId == i);
        CHECK(errorMeta->msgFormat != nullptr);
        CHECK(errorMeta->suggestion != nullptr);
    }
}

/**
 * Test that each error prints as expected with _NEXT message.
 * Results can be found in the _out/build directory
 */
TEST_CASE("dcgm_errors: check full message")
{
    DcgmError d { 0 };

    DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_OK, d); //           "The operation completed successfully."
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_UNKNOWN, d); //      "Unknown error."
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_UNRECOGNIZED, d); // "Unrecognized error code."
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    // replay limit, gpu id, replay errors detected
    DCGM_ERROR_FORMAT_MESSAGE(
        DCGM_FR_PCI_REPLAY_RATE, d, 0, 0, 0); // "Detected more than %u PCIe replays per minute for GPU %u : %d"
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    // dbes deteced, gpu id
    DCGM_ERROR_FORMAT_MESSAGE(
        DCGM_FR_VOLATILE_DBE_DETECTED, d, 0, 0); // "Detected %d volatile double-bit ECC error(s) in GPU %u."
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    // sbe limit, gpu id, sbes detected
    DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_VOLATILE_SBE_DETECTED,
                              d,
                              0,
                              0,
                              0); // "More than %u single-bit ECC error(s) detected in GPU %u Volatile SBEs: %lld"
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    // gpu id
    DCGM_ERROR_FORMAT_MESSAGE(
        DCGM_FR_PENDING_PAGE_RETIREMENTS, d, 0); // "A pending retired page has been detected in GPU %u."
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    // retired pages detected, gpud id
    DCGM_ERROR_FORMAT_MESSAGE(
        DCGM_FR_RETIRED_PAGES_LIMIT, d, 0, 0); // "%u or more retired pages have been detected in GPU %u. "
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    // retired pages due to dbes detected, gpu id
    DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_RETIRED_PAGES_DBE_LIMIT,
                              d,
                              0,
                              0); // "An excess of %u retired pages due to DBEs have been detected and more than one
                                  // page has been retired due to DBEs in the past week in GPU %u."
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    // gpu id
    DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_CORRUPT_INFOROM, d, 0); // "A corrupt InfoROM has been detected in GPU %u."
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    // gpu id
    DCGM_ERROR_FORMAT_MESSAGE(
        DCGM_FR_CLOCKS_EVENT_THERMAL, d, 0); // "Detected clocks event due to thermal violation in GPU %u."
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    // gpu id
    DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_POWER_UNREADABLE, d, 0); // "Cannot reliably read the power usage for GPU %u."
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    // gpu id
    DCGM_ERROR_FORMAT_MESSAGE(
        DCGM_FR_CLOCKS_EVENT_POWER, d, 0); // "Detected clocks event due to power violation in GPU %u."
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    // nvlink errors detected, nvlink id, error threshold
    DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_NVLINK_ERROR_THRESHOLD,
                              d,
                              0,
                              0,
                              0); // "Detected %ld %s NvLink errors on GPU %u's NVLink which exceeds threshold of %u"
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    // gpu id, nvlink id
    DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_NVLINK_DOWN, d, 0, 0); // "GPU %u's NvLink link %d is currently down"
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    // nvlinks up, expected nvlinks up
    DCGM_ERROR_FORMAT_MESSAGE(
        DCGM_FR_GPU_EXPECTED_NVLINKS_UP, d, 0, 0); // "Only %u NvLinks are up out of the expected %u"
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    // switch id, nvlinks up, expected nvlinks up
    DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_NVSWITCH_EXPECTED_NVLINKS_UP,
                              d,
                              0,
                              0,
                              0); // "NvSwitch %u - Only %u NvLinks are up out of the expected %u"
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    // nvswitch id, nvlink id
    DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_NVSWITCH_FATAL_ERROR, d, 0, 0); // "Detected fatal errors on NvSwitch %u link %u"
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    // nvswitch id, nvlink id
    DCGM_ERROR_FORMAT_MESSAGE(
        DCGM_FR_NVSWITCH_NON_FATAL_ERROR, d, 0, 0); // "Detected nonfatal errors on NvSwitch %u link %u"
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    // nvswitch id, nvlink port
    DCGM_ERROR_FORMAT_MESSAGE(
        DCGM_FR_NVSWITCH_DOWN, d, 0, 0); // "NvSwitch physical ID %u's NvLink port %d is currently down."
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    // file path, error detail
    DCGM_ERROR_FORMAT_MESSAGE(
        DCGM_FR_NO_ACCESS_TO_FILE, d, "path", "error"); // "File %s could not be accessed directly: %s"
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    // purpose for communicating with NVML, NVML error as string, NVML error
    DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_NVML_API, d, "func", "error"); // "Error calling NVML API %s: %s"
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    DCGM_ERROR_FORMAT_MESSAGE(
        DCGM_FR_DEVICE_COUNT_MISMATCH,
        d); // "The number of devices NVML returns is different than the number of devices in /dev."
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    // function name
    DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_BAD_PARAMETER, d, "func"); // "Bad parameter to function %s cannot be processed"
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    // library name, error returned from dlopen
    DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_CANNOT_OPEN_LIB, d, "lib", "error"); // "Cannot open library %s: '%s'"
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    // the name of the denylisted driver
    DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_DENYLISTED_DRIVER, d, "driver"); // "Found driver on the denylist: %s"
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    // the name of the function that wasn't found
    DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_NVML_LIB_BAD, d, "func"); // "Cannot get pointer to %s from libnvidia-ml.so"
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_GRAPHICS_PROCESSES,
                              d); // "NVVS has detected processes with graphics contexts open running on at least one
                                  // GPU. This may cause some tests to fail."
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    // error message from the API call
    DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_HOSTENGINE_CONN, d, "error"); // "Could not connect to the host engine: '%s'"
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    // field name, gpu id
    DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_FIELD_QUERY, d, "field", 0); // "Could not query field %s for GPU %u"
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    // environment variable name
    DCGM_ERROR_FORMAT_MESSAGE(
        DCGM_FR_BAD_CUDA_ENV, d, "env"); // "Found CUDA performance-limiting environment variable '%s'."
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    // gpu id
    DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_PERSISTENCE_MODE, d, 0); // "Persistence mode for GPU %u is disabled."
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    // gpu id, direction (d2h, e.g.), measured bandwidth, expected bandwidth
    DCGM_ERROR_FORMAT_MESSAGE(
        DCGM_FR_LOW_BANDWIDTH,
        d,
        0,
        "dir",
        1.0,
        1.0); // "Bandwidth of GPU %u in direction %s of %.2f did not exceed minimum required bandwidth of %.2f."
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    // gpu id, direction (d2h, e.g.), measured latency, expected latency
    DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_HIGH_LATENCY,
                              d,
                              "lat",
                              0,
                              1.0,
                              1.0); //  "Latency type %s of GPU %u value %.2f exceeded maximum allowed latency of %.2f."
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    // field id
    DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_CANNOT_GET_FIELD_TAG, d, 0); // "Unable to get field information for field id %hu"
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    // field value, field name, gpu id (this message is for fields that should always have a 0 value)
    DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_FIELD_VIOLATION, d, 0, "field", 0); // "Detected %ld %s for GPU %u"
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    // field value, field name, gpu id, allowable threshold
    DCGM_ERROR_FORMAT_MESSAGE(
        DCGM_FR_FIELD_THRESHOLD, d, 0, "field", 0); // "Detected %ld %s for GPU %u which is above the threshold %ld"
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    // field value, field name, gpu id (same as DCGM_FR_FIELD_VIOLATION, but it's a double)
    DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_FIELD_VIOLATION_DBL, d, 1.0, "field", 0); // "Detected %.1f %s for GPU %u"
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    // field value, field name, gpu id, allowable threshold (same as DCGM_FR_FIELD_THRESHOLD, but it's a double)
    DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_FIELD_THRESHOLD_DBL,
                              d,
                              1.0,
                              "field",
                              0,
                              1.0); // "Detected %.1f %s for GPU %u which is above the threshold %.1f"
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    // field name
    DCGM_ERROR_FORMAT_MESSAGE(
        DCGM_FR_UNSUPPORTED_FIELD_TYPE,
        d,
        "field"); // "Field %s is not supported by this API because it is neither an int64 nor a double type."
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    // field name, allowable threshold, observed value, seconds
    DCGM_ERROR_FORMAT_MESSAGE(
        DCGM_FR_FIELD_THRESHOLD_TS,
        d,
        "field",
        0,
        0,
        1.0); // "%s met or exceeded the threshold of %lu per second: %lu at %.1f seconds into the test."
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    // field name, allowable threshold, observed value, seconds (same as DCGM_FR_FIELD_THRESHOLD, but it's a double)
    DCGM_ERROR_FORMAT_MESSAGE(
        DCGM_FR_FIELD_THRESHOLD_TS_DBL,
        d,
        "field",
        1.0,
        1.0,
        1.0); // "%s met or exceeded the threshold of %.1f per second: %.1f at %.1f seconds into the test."
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    // total seconds of violation, gpu id
    DCGM_ERROR_FORMAT_MESSAGE(
        DCGM_FR_THERMAL_VIOLATIONS, d, 1.0, 0); // "There were thermal violations totaling %.1f seconds for GPU %u"
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    // total seconds of violations, first instance, gpu id
    DCGM_ERROR_FORMAT_MESSAGE(
        DCGM_FR_THERMAL_VIOLATIONS_TS,
        d,
        1.0,
        1.0,
        0); // "Thermal violations totaling %.1f seconds started at %.1f seconds into the test for GPU %u"
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    // observed temperature, gpu id, max allowed temperature
    DCGM_ERROR_FORMAT_MESSAGE(
        DCGM_FR_TEMP_VIOLATION,
        d,
        "GPU",
        0,
        0,
        0); // "Temperature %lld of GPU %u exceeded user-specified maximum allowed temperature %lld"
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    // observed HBM temperature, gpu id, max allowed temperature
    DCGM_ERROR_FORMAT_MESSAGE(
        DCGM_FR_TEMP_VIOLATION,
        d,
        "HBM Memory on GPU",
        0,
        0,
        0); // "Temperature %lld of HBM Memory on GPU %u exceeded user-specified maximum allowed temperature %lld"
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    // gpu id, seconds into test, details about clocks event
    DCGM_ERROR_FORMAT_MESSAGE(
        DCGM_FR_CLOCKS_EVENT_VIOLATION, d, 0, 1.0, "error"); // "Clocks event for GPU %u because of "
                                                             // "event starting %.1f seconds into the test. %s"
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    // details about error
    DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_INTERNAL, d, "error"); // "There was an internal error during the test: '%s'"
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    // gpu id, PCIe generation, minimum allowed, parameter to control
    DCGM_ERROR_FORMAT_MESSAGE(
        DCGM_FR_PCIE_GENERATION, d, 0, 0, 0, "param"); // "GPU %u is running at PCI link generation %d, which is below
                                                       // the minimum allowed link generation of %d (parameter '%s')"
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    // gpu id, PCIe width, minimum allowed, parameter to control
    DCGM_ERROR_FORMAT_MESSAGE(
        DCGM_FR_PCIE_WIDTH, d, 0, 0, 0, "param"); // "GPU %u is running at PCI link width %dX, which is below the
                                                  // minimum allowed link generation of %d (parameter '%s')"
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_ABORTED, d); // "Test was aborted early due to user signal"
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    // Test name
    DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_TEST_DISABLED, d, "test"); // "The %s test is skipped for this GPU."
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    // stat name, gpu id
    DCGM_ERROR_FORMAT_MESSAGE(
        DCGM_FR_CANNOT_GET_STAT, d, "stat", 0); // "Unable to generate / collect stat %s for GPU %u"
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    // observed value, minimum allowed, gpu id
    DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_STRESS_LEVEL,
                              d,
                              1.0,
                              1.0,
                              0); // "Max stress level of %.1f did not reach desired stress level of %.1f for GPU %u"
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    // CUDA API name
    DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_CUDA_API, d, "api"); // "Error using CUDA API %s"
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    // count, gpu id
    DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_FAULTY_MEMORY, d, 0, 0); // "Found %d faulty memory elements on GPU %u"
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    // error detail
    DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_CANNOT_SET_WATCHES, d, "error"); // "Unable to add field watches to DCGM: %s"
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    // gpu id
    DCGM_ERROR_FORMAT_MESSAGE(
        DCGM_FR_CUDA_UNBOUND, d, 0); // "Cuda GPU %d is no longer bound to a CUDA context...Aborting"
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    // Test name, gpu id
    DCGM_ERROR_FORMAT_MESSAGE(
        DCGM_FR_ECC_DISABLED, d, "test", 0); // "Skipping test %s because ECC is not enabled on GPU %u"
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    // percentage of memory we tried to allocate, gpu id
    DCGM_ERROR_FORMAT_MESSAGE(
        DCGM_FR_MEMORY_ALLOC, d, "1.0, 0"); // "Couldn't allocate at least %.1f%% of GPU memory on GPU %u"
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    // gpu id
    DCGM_ERROR_FORMAT_MESSAGE(
        DCGM_FR_CUDA_DBE, d, 0); // "CUDA APIs have indicated that a double-bit ECC error has occured on GPU %u."
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    // gpu id
    DCGM_ERROR_FORMAT_MESSAGE(
        DCGM_FR_MEMORY_MISMATCH,
        d,
        0); // "A memory mismatch was detected on GPU %u, but no error was reported by CUDA or NVML."
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    // gpu id, error detail
    DCGM_ERROR_FORMAT_MESSAGE(
        DCGM_FR_CUDA_DEVICE, d, 0, "error"); //     "Unable to find a corresponding CUDA device for GPU %u: '%s'"
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_ECC_UNSUPPORTED,
                              d); // "ECC Memory is not turned on or is unsupported. Skipping test."
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    // gpu id
    DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_ECC_PENDING, d, 0); // "ECC memory for GPU %u is in a pending state."
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    // gpu id, observed bandwidth, required, test name
    DCGM_ERROR_FORMAT_MESSAGE(
        DCGM_FR_MEMORY_BANDWIDTH,
        d,
        0,
        1.0,
        1.0,
        0); // "GPU %u only achieved a memory bandwidth of %.2f GB/s, failing to meet %.2f GB/s for test %d"
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    // power draw observed, field tag, minimum power draw required, gpu id
    DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_TARGET_POWER,
                              d,
                              1.0,
                              "field",
                              1.0,
                              0); // "Max power of %.1f did not reach desired power minimum %s of %.1f for GPU %u"
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    // API name, error detail
    DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_API_FAIL, d, "api", "error"); // "API call %s failed: '%s'"
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    // API name, gpu id, error detail
    DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_API_FAIL_GPU, d, "api", 0, "error"); // "API call %s failed for GPU %u: '%s'"
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    // gpu id, error detail
    DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_CUDA_CONTEXT, d, 0, "error"); // "GPU %u failed to create a CUDA context: %s"
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    // DCGM API name
    DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_DCGM_API, d, "api"); // "Error using DCGM API %s"
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_CONCURRENT_GPUS,
                              d); // "Unable to run concurrent pair bandwidth test without 2 or more gpus. Skipping"
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_TOO_MANY_ERRORS,
                              d); // "This API can only return up to four errors per system. Additional errors were
                                  // found for this system that couldn't be communicated."
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    // error count, gpu id
    DCGM_ERROR_FORMAT_MESSAGE(
        DCGM_FR_NVLINK_CRC_ERROR_THRESHOLD,
        d,
        1.0,
        0); // "%.1f %s NvLink errors found occuring per second on GPU %u, exceeding the limit of 100 per second."
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    // error count, field name, gpu id
    DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_NVLINK_ERROR_CRITICAL,
                              d,
                              0,
                              "field",
                              0); // "Detected %ld %s NvLink errors on GPU %u's NVLink (should be 0)"
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    // gpu id, power limit, power reached
    DCGM_ERROR_FORMAT_MESSAGE(
        DCGM_FR_ENFORCED_POWER_LIMIT,
        d,
        0,
        1.0,
        1.0); // "Enforced power limit on GPU %u set to %.1f, which is too low to attempt to achieve target power %.1f"
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    // memory
    DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_MEMORY_ALLOC_HOST, d, 0); // "Cannot allocate %zu bytes on the host"
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_GPU_OP_MODE,
                              d); // "Skipping plugin due to a GPU being in GPU Operating Mode: LOW_DP."
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    // clock, count
    DCGM_ERROR_FORMAT_MESSAGE(
        DCGM_FR_NO_MEMORY_CLOCKS, d, 0, 0); //  "No memory clocks <= %u MHZ were found in %u supported memory clocks."
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    // clock, count, clock
    DCGM_ERROR_FORMAT_MESSAGE(
        DCGM_FR_NO_GRAPHICS_CLOCKS,
        d,
        0,
        0,
        0); // "No graphics clocks <= %u MHZ were found in %u supported graphics clocks for memory clock %u MHZ."
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    DCGM_ERROR_FORMAT_MESSAGE(
        DCGM_FR_HAD_TO_RESTORE_STATE, d, "error"); // "Had to restore GPU state on NVML GPU(s): %s"
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_L1TAG_UNSUPPORTED,
                              d); // "This card does not support the L1 cache test. Skipping test."
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_L1TAG_MISCOMPARE, d); // "Detected a miscompare failure in the L1 cache."
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_ROW_REMAP_FAILURE,
                              d); // "GPU %u had uncorrectable memory errors and row remapping failed."
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_UNCONTAINED_ERROR, d); // "GPU had an uncontained error (XID 95)"
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_EMPTY_GPU_LIST, d); // "No valid GPUs passed to plugin"
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    // gpu id
    DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_DBE_PENDING_PAGE_RETIREMENTS,
                              d,
                              0); // "Pending page retirements together with a DBE were detected on GPU %u."
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    // gpu id, rows remapped
    DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_UNCORRECTABLE_ROW_REMAP,
                              d,
                              0,
                              0); // "GPU %u had uncorrectable memory errors and %u rows were remapped"
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    // gpu id
    DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_PENDING_ROW_REMAP,
                              d,
                              0); // "GPU %u has uncorrectable memory errors and row remappings are pending"
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    // gpu id, test name
    DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_BROKEN_P2P_MEMORY_DEVICE,
                              d,
                              0,
                              0,
                              "test"); // "GPU %u was unsuccessfully written to by %u in a peer-to-peer test: %s"
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    // gpu id, test name
    DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_BROKEN_P2P_WRITER_DEVICE,
                              d,
                              0,
                              0,
                              "test"); // "GPU %u unsuccessfully wrote data to %u in a peer-to-peer test: %s"
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_NVSWITCH_NVLINK_DOWN, d, 0, 0); //         "NVSwitch %u's NvLink %u is down."
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_EUD_BINARY_PERMISSIONS, d); //       "" /* See message inplace */
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_EUD_NON_ROOT_USER, d); //            "" /* See message inplace */
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_EUD_SPAWN_FAILURE, d); //            "" /* See message inplace */
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_EUD_TIMEOUT, d); //                  "" /* See message inplace */
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_EUD_ZOMBIE, d); //                   "" /* See message inplace */
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_EUD_NON_ZERO_EXIT_CODE, d); //       "" /* See message inplace */
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_EUD_TEST_FAILED, d); //              "" /* See message inplace */
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    // directory
    DCGM_ERROR_FORMAT_MESSAGE(
        DCGM_FR_FILE_CREATE_PERMISSIONS,
        d,
        "dir"); // "The DCGM Diagnostic does not have permissions to create a file in directory '%s'"
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_PAUSE_RESUME_FAILED, d); // "" /* See message inplace */
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    // gpu id
    DCGM_ERROR_FORMAT_MESSAGE(
        DCGM_FR_PCIE_H_REPLAY_VIOLATION, d, 0); // "GPU %u had PCIe replays, see dmesg for more information"
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    // xid error, gpu id
    DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_XID_ERROR, d, 0, 0); // "Detected XID %u for GPU %u"
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    // count, field, gpu id
    DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_SBE_VIOLATION, d, 0, "field", 0); // "Detected %ld %s for GPU %u"
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    // count, field, gpu id
    DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_DBE_VIOLATION, d, 0, "field", 0); // "Detected %ld %s for GPU %u"
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    // count, field, gpu id
    DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_PCIE_REPLAY_VIOLATION, d, 0, "field", 0); // "Detected %ld %s for GPU %u"
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    // count, field, gpu id, threshold
    DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_SBE_THRESHOLD_VIOLATION,
                              d,
                              0,
                              "field",
                              0,
                              0); // "Detected %ld %s for GPU %u which is above the threshold %ld"
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    // count, field, gpu id, threshold
    DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_DBE_THRESHOLD_VIOLATION,
                              d,
                              0,
                              "field",
                              0,
                              0); // "Detected %ld %s for GPU %u which is above the threshold %ld"
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_PCIE_REPLAY_THRESHOLD_VIOLATION,
                              d,
                              0,
                              "field",
                              0,
                              0); // "Detected %ld %s for GPU %u which is above the threshold %ld"
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_CUDA_FM_NOT_INITIALIZED, d);
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_SXID_ERROR, d, 1);
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_GFLOPS_THRESHOLD_VIOLATION, d, 100.0, "GFLOPs", 0, 65.0);
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
    std::string_view expected_msg(
        "Detected 100.00 GFLOPs for GPU 0 which is below the threshold 65.00 " DCGM_FR_GFLOPS_THRESHOLD_VIOLATION_NEXT);
    CHECK(d.GetMessage() == expected_msg);
    // count, gpu id
    DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_NAN_VALUE, d, 0, 0); // "Found %ld NaN-value memory elements on GPU %u"
    WARN(d.GetMessage());
    CHECK(d.GetMessage().length() < DCGM_ERR_MSG_LENGTH);
}
