/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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

//! The payload for the SDC binary.

use serde::{Deserialize, Serialize};

/// The test result.
///
/// * `return_code` - The return code of the test.
/// * `test_id` - Human-readable test name.
/// * `test_num` - The test number in the test suite.
/// * `test_suite_len` - The total number of tests in the test suite.
/// * `test_status` - The status of the test. One of pass, fail, running.
/// * `test_details` - Description of the `return_code`. 'OK' if the test passed.
/// * `device_type` - The type of the device. Should be 'cpu' for CPU SDC tests.
/// * `device_ids` - The ids of the devices.
///
#[allow(dead_code)]
#[derive(Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct TestResult {
    /// The return code reported by the test.
    pub(crate) return_code: i32,
    /// The human-readable name of the test.
    pub(crate) test_id: String,
    /// The test's number within the test suite.
    pub(crate) test_num: u32,
    /// The total number of tests in the test suite.
    pub(crate) test_suite_len: u32,
    /// The test status, such as `pass`, `fail`, or `running`.
    pub(crate) test_status: String,
    /// A description of the return code, or `OK` when the test passes.
    pub(crate) test_details: String,
    /// The device type tested, which is expected to be `cpu`.
    pub(crate) device_type: String,
    /// The identifiers of the devices included in the test.
    pub(crate) device_ids: Vec<String>,
}

/// The payload for the test result.
///
/// * `test_result` - The test result.
///
#[derive(Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct TestResultPayload {
    /// The test result carried by this payload.
    pub(crate) test_result: TestResult,
}

/// The version of the SDC binary.
///
/// * `version` - The version of the SDC binary.
///
#[derive(Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct CpuSdcVersion {
    /// The CPU SDC binary version.
    pub(crate) version: String,
}

/// The payload for the version of the SDC binary.
///
/// * `sdc` - The version of the SDC binary.
///
#[derive(Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct VersionPayload {
    /// The CPU SDC version carried by this payload.
    pub(crate) sdc: CpuSdcVersion,
}

/// The possible payloads for the SDC binary.
///
/// * `version` - The version of the SDC binary.
/// * `test_result` - The test result.
///
#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum ProtocolMessage {
    Version(VersionPayload),
    TestResult(TestResultPayload),
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    const TEST_SDC_VERSION: &str = "1.2.3";
    const TEST_ID_CPU_SDC_SHORT: &str = "cpu_sdc_short";
    const TEST_STATUS_FAIL: &str = "fail";
    const TEST_DETAILS_MISMATCH: &str = "mismatch detected";
    const TEST_DEVICE_TYPE_CPU: &str = "cpu";
    const TEST_DEVICE_ID_0: &str = "0";
    const TEST_DEVICE_ID_1: &str = "1";

    #[test]
    /// Verifies deserialization of an SDC version message.
    /// Input: none. Output: `()` when deserialization assertions pass.
    fn deserializes_version_payload() {
        let payload: ProtocolMessage = serde_json::from_value(serde_json::json!({
            "sdc": {
                "version": TEST_SDC_VERSION,
            },
        }))
        .expect("version payload should deserialize");

        assert_eq!(
            payload,
            ProtocolMessage::Version(VersionPayload {
                sdc: CpuSdcVersion {
                    version: String::from(TEST_SDC_VERSION)
                }
            })
        );
    }

    #[test]
    /// Verifies deserialization of an SDC test-result message.
    /// Input: none. Output: `()` when deserialization assertions pass.
    fn deserializes_test_result_payload() {
        let payload: ProtocolMessage = serde_json::from_value(serde_json::json!({
            "test_result": {
                "return_code": 7,
                "test_id": TEST_ID_CPU_SDC_SHORT,
                "test_num": 1,
                "test_suite_len": 2,
                "test_status": TEST_STATUS_FAIL,
                "test_details": TEST_DETAILS_MISMATCH,
                "device_type": TEST_DEVICE_TYPE_CPU,
                "device_ids": [TEST_DEVICE_ID_0, TEST_DEVICE_ID_1],
            },
        }))
        .expect("test result payload should deserialize");

        assert_eq!(
            payload,
            ProtocolMessage::TestResult(TestResultPayload {
                test_result: TestResult {
                    return_code: 7,
                    test_id: String::from(TEST_ID_CPU_SDC_SHORT),
                    test_num: 1,
                    test_suite_len: 2,
                    test_status: String::from(TEST_STATUS_FAIL),
                    test_details: String::from(TEST_DETAILS_MISMATCH),
                    device_type: String::from(TEST_DEVICE_TYPE_CPU),
                    device_ids: vec![String::from(TEST_DEVICE_ID_0), String::from(TEST_DEVICE_ID_1)],
                }
            })
        );
    }

    #[test]
    /// Verifies that payloads with an unknown shape are rejected.
    /// Input: none. Output: `()` when rejection assertions pass.
    fn rejects_unknown_payload_shape() {
        let err = serde_json::from_value::<ProtocolMessage>(serde_json::json!({ "unexpected": true }))
            .expect_err("unknown payload should be rejected");

        assert!(err.is_data());
    }

    #[test]
    /// Verifies that a payload cannot contain both supported message shapes.
    /// Input: none. Output: `()` when rejection assertions pass.
    fn rejects_payload_with_both_version_and_test_result_shapes() {
        let err = serde_json::from_value::<ProtocolMessage>(serde_json::json!({
            "sdc": {
                "version": TEST_SDC_VERSION,
            },
            "test_result": {
                "return_code": 7,
                "test_id": TEST_ID_CPU_SDC_SHORT,
                "test_num": 1,
                "test_suite_len": 2,
                "test_status": TEST_STATUS_FAIL,
                "test_details": TEST_DETAILS_MISMATCH,
                "device_type": TEST_DEVICE_TYPE_CPU,
                "device_ids": [TEST_DEVICE_ID_0, TEST_DEVICE_ID_1],
            },
        }))
        .expect_err("ambiguous mixed-shape payload should be rejected");

        assert!(err.is_data());
    }
}
