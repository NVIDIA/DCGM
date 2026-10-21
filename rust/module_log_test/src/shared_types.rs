/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

#![allow(dead_code, reason = "testing code")]

pub type LoggerCallbackFp = unsafe extern "C" fn(arg1: *const Record);
#[allow(non_camel_case_types, reason = "mimicking nvvs plugin code")]
pub type hostEngineAppenderCallbackFp_t = ::std::option::Option<LoggerCallbackFp>;

#[repr(C)]
#[derive(Debug, Copy, Clone, Hash)]
pub struct Record {
    pub message: *const ::std::os::raw::c_char,
    pub func: *const ::std::os::raw::c_char,
    pub file: *const ::std::os::raw::c_char,
    pub object: *const ::std::os::raw::c_void,
    pub line: usize,
    pub tid: ::std::os::raw::c_uint,
    pub severity: u32,
    pub time: usize,
}
