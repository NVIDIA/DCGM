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
#ifndef DCGM_GETNTHMAPSTOKEN_H
#define DCGM_GETNTHMAPSTOKEN_H


#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief This function retrieves the Nth token from a line of /proc/<pid>/maps.
 * @param[in,out] line The original line to split.
 *                     The memory that this argument points to will be modified by the underlying strtok.
 * @param[in] n The required column/token number. This starts from 1.
 * @return Pointer to the Nth column/token or \c NULL if not found.
 */
char *GetNthMapsToken(char *line, int n) __attribute__((visibility("hidden")));

#ifdef __cplusplus
} //extern "C"
#endif


#endif //DCGM_GETNTHMAPSTOKEN_H
