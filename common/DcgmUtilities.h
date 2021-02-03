/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
#ifndef DCGM_UTILITIES_H
#define DCGM_UTILITIES_H

#include <string>
#include <vector>

/*************************************************************************/
/*************************************************************************
 * Utility methods for DCGM
 *************************************************************************/

/*************************************************************************/
/*
 * Creates a child process and executes the command given by args where args is an argv style array of strings.
 *
 * @param args: (IN) argv style args given as a vector of strings. The program to execute is the first element of the
 * vector.
 * @param infp: (OUT) pointer to a file descriptor for the child process's STDIN. Pass in NULL to ignore STDIN.
 * @param outfp: (OUT) pointer to a file descriptor for the child process's STDOUT. Cannot be NULL.
 * @param errfp: (OUT) pointer to a file descriptor for the child process's STDERR. Pass in NULL to ignore STDERR.
 * @param stderrToStdout: (IN) if true, child's stderr will be redirected to outfp and errfp will be ignored.
 *
 * @return: pid of the child process.
 *          If the returned pid is < 0, then there was an error creating the child process or outfp was NULL.
 *          Errors are logged using the PRINT_ERROR macro.
 *
 * Notes:
 * - It is caller's reponsibility to close the file descriptors associated with infp, outfp, errfp (if non-null
 *   values are given). If you want to run a process and ignore its output, redirect the descriptor to /dev/null.
 *
 * - Caller is responsible for waiting for child to exit (if desired).
 */
pid_t DcgmUtilForkAndExecCommand(std::vector<std::string> &args,
                                 int *infp,
                                 int *outfp,
                                 int *errfp,
                                 bool stderrToStdout);

#endif // DCGM_UTILITIES_H
