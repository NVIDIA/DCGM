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

#include "DcgmTaskRunner.h"
#include "DcgmLogging.h"
#include <stdexcept>

DcgmTaskRunner::~DcgmTaskRunner()
{
    /* Wait for the worker. DcgmThread::StopAndWait() will call DcgmThread::Stop(),
       which will trigger OnStop() to wake up our thread */
    try
    {
        StopAndWait(60000);
    }
    catch (const std::exception &e)
    {
        DCGM_LOG_ERROR << "DcgmTaskRunner::~DcgmTaskRunner(): " << e.what();
    }
    catch (...)
    {
        DCGM_LOG_ERROR << "DcgmTaskRunner::~DcgmTaskRunner(): Unknown exception";
    }
}

void DcgmTaskRunner::OnStop()
{
    DcgmNs::TaskRunner::Stop();
}

void DcgmTaskRunner::run()
{
    using DcgmNs::TaskRunner;
    while (ShouldStop() == 0)
    {
        if (TaskRunner::Run() != TaskRunner::RunResult::Ok)
        {
            break;
        }
    }
}
