/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
#pragma once

#include "TaskRunner.hpp"

#include "DcgmMutex.h"
#include "DcgmThread.h"
#include "dcgm_structs.h"
#include <condition_variable>
#include <queue>

class DcgmTaskRunner
    : public DcgmThread
    , public DcgmNs::TaskRunner
{
public:
    ~DcgmTaskRunner() override;

    /** Virtual method inherited from DcgmThread that is the main() for this thread */
    void run() override;

    /**
     * Virtual method of DcgmThread that is called when DcgmThread::Stop() is
     * called. We use this to signal m_condition to wake up our worker so it
     * will shut down.
     */
    void OnStop() override;
};
