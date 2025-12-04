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

#include "CudaWorkerThread.hpp"
#include <cuda.h>         // For CUdevice
#include <dcgm_structs.h> // For dcgmReturn_t

/**
 * Stub implementation of CudaWorkerThread for unit testing
 * This provides minimal functionality without real CUDA dependencies
 */

CudaWorkerThread::CudaWorkerThread()
{
    // Minimal stub - do nothing
}

CudaWorkerThread::~CudaWorkerThread()
{
    // Minimal stub - do nothing
}

void CudaWorkerThread::Shutdown()
{
    // Minimal stub - do nothing
}

void CudaWorkerThread::SetWorkerToIdle()
{
    // Minimal stub - do nothing
}

void CudaWorkerThread::SetWorkloadAndTarget(unsigned int /* fieldId */,
                                            double /* loadTarget */,
                                            bool /* blockOnCompletion */,
                                            bool /* preferCublas */)
{
    // Minimal stub - do nothing
}

double CudaWorkerThread::GetCurrentAchievedLoad()
{
    return 0.0; // Return default value for testing
}

dcgmReturn_t CudaWorkerThread::Init(CUdevice /* device */)
{
    // Minimal stub - return success
    return DCGM_ST_OK;
}

void CudaWorkerThread::SetPeerByBusId(std::string /* peerBusId */)
{
    // Minimal stub - do nothing
}

void CudaWorkerThread::run()
{
    // Minimal stub - do nothing (virtual method from base class)
}
