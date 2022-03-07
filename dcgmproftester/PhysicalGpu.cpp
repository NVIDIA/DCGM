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
#include "PhysicalGpu.h"
#include "Arguments.h"
#include "DcgmCacheManager.h"
#include "DcgmLogging.h"
#include "DcgmProfTester.h"
#include "DcgmSettings.h"
#include "Entity.h"
#include "Reporter.h"
#include "dcgm_fields.h"
#include "dcgm_fields_internal.h"
#include "timelib.h"
#include "vector_types.h"
#include <cublas_proxy.hpp>
#include <cuda.h>
#include <dcgm_agent.h>

#if (CUDA_VERSION_USED >= 11)
#include "DcgmDgemm.hpp"
#endif

#include <tclap/Arg.h>
#include <tclap/CmdLine.h>
#include <tclap/SwitchArg.h>
#include <tclap/ValueArg.h>
#include <tclap/ValuesConstraint.h>

#include <chrono>
#include <csignal>
#include <cstdint>
#include <iterator>
#include <map>
#include <memory>
#include <streambuf>
#include <string>
#include <sys/types.h>
#include <system_error>
#include <unistd.h>
#include <vector>

namespace DcgmNs::ProfTester
{
static const std::streamsize MaxStreamLength { std::numeric_limits<std::streamsize>::max() };

// Dump field Values returned.
void PhysicalGpu::ValuesDump(std::map<Entity, dcgmFieldValue_v1> &values, ValueType type, double divisor)
{
    if (IsMIG())
    {
        info_reporter << "M:";
    }

    if (values.size() > 1)
    {
        info_reporter << "{ ";
    }

    bool first { true };

    for (auto &[entity, fieldValue] : values)
    {
        if (values.size() > 1)
        {
            if (!first)
            {
                info_reporter << ", ";
            }
            else
            {
                first = false;
            }

            switch (entity.m_entity.entityGroupId)
            {
                default:
                case DCGM_FE_NONE:
                    info_reporter << "?: ";
                    break;

                case DCGM_FE_GPU:
                    info_reporter << "GPU: ";
                    break;

                case DCGM_FE_VGPU:
                    info_reporter << "VGPU: ";
                    break;

                case DCGM_FE_SWITCH:
                    info_reporter << "SW: ";
                    break;

                case DCGM_FE_GPU_I:
                    info_reporter << "GI: ";
                    break;

                case DCGM_FE_GPU_CI:
                    info_reporter << "CI: ";
                    break;
            }
        }

        double value;

        switch (type)
        {
            case PhysicalGpu::ValueType::Int64:
                value = fieldValue.value.i64;
                value /= divisor;
                info_reporter << value;

                break;

            case PhysicalGpu::ValueType::Double:
                value = fieldValue.value.dbl;
                value /= divisor;
                info_reporter << value;

                break;

            case PhysicalGpu::ValueType::String:
                info_reporter << "<string>";

                break;

            case PhysicalGpu::ValueType::Blob:
                info_reporter << "<blob>";

                break;

            default:
                info_reporter << "<unknown>";

                break;
        }
    }

    if (values.size() > 1)
    {
        info_reporter << " }";
    }
}

static dcgmReturn_t DefaultTickHandler(size_t index,
                                       bool valid,
                                       std::map<Entity, dcgmFieldValue_v1> &values,
                                       DistributedCudaContext &worker);

/*****************************************************************************/
/* ctor/dtor */
PhysicalGpu::PhysicalGpu(std::shared_ptr<DcgmProfTester> tester,
                         unsigned int gpuId,
                         const Arguments_t::Parameters &parameters)
    : m_tester(tester)
    , m_gpuId(gpuId)
    , m_responseFn(nullptr)
{
    memset(&m_dcgmDeviceAttr, 0, sizeof(m_dcgmDeviceAttr));
    SetParameters(parameters);
}

/*****************************************************************************/
PhysicalGpu::~PhysicalGpu()
{
    DestroyDcgmGroups();
}

/*****************************************************************************/
void PhysicalGpu::SetParameters(const Arguments_t::Parameters &parameters)
{
    m_parameters = parameters;
    SetTickHandler(DefaultTickHandler);
}

/*****************************************************************************/
dcgmReturn_t PhysicalGpu::CheckVirtualizationMode(void)
{
    dcgmGroupEntityPair_t entity { DCGM_FE_GPU, m_gpuId };
    unsigned short fieldId { DCGM_FI_DEV_VIRTUAL_MODE };
    dcgmFieldValue_v2 value {};

    dcgmReturn_t dcgmReturn
        = dcgmEntitiesGetLatestValues(m_dcgmHandle, &entity, 1, &fieldId, 1, DCGM_FV_FLAG_LIVE_DATA, &value);
    if (dcgmReturn != DCGM_ST_OK || value.status != DCGM_ST_OK)
    {
        warn_reporter << "Skipping GPU virtualization mode check due to nonzero dcgmReturn " << dcgmReturn
                      << " or valueStatus " << value.status
                      << ". Perfworks may still return an error if the vGPU mode is not supported."
                      << warn_reporter.new_line;

        return DCGM_ST_OK;
    }

    if (value.value.i64 != DCGM_GPU_VIRTUALIZATION_MODE_NONE
        && value.value.i64 != DCGM_GPU_VIRTUALIZATION_MODE_PASSTHROUGH)
    {
        error_reporter << "Virtualization mode " << value.value.i64 << " is unsupported." << error_reporter.new_line;

        return DCGM_ST_PROFILING_NOT_SUPPORTED;
    }

    return DCGM_ST_OK;
}


/*****************************************************************************/
dcgmReturn_t PhysicalGpu::Init(dcgmHandle_t dcgmHandle)
{
    dcgmReturn_t dcgmReturn = DCGM_ST_OK;

    m_dcgmHandle             = dcgmHandle;
    m_dcgmDeviceAttr.version = dcgmDeviceAttributes_version;
    dcgmReturn               = dcgmGetDeviceAttributes(m_dcgmHandle, m_gpuId, &m_dcgmDeviceAttr);
    if (dcgmReturn != DCGM_ST_OK)
    {
        error_reporter << "dcgmGetDeviceAttributes() returned " << dcgmReturn << "." << error_reporter.new_line;

        return dcgmReturn;
    }

    /* See if the virtualization mode of the GPU is supported */
    dcgmReturn = CheckVirtualizationMode();
    if (dcgmReturn != DCGM_ST_OK)
    {
        /* Logged by the call */
        return dcgmReturn;
    }

    return dcgmReturn;
}

std::shared_ptr<DistributedCudaContext> PhysicalGpu::AddSlice(
    std::shared_ptr<std::map<dcgm_field_entity_group_t, dcgm_field_eid_t>> entities,
    dcgmGroupEntityPair_t &entity,
    const std::string &deviceId)

{
    std::shared_ptr<DistributedCudaContext> foundCudaContext { nullptr };

    for (auto cudaContext : m_dcgmCudaContexts)
    {
        if (cudaContext->EntityId().entityGroupId != entity.entityGroupId)
        {
            continue;
        }

        if (cudaContext->EntityId().entityId != entity.entityId)
        {
            continue;
        }

        // We found the worker slice!
        foundCudaContext = cudaContext;

        break;
    }

    if (foundCudaContext == nullptr)
    {
        foundCudaContext = std::make_shared<DistributedCudaContext>(shared_from_this(), entities, entity, deviceId);

        /**
         * When we  get the first worker, we get the compute capabilities of
         * its GPU as it will be the same for all MIG slices, and the only
         * workers for a GPU will be its own MIG slices.
         */
        if (m_workers == 0)
        {
            dcgmGroupEntityPair_t entity;
            unsigned short fieldId { DCGM_FI_DEV_CUDA_COMPUTE_CAPABILITY };
            dcgmFieldValue_v2 value {};

            entity.entityGroupId = DCGM_FE_GPU;
            entity.entityId      = (*entities)[DCGM_FE_GPU];

            dcgmReturn_t dcgmReturn
                = dcgmEntitiesGetLatestValues(m_dcgmHandle, &entity, 1, &fieldId, 1, DCGM_FV_FLAG_LIVE_DATA, &value);

            if (dcgmReturn != DCGM_ST_OK)
            {
                /**
                 * foundCudaContext will be smart pointer garbage collected.
                 */

                return nullptr; // Failure
            }

            m_majorComputeCapability = DCGM_CUDA_COMPUTE_CAPABILITY_MAJOR(value.value.i64) >> 16;
            m_minorComputeCapability = DCGM_CUDA_COMPUTE_CAPABILITY_MINOR(value.value.i64);
        }

        /**
         * We count the worker slices separately from the worker vector size
         * since the worker vector can be cleared when all work and reporting
         * is complete or if an error occurs.
         */
        m_dcgmCudaContexts.push_back(foundCudaContext);
        m_workers++;
    }
    else
    {
        foundCudaContext->ReInitialize(entities, deviceId);
    }

    foundCudaContext->SetTries(m_parameters.m_syncCount);

    return foundCudaContext;
}

bool PhysicalGpu::ValueGet(std::map<Entity, dcgmFieldValue_v1> &values,
                           std::map<dcgm_field_entity_group_t, dcgm_field_eid_t> &entities,
                           dcgm_field_entity_group_t entityGroupId,
                           ValueType type,
                           double divisor,
                           double &value)
{
    dcgm_field_eid_t entityId;

    if (IsMIG())
    {
        if (entities.find(entityGroupId) == entities.end())
        {
            return false;
        }

        entityId = entities[entityGroupId];
    }
    else
    {
        entityGroupId = DCGM_FE_GPU;
        entityId      = m_gpuId;
    }

    Entity entity(entityGroupId, entityId);

    if (values.find(entity) == values.end())
    {
        return false;
    }

    switch (type)
    {
        case ValueType::Int64:
            value = values[entity].value.i64;
            value /= divisor;

            break;

        case ValueType::Double:
            value = values[entity].value.dbl;
            value /= divisor;

            break;

        case ValueType::String:
        case ValueType::Blob:
        default:
            return false;
    }

    return true;
}

// Abort child processes. Something went very wrong.
void PhysicalGpu::AbortChildren(unsigned int workers)
{
    if (m_dcgmCudaContexts.empty())
    {
        /**
         * We can get called here if there never were any workers, i.e. a MIG
         * configuration on a GPU with no actual MIG slices.
         */

        return;
    }

    if (workers == 0)
    {
        workers = m_workers;
    }

    for (unsigned int i = 0; i < workers; i++)
    {
        /*
         * Resetting a DistributedCudaContext on the parent process side will
         * close the pipe file descriptors. When the child process tries
         * to write or read it will get an error, and exit. Since we ignore
         * SIGCHLD, there will be no zombies left. This also closes the file
         * descriptors on the parent side, so any subsequent selects will
         * fail.
         *
         * It is important that we report the worker as failed BEFORE we
         * reset it as resetting it clears it's file descriptors and the
         * tester might wish to interrogate it as to WHICH read (from the
         * tester's side) file descriptor to ignore (or for other information).
         */

        m_tester->ReportWorkerFailed(m_dcgmCudaContexts[i]);
        m_dcgmCudaContexts[i]->Reset();
        m_reportedWorkers++; // We count this one as reported, as it is caput.
    }

    DestroyDcgmGroups();
    m_dcgmCudaContexts.clear();
    m_workers    = 0;
    m_lastWorker = nullptr;
}

/* Nuke child processes. This tries to be gentle before aborting it. It
 * should be used if the children have already started processing a request
 * as opposed to not. Generally Aborting the child will cause it to die
 * from SIGPIPE when it can't do I/O on it's pipes, sending an "X" will let
 * it try to exit cleanly if it is done working and waiting for another
 * command.
 */
void PhysicalGpu::NukeChildren(bool force)
{
    for (auto &worker : m_dcgmCudaContexts)
    {
        worker->Command("X\n"); // tell them nicely...
    }

    if (force)
    {
        AbortChildren(); // ... but firmly.
    }

    m_dcgmCudaContexts.clear();
    m_workers    = 0;
    m_lastWorker = nullptr;
}

static dcgmReturn_t DefaultTickHandler(size_t /*index*/,
                                       bool /*valid*/,
                                       std::map<Entity, dcgmFieldValue_v1> & /*values*/,
                                       DistributedCudaContext &worker)
{
    /**
     * Skip to end of notification line. By default, the worker should not
     * send more than one like of Tick notification data.
     */

    worker.Input().ignore(MaxStreamLength, '\n');

    return DCGM_ST_OK;
}


// Set per-subtest tick handler.
void PhysicalGpu::SetTickHandler(TickHandlerType tickHandler)
{
    m_tickHandler = std::move(tickHandler);
}

// Validate measured value within valid range or against expected value.
bool PhysicalGpu::Validate(double expected, double current, double measured, double howFarIn, bool prevValidate)
{
    if (!m_parameters.m_validate)
    {
        return true;
    }

    if ((howFarIn < (m_parameters.m_waitToCheck / 100.0)) && !m_parameters.m_fast)
    {
        return true;
    }

    if (m_parameters.m_valueValid)
    {
        if (measured < m_parameters.m_minValue)
        {
            if (!m_parameters.m_fast)
            {
                warn_reporter << "Field " << m_parameters.m_fieldId << " @ " << howFarIn * 100
                              << "% Validation Retry: " << m_parameters.m_minValue << " !< " << measured << " < "
                              << m_parameters.m_maxValue << "." << warn_reporter.new_line;
            }

            return false;
        }

        if (measured > m_parameters.m_maxValue)
        {
            if (!m_parameters.m_fast)
            {
                warn_reporter << "Field " << m_parameters.m_fieldId << " @ " << howFarIn * 100
                              << "% Validation Retry: " << m_parameters.m_minValue << " !< " << measured << " < "
                              << m_parameters.m_maxValue << "." << warn_reporter.new_line;
            }

            return false;
        }

        if (!prevValidate)
        {
            warn_reporter << "Field " << m_parameters.m_fieldId << " @ " << howFarIn * 100
                          << "% Validation Pass: " << m_parameters.m_minValue << " < " << measured << " < "
                          << m_parameters.m_maxValue << "." << warn_reporter.new_line;
        }

        return true;
    }

    double lowExpected  = std::min(expected, current);
    double highExpected = std::max(expected, current);

    if (m_parameters.m_percentTolerance)
    {
        if (measured < lowExpected * (1.0 - m_parameters.m_tolerance / 100.0))
        {
            if (!m_parameters.m_fast)
            {
                warn_reporter << "Field " << m_parameters.m_fieldId << " @ " << howFarIn * 100
                              << "% Validation Retry: " << lowExpected * (1.0 - m_parameters.m_tolerance / 100.0)
                              << " !< " << measured << " < " << highExpected * (1.0 + m_parameters.m_tolerance / 100.0)
                              << "." << warn_reporter.new_line;
            }

            return false;
        }

        if (measured > highExpected * (1.0 + m_parameters.m_tolerance / 100.0))
        {
            if (!m_parameters.m_fast)
            {
                warn_reporter << "Field " << m_parameters.m_fieldId << " @ " << howFarIn * 100
                              << "% Validation Retry: " << lowExpected * (1.0 - m_parameters.m_tolerance / 100.0)
                              << " < " << measured << " !< " << highExpected * (1.0 + m_parameters.m_tolerance / 100.0)
                              << "." << warn_reporter.new_line;
            }

            return false;
        }

        if (!prevValidate)
        {
            warn_reporter << "Field " << m_parameters.m_fieldId << " @ " << howFarIn * 100
                          << "% Validation Pass: " << lowExpected * (1.0 - m_parameters.m_tolerance / 100.0) << " < "
                          << measured << " < " << highExpected * (1.0 + m_parameters.m_tolerance / 100.0) << "."
                          << warn_reporter.new_line;
        }

        return true;
    }

    if (measured < (lowExpected - m_parameters.m_tolerance))
    {
        if (!m_parameters.m_fast)
        {
            warn_reporter << "Field " << m_parameters.m_fieldId << " @ " << howFarIn * 100
                          << "% Validation Retry: " << lowExpected - m_parameters.m_tolerance << "!< " << measured
                          << " < " << highExpected + m_parameters.m_tolerance << "." << warn_reporter.new_line;
        }

        return false;
    }

    if (measured > (highExpected + m_parameters.m_tolerance))
    {
        if (!m_parameters.m_fast)
        {
            warn_reporter << "Field " << m_parameters.m_fieldId << " @ " << howFarIn * 100
                          << "% Validation Retry: " << lowExpected - m_parameters.m_tolerance << "< " << measured
                          << " !< " << highExpected + m_parameters.m_tolerance << "." << warn_reporter.new_line;
        }

        return false;
    }

    if (!prevValidate)
    {
        warn_reporter << "Field " << m_parameters.m_fieldId << " @ " << howFarIn * 100
                      << "% Validation Pass: " << lowExpected - m_parameters.m_tolerance << "< " << measured << " < "
                      << highExpected + m_parameters.m_tolerance << "." << warn_reporter.new_line;
    }

    return true;
}

// Send a command to all child CUDA worker processes.
dcgmReturn_t PhysicalGpu::CommandAll(bool all, const char *format, ...)
{
    std::va_list args;

    for (auto &worker : m_dcgmCudaContexts)
    {
        worker->ClrFailed();

        va_start(args, format);
        int err = worker->Command(format, args);
        va_end(args);

        if (err < 0)
        {
            NukeChildren(true);

            return DCGM_ST_GENERIC_ERROR;
        }
    }

    return DCGM_ST_OK;
}


// Start tests running.
dcgmReturn_t PhysicalGpu::StartTests(void)
{
    m_responseFn      = &PhysicalGpu::ProcessStartingResponse;
    m_startedWorkers  = 0;
    m_finishedWorkers = 0;
    m_reportedWorkers = 0;

    size_t workers { 0 };

    for (auto &worker : m_dcgmCudaContexts)
    {
        if (!worker->IsRunning())
        {
            if (worker->Run() < 0)
            {
                if (workers > 0)
                {
                    AbortChildren(workers);
                }

                error_reporter << "Failed to start CUDA test on " << worker->Device() << error_reporter.new_line;

                return DCGM_ST_GENERIC_ERROR;
            }

            workers++;
        }
        else
        {
            m_startedWorkers++;
        }


        m_tester->ReportWorkerStarted(worker);
    }

    return DCGM_ST_OK;
}

// Actually run tests when all are started.
dcgmReturn_t PhysicalGpu::RunTests(void)
{
    dcgmReturn_t rtSt { DCGM_ST_OK };

    rtSt = CreateDcgmGroups();

    if (rtSt != DCGM_ST_OK)
    {
        return rtSt;
    }

    m_responseFn    = &PhysicalGpu::ProcessRunningResponse;
    m_exitRequested = false;
    m_valid         = true;

    switch (m_parameters.m_fieldId)
    {
        case DCGM_FI_PROF_GR_ENGINE_ACTIVE:
            if (m_parameters.m_targetMaxValue)
                rtSt = RunSubtestSmOccupancyTargetMax();
            else
                rtSt = RunSubtestGrActivity();
            break;

        case DCGM_FI_PROF_SM_ACTIVE:
            if (m_parameters.m_targetMaxValue)
                rtSt = RunSubtestSmOccupancyTargetMax();
            else
                rtSt = RunSubtestSmActivity();
            break;

        case DCGM_FI_PROF_SM_OCCUPANCY:
            if (m_parameters.m_targetMaxValue)
                rtSt = RunSubtestSmOccupancyTargetMax();
            else
                rtSt = RunSubtestSmOccupancy();
            break;

        case DCGM_FI_PROF_PCIE_RX_BYTES:
        case DCGM_FI_PROF_PCIE_TX_BYTES:
            rtSt = RunSubtestPcieBandwidth();
            break;

        case DCGM_FI_PROF_DRAM_ACTIVE:
            rtSt = RunSubtestDramUtil();
            break;

        case DCGM_FI_PROF_PIPE_FP32_ACTIVE:
        case DCGM_FI_PROF_PIPE_FP64_ACTIVE:
        case DCGM_FI_PROF_PIPE_FP16_ACTIVE:
        case DCGM_FI_PROF_PIPE_TENSOR_ACTIVE:
            rtSt = RunSubtestGemmUtil();
            break;

        case DCGM_FI_PROF_NVLINK_RX_BYTES:
        case DCGM_FI_PROF_NVLINK_TX_BYTES:
            rtSt = RunSubtestNvLinkBandwidth();
            break;

        default:
            warn_reporter << "A test for fieldId " << m_parameters.m_fieldId << " has not been implemented yet."
                          << warn_reporter.new_line;

            rtSt = DCGM_ST_GENERIC_ERROR;
            break;
    }

    if (rtSt != DCGM_ST_OK)
    {
        DestroyDcgmGroups();
    }

    return rtSt;
}


// Process a response from a GPU slice.
dcgmReturn_t PhysicalGpu::ProcessResponse(std::shared_ptr<DistributedCudaContext> worker)

{
    m_workerStarted  = false;
    m_workerRunning  = false;
    m_workerFinished = false;
    m_workerReported = false;

    auto rtSt = (this->*m_responseFn)(worker);

    return rtSt;
}

// Process a response from a GPU slice that is starting up.
dcgmReturn_t PhysicalGpu::ProcessStartingResponse(std::shared_ptr<DistributedCudaContext> worker)
{
    int retval;

    if ((retval = worker->ReadLn()) < 0)
    {
        error_reporter << "ProcessStartingResponse failed to read from a worker." << error_reporter.new_line;
        // Something went wrong.
        // NukeChildren(true);

        return DCGM_ST_GENERIC_ERROR;
    }

    if (retval == 0)
    {
        // We don't have a complete response yet.
        return DCGM_ST_OK;
    }

    dcgmReturn_t rtSt { DCGM_ST_OK };

    char c;

    worker->Input() >> c;

    switch (c)
    {
        /* Failure: we take note, consume line, and count worker. */
        case 'F': // F -- fail
            m_failedWorkers++;
            m_finishedWorkers++;
            worker->SetFailed();
            worker->SetFinished();
            [[fallthrough]];

        case 'P': // P\n -- pass
            worker->Input().ignore(MaxStreamLength, '\n');
            m_startedWorkers++;
            m_workerStarted = true;

            break;

            /* Unexpected: we consume the message, and hope for the best. */
        case 'E': // E <length>\n<errors> - errors
            m_workerReported = true;
            [[fallthrough]];

        case 'M': // M <length>\n<messages> - messages
            std::size_t length;
            worker->Input() >> length;
            worker->Input().ignore(MaxStreamLength, '\n');
            length -= worker->Input().rdbuf()->in_avail();

            if (length > 0)
            {
                worker->Read(length); // Read rest.
                worker->Input().ignore(MaxStreamLength, '\n');
            }

            warn_reporter << "Received unexpected worker status: " << c << warn_reporter.new_line;

            break;

            /* Unexpected: we consume the line, and hope for the best. */
        case 'S': // S\n -- start
            m_workerRunning = true;
            [[fallthrough]];

        case 'D': // D <part> <parts>\n -- done part of test
        case 'T': // T <data> - tick

            // Unknown: we consume the line, and hope for the best.
        default:
            worker->Input().ignore(MaxStreamLength, '\n');
            warn_reporter << "Received unexpected worker status: " << c << warn_reporter.new_line;

            break;
    }

    if (AllFinished())
    {
        m_responseFn = &PhysicalGpu::ProcessFinishedResponse;
        rtSt         = CommandAll(true, "M\nE\n");
    }
    else if (AllStarted())
    {
        /**
         * We no longer start running tests here, since the ProcessResponse()
         * dispatcher (that ultimately called us) wants to limit how many
         * GPUs are tested in parallel. By it calling AllStarted() before
         * and after ProcessResponse() it can determine if all (if any) MIG
         * slices have started and thus if RunTests() can be called.
         */
        // rtSt = RunTests();
        rtSt = DCGM_ST_OK;
    }

    if (rtSt == DCGM_ST_NOT_SUPPORTED)
    {
        CommandAll(true, "M\nE\n");
        NukeChildren(true);
        rtSt = DCGM_ST_OK; // We don't do the test and mark it passed.
    }

    return rtSt;
}


// Process a response from a GPU slice that is running tests.
dcgmReturn_t PhysicalGpu::ProcessRunningResponse(std::shared_ptr<DistributedCudaContext> worker)
{
    if (m_reportedWorkers >= m_workers) // We finished early
    {
        return DCGM_ST_OK;
    }

    if (m_finishedWorkers >= m_workers)
    {
        error_reporter << "ProcessRunningResponse called when all workers finished." << error_reporter.new_line;

        return DCGM_ST_GENERIC_ERROR;
    }

    int retval;

    if ((retval = worker->ReadLn()) < 0)
    {
        error_reporter << "ProcessRunningResponse failed to read from a worker." << error_reporter.new_line;
        // Something went wrong.
        // NukeChildren(true);

        return DCGM_ST_GENERIC_ERROR;
    }

    if (retval == 0)
    {
        // We don't have a complete response yet.

        return DCGM_ST_OK;
    }

    /**
     * This represents the desired dcgm value read. It is only read on "valid"
     * ticks from workers, but is always passed to the tick handler (along with
     * a bool indicating whether it is valid or not).
     *
     * For a tick to be "valid" it must have come from the last worker to
     * have sent it's first response in a work part. So, if there are seven
     * workers, and worker five is the last one in a part to send it's first
     * response in that part, making the total active workers in that part
     * seven (i.e. all of them), ticks by worker five (including that first one
     * in that part), will be marked valid, cause the desired dcgm field to be
     * read, this value updated, and indicated as valid when the particular tick
     * handler is called.
     *
     * It is important for (a) all workers to complete one part before moving
     * on to the rest and (b) the skew between worker ticks be less than the
     * shortest time any worker spends between ticks as we use a simple counter
     * to determine how many workers are busy and only track one part at a time.
     */

    std::map<Entity, dcgmFieldValue_v1> values;

    dcgmReturn_t rtSt { DCGM_ST_OK };
    size_t workerIdx { 0 };
    std::map<Entity, dcgmFieldValue_v1> workerValues;
    dcgmReturn_t dcgmReturn;
    char c;

    worker->Input() >> c;

    switch (c)
    {
        case 'D': // D <part> <parts>\n -- done part of test
            std::size_t part, parts;
            worker->Input() >> part;
            worker->Input() >> parts;
            worker->SetParts(part, parts);
            worker->Input().ignore(MaxStreamLength, '\n');

            break;

        case 'E': // E <length>\n<errors> -- error messages
                  /* We don't expect this unless we asked for it, but we
                   * consume it any way.
                   */
            m_workerReported = true;
            [[fallthrough]];

        case 'M': // M <length>\n<messages> - messages
            /* We don't expect this unless we asked for it, but we consume it
             * any way.
             */
            std::size_t length;
            worker->Input() >> length;
            worker->Input().ignore(MaxStreamLength, '\n');
            length -= worker->Input().rdbuf()->in_avail();

            if (length > 0)
            {
                worker->Read(length); // read rest.
                worker->Input().ignore(MaxStreamLength, '\n');
            }

            break;

        case 'F': // F -- fail
            // Something in the worker failed. This is bad news.
            worker->Input().ignore(MaxStreamLength, '\n');
            m_failedWorkers++;
            m_finishedWorkers++;
            worker->SetFailed();
            worker->SetFinished();
            rtSt = DCGM_ST_GENERIC_ERROR;

            /* We will never have a complete set of workers anymore,
             * so we won't have valid results. We do have to wait
             * for all workers to finish though. Results will get
             * sent back and logged, but won't "count".
             */
            break;

        case 'P': // P\n -- pass
            // The worker finished!
            worker->Input().ignore(MaxStreamLength, '\n');
            m_workerFinished = true;
            m_finishedWorkers++;
            worker->SetFinished();

            break;

        case 'S': // S\n -- start
            // The worker started.
            worker->Input().ignore(MaxStreamLength, '\n');

            break;

        case 'T': // T <data> - tick
                  // The worker asked us to gather data.
            /* This has to be handled specifically to the test we are running,
             * and the part we are in. 'D' responses indicate when we have
             * completed a part.
             *
             * We expect all the workers to complete a "Tick" before any worker
             * completes the next "Tick". Even with a fast 1 ms "Tick" interval,
             * this proves to be the case. This allows just counting how many
             * workers reported "Ticks" to determine when all of them reported
             * a given "Tick".
             */

            worker->GetLatestDcgmValues(values);

            /* We have the rest of the line of Tick notification in the workers.
             * Input() stringstream (and possibly more if it is a multi-line
             * Tick notification (though we do not expect them).
             *
             * If there is any worker specific  stuff to do, do it here. By
             * default, our handler simply consumes the rest of the Tick
             * response line.
             */

            /**
             * Yes, this is a bit inefficient, but we expect no more than 8
             * workers per physical GPU.
             */

            for (auto localWorker : m_dcgmCudaContexts)
            {
                if (worker == localWorker)
                {
                    break;
                }

                workerIdx++;
            }

            for (auto &[type, id] : worker->Entities())
            {
                Entity entity(type, id);

                workerValues[entity] = values[entity];
            }

            dcgmReturn = m_tickHandler(workerIdx, true, workerValues, *worker);

            if (dcgmReturn == DCGM_ST_OK)
            {
                if (m_parameters.m_fast) // Woot! Got a pass in fast mode!
                {
                    m_valid         = true;
                    m_exitRequested = true;
                }
                else if (IsSynchronous())
                {
                    worker->SetTries(m_parameters.m_syncCount);
                    worker->Command("A\n"); // Advance to the next activity.
                }
            }
            else if (dcgmReturn == DCGM_ST_PENDING)
            {
                /**
                 * We were told the desired activity level was not reached.
                 *
                 * We keep retrying until it is either reached, or we are out
                 * of retries.
                 */

                unsigned int tries = worker->GetTries();

                if (tries > 0) // We can try again
                {
                    worker->SetTries(tries - 1);

                    // Demote error to none, decrement try count, and retry.
                    dcgmReturn = DCGM_ST_OK;
                }
            }

            if (dcgmReturn != DCGM_ST_OK)
            {
                /**
                 * We detected a validation error, so we request all
                 * workers on this GPU to terminate early (if not in fast mode),
                 * and we take note of it. Normally, if we don't indicate the
                 * value as valid, the Tick Handler will not validate and report
                 * validation errors, but it may report other kinds of errors.
                 *
                 * It is still important to call it in this case, so as to
                 * read it's tick request, etc. For multi-part tests, the last
                 * first worker to start may vary from part to part, so we
                 * have to keep our communication path in sync.
                 *
                 * In "fast" mode it is a bit different: We tolerate failures,
                 * but exit on the first success.
                 */

                m_valid         = false;
                m_exitRequested = !m_parameters.m_fast;

                if ((dcgmReturn == DCGM_ST_PENDING) && !m_exitRequested && IsSynchronous())
                {
                    /**
                     * We failed, but we can't exit in case we pass, so we
                     * advance anyway if operating synchronously.
                     */
                    worker->Command("A\n"); // Advance to the next activity.
                }
            }

            if (m_exitRequested)
            {
                /**
                 * Early exit was requested by the tick handler, either because
                 * it failed to validate, or it validated after the requisite
                 * waiting period, and we are to terminate the test early in
                 * that case (short circuit operation).
                 */
                CommandAll(true, "Q\n");
            }

            break;

        default:
            // Unknown: we consume the line, and hope for the best.
            worker->Input().ignore(MaxStreamLength, '\n');
            warn_reporter << "Received unexpected worker status: " << c << warn_reporter.new_line;

            break;
    }

    if (AllFinished())
    {
        m_responseFn = &PhysicalGpu::ProcessFinishedResponse;
        auto rtSt2   = CommandAll(true, "M\nE\n");

        if (rtSt2 != DCGM_ST_OK)
        {
            rtSt = rtSt2;
        }
    }

    return rtSt;
}

// Process a response from a GPU slice that has finished running tests.
dcgmReturn_t PhysicalGpu::ProcessFinishedResponse(std::shared_ptr<DistributedCudaContext> worker)
{
    if (m_finishedWorkers < m_workers)
    {
        error_reporter << "ProcessFinishedResponse called before all workers finished." << error_reporter.new_line;

        return DCGM_ST_GENERIC_ERROR;
    }

    if (m_reportedWorkers >= m_workers)
    {
        error_reporter << "ProcessFinishedResponse called after all workers reported." << error_reporter.new_line;

        return DCGM_ST_GENERIC_ERROR;
    }

    int retval;

    /* At this point, all workers have stopped. Some may have failed, and all
     * might have errors or messages to report (generally errors are present
     * only in the face of reported failure, but this is not a hard
     * requirement). Unless we nuked a worker, it will be waiting to be told
     * to exit ("X\n" command).
     *
     * We can pick up and report messages and errors here. These will be
     * asynchronous to other notifications (Start, Ticks, parts Done, Fail,
     * Pass)
     *
     * We should have previously asked all workers for any messages or errors
     * and asked for errors after messages, because that is where we count the
     * interaction done.
     */

    if ((retval = worker->ReadLn()) < 0)
    {
        error_reporter << "ProcessFinishedResponse failed to read from a "
                       << "worker." << error_reporter.new_line;
        // Something wen't wrong.
        // NukeChildren(true);

        return DCGM_ST_GENERIC_ERROR;
    }

    if (retval == 0)
    {
        // We don't have a complete response yet.
        return DCGM_ST_OK;
    }

    dcgmReturn_t rtSt { DCGM_ST_OK };

    char c;

    worker->Input() >> c;

    const char *label = "Message";

    switch (c)
    {
        case 'E': // E <length>\n<errors> - errors
            label = "Error";
            m_reportedWorkers++;
            m_workerReported = true;

            EndSubtest();

            /*rtSt = */ DestroyDcgmGroups();
            [[fallthrough]];

        case 'M': // M <length>\n<messages> - messages
            std::size_t length;
            worker->Input() >> length;
            worker->Input().ignore(MaxStreamLength, '\n');
            length -= (worker->Input().tellp() - worker->Input().tellg());

            if (length > 0)
            {
                worker->Read(length); // read

                /**
                 * Yes, this is a bit inefficient, but we expect no more than 8
                 * workers per physical GPU.
                 */

                size_t workerIdx { 0 };

                for (auto localWorker : m_dcgmCudaContexts)
                {
                    if (worker == localWorker)
                    {
                        break;
                    }

                    workerIdx++;
                }

                if (m_parameters.m_report)
                {
                    info_reporter << "Worker " << m_gpuId << ":" << workerIdx << "[" << m_parameters.m_fieldId
                                  << "]: " << label << ": ";

                    /**
                     * Apparently we can't send big streambufs to the logger,
                     * so we have to break things up.
                     */
                    auto sbuf = worker->Input().rdbuf();

                    do
                    {
                        char ch = sbuf->sgetc();

                        if (ch == '\n')
                        {
                            info_reporter << info_reporter.new_line;
                        }
                        else
                        {
                            info_reporter << ch;
                        }
                    } while (sbuf->snextc() != EOF);
                }
                else
                {
                    worker->Input().str("");
                }
            }

            worker->Input().ignore(worker->Input().tellp() - worker->Input().tellg());

            /* This is predicated on the worker not sending anything else
             * without being prompted after responding to an E(rror) or
             * M(essage) request. Those should only be done when the worker is
             * idle.
             */
            break;

            /* Failure: we take note, consume line, and count worker. */
        case 'F': // F -- fail
            worker->Input().ignore(MaxStreamLength, '\n');
            worker->SetFailed();
            rtSt = DCGM_ST_GENERIC_ERROR;
            break;

            /* Unexpected: we consume the line, and hope for the best. */
        case 'D': // D <part> <parts>\n -- done part of test
        case 'P': // P\n -- pass
        case 'S': // S\n -- start
        case 'T': // T <data> - tick

            // Unknown: we consume the line, and hope for the best.
        default:
            worker->Input().ignore(MaxStreamLength, '\n');
            warn_reporter << "Received unexpected worker status: " << c << warn_reporter.new_line;

            break;
    }

    return rtSt;
}


/*****************************************************************************/

bool PhysicalGpu::IsMIG(void) const
{
    return (m_dcgmDeviceAttr.settings.migModeEnabled != 0);
}

unsigned int PhysicalGpu::GetGpuId(void) const
{
    return m_gpuId;
}

std::string PhysicalGpu::GetGpuBusId(void) const
{
    return std::string(m_dcgmDeviceAttr.identifiers.pciBusId);
}

dcgmHandle_t PhysicalGpu::GetHandle(void) const
{
    return m_dcgmHandle;
}

uint16_t PhysicalGpu::GetFieldId(void) const
{
    return m_parameters.m_fieldId;
}

bool PhysicalGpu::GetDcgmValidation(void) const
{
    return !m_parameters.m_noDcgmValidation;
}

bool PhysicalGpu::IsSynchronous(void) const
{
    return m_parameters.m_syncCount > 0;
}

bool PhysicalGpu::WorkerStarted(void) const
{
    return m_workerStarted;
}

bool PhysicalGpu::WorkerRunning(void) const
{
    return m_workerRunning;
}

bool PhysicalGpu::WorkerFinished(void) const
{
    return m_workerFinished;
}

bool PhysicalGpu::WorkerReported(void) const
{
    return m_workerReported;
}

bool PhysicalGpu::AllStarted(void) const
{
    return m_startedWorkers >= m_workers;
}


bool PhysicalGpu::AllFinished(void) const
{
    return m_finishedWorkers >= m_workers;
}


bool PhysicalGpu::AllReported(void) const
{
    return m_reportedWorkers >= m_workers;
}


bool PhysicalGpu::AnyFailed(void) const
{
    return m_failedWorkers > 0;
}

bool PhysicalGpu::IsValidated(void) const
{
    return m_valid;
}

/*****************************************************************************/
dcgmReturn_t PhysicalGpu::BeginSubtest(std::string testTitle, std::string testTag, bool isLinearTest)
{
    if (m_subtestInProgress)
        EndSubtest();

    if (m_parameters.m_dvsOutput)
        printf("&&&& RUNNING %s\n", testTag.c_str());

    m_subtestDcgmValues.clear();
    m_subtestGenValues.clear();
    m_subtestTitle    = testTitle;
    m_subtestTag      = testTag;
    m_subtestIsLinear = isLinearTest;

    m_subtestInProgress = true;
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t PhysicalGpu::EndSubtest(void)
{
    if (!m_subtestInProgress)
        return DCGM_ST_OK; /* Nothing to do */

    /* Todo: check the two arrays to make sure they are similar
     */

    m_subtestInProgress = false;

    char filename[128] = { 0 };
    snprintf(filename, sizeof(filename), "%s_%d.results", m_subtestTag.c_str(), m_gpuId);
    FILE *testResultsFp = fopen(filename, "wt");
    if (!testResultsFp)
    {
        int cachedErrno = errno;

        error_reporter << "Unable to open " << filename << " (errno " << cachedErrno
                       << "). Can't write test results file." << error_reporter.new_line;

        return DCGM_ST_GENERIC_ERROR;
    }

    fprintf(testResultsFp, "# TestTitle: \"%s\"\n", m_subtestTitle.c_str());

    fprintf(testResultsFp, "# Columns: \"generated\", \"dcgm\"\n");

    for (size_t i = 0; i < m_subtestDcgmValues.size(); i++)
    {
        fprintf(testResultsFp, "%.3f, %.3f\n", m_subtestGenValues[i], m_subtestDcgmValues[i]);
    }

    fprintf(testResultsFp, "# TestResult: %s\n", m_valid ? "PASSED" : "FAILED");
    fprintf(testResultsFp, "# TestResultReason: \n"); /* Todo: Populate with optional text */

    if (m_parameters.m_dvsOutput)
        printf("&&&& %s %s\n", m_valid ? "PASSED" : "FAILED", m_subtestTag.c_str());

    fclose(testResultsFp);
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t PhysicalGpu::AppendSubtestRecord(double generatedValue, double dcgmValue)
{
    m_subtestGenValues.push_back(generatedValue);
    m_subtestDcgmValues.push_back(dcgmValue);
    return DCGM_ST_OK;
}

struct cbData
{
    unsigned int m_gpuId;
    std::map<Entity, std::vector<dcgmFieldValue_v1>> &m_values;

    cbData(unsigned int gpuId, std::map<Entity, std::vector<dcgmFieldValue_v1>> &values)
        : m_gpuId(gpuId)
        , m_values(values)
    {}
};


/*****************************************************************************/
dcgmReturn_t PhysicalGpu::CreateDcgmGroups(void)
{
    dcgmReturn_t dcgmReturn = DCGM_ST_OK;

    if (m_parameters.m_noDcgmValidation)
    {
        DCGM_LOG_INFO << "Skipping CreateDcgmGroups() since DCGM validation is disabled.";

        return DCGM_ST_OK;
    }

    char groupName[32] = { 0 };
    snprintf(groupName, sizeof(groupName), "dpt_%d_%u", getpid(), m_gpuId);

    unsigned short fieldId = m_parameters.m_fieldId;

    dcgmReturn = dcgmFieldGroupCreate(m_dcgmHandle, 1, &fieldId, groupName, &m_fieldGroupId);

    if (dcgmReturn != DCGM_ST_OK)
    {
        error_reporter << "dcgmFieldGroupCreate() returned " << dcgmReturn << "." << error_reporter.new_line;

        return dcgmReturn;
    }

    return dcgmReturn;
}

/*****************************************************************************/
dcgmReturn_t PhysicalGpu::DestroyDcgmGroups(void)
{
    dcgmReturn_t dcgmReturn = DCGM_ST_OK;

    if (m_parameters.m_noDcgmValidation)
    {
        return DCGM_ST_OK;
    }

    dcgmReturn = dcgmFieldGroupDestroy(m_dcgmHandle, m_fieldGroupId);
    if (dcgmReturn != DCGM_ST_OK)
    {
        error_reporter << "dcgmFieldGroupDestroy() returned " << dcgmReturn << "." << error_reporter.new_line;

        return dcgmReturn;
    }

    // We turn this off to note that we destroyed DCGM groups.
    m_parameters.m_noDcgmValidation = true;

    return dcgmReturn;
}


/*****************************************************************************/
dcgmFieldGrp_t PhysicalGpu::GetFieldGroupId(void) const
{
    return m_fieldGroupId;
}

/*****************************************************************************/
dcgmReturn_t PhysicalGpu::RunSubtestSmOccupancyTargetMax(void)
{
    dcgmReturn_t rtSt;

    SetTickHandler([this, firstTick = false](size_t index,
                                             bool valid,
                                             std::map<Entity, dcgmFieldValue_v1> &values,
                                             DistributedCudaContext &worker) mutable -> dcgmReturn_t {
        unsigned int part;
        unsigned int parts;
        bool firstPartTick = worker.IsFirstTick();

        worker.GetParts(part, parts);

        if (firstPartTick)
        {
            if (!firstTick) // First per-test per-GPU code here.
            {
                firstTick = true;

                BeginSubtest("SM Occupancy - Max", "sm_occupancy_max", false);
            }

            if (!m_tester->IsFirstTick()) // First per test code here.
            {
                m_tester->SetFirstTick();

                if (m_parameters.m_report)
                {
                    auto ss    = std::cout.precision();
                    auto flags = std::cout.flags();

                    info_reporter << std::fixed;
                    info_reporter << "GPU " << m_gpuId << ": "
                                  << "Testing Maximum "
                                  << "SmOccupancy/SmActivity/GrActivity "
                                  << "for " << std::setprecision(3) << m_parameters.m_duration << " seconds."
                                  << info_reporter.new_line;

                    info_reporter << "-------------------------------------------------------------"
                                  << info_reporter.new_line;

                    std::cout.precision(ss);
                    std::cout.flags(flags);
                }
            }
        }

        double howFarIn;
        double howFarInCur;
        double timeOffset;
        unsigned int threadsPerSm;

        worker.Input() >> howFarIn;
        worker.Input() >> howFarInCur;
        worker.Input() >> timeOffset;
        worker.Input() >> threadsPerSm;
        worker.Input().ignore(MaxStreamLength, '\n');

        if (valid)
        {
            if (m_parameters.m_report)
            {
                auto ss    = std::cout.precision();
                auto flags = std::cout.flags();

                info_reporter << std::fixed;

                info_reporter << "Worker " << m_gpuId << ":" << index << "[" << m_parameters.m_fieldId
                              << "]: SmOccupancy generated 1.0, dcgm " << std::setprecision(3);

                ValuesDump(values, ValueType::Double, 1.0);
                //<< value.value.dbl

                info_reporter << " at " << std::setprecision(3) << timeOffset << " seconds. threadsPerSm "
                              << threadsPerSm << "." << info_reporter.new_line;

                std::cout.precision(ss);
                std::cout.flags(flags);
            }

            double value;

            worker.SetValidated(ValueGet(values, worker.Entities(), DCGM_FE_GPU_CI, ValueType::Double, 1.0, value)
                                && Validate(1.0, 1.0, value, howFarIn, worker.GetValidated()));

            AppendSubtestRecord(1.0, value);
        }

        return worker.GetValidated() ? DCGM_ST_OK : DCGM_ST_PENDING;
    });

    rtSt = CommandAll(false,
                      "R %u %.3f %.3f %s\n",
                      m_parameters.m_fieldId,
                      m_parameters.m_duration,
                      m_parameters.m_reportInterval,
                      m_parameters.m_targetMaxValue ? "true" : "false");

    return rtSt;
}

/*****************************************************************************/
dcgmReturn_t PhysicalGpu::RunSubtestSmOccupancy(void)
{
    /* Generate SM occupancy */

    dcgmReturn_t rtSt;

    SetTickHandler([this, firstTick = false, nextPart = 0](size_t index,
                                                           bool valid,
                                                           std::map<Entity, dcgmFieldValue_v1> &values,
                                                           DistributedCudaContext &worker) mutable -> dcgmReturn_t {
        unsigned int part;
        unsigned int parts;
        bool firstPartTick = worker.IsFirstTick();

        worker.GetParts(part, parts);

        if (part == 0)
        {
            if (firstPartTick)
            {
                if (!firstTick) // First per-test part per-GPU code here.
                {
                    firstTick = true;
                    nextPart  = 1;

                    BeginSubtest("SM Occupancy - Num Threads", "sm_occupancy_num_threads", true);
                }

                if (!m_tester->IsFirstTick()) // First per test part code here.
                {
                    m_tester->SetFirstTick();
                    m_tester->SetNextPart(1);

                    if (m_parameters.m_report)
                    {
                        auto ss    = std::cout.precision();
                        auto flags = std::cout.flags();

                        info_reporter << std::fixed;

                        info_reporter << "GPU " << m_gpuId << ": "
                                      << "Testing SmOccupancy scaling by "
                                      << "num threads for " << std::setprecision(3) << m_parameters.m_duration / 3.0
                                      << " seconds." << info_reporter.new_line;

                        info_reporter << "-------------------------------------------------------------"
                                      << info_reporter.new_line;

                        std::cout.precision(ss);
                        std::cout.flags(flags);
                    }
                }
            }

            double howFarIn;
            double prevOccupancy;
            double curOccupancy;
            double timeOffset;
            unsigned int threadsPerSm;
            unsigned int maxThreadsPerMultiProcessor;

            worker.Input() >> howFarIn;
            worker.Input() >> prevOccupancy;
            worker.Input() >> curOccupancy;
            worker.Input() >> timeOffset;
            worker.Input() >> threadsPerSm;
            worker.Input() >> maxThreadsPerMultiProcessor;
            worker.Input().ignore(MaxStreamLength, '\n');

            if (valid)
            {
                if (m_parameters.m_report)
                {
                    auto ss    = std::cout.precision();
                    auto flags = std::cout.flags();

                    info_reporter << std::fixed;

                    info_reporter << "Worker " << m_gpuId << ":" << index << "[" << m_parameters.m_fieldId
                                  << "]: SmOccupancy generated " << std::setprecision(3) << prevOccupancy << "/"
                                  << curOccupancy << ", dcgm ";

                    ValuesDump(values, ValueType::Double, 1.0);
                    //<< value.value.dbl

                    info_reporter << " at " << std::setprecision(3) << timeOffset << " seconds. threadsPerSm "
                                  << threadsPerSm << " / " << maxThreadsPerMultiProcessor << "."
                                  << info_reporter.new_line;

                    std::cout.precision(ss);
                    std::cout.flags(flags);
                }

                double value;

                worker.SetValidated(ValueGet(values, worker.Entities(), DCGM_FE_GPU_CI, ValueType::Double, 1.0, value)
                                    && Validate(prevOccupancy, curOccupancy, value, howFarIn, worker.GetValidated()));

                AppendSubtestRecord(prevOccupancy, value);
            }
        }
        else if (part == 1)
        {
            if (firstPartTick)
            {
                if (nextPart == 1)
                {
                    nextPart  = 2;
                    firstTick = false;
                }

                if (!firstTick) // First per-test part per-GPU code here.
                {
                    firstTick = true;

                    EndSubtest();

                    if (m_parameters.m_report)
                    {
                        info_reporter << "Worker " << m_gpuId << ":" << index << "[" << m_parameters.m_fieldId
                                      << "]: Slept to let previous test fall off." << info_reporter.new_line;
                    }

                    BeginSubtest("SM Occupancy - SM Count", "sm_occupancy_sm_count", true);
                }

                if (m_tester->GetNextPart() == 1)
                {
                    m_tester->SetNextPart(2);
                    m_tester->SetFirstTick(false);
                }

                if (!m_tester->IsFirstTick()) // First per test part code here.
                {
                    m_tester->SetFirstTick();

                    if (m_parameters.m_report)
                    {
                        auto ss    = std::cout.precision();
                        auto flags = std::cout.flags();

                        info_reporter << std::fixed;

                        info_reporter << "GPU " << m_gpuId << ": "
                                      << "Testing SmOccupancy scaling "
                                      << "by SM count for " << std::setprecision(3) << m_parameters.m_duration / 3.0
                                      << " seconds." << info_reporter.new_line;

                        info_reporter << "----------------------------------------------------------"
                                      << info_reporter.new_line;

                        std::cout.precision(ss);
                        std::cout.flags(flags);
                    }
                }
            }

            double howFarIn;
            double prevOccupancy;
            double curOccupancy;
            double timeOffset;
            unsigned int numSms;
            unsigned int multiProcessorCount;

            worker.Input() >> howFarIn;
            worker.Input() >> prevOccupancy;
            worker.Input() >> curOccupancy;
            worker.Input() >> timeOffset;
            worker.Input() >> numSms;
            worker.Input() >> multiProcessorCount;
            worker.Input().ignore(MaxStreamLength, '\n');

            if (valid)
            {
                if (m_parameters.m_report)
                {
                    auto ss    = std::cout.precision();
                    auto flags = std::cout.flags();

                    info_reporter << std::fixed << std::setprecision(3);

                    info_reporter << "Worker " << m_gpuId << ":" << index << "[" << m_parameters.m_fieldId
                                  << "]: SmOccupancy generated " << prevOccupancy << "/" << curOccupancy << ", dcgm ";

                    ValuesDump(values, ValueType::Double, 1.0);
                    //<< value.value.dbl

                    info_reporter << " at " << std::setprecision(3) << timeOffset << " seconds."
                                  << " numSms " << numSms << " / " << multiProcessorCount << "."
                                  << info_reporter.new_line;

                    std::cout.precision(ss);
                    std::cout.flags(flags);
                }

                double value;

                worker.SetValidated(ValueGet(values, worker.Entities(), DCGM_FE_GPU_CI, ValueType::Double, 1.0, value)
                                    && Validate(prevOccupancy, curOccupancy, value, howFarIn, worker.GetValidated()));

                AppendSubtestRecord(prevOccupancy, value);
            }
        }
        else if (part == 2)
        {
            if (firstPartTick)
            {
                if (nextPart == 2)
                {
                    nextPart  = 3;
                    firstTick = false;
                }

                if (!firstTick) // First per-test part per-GPU code here.
                {
                    firstTick = true;

                    EndSubtest();

                    if (m_parameters.m_report)
                    {
                        info_reporter << "Worker " << m_gpuId << ":" << index << "[" << m_parameters.m_fieldId
                                      << "]: Slept to let previous test fall off." << info_reporter.new_line;
                    }

                    BeginSubtest("SM Occupancy - CPU Sleeps", "sm_occupancy_cpu_sleeps", true);
                }

                if (m_tester->GetNextPart() == 2)
                {
                    m_tester->SetNextPart(3);
                    m_tester->SetFirstTick(false);
                }

                if (!m_tester->IsFirstTick()) // First per test part code here.
                {
                    m_tester->SetFirstTick();

                    if (m_parameters.m_report)
                    {
                        auto ss    = std::cout.precision();
                        auto flags = std::cout.flags();

                        info_reporter << std::fixed;

                        info_reporter << "GPU " << m_gpuId << ": "
                                      << "Testing SmOccupancy scaling "
                                      << "by CPU sleeps  for " << std::setprecision(3) << m_parameters.m_duration / 3.0
                                      << " seconds." << info_reporter.new_line;

                        info_reporter << "----------------------------------------------------------"
                                      << info_reporter.new_line;

                        std::cout.precision(ss);
                        std::cout.flags(flags);
                    }
                }
            }

            double howFarIn;
            double prevOccupancy;
            double curOccupancy;
            double timeOffset;

            worker.Input() >> howFarIn;
            worker.Input() >> prevOccupancy;
            worker.Input() >> curOccupancy;
            worker.Input() >> timeOffset;
            worker.Input().ignore(MaxStreamLength, '\n');

            if (valid)
            {
                if (m_parameters.m_report)
                {
                    auto ss    = std::cout.precision();
                    auto flags = std::cout.flags();

                    info_reporter << std::fixed << std::setprecision(3);

                    info_reporter << "Worker " << m_gpuId << ":" << index << "[" << m_parameters.m_fieldId
                                  << "]: SmOccupancy generated " << prevOccupancy << "/" << curOccupancy << ", dcgm ";

                    ValuesDump(values, ValueType::Double, 1.0);
                    //<< value.value.dbl

                    info_reporter << " at " << std::setprecision(3) << timeOffset << " seconds."
                                  << info_reporter.new_line;

                    std::cout.precision(ss);
                    std::cout.flags(flags);
                }

                double value;

                worker.SetValidated(ValueGet(values, worker.Entities(), DCGM_FE_GPU_CI, ValueType::Double, 1.0, value)
                                    && Validate(prevOccupancy, curOccupancy, value, howFarIn, worker.GetValidated()));

                AppendSubtestRecord(prevOccupancy, value);
            }
        }

        return worker.GetValidated() ? DCGM_ST_OK : DCGM_ST_PENDING;
    });

    rtSt = CommandAll(false,
                      "R %u %.3f %.3f %s\n",
                      m_parameters.m_fieldId,
                      m_parameters.m_duration,
                      m_parameters.m_reportInterval,
                      m_parameters.m_targetMaxValue ? "true" : "false");

    return rtSt;
}

/*****************************************************************************/
dcgmReturn_t PhysicalGpu::RunSubtestSmActivity(void)
{
    dcgmReturn_t rtSt;

    SetTickHandler([this, firstTick = false, nextPart = 0](size_t index,
                                                           bool valid,
                                                           std::map<Entity, dcgmFieldValue_v1> &values,
                                                           DistributedCudaContext &worker) mutable -> dcgmReturn_t {
        unsigned int part;
        unsigned int parts;
        bool firstPartTick = worker.IsFirstTick();

        worker.GetParts(part, parts);

        if (part == 0)
        {
            if (firstPartTick)
            {
                if (!firstTick) // First per-test part per-GPU code here.
                {
                    firstTick = true;
                    nextPart  = 1;

                    BeginSubtest("SM Activity - SM Count", "sm_activity_sm_count", true);
                }

                if (!m_tester->IsFirstTick()) // First per test part code here.
                {
                    m_tester->SetFirstTick();
                    m_tester->SetNextPart(1);

                    if (m_parameters.m_report)
                    {
                        auto ss    = std::cout.precision();
                        auto flags = std::cout.flags();

                        info_reporter << std::fixed;

                        info_reporter << "GPU " << m_gpuId << ": "
                                      << "Testing SmActivity scaling by SM count "
                                      << "for " << std::setprecision(3) << m_parameters.m_duration / 2.0 << " seconds."
                                      << info_reporter.new_line;

                        info_reporter << "---------------------------------------------------------"
                                      << info_reporter.new_line;

                        std::cout.precision(ss);
                        std::cout.flags(flags);
                    }
                }
            }

            double howFarIn;
            double prevSmActivity;
            double curSmActivity;
            double timeOffset;
            unsigned int numSms;
            unsigned int multiProcessorCount;

            worker.Input() >> howFarIn;
            worker.Input() >> prevSmActivity;
            worker.Input() >> curSmActivity;
            worker.Input() >> timeOffset;
            worker.Input() >> numSms;
            worker.Input() >> multiProcessorCount;
            worker.Input().ignore(MaxStreamLength, '\n');

            if (valid)
            {
                if (m_parameters.m_report)
                {
                    auto ss    = std::cout.precision();
                    auto flags = std::cout.flags();

                    info_reporter << std::fixed << std::setprecision(3);

                    info_reporter << "Worker " << m_gpuId << ":" << index << "[" << m_parameters.m_fieldId
                                  << "]: SmActivity generated " << prevSmActivity << "/" << curSmActivity << ", dcgm ";

                    ValuesDump(values, ValueType::Double, 1.0);
                    //<< value.value.dbl

                    info_reporter << " at " << std::setprecision(3) << timeOffset << " seconds. numSms " << numSms
                                  << " / " << multiProcessorCount << "." << info_reporter.new_line;

                    std::cout.precision(ss);
                    std::cout.flags(flags);
                }

                double value;

                worker.SetValidated(ValueGet(values, worker.Entities(), DCGM_FE_GPU_CI, ValueType::Double, 1.0, value)
                                    && Validate(prevSmActivity, curSmActivity, value, howFarIn, worker.GetValidated()));

                AppendSubtestRecord(prevSmActivity, value);
            }
        }
        else if (part == 1)
        {
            if (firstPartTick)
            {
                if (nextPart == 1)
                {
                    nextPart  = 2;
                    firstTick = false;
                }

                if (!firstTick) // First per-test part per-GPU code here.
                {
                    firstTick = true;

                    EndSubtest();

                    if (m_parameters.m_report)
                    {
                        info_reporter << "Worker " << m_gpuId << ":" << index << "[" << m_parameters.m_fieldId
                                      << "]: Slept to let previous test fall off." << info_reporter.new_line;
                    }

                    BeginSubtest("SM Activity - CPU Sleeps", "sm_activity_cpu_sleeps", true);
                }

                if (m_tester->GetNextPart() == 1)
                {
                    m_tester->SetNextPart(2);
                    m_tester->SetFirstTick(false);
                }

                if (!m_tester->IsFirstTick()) // First per test part code here.
                {
                    m_tester->SetFirstTick();

                    if (m_parameters.m_report)
                    {
                        auto ss    = std::cout.precision();
                        auto flags = std::cout.flags();

                        info_reporter << std::fixed;

                        info_reporter << "GPU " << m_gpuId << ": "
                                      << "Testing SmActivity scaling by CPU sleeps"
                                      << " for " << std::setprecision(3) << m_parameters.m_duration / 2.0 << " seconds."
                                      << info_reporter.new_line;

                        std::cout.precision(ss);
                        std::cout.flags(flags);
                    }
                }
            }

            double howFarIn;
            double prevSmActivity;
            double curSmActivity;
            double timeOffset;

            worker.Input() >> howFarIn;
            worker.Input() >> prevSmActivity;
            worker.Input() >> curSmActivity;
            worker.Input() >> timeOffset;
            worker.Input().ignore(MaxStreamLength, '\n');

            if (valid)
            {
                if (m_parameters.m_report)
                {
                    auto ss    = std::cout.precision();
                    auto flags = std::cout.flags();

                    info_reporter << std::fixed << std::setprecision(3);

                    info_reporter << "Worker " << m_gpuId << ":" << index << "[" << m_parameters.m_fieldId
                                  << "]: SmActivity: generated " << prevSmActivity << "/" << curSmActivity << ", dcgm ";

                    ValuesDump(values, ValueType::Double, 1.0);
                    //<< value.value.dbl

                    info_reporter << " at " << std::setprecision(3) << timeOffset << " seconds."
                                  << info_reporter.new_line;

                    std::cout.precision(ss);
                    std::cout.flags(flags);
                }

                double value;

                worker.SetValidated(ValueGet(values, worker.Entities(), DCGM_FE_GPU_CI, ValueType::Double, 1.0, value)
                                    && Validate(prevSmActivity, curSmActivity, value, howFarIn, worker.GetValidated()));

                AppendSubtestRecord(prevSmActivity, value);
            }
        }

        return worker.GetValidated() ? DCGM_ST_OK : DCGM_ST_PENDING;
    });

    rtSt = CommandAll(false,
                      "R %u %.3f %.3f %s\n",
                      m_parameters.m_fieldId,
                      m_parameters.m_duration,
                      m_parameters.m_reportInterval,
                      m_parameters.m_targetMaxValue ? "true" : "false");

    return rtSt;
}

/*****************************************************************************/
dcgmReturn_t PhysicalGpu::RunSubtestGrActivity(void)
{
    dcgmReturn_t rtSt;

    SetTickHandler([this, firstTick = false](size_t index,
                                             bool valid,
                                             std::map<Entity, dcgmFieldValue_v1> &values,
                                             DistributedCudaContext &worker) mutable -> dcgmReturn_t {
        unsigned int part;
        unsigned int parts;
        bool firstPartTick = worker.IsFirstTick();

        worker.GetParts(part, parts);

        if (firstPartTick)
        {
            if (!firstTick) // First per-test per-GPU code here.
            {
                firstTick = true;

                BeginSubtest("Graphics Activity", "gr_activity", true);
            }

            if (!m_tester->IsFirstTick()) // Add first per test code here.
            {
                m_tester->SetFirstTick();
            }
        }

        double howFarIn;
        double prevHowFarIn;
        double curHowFarIn;
        double timeOffset;

        worker.Input() >> howFarIn;
        worker.Input() >> prevHowFarIn;
        worker.Input() >> curHowFarIn;
        worker.Input() >> timeOffset;
        worker.Input().ignore(MaxStreamLength, '\n');

        if (valid)
        {
            if (m_parameters.m_report)
            {
                auto ss    = std::cout.precision();
                auto flags = std::cout.flags();

                info_reporter << std::fixed << std::setprecision(3);

                info_reporter << "Worker " << m_gpuId << ":" << index << "[" << m_parameters.m_fieldId
                              << "]: GrActivity: generated " << prevHowFarIn << "/" << curHowFarIn << ", dcgm ";

                ValuesDump(values, ValueType::Double, 1.0); //<< value.value.dbl

                info_reporter << " at " << std::setprecision(3) << timeOffset << " seconds." << info_reporter.new_line;

                std::cout.precision(ss);
                std::cout.flags(flags);
            }

            double value;

            worker.SetValidated(ValueGet(values, worker.Entities(), DCGM_FE_GPU_CI, ValueType::Double, 1.0, value)
                                && Validate(prevHowFarIn, curHowFarIn, value, howFarIn, worker.GetValidated()));

            AppendSubtestRecord(howFarIn, value);
        }

        return worker.GetValidated() ? DCGM_ST_OK : DCGM_ST_PENDING;
    });

    rtSt = CommandAll(false,
                      "R %u %.3f %.3f %s\n",
                      m_parameters.m_fieldId,
                      m_parameters.m_duration,
                      m_parameters.m_reportInterval,
                      m_parameters.m_targetMaxValue ? "true" : "false");

    return rtSt;
}

/*****************************************************************************/
dcgmReturn_t PhysicalGpu::RunSubtestPcieBandwidth(void)
{
    dcgmReturn_t rtSt;

    SetTickHandler(
        [this, firstTick = false, fieldHeading = "", subtestTag = "", prevGpuPerSecond = 0.0, curGpuPerSecond = 0.0](
            size_t index,
            bool valid,
            std::map<Entity, dcgmFieldValue_v1> &values,
            DistributedCudaContext &worker) mutable -> dcgmReturn_t {
            unsigned int part;
            unsigned int parts;
            bool firstPartTick = worker.IsFirstTick();

            worker.GetParts(part, parts);

            if (firstPartTick)
            {
                if (!firstTick) // First per-test per-GPU code here.
                {
                    firstTick = true;

                    if (m_parameters.m_fieldId == DCGM_FI_PROF_PCIE_RX_BYTES)
                    {
                        fieldHeading = "PcieRxBytes";
                        subtestTag   = "pcie_rx_bytes";
                    }
                    else
                    {
                        fieldHeading = "PcieTxBytes";
                        subtestTag   = "pcie_tx_bytes";
                    }

                    BeginSubtest(fieldHeading, subtestTag, false);
                }

                if (!m_tester->IsFirstTick()) // Add first per test code here.
                {
                    m_tester->SetFirstTick();
                }
            }

            double howFarIn;
            double prevPerSecond;
            double curPerSecond;

            worker.Input() >> howFarIn;
            worker.Input() >> prevPerSecond;
            worker.Input() >> curPerSecond;
            worker.Input().ignore(MaxStreamLength, '\n');

            /*
             * We need to compare across the whole GPU, so we update whole-GPU
             * activity, subtracting the previous activity, and adding the current
             * activity generated.
             */

            curGpuPerSecond += curPerSecond - prevPerSecond;

            if (valid)
            {
                dcgmGroupEntityPair_t entity;
                unsigned short fieldId { DCGM_FI_DEV_PCIE_LINK_GEN };
                dcgmFieldValue_v2 value2 {};

                entity.entityGroupId = DCGM_FE_GPU;
                entity.entityId      = worker.Entities()[DCGM_FE_GPU];

                dcgmReturn_t dcgmReturn = dcgmEntitiesGetLatestValues(
                    m_dcgmHandle, &entity, 1, &fieldId, 1, DCGM_FV_FLAG_LIVE_DATA, &value2);

                if (dcgmReturn != DCGM_ST_OK)
                {
                    error_reporter << "dcgmEntitiesGetLatestValues failed with " << dcgmReturn << " for gpuId "
                                   << m_gpuId << " PCIE test." << error_reporter.new_line;

                    return DCGM_ST_NOT_SUPPORTED;
                }

                unsigned long long pcieVersion = value2.value.i64;

                fieldId = DCGM_FI_DEV_PCIE_LINK_WIDTH;

                dcgmReturn = dcgmEntitiesGetLatestValues(
                    m_dcgmHandle, &entity, 1, &fieldId, 1, DCGM_FV_FLAG_LIVE_DATA, &value2);

                if (dcgmReturn != DCGM_ST_OK)
                {
                    error_reporter << "dcgmEntitiesGetLatestValues failed with " << dcgmReturn << " for gpuId "
                                   << m_gpuId << " PCIE test." << error_reporter.new_line;

                    return DCGM_ST_NOT_SUPPORTED;
                }

                unsigned long long pcieLanes = value2.value.i64;

                // 16x rate for PCIE versions, starting with 1.0, through 5.0
                static double rateX16[] = {
                    2.50 * 8 / 10 * 16 / 8,    // 1.0 GT/s * coding rate * 16 lines / 8 bits per byte
                    5.00 * 8 / 10 * 16 / 8,    // 2.0
                    8.00 * 128 / 130 * 16 / 8, // 3.0
                    16.0 * 128 / 130 * 16 / 8, // 4.0
                    32.0 * 128 / 130 * 16 / 8, // 5.0
                    64.0 * 128 / 130 * 16 / 8  // 6.0 Yes, there is a FEC, but it is part of the line coding
                };

                if (pcieVersion > (sizeof(rateX16) / sizeof(rateX16[0])))
                {
                    // Cap the version at the max we know about.
                    pcieVersion = sizeof(rateX16) / sizeof(rateX16[0]);
                }

                double expectedRate = rateX16[pcieVersion - 1] / 16 * pcieLanes * 1000.0;

                if (m_parameters.m_report)
                {
                    auto ss    = std::cout.precision();
                    auto flags = std::cout.flags();

                    info_reporter << std::fixed << std::setprecision(0);

                    info_reporter << "Worker " << m_gpuId << ":" << index << "[" << m_parameters.m_fieldId
                                  << "]: " << fieldHeading << " generated " << prevGpuPerSecond << "/"
                                  << curGpuPerSecond << " (" << prevPerSecond << "/" << curPerSecond << ")"
                                  << ", dcgm ";

                    ValuesDump(values, ValueType::Int64, 1000.0 * 1000.0);
                    //<< dcgmValue RSH - i64 / 10^6

                    info_reporter << " MiB/sec (";

                    std::cout.precision(ss);
                    std::cout.flags(flags);
                }

                double value;

                /*
                 * Restore validation to driven vs. measured instead of against
                 * speed of light maximum. We also scale by the number of workers
                 * since the value read is for the whole GPU and the value driven
                 * is per worker, and driving is distributed evenly among workers,
                 * regardless of the partition size of the per-worker MIG slice.
                 */
                auto validated
                    = ValueGet(values, worker.Entities(), DCGM_FE_GPU, ValueType::Int64, 1000.0 * 1000.0, value);

                AppendSubtestRecord(prevPerSecond, value);

                if (m_parameters.m_report)
                {
                    auto ss    = std::cout.precision();
                    auto flags = std::cout.flags();

                    info_reporter << std::fixed << std::setprecision(0);

                    info_reporter << 100.0 * value / expectedRate << "% speed-of-light)"
                                  << ", PCIE version/lanes: " << pcieVersion << "/" << pcieLanes << "."
                                  << info_reporter.new_line;

                    std::cout.precision(ss);
                    std::cout.flags(flags);
                }

                // We clamp near zero to avoid problems with noise.
                if (value < 0.5)
                {
                    value = 0.0;
                }

                if (curGpuPerSecond < 0.5)
                {
                    curGpuPerSecond = 0.0;
                }

                worker.SetValidated(
                    validated && Validate(prevGpuPerSecond, curGpuPerSecond, value, howFarIn, worker.GetValidated()));
            }

            prevGpuPerSecond = curGpuPerSecond;

            return worker.GetValidated() ? DCGM_ST_OK : DCGM_ST_PENDING;
        });

    rtSt = CommandAll(false,
                      "R %u %.3f %.3f %s\n",
                      m_parameters.m_fieldId,
                      m_parameters.m_duration,
                      m_parameters.m_reportInterval,
                      m_parameters.m_targetMaxValue ? "true" : "false");

    return rtSt;
}

/*****************************************************************************/
dcgmReturn_t PhysicalGpu::HelperGetBestNvLinkPeer(std::string &peerPciBusId, unsigned int &nvLinks)
{
    int i;

    dcgmDeviceTopology_v1 deviceTopo;
    memset(&deviceTopo, 0, sizeof(deviceTopo));
    deviceTopo.version = dcgmDeviceTopology_version1;

    dcgmReturn_t dcgmReturn = dcgmGetDeviceTopology(m_dcgmHandle, m_gpuId, &deviceTopo);
    if (dcgmReturn != DCGM_ST_OK)
    {
        DCGM_LOG_WARNING << "dcgmGetDeviceTopology failed with " << dcgmReturn << " for gpuId " << m_gpuId << "."
                         << Reporter::Flags::new_line;

        return DCGM_ST_NOT_SUPPORTED;
    }

    int nvLinkMasks = DCGM_TOPOLOGY_NVLINK1 | DCGM_TOPOLOGY_NVLINK2 | DCGM_TOPOLOGY_NVLINK3 | DCGM_TOPOLOGY_NVLINK4
                      | DCGM_TOPOLOGY_NVLINK5 | DCGM_TOPOLOGY_NVLINK6 | DCGM_TOPOLOGY_NVLINK7 | DCGM_TOPOLOGY_NVLINK8
                      | DCGM_TOPOLOGY_NVLINK9 | DCGM_TOPOLOGY_NVLINK10 | DCGM_TOPOLOGY_NVLINK11
                      | DCGM_TOPOLOGY_NVLINK12;

    /* Find the GPU we have the most NvLink connections to */
    unsigned int bestGpuId   = 0;
    unsigned int maxNumLinks = 0;
    for (i = 0; i < (int)deviceTopo.numGpus; i++)
    {
        if (!(deviceTopo.gpuPaths[i].path & nvLinkMasks))
            continue;

        /* More links = higher mask value */
        unsigned int numNvLinks = __builtin_popcount(deviceTopo.gpuPaths[i].localNvLinkIds);

        if (numNvLinks < maxNumLinks)
            continue;

        bestGpuId   = deviceTopo.gpuPaths[i].gpuId;
        maxNumLinks = numNvLinks;
    }

    if (maxNumLinks == 0)
    {
        warn_reporter << "gpuId " << m_gpuId << " has no NvLink peers. Skipping test." << warn_reporter.new_line;

        return DCGM_ST_NOT_SUPPORTED;
    }

    dcgmDeviceAttributes_v2 peerDeviceAttr;
    memset(&peerDeviceAttr, 0, sizeof(peerDeviceAttr));
    peerDeviceAttr.version = dcgmDeviceAttributes_version2;
    dcgmReturn             = dcgmGetDeviceAttributes(m_dcgmHandle, bestGpuId, &peerDeviceAttr);
    if (dcgmReturn != DCGM_ST_OK)
    {
        error_reporter << "dcgmGetDeviceAttributes failed with " << dcgmReturn << " for gpuId " << bestGpuId << "."
                       << error_reporter.new_line;

        return dcgmReturn;
    }

    peerPciBusId = std::string(peerDeviceAttr.identifiers.pciBusId);
    nvLinks      = maxNumLinks;

    DCGM_LOG_INFO << "The best peer of gpuId %u is gpuId " << bestGpuId << ", numLinks " << nvLinks << ".";

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t PhysicalGpu::RunSubtestNvLinkBandwidth(void)
{
    dcgmReturn_t rtSt;

    if (m_dcgmDeviceAttr.settings.migModeEnabled != 0) // MIG: can not do this.
    {
        warn_reporter << "GPU " << m_gpuId << ": Can not run NvLink tests in MIG mode." << warn_reporter.new_line;

        return DCGM_ST_GENERIC_ERROR;
    }

    std::string peerPciBusId;
    unsigned int nvLinks;

    /* Get our best peer to do NvLink copies to */
    rtSt = HelperGetBestNvLinkPeer(peerPciBusId, nvLinks);
    if (rtSt != DCGM_ST_OK)
    {
        return rtSt;
    }

    SetTickHandler([this, nvLinks, firstTick = false, fieldHeading = "", subtestTag = ""](
                       size_t index,
                       bool valid,
                       std::map<Entity, dcgmFieldValue_v1> &values,
                       DistributedCudaContext &worker) mutable -> dcgmReturn_t {
        unsigned int part;
        unsigned int parts;
        bool firstPartTick = worker.IsFirstTick();

        worker.GetParts(part, parts);

        if (m_parameters.m_fieldId == DCGM_FI_PROF_NVLINK_RX_BYTES)
        {
            fieldHeading = "NvLinkRxBytes";
            subtestTag   = "nvlink_rx_bytes";
        }
        else
        {
            fieldHeading = "NvLinkTxBytes";
            subtestTag   = "nvlink_tx_bytes";
        }

        if (firstPartTick)
        {
            if (!firstTick) // First per-test per-GPU code here.
            {
                firstTick = true;

                BeginSubtest(fieldHeading, subtestTag, false);
            }

            if (!m_tester->IsFirstTick()) // Add first per test code here.
            {
                m_tester->SetFirstTick();
            }
        }

        unsigned int nvLinkMbPerSec { 0 };

        switch (m_majorComputeCapability)
        {
            case 0:
            case 1:
            case 2:
            case 3:
            case 4:
            case 5:
                break;

            case 6: // PASCAL
                nvLinkMbPerSec = 20000;
                break;

            case 7: // VOLTA/TURING
                nvLinkMbPerSec = 25000;
                break;

            case 8: // AMPERE
                nvLinkMbPerSec = 50000;
                break;

            default:
                break;
        }

        nvLinkMbPerSec *= nvLinks;

        double howFarIn;
        double prevPerSecond;
        double curPerSecond;

        worker.Input() >> howFarIn;
        worker.Input() >> prevPerSecond;
        worker.Input() >> curPerSecond;
        worker.Input().ignore(MaxStreamLength, '\n');

        if (valid)
        {
            if (m_parameters.m_report)
            {
                auto ss    = std::cout.precision();
                auto flags = std::cout.flags();

                info_reporter << std::fixed << std::setprecision(0);

                info_reporter << "Worker " << m_gpuId << ":" << index << "[" << m_parameters.m_fieldId
                              << "]: " << fieldHeading << " generated " << prevPerSecond << "/" << curPerSecond
                              << ", dcgm ";

                ValuesDump(values, ValueType::Int64, 1000.0 * 1000.0);
                // << value.value.i64 / 1000 / 1000

                info_reporter << " MiB/sec (";

                std::cout.precision(ss);
                std::cout.flags(flags);
            }

            double value;
            auto validated
                = ValueGet(values, worker.Entities(), DCGM_FE_GPU_CI, ValueType::Int64, 1000.0 * 1000.0, value);

            AppendSubtestRecord(prevPerSecond, value);

            if (m_parameters.m_report)
            {
                auto ss    = std::cout.precision();
                auto flags = std::cout.flags();

                info_reporter << std::fixed << std::setprecision(0);

                info_reporter << 100.0 * value / nvLinkMbPerSec << "% speed-of-light)." << info_reporter.new_line;

                std::cout.precision(ss);
                std::cout.flags(flags);
            }

            worker.SetValidated(validated
                                && Validate(prevPerSecond, curPerSecond, value, howFarIn, worker.GetValidated()));
        }

        return worker.GetValidated() ? DCGM_ST_OK : DCGM_ST_PENDING;
    });

    rtSt = CommandAll(false,
                      "R %u %.3f %.3f %s %s\n",
                      m_parameters.m_fieldId,
                      m_parameters.m_duration,
                      m_parameters.m_reportInterval,
                      m_parameters.m_targetMaxValue ? "true" : "false",
                      peerPciBusId.c_str());

    return rtSt;
}


/*****************************************************************************/
dcgmReturn_t PhysicalGpu::RunSubtestDramUtil(void)
{
    dcgmReturn_t rtSt;

    SetTickHandler([this, firstTick = false](size_t index,
                                             bool valid,
                                             std::map<Entity, dcgmFieldValue_v1> &values,
                                             DistributedCudaContext &worker) mutable -> dcgmReturn_t {
        unsigned int part;
        unsigned int parts;
        bool firstPartTick = worker.IsFirstTick();

        worker.GetParts(part, parts);

        if (firstPartTick)
        {
            if (!firstTick) // First per-test per-GPU code here.
            {
                firstTick = true;

                BeginSubtest("DRAM Activity", "dram_activity", false);
            }

            if (!m_tester->IsFirstTick()) // Add first per test code here.
            {
                m_tester->SetFirstTick();
            }
        }

        double howFarIn;
        double prevDramAct;
        double curDramAct;
        double prevPerSecond;
        unsigned int eccAffectsBandwidth;

        worker.Input() >> howFarIn;
        worker.Input() >> prevDramAct;
        worker.Input() >> curDramAct;
        worker.Input() >> prevPerSecond;
        worker.Input() >> eccAffectsBandwidth;
        worker.Input().ignore(MaxStreamLength, '\n');

        double bandwidthDivisor = 1.0;
        if (eccAffectsBandwidth != 0)
        {
            bandwidthDivisor = 9.0 / 8.0; /* 1 parity bit per 8 bits = 9/8ths expected bandwidth */
        }

        if (valid)
        {
            if (m_parameters.m_report)
            {
                auto ss    = std::cout.precision();
                auto flags = std::cout.flags();

                info_reporter << std::fixed << std::setprecision(3);

                info_reporter << "Worker " << m_gpuId << ":" << index << "[" << m_parameters.m_fieldId
                              << "]: DramUtil generated " << prevDramAct << "/" << curDramAct << ", dcgm ";


                ValuesDump(values, ValueType::Double, bandwidthDivisor); //<< value.value.dbl

                info_reporter << " (" << std::setprecision(1) << prevPerSecond << " GiB/sec)"
                              << ", bandwidthDivisor " << std::setprecision(3) << bandwidthDivisor << "."
                              << info_reporter.new_line;

                std::cout.precision(ss);
                std::cout.flags(flags);
            }

            double value;

            worker.SetValidated(
                ValueGet(values, worker.Entities(), DCGM_FE_GPU_I, ValueType::Double, bandwidthDivisor, value)
                && Validate(prevDramAct, curDramAct, value, howFarIn, worker.GetValidated()));

            AppendSubtestRecord(prevDramAct, value);
        }

        return worker.GetValidated() ? DCGM_ST_OK : DCGM_ST_PENDING;
    });

    rtSt = CommandAll(false,
                      "R %u %.3f %.3f %s\n",
                      m_parameters.m_fieldId,
                      m_parameters.m_duration,
                      m_parameters.m_reportInterval,
                      m_parameters.m_targetMaxValue ? "true" : "false");

    return rtSt;
}

/*****************************************************************************/
/**
 * CUDA version 11 on A10x (x other than 0) is currently unoptimized for
 * FP32 and TENSOR operations. We check for this and adjust our activity
 * expectations.
 */
bool PhysicalGpu::IsComputeUnoptimized(unsigned int fieldId)
{
#if (CUDA_VERSION_USED >= 11)
    switch (fieldId)
    {
        case DCGM_FI_PROF_PIPE_FP32_ACTIVE:
        case DCGM_FI_PROF_PIPE_TENSOR_ACTIVE:
            return (((m_majorComputeCapability == 8) && (m_minorComputeCapability >= 6))
                    || (m_majorComputeCapability > 8));

        default:
            return false;
    }
#else
    return false;
#endif
}

/*****************************************************************************/
/**
 * On A10x or TU104, some GPUs have non-deterministic hardware capabilities.
 * This checks for that for the field in question.
 */
bool PhysicalGpu::IsHardwareNonDeterministic(unsigned int fieldId)
{
    switch (fieldId)
    {
        case DCGM_FI_PROF_PIPE_TENSOR_ACTIVE:
            return ((m_majorComputeCapability == 8) && (m_minorComputeCapability == 6))
                   || ((m_majorComputeCapability == 7) && (m_minorComputeCapability == 5));

        default:
            return false;
    }
}

/*****************************************************************************/
dcgmReturn_t PhysicalGpu::RunSubtestGemmUtil(void)
{
    dcgmReturn_t rtSt;

    const char *testHeader;
    const char *testTag;
    ////double prevValue { 0 };

    switch (m_parameters.m_fieldId)
    {
        case DCGM_FI_PROF_PIPE_FP32_ACTIVE:
            testHeader = "Fp32EngineActive";
            testTag    = "fp32_active";
            break;
        case DCGM_FI_PROF_PIPE_FP64_ACTIVE:
            testHeader = "Fp64EngineActive";
            testTag    = "fp64_active";
            break;
        case DCGM_FI_PROF_PIPE_FP16_ACTIVE:
            testHeader = "Fp16EngineActive";
            testTag    = "fp16_active";
            break;
        case DCGM_FI_PROF_PIPE_TENSOR_ACTIVE:
            testHeader = "TensorEngineActive";
            testTag    = "tensor_active";
            break;
        default:
            error_reporter << "fieldId " << m_parameters.m_fieldId << " is unhandled." << error_reporter.new_line;

            return DCGM_ST_GENERIC_ERROR;
    }

    double limit = IsComputeUnoptimized(m_parameters.m_fieldId) ? 0.4 : 0.5;

    limit = IsHardwareNonDeterministic(m_parameters.m_fieldId) ? 0.2 : limit;

    SetTickHandler(
        [this, firstTick = false, testHeader, testTag, limit](size_t index,
                                                              bool valid,
                                                              std::map<Entity, dcgmFieldValue_v1> &values,
                                                              DistributedCudaContext &worker) mutable -> dcgmReturn_t {
            unsigned int part;
            unsigned int parts;
            bool firstPartTick = worker.IsFirstTick();

            worker.GetParts(part, parts);

            if (firstPartTick)
            {
                if (!firstTick) // First per-test per-GPU code here.
                {
                    firstTick = true;

                    BeginSubtest(testHeader, testTag, false);
                }

                if (!m_tester->IsFirstTick()) // Add first per test code here.
                {
                    m_tester->SetFirstTick();
                }
            }

            double howFarIn;
            double gflops;
            double gflops2;

            worker.Input() >> howFarIn;
            worker.Input() >> gflops;
            worker.Input() >> gflops2;
            worker.Input().ignore(MaxStreamLength, '\n');

            if (valid)
            {
                if (m_parameters.m_report)
                {
                    auto ss    = std::cout.precision();
                    auto flags = std::cout.flags();

                    info_reporter << std::fixed << std::setprecision(3);

                    info_reporter << "Worker " << m_gpuId << ":" << index << "[" << m_parameters.m_fieldId
                                  << "]: " << testHeader << ": generated ???, dcgm ";

                    ValuesDump(values, ValueType::Double, 1.0); //<< value.value.dbl

                    info_reporter << " (" << std::setprecision(1) << gflops << " gflops)." << info_reporter.new_line;

                    std::cout.precision(ss);
                    std::cout.flags(flags);
                }

                double value;

                worker.SetValidated(ValueGet(values, worker.Entities(), DCGM_FE_GPU_CI, ValueType::Double, 1.0, value)
                                    && Validate(limit, 0.9, value, howFarIn, worker.GetValidated()));

                ////prevValue = value;

                AppendSubtestRecord(0.0, value);
            }

            return worker.GetValidated() ? DCGM_ST_OK : DCGM_ST_PENDING;
        });

    rtSt = CommandAll(false,
                      "R %u %.3f %.3f %s\n",
                      m_parameters.m_fieldId,
                      m_parameters.m_duration,
                      m_parameters.m_reportInterval,
                      m_parameters.m_targetMaxValue ? "true" : "false");

    return rtSt;
}

} // namespace DcgmNs::ProfTester
