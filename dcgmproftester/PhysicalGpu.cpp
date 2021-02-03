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
#include "PhysicalGpu.h"
#include "Arguments.h"
#include "DcgmCacheManager.h"
#include "DcgmLogging.h"
#include "DcgmProfTester.h"
#include "DcgmSettings.h"
#include "Entity.h"
#include "dcgm_fields.h"
#include "dcgm_fields_internal.h"
#include "timelib.h"
#include "vector_types.h"
#include <DcgmLogging.h>
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
static void ValuesDump(std::map<Entity, dcgmFieldValue_v1> &values, PhysicalGpu::ValueType type, double divisor)
{
    if (values.size() > 1)
    {
        std::cout << "{ ";
    }

    bool first { true };

    for (auto &[entity, fieldValue] : values)
    {
        if (values.size() > 1)
        {
            if (!first)
            {
                std::cout << ", ";
            }
            else
            {
                first = false;
            }

            switch (entity.m_entity.entityGroupId)
            {
                default:
                case DCGM_FE_NONE:
                    std::cout << "?: ";
                    break;

                case DCGM_FE_GPU:
                    std::cout << "GPU: ";
                    break;

                case DCGM_FE_VGPU:
                    std::cout << "VGPU: ";
                    break;

                case DCGM_FE_SWITCH:
                    std::cout << "SW: ";
                    break;

                case DCGM_FE_GPU_I:
                    std::cout << "GI: ";
                    break;

                case DCGM_FE_GPU_CI:
                    std::cout << "CI: ";
                    break;
            }
        }

        double value;

        switch (type)
        {
            case PhysicalGpu::ValueType::Int64:
                value = fieldValue.value.i64;
                value /= divisor;
                std::cout << value;

                break;

            case PhysicalGpu::ValueType::Double:
                value = fieldValue.value.dbl;
                value /= divisor;
                std::cout << value;

                break;

            case PhysicalGpu::ValueType::String:
                std::cout << "<string>";

                break;

            case PhysicalGpu::ValueType::Blob:
                std::cout << "<blob>";

                break;

            default:
                std::cout << "<unknown>";

                break;
        }
    }

    if (values.size() > 1)
    {
        std::cout << " }";
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
        printf(
            "Skipping GPU virtualization mode check due to nonzero dcgmReturn %d or valueStatus %d. Perfworks may still return an error if the vGPU mode is not supported.",
            (int)dcgmReturn,
            value.status);
        return DCGM_ST_OK;
    }

    if (value.value.i64 != DCGM_GPU_VIRTUALIZATION_MODE_NONE
        && value.value.i64 != DCGM_GPU_VIRTUALIZATION_MODE_PASSTHROUGH)
    {
        fprintf(stderr, "Virtualization mode %ld is unsupported.", value.value.i64);
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
        fprintf(stderr, "dcgmGetDeviceAttributes() returned %d\n", dcgmReturn);
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
bool PhysicalGpu::Validate(double expected, double current, double measured, double howFarIn)
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
                std::cout << "Field " << m_parameters.m_fieldId << " @ " << howFarIn * 100
                          << "% Validation Fail: " << m_parameters.m_minValue << " !< " << measured << " < "
                          << m_parameters.m_maxValue << std::endl;
            }

            return false;
        }

        if (measured > m_parameters.m_maxValue)
        {
            if (!m_parameters.m_fast)
            {
                std::cout << "Field " << m_parameters.m_fieldId << " @ " << howFarIn * 100
                          << "% Validation Fail: " << m_parameters.m_minValue << " !< " << measured << " < "
                          << m_parameters.m_maxValue << std::endl;
            }

            return false;
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
                std::cout << "Field " << m_parameters.m_fieldId << " @ " << howFarIn * 100
                          << "% Validation Fail: " << lowExpected * (1.0 - m_parameters.m_tolerance / 100.0) << " !< "
                          << measured << " < " << highExpected * (1.0 + m_parameters.m_tolerance / 100.0) << std::endl;
            }

            return false;
        }

        if (measured > highExpected * (1.0 + m_parameters.m_tolerance / 100.0))
        {
            if (!m_parameters.m_fast)
            {
                std::cout << "Field " << m_parameters.m_fieldId << " @ " << howFarIn * 100
                          << "% Validation Fail: " << lowExpected * (1.0 - m_parameters.m_tolerance / 100.0) << " < "
                          << measured << " !< " << highExpected * (1.0 + m_parameters.m_tolerance / 100.0) << std::endl;
            }

            return false;
        }

        return true;
    }

    if (measured < (lowExpected - m_parameters.m_tolerance))
    {
        if (!m_parameters.m_fast)
        {
            std::cout << "Field " << m_parameters.m_fieldId << " @ " << howFarIn * 100
                      << "% Validation Fail: " << lowExpected - m_parameters.m_tolerance << "!< " << measured << " < "
                      << highExpected + m_parameters.m_tolerance << std::endl;
        }

        return false;
    }

    if (measured > (highExpected + m_parameters.m_tolerance))
    {
        if (!m_parameters.m_fast)
        {
            std::cout << "Field " << m_parameters.m_fieldId << " @ " << howFarIn * 100
                      << "% Validation Fail: " << lowExpected - m_parameters.m_tolerance << "< " << measured << " !< "
                      << highExpected + m_parameters.m_tolerance << std::endl;
        }

        return false;
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

                DCGM_LOG_ERROR << "Failed to start CUDA test on " << worker->Device();
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
            DCGM_LOG_WARNING << "A test for fieldId " << m_parameters.m_fieldId << " has not been implemented yet.";

            rtSt = DCGM_ST_GENERIC_ERROR;
            break;
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
        DCGM_LOG_ERROR << "ProcessStartingResponse failed to read from a worker.";
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

            DCGM_LOG_WARNING << "Received unexpected worker status: " << c;
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
            DCGM_LOG_WARNING << "Received unexpected worker status: " << c;
            break;
    }

    if (AllFinished())
    {
        m_responseFn = &PhysicalGpu::ProcessFinishedResponse;
        rtSt         = CommandAll(true, "M\nE\n");
    }
    else if (AllStarted())
    {
        rtSt = RunTests();
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
        DCGM_LOG_ERROR << "ProcessRunningResponse called when all workers finished";

        return DCGM_ST_GENERIC_ERROR;
    }

    int retval;

    if ((retval = worker->ReadLn()) < 0)
    {
        DCGM_LOG_ERROR << "ProcessRunningResponse failed to read from a worker.";
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
            DCGM_LOG_WARNING << "Received unexpected worker status: " << c;

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
        DCGM_LOG_ERROR << "ProcessFinishedResponse called before all workers finished";

        return DCGM_ST_GENERIC_ERROR;
    }

    if (m_reportedWorkers >= m_workers)
    {
        DCGM_LOG_ERROR << "ProcessFinishedResponse called after all workers reported";

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
    // rtSt = CommandAll("M\nE\n");

    if ((retval = worker->ReadLn()) < 0)
    {
        DCGM_LOG_ERROR << "ProcessFinishedResponse failed to read from a "
                       << "worker.";
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
                    std::cout << "Worker " << m_gpuId << ":" << workerIdx << "[" << m_parameters.m_fieldId
                              << "]: " << label << ": " << worker->Input().rdbuf() << std::endl;
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
            DCGM_LOG_WARNING << "Received unexpected worker status: " << c;
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
        fprintf(stderr, "Unable to open %s (errno %d). Can't write test results file.\n", filename, cachedErrno);
        return DCGM_ST_GENERIC_ERROR;
    }

    fprintf(testResultsFp, "# TestTitle: \"%s\"\n", m_subtestTitle.c_str());

    fprintf(testResultsFp, "# Columns: \"generated\", \"dcgm\"\n");

    for (size_t i = 0; i < m_subtestDcgmValues.size(); i++)
    {
        fprintf(testResultsFp, "%.3f, %.3f\n", m_subtestGenValues[i], m_subtestDcgmValues[i]);
    }

    fprintf(testResultsFp, "# TestResult: PASSED\n"); /* Todo: check and actually change this to WARNING or FAILED */
    fprintf(testResultsFp, "# TestResultReason: \n"); /* Todo: Populate with optional text */

    if (m_parameters.m_dvsOutput)
        printf("&&&& PASSED %s\n", m_subtestTag.c_str());

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
        printf("Skipping CreateDcgmGroups() since DCGM validation is disabled\n");
        return DCGM_ST_OK;
    }

    char groupName[32] = { 0 };
    snprintf(groupName, sizeof(groupName), "dpt_%d_%u", getpid(), m_gpuId);

    unsigned short fieldId = m_parameters.m_fieldId;

    dcgmReturn = dcgmFieldGroupCreate(m_dcgmHandle, 1, &fieldId, groupName, &m_fieldGroupId);

    if (dcgmReturn != DCGM_ST_OK)
    {
        fprintf(stderr, "dcgmFieldGroupCreate() returned %d\n", dcgmReturn);
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
        fprintf(stderr, "dcgmFieldGroupDestroy() returned %d\n", dcgmReturn);
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
        bool validated { true };
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

                    std::cout << std::fixed;
                    std::cout << "GPU " << m_gpuId << ": "
                              << "Testing Maximum "
                              << "SmOccupancy/SmActivity/GrActivity "
                              << "for " << std::setprecision(3) << m_parameters.m_duration << " seconds." << std::endl;

                    std::cout << "-------------------------------------------------------------" << std::endl;

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

                std::cout << std::fixed;

                std::cout << "Worker " << m_gpuId << ":" << index << "[" << m_parameters.m_fieldId
                          << "]: SmOccupancy generated 1.0, dcgm " << std::setprecision(3);

                ValuesDump(values, ValueType::Double, 1.0);
                //<< value.value.dbl

                std::cout << " at " << std::setprecision(3) << timeOffset << " seconds. threadsPerSm " << threadsPerSm
                          << std::endl;

                std::cout.precision(ss);
                std::cout.flags(flags);
            }

            double value;

            validated = ValueGet(values, worker.Entities(), DCGM_FE_GPU_CI, ValueType::Double, 1.0, value)
                        && Validate(1.0, 1.0, value, howFarIn);

            AppendSubtestRecord(1.0, value);
        }

        return validated ? DCGM_ST_OK : DCGM_ST_PENDING;
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
        bool validated { true };
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

                        std::cout << std::fixed;

                        std::cout << "GPU " << m_gpuId << ": "
                                  << "Testing SmOccupancy scaling by "
                                  << "num threads for " << std::setprecision(3) << m_parameters.m_duration / 3.0
                                  << " seconds." << std::endl;

                        std::cout << "-------------------------------------------------------------" << std::endl;

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

                    std::cout << std::fixed;

                    std::cout << "Worker " << m_gpuId << ":" << index << "[" << m_parameters.m_fieldId
                              << "]: SmOccupancy generated " << std::setprecision(3) << prevOccupancy << "/"
                              << curOccupancy << ", dcgm ";

                    ValuesDump(values, ValueType::Double, 1.0);
                    //<< value.value.dbl

                    std::cout << " at " << std::setprecision(3) << timeOffset << " seconds. threadsPerSm "
                              << threadsPerSm << " / " << maxThreadsPerMultiProcessor << std::endl;

                    std::cout.precision(ss);
                    std::cout.flags(flags);
                }

                double value;

                validated &= ValueGet(values, worker.Entities(), DCGM_FE_GPU_CI, ValueType::Double, 1.0, value)
                             && Validate(prevOccupancy, curOccupancy, value, howFarIn);

                AppendSubtestRecord(prevOccupancy, value);
            }
        }
        else if (part == 1)
        {
            if (firstPartTick)
            {
                if (m_parameters.m_report)
                {
                    std::cout << "Worker " << m_gpuId << ":" << index << "[" << m_parameters.m_fieldId
                              << "]: Slept to let previous test fall off." << std::endl;
                }

                if (nextPart == 1)
                {
                    nextPart  = 2;
                    firstTick = false;
                }

                if (!firstTick) // First per-test part per-GPU code here.
                {
                    firstTick = true;

                    EndSubtest();

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

                        std::cout << std::fixed;

                        std::cout << "GPU " << m_gpuId << ": "
                                  << "Testing SmOccupancy scaling "
                                  << "by SM count for " << std::setprecision(3) << m_parameters.m_duration / 3.0
                                  << " seconds." << std::endl;

                        std::cout << "----------------------------------------------------------" << std::endl;

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

                    std::cout << std::fixed << std::setprecision(3);

                    std::cout << "Worker " << m_gpuId << ":" << index << "[" << m_parameters.m_fieldId
                              << "]: SmOccupancy generated " << prevOccupancy << "/" << curOccupancy << ", dcgm ";

                    ValuesDump(values, ValueType::Double, 1.0);
                    //<< value.value.dbl

                    std::cout << " at " << std::setprecision(3) << timeOffset << " seconds."
                              << " numSms " << numSms << " / " << multiProcessorCount << std::endl;

                    std::cout.precision(ss);
                    std::cout.flags(flags);
                }

                double value;

                validated &= ValueGet(values, worker.Entities(), DCGM_FE_GPU_CI, ValueType::Double, 1.0, value)
                             && Validate(prevOccupancy, curOccupancy, value, howFarIn);

                AppendSubtestRecord(prevOccupancy, value);
            }
        }
        else if (part == 2)
        {
            if (firstPartTick)
            {
                if (m_parameters.m_report)
                {
                    std::cout << "Worker " << m_gpuId << ":" << index << "[" << m_parameters.m_fieldId
                              << "]: Slept to let previous test fall off." << std::endl;
                }

                if (nextPart == 2)
                {
                    nextPart  = 3;
                    firstTick = false;
                }

                if (!firstTick) // First per-test part per-GPU code here.
                {
                    firstTick = true;

                    EndSubtest();

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

                        std::cout << std::fixed;

                        std::cout << "GPU " << m_gpuId << ": "
                                  << "Testing SmOccupancy scaling "
                                  << "by CPU sleeps  for " << std::setprecision(3) << m_parameters.m_duration / 3.0
                                  << " seconds." << std::endl;

                        std::cout << "----------------------------------------------------------" << std::endl;

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

                    std::cout << std::fixed << std::setprecision(3);

                    std::cout << "Worker " << m_gpuId << ":" << index << "[" << m_parameters.m_fieldId
                              << "]: SmOccupancy generated " << prevOccupancy << "/" << curOccupancy << ", dcgm ";

                    ValuesDump(values, ValueType::Double, 1.0);
                    //<< value.value.dbl

                    std::cout << " at " << std::setprecision(3) << timeOffset << " seconds." << std::endl;

                    std::cout.precision(ss);
                    std::cout.flags(flags);
                }

                double value;

                validated &= ValueGet(values, worker.Entities(), DCGM_FE_GPU_CI, ValueType::Double, 1.0, value)
                             && Validate(prevOccupancy, curOccupancy, value, howFarIn);

                AppendSubtestRecord(prevOccupancy, value);
            }
        }

        return validated ? DCGM_ST_OK : DCGM_ST_PENDING;
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
        bool validated { true };
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

                        std::cout << std::fixed;

                        std::cout << "GPU " << m_gpuId << ": "
                                  << "Testing SmActivity scaling by SM count "
                                  << "for " << std::setprecision(3) << m_parameters.m_duration / 2.0 << " seconds."
                                  << std::endl;

                        std::cout << "---------------------------------------------------------" << std::endl;

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

                    std::cout << std::fixed << std::setprecision(3);

                    std::cout << "Worker " << m_gpuId << ":" << index << "[" << m_parameters.m_fieldId
                              << "]: SmActivity generated " << prevSmActivity << "/" << curSmActivity << ", dcgm ";

                    ValuesDump(values, ValueType::Double, 1.0);
                    //<< value.value.dbl

                    std::cout << " at " << std::setprecision(3) << timeOffset << " seconds. numSms " << numSms << " / "
                              << multiProcessorCount << std::endl;

                    std::cout.precision(ss);
                    std::cout.flags(flags);
                }

                double value;

                validated &= ValueGet(values, worker.Entities(), DCGM_FE_GPU_CI, ValueType::Double, 1.0, value)
                             && Validate(prevSmActivity, curSmActivity, value, howFarIn);

                AppendSubtestRecord(prevSmActivity, value);
            }
        }
        else if (part == 1)
        {
            if (firstPartTick)
            {
                if (m_parameters.m_report)
                {
                    std::cout << "Worker " << m_gpuId << ":" << index << "[" << m_parameters.m_fieldId
                              << "]: Slept to let previous test fall off." << std::endl;
                }

                if (nextPart == 1)
                {
                    nextPart  = 2;
                    firstTick = false;
                }

                if (!firstTick) // First per-test part per-GPU code here.
                {
                    firstTick = true;

                    EndSubtest();

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

                        std::cout << std::fixed;

                        std::cout << "GPU " << m_gpuId << ": "
                                  << "Testing SmActivity scaling by CPU sleeps"
                                  << " for " << std::setprecision(3) << m_parameters.m_duration / 2.0 << " seconds."
                                  << std::endl;

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

                    std::cout << std::fixed << std::setprecision(3);

                    std::cout << "Worker " << m_gpuId << ":" << index << "[" << m_parameters.m_fieldId
                              << "]: SmActivity: generated " << prevSmActivity << "/" << curSmActivity << ", dcgm ";

                    ValuesDump(values, ValueType::Double, 1.0);
                    //<< value.value.dbl

                    std::cout << " at " << std::setprecision(3) << timeOffset << " seconds." << std::endl;

                    std::cout.precision(ss);
                    std::cout.flags(flags);
                }

                double value;

                validated &= ValueGet(values, worker.Entities(), DCGM_FE_GPU_CI, ValueType::Double, 1.0, value)
                             && Validate(prevSmActivity, curSmActivity, value, howFarIn);

                AppendSubtestRecord(prevSmActivity, value);
            }
        }

        return validated ? DCGM_ST_OK : DCGM_ST_PENDING;
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
        bool validated { true };
        unsigned int part;
        unsigned int parts;
        bool firstPartTick = worker.IsFirstTick();

        worker.GetParts(part, parts);

        if (firstPartTick)
        {
            if (firstTick) // First per-test per-GPU code here.
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

                std::cout << std::fixed << std::setprecision(3);

                std::cout << "Worker " << m_gpuId << ":" << index << "[" << m_parameters.m_fieldId
                          << "]: GrActivity: generated " << prevHowFarIn << "/" << curHowFarIn << ", dcgm ";

                ValuesDump(values, ValueType::Double, 1.0); //<< value.value.dbl

                std::cout << " at " << std::setprecision(3) << timeOffset << " seconds." << std::endl;

                std::cout.precision(ss);
                std::cout.flags(flags);
            }

            double value;

            validated &= ValueGet(values, worker.Entities(), DCGM_FE_GPU_CI, ValueType::Double, 1.0, value)
                         && Validate(prevHowFarIn, curHowFarIn, value, howFarIn);

            AppendSubtestRecord(howFarIn, value);
        }

        return validated ? DCGM_ST_OK : DCGM_ST_PENDING;
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

    SetTickHandler([this, firstTick = false, fieldHeading = "", subtestTag = ""](
                       size_t index,
                       bool valid,
                       std::map<Entity, dcgmFieldValue_v1> &values,
                       DistributedCudaContext &worker) mutable -> dcgmReturn_t {
        bool validated { true };
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

        if (valid)
        {
            dcgmGroupEntityPair_t entity;
            unsigned short fieldId { DCGM_FI_DEV_PCIE_LINK_GEN };
            dcgmFieldValue_v2 value2 {};

            entity.entityGroupId = DCGM_FE_GPU;
            entity.entityId      = worker.Entities()[DCGM_FE_GPU];

            dcgmReturn_t dcgmReturn
                = dcgmEntitiesGetLatestValues(m_dcgmHandle, &entity, 1, &fieldId, 1, DCGM_FV_FLAG_LIVE_DATA, &value2);

            if (dcgmReturn != DCGM_ST_OK)
            {
                fprintf(stderr,
                        "dcgmEntitiesGetLatestValues failed with %d for gpuId %u PCIE test.\n",
                        dcgmReturn,
                        m_gpuId);

                return DCGM_ST_NOT_SUPPORTED;
            }

            unsigned long long pcieVersion = value2.value.i64;

            fieldId = DCGM_FI_DEV_PCIE_LINK_WIDTH;

            dcgmReturn
                = dcgmEntitiesGetLatestValues(m_dcgmHandle, &entity, 1, &fieldId, 1, DCGM_FV_FLAG_LIVE_DATA, &value2);

            if (dcgmReturn != DCGM_ST_OK)
            {
                fprintf(stderr,
                        "dcgmEntitiesGetLatestValues failed with %d for gpuId %u PCIE test.\n",
                        dcgmReturn,
                        m_gpuId);

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

                std::cout << std::fixed << std::setprecision(0);

                std::cout << "Worker " << m_gpuId << ":" << index << "[" << m_parameters.m_fieldId
                          << "]: " << fieldHeading << " generated " << prevPerSecond << "/" << curPerSecond
                          << ", dcgm ";

                ValuesDump(values, ValueType::Int64, 1000.0 * 1000.0);
                //<< dcgmValue RSH - i64 / 10^6

                std::cout << " MiB/sec (";

                std::cout.precision(ss);
                std::cout.flags(flags);
            }

            double value;

            validated &= ValueGet(values, worker.Entities(), DCGM_FE_GPU, ValueType::Int64, 1000.0 * 1000.0, value)
                         && Validate(80.0, 90.0, 100.0 * value / expectedRate, howFarIn);

            AppendSubtestRecord(prevPerSecond, value);

            if (m_parameters.m_report)
            {
                auto ss    = std::cout.precision();
                auto flags = std::cout.flags();

                std::cout << std::fixed << std::setprecision(0);

                std::cout << 100.0 * value / expectedRate << "% speed-of-light)" << std::endl;

                std::cout.precision(ss);
                std::cout.flags(flags);
            }
        }

        return validated ? DCGM_ST_OK : DCGM_ST_PENDING;
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
        fprintf(stderr, "dcgmGetDeviceTopology failed with %d for gpuId %u.\n", dcgmReturn, m_gpuId);

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

    if (!maxNumLinks)
    {
        fprintf(stderr, "gpuId %u has no NvLink peers. Skipping test.\n", m_gpuId);
        return DCGM_ST_NOT_SUPPORTED;
    }

    dcgmDeviceAttributes_v2 peerDeviceAttr;
    memset(&peerDeviceAttr, 0, sizeof(peerDeviceAttr));
    peerDeviceAttr.version = dcgmDeviceAttributes_version2;
    dcgmReturn             = dcgmGetDeviceAttributes(m_dcgmHandle, bestGpuId, &peerDeviceAttr);
    if (dcgmReturn != DCGM_ST_OK)
    {
        fprintf(stderr, "dcgmGetDeviceAttributes failed with %d for gpuId %u.\n", dcgmReturn, bestGpuId);
        return dcgmReturn;
    }

    peerPciBusId = std::string(peerDeviceAttr.identifiers.pciBusId);
    nvLinks      = maxNumLinks;

    fprintf(stdout, "The best peer of gpuId %u is gpuId %u, numLinks %u.\n", m_gpuId, bestGpuId, nvLinks);

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t PhysicalGpu::RunSubtestNvLinkBandwidth(void)
{
    dcgmReturn_t rtSt;

    if (m_dcgmDeviceAttr.settings.migModeEnabled != 0) // MIG: can not do this.
    {
        std::cerr << "GPU " << m_gpuId << ": "
                  << "Can not run NvLink tests in MIG mode." << std::endl;

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
        bool validated { true };
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

        dcgmGroupEntityPair_t entity;
        unsigned short fieldId { DCGM_FI_DEV_CUDA_COMPUTE_CAPABILITY };
        dcgmFieldValue_v2 value2 {};

        entity.entityGroupId = DCGM_FE_GPU;
        entity.entityId      = worker.Entities()[DCGM_FE_GPU];

        dcgmReturn_t dcgmReturn
            = dcgmEntitiesGetLatestValues(m_dcgmHandle, &entity, 1, &fieldId, 1, DCGM_FV_FLAG_LIVE_DATA, &value2);

        if (dcgmReturn != DCGM_ST_OK)
        {
            fprintf(
                stderr, "dcgmEntitiesGetLatestValues failed with %d for gpuId %u NvLink test.\n", dcgmReturn, m_gpuId);

            return DCGM_ST_NOT_SUPPORTED;
        }

        unsigned int majorVersion = (value2.value.i64 & 0xffff0000) >> 16;

        unsigned int nvLinkMbPerSec { 0 };

        switch (majorVersion)
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

                std::cout << std::fixed << std::setprecision(0);

                std::cout << "Worker " << m_gpuId << ":" << index << "[" << m_parameters.m_fieldId
                          << "]: " << fieldHeading << " generated " << prevPerSecond << "/" << curPerSecond
                          << ", dcgm ";

                ValuesDump(values, ValueType::Int64, 1000.0 * 1000.0);
                // << value.value.i64 / 1000 / 1000

                std::cout << " MiB/sec (";

                std::cout.precision(ss);
                std::cout.flags(flags);
            }

            double value;

            validated &= ValueGet(values, worker.Entities(), DCGM_FE_GPU_CI, ValueType::Int64, 1000.0 * 1000.0, value)
                         && Validate(prevPerSecond, curPerSecond, value, howFarIn);

            AppendSubtestRecord(prevPerSecond, value);

            if (m_parameters.m_report)
            {
                auto ss    = std::cout.precision();
                auto flags = std::cout.flags();

                std::cout << std::fixed << std::setprecision(0);

                std::cout << 100.0 * value / nvLinkMbPerSec << "% speed-of-light)" << std::endl;

                std::cout.precision(ss);
                std::cout.flags(flags);
            }
        }

        return validated ? DCGM_ST_OK : DCGM_ST_PENDING;
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
        bool validated { true };
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
        unsigned int eccSupport;

        worker.Input() >> howFarIn;
        worker.Input() >> prevDramAct;
        worker.Input() >> curDramAct;
        worker.Input() >> prevPerSecond;
        worker.Input() >> eccSupport;
        worker.Input().ignore(MaxStreamLength, '\n');

        if (valid)
        {
            if (m_parameters.m_report)
            {
                auto ss    = std::cout.precision();
                auto flags = std::cout.flags();

                std::cout << std::fixed << std::setprecision(3);

                std::cout << "Worker " << m_gpuId << ":" << index << "[" << m_parameters.m_fieldId
                          << "]: DramUtil generated " << prevDramAct << "/" << curDramAct << ", dcgm ";


                ValuesDump(values, ValueType::Double, (eccSupport != 0) ? 9.0 / 8.0 : 1.0); //<< value.value.dbl

                std::cout << " (" << std::setprecision(1) << prevPerSecond << " GiB/sec)" << std::endl;

                std::cout.precision(ss);
                std::cout.flags(flags);
            }

            double value;

            validated &= ValueGet(values,
                                  worker.Entities(),
                                  DCGM_FE_GPU_I,
                                  ValueType::Double,
                                  (eccSupport != 0) ? 9.0 / 8.0 : 1.0,
                                  value)
                         && Validate(prevDramAct, curDramAct, value, howFarIn);

            AppendSubtestRecord(prevDramAct, value);
        }

        return validated ? DCGM_ST_OK : DCGM_ST_PENDING;
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
            fprintf(stderr, "fieldId %u is unhandled.\n", m_parameters.m_fieldId);

            return DCGM_ST_GENERIC_ERROR;
    }

    SetTickHandler([this, firstTick = false, testHeader, testTag /*, prevValue*/](
                       size_t index,
                       bool valid,
                       std::map<Entity, dcgmFieldValue_v1> &values,
                       DistributedCudaContext &worker) mutable -> dcgmReturn_t {
        bool validated { true };
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

                std::cout << std::fixed << std::setprecision(3);

                std::cout << "Worker " << m_gpuId << ":" << index << "[" << m_parameters.m_fieldId
                          << "]: " << testHeader << ": generated ???, dcgm ";

                ValuesDump(values, ValueType::Double, 1.0); //<< value.value.dbl

                std::cout << " (" << std::setprecision(1) << gflops << " gflops)" << std::endl;

                std::cout.precision(ss);
                std::cout.flags(flags);
            }

            double value;

            validated &= ValueGet(values, worker.Entities(), DCGM_FE_GPU_CI, ValueType::Double, 1.0, value)
                         && Validate(0.90, 0.90, value, howFarIn);

            ////prevValue = value;

            AppendSubtestRecord(0.0, value);
        }

        return validated ? DCGM_ST_OK : DCGM_ST_PENDING;
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
