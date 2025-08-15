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
#include <DcgmLogging.h>
#include <cuda.h>

#include "Arguments.h"
#include "DcgmCacheManager.h"
#include "DcgmProfTester.h"
#include "DcgmSettings.h"
#include "Entity.h"
#include "PhysicalGpu.h"
#include "Reporter.h"
#include "dcgm_fields.h"
#include "dcgm_fields_internal.hpp"
#include "timelib.h"
#include "vector_types.h"
#include <cublas_proxy.hpp>
#include <dcgm_agent.h>

#include <tclap/Arg.h>
#include <tclap/CmdLine.h>
#include <tclap/SwitchArg.h>
#include <tclap/ValueArg.h>
#include <tclap/ValuesConstraint.h>

#include <chrono>
#include <climits>
#include <csignal>
#include <cstdint>
#include <ctime> //RSH
#include <fmt/format.h>
#include <iomanip>  //RSH
#include <iostream> //RSH
#include <iterator>
#include <map>
#include <memory>
#include <streambuf>
#include <string>
#include <sys/types.h>
#include <system_error>
#include <unistd.h>
#include <unordered_set>
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

                case DCGM_FE_LINK:
                    info_reporter << "LI: ";
                    break;
            }
        }

        double value;

        switch (type)
        {
            case PhysicalGpu::ValueType::Int64:
                if (fieldValue.value.i64 == DCGM_INT64_BLANK)
                {
                    info_reporter << "<unknown>";
                }
                else
                {
                    value = fieldValue.value.i64;
                    value /= divisor;
                    info_reporter << value;
                }

                break;

            case PhysicalGpu::ValueType::Double:
                if (fieldValue.value.dbl == DCGM_FP64_BLANK)
                {
                    info_reporter << "<unknown>";
                }
                else
                {
                    value = fieldValue.value.dbl;
                    value /= divisor;
                    info_reporter << value;
                }

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
    : m_tester(std::move(tester))
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
    /**
     * We prospectively add a worker slice for this GPU. It may already be added
     * or it may need to be removed.
     *
     * Bacause each worker requires a CUDA context, and CUDA contexts are
     * expensive in terms of time to create, once a worker is created with one,
     * it remains added to the Physical GPU. "Adding" it again is simply a NO
     * OP. However, sometimes we only want one CPU Instance (CI) worker per
     * each GPU Instance (GI) on a physical GPU. This is the case for
     * DRAM Util tests.
     *
     * As we proceed through the tests on a GPU, it may be necessary to
     * inactivate or reactivate per-CI workers. So, AddSlice does not just
     * add (perhaps already added) workers, but reassigns them to an inactive
     * list, or removes them from it, as required.
     */

    std::shared_ptr<DistributedCudaContext> foundCudaContext { nullptr };
    std::shared_ptr<DistributedCudaContext> foundGiCudaContext { nullptr };

    dcgm_field_eid_t gi { 0 };
    dcgm_field_eid_t ci { 0 };

    /**
     * In the MIG case, we may want to restrict ourselves for one CI per GI,
     * so we get our prospective worker's GI and CI for comparison.
     */
    if (IsMIG())
    {
        gi = (*entities)[DCGM_FE_GPU_I];
        ci = (*entities)[DCGM_FE_GPU_CI];
    }

    auto cudaContextIt = m_dcgmCudaContexts.begin();
    bool firstCi { false };

    /**
     * We iterate through the active workers to see if we have alreaded added
     * this one, or at least one with the same GI. The latter matters in the
     * MIG case for DRAM Util tests, as we may only want one CI per GI.
     */
    for (; cudaContextIt != m_dcgmCudaContexts.end(); cudaContextIt++)
    {
        auto cudaContext = *cudaContextIt;

        if (cudaContext->EntityId().entityGroupId != entity.entityGroupId)
        {
            /**
             * This is not even the same kind of worker (whole GPU/slice).
             * Keep looking.
             */
            continue;
        }

        if (IsMIG() && (foundGiCudaContext == nullptr) && (cudaContext->Entities()[DCGM_FE_GPU_I] == gi))
        {
            /**
             * We found a worker with the same GI. Perhaps we don't need to add
             * a new one if we only allow one CI per GI. We have to gate on
             * IsMIG() because non-MIG workers will not have an DCGM_FE_GPU_I
             * metric entity,
             */
            foundGiCudaContext = cudaContext;

            /**
             * if the GI matches and the CI matches, we have an exact match.
             * This worker was already added.
             */
            firstCi = cudaContext->Entities()[DCGM_FE_GPU_CI] == ci;
        }

        if (cudaContext->EntityId().entityId != entity.entityId)
        {
            /**
             * It's not the same worker. Keep looking.
             */
            continue;
        }

        /**
         * We found a matching worker.
         */
        foundCudaContext = std::move(cudaContext);

        break;
    }

    if (foundCudaContext == nullptr)
    {
        /**
         * We did not find the worker already added. We consider adding it
         * unless we can not have more than one CI per GI.
         */

        if (foundGiCudaContext != nullptr)
        {
            /**
             * In the case of Dram Util tests, we only allow one CI per GI.
             *
             * Note that foundGiCudaContext can only not be nullptr in the MIG
             * case so we don't have to gate on it again.
             */
            if (m_parameters.m_fieldId == DCGM_FI_PROF_DRAM_ACTIVE)
            {
                /**
                 * We found a matching GI, but not CI, and we don't allow
                 * multiple CIs per GI for DRAM Util tests. So, this worker
                 * can be used. Note that the same worker might be used for
                 * multiple prospective CI workers but only one will actually
                 * run.
                 */
                foundGiCudaContext->ReInitialize();
                foundGiCudaContext->SetTries(m_parameters.m_syncCount);

                return foundGiCudaContext;
            }

            /**
             * We found a matching GI but not the CI, and we allow multiple CIs
             * per GI. We need to check if it is on in the suspended CI map. It
             * can get there if a previous test allowed multiple CIs per GI but
             * another one after it did not.
             */

            auto inactive_it = m_inactiveCudaContexts.find(entity_id_t(gi, ci));

            if (inactive_it != m_inactiveCudaContexts.end())
            {
                /**
                 * We found an inactive worker. Make it active again!
                 */
                foundCudaContext = inactive_it->second;
                m_inactiveCudaContexts.extract(inactive_it);
                m_dcgmCudaContexts.push_back(foundCudaContext);

                m_workers++;

                foundCudaContext->ReInitialize();
                foundCudaContext->SetTries(m_parameters.m_syncCount);

                return foundCudaContext;
            }

            /**
             * We found a worker with a matching GI, but not CI. We can
             * run multiple CIs per GI, but no such CI was previously made
             * inactive. So, we fall through and create one. This is actually
             * a newly created worker and is how all workers are initially
             * created.
             */
        }

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
        /**
         * We found a worker. We need to see if we can let it run or if it
         * matches one with the same GI and we only allow one CI per GI. In that
         * case, we have to make the found worker inactive, and return the one
         * with the same GI.
         */

        if (foundGiCudaContext != nullptr)
        {
            /**
             * In the case of Dram Util tests, we only allow one CI per GI,
             */
            if ((m_parameters.m_fieldId == DCGM_FI_PROF_DRAM_ACTIVE) && !firstCi)
            {
                m_inactiveCudaContexts.insert(std::pair(entity_id_t(gi, ci), *cudaContextIt));
                m_dcgmCudaContexts.erase(cudaContextIt);

                --m_workers;

                foundGiCudaContext->ReInitialize();
                foundGiCudaContext->SetTries(m_parameters.m_syncCount);

                return foundGiCudaContext;
            }

            /**
             * We found a matching GI and the CI, and we allow multiple CIs
             * per GI. So, we fall through and ReInitialize it.
             */
        }

        foundCudaContext->ReInitialize();
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
            if (values[entity].value.i64 == DCGM_INT64_BLANK)
            {
                return false;
            }
            value = values[entity].value.i64;
            value /= divisor;

            break;

        case ValueType::Double:
            if (values[entity].value.dbl == DCGM_FP64_BLANK)
            {
                return false;
            }
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

    if (all)
    {
        // Reset failed worker count as we are commanding all of them.
        m_failedWorkers = 0;
    }

    for (auto &worker : m_dcgmCudaContexts)
    {
        if (all && worker->Failed())
        {
            // Reset worker failure status.
            worker->ClrFailed();
        }

        if (worker->Failed())
        {
            continue;
        }

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
    if (m_dcgmCudaContexts.size() == 0)
    {
        DCGM_LOG_ERROR << "Error: no GPUs or MIG instances to run on.";

        return DCGM_ST_NO_DATA;
    }

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

    return CreateDcgmGroups();
}

// Actually run tests when all are started.
dcgmReturn_t PhysicalGpu::RunTests(void)
{
    dcgmReturn_t rtSt { DCGM_ST_OK };

    /*
    rtSt = CreateDcgmGroups();

    if (rtSt != DCGM_ST_OK)
    {
        return rtSt;
    }
    */

    if (AllFinished())
    {
        DCGM_LOG_INFO << "RunTests aborted because already finished";

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

        case DCGM_FI_PROF_PIPE_TENSOR_ACTIVE:
            rtSt = RunSubtestGemmUtil();
            break;

        case DCGM_FI_PROF_PIPE_FP64_ACTIVE:
        case DCGM_FI_PROF_PIPE_FP32_ACTIVE:
        case DCGM_FI_PROF_PIPE_FP16_ACTIVE:
            rtSt = UseCublas() ? RunSubtestGemmUtil() : RunSubtestDataTypeActive();
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

    auto rtSt = (this->*m_responseFn)(std::move(worker));

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

            if (IsSynchronous())
            {
                /*
                 * Even if the last measurement failed validation, we had one
                 * at the same activity level pass (otherwise we would not have
                 * sent the '"A'dvance message to finish), so we let it go.
                 */

                if (!m_valid)
                {
                    info_reporter << "Ignored a validation failure after last activity level success."
                                  << info_reporter.new_line;

                    m_valid = true;
                }
            }

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
                /**
                 * If an exit was requested, we either had a fast pass, in which
                 * case m_valid is already true, or we had a failure after
                 * enough retries, and m_valid is false. We do not want to reset
                 * it in the latter case.
                 *
                 * If m_valid is false, but m_exitRequested is not true, then
                 * we have recovered by retrying, and we can reset it to true.
                 *
                 * This fixes the problem of ignored failures in subsequent
                 * steps in a multi-step test where the previous step failed,
                 * we set m_exitRequested, and fell through to the next step,
                 * where we ignore the first few failures.
                 *
                 * https://nvbugswb.nvidia.com/NvBugs5/SWBug.aspx?bugid=4014448
                 */
                if (!m_exitRequested)
                {
                    m_valid = true;
                }

                if (m_parameters.m_fast) // Woot! Got a pass in fast mode!
                {
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
                }
                else
                {
                    dcgmReturn = DCGM_ST_GENERIC_ERROR;
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

                m_valid = false;

                if (dcgmReturn != DCGM_ST_PENDING)
                {
                    m_exitRequested = !m_parameters.m_fast;
                }

                if ((dcgmReturn != DCGM_ST_PENDING) && !m_exitRequested && IsSynchronous())
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
                    info_reporter << fmt::format(
                        "Worker {:d}:{:d}[{:d}]: {}: ", m_gpuId, workerIdx, m_parameters.m_fieldId, label);

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

bool PhysicalGpu::UseCublas(void) const
{
    return (m_parameters.m_cublas);
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


bool PhysicalGpu::AnyWorkerRequestFailed(void) const
{
    return m_failedWorkers > 0;
}

bool PhysicalGpu::IsValidated(void) const
{
    /*
     * Gate reported measurement validity by the last operation actually
     * succeeding.
     */

    return m_valid && !AnyWorkerRequestFailed();
}

bool PhysicalGpu::Advance(DistributedCudaContext &ignoreWorker, unsigned int activity)
{
    if (!ignoreWorker.GetValidated())
    {
        return false;
    }

    unsigned int minActivity { UINT_MAX - 1 };
    unsigned int minPart { UINT_MAX - 1 };

    unsigned int ignoreWorkerPart;
    unsigned int ignoreWorkerParts;

    unsigned int workerIdx { 0 };
    unsigned int ignoreWorkerIdx { 0 };

    ignoreWorker.GetParts(ignoreWorkerPart, ignoreWorkerParts);

    unsigned int part, parts;

    for (auto &worker : m_dcgmCudaContexts)
    {
        if (&*worker == &ignoreWorker)
        {
            break;
        }

        ignoreWorkerIdx++;
    }

    auto ignoreWorkerActivity = ignoreWorker.GetActivity();

    if (ignoreWorkerActivity > activity)
    {
        activity = ignoreWorkerActivity;
    }

    for (auto &worker : m_dcgmCudaContexts)
    {
        if (&*worker != &ignoreWorker)
        {
            worker->GetParts(part, parts);

            if (part <= minPart)
            {
                if (part < minPart)
                {
                    minPart     = part;
                    minActivity = UINT_MAX - 1;
                }

                auto workerActivity = worker->GetActivity();

                if (workerActivity < minActivity)
                {
                    minActivity = workerActivity;
                }
            }
        }

        workerIdx++;
    }

    if ((ignoreWorkerPart > minPart) || (activity > minActivity))
    {
        // We are too far ahead.

        // Don't penalize worker
        ignoreWorker.SetTries(ignoreWorker.GetTries() + 1);

        return false; // Don't advance.
    }

    ignoreWorker.SetActivity(activity + 1);

    return true;
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
    m_subtestTitle    = std::move(testTitle);
    m_subtestTag      = std::move(testTag);
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
                    info_reporter << fmt::format(
                        "GPU {:d}: Testing Maximum SmOccupancy/SmActivity/GrActivity for {:#.3} seconds.",
                        m_gpuId,
                        m_parameters.m_duration)
                                  << info_reporter.new_line;

                    info_reporter << "-------------------------------------------------------------"
                                  << info_reporter.new_line;
                }
            }
        }

        unsigned int activity;
        double howFarIn;
        double howFarInCur;
        double timeOffset;
        unsigned int threadsPerSm;

        worker.Input() >> activity;
        worker.Input() >> howFarIn;
        worker.Input() >> howFarInCur;
        worker.Input() >> timeOffset;
        worker.Input() >> threadsPerSm;
        worker.Input().ignore(MaxStreamLength, '\n');

        if (valid)
        {
            if (m_parameters.m_report)
            {
                info_reporter << fmt::format(
                    "Worker {:d}:{:d}[{:d}]: SmOccupancy generated 1.0, dcgm", m_gpuId, index, m_parameters.m_fieldId);

                ValuesDump(values, ValueType::Double, 1.0);

                info_reporter << fmt::format(" at {:#.3} seconds. threadPerSm {:d}.", timeOffset, threadsPerSm)
                              << info_reporter.new_line;
            }

            double value;

            worker.SetValidated(ValueGet(values, worker.Entities(), DCGM_FE_GPU_CI, ValueType::Double, 1.0, value)
                                && Validate(1.0, 1.0, value, howFarIn, worker.GetValidated()));

            AppendSubtestRecord(1.0, value);
        }

        return Advance(worker, activity) ? DCGM_ST_OK : DCGM_ST_PENDING;
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
        unsigned int activity { 0 };
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
                        info_reporter << fmt::format(
                            "GPU {:d}: Testing SmOccupancy scaling by num threads for {:#.3} seconds.",
                            m_gpuId,
                            m_parameters.m_duration)
                                      << info_reporter.new_line;
                        info_reporter << "-------------------------------------------------------------"
                                      << info_reporter.new_line;
                    }
                }
            }

            double howFarIn;
            double prevOccupancy;
            double curOccupancy;
            double timeOffset;
            unsigned int threadsPerSm;
            unsigned int maxThreadsPerMultiProcessor;

            worker.Input() >> activity;
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
                    info_reporter << fmt::format("Worker {:d}:{:d}[{:d}]: SmOccupancy generated {:#.3}/{:#.3}, dcgm ",
                                                 m_gpuId,
                                                 index,
                                                 m_parameters.m_fieldId,
                                                 prevOccupancy,
                                                 curOccupancy);

                    ValuesDump(values, ValueType::Double, 1.0);

                    info_reporter << fmt::format(" at {:#.3} seconds. threadsPerSm {:d} / {:d}.",
                                                 timeOffset,
                                                 threadsPerSm,
                                                 maxThreadsPerMultiProcessor)
                                  << info_reporter.new_line;
                }

                double value;

                worker.SetValidated(
                    ValueGet(values, worker.Entities(), DCGM_FE_GPU /*_CI*/, ValueType::Double, 1.0, value)
                    && Validate(prevOccupancy, curOccupancy, value, howFarIn, worker.GetValidated()));

                AppendSubtestRecord(prevOccupancy, value);
            }
        }

        return Advance(worker, activity) ? DCGM_ST_OK : DCGM_ST_PENDING;
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
        unsigned int activity { 0 };
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
                        info_reporter << fmt::format(
                            "GPU {:d}: Testing SmActivity scaling by SM count  for {:#.3} seconds.",
                            m_gpuId,
                            m_parameters.m_duration)
                                      << info_reporter.new_line;

                        info_reporter << "---------------------------------------------------------"
                                      << info_reporter.new_line;
                    }
                }
            }

            double howFarIn;
            double prevSmActivity;
            double curSmActivity;
            double timeOffset;
            unsigned int numSms;
            unsigned int multiProcessorCount;

            worker.Input() >> activity;
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
                    info_reporter << fmt::format("Worker {:d}:{:d}[{:d}]: SmActivity generated {:#.3}/{:#.3}, dcgm ",
                                                 m_gpuId,
                                                 index,
                                                 m_parameters.m_fieldId,
                                                 prevSmActivity,
                                                 curSmActivity);

                    ValuesDump(values, ValueType::Double, 1.0);

                    info_reporter << fmt::format(
                        " at {:#.3} seconds. numSms {:d} / {:d}.", timeOffset, numSms, multiProcessorCount)
                                  << info_reporter.new_line;
                }

                double value;

                worker.SetValidated(ValueGet(values, worker.Entities(), DCGM_FE_GPU, ValueType::Double, 1.0, value)
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
                        info_reporter << fmt::format("Worker {:d}:{:d}[{:d}]: Slept to let previous test fall off.",
                                                     m_gpuId,
                                                     index,
                                                     m_parameters.m_fieldId)
                                      << info_reporter.new_line;
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
                        info_reporter << fmt::format(
                            "GPU {:d}: Testing SmActivity scaling by CPU sleeps for {:#.3} seconds",
                            m_gpuId,
                            m_parameters.m_duration / 2.0)
                                      << info_reporter.new_line;
                    }
                }
            }

            double howFarIn;
            double prevSmActivity;
            double curSmActivity;
            double timeOffset;

            worker.Input() >> activity;
            worker.Input() >> howFarIn;
            worker.Input() >> prevSmActivity;
            worker.Input() >> curSmActivity;
            worker.Input() >> timeOffset;
            worker.Input().ignore(MaxStreamLength, '\n');

            if (valid)
            {
                if (m_parameters.m_report)
                {
                    info_reporter << fmt::format("Worker {:d}:{:d}[{:d}]: SmActivity generated {:#.3}/{:#.3}, dcgm ",
                                                 m_gpuId,
                                                 index,
                                                 m_parameters.m_fieldId,
                                                 prevSmActivity,
                                                 curSmActivity);

                    ValuesDump(values, ValueType::Double, 1.0);

                    info_reporter << fmt::format(" at {:#.3} seconds.", timeOffset) << info_reporter.new_line;
                }

                double value;

                worker.SetValidated(ValueGet(values, worker.Entities(), DCGM_FE_GPU, ValueType::Double, 1.0, value)
                                    && Validate(prevSmActivity, curSmActivity, value, howFarIn, worker.GetValidated()));

                AppendSubtestRecord(prevSmActivity, value);
            }
        }

        return Advance(worker, activity) ? DCGM_ST_OK : DCGM_ST_PENDING;
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
        unsigned int activity { 0 };
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

        worker.Input() >> activity;
        worker.Input() >> howFarIn;
        worker.Input() >> prevHowFarIn;
        worker.Input() >> curHowFarIn;
        worker.Input() >> timeOffset;
        worker.Input().ignore(MaxStreamLength, '\n');

        if (valid)
        {
            if (m_parameters.m_report)
            {
                info_reporter << fmt::format("Worker {:d}:{:d} [{:d}]: GrActivity: generated {:#.3}/{:#.3}, dcgm ",
                                             m_gpuId,
                                             index,
                                             m_parameters.m_fieldId,
                                             prevHowFarIn,
                                             curHowFarIn);

                ValuesDump(values, ValueType::Double, 1.0); //<< value.value.dbl

                info_reporter << fmt::format(" at {:#.3} seconds. {:d}", timeOffset, activity)
                              << info_reporter.new_line;
            }

            double value;

            worker.SetValidated(ValueGet(values, worker.Entities(), DCGM_FE_GPU_CI, ValueType::Double, 1.0, value)
                                && Validate(prevHowFarIn, curHowFarIn, value, howFarIn, worker.GetValidated()));

            AppendSubtestRecord(howFarIn, value);
        }

        return Advance(worker, activity) ? DCGM_ST_OK : DCGM_ST_PENDING;
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
            unsigned int activity { 0 };
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

            worker.Input() >> activity;
            worker.Input() >> howFarIn;
            worker.Input() >> prevPerSecond;
            worker.Input() >> curPerSecond;
            worker.Input().ignore(MaxStreamLength, '\n');

            /*
             * We need to compare across the whole GPU, so we update whole-GPU
             * activity, subtracting the previous activity, and adding the
             * currentactivity generated.
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
                    info_reporter << fmt::format(
                        "Worker {:d}:{:d}[{:d}]: {} generated {:#.3}/{:#.3} ({:#.3}/{:#.3}). dcgm ",
                        m_gpuId,
                        index,
                        m_parameters.m_fieldId,
                        fieldHeading,
                        prevGpuPerSecond,
                        curGpuPerSecond,
                        prevPerSecond,
                        curPerSecond);

                    ValuesDump(values, ValueType::Int64, 1000.0 * 1000.0);

                    info_reporter << " MiB/sec (";
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
                    info_reporter << fmt::format("{:#.3}% speed-of-light), PCIE version/lanes: {:d}/{:d}.",
                                                 100.0 * value / expectedRate,
                                                 pcieVersion,
                                                 pcieLanes)
                                  << info_reporter.new_line;
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

            return Advance(worker, activity) ? DCGM_ST_OK : DCGM_ST_PENDING;
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
dcgmReturn_t PhysicalGpu::HelperGetCudaVisibleGPUs(std::string &cudaVisibleGPUs, dcgmGroupEntityPair_t &entity)
{
    int i;

    dcgmDeviceTopology_v1 deviceTopo {
        .version = dcgmDeviceTopology_version1, .cpuAffinityMask = {}, .numGpus = 0, .gpuPaths = {}
    };

    if (entity.entityGroupId == DCGM_FE_GPU) // A physical GPU
    {
        dcgmReturn_t dcgmReturn = dcgmGetDeviceTopology(m_dcgmHandle, m_gpuId, &deviceTopo);

        if ((dcgmReturn != DCGM_ST_OK) && (dcgmReturn != DCGM_ST_NOT_SUPPORTED))
        {
            DCGM_LOG_ERROR << "dcgmGetDeviceTopology failed with " << dcgmReturn << " for gpuId " << m_gpuId << ".";

            return dcgmReturn;
        }
    }
    else if (entity.entityGroupId != DCGM_FE_GPU_CI) // Not a MIG slice either
    {
        DCGM_LOG_ERROR << "HelperGetCudaVisibleGPUs called with non-GPU or non-FI Entity Group Id for gpuId " << m_gpuId
                       << ".";

        return DCGM_ST_BADPARAM;
    }

    // Iterate through NvLink-connected GPUs.

    cudaVisibleGPUs = "";

    std::string separator = "";

    /**
     * The first iteration through the loop gets the device ID for the device
     * or MIG slice itself. Subsequent iterations get the device ID for
     * reachable NvLink devices in the non-MIG case.
     */

    for (i = -1; i < (int)deviceTopo.numGpus; i++)
    {
        dcgmGroupEntityPair_t tmpEntity { entity.entityGroupId,
                                          i < 0 ? entity.entityId : deviceTopo.gpuPaths[i].gpuId };

        unsigned short fieldId { DCGM_FI_DEV_CUDA_VISIBLE_DEVICES_STR };

        dcgmFieldValue_v2 value {};

        dcgmReturn_t dcgmReturn
            = dcgmEntitiesGetLatestValues(m_dcgmHandle, &tmpEntity, 1, &fieldId, 1, DCGM_FV_FLAG_LIVE_DATA, &value);

        if (dcgmReturn != DCGM_ST_OK || value.status != DCGM_ST_OK)
        {
            DCGM_LOG_ERROR << "Could not map Entity ID [" << tmpEntity.entityGroupId << "," << tmpEntity.entityId
                           << "] to prospective CUDA_VISIBLE_DEVICES environment variable (" << (int)dcgmReturn
                           << "), status: " << value.status;

            return DCGM_ST_GENERIC_ERROR;
        }

        if ((tmpEntity.entityGroupId != DCGM_FE_GPU_CI) && (strncmp(value.value.str, "MIG", 3) == 0))
        {
            DCGM_LOG_INFO << "Entity ID [" << entity.entityGroupId << "," << entity.entityId << "] maps to MIG GPU "
                          << value.value.str << "; ignored for CUDA_VISIBLE_DEVICES";

            continue;
        }

        cudaVisibleGPUs += separator + value.value.str;
        separator = ",";
    }

    return DCGM_ST_OK;
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
                      | DCGM_TOPOLOGY_NVLINK9 | DCGM_TOPOLOGY_NVLINK10 | DCGM_TOPOLOGY_NVLINK11 | DCGM_TOPOLOGY_NVLINK12
                      | DCGM_TOPOLOGY_NVLINK13 | DCGM_TOPOLOGY_NVLINK14 | DCGM_TOPOLOGY_NVLINK15
                      | DCGM_TOPOLOGY_NVLINK16 | DCGM_TOPOLOGY_NVLINK17 | DCGM_TOPOLOGY_NVLINK18;

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

    dcgmDeviceAttributes_t peerDeviceAttr;
    memset(&peerDeviceAttr, 0, sizeof(peerDeviceAttr));
    peerDeviceAttr.version = dcgmDeviceAttributes_version;
    dcgmReturn             = dcgmGetDeviceAttributes(m_dcgmHandle, bestGpuId, &peerDeviceAttr);
    if (dcgmReturn != DCGM_ST_OK)
    {
        error_reporter << "dcgmGetDeviceAttributes failed with " << dcgmReturn << " for gpuId " << bestGpuId << "."
                       << error_reporter.new_line;

        return dcgmReturn;
    }

    peerPciBusId = std::string(peerDeviceAttr.identifiers.pciBusId);
    nvLinks      = maxNumLinks;

    DCGM_LOG_INFO << "The best peer of gpuId " << m_gpuId << " is gpuId " << bestGpuId << ", numLinks " << nvLinks
                  << ".";

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
        unsigned int activity { 0 };
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
            case 9: // HOPPER
                nvLinkMbPerSec = 50000;
                break;

            case 10: // BLACKWELL
                nvLinkMbPerSec = 100000;
                break;

            default:
                break;
        }

        nvLinkMbPerSec *= nvLinks;

        double howFarIn;
        double prevPerSecond;
        double curPerSecond;

        worker.Input() >> activity;
        worker.Input() >> howFarIn;
        worker.Input() >> prevPerSecond;
        worker.Input() >> curPerSecond;
        worker.Input().ignore(MaxStreamLength, '\n');

        if (valid)
        {
            if (m_parameters.m_report)
            {
                info_reporter << fmt::format("Worker {:d}:{:d}[{:d}]: {} generated {:#.3}/{:#.3}, dcgm ",
                                             m_gpuId,
                                             index,
                                             m_parameters.m_fieldId,
                                             fieldHeading,
                                             prevPerSecond,
                                             curPerSecond);

                ValuesDump(values, ValueType::Int64, 1000.0 * 1000.0);
                // << value.value.i64 / 1000 / 1000

                info_reporter << " MiB/sec (";
            }

            double value;
            auto validated
                = ValueGet(values, worker.Entities(), DCGM_FE_GPU_CI, ValueType::Int64, 1000.0 * 1000.0, value);

            AppendSubtestRecord(prevPerSecond, value);

            if (m_parameters.m_report)
            {
                info_reporter << fmt::format("{:#.3}% speed-of-light).", 100.0 * value / nvLinkMbPerSec)
                              << info_reporter.new_line;
            }

            worker.SetValidated(validated
                                && Validate(prevPerSecond, curPerSecond, value, howFarIn, worker.GetValidated()));
        }

        return Advance(worker, activity) ? DCGM_ST_OK : DCGM_ST_PENDING;
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

    /* Track CI bandwidth for all CIs under a GI. */
    class GiDramBandwidth_t
    {
    private:
        double m_maxGiDramBandwidth { 0.0 };
        double m_giDramBandwidth { 0.0 };
        double m_lastActivity { 0.0 };

    public:
        GiDramBandwidth_t()  = default;
        ~GiDramBandwidth_t() = default;

        /**
         * This updates the running per-GI theoretical DRAM bandwith, updates
         * the measured GI bandwidth from incremental CI bandwith changes, and
         * returns the fraction of CI bandwith per maximum bandwidth
         * to compare against measured DRAM bandwidth stress.
         *
         * Clearly, until each CI is "seen" at least once, this will not give
         * correct values, but it will completely converge to the correct value
         * after all CI bandwiths have been reported once.
         *
         * While this handles the general case of multiple CIs generating DRAM
         * activity, in practice, for this test, only one CI is active.
         */
        double UpdateCiBandwidth(double maxCiBandwidth, double prevCiBandwidth, double curCiBandwidth)
        {
            m_maxGiDramBandwidth = maxCiBandwidth;
            m_giDramBandwidth -= prevCiBandwidth;
            m_giDramBandwidth += curCiBandwidth;

            m_lastActivity = m_giDramBandwidth / m_maxGiDramBandwidth;

            return m_lastActivity;
        }

        double GetLastActivity(void) const
        {
            return m_lastActivity;
        }
    };

    std::map<dcgm_field_eid_t, GiDramBandwidth_t> giActivity;

    SetTickHandler([this, firstTick = false, giActivity](size_t index,
                                                         bool valid,
                                                         std::map<Entity, dcgmFieldValue_v1> &values,
                                                         DistributedCudaContext &worker) mutable -> dcgmReturn_t {
        unsigned int part;
        unsigned int parts;
        unsigned int activity { 0 };
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
        double maxCiBandwidth;

        worker.Input() >> activity;
        worker.Input() >> howFarIn;
        worker.Input() >> prevDramAct;
        worker.Input() >> curDramAct;
        worker.Input() >> prevPerSecond;
        worker.Input() >> maxCiBandwidth;
        worker.Input() >> eccAffectsBandwidth;
        worker.Input().ignore(MaxStreamLength, '\n');

        double bandwidthDivisor = 1.0;

        if (eccAffectsBandwidth != 0)
        {
            bandwidthDivisor = 9.0 / 8.0; /* 1 parity bit per 8 bits = 9/8ths expected bandwidth */
        }

        if (valid)
        {
            double prevGpuDramAct = prevDramAct;
            double curGpuDramAct  = curDramAct;

            dcgm_field_eid_t gi { 0 };
            dcgm_field_eid_t ci { 0 };

            /**
             * In the MIG case, we can only measure per GI but we might generate
             * activity per CI, so we aggregate all the reported CI activity
             * per GI. We do this by subtracting the LAST reported activity
             * for a CI from the GI and adding the CURRENT reported activity
             * to the GI. We handle the boundary case of the first reported
             * activity by presuming both the initial running total and the last
             * reported activity are 0.0.
             *
             * Normally, we will only be configured to drive one CI per GI
             * for DRAMUtil and, this can not be changed without a code change.
             * But, we handle the general case here none the less.
             */

            if (IsMIG())
            {
                gi = worker.Entities()[DCGM_FE_GPU_I];
                ci = worker.Entities()[DCGM_FE_GPU_CI];

                double curCiBandwidth  = maxCiBandwidth * curDramAct;
                double prevCiBandwidth = maxCiBandwidth * prevDramAct;

                prevGpuDramAct = giActivity[gi].GetLastActivity();
                curGpuDramAct  = giActivity[gi].UpdateCiBandwidth(maxCiBandwidth, prevCiBandwidth, curCiBandwidth);
            }

            if (m_parameters.m_report)
            {
                info_reporter << fmt::format("Worker {:d}", m_gpuId);

                if (IsMIG())
                {
                    info_reporter << fmt::format("/{:d}/{:d}", gi, ci);
                }

                info_reporter << fmt::format(":{:d}[{:d}]: DramUtil generated {:#.3}/{:#.3}, dcgm ",
                                             index,
                                             m_parameters.m_fieldId,
                                             prevGpuDramAct,
                                             curGpuDramAct);


                ValuesDump(values, ValueType::Double, bandwidthDivisor); //<< value.value.dbl

                info_reporter << fmt::format(
                    " ({:#.3} GiB/sec). bandwidthDivisor {:#.3}", prevPerSecond, bandwidthDivisor)
                              << info_reporter.new_line;
            }

            double value;

            worker.SetValidated(
                ValueGet(values, worker.Entities(), DCGM_FE_GPU_I, ValueType::Double, bandwidthDivisor, value)
                && Validate(prevGpuDramAct, curGpuDramAct, value, howFarIn, worker.GetValidated()));

            AppendSubtestRecord(prevDramAct, value);
        }

        return Advance(worker, activity) ? DCGM_ST_OK : DCGM_ST_PENDING;
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
dcgmReturn_t PhysicalGpu::RunSubtestDataTypeActive(void)
{
    dcgmReturn_t rtSt;

    const char *testHeader;
    const char *testTag;

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
            unsigned int activity { 0 };
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
            double target;
            double target2;

            worker.Input() >> activity;
            worker.Input() >> howFarIn;
            worker.Input() >> target;
            worker.Input() >> target2;
            worker.Input().ignore(MaxStreamLength, '\n');

            if (valid)
            {
                if (m_parameters.m_report)
                {
                    info_reporter << fmt::format(
                        "Worker {:d}:{:d}[{:d}]: {}: target ", m_gpuId, index, m_parameters.m_fieldId, testHeader);

                    if (target == 1.0)
                    {
                        info_reporter << "max";
                    }
                    else
                    {
                        info_reporter << fmt::format("{:#.3}", target);
                    }

                    info_reporter << ", dcgm ";

                    ValuesDump(values, ValueType::Double, 1.0); //<< value.value.dbl

                    info_reporter << info_reporter.new_line;
                }

                double value;

                /**
                 * We compute a hard maximum permitted value limited to 1.0.
                 */
                double maxValue = 1.0;

                if (m_parameters.m_percentTolerance)
                {
                    maxValue /= (1.0 + m_parameters.m_tolerance / 100.0);
                }
                else
                {
                    maxValue -= m_parameters.m_tolerance;
                }

                worker.SetValidated(ValueGet(values, worker.Entities(), DCGM_FE_GPU_CI, ValueType::Double, 1.0, value)
                                    && Validate(limit, maxValue, value, howFarIn, worker.GetValidated()));

                AppendSubtestRecord(0.0, value);
            }

            return Advance(worker, activity) ? DCGM_ST_OK : DCGM_ST_PENDING;
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
            unsigned int activity { 0 };
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

            worker.Input() >> activity;
            worker.Input() >> howFarIn;
            worker.Input() >> gflops;
            worker.Input() >> gflops2;
            worker.Input().ignore(MaxStreamLength, '\n');

            if (valid)
            {
                if (m_parameters.m_report)
                {
                    info_reporter << fmt::format("Worker {:d}:{:d} [{:d}]: {}:  generated ???, dcgm ",
                                                 m_gpuId,
                                                 index,
                                                 m_parameters.m_fieldId,
                                                 testHeader);

                    ValuesDump(values, ValueType::Double, 1.0);

                    info_reporter << fmt::format(" ({:#.3} gflops).", gflops) << info_reporter.new_line;
                }

                double value;

                worker.SetValidated(ValueGet(values, worker.Entities(), DCGM_FE_GPU_CI, ValueType::Double, 1.0, value)
                                    && Validate(limit, 0.9, value, howFarIn, worker.GetValidated()));

                AppendSubtestRecord(0.0, value);
            }

            return Advance(worker, activity) ? DCGM_ST_OK : DCGM_ST_PENDING;
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
