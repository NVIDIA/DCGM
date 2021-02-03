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
#include "DcgmProfTester.h"
#include "Arguments.h"
#include "DcgmLogging.h"
#include "DcgmSettings.h"
#include "DistributedCudaContext.h"
#include "Entity.h"
#include "PhysicalGpu.h"
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
#include <cmath>
#include <csignal>
#include <cstdint>
#include <iterator>
#include <map>
#include <memory>
#include <string>
#include <sys/types.h>
#include <system_error>
#include <unistd.h>
#include <vector>

using namespace DcgmNs::ProfTester;

/*****************************************************************************/
/* ctor/dtor */
DcgmProfTester::DcgmProfTester()
    : m_argumentSet("dcgmproftester", DCGMPROFTESTER_VERSION)
{
    m_duration          = 30.0; // Total for all equally long parts, in seconds.
    m_reportingInterval = 1.0;  // Report this frequently, in seconds.
    m_targetMaxValue    = false;
    m_dcgmIsInitialized = false;
    m_dcgmHandle        = (dcgmHandle_t) nullptr;
    m_groupId           = (dcgmGpuGrp_t) nullptr;
    m_fieldGroupId      = (dcgmFieldGrp_t) nullptr;
    m_testFieldId       = DCGM_FI_PROF_SM_ACTIVE;
    m_sinceTimestamp    = 0;
    m_startDcgm         = true;
    m_dvsOutput         = false;

    FD_ZERO(&m_parentReadFds);
}

/*****************************************************************************/
DcgmProfTester::~DcgmProfTester()
{
    if (m_dcgmIsInitialized)
    {
        dcgmShutdown();
    }
}

/*****************************************************************************/
dcgmReturn_t DcgmProfTester::ParseCommandLine(int argc, char *argv[])
{
    // m_argumentSet.AddDefault(DCGM_FI_PROF_GR_ENGINE_ACTIVE, ValueRange_t(0.50, 1.0));
    // m_argumentSet.AddDefault(DCGM_FI_PROF_SM_ACTIVE, ValueRange_t(0.50, 1.0));
    // m_argumentSet.AddDefault(DCGM_FI_PROF_SM_OCCUPANCY, ValueRange_t(0.50, 1.0));
    // m_argumentSet.AddDefault(DCGM_FI_PROF_PIPE_TENSOR_ACTIVE, ValueRange_t(0.50, 1.0));
    // m_argumentSet.AddDefault(DCGM_FI_PROF_DRAM_ACTIVE, ValueRange_t(485.0, 500.0));
    m_argumentSet.AddDefault(DCGM_FI_PROF_PIPE_FP64_ACTIVE, ValueRange_t(0.50, 1.0));
    m_argumentSet.AddDefault(DCGM_FI_PROF_PIPE_FP32_ACTIVE, ValueRange_t(0.50, 1.0));
    m_argumentSet.AddDefault(DCGM_FI_PROF_PIPE_FP16_ACTIVE, ValueRange_t(0.50, 1.0));
    // m_argumentSet.AddDefault(DCGM_FI_PROF_PCIE_TX_BYTES, ValueRange_t(0.0, 150000000.0));
    // m_argumentSet.AddDefault(DCGM_FI_PROF_PCIE_RX_BYTES, ValueRange_t(0.0, 150000000.0));

    return m_argumentSet.Parse(argc, argv);
}

/*****************************************************************************/
dcgmReturn_t DcgmProfTester::CreateDcgmGroups(short unsigned int fieldId)
{
    dcgmReturn_t dcgmReturn = DCGM_ST_OK;

    if (!m_startDcgm)
    {
        printf("Skipping CreateDcgmGroups() since DCGM validation is disabled\n");
        return DCGM_ST_OK;
    }

    char groupName[32] = { 0 };
    snprintf(groupName, sizeof(groupName), "dpt_%d", getpid());

    dcgmReturn = dcgmGroupCreate(m_dcgmHandle, DCGM_GROUP_EMPTY, groupName, &m_groupId);

    if (dcgmReturn != DCGM_ST_OK)
    {
        fprintf(stderr, "dcgmGroupCreate() returned %d\n", dcgmReturn);
        return dcgmReturn;
    }

    for (auto &[gpuId, _] : m_gpus)
    {
        dcgmReturn = dcgmGroupAddEntity(m_dcgmHandle, m_groupId, DCGM_FE_GPU, gpuId);

        if (dcgmReturn != DCGM_ST_OK)
        {
            fprintf(stderr, "dcgmGroupAddEntity() returned %d\n", dcgmReturn);
            return dcgmReturn;
        }
    }

    /* Note: using groupName again on purpose since field groups and GPU groups are keyed separately */
    dcgmReturn = dcgmFieldGroupCreate(m_dcgmHandle, 1, &fieldId, groupName, &m_fieldGroupId);
    if (dcgmReturn != DCGM_ST_OK)
    {
        fprintf(stderr, "dcgmFieldGroupCreate() returned %d\n", dcgmReturn);
        return dcgmReturn;
    }

    return dcgmReturn;
}

dcgmReturn_t DcgmProfTester::DestroyDcgmGroups(void)
{
    dcgmReturn_t dcgmReturn = DCGM_ST_OK;

    if (!m_startDcgm)
    {
        return DCGM_ST_OK;
    }

    dcgmReturn = dcgmGroupDestroy(m_dcgmHandle, m_groupId);

    if (dcgmReturn != DCGM_ST_OK)
    {
        fprintf(stderr, "dcgmGroupDestroy() returned %d\n", dcgmReturn);
        return dcgmReturn;
    }

    dcgmReturn = dcgmFieldGroupDestroy(m_dcgmHandle, m_fieldGroupId);
    if (dcgmReturn != DCGM_ST_OK)
    {
        fprintf(stderr, "dcgmFieldGroupDestroy() returned %d\n", dcgmReturn);
        return dcgmReturn;
    }

    return dcgmReturn;
}

/*****************************************************************************/
dcgmReturn_t DcgmProfTester::WatchFields(long long updateIntervalUsec, double maxKeepAge, unsigned int testFieldId)
{
    dcgmReturn_t dcgmReturn = DCGM_ST_OK;

    int maxKeepSamples = 0; /* Use maxKeepAge instead */

    if (!m_startDcgm)
    {
        printf("Skipping WatchFields() since DCGM validation is disabled\n");
        return DCGM_ST_OK;
    }

    dcgmReturn
        = dcgmWatchFields(m_dcgmHandle, m_groupId, m_fieldGroupId, updateIntervalUsec, maxKeepAge, maxKeepSamples);
    if (dcgmReturn == DCGM_ST_REQUIRES_ROOT)
    {
        fprintf(stderr, "Profiling requires running as root.\n");
    }
    else if (dcgmReturn == DCGM_ST_PROFILING_NOT_SUPPORTED)
    {
        fprintf(stderr, "Profiling is not supported.\n");
    }
    else if (dcgmReturn == DCGM_ST_INSUFFICIENT_DRIVER_VERSION)
    {
        fprintf(stderr,
                "Either your driver is older than 418.75 (TRD3) or you "
                "are not running dcgmproftester as root.\n");
    }
    else if (dcgmReturn == DCGM_ST_IN_USE)
    {
        fprintf(stderr,
                "Another process is already using the profiling infrastucture. "
                "If nv-hostengine is running on your box, please kill it before running "
                "dcgmproftester or use the --no-dcgm-validation option to only generate a workload\n");
    }
    else if (dcgmReturn == DCGM_ST_NOT_SUPPORTED)
    {
        fprintf(stderr, "Field ID %u is is not supported for your GPU.\n", testFieldId);
    }
    else if (dcgmReturn == DCGM_ST_GROUP_INCOMPATIBLE)
    {
        fprintf(
            stderr,
            "dcgmproftester can only test homogeneous GPUs. Please use -i to pass a list of GPUs that are the same SKU.\n");
    }
    else if (dcgmReturn != DCGM_ST_OK)
    {
        fprintf(stderr, "dcgmWatchFields() returned %d\n", dcgmReturn);
    }

    return dcgmReturn;
}

/*****************************************************************************/
dcgmReturn_t DcgmProfTester::UnwatchFields(void)
{
    dcgmReturn_t dcgmReturn;

    if (!m_startDcgm)
    {
        printf("Skipping UnwatchFields() since DCGM validation is disabled\n");
        return DCGM_ST_OK;
    }

    dcgmReturn = dcgmUnwatchFields(m_dcgmHandle, m_groupId, m_fieldGroupId);
    if (dcgmReturn != DCGM_ST_OK)
    {
        fprintf(stderr, "dcgmUnwatchFields() returned %d\n", dcgmReturn);
        return dcgmReturn;
    }

    return dcgmReturn;
}


/*****************************************************************************/
dcgmReturn_t DcgmProfTester::DcgmInit(void)
{
    dcgmReturn_t dcgmReturn = dcgmInit();
    if (dcgmReturn != DCGM_ST_OK)
    {
        fprintf(stderr, "dcgmInit() returned %d\n", dcgmReturn);
        return dcgmReturn;
    }

    dcgmReturn = dcgmStartEmbedded(DCGM_OPERATION_MODE_AUTO, &m_dcgmHandle);
    if (dcgmReturn != DCGM_ST_OK)
    {
        fprintf(stderr, "dcgmStartEmbedded() returned %d\n", dcgmReturn);
    }

    m_dcgmIsInitialized = true;

    return dcgmReturn;
}

/*****************************************************************************/

/**
 * Here, we initialize the list of physical GPUs, and set up Watch Groups
 * as necessary.
 */
dcgmReturn_t DcgmProfTester::InitializeGpus(const Arguments_t &arguments)
{
    /**
     * Here, we add nullptr placeholders for any new GPU IDs we want to test.
     */
    for (auto gpuId : arguments.m_gpuIds)
    {
        m_gpus.insert({ gpuId, nullptr });
    }

    /**
     * Here we get a list of all the GPUs in the system. We do this so we can
     * exclude GPUs from the argument that don't actually exist.
     */

    int count = 0;
    unsigned int gpuIdList[DCGM_MAX_NUM_DEVICES];
    dcgmReturn_t dcgmReturn = dcgmGetAllDevices(m_dcgmHandle, gpuIdList, &count);

    if (dcgmReturn != DCGM_ST_OK)
    {
        fprintf(stderr, "dcgmGetAllDevices() returned %d\n", dcgmReturn);
        return dcgmReturn;
    }

    if (count < 1)
    {
        fprintf(stderr, "DCGM found 0 GPUs. There's nothing to test on.\n");
        return DCGM_ST_GENERIC_ERROR;
    }

    /**
     * If we have not listed ANY GPUs on the command line, we test on all
     * of them.
     */

    bool allGpus = (arguments.m_gpuIds.size() == 0);

    for (int i = 0; i < count; i++)
    {
        unsigned int gpuId = gpuIdList[i];

        /**
         * Add a GPU to our list if it isn't already on it.
         */
        auto it = allGpus ? m_gpus.insert({ gpuId, nullptr }).first : m_gpus.find(gpuId);

        if (it != m_gpus.end())
        {
            /**
             * If the GPU was newly added to our list, we need to create
             * a PhysicalGpu object to track it.
             */
            if (it->second == nullptr) // Initialized for the first time.
            {
                it->second = std::make_shared<PhysicalGpu>(shared_from_this(), gpuIdList[i], arguments.m_parameters);

                dcgmReturn_t retVal = it->second->Init(m_dcgmHandle);

                if (retVal != DCGM_ST_OK)
                {
                    std::cout << "GPU " << gpuId << " could not be initialized. Returns: " << retVal << std::endl;
                    it->second = nullptr; // Don't try and test this one.
                }
            }
            else
            {
                /**
                 * We already initialized this GPU and ran a test on it.
                 * It should be ready for another test.
                 *
                 * TODO: need to decide when to remove GPUs.
                 */
                it->second->SetParameters(arguments.m_parameters);
            }
        }
    }

    // Delete m_gpu entries that were not found or are no longer specified.
    auto it = m_gpus.cbegin();

    while (it != m_gpus.end())
    {
        /**
         * Check if the GPU specified actually exists. If so, a PhysicalGpu
         * will be present.
         */
        if (it->second == nullptr)
        {
            std::cout << "GPU " << it->first << " does not exist or can't be tested." << std::endl;
            it = m_gpus.erase(it);

            continue;
        }

        if (!allGpus)
        {
            /**
             * Check if the GPU specified was indicated on the command line.
             * If not, it will be removed.
             */
            bool deletedGpu { true };

            for (auto gpuId : arguments.m_gpuIds)
            {
                if (it->first == gpuId)
                {
                    deletedGpu = false;

                    break;
                }
            }

            if (deletedGpu)
            {
                it = m_gpus.erase(it);
            }
        }

        it++;
    }

    dcgmReturn = InitializeGpuInstances();

    if (dcgmReturn != DCGM_ST_OK)
        return dcgmReturn;

    dcgmReturn = CreateDcgmGroups(arguments.m_parameters.m_fieldId);
    if (dcgmReturn != DCGM_ST_OK)
        return dcgmReturn;

    return DCGM_ST_OK;
}

/*****************************************************************************/

/**
 * Here, we shutdown the list of physical GPUs, and clear up Watch Groups
 * as necessary.
 */
dcgmReturn_t DcgmProfTester::ShutdownGpus(void)
{
    dcgmReturn_t dcgmReturn;

    dcgmReturn = ShutdownGpuInstances();
    if (dcgmReturn != DCGM_ST_OK)
    {
        return dcgmReturn;
    }

    dcgmReturn = DestroyDcgmGroups();
    if (dcgmReturn != DCGM_ST_OK)
    {
        return dcgmReturn;
    }

    return DCGM_ST_OK;
}


/*****************************************************************************/
dcgmReturn_t DcgmProfTester::Init(int argc, char *argv[])
{
    dcgmReturn_t dcgmReturn = DCGM_ST_OK;

    dcgmReturn = ParseCommandLine(argc, argv);
    if (dcgmReturn != DCGM_ST_OK)
        return dcgmReturn;

    /* Start DCGM. We will initialize CUDA in each of the per-slice (only one
       if not MIG) worker processes.
    */
    dcgmReturn = DcgmInit();
    return dcgmReturn;
}

// Abort Physical GPU processes. Something went very wrong.
void DcgmProfTester::AbortOtherChildren(unsigned int gpuId)
{
    /**
     * We don't abort the one specified as it already aborted itself. And we
     * don't abort more after we find ourselves because we were called to abort
     * all the ones initialized before the one specified in a similar (ordered)
     * map iteration.
     */

    for (auto &[id, gpu] : m_gpus)
    {
        if (id == gpuId)
        {
            break;
        }

        gpu->AbortChildren();
    }

    m_dcgmCudaContexts.clear();
}

/* Nuke child processes. This tries to be gentle before aborting it. It
 * should be used if the children have already started processing a request
 * as opposed to not. Generally Aborting the child will cause it to die
 * from SIGPIPE when it can't do I/O on its pipes, sending an "X" will let
 * it try to exit cleanly if it is done working and waiting for another
 * command.
 */
void DcgmProfTester::NukeChildren(bool force)
{
    for (auto &[gpuId, gpu] : m_gpus)
    {
        gpu->NukeChildren(force);
    }

    m_dcgmCudaContexts.clear();
}

/*****************************************************************************/

/**
 * One-of reporting management for use by Physical GPU classes to coordinate
 * their reporting.
 */


//  Is this the first tick in the current part from any GPU?
bool DcgmProfTester::IsFirstTick(void) const
{
    return m_isFirstTick;
}

// Set (or clear)  the first tick in the current part from any GPU.
void DcgmProfTester::SetFirstTick(bool value)
{
    m_isFirstTick = value;
}

// Get the next test part we expect.
unsigned int DcgmProfTester::GetNextPart(void) const
{
    return m_nextPart;
}

// Get the next test part we expect.
void DcgmProfTester::SetNextPart(unsigned int value)
{
    m_nextPart = value;
}


/*****************************************************************************/
void DcgmProfTester::ReportWorkerStarted(std::shared_ptr<DistributedCudaContext> worker)
{
    int fd = worker->GetReadFD();

    m_dcgmCudaContexts[fd] = worker;

    FD_SET(fd, &m_parentReadFds);

    if (fd > m_maxFd)
    {
        m_maxFd = fd;
    }
}


/*****************************************************************************/
void DcgmProfTester::ReportWorkerFailed(std::shared_ptr<DistributedCudaContext> worker)
{
    int fd = worker->GetReadFD();

    FD_CLR(fd, &m_parentReadFds);

    if (fd > m_maxFd)
    {
        m_maxFd = fd;
    }
}


/*****************************************************************************/
dcgmReturn_t DcgmProfTester::RunTests(double reportingInterval, double duration, unsigned int testFieldId)
{
    dcgmReturn_t rtSt { DCGM_ST_OK };

    SetFirstTick(false);
    SetNextPart(0);

    for (auto &gpuInstance : m_gpuInstances)
    {
        std::shared_ptr<DistributedCudaContext> worker { nullptr };
        std::string cudaVisibleDevices { "" };

        auto entities = std::make_shared<std::map<dcgm_field_entity_group_t, dcgm_field_eid_t>>();

        unsigned short fieldId { DCGM_FI_DEV_CUDA_VISIBLE_DEVICES_STR };
        dcgmGroupEntityPair_t entity;

        (*entities)[DCGM_FE_GPU] = gpuInstance.m_gpuId;

        if (gpuInstance.m_isMig)
        {
            if ((testFieldId != DCGM_FI_PROF_PCIE_TX_BYTES) && (testFieldId != DCGM_FI_PROF_PCIE_RX_BYTES))
            {
                (*entities)[DCGM_FE_GPU_I]  = gpuInstance.m_gi;
                (*entities)[DCGM_FE_GPU_CI] = gpuInstance.m_ci;
            }

            entity.entityGroupId = DCGM_FE_GPU_CI;
            entity.entityId      = gpuInstance.m_ci;

            dcgmFieldValue_v2 value {};

            dcgmReturn_t dcgmReturn
                = dcgmEntitiesGetLatestValues(m_dcgmHandle, &entity, 1, &fieldId, 1, DCGM_FV_FLAG_LIVE_DATA, &value);

            if (dcgmReturn != DCGM_ST_OK || value.status != DCGM_ST_OK)
            {
                AbortOtherChildren(gpuInstance.m_gpuId);

                DCGM_LOG_ERROR << "Could not map Entity ID [" << entity.entityGroupId << "," << entity.entityId
                               << "] to CUDA_VISIBLE_DEVICES environment variable (" << (int)dcgmReturn << "), "
                               << value.status << ")";

                return DCGM_ST_GENERIC_ERROR;
            }

            cudaVisibleDevices = value.value.str;
        }
        else
        {
            entity.entityGroupId = DCGM_FE_GPU;
            entity.entityId      = gpuInstance.m_gpuId;
            cudaVisibleDevices   = "";
        }

        worker = m_gpus[gpuInstance.m_gpuId]->AddSlice(entities, entity, cudaVisibleDevices);

        if (worker == nullptr) // Failure to add a worker slice.
        {
            return DCGM_ST_GENERIC_ERROR;
        }
    }

    static const unsigned int cUsecInSec = 1000000;

    for ([[maybe_unused]] auto &[gpuId, gpu] : m_gpus)
    {
        rtSt = gpu->StartTests(); // Will not start workers already running.

        if ((rtSt == DCGM_ST_OK) && gpu->AllStarted())
        {
            /**
             * We are already started, so we just need to run tests.
             * This can happen when we reuse started GPU (or GPU MIG slice)
             * worker processes. Normally, we are not AllStarted until the GPU
             * (or GPU MIG slice) workers report that they started to the
             * Physical GPU object. When that happens, the Physical GPU
             * advances to the Running state and actually start the test either
             * on the entire GPU or the MIG slice worker processes.
             *
             * But, in this case, the GPU workers were already started, so there
             * are no starting indications from worker processes to drive the
             * transition from Starting to Running, so we request that
             * transition here.
             */

            rtSt = gpu->RunTests();
        }

        if (rtSt != DCGM_ST_OK)
        {
            AbortOtherChildren(gpuId);

            return rtSt;
        }
    }

    dcgmReturn_t dcgmReturn = WatchFields(cUsecInSec * reportingInterval, duration, testFieldId);

    if (dcgmReturn != DCGM_ST_OK)
    {
        NukeChildren(true);

        return dcgmReturn;
    }

    /*
     * We have now started up all worker processes. We should check if they are
     * ready to receive commands. They send back P\n if initialization passed,
     * and F\n if it failed. If they are crashed, they have closed their side
     * of the pipes between us.
     */

    fd_set rfds;
    size_t physicalGPUsReported { 0 };

    while (physicalGPUsReported < m_gpus.size())
    {
        std::memcpy(&rfds, &m_parentReadFds, sizeof(fd_set));

        timeval tv;

        // Wait up to twice the duration, per GPU instance, rounded up.
        double timeout = duration * m_gpuInstances.size() * 2.0 + 0.5;
        double timeoutInt;
        double timeoutFrac;

        timeoutFrac = modf(timeout, &timeoutInt);

        tv.tv_sec  = (decltype(tv.tv_sec))(timeoutInt);
        tv.tv_usec = (decltype(tv.tv_usec))(timeoutFrac * 1000000.0);

        int readyWorkers = select(m_maxFd + 1, &rfds, nullptr, nullptr, &tv);

        if (readyWorkers < 0) // select error
        {
            NukeChildren(true);
            UnwatchFields();

            return DCGM_ST_GENERIC_ERROR;
        }

        if (readyWorkers == 0) // timeout
        {
            NukeChildren(true);
            UnwatchFields();

            return DCGM_ST_GENERIC_ERROR;
        }

        // Handle workers with data.

        for (unsigned int fd = 0; (fd <= m_maxFd) && (readyWorkers > 0); fd++)
        {
            if (!FD_ISSET(fd, &rfds))
            {
                // This worker has no response yet.
                continue;
            }

            readyWorkers--;

            auto worker = m_dcgmCudaContexts[fd];

            if (worker == nullptr)
            {
                fprintf(stderr, "*** NULLPTR worker on fd %d\n", fd);

                continue;
            }

            auto physicalGpu     = worker->GetPhysicalGpu();
            auto alreadyReported = physicalGpu->AllReported();

            rtSt = physicalGpu->ProcessResponse(worker);

            if (physicalGpu->WorkerReported())
            {
                FD_CLR(fd, &m_parentReadFds);
                m_dcgmCudaContexts[fd] = nullptr;

                // We only check this when it makes a difference.
                if (fd == m_maxFd)
                {
                    while ((m_maxFd > 0) && !FD_ISSET(m_maxFd, &m_parentReadFds))
                    {
                        --m_maxFd;
                    }
                }
            }

            // Check if this physical GPU finished reporting results.
            if (!alreadyReported && physicalGpu->AllReported())
            {
                physicalGPUsReported++;
            }
        }
    }

    UnwatchFields();

    unsigned int i = 0;

    bool failed { false };

    for (auto &[_, gpu] : m_gpus)
    {
        if (!gpu->IsValidated())
        {
            std::cout << "GPU " << i << ", TestField " << testFieldId << " test FAILED" << std::endl;

            failed = true;
        }

        i++;
    }

    return failed ? DCGM_ST_GENERIC_ERROR : DCGM_ST_OK;
}


/**
 * Initializes CUDA contexts - one per physical CPU or one per MIG slice.
 *
 * @return Returns DCGM_ST_OK on success, other on error.
 */
dcgmReturn_t DcgmProfTester::InitializeGpuInstances(void)
{
    dcgmMigHierarchy_v2 hierarchy {};

    hierarchy.version = dcgmMigHierarchy_version2;

    dcgmReturn_t ret = dcgmGetGpuInstanceHierarchy(m_dcgmHandle, &hierarchy);
    if (ret != DCGM_ST_OK)
    {
        DCGM_LOG_ERROR << "Failed to enumerate GPU instances: " << errorString(ret);

        return ret;
    }

    try
    {
        std::unordered_map<dcgm_field_eid_t, dcgmMigHierarchyInfo_v2 *> gpuInstances;

        // Count GPU instances and per-GPU instance Compute instances.
        for (auto *p = hierarchy.entityList; p < hierarchy.entityList + hierarchy.count; p++)
        {
            if ((p->entity.entityGroupId == DCGM_FE_GPU_I) && (p->parent.entityGroupId == DCGM_FE_GPU)
                && (m_gpus.find(p->parent.entityId) != m_gpus.end()))
            {
                gpuInstances[p->entity.entityId] = p;
            }
            else if ((p->entity.entityGroupId == DCGM_FE_GPU_CI) && (p->parent.entityGroupId == DCGM_FE_GPU_I)
                     && (gpuInstances.find(p->parent.entityId) != gpuInstances.end()))
            {
                m_gpuInstances.emplace_back(gpuInstances[p->parent.entityId]->parent.entityId,
                                            gpuInstances[p->parent.entityId]->entity.entityId,
                                            p->entity.entityId);
            }
        }

        // Pick up non-MIG GPUs.
        for (auto &[gpuId, gpu] : m_gpus)
        {
            if (!gpu->IsMIG())
            {
                m_gpuInstances.emplace_back(gpuId);
            }
        }
    }
    catch (...) // Generally std::bad_alloc but we catch all.
    {
        DCGM_LOG_ERROR << "Cannot allocate CUDA worker processes";
        return DCGM_ST_GENERIC_ERROR;
    }

    return DCGM_ST_OK;
}

/**
 * Shuts down CUDA contexts - one per physical CPU or one per MIG slice.
 *
 * @return Returns DCGM_ST_OK on success, other on error.
 */
dcgmReturn_t DcgmProfTester::ShutdownGpuInstances(void)
{
    m_gpuInstances.clear();

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmProfTester::Process(std::function<dcgmReturn_t(std::shared_ptr<Arguments_t> arguments)> processFn)
{
    return m_argumentSet.Process([this, processFn](std::shared_ptr<Arguments_t> arguments) {
        m_startDcgm = !arguments->m_parameters.m_noDcgmValidation;

        return processFn(arguments);
    });
}


/*****************************************************************************/
int main(int argc, char **argv)
{
    try
    {
        std::shared_ptr<DcgmProfTester> dpt { std::make_shared<DcgmProfTester>() };

        dcgmReturn_t dcgmReturn;

        int cudaLoaded = 0;
        auto cuResult  = cuDriverGetVersion(&cudaLoaded);
        if (cuResult != CUDA_SUCCESS)
        {
            fprintf(stderr, "Unable to validate Cuda version. cuDriverGetVersion returned %d\n", cuResult);
            return -1;
        }

        // CUDA_VERSION_USED is defined in CMakeLists.txt file
        if ((cudaLoaded / 1000) != CUDA_VERSION_USED)
        {
            fprintf(
                stderr,
                "Wrong version of dcgmproftester is used. Expected Cuda version is %d; Installed Cuda version is %d.\n",
                CUDA_VERSION_USED,
                cudaLoaded / 1000);
            return -1;
        }

        // We do this to avoid zombies. We don't care about worker exit codes.
        struct sigaction sa;
        sa.sa_handler = SIG_IGN; // handle signal by ignoring
        sigemptyset(&sa.sa_mask);
        sa.sa_flags = 0;
        if (sigaction(SIGCHLD, &sa, 0) == -1)
        {
            fprintf(stderr, "Could not ignore SIGCHLD from worker threads.\n");
            return -1;
        }

        DcgmLogging::init("dcgmproftester.log", DcgmLoggingSeverityWarning);

        dcgmReturn = dpt->Init(argc, argv);
        if (dcgmReturn)
        {
            fprintf(stderr, "Error %d from Init(). Exiting.\n", dcgmReturn);
            return -((int)dcgmReturn);
        }

        dcgmReturn = dpt->Process([dpt](std::shared_ptr<DcgmNs::ProfTester::Arguments_t> arguments) -> dcgmReturn_t {
            dcgmReturn_t dcgmReturn = dpt->InitializeGpus(*arguments);

            if (dcgmReturn)
            {
                fprintf(stderr, "Error %d from InitializeGpus(). Exiting.\n", dcgmReturn);

                return dcgmReturn;
            }

            dcgmReturn_t st = dpt->RunTests(arguments->m_parameters.m_reportInterval,
                                            arguments->m_parameters.m_duration,
                                            arguments->m_parameters.m_fieldId);

            dcgmReturn = dpt->ShutdownGpus();

            if (dcgmReturn)
            {
                fprintf(stderr, "Error %d from ShutdownGpus(). Exiting.\n", dcgmReturn);

                return dcgmReturn;
            }

            return st;
        });

        return -dcgmReturn;
    }
    catch (std::runtime_error const &ex)
    {
        fprintf(stderr, "Uncaught exception occured: %s\n", ex.what());
        return -((int)DCGM_ST_GENERIC_ERROR);
    }
    catch (...)
    {
        fprintf(stderr, "Uncaught unexpected exception occured\n");
        return -((int)DCGM_ST_GENERIC_ERROR);
    }

    return 0;
}
