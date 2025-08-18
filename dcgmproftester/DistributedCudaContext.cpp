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

#include "DistributedCudaContext.h"
#include "PhysicalGpu.h"
#include <dcgm_agent.h>
#include <dcgm_structs.h>

#include <algorithm>
#include <cerrno>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <fcntl.h>
#include <iomanip>
#include <iostream> // for debugging
#include <libgen.h> // for dirname
#include <sys/prctl.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <system_error>
#include <timelib.h>
#include <unistd.h>
#include <vector>

using namespace Dcgm;


namespace DcgmNs::ProfTester
{
// Initialize object. Call from worker process.
dcgmReturn_t DistributedCudaContext::Init(int inFd, int outFd)
{
    int st;
    CUresult cuSt;

    if (m_isInitialized)
    {
        m_message << "DCGM Cuda Context already initialized.\n\n";
        return DCGM_ST_OK;
    }

    m_inFd  = inFd;
    m_outFd = outFd;

    // Make I/O in child non-blocking.
    st = fcntl(m_inFd, F_SETFL, O_NONBLOCK);
    if (st == -1)
    {
        return DCGM_ST_GENERIC_ERROR;
    }

    /**
     * With MIG, we have to use CUDA_VISIBLE_DEVICES to identify the MIG
     * partition "device", along with a device ID of 0. We can not run
     * NvLink tests with MIG and we check to ensure we do not try. Without MIG,
     * we set CUDA_VISIBLE_DEVICES to all non-MIG GPUs so NvLink tests work, and
     * we set the device ID properly.
     */

    st = setenv("CUDA_VISIBLE_DEVICES", m_cudaVisibleDevices.c_str(), 1);
    if (st != 0)
    {
        m_error << "std::setenv returned" << st << '\n';

        return DCGM_ST_GENERIC_ERROR;
    }

    m_message << "std::setenv successfully set CUDA_VISIBLE_DEVICES to " << m_cudaVisibleDevices.c_str() << '\n';

    if (cuSt = cuInit(0); cuSt != CUDA_SUCCESS)
    {
        const char *errorString { nullptr };
        cuGetErrorString(cuSt, &errorString);

        m_error << "cuInit returned " << static_cast<unsigned int>(cuSt) << " - " << errorString << " for "
                << (GetPhysicalGpu()->IsMIG() ? m_cudaVisibleDevices.c_str() : std::to_string(m_device)) << '\n';

        return DCGM_ST_GENERIC_ERROR;
    }

    if (!GetPhysicalGpu()->IsMIG())
    {
        /**
         * Without MIG, we can get the physical device ID.
         */

        std::string busId = GetPhysicalGpu()->GetGpuBusId().c_str();

        if (cuSt = cuDeviceGetByPCIBusId(&m_device, busId.c_str()); cuSt != CUDA_SUCCESS)
        {
            const char *errorString { nullptr };
            cuGetErrorString(cuSt, &errorString);
            m_error << "cuDeviceGetByPCIBusId returned " << static_cast<unsigned int>(cuSt) << " - " << errorString
                    << " for " << busId.c_str() << '\n';
            return DCGM_ST_GENERIC_ERROR;
        }

        m_message << "Bus ID " << busId << " mapped to cuda device ID " << m_device << "\n";
    }
    else
    {
        m_device = 0;
    }

    std::pair<int *, CUdevice_attribute const> const attributes[] = {
        { &m_attributes.m_maxThreadsPerMultiProcessor, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR },
        { &m_attributes.m_multiProcessorCount, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT },
        { &m_attributes.m_sharedMemPerMultiprocessor, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR },
        { &m_attributes.m_computeCapabilityMajor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR },
        { &m_attributes.m_computeCapabilityMinor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR },
        { &m_attributes.m_memoryBusWidth, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH },
        { &m_attributes.m_maxMemoryClockMhz, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE },
        { &m_attributes.m_eccSupport, CU_DEVICE_ATTRIBUTE_ECC_ENABLED },
    };

    for (auto [value, cuDevAttr] : attributes)
    {
        bool const isMIG = GetPhysicalGpu()->IsMIG();
        if (cuSt = cuDeviceGetAttribute(value, cuDevAttr, m_device); cuSt != CUDA_SUCCESS)
        {
            const char *errorString { nullptr };
            cuGetErrorString(cuSt, &errorString);
            m_error << "cuDeviceGetAttribute(" << static_cast<unsigned int>(cuDevAttr) << ") returned "
                    << static_cast<unsigned int>(cuSt) << " - " << errorString << " for "
                    << (isMIG ? m_cudaVisibleDevices.c_str() : std::to_string(m_device)) << "\n";
            return DCGM_ST_GENERIC_ERROR;
        }
    }

    m_attributes.m_computeCapability
        = (double)m_attributes.m_computeCapabilityMajor + ((double)m_attributes.m_computeCapabilityMinor / 10.0);

    /* Convert to MHz */
    m_attributes.m_maxMemoryClockMhz /= 1000;

    /**
     * memory bandwidth in bytes = memClockMhz * 1000000 bytes per MiB *
     * 2 copies per cycle.bitWidth / 8 bits per byte.
     */
    m_attributes.m_maxMemBandwidth
        = (double)m_attributes.m_maxMemoryClockMhz * 1000000.0 * 2.0 * (double)m_attributes.m_memoryBusWidth / 8.0;

    m_message << "DCGM CudaContext Init completed successfully." << '\n' << '\n';

    m_isInitialized = true;
    return DCGM_ST_OK;
}

// Return worker attributes
const DistributedCudaContext::Attributes &DistributedCudaContext::GetAttributes(void) const
{
    return m_attributes;
}


// Reset object.
void DistributedCudaContext::Reset(bool keepDcgmGroups)
{
    if (m_inFd >= 0)
    {
        close(m_inFd);
        m_inFd = -1;
    }

    if (m_outFd >= 0)
    {
        close(m_outFd);
        m_outFd = -1;
    }

    {
        std::stringstream temp;
        m_input << "";
        m_input.swap(temp);
    }

    {
        std::stringstream temp;
        m_message << "";
        m_message.swap(temp);
    }

    {
        std::stringstream temp;
        m_error << "";
        m_error.swap(temp);
    }

    if (!keepDcgmGroups)
    {
        DestroyDcgmGroups();
    }

    m_isInitialized = false;
    m_failed        = false;
    // m_pid         = 0; -- preserve this to kill child if necessary.
    m_part      = 0;
    m_parts     = 0;
    m_activity  = 0;
    m_tick      = false;
    m_firstTick = false;
    m_wait      = false;
    m_buffered  = false;
    m_finished  = false;
    m_validated = true;
}


// Send a command.
//
// This can also be used to send a response, since the parent and child pipe
// file descriptors are symmetrical. But, it is better to use the Respond()
// function as it throws an exception on error. That is usually catastrophic
// in the child.
int DistributedCudaContext::Command(const char *format, std::va_list args)
{
    return vdprintf(m_outFd, format, args);
}

// Send a command.
//
// This wraps Command that takes a va_list.
int DistributedCudaContext::Command(const char *format, ...)
{
    std::va_list args;

    va_start(args, format);

    int err = Command(format, args);

    va_end(args);

    return err;
}

// Read from peer.
//
// If we're the child, we read from the parent. If we're the parent,
// we read from the child. As this is a blocking call, the parent may wish
// to use select() on an fd_set before reading. We read up to 1000 bytes at a
// time. We return true when the requested data has been read, or false on
// error.
//
// This is really intended for the parent process to read a known number
// of bytes (such as an indicated E(rror) or M(essage) response. It does mean
// the parent process is "greedy" in that once it is notified that such a
// response is available, it will block read it until it has it all, but
// it avoids the parent having to keep partial response state for a number of
// workers.
//
bool DistributedCudaContext::Read(size_t toRead)
{
    char buffer[1001];
    int rtSt;

    do
    {
        size_t thisRead = toRead;

        if (thisRead > (sizeof(buffer) - 1))
        {
            thisRead = sizeof(buffer) - 1;
        }

        rtSt = read(m_inFd, &buffer, thisRead);

        if (rtSt <= 0)
        {
            return false;
        }

        buffer[rtSt] = '\0';

        m_input << buffer;
        toRead -= rtSt;
    } while (toRead > 0);

    return true;
}


// Read from our peer.
//
// If we're the child, we read a command from the parent. If we're the parent,
// we read a response from the child.
//
// Returns 1 on a successful read, 0 on no data (if non-blocking), and negative
// on error.
int DistributedCudaContext::ReadLn(void)
{
    char c;
    int rtSt;

    while (((rtSt = read(m_inFd, &c, 1)) > 0))
    {
        m_input << c;

        if (c == '\n')
        {
            break;
        }
    }

    if ((rtSt == -1) && (errno == EAGAIN)) // no data
    {
        rtSt = 0;
    }

    return rtSt;
}

// Check for synchronization command.
int DistributedCudaContext::ReadLnCheck(unsigned int &activity, bool &earlyQuit)
{
    int retSt { 0 };

    earlyQuit = false;

    if (GetPhysicalGpu()->IsSynchronous())
    {
        retSt = ReadLn();

        if (retSt == 0)
        {
            // Repeat.
            --activity;
        }
        else if (retSt > 0)
        {
            char command;

            m_input >> command;

            // Ignore rest of synchronization command.
            m_input.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

            if (command == 'Q')
            {
                earlyQuit = true;
            }
            else if (command != 'A')
            {
                retSt = -1;
            }
        }
    }

    return retSt;
}


// Get input file descriptor.
int DistributedCudaContext::GetReadFD(void) const
{
    return m_inFd;
}


// Get input stream.
std::stringstream &DistributedCudaContext::Input(void)
{
    return m_input;
}


// Set part of parts.
//
// This is intended to be called on the parent process to keep track of the
// part that completed and total number of parts. The ticks sent by various
// tests have a different syntax and or semantics in different parts, though
// a consistent syntax and semantics within a part. The "D"one response
// indicates when a new part starts. (The first part is always numbered 0.)
void DistributedCudaContext::SetParts(unsigned int part, unsigned int parts)
{
    m_part      = part;
    m_parts     = parts;
    m_tick      = false;
    m_firstTick = true;
}

// Get part of parts.
//
// This is intended to be called on the parent process to identify what part
// we are in. We don't know the number of parts until the first one is reported
// indicating the number of parts.
void DistributedCudaContext::GetParts(unsigned int &part, unsigned int &parts) const
{
    part  = m_part;
    parts = m_parts;
}

// Lets us keep track of the first tick of a part of work.
bool DistributedCudaContext::IsFirstTick(void) const
{
    return m_firstTick;
}

// Request early test termination. Call from main process.
void DistributedCudaContext::Exit(void)
{
    Command("X\nX\n"); // Request exit;
    m_pid = 0;         // Forget about child.
}

// Tell this worker it is finished processing data.
void DistributedCudaContext::SetFinished(void)
{
    m_finished = true;
}

// Determine if we have finished processing data.
bool DistributedCudaContext::Finished(void) const
{
    return m_finished;
}

// Tell this worker it is in a failed state. Intended to be called from parent.
void DistributedCudaContext::SetFailed(void)
{
    m_failed = true;
}

// Tell this worker it is in a non-failed state. To be called from parent.
void DistributedCudaContext::ClrFailed(void)
{
    m_failed = false;
}

// Determine if we are in a failed state.
bool DistributedCudaContext::Failed(void) const
{
    return m_failed;
}

// Set this worker's tries. Intended to be called from parent.
void DistributedCudaContext::SetTries(unsigned int tries)
{
    m_tries = tries;
}

// Return number of tries available for a given activity level.
unsigned int DistributedCudaContext::GetTries(void) const
{
    return m_tries;
}

// Set this worker's test depth.
void DistributedCudaContext::SetActivity(unsigned int activity)
{
    m_activity = activity;
}

// Return this worker's test depth.
unsigned int DistributedCudaContext::GetActivity(void) const
{
    return m_activity;
}

// Set this worker's validation status.
void DistributedCudaContext::SetValidated(bool validated)
{
    m_validated = validated;
}

// Return this worker's validation status.
bool DistributedCudaContext::GetValidated(void) const
{
    return m_validated;
}

// Send back a response.
//
// Throw exception on EOF or other error, so we can be sure to kill this
// process. This is private, unlike Command, since it is expected to be used
// in the child process.
void DistributedCudaContext::Respond(const char *format, ...)
{
    std::va_list args;

    va_start(args, format);

    int err = Command(format, args);

    va_end(args);

    RespondException ex(err);

    if (err < 0)
    {
        throw ex;
    }
}


// Move other object to this.
void DistributedCudaContext::MoveFrom(DistributedCudaContext &&other)
{
    if (this == &other)
    {
        return;
    }

    m_groupId       = other.m_groupId;
    other.m_groupId = 0;

    m_cudaVisibleDevices = std::move(other.m_cudaVisibleDevices);

    m_input   = std::move(other.m_input);
    m_message = std::move(other.m_message);
    m_error   = std::move(other.m_error);

    std::swap(m_inFd, other.m_inFd);
    std::swap(m_outFd, other.m_outFd);

    m_physicalGpu       = other.m_physicalGpu;
    other.m_physicalGpu = nullptr;

    m_device         = other.m_device;
    m_attributes     = other.m_attributes;
    m_isInitialized  = other.m_isInitialized;
    m_failed         = other.m_failed;
    m_pid            = other.m_pid;
    m_part           = other.m_part;
    m_parts          = other.m_parts;
    m_activity       = other.m_activity;
    m_tick           = other.m_tick;
    m_testFieldId    = other.m_testFieldId;
    m_duration       = other.m_duration;
    m_reportInterval = other.m_reportInterval;
    m_targetMaxValue = other.m_targetMaxValue;
    m_wait           = other.m_wait;
    m_buffered       = other.m_buffered;
    m_finished       = other.m_finished;
    m_validated      = other.m_validated;

    other.Reset();
}


DistributedCudaContext::DistributedCudaContext(
    std::shared_ptr<PhysicalGpu> physicalGpu,
    std::shared_ptr<std::map<dcgm_field_entity_group_t, dcgm_field_eid_t>> entities,
    const dcgmGroupEntityPair_t &entity,
    const std::string &cudaVisibleDevices)
    : m_physicalGpu(std::move(physicalGpu))
    , m_cudaVisibleDevices(cudaVisibleDevices)
    , m_entity(entity)
{
    ReInitialize(std::move(entities));
}


DistributedCudaContext::DistributedCudaContext(DistributedCudaContext &&other)
    : m_entity({})
{
    MoveFrom(std::move(other));
}


DistributedCudaContext::~DistributedCudaContext()
{
    Reset();

    if (m_pid != 0) // We are the parent process, kill child if necessary
    {
        usleep(200000); // Wait a little bit.

        pid_t result = waitpid(m_pid, NULL, WNOHANG);

        if (result == 0)
        {
            // Child still alive
            kill(m_pid, SIGKILL);
        }
    }
}


void DistributedCudaContext::ReInitialize()
{
    CreateDcgmGroups();
}


void DistributedCudaContext::ReInitialize(
    std::shared_ptr<std::map<dcgm_field_entity_group_t, dcgm_field_eid_t>> entities)
{
    m_entities = std::move(entities);
    CreateDcgmGroups();
}


DistributedCudaContext &DistributedCudaContext::operator=(DistributedCudaContext &&other) noexcept
{
    Reset();
    MoveFrom(std::move(other));
    other.Reset();

    return *this;
}


std::shared_ptr<PhysicalGpu> DistributedCudaContext::GetPhysicalGpu(void)
{
    return m_physicalGpu;
}

const std::string &DistributedCudaContext::Device(void) const
{
    return m_cudaVisibleDevices;
}

const dcgmGroupEntityPair_t &DistributedCudaContext::EntityId(void) const
{
    return m_entity;
}

std::map<dcgm_field_entity_group_t, dcgm_field_eid_t> &DistributedCudaContext::Entities(void) const
{
    return *m_entities;
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
dcgmReturn_t DistributedCudaContext::CreateDcgmGroups(void)
{
    dcgmReturn_t dcgmReturn = DestroyDcgmGroups();

    if (dcgmReturn != DCGM_ST_OK)
    {
        return dcgmReturn;
    }

    if (!GetPhysicalGpu()->GetDcgmValidation())
    {
        printf("Skipping CreateDcgmGroups() since DCGM validation is disabled\n");
        return DCGM_ST_OK;
    }

    char groupName[256] = { 0 };
    snprintf(groupName, sizeof(groupName), "dpt_%d_%s", getpid(), Device().c_str());

    dcgmReturn = dcgmGroupCreate(GetPhysicalGpu()->GetHandle(), DCGM_GROUP_EMPTY, groupName, &m_groupId);

    if (dcgmReturn != DCGM_ST_OK)
    {
        fprintf(stderr, "dcgmGroupCreate() returned %d\n", dcgmReturn);
        return dcgmReturn;
    }

    for (auto &[type, id] : *m_entities)
    {
        /**
         * Remove it in case it is already there as we can't add it more than
         * once.
         */

        dcgmGroupRemoveEntity(GetPhysicalGpu()->GetHandle(), m_groupId, type, id);
        dcgmReturn = dcgmGroupAddEntity(GetPhysicalGpu()->GetHandle(), m_groupId, type, id);

        if (dcgmReturn != DCGM_ST_OK)
        {
            fprintf(stderr, "dcgmGroupAddEntity() returned %d\n", dcgmReturn);
            return dcgmReturn;
        }
    }


    return dcgmReturn;
}

/*****************************************************************************/
dcgmReturn_t DistributedCudaContext::DestroyDcgmGroups(void)
{
    dcgmReturn_t dcgmReturn = DCGM_ST_OK;

    if (m_groupId != 0)
    {
        if (!GetPhysicalGpu()->GetDcgmValidation())
        {
            return DCGM_ST_OK;
        }

        dcgmReturn = dcgmGroupDestroy(GetPhysicalGpu()->GetHandle(), m_groupId);

        if (dcgmReturn != DCGM_ST_OK)
        {
            fprintf(stderr, "dcgmGroupDestroy() returned %d\n", dcgmReturn);
            return dcgmReturn;
        }

        m_groupId = 0;
    }

    return dcgmReturn;
}

/*****************************************************************************/
/**
   ReportIntervalSleep sleeps for m_reportInterval seconds, to microsecond
   resolution. It is basically a wrapper around usleep(useconds_t t) which
   accepts a useconds_t argument capable of holding integral values in the
   range of [0,1000000] and MAYBE more.

   To be portable, we don't call usleep with values greater than 1000000.0
   (which would be implicitly truncated to an integral type), but call this
   instead.
 */
/*****************************************************************************/
void DistributedCudaContext::ReportIntervalSleep(void) const
{
    unsigned long sleepSeconds  = static_cast<unsigned long>(m_reportInterval);
    unsigned long usleepSeconds = static_cast<useconds_t>((m_reportInterval - sleepSeconds) * 1000000.0);

    if (sleepSeconds > 0)
    {
        sleep(sleepSeconds);
    }

    if (usleepSeconds > 0)
    {
        usleep(usleepSeconds);
    }
}

/*****************************************************************************/
static int dptGetLatestDcgmValueCB(dcgm_field_entity_group_t entityGroupId,
                                   dcgm_field_eid_t entityId,
                                   dcgmFieldValue_v1 *values,
                                   int numValues,
                                   void *userData)
{
    if (userData == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }

    cbData *data = (cbData *)userData;

    /**
     * If there's no data yet, status of the first record will be
     * DCGM_ST_NO_DATA
     */
    if (values[0].status == DCGM_ST_NO_DATA)
    {
        DCGM_LOG_WARNING << "Got DCGM_ST_NO_DATA. Timing may be off by one "
                         << "cycle.";

        return DCGM_ST_OK;
    }
    else if (values[0].status != DCGM_ST_OK)
    {
        DCGM_LOG_ERROR << "Got unknown error " << errorString((dcgmReturn_t)values[0].status);

        return DCGM_ST_OK;
    }

    dcgmGroupEntityPair_t entity;

    entity.entityGroupId = entityGroupId;
    entity.entityId      = entityId;

    data->m_values[entity].insert(data->m_values[entity].end(), values, values + numValues);

    return DCGM_ST_OK;
}


/*****************************************************************************/
dcgmReturn_t DistributedCudaContext::GetLatestDcgmValues(std::map<Entity, dcgmFieldValue_v1> &values)
{
    dcgmReturn_t dcgmReturn;
    long long nextSinceTimestamp = 0;

    if (!GetPhysicalGpu()->GetDcgmValidation())
        return DCGM_ST_OK;

    cbData data(GetPhysicalGpu()->GetGpuId(), m_dcgmValues);

    dcgmReturn = dcgmGetValuesSince_v2(GetPhysicalGpu()->GetHandle(),
                                       m_groupId,
                                       GetPhysicalGpu()->GetFieldGroupId(),
                                       m_sinceTimestamp,
                                       &nextSinceTimestamp,
                                       dptGetLatestDcgmValueCB,
                                       &data);

    if (dcgmReturn == DCGM_ST_OK)
    {
        m_sinceTimestamp = nextSinceTimestamp;

        for (auto &[entity, readValues] : m_dcgmValues)
        {
            if (readValues.size() > 0)
            {
                values[entity] = readValues[readValues.size() - 1];
            }
        }

        return DCGM_ST_OK;
    }

    fprintf(stderr, "dcgmGetValuesSince returned %d\n", dcgmReturn);
    return dcgmReturn;
}

/*****************************************************************************/
int DistributedCudaContext::RunSubtestSmOccupancyTargetMax(void)
{
    /* Generate SM activity */

    int retSt { 0 };

    auto duration           = m_duration;
    double startTime        = timelib_dsecSince1970();
    double now              = timelib_dsecSince1970();
    double tick             = startTime + m_reportInterval;
    unsigned int activities = (duration + m_reportInterval / 2.0) / m_reportInterval;

    m_cudaWorker.SetWorkerToIdle();

    m_input.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    Respond("S\nD 0 1\n");

    m_cudaWorker.SetWorkloadAndTarget(DCGM_FI_PROF_SM_OCCUPANCY, 1.0, true);

    for (unsigned int activity = 0; activity < activities; activity++)
    {
        double howFarIn = 1.0 * activity / activities;

        ReportIntervalSleep();

        now = timelib_dsecSince1970();

        if (now > tick)
        {
            tick = now + m_reportInterval;
            Respond("T %u %0.3f %0.3f %0.3f %u\n",
                    activity,
                    howFarIn,
                    howFarIn,
                    now - startTime,
                    m_attributes.m_maxThreadsPerMultiProcessor);
        }

        bool earlyQuit { false };

        retSt = ReadLnCheck(activity, earlyQuit);

        if ((retSt < 0) || earlyQuit)
        {
            break;
        }
    }

    Respond((retSt < 0) ? "F\n" : "D 1 1\nP\n");

    m_cudaWorker.SetWorkerToIdle();

    return retSt;
}

/*****************************************************************************/
int DistributedCudaContext::RunSubtestSmOccupancy(void)
{
    /* Generate SM occupancy */

    int retSt { 0 };

    auto duration           = m_duration;
    double now              = timelib_dsecSince1970();
    double startTime        = timelib_dsecSince1970();
    double tick             = startTime + m_reportInterval;
    unsigned int activities = (duration + m_reportInterval / 2.0) / m_reportInterval;

    m_cudaWorker.SetWorkerToIdle();

    double curOccupancy  = 0.0;
    double prevOccupancy = 0.0;

    m_input.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    Respond("S\nD 0 1\n");

    double sleepKernelInterval  = std::max(0.005, m_reportInterval / 100.0);
    unsigned int innerLoopCount = m_reportInterval / sleepKernelInterval;

    sleepKernelInterval = m_reportInterval / innerLoopCount;

    /**
     * We start activity at least 20 percent of the way in since it can never be
     * exactly 0.
     */
    for (unsigned int activity = 0; activity < activities; activity++)
    {
        double howFarIn = 1.0 * activity / activities;

        m_cudaWorker.SetWorkloadAndTarget(DCGM_FI_PROF_SM_OCCUPANCY, howFarIn);
        curOccupancy = ((double)(unsigned int)(howFarIn * m_attributes.m_maxThreadsPerMultiProcessor))
                       / m_attributes.m_maxThreadsPerMultiProcessor;

        ReportIntervalSleep();

        now = timelib_dsecSince1970();
        if (now > tick)
        {
            Respond("T %u %0.3f %0.3f %.3f %.3f %u %u\n",
                    activity,
                    howFarIn,
                    prevOccupancy,
                    curOccupancy,
                    now - startTime,
                    (unsigned int)(howFarIn * m_attributes.m_maxThreadsPerMultiProcessor),
                    m_attributes.m_maxThreadsPerMultiProcessor);

            tick = now + m_reportInterval;
        }

        prevOccupancy = howFarIn;

        bool earlyQuit { false };

        retSt = ReadLnCheck(activity, earlyQuit);

        if ((retSt < 0) || earlyQuit)
        {
            break;
        }
    }

    m_cudaWorker.SetWorkerToIdle();

    Respond((retSt < 0) ? "F\n" : "D 1 1\nP\n");

    return 0;
}

/*****************************************************************************/
int DistributedCudaContext::RunSubtestSmActivity(void)
{
    /* Generate SM activity */

    int retSt { 0 };

    auto duration           = m_duration;
    double now              = timelib_dsecSince1970();
    double startTime        = timelib_dsecSince1970();
    double tick             = startTime + m_reportInterval;
    double curSmActivity    = 0.0;
    double prevSmActivity   = 0.0;
    unsigned int activities = (duration + m_reportInterval / 2.0) / m_reportInterval;

    m_cudaWorker.SetWorkerToIdle();

    m_input.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    Respond("S\nD 0 1\n");

    for (unsigned int activity = 0; activity < activities; activity++)
    {
        double howFarIn = 1.0 * activity / activities;

        m_cudaWorker.SetWorkloadAndTarget(DCGM_FI_PROF_SM_ACTIVE, howFarIn);
        curSmActivity = ((double)(unsigned int)(howFarIn * m_attributes.m_multiProcessorCount))
                        / m_attributes.m_multiProcessorCount;

        ReportIntervalSleep();

        now = timelib_dsecSince1970();
        if (now > tick)
        {
            Respond("T %u %0.3f %0.3f %2.3f %2.3f %u %u\n",
                    activity,
                    howFarIn,
                    prevSmActivity,
                    curSmActivity,
                    now - startTime,
                    (unsigned int)(howFarIn * m_attributes.m_multiProcessorCount),
                    m_attributes.m_multiProcessorCount);

            tick = now + m_reportInterval;
        }

        prevSmActivity = curSmActivity;

        bool earlyQuit { false };

        retSt = ReadLnCheck(activity, earlyQuit);

        if ((retSt < 0) || earlyQuit)
        {
            break;
        }
    }

    m_cudaWorker.SetWorkerToIdle();

    Respond((retSt < 0) ? "F\n" : "D 1 1\nP\n");

    return 0;
}

/*****************************************************************************/
int DistributedCudaContext::RunSubtestGrActivity(void)
{
    /* Generate graphics and SM activity */

    int retSt { 0 };

    auto duration           = m_duration;
    double now              = timelib_dsecSince1970();
    double startTime        = timelib_dsecSince1970();
    double tick             = startTime + m_reportInterval;
    unsigned int activities = (duration + m_reportInterval / 2.0) / m_reportInterval;

    m_cudaWorker.SetWorkerToIdle();

    m_input.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    Respond("S\nD 0 1\n");

    double sleepKernelInterval  = std::max(0.005, m_reportInterval / 100.0);
    unsigned int innerLoopCount = m_reportInterval / sleepKernelInterval;

    sleepKernelInterval = m_reportInterval / innerLoopCount;

    double prevHowFarIn = 0.0; // 1.0 * std::max(1U, (activities + 10) / 20) / activities;

    for (unsigned int activity = 0; activity < activities; activity++)
    {
        double howFarIn = 1.0 * activity / activities;

        m_cudaWorker.SetWorkloadAndTarget(DCGM_FI_PROF_GR_ENGINE_ACTIVE, howFarIn);
        ReportIntervalSleep();

        now = timelib_dsecSince1970();

        if (now > tick)
        {
            Respond("T %u %0.3f %0.3f %0.3f %0.3f\n", activity, howFarIn, prevHowFarIn, howFarIn, now - startTime);
            tick = now + m_reportInterval;
        }

        prevHowFarIn = howFarIn;

        bool earlyQuit { false };

        retSt = ReadLnCheck(activity, earlyQuit);

        if ((retSt < 0) || earlyQuit)
        {
            break;
        }
    }

    m_cudaWorker.SetWorkerToIdle();

    Respond((retSt < 0) ? "F\n" : "D 1 1\nP\n");

    return 0;
}

/*****************************************************************************/
int DistributedCudaContext::RunSubtestPcieBandwidth(void)
{
    constexpr int HOPPER_COMPAT = 9;
    int retSt { 0 };

    /* Setting target to 1.0 since it's ignored anyway. We can update
       FieldWorkerPciRxTxBytes in the future if we need specific targets */
    m_cudaWorker.SetWorkloadAndTarget(m_testFieldId, 1.0, true);

    /* Set timers after we've allocated memory since that takes a while */

    auto duration           = m_duration;
    unsigned int activities = (duration + m_reportInterval / 2.0) / m_reportInterval;

    m_input.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    Respond("S\nD 0 1\n");

    double perSecond = 0.0;

    for (unsigned int activity = 0; activity <= activities; activity++)
    {
        ReportIntervalSleep();

        double howFarIn      = 1.0 * activity / activities;
        double prevPerSecond = perSecond;
        perSecond            = m_cudaWorker.GetCurrentAchievedLoad();
        perSecond /= 1000000.0; /* Bytes -> MiB */

        /* Adjust for PCIe protocol overhead per chip generation */
        if (m_attributes.m_computeCapabilityMajor >= HOPPER_COMPAT)
        {
            perSecond *= 1.08; /* Consistently saw an 8% difference in testing */
        }
        else
        {
            perSecond *= 1.123; /* We've seen a 12.3% overhead in testing, verified secondarily by looking at
                                   nvidia-smi dmon -s */
        }

        Respond("T %u %0.3f %0.3f %0.3f\n", activity, howFarIn, prevPerSecond, perSecond);

        bool earlyQuit { false };

        retSt = ReadLnCheck(activity, earlyQuit);

        if ((retSt < 0) || earlyQuit)
        {
            break;
        }
    }

    m_cudaWorker.SetWorkerToIdle();

    // coverity[dead_error_condition] - Leaving in case retSt is nonzero in the future
    Respond((retSt < 0) ? "F\n" : "D 1 1\nP\n");

    return retSt;
}


/*****************************************************************************/
int DistributedCudaContext::RunSubtestNvLinkBandwidth(void)
{
    int retSt { 0 };

    m_cudaWorker.SetWorkerToIdle();

    /* Get our best peer to do NvLink copies to. Copy in rest of request line
     * and parse it.
     */
    std::string PeerBusId;
    m_input >> PeerBusId;
    m_cudaWorker.SetPeerByBusId(std::move(PeerBusId));

    m_cudaWorker.SetWorkerToIdle();

    m_input.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    unsigned int activities = (m_duration + m_reportInterval / 2.0) / m_reportInterval;

    /* Load target is irrelevant because we target SOL anyway */
    m_cudaWorker.SetWorkloadAndTarget(m_testFieldId, 1.0, true);

    Respond("S\nD 0 1\n");

    double perSecond     = 0.0;
    double prevPerSecond = 0.0;

    for (unsigned int activity = 0; activity <= activities; activity++)
    {
        ReportIntervalSleep();

        double howFarIn = 1.0 * activity / activities;
        prevPerSecond   = perSecond;
        perSecond       = m_cudaWorker.GetCurrentAchievedLoad();
        perSecond /= 1000000.0; /* Bytes -> MiB */

        Respond("T %u %0.3f %0.3f %0.3f\n", activity, howFarIn, prevPerSecond, perSecond);

        bool earlyQuit { false };

        retSt = ReadLnCheck(activity, earlyQuit);

        if ((retSt < 0) || earlyQuit)
        {
            break;
        }
    }

    m_cudaWorker.SetWorkerToIdle();

    Respond((retSt < 0) ? "F\n" : "D 1 1\nP\n");

    return retSt;
}

/*****************************************************************************/
bool DistributedCudaContext::EccAffectsDramBandwidth(void)
{
    /* Assert that m_attributes has been populated */
    assert(m_attributes.m_computeCapabilityMajor != 0);

    /* A10x */
    if ((m_attributes.m_computeCapabilityMajor == 8) && (m_attributes.m_computeCapabilityMinor >= 3))
    {
        return true;
    }
    /* TU10x */
    else if ((m_attributes.m_computeCapabilityMajor == 7) && (m_attributes.m_computeCapabilityMinor >= 5))
    {
        return true;
    }

    return false;
}

/*****************************************************************************/
int DistributedCudaContext::RunSubtestDramUtil(void)
{
    int retSt { 0 };

    m_input.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    /* This has always been full speed. load target is ignored */
    m_cudaWorker.SetWorkloadAndTarget(DCGM_FI_PROF_DRAM_ACTIVE, 1.0, true);

    Respond("S\nD 0 1\n");

    double prevDramAct       = 0.0;
    bool eccAffectsBandwidth = EccAffectsDramBandwidth() && (m_attributes.m_eccSupport > 0);
    unsigned int activities  = (m_duration + m_reportInterval / 2.0) / m_reportInterval;
    double perSecond         = 0.0;
    double utilRate          = 0.0;

    for (unsigned int activity = 0; activity <= activities; activity++)
    {
        ReportIntervalSleep();

        double howFarIn      = 1.0 * activity / activities;
        double prevPerSecond = perSecond;
        perSecond            = m_cudaWorker.GetCurrentAchievedLoad();
        prevDramAct          = utilRate;
        utilRate             = perSecond / m_attributes.m_maxMemBandwidth;
        perSecond /= 1000000000.0;

        Respond("T %u %0.3f %0.3f %0.3f %0.3f %0.3f %1u\n",
                activity,
                howFarIn,
                prevDramAct,
                utilRate,
                prevPerSecond,
                m_attributes.m_maxMemBandwidth,
                eccAffectsBandwidth ? 1 : 0);

        bool earlyQuit { false };

        retSt = ReadLnCheck(activity, earlyQuit);

        if ((retSt < 0) || earlyQuit)
        {
            break;
        }
    }

    m_cudaWorker.SetWorkerToIdle();

    // coverity[dead_error_condition] - Leaving in case retSt is nonzero in the future
    Respond((retSt < 0) ? "F\n" : "D 1 1\nP\n");

    return retSt;
}

/*****************************************************************************/
int DistributedCudaContext::RunSubtestGemmUtil(void)
{
    int retSt { 0 };

    double now, startTime; // endTime;

    m_input.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    /* This has always been full speed. load target is ignored */
    m_cudaWorker.SetWorkloadAndTarget(m_testFieldId, 1.0, true, m_physicalGpu->UseCublas());

    /* Run our test, noting we started (S\n) and have completed zero sub-parts
     *  of one (D 0 1\n).
     */
    Respond("S\nD 0 1\n");

    now       = timelib_dsecSince1970();
    startTime = now;
    // endTime   = now + m_duration;
    unsigned int activities = (m_duration + m_reportInterval / 2.0) / m_reportInterval;

    for (unsigned int activity = 0; activity <= activities; activity++)
    {
        ReportIntervalSleep();

        double howFarIn = std::max(1.0, (now - startTime) / m_duration);
        double gflops   = m_cudaWorker.GetCurrentAchievedLoad() / 1000000000.0;

        Respond("T %u %0.3f %0.3f %0.3f\n", activity, howFarIn, gflops, gflops);

        bool earlyQuit { false };

        retSt = ReadLnCheck(activity, earlyQuit);

        if ((retSt < 0) || earlyQuit)
        {
            break;
        }

        now = timelib_dsecSince1970();
    }

    m_cudaWorker.SetWorkerToIdle();

    // coverity[dead_error_condition] - Leaving in case retSt is nonzero in the future
    Respond((retSt < 0) ? "F\n" : "D 1 1\nP\n");

    return retSt;
}

/*****************************************************************************/
int DistributedCudaContext::RunSubtestDataTypeActive(void)
{
    /* Generate per-datatype activity */

    int retSt { 0 };

    auto duration             = m_duration;
    double now                = timelib_dsecSince1970();
    double startTime          = timelib_dsecSince1970();
    double tick               = startTime + m_reportInterval;
    double prevTargetActivity = 0.0;
    unsigned int activities   = (duration + m_reportInterval) / m_reportInterval;
    m_cudaWorker.SetWorkerToIdle();

    m_input.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    Respond("S\nD 0 1\n");

    for (unsigned int activity = 0; activity < activities; activity++)
    {
        double howFarIn       = 1.0 * activity / activities;
        double targetActivity = 1.0; /* Go for 100% for now to match old behavior */

        m_cudaWorker.SetWorkloadAndTarget(m_testFieldId, targetActivity);

        ReportIntervalSleep();

        now = timelib_dsecSince1970();
        if (now > tick)
        {
            Respond("T %u %0.3f %0.3f %2.3f %2.3f %u %u\n",
                    activity,
                    howFarIn,
                    prevTargetActivity,
                    targetActivity,
                    now - startTime,
                    (unsigned int)(howFarIn * m_attributes.m_multiProcessorCount),
                    m_attributes.m_multiProcessorCount);

            tick = now + m_reportInterval;
        }

        prevTargetActivity = targetActivity;

        bool earlyQuit { false };

        retSt = ReadLnCheck(activity, earlyQuit);

        if ((retSt < 0) || earlyQuit)
        {
            break;
        }
    }

    m_cudaWorker.SetWorkerToIdle();

    // coverity[dead_error_condition] - Leaving in case retSt is nonzero in the future
    Respond((retSt < 0) ? "F\n" : "D 1 1\nP\n");

    return 0;
}

/*****************************************************************************/

// Dispatch test to be run.
int DistributedCudaContext::RunTest(void)
{
    int retSt = 0;

    m_message << "CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR: " << m_attributes.m_maxThreadsPerMultiProcessor
              << '\n';

    m_message << "CUDA_VISIBLE_DEVICES: " << m_cudaVisibleDevices.c_str() << '\n';

    m_message << "CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: " << m_attributes.m_multiProcessorCount << '\n';

    m_message << "CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR: "
              << m_attributes.m_sharedMemPerMultiprocessor << '\n';

    m_message << "CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: " << m_attributes.m_computeCapabilityMajor << '\n';

    m_message << "CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: " << m_attributes.m_computeCapabilityMinor << '\n';

    m_message << "CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH: " << m_attributes.m_memoryBusWidth << '\n';

    m_message << "CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE: " << m_attributes.m_maxMemoryClockMhz << '\n';

    m_message << "Max Memory bandwidth: " << std::fixed << std::setprecision(0) << m_attributes.m_maxMemBandwidth
              << " bytes (" << std::setprecision(1) << m_attributes.m_maxMemBandwidth / (1.0e9) << " GiB)" << '\n';

    m_message << "CU_DEVICE_ATTRIBUTE_ECC_SUPPORT: " << ((m_attributes.m_eccSupport > 0) ? "true" : "false") << '\n';

    dcgmReturn_t dcgmReturn = m_cudaWorker.Init(m_device);
    if (dcgmReturn != DCGM_ST_OK)
    {
        DCGM_LOG_ERROR << "m_cudaWorker.Init failed with " << dcgmReturn;
        return -1;
    }

    switch (m_testFieldId)
    {
        case DCGM_FI_PROF_GR_ENGINE_ACTIVE:
            if (m_targetMaxValue)
                retSt = RunSubtestSmOccupancyTargetMax();
            else
                retSt = RunSubtestGrActivity();
            break;

        case DCGM_FI_PROF_SM_ACTIVE:
            if (m_targetMaxValue)
                retSt = RunSubtestSmOccupancyTargetMax();
            else
                retSt = RunSubtestSmActivity();
            break;

        case DCGM_FI_PROF_SM_OCCUPANCY:
            if (m_targetMaxValue)
                retSt = RunSubtestSmOccupancyTargetMax();
            else
                retSt = RunSubtestSmOccupancy();
            break;

        case DCGM_FI_PROF_PCIE_RX_BYTES:
        case DCGM_FI_PROF_PCIE_TX_BYTES:
            retSt = RunSubtestPcieBandwidth();
            break;

        case DCGM_FI_PROF_DRAM_ACTIVE:
            retSt = RunSubtestDramUtil();
            break;

        case DCGM_FI_PROF_PIPE_FP32_ACTIVE:
        case DCGM_FI_PROF_PIPE_FP64_ACTIVE:
        case DCGM_FI_PROF_PIPE_FP16_ACTIVE:
            retSt = m_physicalGpu->UseCublas() ? RunSubtestGemmUtil() : RunSubtestDataTypeActive();
            break;

        case DCGM_FI_PROF_PIPE_TENSOR_ACTIVE:
            retSt = RunSubtestGemmUtil();
            break;

        case DCGM_FI_PROF_NVLINK_RX_BYTES:
        case DCGM_FI_PROF_NVLINK_TX_BYTES:
            retSt = RunSubtestNvLinkBandwidth();
            break;

        default:
            m_error << "A test for fieldId " << m_testFieldId << " has not been implemented yet." << '\n';
            m_input.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            retSt = -1;
            break;
    }

    return retSt;
}


// Are we already running?
bool DistributedCudaContext::IsRunning(void) const
{
    return m_pid != 0;
}


// Run a test in a sub-process.
int DistributedCudaContext::Run(void)
{
    extern std::atomic_bool g_signalCaught;

    int retSt     = 0;
    int exitValue = 0;
    int toChildPipe[2];  // parent writes to this, child reads from this
    int toParentPipe[2]; // parent reads from this, child writes to this

    // TODO: SET GPU CUDA environment here or before entry.

    if ((retSt = pipe(toChildPipe)) != 0)
    {
        return retSt;
    }

    if ((retSt = pipe(toParentPipe)) != 0)
    {
        close(toChildPipe[0]);
        close(toChildPipe[1]);
        return retSt;
    }

    /**
     * We must close any CUDA contexts and file descriptors we might have open.
     * But, we must keep any monitoring DCGM Groups intact.
     */
    Reset(true);

    pid_t pid = fork();

    if (pid < 0) // error
    {
        return pid;
    }

    if (pid != 0) // parent
    {
        if ((retSt = (close(toChildPipe[0]) < 0) || (close(toParentPipe[1]) < 0)))
        {
            close(toParentPipe[0]); // child can't read and will eventually die
            return retSt;
        }

        m_inFd  = toParentPipe[0];
        m_outFd = toChildPipe[1];

        int st = 0;

        // Make I/O in parent non-blocking.
        st = fcntl(m_inFd, F_SETFL, O_NONBLOCK);
        if (st == -1)
        {
            return -1;
        }

        st = fcntl(m_outFd, F_SETFL, O_NONBLOCK);
        if (st == -1)
        {
            return -1;
        }

        m_pid = pid;
        m_input.str("");
        m_input.clear();
        return pid; // monitor and wait for child
    }

    // We are in the child.

    close(STDIN_FILENO);
    close(STDOUT_FILENO);
    close(STDERR_FILENO);

    /* Ask for a SIGHUP when our parent dies */
    prctl(PR_SET_PDEATHSIG, SIGHUP);

    if ((retSt = (close(toChildPipe[1]) < 0) || (close(toParentPipe[0]) < 0)))
    {
        m_error << "child CUDA process could not clean up pipes." << '\n';

        // We can still communicate, so we do not consider this a failure.
    }
    else if (Init(toChildPipe[0], toParentPipe[1]) != DCGM_ST_OK)
    {
        m_error << "failed to initialize, waiting to be killed." << '\n';
        m_failed = true;
    }

    try
    {
        /* Tell parent we are ready for commands and convey our initialization
         * status.
         */

        Respond(m_failed ? "F\n" : "P\n");

        // We process commands until we are told not to or we crash.
        for (char op = '\0'; op != 'X';)
        {
            retSt = ReadLn();

            if ((retSt < 0) || g_signalCaught) // FD reading error or signal interrupt
            {
                /* Parent will see write pipe closed and get SIGCHLD. No point
                 * to setting m_failed, or an error, as process is gone. This
                 * is a bad one. Of course, parent could have died, closing
                 * the write end of the pipe, and that causes us to die.
                 */
                exitValue = (int)DCGM_ST_GENERIC_ERROR;
                break;
            }
            else if (retSt == 0)
            {
                usleep(50);

                continue;
            }

            m_input >> op; // get operation

            switch (op)
            {
                case 'X': // exit
                    break;

                case 'Q': // ignore
                    break;

                case 'R':                        // run
                    m_input >> m_testFieldId;    // get test ID
                    m_input >> m_duration;       // get duration
                    m_input >> m_reportInterval; // get report interval

                    // Get targetMaxValue flag.
                    m_input >> std::boolalpha >> m_targetMaxValue;

                    if ((retSt = RunTest()) < 0) // And run the test!
                    {
                        m_error << "test " << m_testFieldId << " failed to run: " << retSt << '\n';
                    }

                    break;

                case 'E': // get errors -- clear at start of each test & here
                {
                    std::stringstream temp;
                    std::string errors   = m_error.str();
                    unsigned long length = errors.length();

                    Respond("E %lu\n%s", length, errors.c_str());
                    m_error.swap(temp);

                    break;
                }


                case 'M': // get messages -- clear at start of each test & here
                {
                    std::stringstream temp;
                    std::string message  = m_message.str();
                    unsigned long length = message.length();

                    Respond("M %lu\n%s", length, message.c_str());
                    m_message.swap(temp);
                    break;
                }

                case 'A': // extraneous 'A'dvanced sync command.
                    m_input.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                    break;

                default: // unknown command
                    m_error << "unknown command " << op << '\n';
                    break;
            }
        }
    }
    catch (...)
    {
        // catastrophic error
        exitValue = (int)DCGM_ST_GENERIC_ERROR;
    }

    /* Note : No longer calling Reset() here before shutdown because that call hangs */

    /* Make sure our worker thread is inactive before exiting to avoid queuing cuda work
       at the same time the cuda atexit() handlers are running */
    m_cudaWorker.Shutdown();

    /* In the future, we shouldn't exit here, but that will require a more substantial
       refactor of DistributedCudaContext and our forked worker */
    exit(exitValue);
}

} // namespace DcgmNs::ProfTester
