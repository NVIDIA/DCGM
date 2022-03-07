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

#include "Arguments.h"
#include "DcgmProfTester.h"
#include "DistributedCudaContext.h"
#include "Entity.h"

#include <cuda.h>
#include <dcgm_structs.h>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>
namespace DcgmNs::ProfTester
{
/*****************************************************************************/
class PhysicalGpu : public std::enable_shared_from_this<PhysicalGpu>
{
public:
    enum class ValueType
    {
        Int64,
        Double,
        String,
        Blob
    };

    struct entity_id_t
    {
        unsigned int m_gi;
        unsigned int m_ci;

        entity_id_t(unsigned int gi, unsigned int ci)
            : m_gi(gi)
            , m_ci(ci)
        {}
    };

    /*************************************************************************/
    /* ctor/dtor */
    PhysicalGpu(std::shared_ptr<DcgmProfTester> tester, unsigned int gpuId, const Arguments_t::Parameters &parameters);

    ~PhysicalGpu();

    /*************************************************************************/


    /*************************************************************************/
    /*
     * Set (or reset), the test parameters. This allows the Physical GPU to be
     * used to run more than one test. The previous test has to have finished,
     * though.
     */
    void SetParameters(const Arguments_t::Parameters &parameters);

    /*************************************************************************/
    /*
     * Initialize cuda to be able to run (on MIG partitions too, if in MIG
     * mode).
     *
     * @param dcgmHandle: a dcgmHandle_t
     *
     * Returns 0 on success. !0 on failure.
     */
    dcgmReturn_t Init(dcgmHandle_t dcgmHandle);


    /*************************************************************************/
    /**
     * This function is used to associate CUDA contexts for tests with the
     * physical GPU. Either a single call to AddSlice is made, for the whole
     * GPU, or multiple calls identifying the MIG slice via nvmlEntityId.
     *
     * @param entities - map of field values to be captured for this device
     * @param deviceId - CUDA context environment variable to select this device
     *
     * @returns - a shared_ptr to a DistributedCudaContext
     *
     */
    std::shared_ptr<DistributedCudaContext> AddSlice(
        std::shared_ptr<std::map<dcgm_field_entity_group_t, dcgm_field_eid_t>> entities,
        dcgmGroupEntityPair_t &entity,
        const std::string &deviceId);

    /*************************************************************************/
    /* These functions manage the starting and Running of tests and processing
     * test responses.
     */
    dcgmReturn_t StartTests(void); // Start tests
    dcgmReturn_t RunTests(void);   // Run on GPUs with already started workers

    /*
     * Process Response
     *
     * Process a response from a GPU CI on this physical GPU.
     *
     * @param worker - worker CUDA context smart pointer.
     *
     * @returns dcgmReturn_t indicating result.
     */
    dcgmReturn_t ProcessResponse(std::shared_ptr<DistributedCudaContext> worker);

    void AbortChildren(unsigned int workers = 0); // Call before test start.
    void NukeChildren(bool force = false);        // Call after test start.

    bool IsMIG(void) const;                     // Are we MIG mode (call Init first)
    unsigned int GetGpuId(void) const;          // Return GPU Id.
    std::string GetGpuBusId(void) const;        // Get the BUS ID of our GPU
    dcgmHandle_t GetHandle(void) const;         // return DCGM handle
    unsigned short GetFieldId(void) const;      // return field ID
    dcgmFieldGrp_t GetFieldGroupId(void) const; // return field Group ID

    bool GetDcgmValidation(void) const; // Are we validating?
    bool IsSynchronous(void) const;     // Is operation synchronous?

    /*************************************************************************/
    /* These functions are used to track progress on all CUDA context workers.
     *
     * When all have finished, AllFinished() returns true, and when all have
     * reported their Messages and Errors, AllReported() returns true.
     */

    bool WorkerStarted(void) const;  // Did last operation make a worker start?
    bool WorkerRunning(void) const;  // Did last operation make a worker run?
    bool WorkerFinished(void) const; // Did last operation make a worker finish?
    bool WorkerReported(void) const; // Did last operation make a worker report?

    bool AllStarted(void) const;  // Did all workers start (or fail to)?
    bool AllFinished(void) const; // Did all workers finish?
    bool AllReported(void) const; // Did all workers report?
    bool AnyFailed(void) const;   // Did any workers fail to start?

    bool IsValidated(void) const; // Did all workers pass validation?

private:
    // Tick handler type.
    using TickHandlerType = std::function<dcgmReturn_t(size_t index,
                                                       bool valid,
                                                       std::map<Entity, dcgmFieldValue_v1> &values,

                                                       DcgmNs::ProfTester::DistributedCudaContext &context)>;

    /*************************************************************************/

    dcgmReturn_t CreateDcgmGroups(void);  // Create field group to retrieve
    dcgmReturn_t DestroyDcgmGroups(void); // Destroy field group to retrieve

    /* Child process control. */
    dcgmReturn_t CommandAll(bool all, const char *format, ...);

    /* Subtest methods */
    dcgmReturn_t BeginSubtest(std::string testTitle, std::string testTag, bool isLinearTest);

    dcgmReturn_t EndSubtest(void);
    dcgmReturn_t AppendSubtestRecord(double generatedValue, double dcgmValue);

    /*************************************************************************/
    /**
     *
     * Get a single value of the given entity group type and numeric type.
     *
     * @param values        - field values retrieved
     * @param entityGroupID - type of value desired (if MIG)
     * @param type          - type of entity desired (GPU, GI, CI)
     * @param divisor       - amount field value is to be scaled down by
     * @param value         - location to store value
     *
     * @returns             - true if field value found, false otherwise
     */
    bool ValueGet(std::map<Entity, dcgmFieldValue_v1> &values,
                  std::map<dcgm_field_entity_group_t, dcgm_field_eid_t> &entities,
                  dcgm_field_entity_group_t entityGroupId,
                  ValueType type,
                  double divisor,
                  double &value);

    /* Dump retrieved values. */
    void ValuesDump(std::map<Entity, dcgmFieldValue_v1> &values, ValueType type, double divisor);

    /* Validation */
    bool Validate(double expected, double current, double measured, double howFarIn, bool prevValidated);

    /*************************************************************************/
    /* Individual subtests */
    dcgmReturn_t RunSubtestGrActivity(void);
    dcgmReturn_t RunSubtestSmActivity(void);
    dcgmReturn_t RunSubtestSmOccupancy(void);
    dcgmReturn_t RunSubtestPcieBandwidth(void);
    dcgmReturn_t RunSubtestDramUtil(void);
    dcgmReturn_t RunSubtestGemmUtil(void);
    dcgmReturn_t RunSubtestNvLinkBandwidth(void);
    dcgmReturn_t RunSubtestSmOccupancyTargetMax(void);

    /*************************************************************************/
    /*
     * Method to get the best linked peer.
     *
     * @param peerPciBusId  - peer Bus Id.
     * @param nvLinks       - links available.
     *
     * @return DCGM_ST_OK if it succeeded, something else otherwise.
     */
    dcgmReturn_t HelperGetBestNvLinkPeer(std::string &peerPciBusId, unsigned int &nvLinks);

    /*************************************************************************/
    /*
     * Method to check if the current GPU's virtualization mode is supported.
     *
     * @param none
     * @returns DCGM_ST_OK if the virtualization mode is supported or
     *  DCGM_ST_PROFILING_NOT_SUPPORTED if the virtualization mode is not
     *  supported.
     */
    dcgmReturn_t CheckVirtualizationMode(void);

    /*************************************************************************/
    /*
     * Method to check if the current GPU's compute capabilities are
     * unoptimized. This is generally a compute capability issue.
     *
     * @param field Id.
     * @returns true if unoptimized, false otherwise.
     */
    bool IsComputeUnoptimized(unsigned int fieldId);

    /*************************************************************************/
    /*
     * Method to check if the current GPU's hardware capabilities are
     * not determinable.
     *
     * @param field Id.
     * @returns true if limited, false otherwise.
     */
    bool IsHardwareNonDeterministic(unsigned int fieldId);

    /*************************************************************************/
    /*
     * Set Tick Handler.
     *
     * Since the tick handler varies for each subtest, we allow for the
     * setting of a function to deal with it at the time the subtests are
     * dispatched. A virtual function on the parent side of
     * DistributedCudaContext classes would be more elegant, but as this was
     * initially open-coded, at this point, we preserve that style: The subtest
     * dispatcher sets this, generally to an appropriate lambda that is passed
     * the parent side of the worker context.
     */
    void SetTickHandler(TickHandlerType tickHandler);

    /*************************************************************************/
    /*
     * Process Starting Response
     *
     * Process a response from a GPU CI on this physical GPU. Advances to
     * the running state if all workers have started or failed to do so.
     *
     * @param worker - worker CUDA context smart pointer.
     *
     * @returns dcgmReturn_t indicating result.
     */
    dcgmReturn_t ProcessStartingResponse(std::shared_ptr<DistributedCudaContext> worker);

    /*
     * Process Running Response
     *
     * Process a response from a GPU CI on this physical GPU. Advances to
     * the finished state if all workers have started or failed to do so.
     *
     * @param worker - worker CUDA context smart pointer.
     *
     * @returns dcgmReturn_t indicating result.
     */
    dcgmReturn_t ProcessRunningResponse(std::shared_ptr<DistributedCudaContext> worker);

    /*
     * Process FinishedResponse
     *
     * Process a response from a GPU CI on this physical GPU.
     *
     * @param worker - worker CUDA context smart pointer.
     *
     * @returns dcgmReturn_t indicating result.
     */
    dcgmReturn_t ProcessFinishedResponse(std::shared_ptr<DistributedCudaContext> worker);

    /*************************************************************************/

    std::shared_ptr<DcgmProfTester> m_tester; /* Our controlling tester */

    /* Test parameters */

    unsigned int m_gpuId; /* DCGM GPU ID we are testing on. */

    Arguments_t::Parameters m_parameters; /* Operational test parameters */

    /* CUDA contexts (one per MIG slice or one per whole GPU) */
    std::vector<std::shared_ptr<DistributedCudaContext>> m_dcgmCudaContexts;

    unsigned int m_workers { 0 }; /* Worker Slices added */

    dcgmHandle_t m_dcgmHandle { (uintptr_t) nullptr }; /* Host Engine handle */
    dcgmDeviceAttributes_t m_dcgmDeviceAttr {};        /* DCGM device attributes for this GPU */
    dcgmFieldGrp_t m_fieldGroupId { 0 };               /* Fields we are watching for m_groupId */

    uint64_t m_majorComputeCapability { 0 }; /* Compute capability */
    uint64_t m_minorComputeCapability { 0 }; /* Compute capability */

    /* Cache of values that have been fetched so far. */
    std::map<Entity, std::vector<dcgmFieldValue_v1>> m_dcgmValues;

    long long m_sinceTimestamp { 0 }; /* Cursor for DCGM field value fetching */

    /* Subtest stuff */
    bool m_subtestInProgress {}; /* Is there currently a subtest running? */

    bool m_subtestIsLinear {}; /* Is the subtest linear (true) or static value (false)? */

    std::string m_subtestTitle; /* Current subtest's display name. */

    std::string m_subtestTag; /* Current subtest's tag - used for keys and the output filenames. */

    std::vector<double> m_subtestDcgmValues; /* subtest DCGM values */
    std::vector<double> m_subtestGenValues;  /* subtest generated values */

    /* Worker tracking */

    dcgmReturn_t (PhysicalGpu::*m_responseFn)(std::shared_ptr<DistributedCudaContext>);

    bool m_workerStarted { false };  /* Operation made worker start */
    bool m_workerRunning { false };  /* Operation made worker run */
    bool m_workerFinished { false }; /* operation made worker finish */
    bool m_workerReported { false }; /* operation made worker report */

    size_t m_startedWorkers { 0 };  /* workers that started */
    size_t m_failedWorkers { 0 };   /* workers that failed to start */
    size_t m_busyWorkers { 0 };     /* workers that started */
    size_t m_finishedWorkers { 0 }; /* workers that finished */
    size_t m_reportedWorkers { 0 }; /* workers that reported */

    bool m_sawLastWorker { false }; /* last worker started current part */

    /* last worker to start current part */
    std::shared_ptr<DistributedCudaContext> m_lastWorker { nullptr };

    TickHandlerType m_tickHandler; /* Tick handler. */

    bool m_exitRequested { false }; /* whether early exit is requested */
    bool m_valid { true };          /* Whether the test passes. */
};

} // namespace DcgmNs::ProfTester
