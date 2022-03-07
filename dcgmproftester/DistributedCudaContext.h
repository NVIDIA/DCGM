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

#include <dcgm_structs.h>

#include <cstdarg>
#include <cuda.h>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

/* DCGM Distributed Cuda Context
 *
 * This class encapsulates the state of each process executing CUDA code,
 * typically one per MIG GPU instance (or one when MIG is not used).
 *
 * It is replicated between a DCGM parent process and (for each GPU instance)
 * CUDA worker process via a fork made in the DCGM parent process. After
 * forking, communication between the two processes takes place over unnamed
 * pipes created in the parent process prior to the fork.
 *
 * The worker process will respond with "P\n" (pass) or "F\n" fail over the
 * pipe to the parent when it forks. In both cases, it will wait for commands,
 * though the only reasonable one to send is "X\n" (exit). Upon receipt, it
 * will exit, closing its end of communication pipes.
 *
 * To start a test, the parent sends an "R <Test ID> <Duration>
 * <Reporting Interval> <Max Target Value> ...\n" command, with the arbitrary
 * parameters depending on the specific test requested.
 *
 * Test ID is an unsigned integer numeric test identifier.
 *
 * Duration is a double indicating the duration of the test.
 *
 * Reporting Interval is a double indicating the minimum interval at which
 * status reports (Ticks) are sent back from the worker to the parent process.
 * Generally, waiting for CUDA operations to complete sets a lower bound on
 * what this is, regardless of what is specified. But, for rapid CUDA operations
 * it can be used to regulate the reporting rate and bandwidth across the pipe
 * to the parent process.
 *
 * Max Target Value is a boolean (true | false) indicating whether the test
 * should ramp up slowly or not.
 *
 * These encompass the parameters that are sent to all tests, as they are common
 * Additional, test-specific parameters can follow them, up to the newline
 * ending the command.
 *
 * When the test successfully starts, the worker process sends back "S\n" on
 * the pipe to the parent process.
 *
 * When a sub-test completes, the worker process sends back "D <part> <parts>\n"
 * to the parent process. Here, <parts> are the number of distinct sub-tests
 * of the test, and <part> is the ordinal of the one that just finished
 * (starting from 1). Within a test, the format of data that is sent back on
 * a "Tick" is consistent within a sub-test, but may vary between sub-tests.
 *
 * While a sub-test is running, no more frequently than indicated by the
 * Reporting Interval, "Tick" status reports will be sent from the worker
 * process to the the parent. Each "Tick" takes the form "T ...\n" where the
 * format depends on the test and sub-test.
 *
 * When a test completes, it returns either "P\n" (pass)  or "F\n" (fail) and
 * awaits further commands. At this point a new test can be requested with a
 * "R ...\n" command. More commonly, an "X\n" command will be sent to cause
 * the worker process to exit. It is not expected that the parent process will
 * seek to recover the exit code and ignore SIGCHLD so as to not create zombies
 * if the parent exists before any workers.
 *
 * As a test runs, it may accumulate informative messages or outright errors
 * (more commonly when it fails with an "F\n" response). After the test
 * completes, these can be destructively interrogated by the parent process
 * using the "M\n" (messages) or "E\n" (errors) commands. The worker process
 * responds with "M <length>\n<message of length specified>" or "E <length>\n
 * <error of length specified>", respectively. These commands should only be
 * sent once a test completes to gather forensics.
 *
 * Because reading from, or writing to, a closed pipe results in a SIGPIPE
 * signal, uncerimoniously closing the pipe (say on the unexpected exit of a
 * crashed peer process) will cause the current process to crash instead of
 * leaving behind a zombie. This is currently intentional. It should not happen.
 */

namespace DcgmNs::ProfTester
{
class PhysicalGpu;
struct Entity;

class DistributedCudaContext
{
public:
    // Default constructor.
    DistributedCudaContext() = delete;

    // Constructor expecting device identification.
    /**
     * \brief Establish CUDA connection to indicated device (or MIG).
     *
     * @param cudaVisibleDevices string to set CUDA_VISIBLE_DEVICES to.
     *
     * @return None.
     */
    DistributedCudaContext(std::shared_ptr<DcgmNs::ProfTester::PhysicalGpu>,

                           std::shared_ptr<std::map<dcgm_field_entity_group_t, dcgm_field_eid_t>>,
                           const dcgmGroupEntityPair_t &,
                           const std::string &);

    // Move constructors.
    DistributedCudaContext(DistributedCudaContext &) = delete;  //!< no copy
    DistributedCudaContext(DistributedCudaContext &&) noexcept; //!< but move

    // Destructor.
    ~DistributedCudaContext();

    // Assignments operators.
    DistributedCudaContext &operator=(DistributedCudaContext &) = delete;
    DistributedCudaContext &operator                            =(DistributedCudaContext &&) noexcept;

    /** @name DeviceID
     * Device moniker and physical GPU relationships.
     * @{
     */

    /**
     * \brief Return Physical GPU.
     *
     * @return a PhysicalGpu shared pointer is returned.
     */
    std::shared_ptr<PhysicalGpu> GetPhysicalGpu(void);

    /**
     * \brief Return Device name.
     *
     * @return a Device name string reference.
     */
    const std::string &Device(void) const;

    /**
     * \brief Return Entity information.
     *
     * @return reference to a dcgmGroupEntityPair.
     *
     * This is basically another version of the DeviceId, but is not mangled
     * for NvLink tests to be empty.
     */
    const dcgmGroupEntityPair_t &EntityId(void) const;

    /**
     * \brief Return Entity information.
     *
     * @return a
     * share_ptr<std::map<dcgm_field_entity_group_t, dcgm_field_eid_t>> map of
     * field entities to watch and retrieve.
     */
    std::map<dcgm_field_entity_group_t, dcgm_field_eid_t> &Entities(void) const;

    /**@}*/

    /** @name StateManagement
     * State management.
     * @{
     */

    /**
     * \brief Reinitialize reinitializes the worker. Called on main side.
     */
    void ReInitialize(std::shared_ptr<std::map<dcgm_field_entity_group_t, dcgm_field_eid_t>>, const std::string &);

    /**
     * \brief Initialize worker side.
     *
     * This initializes communications on the worker process. It should only
     * be called in the worker process.
     *
     * @param inFd  An int specifying the input file descriptor for commands.
     * @param outFd An int specifying the output file descriptor for responses.
     *
     * @return A dcgmReturn_t is returned indicating success or failure.
     */
    dcgmReturn_t Init(int inFd, int outFd);

    /**
     * \brief Reset state of DistributedCudaContext.
     *
     * This resets the state of the DistributedCudaContext. File Descriptors are
     * closed, messages and errors are erased, and everything is restored to
     * the default state.
     *
     * @param keepDcgmGroups A bool to indicate if dcgmfield groups should be
     *                       preserved. Default is false.
     *
     * @return None.
     */
    void Reset(bool keepDcgmGroups = false);

    /**
     * \brief Run the worker.
     *
     * This is designed to be called in the parent process to start running
     * the worker process. It establishes the communication pipes and file
     * descriptors in both parent and worker process.
     *
     * @return An int is returned, negative for error, zero in the worker
     *          process, and the process ID in the parent process.
     */
    int Run(void);

    /**
     * \brief Is the worker already running?
     *
     * This is designed to be called in the parent process to determine if
     * the worker is already running. If so, there is no need to start it.
     *
     * @return A bool is returned, true if the worker is running, false
     * otherwise.
     */
    bool IsRunning(void) const;

    /** @name InterProcs
     * Inter-process pipe I/O methods
     * @{
     */

    /**
     * \brief Send a command from parent to worker.
     *
     * This send a command from the parent process to the worker process. It
     * is not normally used directly but from within a wrapper function that
     * also accepts variable argument lists.
     *
     * @param format A vdprintf for the subsequent arguments.
     * @param args   Arguments for the command.
     *
     * @return An int is returned reflective of the underlying vdprintf. In
     *          general, negative indicates an error.
     */
    int Command(const char *format, std::va_list args);

    /**
     * \brief Send a command from parent to worker.
     *
     * This send a command from the parent process to the worker process.
     *
     * @param format A vdprintf for the subsequent arguments.
     * @param ...    Arguments for the command.
     *
     * @return An int is returned reflective of the underlying vdprintf. In
     *          general, negative indicates an error.
     */
    int Command(const char *format, ...);

    /**
     * \brief Read from input pipe into the input stringstream.
     *
     * This reads from the input pipe (command on the worker side, response on
     * the parent side) into the input stringstream.
     *
     * @param toRead The number of characters to read. This will return only
     *               on error or when the of bytes indicated has been read.
     *               It really  is designed to greedily get the response from
     *               M(essage) and E(rror) commands.
     *
     * @return true on completion, false on error.
     */
    bool Read(size_t toRead);

    /**
     * \brief Read a line from input pipe to input stringstream.
     *
     * This reads a line from the input pipe (command on the worker process,
     * response on the parent process) into the input strinstream. On a non-
     * blocking pipe it can return without a complete line read. The parent
     * process is expected to read responses non-blockingly and the worker
     * process is expected to read commands blockingly.
     *
     * @return An int is returned indicative of status: negative on error (
     *          other than no data on a non-blocking file descriptor), zero
     *          on no complete line (only on non-blocking file descriptor in
     *          the parent process), and non-zero (generally 1), when a line
     *          is available.
     */
    int ReadLn(void);

    /**
     * \brief Get the input file descriptor.
     *
     * This returns the input file descriptor. It is intended to aid in the
     * setting of file descriptor masks for select() operations in the parent
     * process to know which file descriptors to monitor for data. It is not
     * intended to be used to get a file descriptor for direct I/O.
     *
     * @return An int file descriptor is returned, negative on error.
     */
    int GetReadFD(void) const;

    /**
     * \brief Return a reference to the input stringstream.
     *
     * A reference to the input (command for worker processes, response for
     * parent process) stringstream is returned so one can parse information
     * out of it. Generally one calls Read() or ReadLn() to retrieve data from
     * the input file descriptor to the input stringstream, and then parses
     * from that.
     *
     * @return A reference to the input stringstream is returned.
     */
    std::stringstream &Input(void);

    /**@}*/

    /** @name TestParts
     * Test part functions.
     * @{
     */

    /**
     * \brief Set the part and number of parts for the current sub-test.
     *
     * This is intended to be called by the parent process to keep track of
     * the current sub-test part and number of parts. The Tick Handler for
     * a given test will make use of this information to know how to parse
     * Tick responses from the test. This is typically called in the "D"
     * response handling to update the parent side of the DistributedCudaContext
     * so the current TickHandler lambda can make use of it to parse subsequent
     * Ticks properly.
     *
     * @param part  An unsigned int indicating the current part.
     * @param parts An unsigned int indicating the total number of parts.
     *
     * @return None.
     */
    void SetParts(unsigned int part, unsigned int parts);

    /**
     * \brief Get the part and number of parts for the current sub-test.
     *
     * This is intended to be called by the parent process Tick Handler lambda
     * to find out what sub-test part and total parts of a test a CUDA worker
     * process is in. That determines how the Tick Handler should parse Tick
     * notifications.
     *
     * @param part  An unsigned int reference indicating the current part.
     *
     * @param parts An unsigned int reference indicating the total number of
     *              parts.
     *
     * @return None.
     */
    void GetParts(unsigned int &part, unsigned int &parts) const;

    /**@}*/

    /** @name TickManagement
     * Tick state management.
     * @{
     */

    /**
     * \brief Whether it is first for a sub-test part.
     *
     * This is intended to be called in a parent process Tick Handler for it
     * to tell whether it is the first tick of a sub-test part so it can do
     * any per-sub-test part processing.
     *
     * @return Returns true if this is the first tick in a sub-test part.
     */
    bool IsFirstTick(void) const;

    /**
     * \brief Request an early, orderly, exit.
     *
     * This is intended to be called if we wish the test to exit early, that is
     * we have a short-circuit situation where we declare a pass as soon as
     * we see a successful validation after running long enough.
     *
     * @return None
     */
    void Exit(void);

    /**
     * \brief Tell this worker it is finished.
     *
     * We call this if there is data buffered to be processed and to not
     * try and read data from the pipe first.
     */
    void SetFinished(void);

    /**
     * \brief Determine if we are finished.
     *
     * We call this to see if we have finished processing data.
     */
    bool Finished(void) const;

    /**
     * \brief Note that this worker it has failed.
     *
     * The main process notes this worker is in a failed state.
     */
    void SetFailed(void);

    /**
     * \brief Note that this worker has been restored from a failed state.
     *
     * The main process notes this worker is in a non-failed state.
     */
    void ClrFailed(void);

    /**
     * \brief Determine if we have failed.
     *
     * We call this to see if we are in a failed state.
     */
    bool Failed(void) const;

    /**
     * \brief Set this worker's synchronized tries.
     *
     * The main process sets the number of activity level tries available.
     */
    void SetTries(unsigned int tries);

    /**
     * \brief Get worker's activity tries available.
     *
     * We call this to see if we can try again.
     */
    unsigned int GetTries(void) const;

    /**
     * \brief Set this worker's validated status.
     *
     * The main process sets validated status of the last test.
     */
    void SetValidated(bool validated);

    /**
     * \brief Get worker's validated status.
     *
     * Returns the last test validated status.
     */
    bool GetValidated(void) const;

    /**@}*/

    /** @name Field value retrieval.
     * Handles field value retrieval.
     * @{
     */

    // Retrieve field group values
    dcgmReturn_t GetLatestDcgmValues(std::map<Entity, dcgmFieldValue_v1> &);

    /**@}*/

private:
    /** @name ExceptionClass
     * Exception class.
     * @{
     */

    /**
     * \brief An exception to be thrown on a command error.
     *
     * We do not try to catch this exception. We WANT to crash and exit. This
     * will cause the pipes between parent and workers to close and the
     * workers to either crash on SIGPIPE or exit. This is used in extremis
     * so that all processes terminate.
     */
    class RespondException : public std::runtime_error
    {
    private:
        int m_error { 0 }; //!< keep track of error if interested

    public:
        /**
         *
         * \brief Constructor accepting error.
         *
         * We just take note of the error.
         *
         * @param An int representing the communication error.
         *
         * @return None.
         */
        explicit RespondException(int error)
            : std::runtime_error("")
            , m_error(error)
        {}

        /**
         * \brief No assignment operator.
         */
        RespondException &operator=(const RespondException &) = delete;

        /**
         * \brief Retrieve error.
         *
         * If this exception is caught, one can retrieve the error.
         *
         * @return An integer representing the error is returned.
         */
        int Error(void)
        {
            return m_error;
        }
    };

    /**@}*/

    /** @name UtilityFunctions
     * Utility functions.
     * @{
     */

    /**
     * \brief LoadModule loads the CUDA module.
     *
     * @return A dcgmReturn_t is returned indicating success or failure.
     */
    dcgmReturn_t LoadModule(void);

    /**
     * \brief ReadLnCheck checks for synchronous operation commands.
     *
     * Activity is decremented if we are to stay at the current activity level,
     * (0 return value), and a return value of -1 indicates an error, and +1
     * a request to proceed to the next activity level.
     *
     * @return An int is returned indicating synchronization success (>0),
     * repetition (= 0), or failure (<0).
     */
    int ReadLnCheck(unsigned int &activity);

    /**
     * \brief Move DistributedCudaContext rvalue into another.
     *
     * This exists for the benefit of move constructors and assignment
     * operators.
     *
     * @param other An rvalue reference to a DistributedCudaContext.
     *
     * @return None.
     */
    void MoveFrom(DistributedCudaContext &&other);

    /**
     *
     * \brief Respond from the worker to the parent process.
     *
     * This is really just a convenient wrapper around Command, which might
     * be confusing to use when sending a response, but the pipes are
     * symmetrical between parent and worker.
     *
     * @param format A char *vdprintf format string.
     *
     * @param ... parameters to be sent back.
     *
     * @ returns Nothing. If we can't respond, an exception is thrown.
     */
    void Respond(const char *format, ...);

    /**
     * \brief Main CUDA kernel.
     *
     * This calls the main CUDA kernel.
     *
     * @param numSms       Number of SMs to use.
     * @param threadsPerSm Threads per SM.
     * @param runForUsecs  Microseconds to run for,
     *
     * @return An int representing the result is returned.
     */
    int RunSleepKernel(unsigned int numSms, unsigned int threadsPerSm, unsigned int runForUsec);

    /**
     * \brief Helper to return whether ECC adds overhead to dram bandwidth for this SKU.
     *
     * Uses m_attributes
     *
     * @return true if ECC affects bandwidth. false if not.
     */
    bool EccAffectsDramBandwidth(void);

    /**@}*/

    /** @name Field value retrieval.
     * Routines to establish and destroy field value retrieval contexts.
     * @{
     */

    dcgmReturn_t CreateDcgmGroups(void);  // Create field group to retrieve
    dcgmReturn_t DestroyDcgmGroups(void); // Destroy field group to retrieve

    /**@}*/

    /** @name Subtests
     * Individual sub-tests.
     * @{
     */

    int RunSubtestSmOccupancyTargetMax(void);
    int RunSubtestSmOccupancy(void);
    int RunSubtestSmActivity(void);
    int RunSubtestGrActivity(void);
    int RunSubtestPcieBandwidth(void);
    int RunSubtestNvLinkBandwidth(void);
    int RunSubtestDramUtil(void);
    int RunSubtestGemmUtil(void);

    /**@}*/

    /** @name TestDiscriminator
     * Test discriminator top dispatch tests.
     * @{
     */

    /**
     *
     * \brief Run the test indicated by m_testFieldId.
     */
    int RunTest(void);

    /**@}*/

    /** @name DataMembers
     * Data members representing state.
     * @{
     */

    // Physical GPU.
    std::shared_ptr<PhysicalGpu> m_physicalGpu;

    // Group of GPUs we are watching.
    dcgmGpuGrp_t m_groupId { 0 };

    // Cursor for DCGM field value fetching.
    long long m_sinceTimestamp { 0 };

    // Cache of values that have been fetched so far.
    std::map<Entity, std::vector<dcgmFieldValue_v1>> m_dcgmValues;

    // CUDA_VISIBLE_DEVICES environment variable to set in worker.
    std::string m_cudaVisibleDevices {};

    // Input stream, filled by Read*. Called from Run() in worker.
    std::stringstream m_input {};

    // Message and error streams, filled by subtests, above.
    std::stringstream m_message {}; //!< message to return
    std::stringstream m_error {};   //!< error to return

    // Communication pipe file descriptors, set by Init().
    int m_inFd { -1 };  //!< file descriptor we read from
    int m_outFd { -1 }; //!< file descriptor we write to

    // Primary CUDA attributes.
    CUdevice m_device { 0 };          //!< Cuda ordinal of the device to use
    CUcontext m_context { nullptr };  //!< Cuda context
    CUfunction m_cuFuncWaitNs {};     //!< Pointer to waitNs() CUDA kernel
    CUfunction m_cuFuncWaitCycles {}; //!< Pointer to waitCycles() CUDA kernel

    CUmodule m_module { nullptr }; //!< .PTX file that belongs to m_context

    // Simple attributes, computed by Init().
    struct
    {
        int m_maxThreadsPerMultiProcessor {}; //!< threads per multiprocessor
        int m_multiProcessorCount {};         //!< multiprocessors
        int m_sharedMemPerMultiprocessor {};  //!< shared mem per multiprocessor
        int m_computeCapabilityMajor {};      //!< compute capability major num.
        int m_computeCapabilityMinor {};      //!< compute capability minor num.
        int m_computeCapability {};           //!< combined compute capability
        int m_memoryBusWidth {};              //!< memory bus bandwidth
        int m_maxMemoryClockMhz {};           //!< max. memory clock rate (MHz)
        double m_maxMemBandwidth {};          //!< max. memory bandwidth
        int m_eccSupport {};                  //!< ECC support enabled.
    } m_attributes;

    //<! Entity that describes GPU (non-MIG) or GI (MIG)
    dcgmGroupEntityPair_t m_entity;

    //<! field entities to watch and check.
    std::shared_ptr<std::map<dcgm_field_entity_group_t, dcgm_field_eid_t>> m_entities;

    bool m_isInitialized { false }; //<! whether initialized
    bool m_failed { false };        //<! whether passed startup

    // These only make sense on the parent side, controlling the worker.
    int m_pid { 0 };            //<! worker process pid
    unsigned int m_tries { 0 }; //<! synchronous tries available
    unsigned int m_part { 0 };  //<! worker process part complete
    unsigned int m_parts { 0 }; //<! worker process total parts
    bool m_tick { false };      //<! whether tick seen in current part;
    bool m_firstTick { false }; //<! whether this is first tick of part;
    bool m_wait { false };      //<! wait to process a tick
    bool m_buffered { false };  //<! we have buffered data to process.
    bool m_finished { false };  //<! worker is finished
    bool m_validated { true };  //<! last test passed

    /**@}*/

    /** @name PerTestParameters
     * Per-test parameters.
     * These are not moved between instances, and change for every test.
     * @{
     */

    /**
     * Field ID we are testing. This will determine which subtest gets called.
     */
    unsigned int m_testFieldId { 1002 };

    double m_duration { 30.0 };      //!< Test duration in seconds.
    double m_reportInterval { 1.0 }; //!< Test report interval in seconds.
    unsigned int m_syncCount { 0 };  //!< Times we try to reach activity level.

    /**
     * Whether (true) or not (false) we should just target the maximum value
     * for m_testFieldId instead of stair stepping from 0 to 100%.
     */
    bool m_targetMaxValue { false };

    /**@}*/
};

} // namespace DcgmNs::ProfTester
