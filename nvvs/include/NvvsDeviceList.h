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
#ifndef NVVSDEVICELIST_H
#define NVVSDEVICELIST_H

#include "Plugin.h"
#include <vector>

/*****************************************************************************/
#define NVVS_DL_AFFINITY_SIZE 16

/* Struct for storing the device's state */
typedef struct nvvs_device_state_t
{
    int populated;            /* Has this struct already been populated? Prevents overwriting
                                 1=Yes. 0=No */
    int computeModeSupported; /* 1 yes. 0 no. Following fields are only valid for == 1 */
    unsigned int computeMode;

} nvvs_device_state_t, *nvvs_device_state_p;

/*****************************************************************************/
/* Interface class to represent a single production/mock NVVS GPU */
class NvvsDeviceBase
{
public:
    virtual ~NvvsDeviceBase() = default;

    virtual int Init(std::string const &testName, unsigned int gpuId) = 0;
    virtual int RestoreState(std::string const &testName)             = 0;
    virtual unsigned int GetGpuId()                                   = 0;
};

/*****************************************************************************/
/* Class to represent a single NVVS GPU */
class NvvsDevice : public NvvsDeviceBase
{
public:
    NvvsDevice(Plugin *plugin);
    ~NvvsDevice();

    /*************************************************************************/
    /* Initialize this object from its GPU id, saving the current
     * state in the process
     *
     * Returns 0 on success.
     *         Nonzero on failure
     **/
    int Init(std::string const &testName, unsigned int gpuId) override;

    /*************************************************************************/
    /*
     * Back up the state of this device to the passed in struct for later
     * restoration or comparison to an already-saved struct
     *
     * Returns 0 on success.
     *        !0 on error
     *
     */
    int SaveState(std::string const &testName, nvvs_device_state_t *savedState);


    /*************************************************************************/
    /*
     * Get the GPU id associated with this device
     *
     */
    unsigned int GetGpuId() override;

    /*
     * Try to set the CPU affinity of this device
     *
     * Returns 0 on success.
     *        !0 on error
     */

    /*************************************************************************/
    /*
     * Restore any state that has changed since the object was instantiated
     *
     * Returns: 0 if there was no state to restore
     *         <0 on error
     *         >0 if state was restored
     *
     */
    int RestoreState(std::string const &testName) override;

    /*************************************************************************/
    /*
     * Reports on whether this NVVS device is currently considered idle
     *
     * idleTemp: Temperature that is considered idle. Ex: 50.0 = 50 celcius
     * idlePowerWatts: Power draw that is considered idle. Ex: 100.0 = 100 watts
     *
     * Returns: 1 if device is currently considered idle
     *          0 if device is NOT currently considered idle
     *         <0 on error
     *
     */
    int IsIdle(double idleTemp, double idlePowerWatts);

    /*************************************************************************/
    /*
     * Wrapper functions for logging to our associated plugin or the log file
     * if we don't have an associated plugin
     *
     * failPlugin: If this is 1, we will either set the plugin object to be failed
     *             if the plugin object is non-null or we will throw an exception
     *             to be caught at a higher level. 0=just log it
     */

    void RecordWarning(std::string const &testName, DcgmError const &d, int failPlugin);
    void RecordInfo(std::string const &testName, char const *logText);

    /*************************************************************************/

private:
    unsigned int m_gpuId; /* GPU id */
    // int m_cudaIndex;          /* Cuda index of the GPU */

    Plugin *m_plugin; /* Plugin object to log to */
    // struct cudaDeviceProp m_cudaDeviceProp;

    nvvs_device_state_t m_savedState; /* Saved state of the GPU when we first
                                         called Init() */
};

/*****************************************************************************/
/* Class to represent all of the devices our tests will run on */
class NvvsDeviceList
{
public:
    /*************************************************************************/
    /* Constructor.
     *
     * plugin: Plugin object to log to. NULL=not inside plugin
     *
     */
    explicit NvvsDeviceList(Plugin *plugin);

    // Avoid auto-generating the copy constructor/copy assignment as we use std::vector<std::unique_ptr<NvvsDeviceBase>>
    NvvsDeviceList(NvvsDeviceList const &)            = delete;
    NvvsDeviceList &operator=(NvvsDeviceList const &) = delete;

    NvvsDeviceList(NvvsDeviceList &&)            = default;
    NvvsDeviceList &operator=(NvvsDeviceList &&) = default;

    /*************************************************************************/
    /* Destructor */
    ~NvvsDeviceList();

    /*************************************************************************/
    /*
     * Initialize the object from a list of GPU ids
     *
     * Returns: 0 on success
     *         <0 on error
     */
    int Init(std::string const &testName, std::vector<unsigned int> gpuIds);

    /*************************************************************************/
    /*
     * Wrapper functions for logging to our associated plugin or the log file
     * if we don't have an associated plugin
     *
     * failPlugin: If this is 1, we will either set the plugin object to be failed
     *             if the plugin object is non-null or we will throw an exception
     *             to be caught at a higher level. 0=just log it
     */

    void RecordWarning(std::string const &testName, DcgmError const &d, int failPlugin);
    void RecordInfo(std::string const &testName, char const *logText);

    /*************************************************************************/
    /*
     * Restore any device state that has changed since the object was instantiated
     *
     * failOnRestore: Should a fatal error be set in the plugin/thrown if any state
     *                was left changed? 1=yes. 0=no
     *
     * Returns: 0 if there was no state to restore
     *         <0 on error
     *         >0 if state was restored
     *
     */
    int RestoreState(std::string const &testName, int failOnRestore);

    /*************************************************************************/
    /*
     * Wait for all GPUs managed by this object to return to idle
     *
     * timeout: Timeout in seconds to wait before returning regardless of idle
     *          state. <0.0 = use default timeout (currently 60 seconds)
     * idleTemp: Idle temperature (degrees celcius) to be below to be considered idle.
     *           <0.0 = pick a reasonable default (currently 65.0)
     * idlePowerWatts: Idle power (watts) to be below to be considered idle.
     *           <0.0 = pick a reasonable default (currently 100.0)
     *
     * Returns: 0 if reached idle state
     *          1 if timed out before reaching idle state
     *         <0 on other error
     *
     */
    int WaitForIdle(double timeout, double idleTemp, double idlePowerWatts);

    /*************************************************************************/

private:
    Plugin *m_plugin; /* Plugin object to log to. Can be null */
    std::vector<std::unique_ptr<NvvsDeviceBase>> m_devices;
    friend class NvvsDeviceListTest;
};

/*****************************************************************************/

#endif // NVVSDEVICELIST_H
