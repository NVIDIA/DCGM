/*
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

#include <map>

#include <DcgmCoreProxy.h>
#include <DcgmNvSwitchManagerBase.h>
#include <DcgmWatchTable.h>
#include <dcgm_nvsdm.h>
#include <dcgm_structs.h>

namespace DcgmNs
{

struct NvsdmPort
{
    unsigned int id;
    nvsdmPort_t port;
    dcgmNvLinkLinkState_t state;
    unsigned int nvsdmDeviceId;
    unsigned int num;
    uint16_t lid;
    uint64_t guid;
    uint8_t gid[16];
};

struct NvsdmDevice
{
    unsigned int id;
    nvsdmDevice_t device;
    std::string longName;
    unsigned int portIds[DCGM_NVLINK_MAX_LINKS_PER_NVSWITCH];
    unsigned int numOfPorts;
    uint16_t devID;
    uint32_t vendorID;
    uint64_t guid;
};

class DcgmNvsdmManager : public DcgmNvSwitchManagerBase
{
public:
    /*************************************************************************/
    explicit DcgmNvsdmManager(dcgmCoreCallbacks_t *dcc);

    /*************************************************************************/
    ~DcgmNvsdmManager() override;

    /*************************************************************************/
    /*
     * Attach to NVSDM and scan for NvSwitches + link statuses
     *
     * Returns DCGM_ST_OK on success.
     *         DCGM_ST_? on error.
     */
    dcgmReturn_t AttachToNvsdm();

    /*************************************************************************/
    /*
     * Detach from NVSDM, unloading its driver.
     *
     * Returns DCGM_ST_OK on success.
     *         DCGM_ST_? on error.
     */
    dcgmReturn_t DetachFromNvsdm();

    /*************************************************************************/
    /**
     * Read switch status for all switches and update their information
     */
    dcgmReturn_t ReadNvSwitchStatusAllSwitches() override;

    /*************************************************************************/
    /**
     * Update fields for NVSWITCH from NSCQ library.
     *
     * @param fieldId[in]  - Field id to update.
     * @param buf[out]     - DcgmFvBuffer type which needs to be populated with fieldId value.
     * @param entities[in] - Vector of dcgm_field_update_info_t to access entityGroupId and entityId.
     * @param now[in]      - Current time
     *
     * Returns DCGM_ST_OK on success.
     *         DCGM_ST_? on error.
     */
    dcgmReturn_t UpdateFieldsFromNvswitchLibrary(unsigned short fieldId,
                                                 DcgmFvBuffer &buf,
                                                 const std::vector<dcgm_field_update_info_t> &entities,
                                                 timelib64_t now) override;

    /*************************************************************************/
    /**
     * Initialize Switch Manager
     *
     * Calling this method creates an NVSDM session for communicating with
     * switches and NVLINKs
     *
     *
     * @return DCGM_ST_OK:            Successfully initialized
     *         DCGM_ST_BADPARAM:      Already initialized
     *         DCGM_ST_NVML_ERROR:    NVSDM returned an error during session creation
     */
    dcgmReturn_t Init() override;

    /**
     * @brief Pauses the Switch Manager.
     *
     * When paused, the Switch Manager will detach from NVSDM and not update any fields.
     * The paused state is not considered as uninitialized and any update function will return \c DCGM_ST_STALE_DATA
     * instead of \c DCGM_ST_UNINITIALIZED.
     *
     *
     * @return \c DCGM_ST_OK:            Successfully paused
     * @return \c DCGM_ST_UNINITIALIZED: Switch Manager is not initialized
     */
    dcgmReturn_t Pause() override;
    /**
     * @brief Resumes the Switch Manager.
     *
     * When resumed, the Switch Manager will reattach to NVSDM.
     *
     *
     * @return \c DCGM_ST_OK:           Successfully resumed
     * @return Error code similar to \c Init() otherwise
     */
    dcgmReturn_t Resume() override;

    /*************************************************************************/
    /**
     * Get the name of the active backend
     *
     * @param msg[in/out]          - Message object querying backend name
     *
     * Returns DCGM_ST_OK on success.
     *         DCGM_ST_? on error.
     */
    dcgmReturn_t GetBackend(dcgm_nvswitch_msg_get_backend_t *msg) override;

#ifndef DCGM_NVSWITCH_TEST // Allow nvsdm tests to peek in
protected:
#endif
    std::vector<NvsdmDevice> m_nvsdmDevices;                // NVSDM Devices
    std::vector<NvsdmPort> m_nvsdmPorts;                    // NVSDM Ports
    bool m_attachedToNvsdm = false;                         // Have we attached to nvsdm yet? */
    DcgmNvSwitchError m_fatalErrors[DCGM_MAX_NUM_SWITCHES]; // Fatal errors. Max 1 per switch
    bool m_paused                = false;                   // Is the Switch Manager paused?
    unsigned int m_numNvsdmPorts = 0;                       // Number of entries in m_nvsdmPorts that are valid

    /*************************************************************************/
    /**
     * Returns true if the specified switch id is present on the host, false otherwise
     */
    bool IsValidNvSwitchId(dcgm_field_eid_t entityId);

    /*************************************************************************/
    /**
     * Returns true if the specified link id is present on the host, false otherwise
     */
    bool IsValidNvLinkId(dcgm_field_eid_t entityId);

    /*************************************************************************/
    /**
     * Handles composite fieldIds to aggregate telemetry values.
     */
    dcgmReturn_t HandleCompositeFieldId(const dcgm_field_entity_group_t entityGroupId,
                                        const unsigned int entityId,
                                        unsigned short fieldId,
                                        nvsdmTelemParam_t &param,
                                        timelib64_t now,
                                        DcgmFvBuffer &buf);

    /*************************************************************************/
    /**
     * Returns a pointer to the internal NvSwitch object by looking up its
     * physical ID (entityId). Returns nullptr on error.
     *
     * For links (entityGroupId==DCGM_FE_LINK),
     *
     * @return  Pointer to the internal NvSwitch object on success
     *          nullptr on error
     */
    dcgm_nvswitch_info_t *GetNvSwitchObject(dcgm_field_entity_group_t entityGroupId,
                                            dcgm_field_eid_t entityId) override;

    /*************************************************************************/
    /**
     * Scan the NVSDM driver for NvSwitches and add any new switches to m_nvSwitches.
     * This also sets the link state for each port of the nvSwitch.
     */
    dcgmReturn_t AttachNvSwitches();


    /*************************************************************************/
    /**
     * Read link status for each link belonging to any switch and update switch
     * information accordingly
     */
    dcgmReturn_t ReadLinkStatesAllSwitches() override;

    /*************************************************************************/
    /**
     * Read fatal errors for all switches
     */
    dcgmReturn_t ReadNvSwitchFatalErrorsAllSwitches() override;

    /*************************************************************************/
    /**
     * Helper to buffer blank values for all affected entities of a fieldId
     */
    void BufferBlankValueForAllEntities(unsigned short fieldId,
                                        DcgmFvBuffer &buf,
                                        const std::vector<dcgm_field_update_info_t> &entities);

    /*************************************************************************/
    /**
     * Scan the NVSDM driver for nvsdmPorts and add any new Ports to m_nvsdmPorts.
     */
    dcgmReturn_t AttachNvLinks();

    /*************************************************************************/
    /**
     * Populate linkIds with the ids of each NvLink present on the system.
     *
     * @param count[in/out]    - passed in as the maximum number of link ids that can be placed in linkIds. Set
     *                           to the number of link ids placed in linkIds
     * @param linkIds[out]     - populated with the physical ids of the NvLinks on the host.
     * @param flags            - specify a characteristic to get a subset of link ids placed in linkIds
     *                           (unused for now)
     *
     * @return DCGM_ST_OK:                all desired link ids added to buffer
     *         DCGM_ST_INSUFFICIENT_SIZE: the buffer wasn't big enough to add all desired link ids
     */
    dcgmReturn_t GetNvLinkList(unsigned int &count, unsigned int *linkIds, int64_t flags) override;

    /*************************************************************************/
    /**
     * Process a dcgm_nvswitch_msg_get_entity_status_t message.
     */
    dcgmReturn_t GetEntityStatus(dcgm_nvswitch_msg_get_entity_status_t *msg) override;

    /**
     * @brief Returns the connection status of the Switch Manager.
     * @return \c ConnectionStatus::Ok:           The Switch Manager is connected to the NVSDM
     * @return \c ConnectionStatus::Disconnected: The Switch Manager is not connected to the NVSDM
     * @return \c ConnectionStatus::Paused:       The Switch Manager is paused and not connected to the NVSDM
     */
    ConnectionStatus CheckConnectionStatus() const override;
};

} // namespace DcgmNs
