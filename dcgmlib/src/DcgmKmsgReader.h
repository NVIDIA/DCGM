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

#pragma once

#include <DcgmMutex.h>
#include <DcgmThread.h>

#include <cstdint>
#include <unordered_set>
#include <vector>

struct KmsgXidData
{
    char pciBdf[200];     //!< GPU's PCI slot information in Bus:Domain.Function notation. For example, 0000:01:00.0
    int64_t timestamp {}; //!< Timestamp of XID
    uint32_t xid {};      //!< XID value
};

/**
 * @brief Reads the given environment variable, and adds/removes
 *        values to the provided set.
 *        For example, given __DCGM_XID_KMSG__=34,-2,1, adds 1 and 34 to the
 *        set, and removes 2 from the set if present.
 * @param envVar - Name of the environment variable
 * @param xidSet - Set of XIDs to update
 */
void ReadEnvXidAndUpdate(std::string_view envVar, std::unordered_set<uint32_t> &xidSet);

/**
 * @brief Reads the __DCGM_TEST_KMSG_FILENAME__ environment variable, and
 *        updates the provided string with the value if the value is not
 *        empty.
 * @param filename - string to update
 */
void ReadEnvKmsgFilenameAndUpdate(std::string &filename);

/**
 * @brief Parses the given buffer for XID information, and stores the GPU's PCI
 *        slot information, timestamp and XID value in the KmsgXidData struct.
 * @param buffer - Buffer to parse for XID information
 * @return unique_ptr to KmsgXidData struct with the parsed XID information
 */
std::unique_ptr<KmsgXidData> ParseKmsgLineForXid(std::string_view buffer);

/**
 * @brief Converts a timespec timestamp to timestamp in microseconds.
 * @param ts - timestamp to convert
 * @return timestamp in microseconds
 */
long long timespec_to_us(struct timespec ts);

/**
 * @brief Converts a timeval timestamp to timestamp in microseconds.
 * @param ts - timestamp to convert
 * @return timestamp in microseconds
 */
long long timeval_to_us(struct timeval ts);

/**
 * @brief Calculates the system time (in microseconds) with respect to the
 *        given monotonic timestamp value (in microseconds).
 *        Returns given timestamp on error.
 * @param timestamp - timestamp to convert
 * @return converted time since 1970 epoch (in microseconds)
 */
long long GetTimeSinceEpochFromMonoticTs(const long long timestamp);


class DcgmKmsgReaderThread : public DcgmThread
{
private:
    std::vector<std::unique_ptr<KmsgXidData>> m_parsedKmsgXids;
    std::unordered_set<uint32_t> m_xidsToParse;
    std::unique_ptr<DcgmMutex> m_mutex;
    std::string m_kmsgFilename;
    uint32_t m_pollIntervalUs = 5000;

public:
    DcgmKmsgReaderThread();
    ~DcgmKmsgReaderThread() override = default;
    void run() override;

    /**
     * @brief Returns the parsed XID vector, and clears it.
     * @return vector of unique_ptrs to the parsed KmsgXidData structs
     */
    std::vector<std::unique_ptr<KmsgXidData>> GetParsedKmsgXids();

    /**
     * @brief Returns the file descriptor polling interval in microseconds.
     * @return poll interval in us
     */
    unsigned int GetPollInterval() const;
};