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

#include "DcgmKmsgReader.h"

#include <DcgmLogging.h>
#include <DcgmStringHelpers.h>
#include <DcgmUtilities.h>

#include <boost/regex.hpp>

#include <cassert>
#include <errno.h>
#include <fcntl.h>
#include <filesystem>
#include <poll.h>
#include <regex>
#include <set>
#include <string.h>
#include <unistd.h>

void ReadEnvXidAndUpdate(std::string_view envVar, std::unordered_set<uint32_t> &xidsToParse)
{
    char *xidGetEnv = getenv(envVar.data());
    if (!xidGetEnv)
    {
        log_debug("Env variable {} unset. Not loading", envVar.data());
        return;
    }
    std::string xidEnvOverride(xidGetEnv);
    auto xidStrings = dcgmTokenizeString(xidEnvOverride, ",");
    for (auto const &xidString : xidStrings)
    {
        uint32_t newXid;
        try
        {
            if (!xidString.starts_with("-"))
            {
                newXid = std::stoul(xidString);
                xidsToParse.insert(newXid);
                log_debug("Added {} to kmsg XID set", newXid);
            }
            else
            {
                newXid = std::stoul(xidString.substr(1));
                xidsToParse.erase(newXid);
                log_debug("Removed {} from kmsg XID set", newXid);
            }
        }
        catch (std::invalid_argument const &)
        {
            log_warning("Env variable {} not formatted correctly. Specify comma-separated numbers.", envVar.data());
            return;
        }
    }
}

long long timespec_to_us(struct timespec ts)
{
    return static_cast<long long>(ts.tv_sec) * 1000 * 1000 + ts.tv_nsec / 1000;
}

long long timeval_to_us(struct timeval ts)
{
    return static_cast<long long>(ts.tv_sec) * 1000 * 1000 + ts.tv_usec;
}

long long GetTimeSinceEpochFromMonoticTs(const long long timestamp)
{
    struct timeval now
    {};
    int ret = gettimeofday(&now, NULL);
    if (ret < 0)
    {
        log_debug("gettimeofday returned error: %s", strerror(errno));
        return timestamp;
    }
    struct timespec boot_ts
    {};

#ifdef CLOCK_BOOTTIME
    // Use CLOCK_BOOTTIME if defined because that accurately takes into
    // account system standby time.
    ret = clock_gettime(CLOCK_BOOTTIME, &boot_ts);
    if (ret < 0)
    {
        log_debug("clock_gettime CLOCK_BOOTTIME returned error: %s", strerror(errno));
        return timestamp;
    }
#else
    ret = clock_gettime(CLOCK_MONOTONIC, &boot_ts);
    if (ret < 0)
    {
        log_debug("clock_gettime CLOCK_MONOTONIC returned error: %s", strerror(errno));
        return timestamp;
    }
#endif

    long long boottime = timeval_to_us(now) - timespec_to_us(boot_ts);
    return boottime + timestamp;
}

std::unique_ptr<KmsgXidData> ParseKmsgLineForXid(std::string_view buffer)
{
    static boost::regex exp(R"(^\d+,\d+,(\d+),.*;NVRM: Xid \(PCI:(.*)\): (\d+),.*)");
    boost::cmatch match;
    if (boost::regex_match(buffer.begin(), buffer.end(), match, exp))
    {
        if (match.size() != 4)
        {
            log_debug("Kmsg regex match incorrect");
            return nullptr;
        }
        std::unique_ptr<KmsgXidData> newXid = std::make_unique<KmsgXidData>();
        try
        {
            long long monotonicTimestamp = std::stoll(match[1].str());
            newXid->timestamp            = GetTimeSinceEpochFromMonoticTs(monotonicTimestamp);
            SafeCopyTo(newXid->pciBdf, std::string(match[2].str() + ".0").c_str());
            newXid->xid = std::stoul(match[3].str());
            return newXid;
        }
        catch (std::out_of_range const &e)
        {
            log_error("Kmsg line could not be parsed correctly, exception: {}", e.what());
        }
    }
    return nullptr;
}

void ReadEnvKmsgFilenameAndUpdate(std::string &filename)
{
    char *filenameGetEnv = getenv("__DCGM_TEST_KMSG_FILENAME__");
    if (!filenameGetEnv)
    {
        log_debug("__DCGM_TEST_KMSG_FILENAME__ unset. Not loading");
        return;
    }
    std::string kmsgFilenameOverride = filenameGetEnv;
    if (!kmsgFilenameOverride.empty())
    {
        log_debug("Parsing {} instead of {}", kmsgFilenameOverride, filename);
        filename = kmsgFilenameOverride;
        if (!std::filesystem::exists(kmsgFilenameOverride))
        {
            log_error("__DCGM_TEST_KMSG_FILENAME__ value incorrect. {} does not exist", kmsgFilenameOverride);
        }
    }
}

DcgmKmsgReaderThread::DcgmKmsgReaderThread()
    : m_xidsToParse({ 79, 119, 120 })
    , m_mutex(std::make_unique<DcgmMutex>(0))
    , m_kmsgFilename("/dev/kmsg")
{
    try
    {
        ReadEnvXidAndUpdate("__DCGM_XID_KMSG__", m_xidsToParse);
        ReadEnvKmsgFilenameAndUpdate(m_kmsgFilename);
    }
    catch (const std::exception &e)
    {
        log_warning("Exception in DcgmKmsgReader::DcgmKmsgReader: {}", e.what());
    }
}

std::vector<std::unique_ptr<KmsgXidData>> DcgmKmsgReaderThread::GetParsedKmsgXids()
{
    DcgmLockGuard lg(m_mutex.get());
    return std::exchange(m_parsedKmsgXids, {});
}

uint32_t DcgmKmsgReaderThread::GetPollInterval() const
{
    return m_pollIntervalUs;
}

void DcgmKmsgReaderThread::run()
{
    struct pollfd pfds
    {};
    constexpr uint32_t MAX_RECORD_SIZE = 2048; // Based on PRINTK_MESSAGE_MAX
    auto pollBlockForMs                = 0;    // return immediately if not ready
    bool errorCondition                = false;
    bool sleepNow                      = false;

    int kmsgFd = open(m_kmsgFilename.c_str(), O_RDONLY | O_NONBLOCK);
    if (kmsgFd < 0)
    {
        log_debug("File {} could not be opened because of error: {}", m_kmsgFilename, strerror(errno));
        errorCondition = true;
    }

    DcgmNs::Utils::FileHandle kmsgFileHandle { kmsgFd };
    pfds.fd     = kmsgFileHandle.Get();
    pfds.events = POLLIN;

    while (!ShouldStop() && !errorCondition)
    {
        if (sleepNow)
        {
            usleep(m_pollIntervalUs);
            sleepNow = false;
        }
        int pollRet = poll(&pfds, 1, pollBlockForMs);
        if (pollRet < 0)
        {
            log_debug("Poll returned error: {}", strerror(errno));
            errorCondition = true;
        }
        else if (pollRet == 0) // nothing to read at this time, sleep and try again
        {
            sleepNow = true;
            continue;
        }
        else // data available to read
        {
            if (pfds.revents & POLLIN)
            {
                char readBuffer[MAX_RECORD_SIZE];
                bool readAgain;
                do
                {
                    readAgain = false;
                    // /dev/kmsg reads are one record at a time
                    ssize_t readRet = read(kmsgFileHandle.Get(), readBuffer, sizeof(readBuffer));
                    if (readRet < 0)
                    {
                        if (errno == EAGAIN || errno == EWOULDBLOCK)
                        {
                            log_debug("EWOULDBLOCK recorded");
                            sleepNow = true;
                        }
                        else if (errno == EPIPE)
                        {
                            log_debug("Read kmsg returned EPIPE, data overwritten in circular buffer, reading again");
                            readAgain = true;
                            continue;
                        }
                        else
                        {
                            log_debug("Read kmsg errored: {} ", strerror(errno));
                            errorCondition = true;
                            break;
                        }
                    }
                    else
                    {
                        std::string_view readBuffer_sv(readBuffer, readRet);
                        std::unique_ptr<KmsgXidData> newXid = ParseKmsgLineForXid(readBuffer_sv);
                        if (newXid && m_xidsToParse.contains(newXid->xid))
                        {
                            DcgmLockGuard lg(m_mutex.get());
                            m_parsedKmsgXids.emplace_back(std::move(newXid));
                            log_debug("Adding parsed XID {} to kmsg XIDs", m_parsedKmsgXids.back()->xid);
                        }
                    }
                } while (readAgain && !ShouldStop());
            }
            else // POLLHUP or POLLERR
            {
                log_debug("Poll kmsg revents error. Revents {}", pfds.revents);
                errorCondition = true;
            }
        }
    }
}