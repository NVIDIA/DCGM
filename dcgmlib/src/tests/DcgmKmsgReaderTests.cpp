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
#include <DcgmMutex.h>
#include <DcgmThread.h>

#include <catch2/catch_all.hpp>
#include <fmt/format.h>

#include <chrono>
#include <ctime>
#include <list>
#include <thread>
#include <unistd.h>
#include <unordered_set>

namespace
{
DcgmMutex envMutex(0);
std::string XidEnvKey          = "__DCGM_XID_KMSG__";
std::string KmsgFilenameEnvKey = "__DCGM_TEST_KMSG_FILENAME__";

int SetEnv(const std::string &key, const std::string &value)
{
    DcgmLockGuard lock(&envMutex);
    return setenv(key.c_str(), value.c_str(), 0);
}

int UnsetEnv(const std::string &key)
{
    DcgmLockGuard lock(&envMutex);
    return unsetenv(key.c_str());
}
} //namespace

TEST_CASE("ParseKmsgLineForXid")
{
    std::string testPciBdf     = "0000:01:00";
    std::string expectedPciBdf = testPciBdf + ".0";
    int64_t insertedTimestamp  = 437812449;
    int64_t expectedTimestamp  = GetTimeSinceEpochFromMonoticTs(insertedTimestamp);
    uint32_t expectedXid       = 43;
    SECTION("validBufferReturnsParsedXidPtr")
    {
        std::string validBuffer
            = fmt::format("4,1227,{},c;NVRM: Xid (PCI:{}): {}, pid=1814139, name=cuda_assert_64b, Ch 00000018",
                          insertedTimestamp,
                          testPciBdf,
                          expectedXid);

        std::unique_ptr<KmsgXidData> parsedKmsgXid = ParseKmsgLineForXid(validBuffer.c_str());
        REQUIRE(parsedKmsgXid != nullptr);
        CHECK(std::string(parsedKmsgXid->pciBdf) == expectedPciBdf);
        // Since the timestamps are in units of microseconds, there is a slight
        // offset in the timestamps most times.
        CHECK(std::abs(parsedKmsgXid->timestamp - expectedTimestamp) <= 100);
        CHECK(parsedKmsgXid->xid == expectedXid);
    }

    SECTION("emptyBufferReturnsNullptr")
    {
        std::string emptyBuffer;
        std::unique_ptr<KmsgXidData> parsedKmsgXid = ParseKmsgLineForXid(emptyBuffer.c_str());
        REQUIRE(parsedKmsgXid == nullptr);
    }

    SECTION("randomTextBufferReturnsNullptr")
    {
        std::string randomBuffer                   = "rubbish";
        std::unique_ptr<KmsgXidData> parsedKmsgXid = ParseKmsgLineForXid(randomBuffer.c_str());
        REQUIRE(parsedKmsgXid == nullptr);
    }

    SECTION("malformedBufferReturnsNullptr")
    {
        std::string malformedBuffer
            = "4,1227,437812449932,c;NVRM: Xid PCI:0000:01:00: 43, pid=1814139, name=cuda_assert_64b, Ch 00000018";
        std::unique_ptr<KmsgXidData> parsedKmsgXid = ParseKmsgLineForXid(malformedBuffer.c_str());
        REQUIRE(parsedKmsgXid == nullptr);
    }


    SECTION("validBufferWithExtraPrefixFieldsReturnsParsedXidPtr")
    {
        std::string additionalPrefixBuffer
            = fmt::format("4,1227,{},c,1,-;NVRM: Xid (PCI:{}): {}, pid=1814139, name=cuda_assert_64b, Ch 00000018",
                          insertedTimestamp,
                          testPciBdf,
                          expectedXid);

        std::unique_ptr<KmsgXidData> parsedKmsgXid = ParseKmsgLineForXid(additionalPrefixBuffer.c_str());
        REQUIRE(parsedKmsgXid != nullptr);
        CHECK(std::string(parsedKmsgXid->pciBdf) == expectedPciBdf);
        CHECK(std::abs(parsedKmsgXid->timestamp - expectedTimestamp) <= 100);
        CHECK(parsedKmsgXid->xid == expectedXid);
    }

    SECTION("validBufferWithLastEmptyPrefixFieldsReturnsParsedXidPtr")
    {
        std::string additionalPrefixBuffer
            = fmt::format("4,1227,{},c;NVRM: Xid (PCI:{}): {}, pid=1814139, name=cuda_assert_64b, Ch 00000018",
                          insertedTimestamp,
                          testPciBdf,
                          expectedXid);

        std::unique_ptr<KmsgXidData> parsedKmsgXid = ParseKmsgLineForXid(additionalPrefixBuffer.c_str());
        REQUIRE(parsedKmsgXid != nullptr);
        CHECK(std::string(parsedKmsgXid->pciBdf) == expectedPciBdf);
        CHECK(std::abs(parsedKmsgXid->timestamp - expectedTimestamp) <= 100);
        CHECK(parsedKmsgXid->xid == expectedXid);
    }
}

TEST_CASE("ReadEnvXidAndUpdate")
{
    std::unordered_set<unsigned int> defaultXids = { 79, 119 };

    SECTION("Add new XID to set")
    {
        std::unordered_set<unsigned int> testXids = defaultXids;
        SetEnv(XidEnvKey, "3");
        ReadEnvXidAndUpdate(XidEnvKey, testXids);
        CHECK(testXids.size() == 3);
        std::unordered_set<unsigned int> expectedXids = { 3, 79, 119 };
        CHECK(testXids == expectedXids);
        UnsetEnv(XidEnvKey);
    }
    SECTION("Remove existing XID from set")
    {
        std::unordered_set<unsigned int> testXids = defaultXids;
        SetEnv(XidEnvKey, "-79");
        ReadEnvXidAndUpdate(XidEnvKey, testXids);
        CHECK(testXids.size() == 1);
        std::unordered_set<unsigned int> expectedXids = { 119 };
        CHECK(testXids == expectedXids);
        UnsetEnv(XidEnvKey);
    }

    SECTION("Remove non-existent XID from set")
    {
        std::unordered_set<unsigned int> testXids = defaultXids;
        SetEnv(XidEnvKey, "-3");
        ReadEnvXidAndUpdate(XidEnvKey, testXids);
        CHECK(testXids.size() == 2);
        CHECK(testXids == defaultXids);
        UnsetEnv(XidEnvKey);
    }

    SECTION("Remove existing XID and add new XID to set")
    {
        std::unordered_set<unsigned int> testXids = defaultXids;
        SetEnv(XidEnvKey, "-79,3");
        ReadEnvXidAndUpdate(XidEnvKey, testXids);
        CHECK(testXids.size() == 2);
        std::unordered_set<unsigned int> expectedXids = { 3, 119 };
        CHECK(testXids == expectedXids);
        UnsetEnv(XidEnvKey);
    }
    SECTION("Set invalid format env value")
    {
        std::unordered_set<unsigned int> testXids = std::move(defaultXids);
        SetEnv(XidEnvKey, "-79, 3");
        ReadEnvXidAndUpdate(XidEnvKey, testXids);
        CHECK(testXids.size() == 2);

        SetEnv(XidEnvKey, " 3");
        ReadEnvXidAndUpdate(XidEnvKey, testXids);
        CHECK(testXids.size() == 2);

        SetEnv(XidEnvKey, "rubbish");
        ReadEnvXidAndUpdate(XidEnvKey, testXids);
        CHECK(testXids.size() == 2);
        UnsetEnv(XidEnvKey);
    }
}

class PipeKmsgThread : public DcgmThread
{
private:
    std::string m_pipeName;
    std::list<std::string> m_messages;
    uint64_t m_usWriteInterval;
    void SetFlagOnFd(int fd, int flag);
    std::string FormatToKmsgString(const std::string &, uint64_t);

public:
    std::atomic_bool m_pipeReady = false;
    PipeKmsgThread()             = delete;
    PipeKmsgThread(const std::string &, const std::list<std::string> &, uint64_t);
    virtual ~PipeKmsgThread() = default;
    void run() override;
    bool GetPipeReady();
};

PipeKmsgThread::PipeKmsgThread(const std::string &pipeName,
                               const std::list<std::string> &strings,
                               uint64_t usWriteInterval)
    : m_pipeName(pipeName)
    , m_messages(strings)
    , m_usWriteInterval(usWriteInterval)
{}

void PipeKmsgThread::SetFlagOnFd(int fd, int flag)
{
    int curFlags = fcntl(fd, F_GETFL);
    if (curFlags < 0)
    {
        log_debug("fcntl F_GETFL returned error: {}", strerror(errno));
        return;
    }
    curFlags |= flag;
    int fcntlRet = fcntl(fd, F_SETFL, curFlags);
    if (fcntlRet < 0)
    {
        log_debug("fcntl F_SETFL returned error: {}", strerror(errno));
    }
}

bool PipeKmsgThread::GetPipeReady()
{
    return m_pipeReady;
}

std::string PipeKmsgThread::FormatToKmsgString(const std::string &message, uint64_t sequenceNumber)
{
    struct timespec mono_ts
    {};
    int ret = clock_gettime(CLOCK_MONOTONIC, &mono_ts);
    if (ret < 0)
    {
        log_debug("clock_gettime CLOCK_MONOTONIC returned error: %s", strerror(errno));
    }
    auto timestamp = timespec_to_us(mono_ts);
    auto priority  = 1;
    return fmt::format("{},{},{},-;{}\n", priority, sequenceNumber, timestamp, message);
}

void PipeKmsgThread::run()
{
    int mkfifoRet = mkfifo(m_pipeName.c_str(), 0755);
    if (mkfifoRet < 0)
    {
        log_debug("mkfifo returned an error: {}", strerror(errno));
        return;
    }
    m_pipeReady = true;
    int kmsgFd  = open(m_pipeName.c_str(), O_WRONLY);
    if (kmsgFd < 0)
    {
        log_debug("File {} could not be opened because of error: {}", m_pipeName.c_str(), strerror(errno));
        return;
    }
    /*
     * Mock /dev/kmsg by using this pipe in "packet" mode. Every read and write is done
     * one "packet" at a time. This is similar to /dev/kmsg where reads are done one record
     * at a time.
     */
    SetFlagOnFd(kmsgFd, O_DIRECT);

    uint64_t sequenceNumber = 1;
    while (!ShouldStop() && !m_messages.empty())
    {
        auto formattedMsg = FormatToKmsgString(m_messages.front(), sequenceNumber++);
        int writeRet      = write(kmsgFd, formattedMsg.c_str(), formattedMsg.length() + 1);
        if (writeRet < 0)
        {
            log_debug("Write returned error: ", strerror(errno));
        }
        else
        {
            m_messages.pop_front();
        }
        usleep(m_usWriteInterval);
    }

    int removeRet = remove(m_pipeName.c_str());
    if (removeRet < 0)
    {
        log_debug("remove {} returned an error: {}", m_pipeName.c_str(), strerror(errno));
    }
    m_pipeReady = false;
}

TEST_CASE("Test kmsg pipe")
{
    std::string filename                      = "/tmp/kmsg";
    std::list<std::string> xidMessagesToWrite = {
        "filler sfdsfdslkfjdslgjflkdsjf",
        "filler sfdsfdslkfjdslgjflkdsjf",
        "filler sfdsfdslkfjdslgjflkdsjf",
        "filler sfdsfdslkfjdslgjflkdsjf",
        "filler sfdsfdslkfjdslgjflkdsjf",
        "filler sfdsfdslkfjdslgjflkdsjf",
        "NVRM: filler",
        "NVRM: GPU at PCI:0000:01:00: GPU-aeab3757-56a6-7e30-3f7c-1f322454bde2",
        "NVRM: GPU Board Serial Number: 0322518027985",
        "filler sfdsfdslkfjdslgjflkdsjf",
        "filler sfdsfdslkfjdslgjflkdsjf",
        "NVRM: Xid (PCI:0000:00:16): 119, pid=36055, name=dcgm-exporter, Timeout after 6s of waiting for RPC response from GPU0 GSP! Expected function 76 (GSP_RM_CONTROL) (0x90cc0301 0xc).",
        "NVRM: GPU at PCI:0000:01:00: GPU-aeab3757-56a6-7e30-3f7c-1f322454bde2",
        "NVRM: filler",
        "NVRM: filler",
        "filler sfdsfdslkfjdslgjflkdsjf",
        "filler sfdsfdslkfjdslgjflkdsjf",
        "NVRM: GPU Board Serial Number: 0322518027985",
        "filler sfdsfdslkfjdslgjflkdsjf",
        "filler sfdsfdslkfjdslgjflkdsjf",
        "NVRM: Xid (PCI:0000:01:00): 43, pid=1814139, name=cuda_assert_64b, Ch 00000018",
        "filler sfdsfdslkfjdslgjflkdsjf",
        "filler sfdsfdslkfjdslgjflkdsjf",
        "filler sfdsfdslkfjdslgjflkdsjf",
        "filler sfdsfdslkfjdslgjflkdsjf",
        "NVRM: filler",
        "NVRM: filler",
        "NVRM: Xid (PCI:0000:2a:00): 79, pid='<unknown>', name=<unknown>, GPU has fallen off the bus",
        "filler sfdsfdslkfjdslgjflkdsjf",
        "filler sfdsfdslkfjdslgjflkdsjf",
        "filler sfdsfdslkfjdslgjflkdsjf",
        "filler sfdsfdslkfjdslgjflkdsjf",
    };

    SetEnv(KmsgFilenameEnvKey, filename);
    SECTION("Verifies reader reads and parses all available XIDs with a much slower writer")
    {
        DcgmKmsgReaderThread readerThread {};
        uint64_t usWriteInterval = readerThread.GetPollInterval() * 5;
        PipeKmsgThread writerThread { filename, xidMessagesToWrite, usWriteInterval };
        writerThread.Start();

        DcgmMutex wThreadMutex(0);
        std::condition_variable cv;
        std::function<bool()> cvFunc = std::bind(&PipeKmsgThread::GetPipeReady, &writerThread);
        int waitForPipeReady         = wThreadMutex.CondWait(cv, 100, cvFunc);
        REQUIRE(waitForPipeReady == DCGM_MUTEX_ST_OK);

        readerThread.Start();

        unsigned int usWaitForThreads = xidMessagesToWrite.size() * usWriteInterval + usWriteInterval;
        std::this_thread::sleep_for(std::chrono::microseconds(usWaitForThreads));
        REQUIRE(writerThread.StopAndWait(usWriteInterval * 1000) == 0);
        REQUIRE(readerThread.StopAndWait(readerThread.GetPollInterval() * 1000) == 0);

        std::vector<std::unique_ptr<KmsgXidData>> parsedXids = readerThread.GetParsedKmsgXids();
        REQUIRE(parsedXids.size() == 2);
        CHECK(parsedXids[0]->xid == 119);
        CHECK(std::string(parsedXids[0]->pciBdf) == "0000:00:16.0");

        CHECK(parsedXids[1]->xid == 79);
        CHECK(std::string(parsedXids[1]->pciBdf) == "0000:2a:00.0");

        parsedXids.clear();

        parsedXids = readerThread.GetParsedKmsgXids();
        REQUIRE(parsedXids.size() == 0);
    }


    SECTION("Verifies reader does not block on stop with a much slower writer")
    {
        DcgmKmsgReaderThread readerThread {};
        uint64_t usWriteInterval = readerThread.GetPollInterval() * 5;
        PipeKmsgThread writerThread { filename, xidMessagesToWrite, usWriteInterval };
        writerThread.Start();

        DcgmMutex wThreadMutex(0);
        std::condition_variable cv;
        std::function<bool()> cvFunc = std::bind(&PipeKmsgThread::GetPipeReady, &writerThread);
        int waitForPipeReady         = wThreadMutex.CondWait(cv, 100, cvFunc);
        REQUIRE(waitForPipeReady == DCGM_MUTEX_ST_OK);

        readerThread.Start();

        unsigned int usWaitForThreads = xidMessagesToWrite.size() / 4 * usWriteInterval + usWriteInterval;
        std::this_thread::sleep_for(std::chrono::microseconds(usWaitForThreads));
        REQUIRE(writerThread.StopAndWait(usWriteInterval * 1000) == 0);
        REQUIRE(readerThread.StopAndWait(readerThread.GetPollInterval() * 1000) == 0);
    }

    SECTION("Verifies reader does not block on stop with a faster writer")
    {
        // Stuff more filler messages in the list to fill out the pipe
        for (auto i = 0; i < 200; i++)
        {
            xidMessagesToWrite.emplace_front("filler sfdsfdslkfjdslgjflkdsjf");
        }
        DcgmKmsgReaderThread readerThread {};
        uint64_t usWriteInterval = readerThread.GetPollInterval() / 1000;
        PipeKmsgThread writerThread { filename, xidMessagesToWrite, usWriteInterval };
        writerThread.Start();
        DcgmMutex wThreadMutex(0);
        std::condition_variable cv;
        std::function<bool()> cvFunc = std::bind(&PipeKmsgThread::GetPipeReady, &writerThread);
        int waitForPipeReady         = wThreadMutex.CondWait(cv, 100, cvFunc);
        REQUIRE(waitForPipeReady == DCGM_MUTEX_ST_OK);

        readerThread.Start();

        unsigned int usWaitForThreads = xidMessagesToWrite.size() * usWriteInterval + usWriteInterval;
        std::this_thread::sleep_for(std::chrono::microseconds(usWaitForThreads));
        REQUIRE(writerThread.StopAndWait(usWriteInterval * 1000) == 0);
        REQUIRE(readerThread.StopAndWait(readerThread.GetPollInterval() * 1000) == 0);

        std::vector<std::unique_ptr<KmsgXidData>> parsedXids = readerThread.GetParsedKmsgXids();
        REQUIRE(parsedXids.size() == 0);
    }
    UnsetEnv(KmsgFilenameEnvKey);
}

TEST_CASE("Nonexistent test kmsg filename")
{
    SetEnv(KmsgFilenameEnvKey, "");
    SECTION("Reader thread exits early")
    {
        DcgmKmsgReaderThread readerThread {};
        readerThread.Start();
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        CHECK(readerThread.HasExited() == 1);
    }
}