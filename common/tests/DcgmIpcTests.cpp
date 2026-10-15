/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
#include "DcgmIpc.h"
#include "Defer.hpp"
#include "HangDetect.h"
#include "HangDetectMonitor.h"
#include <TempDir.hpp>

#include <catch2/catch_all.hpp>
#include <dcgm_structs.h>
#include <latch>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <utility>

class DcgmIpcTestFixture
{
protected:
    TempDir socketDir      = TempDir::Random("dcgm-ipc-test-");
    std::string socketPath = (socketDir.Path() / "dcgm-ipc-test.sock").string();
    DcgmIpcDomainServerParams_t domainParams;

    // No-op callbacks for tests that don't need message handling
    DcgmIpcProcessMessageFunc_f noopMessage = [](dcgm_connection_id_t, std::unique_ptr<DcgmMessage>, void *) {
    };
    DcgmIpcProcessDisconnectFunc_f noopDisconnect = [](dcgm_connection_id_t, void *) {
    };

    class TcpPortReservation
    {
    public:
        TcpPortReservation(int fd, int port)
            : m_fd(fd)
            , m_port(port)
        {}

        TcpPortReservation(TcpPortReservation const &)            = delete;
        TcpPortReservation &operator=(TcpPortReservation const &) = delete;

        TcpPortReservation(TcpPortReservation &&other) noexcept
            : m_fd(std::exchange(other.m_fd, -1))
            , m_port(std::exchange(other.m_port, 0))
        {}

        TcpPortReservation &operator=(TcpPortReservation &&other) noexcept
        {
            if (this != &other)
            {
                Close();
                m_fd   = std::exchange(other.m_fd, -1);
                m_port = std::exchange(other.m_port, 0);
            }

            return *this;
        }

        ~TcpPortReservation()
        {
            Close();
        }

        [[nodiscard]] int Port() const
        {
            return m_port;
        }

        void Release()
        {
            Close();
        }

        [[nodiscard]] int Fd() const
        {
            return m_fd;
        }

        [[nodiscard]] int ReleaseFd()
        {
            return std::exchange(m_fd, -1);
        }

    private:
        void Close()
        {
            if (m_fd >= 0)
            {
                close(m_fd);
                m_fd = -1;
            }
        }

        int m_fd   = -1;
        int m_port = 0;
    };

    static TcpPortReservation ReserveUnusedTcpPort()
    {
        int fd = socket(AF_INET, SOCK_STREAM, 0);
        REQUIRE(fd >= 0);
        TcpPortReservation reservation(fd, 0);

        sockaddr_in addr {};
        addr.sin_family      = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        addr.sin_port        = 0;

        REQUIRE(bind(reservation.Fd(), reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) == 0);

        socklen_t addrLen = sizeof(addr);
        REQUIRE(getsockname(reservation.Fd(), reinterpret_cast<sockaddr *>(&addr), &addrLen) == 0);

        REQUIRE(listen(reservation.Fd(), 1) == 0);

        return TcpPortReservation(reservation.ReleaseFd(), ntohs(addr.sin_port));
    }

    static DcgmIpcTcpServerParams_t MakeTcpParams(int port, std::string bindAddress = "127.0.0.1")
    {
        DcgmIpcTcpServerParams_t tcpParams;
        tcpParams.bindIPAddress = std::move(bindAddress);
        tcpParams.port          = port;
        return tcpParams;
    }

    static dcgmReturn_t InitTcpServerOnReservedPort(std::unique_ptr<DcgmIpc> &serverOut,
                                                    DcgmIpcTcpServerParams_t &tcpParamsOut,
                                                    DcgmIpcProcessMessageFunc_f processMessage,
                                                    void *processMessageUserData,
                                                    DcgmIpcProcessDisconnectFunc_f processDisconnect,
                                                    void *processDisconnectUserData,
                                                    std::string bindIPAddress = "127.0.0.1")
    {
        dcgmReturn_t result = DCGM_ST_GENERIC_ERROR;

        // Attempt binding the port up to 10 times to virtually eliminate TOCTOU risk in tests
        for (int attempt = 0; attempt < 10; attempt++)
        {
            auto reservedPort = ReserveUnusedTcpPort();
            auto tcpParams    = MakeTcpParams(reservedPort.Port(), bindIPAddress);
            reservedPort.Release();

            // Allows us to build a fresh DcgmIpc object in each attempt
            serverOut = std::make_unique<DcgmIpc>(1);
            result    = serverOut->Init(
                tcpParams, processMessage, processMessageUserData, processDisconnect, processDisconnectUserData);
            if (result == DCGM_ST_OK)
            {
                tcpParamsOut = tcpParams;
                return DCGM_ST_OK;
            }
            serverOut.reset();
        }

        return result;
    }

    /**
     * Common setup for DcgmIpc tests
     */
    DcgmIpcTestFixture()
    {
        unlink(socketPath.c_str());
        domainParams.domainSocketPath = socketPath;
    }

    ~DcgmIpcTestFixture()
    {
        unlink(socketPath.c_str());
    }
};

class DcgmIpcHangDetectFixture : public DcgmIpcTestFixture
{
protected:
    // HangDetect must outlive the monitor that references it
    HangDetect hangDetect;
    HangDetectMonitor hangDetectMonitor { hangDetect };
};

TEST_CASE("DcgmIpc::GetConnectAddressFamilyHint")
{
    CHECK(DcgmNs::Ipc::detail::GetConnectAddressFamilyHint("127.0.0.1") == AF_INET);
    CHECK(DcgmNs::Ipc::detail::GetConnectAddressFamilyHint("::1") == AF_INET6);
    CHECK(DcgmNs::Ipc::detail::GetConnectAddressFamilyHint("ip6-localhost") == AF_UNSPEC);
    CHECK(DcgmNs::Ipc::detail::GetConnectAddressFamilyHint("localhost") == AF_UNSPEC);
}

TEST_CASE("DcgmIpc::ResolveTcpConnectCandidates")
{
    using DcgmNs::Ipc::detail::ConnectCandidate;
    using DcgmNs::Ipc::detail::ResolveTcpConnectCandidates;

    SECTION("IPv4 literal resolves to a single AF_INET candidate")
    {
        std::vector<ConnectCandidate> candidates;
        REQUIRE(ResolveTcpConnectCandidates("127.0.0.1", 5555, candidates) == DCGM_ST_OK);
        REQUIRE(candidates.size() == 1);
        CHECK(candidates[0].addr.ss_family == AF_INET);
    }

    SECTION("IPv6 literal resolves to a single AF_INET6 candidate")
    {
        std::vector<ConnectCandidate> candidates;
        REQUIRE(ResolveTcpConnectCandidates("::1", 5555, candidates) == DCGM_ST_OK);
        REQUIRE(candidates.size() == 1);
        CHECK(candidates[0].addr.ss_family == AF_INET6);
    }

    SECTION("localhost resolves to at least one usable candidate")
    {
        std::vector<ConnectCandidate> candidates;
        REQUIRE(ResolveTcpConnectCandidates("localhost", 5555, candidates) == DCGM_ST_OK);
        REQUIRE(!candidates.empty());
        for (auto const &candidate : candidates)
        {
            CHECK((candidate.addr.ss_family == AF_INET || candidate.addr.ss_family == AF_INET6));
            CHECK(candidate.addrLen > 0);
        }
    }

    SECTION("Unresolvable name fails cleanly")
    {
        std::vector<ConnectCandidate> candidates;
        CHECK(ResolveTcpConnectCandidates("dcgm.invalid.nonexistent.example.", 5555, candidates) != DCGM_ST_OK);
        CHECK(candidates.empty());
    }
}

TEST_CASE("DcgmIpc::SelectFamilyFallbackCandidates")
{
    using DcgmNs::Ipc::detail::ConnectCandidate;
    using DcgmNs::Ipc::detail::SelectFamilyFallbackCandidates;

    /* marker is stored in sin_port / sin6_port so each candidate is distinguishable even
       when two candidates share the same address family. */
    auto makeCandidate = [](sa_family_t family, uint16_t marker) {
        ConnectCandidate candidate {};
        if (family == AF_INET)
        {
            auto &sin         = reinterpret_cast<sockaddr_in &>(candidate.addr);
            sin.sin_family    = AF_INET;
            sin.sin_port      = htons(marker);
            candidate.addrLen = sizeof(sockaddr_in);
        }
        else
        {
            auto &sin6        = reinterpret_cast<sockaddr_in6 &>(candidate.addr);
            sin6.sin6_family  = AF_INET6;
            sin6.sin6_port    = htons(marker);
            candidate.addrLen = sizeof(sockaddr_in6);
        }
        return candidate;
    };

    auto markerOf = [](ConnectCandidate const &c) -> uint16_t {
        if (c.addr.ss_family == AF_INET)
        {
            return ntohs(reinterpret_cast<sockaddr_in const &>(c.addr).sin_port);
        }
        return ntohs(reinterpret_cast<sockaddr_in6 const &>(c.addr).sin6_port);
    };

    SECTION("Dual-stack reduces to one IPv4 then one IPv6")
    {
        /* resolver order: IPv6:10, IPv4:20, IPv6:30, IPv4:40
           expected:       IPv4:20 (first IPv4 in resolver order), IPv6:10 (first IPv6) */
        std::vector<ConnectCandidate> resolved {
            makeCandidate(AF_INET6, 10),
            makeCandidate(AF_INET, 20),
            makeCandidate(AF_INET6, 30),
            makeCandidate(AF_INET, 40),
        };
        auto ordered = SelectFamilyFallbackCandidates(resolved);
        REQUIRE(ordered.size() == 2);
        CHECK(ordered[0].addr.ss_family == AF_INET); // IPv4 first, regardless of resolver order
        CHECK(ordered[0].addrLen == sizeof(sockaddr_in));
        CHECK(markerOf(ordered[0]) == 20); // first IPv4 in resolver order, not a later one
        CHECK(ordered[1].addr.ss_family == AF_INET6);
        CHECK(ordered[1].addrLen == sizeof(sockaddr_in6));
        CHECK(markerOf(ordered[1]) == 10); // first IPv6 in resolver order, not a later one
    }

    SECTION("IPv4-only stays IPv4-only")
    {
        std::vector<ConnectCandidate> resolved { makeCandidate(AF_INET, 11), makeCandidate(AF_INET, 22) };
        auto ordered = SelectFamilyFallbackCandidates(resolved);
        REQUIRE(ordered.size() == 1);
        CHECK(ordered[0].addr.ss_family == AF_INET);
        CHECK(ordered[0].addrLen == sizeof(sockaddr_in));
        CHECK(markerOf(ordered[0]) == 11); // first in resolver order
    }

    SECTION("IPv6-only stays IPv6-only (no wasted IPv4 attempt)")
    {
        std::vector<ConnectCandidate> resolved { makeCandidate(AF_INET6, 33) };
        auto ordered = SelectFamilyFallbackCandidates(resolved);
        REQUIRE(ordered.size() == 1);
        CHECK(ordered[0].addr.ss_family == AF_INET6);
        CHECK(ordered[0].addrLen == sizeof(sockaddr_in6));
        CHECK(markerOf(ordered[0]) == 33);
    }

    SECTION("Empty stays empty")
    {
        CHECK(SelectFamilyFallbackCandidates({}).empty());
    }
}

TEST_CASE_METHOD(DcgmIpcTestFixture, "DcgmIpc::Lifecycle")
{
    SECTION("Construct and destroy without init")
    {
        // Verifies constructor and destructor doesn't crash or hang
        DcgmIpc ipc(1);
    }

    SECTION("Init with domain socket and clean shutdown")
    {
        DcgmIpc ipc(1);
        auto result = ipc.Init(domainParams, noopMessage, nullptr, noopDisconnect, nullptr);
        REQUIRE(result == DCGM_ST_OK);
    }

    SECTION("Init with no server params")
    {
        DcgmIpc ipc(1);
        auto result = ipc.Init(std::nullopt, noopMessage, nullptr, noopDisconnect, nullptr);
        REQUIRE(result == DCGM_ST_OK);
    }

    SECTION("Init with multiple worker threads")
    {
        DcgmIpc ipc(4);
        auto result = ipc.Init(domainParams, noopMessage, nullptr, noopDisconnect, nullptr);
        REQUIRE(result == DCGM_ST_OK);
    }
}

TEST_CASE_METHOD(DcgmIpcTestFixture, "DcgmIpc::Connections")
{
    // Server instance, shared by all sections
    DcgmIpc server(1);
    auto initResult = server.Init(domainParams, noopMessage, nullptr, noopDisconnect, nullptr);
    REQUIRE(initResult == DCGM_ST_OK);

    SECTION("Client connects to server via domain socket")
    {
        DcgmIpc client(1);
        auto clientInit = client.Init(std::nullopt, noopMessage, nullptr, noopDisconnect, nullptr);
        REQUIRE(clientInit == DCGM_ST_OK);

        dcgm_connection_id_t connectionId = DCGM_CONNECTION_ID_NONE;
        auto result                       = client.ConnectDomain(socketPath, connectionId, 5000);

        REQUIRE(result == DCGM_ST_OK);
        REQUIRE(connectionId != DCGM_CONNECTION_ID_NONE);
    }

    SECTION("MonitorSocketFd with socketpair")
    {
        int fds[2];
        int rc = socketpair(AF_UNIX, SOCK_STREAM, 0, fds);
        REQUIRE(rc == 0);

        // Server monitors one end, we hold the other
        dcgm_connection_id_t connectionId = DCGM_CONNECTION_ID_NONE;
        auto result                       = server.MonitorSocketFd(fds[0], connectionId);

        REQUIRE(result == DCGM_ST_OK);
        REQUIRE(connectionId != DCGM_CONNECTION_ID_NONE);

        // Close our end to trigger cleanup on the server side
        close(fds[1]);
    }

    SECTION("Multiple connections get unique connection IDs")
    {
        DcgmIpc client(1);
        auto clientInit = client.Init(std::nullopt, noopMessage, nullptr, noopDisconnect, nullptr);
        REQUIRE(clientInit == DCGM_ST_OK);

        dcgm_connection_id_t connId1 = DCGM_CONNECTION_ID_NONE;
        dcgm_connection_id_t connId2 = DCGM_CONNECTION_ID_NONE;

        auto connectResult1 = client.ConnectDomain(socketPath, connId1, 5000);
        REQUIRE(connectResult1 == DCGM_ST_OK);

        auto connectResult2 = client.ConnectDomain(socketPath, connId2, 5000);
        REQUIRE(connectResult2 == DCGM_ST_OK);

        REQUIRE(connId1 != DCGM_CONNECTION_ID_NONE);
        REQUIRE(connId2 != DCGM_CONNECTION_ID_NONE);
        REQUIRE(connId1 != connId2);
    }

    SECTION("CloseConnection after connect")
    {
        DcgmIpc client(1);
        auto clientInit = client.Init(std::nullopt, noopMessage, nullptr, noopDisconnect, nullptr);
        REQUIRE(clientInit == DCGM_ST_OK);

        dcgm_connection_id_t connectionId = DCGM_CONNECTION_ID_NONE;
        auto connectResult                = client.ConnectDomain(socketPath, connectionId, 5000);
        REQUIRE(connectResult == DCGM_ST_OK);
        REQUIRE(connectionId != DCGM_CONNECTION_ID_NONE);

        auto closeResult = client.CloseConnection(connectionId);
        REQUIRE(closeResult == DCGM_ST_OK);
    }

    SECTION("Client connects to server via TCP")
    {
        std::unique_ptr<DcgmIpc> tcpServer;
        DcgmIpcTcpServerParams_t tcpParams;
        auto serverInit
            = InitTcpServerOnReservedPort(tcpServer, tcpParams, noopMessage, nullptr, noopDisconnect, nullptr);
        REQUIRE(serverInit == DCGM_ST_OK);

        DcgmIpc client(1);
        auto clientInit = client.Init(std::nullopt, noopMessage, nullptr, noopDisconnect, nullptr);
        REQUIRE(clientInit == DCGM_ST_OK);

        dcgm_connection_id_t connectionId = DCGM_CONNECTION_ID_NONE;
        auto result = client.ConnectTcp(tcpParams.bindIPAddress, tcpParams.port, connectionId, 5000);

        REQUIRE(result == DCGM_ST_OK);
        REQUIRE(connectionId != DCGM_CONNECTION_ID_NONE);
    }
}

TEST_CASE_METHOD(DcgmIpcTestFixture, "DcgmIpc::Messaging")
{
    // Captures the first message the server receives
    std::promise<std::vector<char>> receivedPromise;
    auto receivedFuture = receivedPromise.get_future();

    auto capturingHandler = [&](dcgm_connection_id_t, std::unique_ptr<DcgmMessage> msg, void *) {
        auto bytes = msg->GetMsgBytesPtr();
        receivedPromise.set_value(std::vector<char>(bytes->begin(), bytes->end()));
    };

    DcgmIpc server(1);
    auto initResult = server.Init(domainParams, capturingHandler, nullptr, noopDisconnect, nullptr);
    REQUIRE(initResult == DCGM_ST_OK);

    SECTION("Client sends message to server")
    {
        DcgmIpc client(1);
        auto clientInit = client.Init(std::nullopt, noopMessage, nullptr, noopDisconnect, nullptr);
        REQUIRE(clientInit == DCGM_ST_OK);

        dcgm_connection_id_t connectionId = DCGM_CONNECTION_ID_NONE;
        auto connectResult                = client.ConnectDomain(socketPath, connectionId, 5000);
        REQUIRE(connectResult == DCGM_ST_OK);

        std::string payload = "hello from client";
        auto msg            = std::make_unique<DcgmMessage>();
        msg->GetMsgBytesPtr()->assign(payload.begin(), payload.end());
        msg->UpdateMsgHdr(DCGM_MSG_MODULE_COMMAND, DCGM_REQUEST_ID_NONE, DCGM_ST_OK, payload.size());

        auto sendResult = client.SendMessage(connectionId, std::move(msg), true);
        REQUIRE(sendResult == DCGM_ST_OK);

        auto status = receivedFuture.wait_for(std::chrono::seconds(5));
        REQUIRE(status == std::future_status::ready);

        auto received = receivedFuture.get();
        std::string receivedStr(received.begin(), received.end());
        REQUIRE(receivedStr == "hello from client");
    }

    SECTION("Send with waitForSend false")
    {
        DcgmIpc client(1);
        auto clientInit = client.Init(std::nullopt, noopMessage, nullptr, noopDisconnect, nullptr);
        REQUIRE(clientInit == DCGM_ST_OK);

        dcgm_connection_id_t connectionId = DCGM_CONNECTION_ID_NONE;
        auto connectResult                = client.ConnectDomain(socketPath, connectionId, 5000);
        REQUIRE(connectResult == DCGM_ST_OK);

        std::string payload = "fire and forget";
        auto msg            = std::make_unique<DcgmMessage>();
        msg->GetMsgBytesPtr()->assign(payload.begin(), payload.end());
        msg->UpdateMsgHdr(DCGM_MSG_MODULE_COMMAND, DCGM_REQUEST_ID_NONE, DCGM_ST_OK, payload.size());

        auto sendResult = client.SendMessage(connectionId, std::move(msg), false);
        REQUIRE(sendResult == DCGM_ST_OK);

        auto status = receivedFuture.wait_for(std::chrono::seconds(5));
        REQUIRE(status == std::future_status::ready);
    }

    SECTION("TCP client sends message to server")
    {
        std::unique_ptr<DcgmIpc> tcpServer;
        DcgmIpcTcpServerParams_t tcpParams;
        auto serverInit
            = InitTcpServerOnReservedPort(tcpServer, tcpParams, capturingHandler, nullptr, noopDisconnect, nullptr);
        REQUIRE(serverInit == DCGM_ST_OK);

        DcgmIpc client(1);
        auto clientInit = client.Init(std::nullopt, noopMessage, nullptr, noopDisconnect, nullptr);
        REQUIRE(clientInit == DCGM_ST_OK);

        dcgm_connection_id_t connectionId = DCGM_CONNECTION_ID_NONE;
        auto connectResult = client.ConnectTcp(tcpParams.bindIPAddress, tcpParams.port, connectionId, 5000);
        REQUIRE(connectResult == DCGM_ST_OK);

        std::string payload = "hello over tcp";
        auto msg            = std::make_unique<DcgmMessage>();
        msg->GetMsgBytesPtr()->assign(payload.begin(), payload.end());
        msg->UpdateMsgHdr(DCGM_MSG_MODULE_COMMAND, DCGM_REQUEST_ID_NONE, DCGM_ST_OK, payload.size());

        auto sendResult = client.SendMessage(connectionId, std::move(msg), true);
        REQUIRE(sendResult == DCGM_ST_OK);

        auto status = receivedFuture.wait_for(std::chrono::seconds(5));
        REQUIRE(status == std::future_status::ready);

        auto received = receivedFuture.get();
        REQUIRE(std::string(received.begin(), received.end()) == payload);
    }
}

TEST_CASE_METHOD(DcgmIpcTestFixture, "DcgmIpc::MultipleMessages")
{
    int constexpr numMessages = 15;
    std::latch allReceived(numMessages);
    std::vector<std::string> receivedPayloads;

    auto handler = [&](dcgm_connection_id_t, std::unique_ptr<DcgmMessage> msg, void *) {
        auto bytes = msg->GetMsgBytesPtr();
        receivedPayloads.emplace_back(bytes->begin(), bytes->end());
        allReceived.count_down();
    };

    DcgmIpc server(1);
    auto initResult = server.Init(domainParams, handler, nullptr, noopDisconnect, nullptr);
    REQUIRE(initResult == DCGM_ST_OK);

    DcgmIpc client(1);
    auto clientInit = client.Init(std::nullopt, noopMessage, nullptr, noopDisconnect, nullptr);
    REQUIRE(clientInit == DCGM_ST_OK);

    dcgm_connection_id_t connectionId = DCGM_CONNECTION_ID_NONE;
    auto connectResult                = client.ConnectDomain(socketPath, connectionId, 5000);
    REQUIRE(connectResult == DCGM_ST_OK);

    for (int i = 0; i < numMessages; i++)
    {
        std::string payload = "msg_" + std::to_string(i);
        auto msg            = std::make_unique<DcgmMessage>();
        msg->GetMsgBytesPtr()->assign(payload.begin(), payload.end());
        msg->UpdateMsgHdr(DCGM_MSG_MODULE_COMMAND, DCGM_REQUEST_ID_NONE, DCGM_ST_OK, payload.size());

        auto sendResult = client.SendMessage(connectionId, std::move(msg), true);
        REQUIRE(sendResult == DCGM_ST_OK);
    }

    allReceived.wait();

    REQUIRE(static_cast<int>(receivedPayloads.size()) == numMessages);
    for (int i = 0; i < numMessages; i++)
    {
        REQUIRE(receivedPayloads[i] == "msg_" + std::to_string(i));
    }
}

TEST_CASE_METHOD(DcgmIpcTestFixture, "DcgmIpc::ErrorCases")
{
    SECTION("Init with invalid domain socket path fails")
    {
        DcgmIpc ipc(1);

        DcgmIpcDomainServerParams_t badParams;
        badParams.domainSocketPath = "/nonexistent/directory/test.sock";

        auto result = ipc.Init(badParams, noopMessage, nullptr, noopDisconnect, nullptr);
        REQUIRE(result != DCGM_ST_OK);
    }

    SECTION("Init with invalid TCP bind address fails")
    {
        DcgmIpc ipc(1);
        auto reservedPort = ReserveUnusedTcpPort();
        auto badParams    = MakeTcpParams(reservedPort.Port(), "not-an-ip-address");
        reservedPort.Release();

        auto result = ipc.Init(badParams, noopMessage, nullptr, noopDisconnect, nullptr);
        REQUIRE(result == DCGM_ST_GENERIC_ERROR);
    }

    SECTION("ConnectTcp to unused port fails")
    {
        DcgmIpc client(1);
        auto clientInit = client.Init(std::nullopt, noopMessage, nullptr, noopDisconnect, nullptr);
        REQUIRE(clientInit == DCGM_ST_OK);

        auto reservedPort = ReserveUnusedTcpPort();
        auto port         = reservedPort.Port();
        reservedPort.Release();

        dcgm_connection_id_t connectionId = DCGM_CONNECTION_ID_NONE;
        auto result                       = client.ConnectTcp("127.0.0.1", port, connectionId, 1000);
        REQUIRE(result == DCGM_ST_CONNECTION_NOT_VALID);
    }

    SECTION("ConnectDomain to nonexistent socket fails")
    {
        DcgmIpc client(1);
        auto clientInit = client.Init(std::nullopt, noopMessage, nullptr, noopDisconnect, nullptr);
        REQUIRE(clientInit == DCGM_ST_OK);

        auto missingSocketPath            = (socketDir.Path() / "dcgm-ipc-no-server.sock").string();
        dcgm_connection_id_t connectionId = DCGM_CONNECTION_ID_NONE;
        auto result                       = client.ConnectDomain(missingSocketPath, connectionId, 1000);
        REQUIRE(result != DCGM_ST_OK);
    }

    SECTION("MonitorSocketFd with negative fd returns BADPARAM")
    {
        DcgmIpc ipc(1);
        auto initResult = ipc.Init(std::nullopt, noopMessage, nullptr, noopDisconnect, nullptr);
        REQUIRE(initResult == DCGM_ST_OK);

        dcgm_connection_id_t connectionId = DCGM_CONNECTION_ID_NONE;
        auto result                       = ipc.MonitorSocketFd(-1, connectionId);
        REQUIRE(result == DCGM_ST_BADPARAM);
    }

    SECTION("SendMessage to invalid connection ID fails")
    {
        DcgmIpc ipc(1);
        auto initResult = ipc.Init(std::nullopt, noopMessage, nullptr, noopDisconnect, nullptr);
        REQUIRE(initResult == DCGM_ST_OK);

        std::string payload = "should fail";
        auto msg            = std::make_unique<DcgmMessage>();
        msg->GetMsgBytesPtr()->assign(payload.begin(), payload.end());
        msg->UpdateMsgHdr(DCGM_MSG_MODULE_COMMAND, DCGM_REQUEST_ID_NONE, DCGM_ST_OK, payload.size());

        auto result = ipc.SendMessage(999, std::move(msg), true);
        REQUIRE(result == DCGM_ST_CONNECTION_NOT_VALID);
    }

    SECTION("SendMessage after CloseConnection fails")
    {
        std::latch disconnected(1);

        auto disconnectHandler = [&](dcgm_connection_id_t, void *) {
            disconnected.count_down();
        };

        DcgmIpc server(1);
        auto serverInit = server.Init(domainParams, noopMessage, nullptr, disconnectHandler, nullptr);
        REQUIRE(serverInit == DCGM_ST_OK);

        DcgmIpc client(1);
        auto clientInit = client.Init(std::nullopt, noopMessage, nullptr, noopDisconnect, nullptr);
        REQUIRE(clientInit == DCGM_ST_OK);

        dcgm_connection_id_t connectionId = DCGM_CONNECTION_ID_NONE;
        auto connectResult                = client.ConnectDomain(socketPath, connectionId, 5000);
        REQUIRE(connectResult == DCGM_ST_OK);

        auto closeResult = client.CloseConnection(connectionId);
        REQUIRE(closeResult == DCGM_ST_OK);

        // Wait for the server to confirm the disconnect was processed
        disconnected.wait();

        std::string payload = "should fail";
        auto msg            = std::make_unique<DcgmMessage>();
        msg->GetMsgBytesPtr()->assign(payload.begin(), payload.end());
        msg->UpdateMsgHdr(DCGM_MSG_MODULE_COMMAND, DCGM_REQUEST_ID_NONE, DCGM_ST_OK, payload.size());

        auto sendResult = client.SendMessage(connectionId, std::move(msg), true);
        REQUIRE(sendResult == DCGM_ST_CONNECTION_NOT_VALID);
    }

    SECTION("Disconnect callback fires when client drops")
    {
        std::promise<dcgm_connection_id_t> disconnectPromise;
        auto disconnectFuture = disconnectPromise.get_future();

        auto disconnectHandler = [&](dcgm_connection_id_t connId, void *) {
            disconnectPromise.set_value(connId);
        };

        DcgmIpc server(1);
        auto serverInit = server.Init(domainParams, noopMessage, nullptr, disconnectHandler, nullptr);
        REQUIRE(serverInit == DCGM_ST_OK);

        dcgm_connection_id_t clientConnId = DCGM_CONNECTION_ID_NONE;
        {
            // Client lives only inside this scope
            DcgmIpc client(1);
            auto clientInit = client.Init(std::nullopt, noopMessage, nullptr, noopDisconnect, nullptr);
            REQUIRE(clientInit == DCGM_ST_OK);

            auto connectResult = client.ConnectDomain(socketPath, clientConnId, 5000);
            REQUIRE(connectResult == DCGM_ST_OK);
        }

        // Client destructor runs here, closing the connection
        auto status = disconnectFuture.wait_for(std::chrono::seconds(5));
        REQUIRE(status == std::future_status::ready);

        // The server saw some connection disconnect
        auto disconnectedId = disconnectFuture.get();
        REQUIRE(disconnectedId != DCGM_CONNECTION_ID_NONE);
    }

    SECTION("ConnectTcp returns CONNECTION_NOT_VALID when the timeout elapses")
    {
        DcgmIpc client(1);
        auto clientInit = client.Init(std::nullopt, noopMessage, nullptr, noopDisconnect, nullptr);
        REQUIRE(clientInit == DCGM_ST_OK);

        // timeoutMs=0 + TEST-NET-1 (RFC 5737) deterministically hits WaitForConnectHelper's timeout branch
        dcgm_connection_id_t connectionId = DCGM_CONNECTION_ID_NONE;
        auto result                       = client.ConnectTcp("192.0.2.1", 12345, connectionId, 0);
        REQUIRE(result == DCGM_ST_CONNECTION_NOT_VALID);
    }
}

TEST_CASE_METHOD(DcgmIpcTestFixture, "DcgmIpc::EdgeCases")
{
    SECTION("Rapid connect/disconnect cycles")
    {
        DcgmIpc server(1);
        auto serverInit = server.Init(domainParams, noopMessage, nullptr, noopDisconnect, nullptr);
        REQUIRE(serverInit == DCGM_ST_OK);

        // Each iteration destroys the client, exercising cleanup under rapid churn
        for (int i = 0; i < 10; i++)
        {
            DcgmIpc client(1);
            auto clientInit = client.Init(std::nullopt, noopMessage, nullptr, noopDisconnect, nullptr);
            REQUIRE(clientInit == DCGM_ST_OK);

            dcgm_connection_id_t connectionId = DCGM_CONNECTION_ID_NONE;
            auto connectResult                = client.ConnectDomain(socketPath, connectionId, 5000);
            REQUIRE(connectResult == DCGM_ST_OK);
            REQUIRE(connectionId != DCGM_CONNECTION_ID_NONE);
        }
    }

    SECTION("CloseConnection with invalid connection ID")
    {
        DcgmIpc ipc(1);
        auto initResult = ipc.Init(std::nullopt, noopMessage, nullptr, noopDisconnect, nullptr);
        REQUIRE(initResult == DCGM_ST_OK);

        // CloseConnection is fire-and-forget, always returns OK if event_base_once succeeds
        auto result = ipc.CloseConnection(999);
        REQUIRE(result == DCGM_ST_OK);
    }

    SECTION("Zero-length payload")
    {
        std::promise<bool> receivedPromise;
        auto receivedFuture = receivedPromise.get_future();

        auto handler = [&](dcgm_connection_id_t, std::unique_ptr<DcgmMessage> msg, void *) {
            auto bytes = msg->GetMsgBytesPtr();
            receivedPromise.set_value(bytes->empty());
        };

        DcgmIpc server(1);
        auto serverInit = server.Init(domainParams, handler, nullptr, noopDisconnect, nullptr);
        REQUIRE(serverInit == DCGM_ST_OK);

        DcgmIpc client(1);
        auto clientInit = client.Init(std::nullopt, noopMessage, nullptr, noopDisconnect, nullptr);
        REQUIRE(clientInit == DCGM_ST_OK);

        dcgm_connection_id_t connectionId = DCGM_CONNECTION_ID_NONE;
        auto connectResult                = client.ConnectDomain(socketPath, connectionId, 5000);
        REQUIRE(connectResult == DCGM_ST_OK);

        // Send a message with zero-length payload
        auto msg = std::make_unique<DcgmMessage>();
        msg->UpdateMsgHdr(DCGM_MSG_MODULE_COMMAND, DCGM_REQUEST_ID_NONE, DCGM_ST_OK, 0);

        auto sendResult = client.SendMessage(connectionId, std::move(msg), true);
        REQUIRE(sendResult == DCGM_ST_OK);

        auto status = receivedFuture.wait_for(std::chrono::seconds(5));
        REQUIRE(status == std::future_status::ready);
        REQUIRE(receivedFuture.get() == true);
    }

    SECTION("Bidirectional messaging")
    {
        std::promise<std::vector<char>> serverReceivedPromise;
        auto serverReceivedFuture = serverReceivedPromise.get_future();

        // Server captures the message and echoes a reply back via userData
        auto serverMsgHandler = [&](dcgm_connection_id_t connId, std::unique_ptr<DcgmMessage> msg, void *userData) {
            auto bytes = msg->GetMsgBytesPtr();
            serverReceivedPromise.set_value(std::vector<char>(bytes->begin(), bytes->end()));

            auto *srv         = static_cast<DcgmIpc *>(userData);
            std::string reply = "reply from server";
            auto replyMsg     = std::make_unique<DcgmMessage>();
            replyMsg->GetMsgBytesPtr()->assign(reply.begin(), reply.end());
            replyMsg->UpdateMsgHdr(DCGM_MSG_MODULE_COMMAND, DCGM_REQUEST_ID_NONE, DCGM_ST_OK, reply.size());
            srv->SendMessage(connId, std::move(replyMsg), false);
        };

        DcgmIpc server(1);
        auto serverInit = server.Init(domainParams, serverMsgHandler, &server, noopDisconnect, nullptr);
        REQUIRE(serverInit == DCGM_ST_OK);

        std::promise<std::vector<char>> clientReceivedPromise;
        auto clientReceivedFuture = clientReceivedPromise.get_future();

        auto clientHandler = [&](dcgm_connection_id_t, std::unique_ptr<DcgmMessage> msg, void *) {
            auto bytes = msg->GetMsgBytesPtr();
            clientReceivedPromise.set_value(std::vector<char>(bytes->begin(), bytes->end()));
        };

        DcgmIpc client(1);
        auto clientInit = client.Init(std::nullopt, clientHandler, nullptr, noopDisconnect, nullptr);
        REQUIRE(clientInit == DCGM_ST_OK);

        dcgm_connection_id_t connectionId = DCGM_CONNECTION_ID_NONE;
        auto connectResult                = client.ConnectDomain(socketPath, connectionId, 5000);
        REQUIRE(connectResult == DCGM_ST_OK);

        std::string payload = "hello from client";
        auto msg            = std::make_unique<DcgmMessage>();
        msg->GetMsgBytesPtr()->assign(payload.begin(), payload.end());
        msg->UpdateMsgHdr(DCGM_MSG_MODULE_COMMAND, DCGM_REQUEST_ID_NONE, DCGM_ST_OK, payload.size());

        auto sendResult = client.SendMessage(connectionId, std::move(msg), true);
        REQUIRE(sendResult == DCGM_ST_OK);

        auto serverStatus = serverReceivedFuture.wait_for(std::chrono::seconds(5));
        REQUIRE(serverStatus == std::future_status::ready);
        auto serverReceived = serverReceivedFuture.get();
        REQUIRE(std::string(serverReceived.begin(), serverReceived.end()) == "hello from client");

        auto clientStatus = clientReceivedFuture.wait_for(std::chrono::seconds(5));
        REQUIRE(clientStatus == std::future_status::ready);
        auto clientReceived = clientReceivedFuture.get();
        REQUIRE(std::string(clientReceived.begin(), clientReceived.end()) == "reply from server");
    }

    SECTION("Server-initiated CloseConnection drops the client")
    {
        // Disconnect latch fires when the client sees the server-initiated drop
        std::latch clientDisconnected(1);
        auto disconnectHandler = [&](dcgm_connection_id_t, void *) {
            clientDisconnected.count_down();
        };

        // Server tracks the connection ID it accepts so it can close it
        std::promise<dcgm_connection_id_t> serverConnIdPromise;
        auto serverConnIdFuture = serverConnIdPromise.get_future();
        auto serverMsgHandler   = [&](dcgm_connection_id_t connId, std::unique_ptr<DcgmMessage>, void *) {
            serverConnIdPromise.set_value(connId);
        };

        DcgmIpc server(1);
        auto serverInit = server.Init(domainParams, serverMsgHandler, nullptr, noopDisconnect, nullptr);
        REQUIRE(serverInit == DCGM_ST_OK);

        DcgmIpc client(1);
        auto clientInit = client.Init(std::nullopt, noopMessage, nullptr, disconnectHandler, nullptr);
        REQUIRE(clientInit == DCGM_ST_OK);

        dcgm_connection_id_t clientConnId = DCGM_CONNECTION_ID_NONE;
        auto connectResult                = client.ConnectDomain(socketPath, clientConnId, 5000);
        REQUIRE(connectResult == DCGM_ST_OK);

        // Send a message so the server learns its accepted connection ID
        std::string payload = "hello";
        auto msg            = std::make_unique<DcgmMessage>();
        msg->GetMsgBytesPtr()->assign(payload.begin(), payload.end());
        msg->UpdateMsgHdr(DCGM_MSG_MODULE_COMMAND, DCGM_REQUEST_ID_NONE, DCGM_ST_OK, payload.size());
        auto sendResult = client.SendMessage(clientConnId, std::move(msg), true);
        REQUIRE(sendResult == DCGM_ST_OK);

        auto serverConnId = serverConnIdFuture.get();
        auto closeResult  = server.CloseConnection(serverConnId);
        REQUIRE(closeResult == DCGM_ST_OK);

        clientDisconnected.wait();
    }
}

TEST_CASE_METHOD(DcgmIpcTestFixture, "DcgmIpc::TcpLifecycle")
{
    SECTION("Init with TCP, no bind address (all interfaces)")
    {
        // Empty bindIPAddress, binds all interfaces via IPv6 with IPV6_V6ONLY=0
        std::unique_ptr<DcgmIpc> ipc;
        DcgmIpcTcpServerParams_t boundParams;
        auto result = InitTcpServerOnReservedPort(ipc, boundParams, noopMessage, nullptr, noopDisconnect, nullptr, "");
        REQUIRE(result == DCGM_ST_OK);
    }

    SECTION("Init with TCP, IPv4 bind address")
    {
        std::unique_ptr<DcgmIpc> tcpServer;
        DcgmIpcTcpServerParams_t tcpParams;
        auto serverInit
            = InitTcpServerOnReservedPort(tcpServer, tcpParams, noopMessage, nullptr, noopDisconnect, nullptr);
        REQUIRE(serverInit == DCGM_ST_OK);
    }

    SECTION("Init with TCP, IPv6 bind address")
    {
        std::unique_ptr<DcgmIpc> tcpServer;
        DcgmIpcTcpServerParams_t tcpParams;
        auto serverInit
            = InitTcpServerOnReservedPort(tcpServer, tcpParams, noopMessage, nullptr, noopDisconnect, nullptr, "::1");
        REQUIRE(serverInit == DCGM_ST_OK);
    }

    SECTION("Init with TCP, invalid bind address fails")
    {
        // Neither inet_aton nor inet_pton can parse this, returns DCGM_ST_GENERIC_ERROR
        std::unique_ptr<DcgmIpc> tcpServer;
        DcgmIpcTcpServerParams_t tcpParams;
        auto serverInit = InitTcpServerOnReservedPort(
            tcpServer, tcpParams, noopMessage, nullptr, noopDisconnect, nullptr, "not.an.ip");
        REQUIRE(serverInit != DCGM_ST_OK);
    }

    SECTION("Init with TCP, port already in use fails")
    {
        // First instance binds a reserved port successfully
        std::unique_ptr<DcgmIpc> first;
        DcgmIpcTcpServerParams_t boundParams;
        auto firstResult
            = InitTcpServerOnReservedPort(first, boundParams, noopMessage, nullptr, noopDisconnect, nullptr);
        REQUIRE(firstResult == DCGM_ST_OK);

        // Second instance on the same port should fail at bind()
        DcgmIpc second(1);
        auto secondResult = second.Init(boundParams, noopMessage, nullptr, noopDisconnect, nullptr);
        REQUIRE(secondResult != DCGM_ST_OK);
    }
}

TEST_CASE_METHOD(DcgmIpcTestFixture, "DcgmIpc::TcpConnections")
{
    // Server instance, shared by all sections
    std::unique_ptr<DcgmIpc> server;
    DcgmIpcTcpServerParams_t tcpParams;
    auto initResult = InitTcpServerOnReservedPort(
        server, tcpParams, noopMessage, nullptr, noopDisconnect, nullptr, /*bindIPAddress=*/"");
    REQUIRE(initResult == DCGM_ST_OK);

    SECTION("Client connects to server via TCP (IPv4)")
    {
        DcgmIpc client(1);
        auto clientInit = client.Init(std::nullopt, noopMessage, nullptr, noopDisconnect, nullptr);
        REQUIRE(clientInit == DCGM_ST_OK);

        dcgm_connection_id_t connectionId = DCGM_CONNECTION_ID_NONE;
        auto result                       = client.ConnectTcp("127.0.0.1", tcpParams.port, connectionId, 5000);

        REQUIRE(result == DCGM_ST_OK);
        REQUIRE(connectionId != DCGM_CONNECTION_ID_NONE);
    }

    SECTION("Client connects to server via TCP (IPv6)")
    {
        // "::1" is recognized as IPv6 via inet_pton, exercises the family override path
        DcgmIpc client(1);
        auto clientInit = client.Init(std::nullopt, noopMessage, nullptr, noopDisconnect, nullptr);
        REQUIRE(clientInit == DCGM_ST_OK);

        dcgm_connection_id_t connectionId = DCGM_CONNECTION_ID_NONE;
        auto result                       = client.ConnectTcp("::1", tcpParams.port, connectionId, 5000);

        REQUIRE(result == DCGM_ST_OK);
        REQUIRE(connectionId != DCGM_CONNECTION_ID_NONE);
    }

    SECTION("Client connects to server via TCP (IPv6 bracket notation)")
    {
        // "[::1]" is stripped to "::1" before connection
        DcgmIpc client(1);
        auto clientInit = client.Init(std::nullopt, noopMessage, nullptr, noopDisconnect, nullptr);
        REQUIRE(clientInit == DCGM_ST_OK);

        dcgm_connection_id_t connectionId = DCGM_CONNECTION_ID_NONE;
        auto result                       = client.ConnectTcp("[::1]", tcpParams.port, connectionId, 5000);

        REQUIRE(result == DCGM_ST_OK);
        REQUIRE(connectionId != DCGM_CONNECTION_ID_NONE);
    }

    SECTION("Multiple connections get unique connection IDs")
    {
        DcgmIpc client(1);
        auto clientInit = client.Init(std::nullopt, noopMessage, nullptr, noopDisconnect, nullptr);
        REQUIRE(clientInit == DCGM_ST_OK);

        dcgm_connection_id_t connId1 = DCGM_CONNECTION_ID_NONE;
        dcgm_connection_id_t connId2 = DCGM_CONNECTION_ID_NONE;

        auto connectResult1 = client.ConnectTcp("127.0.0.1", tcpParams.port, connId1, 5000);
        REQUIRE(connectResult1 == DCGM_ST_OK);

        auto connectResult2 = client.ConnectTcp("127.0.0.1", tcpParams.port, connId2, 5000);
        REQUIRE(connectResult2 == DCGM_ST_OK);

        REQUIRE(connId1 != DCGM_CONNECTION_ID_NONE);
        REQUIRE(connId2 != DCGM_CONNECTION_ID_NONE);
        REQUIRE(connId1 != connId2);
    }

    SECTION("CloseConnection after connect")
    {
        DcgmIpc client(1);
        auto clientInit = client.Init(std::nullopt, noopMessage, nullptr, noopDisconnect, nullptr);
        REQUIRE(clientInit == DCGM_ST_OK);

        dcgm_connection_id_t connectionId = DCGM_CONNECTION_ID_NONE;
        auto connectResult                = client.ConnectTcp("127.0.0.1", tcpParams.port, connectionId, 5000);
        REQUIRE(connectResult == DCGM_ST_OK);
        REQUIRE(connectionId != DCGM_CONNECTION_ID_NONE);

        auto closeResult = client.CloseConnection(connectionId);
        REQUIRE(closeResult == DCGM_ST_OK);
    }

    SECTION("ConnectTcp to a closed local port fails")
    {
        auto reservedPort = ReserveUnusedTcpPort();
        auto closedPort   = static_cast<unsigned short>(reservedPort.Port());
        reservedPort.Release();

        DcgmIpc client(1);
        auto clientInit = client.Init(std::nullopt, noopMessage, nullptr, noopDisconnect, nullptr);
        REQUIRE(clientInit == DCGM_ST_OK);

        // Kernel returns RST, the connection fails asynchronously via EventCB BEV_EVENT_ERROR
        dcgm_connection_id_t connectionId = DCGM_CONNECTION_ID_NONE;
        auto result                       = client.ConnectTcp("127.0.0.1", closedPort, connectionId, 5000);
        REQUIRE(result != DCGM_ST_OK);
    }
}

TEST_CASE_METHOD(DcgmIpcTestFixture, "DcgmIpc::TcpMessaging")
{
    // Captures the first message the server receives
    std::promise<std::vector<char>> receivedPromise;
    auto receivedFuture = receivedPromise.get_future();

    auto capturingHandler = [&](dcgm_connection_id_t, std::unique_ptr<DcgmMessage> msg, void *) {
        auto bytes = msg->GetMsgBytesPtr();
        receivedPromise.set_value(std::vector<char>(bytes->begin(), bytes->end()));
    };

    std::unique_ptr<DcgmIpc> tcpServer;
    DcgmIpcTcpServerParams_t tcpParams;
    auto serverInit
        = InitTcpServerOnReservedPort(tcpServer, tcpParams, capturingHandler, nullptr, noopDisconnect, nullptr);
    REQUIRE(serverInit == DCGM_ST_OK);

    SECTION("Client sends message to server")
    {
        DcgmIpc client(1);
        auto clientInit = client.Init(std::nullopt, noopMessage, nullptr, noopDisconnect, nullptr);
        REQUIRE(clientInit == DCGM_ST_OK);

        dcgm_connection_id_t connectionId = DCGM_CONNECTION_ID_NONE;
        auto connectResult = client.ConnectTcp(tcpParams.bindIPAddress, tcpParams.port, connectionId, 5000);
        REQUIRE(connectResult == DCGM_ST_OK);

        std::string payload = "hello from client";
        auto msg            = std::make_unique<DcgmMessage>();
        msg->GetMsgBytesPtr()->assign(payload.begin(), payload.end());
        msg->UpdateMsgHdr(DCGM_MSG_MODULE_COMMAND, DCGM_REQUEST_ID_NONE, DCGM_ST_OK, payload.size());

        auto sendResult = client.SendMessage(connectionId, std::move(msg), true);
        REQUIRE(sendResult == DCGM_ST_OK);

        auto status = receivedFuture.wait_for(std::chrono::seconds(5));
        REQUIRE(status == std::future_status::ready);

        auto received = receivedFuture.get();
        std::string receivedStr(received.begin(), received.end());
        REQUIRE(receivedStr == "hello from client");
    }

    SECTION("Send with waitForSend false")
    {
        DcgmIpc client(1);
        auto clientInit = client.Init(std::nullopt, noopMessage, nullptr, noopDisconnect, nullptr);
        REQUIRE(clientInit == DCGM_ST_OK);

        dcgm_connection_id_t connectionId = DCGM_CONNECTION_ID_NONE;
        auto connectResult = client.ConnectTcp(tcpParams.bindIPAddress, tcpParams.port, connectionId, 5000);
        REQUIRE(connectResult == DCGM_ST_OK);

        std::string payload = "fire and forget";
        auto msg            = std::make_unique<DcgmMessage>();
        msg->GetMsgBytesPtr()->assign(payload.begin(), payload.end());
        msg->UpdateMsgHdr(DCGM_MSG_MODULE_COMMAND, DCGM_REQUEST_ID_NONE, DCGM_ST_OK, payload.size());

        auto sendResult = client.SendMessage(connectionId, std::move(msg), false);
        REQUIRE(sendResult == DCGM_ST_OK);

        auto status = receivedFuture.wait_for(std::chrono::seconds(5));
        REQUIRE(status == std::future_status::ready);
    }
}

TEST_CASE_METHOD(DcgmIpcTestFixture, "DcgmIpc::ProtocolErrors")
{
    // Disconnect latch fires when the IPC layer drops the bad connection
    std::latch disconnected(1);
    auto disconnectHandler = [&](dcgm_connection_id_t, void *) {
        disconnected.count_down();
    };

    DcgmIpc server(1);
    auto initResult = server.Init(std::nullopt, noopMessage, nullptr, disconnectHandler, nullptr);
    REQUIRE(initResult == DCGM_ST_OK);

    // socketpair gives us one end the server monitors and one end we own for raw writes
    int fds[2];
    int rc = socketpair(AF_UNIX, SOCK_STREAM, 0, fds);
    REQUIRE(rc == 0);
    DcgmNs::Defer cleanup([&] { close(fds[1]); });

    dcgm_connection_id_t connectionId = DCGM_CONNECTION_ID_NONE;
    auto monitorResult                = server.MonitorSocketFd(fds[0], connectionId);
    REQUIRE(monitorResult == DCGM_ST_OK);

    SECTION("Bad magic number drops the connection")
    {
        // Build a valid header, then corrupt only the magic field
        DcgmMessage msg;
        msg.UpdateMsgHdr(DCGM_MSG_MODULE_COMMAND, DCGM_REQUEST_ID_NONE, DCGM_ST_OK, 0);
        auto *hdr  = msg.GetMessageHdr();
        hdr->msgId = 0xDEADBEEF;

        ssize_t written = write(fds[1], hdr, sizeof(*hdr));
        REQUIRE(written == static_cast<ssize_t>(sizeof(*hdr)));

        // Server reads the bad header, closes the bev, fires the disconnect callback
        disconnected.wait();
    }

    SECTION("Oversized message length drops the connection")
    {
        // Build a valid header, then set length beyond DCGM_PROTO_MAX_MESSAGE_SIZE
        DcgmMessage msg;
        msg.UpdateMsgHdr(DCGM_MSG_MODULE_COMMAND, DCGM_REQUEST_ID_NONE, DCGM_ST_OK, 0);
        auto *hdr   = msg.GetMessageHdr();
        hdr->length = DCGM_PROTO_MAX_MESSAGE_SIZE + 1;

        ssize_t written = write(fds[1], hdr, sizeof(*hdr));
        REQUIRE(written == static_cast<ssize_t>(sizeof(*hdr)));

        disconnected.wait();
    }

    SECTION("Negative message length drops the connection")
    {
        // length is signed in dcgm_message_header_t, negative values must be rejected
        DcgmMessage msg;
        msg.UpdateMsgHdr(DCGM_MSG_MODULE_COMMAND, DCGM_REQUEST_ID_NONE, DCGM_ST_OK, 0);
        auto *hdr   = msg.GetMessageHdr();
        hdr->length = -1;

        ssize_t written = write(fds[1], hdr, sizeof(*hdr));
        REQUIRE(written == static_cast<ssize_t>(sizeof(*hdr)));

        disconnected.wait();
    }
}

TEST_CASE_METHOD(DcgmIpcHangDetectFixture, "DcgmIpc::HangDetect")
{
    SECTION("Init with HangDetectMonitor registers and unregisters the IPC thread")
    {
        // Two-arg constructor wires the monitor into m_workersPool and m_monitor
        DcgmIpc ipc(1, &hangDetectMonitor);
        auto result = ipc.Init(domainParams, noopMessage, nullptr, noopDisconnect, nullptr);
        REQUIRE(result == DCGM_ST_OK);

        // Destructor runs at scope exit, exercising RemoveMonitoredTask on the IPC thread
    }

    SECTION("Init with null HangDetectMonitor falls back to no monitoring")
    {
        // Two-arg constructor with nullptr exercises the disabled-monitoring path in run()
        DcgmIpc ipc(1, nullptr);
        auto result = ipc.Init(domainParams, noopMessage, nullptr, noopDisconnect, nullptr);
        REQUIRE(result == DCGM_ST_OK);
    }
}
