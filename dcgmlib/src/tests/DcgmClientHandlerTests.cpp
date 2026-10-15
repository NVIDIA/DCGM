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

#include <DcgmClientHandler.h>

#include <DcgmProtocol.h>
#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <future>
#include <memory>
#include <optional>
#include <utility>

namespace
{
class TestDcgmClientHandler : public DcgmClientHandler
{
public:
    using DcgmClientHandler::AddBlockingRequest;
    using DcgmClientHandler::AddPersistentRequest;
    using DcgmClientHandler::GetNextRequestId;
    using DcgmClientHandler::m_blockingReqs;
    using DcgmClientHandler::m_connectionAttributes;
    using DcgmClientHandler::m_connectionRequests;
    using DcgmClientHandler::m_persistentReqs;
    using DcgmClientHandler::m_requestId;
    using DcgmClientHandler::ProcessDisconnect;
    using DcgmClientHandler::ProcessMessage;
    using DcgmClientHandler::RemoveBlockingRequest;
    using DcgmClientHandler::RemovePersistentRequest;
};

class TestDcgmRequest : public DcgmRequest
{
public:
    explicit TestDcgmRequest(dcgm_request_id_t requestId)
        : DcgmRequest(requestId)
    {}

    [[nodiscard]] int ProcessMessage(std::unique_ptr<DcgmMessage> msg) override
    {
        processedCount++;
        lastMessageType = msg->GetMsgType();
        lastStatus      = msg->GetMessageHdr()->status;
        return processReturn;
    }

    int processReturn   = DCGM_ST_OK;
    int processedCount  = 0;
    int lastMessageType = 0;
    int lastStatus      = DCGM_ST_OK;
};

constexpr auto c_invalidHostengineHandle = static_cast<dcgmHandle_t>(0x1234);
constexpr auto c_noWait                  = std::chrono::milliseconds(0);

std::unique_ptr<DcgmMessage> MakeMessage(int messageType, dcgm_request_id_t requestId, dcgmReturn_t status)
{
    auto message = std::make_unique<DcgmMessage>();
    message->UpdateMsgHdr(messageType, requestId, status, 0);
    return message;
}
} // namespace

TEST_CASE("DcgmClientHandler: request ids skip the reserved value")
{
    TestDcgmClientHandler handler;

    handler.m_requestId = DCGM_REQUEST_ID_NONE;

    CHECK(handler.GetNextRequestId() == 1);
    CHECK(handler.GetNextRequestId() == 2);
}

TEST_CASE("DcgmClientHandler: blocking request helpers publish responses and clean indexes")
{
    TestDcgmClientHandler handler;
    dcgm_connection_id_t const connectionId = 44;
    dcgm_request_id_t const requestId       = 7;

    auto future = handler.AddBlockingRequest(connectionId, requestId);
    REQUIRE(handler.m_blockingReqs.contains(requestId));
    REQUIRE(handler.m_connectionRequests[connectionId].contains(requestId));

    SECTION("removing with an error satisfies the waiting future")
    {
        handler.RemoveBlockingRequest(connectionId, requestId, DCGM_ST_CONNECTION_NOT_VALID);

        REQUIRE(future.wait_for(c_noWait) == std::future_status::ready);
        auto response = future.get();
        CHECK(response.dcgmReturn == DCGM_ST_CONNECTION_NOT_VALID);
        CHECK(response.response == nullptr);
        CHECK_FALSE(handler.m_blockingReqs.contains(requestId));
        CHECK_FALSE(handler.m_connectionRequests[connectionId].contains(requestId));
    }

    SECTION("a matching response message satisfies the waiting future")
    {
        handler.ProcessMessage(connectionId, MakeMessage(DCGM_MSG_MODULE_COMMAND, requestId, DCGM_ST_OK));

        REQUIRE(future.wait_for(c_noWait) == std::future_status::ready);
        auto response = future.get();
        CHECK(response.dcgmReturn == DCGM_ST_OK);
        REQUIRE(response.response != nullptr);
        CHECK(response.response->GetRequestId() == requestId);
        CHECK(response.response->GetMsgType() == DCGM_MSG_MODULE_COMMAND);

        handler.RemoveBlockingRequest(connectionId, requestId, std::nullopt);
        CHECK_FALSE(handler.m_blockingReqs.contains(requestId));
        CHECK_FALSE(handler.m_connectionRequests[connectionId].contains(requestId));
    }
}

TEST_CASE("DcgmClientHandler: persistent request helpers route async messages")
{
    TestDcgmClientHandler handler;
    dcgm_connection_id_t const connectionId = 55;
    dcgm_request_id_t const requestId       = 9;
    auto request                            = std::make_unique<TestDcgmRequest>(requestId);
    auto *requestPtr                        = request.get();

    handler.AddPersistentRequest(connectionId, std::move(request));
    REQUIRE(handler.m_persistentReqs.contains(requestId));
    REQUIRE(handler.m_connectionRequests[connectionId].contains(requestId));

    SECTION("async messages are delivered to persistent requests without blocking state")
    {
        handler.ProcessMessage(connectionId, MakeMessage(DCGM_MSG_POLICY_NOTIFY, requestId, DCGM_ST_OK));

        CHECK(requestPtr->processedCount == 1);
        CHECK(requestPtr->lastMessageType == DCGM_MSG_POLICY_NOTIFY);
        CHECK(requestPtr->lastStatus == DCGM_ST_OK);
        CHECK(handler.m_persistentReqs.contains(requestId));
    }

    SECTION("removing a persistent request erases request and connection indexes")
    {
        handler.RemovePersistentRequest(connectionId, requestId);

        CHECK_FALSE(handler.m_persistentReqs.contains(requestId));
        CHECK_FALSE(handler.m_connectionRequests[connectionId].contains(requestId));
    }
}

TEST_CASE("DcgmClientHandler: disconnect clears outstanding requests and attributes")
{
    TestDcgmClientHandler handler;
    dcgm_connection_id_t const connectionId = 66;
    dcgm_request_id_t const blockingId      = 11;
    dcgm_request_id_t const persistentId    = 12;

    auto future = handler.AddBlockingRequest(connectionId, blockingId);
    handler.AddPersistentRequest(connectionId, std::make_unique<TestDcgmRequest>(persistentId));
    handler.m_connectionAttributes.emplace(connectionId, "1.2.3~build");

    handler.ProcessDisconnect(connectionId);

    REQUIRE(future.wait_for(c_noWait) == std::future_status::ready);
    CHECK(future.get().dcgmReturn == DCGM_ST_CONNECTION_NOT_VALID);
    CHECK_FALSE(handler.m_blockingReqs.contains(blockingId));
    CHECK_FALSE(handler.m_persistentReqs.contains(persistentId));
    CHECK_FALSE(handler.m_connectionRequests.contains(connectionId));
    CHECK_FALSE(handler.m_connectionAttributes.contains(connectionId));
}

TEST_CASE("DcgmClientHandler: ignores messages that cannot be matched to tracked requests")
{
    TestDcgmClientHandler handler;
    dcgm_connection_id_t const connectionId = 77;

    SECTION("reserved request ids do not create state")
    {
        handler.ProcessMessage(connectionId,
                               MakeMessage(DCGM_MSG_MODULE_COMMAND, DCGM_REQUEST_ID_NONE, DCGM_ST_GENERIC_ERROR));

        CHECK(handler.m_blockingReqs.empty());
        CHECK(handler.m_persistentReqs.empty());
        CHECK(handler.m_connectionRequests.empty());
    }

    SECTION("unknown request ids do not create state")
    {
        handler.ProcessMessage(connectionId, MakeMessage(DCGM_MSG_MODULE_COMMAND, 99, DCGM_ST_OK));

        CHECK(handler.m_blockingReqs.empty());
        CHECK(handler.m_persistentReqs.empty());
        CHECK(handler.m_connectionRequests.empty());
    }
}

SCENARIO("DcgmClientHandler: keeps async, error, and disconnect indexes isolated")
{
    TestDcgmClientHandler handler;

    GIVEN("a blocking request and a persistent request on the same connection")
    {
        dcgm_connection_id_t const connectionId = 88;
        dcgm_request_id_t const blockingId      = 21;
        dcgm_request_id_t const persistentId    = 22;

        auto future               = handler.AddBlockingRequest(connectionId, blockingId);
        auto request              = std::make_unique<TestDcgmRequest>(persistentId);
        auto *requestPtr          = request.get();
        requestPtr->processReturn = DCGM_ST_GENERIC_ERROR;
        handler.AddPersistentRequest(connectionId, std::move(request));

        WHEN("an async notification arrives for the blocking request id")
        {
            handler.ProcessMessage(connectionId, MakeMessage(DCGM_MSG_POLICY_NOTIFY, blockingId, DCGM_ST_OK));

            THEN("the blocking request remains pending and the persistent request is not invoked")
            {
                CHECK(future.wait_for(c_noWait) == std::future_status::timeout);
                CHECK(handler.m_blockingReqs.contains(blockingId));
                CHECK(handler.m_connectionRequests[connectionId].contains(blockingId));
                CHECK(requestPtr->processedCount == 0);
            }
        }

        WHEN("the persistent request returns an error while processing a notification")
        {
            handler.ProcessMessage(connectionId,
                                   MakeMessage(DCGM_MSG_POLICY_NOTIFY, persistentId, DCGM_ST_GENERIC_ERROR));

            THEN("the notification is recorded and the persistent subscription remains installed")
            {
                CHECK(requestPtr->processedCount == 1);
                CHECK(requestPtr->lastMessageType == DCGM_MSG_POLICY_NOTIFY);
                CHECK(requestPtr->lastStatus == DCGM_ST_GENERIC_ERROR);
                CHECK(handler.m_persistentReqs.contains(persistentId));
                CHECK(handler.m_connectionRequests[connectionId].contains(persistentId));
            }
        }

        WHEN("unknown request ids are removed for the same connection")
        {
            dcgm_request_id_t const unknownBlockingId   = 999;
            dcgm_request_id_t const unknownPersistentId = 1000;
            handler.RemoveBlockingRequest(connectionId, unknownBlockingId, DCGM_ST_CONNECTION_NOT_VALID);
            handler.RemovePersistentRequest(connectionId, unknownPersistentId);

            THEN("tracked request state for the connection is preserved")
            {
                CHECK(handler.m_blockingReqs.contains(blockingId));
                CHECK(handler.m_persistentReqs.contains(persistentId));
                CHECK(handler.m_connectionRequests[connectionId].contains(blockingId));
                CHECK(handler.m_connectionRequests[connectionId].contains(persistentId));
                CHECK(future.wait_for(c_noWait) == std::future_status::timeout);
            }
        }
    }

    GIVEN("two connections with outstanding blocking and persistent requests")
    {
        dcgm_connection_id_t const firstConnection  = 91;
        dcgm_connection_id_t const secondConnection = 92;
        dcgm_request_id_t const firstBlockingId     = 31;
        dcgm_request_id_t const secondBlockingId    = 32;
        dcgm_request_id_t const firstPersistentId   = 33;
        dcgm_request_id_t const secondPersistentId  = 34;

        auto firstFuture  = handler.AddBlockingRequest(firstConnection, firstBlockingId);
        auto secondFuture = handler.AddBlockingRequest(secondConnection, secondBlockingId);
        handler.AddPersistentRequest(firstConnection, std::make_unique<TestDcgmRequest>(firstPersistentId));
        handler.AddPersistentRequest(secondConnection, std::make_unique<TestDcgmRequest>(secondPersistentId));
        handler.m_connectionAttributes.emplace(firstConnection, "first");
        handler.m_connectionAttributes.emplace(secondConnection, "second");

        WHEN("the first connection disconnects")
        {
            handler.ProcessDisconnect(firstConnection);

            THEN("only the first connection's requests and attributes are cleared")
            {
                REQUIRE(firstFuture.wait_for(c_noWait) == std::future_status::ready);
                CHECK(firstFuture.get().dcgmReturn == DCGM_ST_CONNECTION_NOT_VALID);
                CHECK(secondFuture.wait_for(c_noWait) == std::future_status::timeout);
                CHECK_FALSE(handler.m_blockingReqs.contains(firstBlockingId));
                CHECK(handler.m_blockingReqs.contains(secondBlockingId));
                CHECK_FALSE(handler.m_persistentReqs.contains(firstPersistentId));
                CHECK(handler.m_persistentReqs.contains(secondPersistentId));
                CHECK_FALSE(handler.m_connectionRequests.contains(firstConnection));
                CHECK(handler.m_connectionRequests[secondConnection].contains(secondBlockingId));
                CHECK(handler.m_connectionRequests[secondConnection].contains(secondPersistentId));
                CHECK_FALSE(handler.m_connectionAttributes.contains(firstConnection));
                CHECK(handler.m_connectionAttributes.contains(secondConnection));
            }
        }
    }
}

SCENARIO("DcgmClientHandler: completes blocking requests without disturbing neighboring indexes")
{
    TestDcgmClientHandler handler;

    GIVEN("two blocking requests on one connection and one on another")
    {
        dcgm_connection_id_t const firstConnection  = 401;
        dcgm_connection_id_t const secondConnection = 402;
        dcgm_request_id_t const firstRequest        = 501;
        dcgm_request_id_t const secondRequest       = 502;
        dcgm_request_id_t const thirdRequest        = 503;

        auto firstFuture  = handler.AddBlockingRequest(firstConnection, firstRequest);
        auto secondFuture = handler.AddBlockingRequest(firstConnection, secondRequest);
        auto thirdFuture  = handler.AddBlockingRequest(secondConnection, thirdRequest);

        WHEN("a normal response arrives for the middle request")
        {
            handler.ProcessMessage(firstConnection, MakeMessage(DCGM_MSG_MODULE_COMMAND, secondRequest, DCGM_ST_OK));

            THEN("only that request future is completed and the other request indexes stay pending")
            {
                REQUIRE(secondFuture.wait_for(c_noWait) == std::future_status::ready);
                auto response = secondFuture.get();
                REQUIRE(response.response != nullptr);
                CHECK(response.dcgmReturn == DCGM_ST_OK);
                CHECK(response.response->GetMessageHdr()->requestId == secondRequest);
                CHECK(response.response->GetMessageHdr()->status == DCGM_ST_OK);

                CHECK(firstFuture.wait_for(c_noWait) == std::future_status::timeout);
                CHECK(thirdFuture.wait_for(c_noWait) == std::future_status::timeout);
                CHECK(handler.m_blockingReqs.contains(firstRequest));
                CHECK(handler.m_blockingReqs.contains(secondRequest));
                CHECK(handler.m_blockingReqs.contains(thirdRequest));
                CHECK(handler.m_connectionRequests[firstConnection].contains(firstRequest));
                CHECK(handler.m_connectionRequests[firstConnection].contains(secondRequest));
                CHECK(handler.m_connectionRequests[secondConnection].contains(thirdRequest));
            }
        }

        WHEN("a blocking request is explicitly removed with no completion status")
        {
            handler.RemoveBlockingRequest(firstConnection, firstRequest, std::nullopt);

            THEN("the future observes the broken promise and bookkeeping for that request is erased")
            {
                REQUIRE(firstFuture.wait_for(c_noWait) == std::future_status::ready);
                REQUIRE_THROWS_AS(firstFuture.get(), std::future_error);
                CHECK_FALSE(handler.m_blockingReqs.contains(firstRequest));
                CHECK_FALSE(handler.m_connectionRequests[firstConnection].contains(firstRequest));
                CHECK(handler.m_blockingReqs.contains(secondRequest));
                CHECK(handler.m_blockingReqs.contains(thirdRequest));
            }
        }
    }
}

SCENARIO("DcgmClientHandler: separates initial responses from persistent notifications")
{
    TestDcgmClientHandler handler;

    GIVEN("a request id tracked as both blocking and persistent")
    {
        dcgm_connection_id_t const connectionId = 601;
        dcgm_request_id_t const requestId       = 701;

        auto future      = handler.AddBlockingRequest(connectionId, requestId);
        auto request     = std::make_unique<TestDcgmRequest>(requestId);
        auto *requestPtr = request.get();
        handler.AddPersistentRequest(connectionId, std::move(request));

        WHEN("the initial module response arrives")
        {
            handler.ProcessMessage(connectionId,
                                   MakeMessage(DCGM_MSG_MODULE_COMMAND, requestId, DCGM_ST_GENERIC_ERROR));

            THEN("the blocking future receives the response and the persistent request is left for later notifications")
            {
                REQUIRE(future.wait_for(c_noWait) == std::future_status::ready);
                auto response = future.get();
                CHECK(response.dcgmReturn == DCGM_ST_OK);
                REQUIRE(response.response != nullptr);
                CHECK(response.response->GetMessageHdr()->status == DCGM_ST_GENERIC_ERROR);
                CHECK(requestPtr->processedCount == 0);
                CHECK(handler.m_persistentReqs.contains(requestId));
                CHECK(handler.m_connectionRequests[connectionId].contains(requestId));
            }
        }

        WHEN("the blocking response is completed and a notification arrives later")
        {
            handler.ProcessMessage(connectionId, MakeMessage(DCGM_MSG_MODULE_COMMAND, requestId, DCGM_ST_OK));
            REQUIRE(future.wait_for(c_noWait) == std::future_status::ready);
            (void)future.get();
            handler.RemoveBlockingRequest(connectionId, requestId, std::nullopt);

            handler.ProcessMessage(connectionId, MakeMessage(DCGM_MSG_POLICY_NOTIFY, requestId, DCGM_ST_NOT_SUPPORTED));

            THEN("the notification is delivered through the persistent request path")
            {
                CHECK(requestPtr->processedCount == 1);
                CHECK(requestPtr->lastMessageType == DCGM_MSG_POLICY_NOTIFY);
                CHECK(requestPtr->lastStatus == DCGM_ST_NOT_SUPPORTED);
                CHECK(handler.m_persistentReqs.contains(requestId));
            }
        }
    }
}

SCENARIO("DcgmClientHandler: validates connection input before network work")
{
    TestDcgmClientHandler handler;

    GIVEN("an invalid endpoint string and a sentinel handle")
    {
        constexpr unsigned int connectionTimeoutMs = 1;
        dcgmHandle_t handle                        = c_invalidHostengineHandle;

        WHEN("a hostengine connection is requested")
        {
            auto const result
                = handler.GetConnHandleForHostEngine("bad-scheme://localhost", &handle, connectionTimeoutMs);

            THEN("the parse failure is reported without changing the caller handle")
            {
                CHECK(result == DCGM_ST_BADPARAM);
                CHECK(handle == c_invalidHostengineHandle);
            }
        }
    }

    GIVEN("connection attributes without outstanding request indexes")
    {
        dcgm_connection_id_t const connectionId = 801;
        handler.m_connectionAttributes.emplace(connectionId, "4.6.0~test");

        WHEN("the connection is closed through the null-handle path and then disconnected")
        {
            handler.CloseConnForHostEngine(DCGM_CONNECTION_ID_NONE);
            handler.ProcessDisconnect(connectionId);

            THEN("connection attributes are cleared even when no requests were tracked")
            {
                CHECK_FALSE(handler.m_connectionAttributes.contains(connectionId));
                CHECK(handler.m_connectionRequests.empty());
            }
        }
    }
}
