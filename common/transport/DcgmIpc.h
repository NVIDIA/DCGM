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

#include "DcgmProtocol.h"
#include <DcgmThread.h>
#include <ThreadPool.hpp>
#include <atomic>
#include <dcgm_structs.h>
#include <event2/buffer.h>
#include <event2/bufferevent.h>
#include <event2/dns.h>
#include <event2/event.h>
#include <event2/thread.h>
#include <functional>
#include <future>
#include <optional>
#include <unordered_map>
#include <unordered_set>

typedef struct
{
    std::string bindIPAddress; /* IPv4/IPv6 address of the NIC to bind to. "" = all NICs */
    int port;                  /* TCP port to bind to */
} DcgmIpcTcpServerParams_t;

typedef struct
{
    std::string domainSocketPath; /* Path to the domain socket file to listen on */
} DcgmIpcDomainServerParams_t;

typedef enum
{
    DCGM_IPC_STATE_NOT_STARTED = 0,
    DCGM_IPC_STATE_RUNNING, /* Initialized successfully */
    DCGM_IPC_STATE_FAILED,  /* Failed to initialze */
    DCGM_IPC_STATE_STOPPED, /* Shut down */
    DCGM_IPC_STATE_MAX      /* Sentinel value */
} DcgmIpcState_t;

typedef enum
{
    DCGM_IPC_CS_UNKNOWN = 0,
    DCGM_IPC_CS_PENDING = 1, /* Are in progress trying to connect */
    DCGM_IPC_CS_ACTIVE  = 2, /* Connection has been established */
    DCGM_IPC_CS_CLOSED  = 3, /* The connection is closed */
    DCGM_IPC_CS_MAX     = 4  /* Sentinel value */
} DcgmIpcConnectionState_t;

/* Callback function to pass to DcgmIpc::Init that will process any messages received by clients.
   This will be invoked on a separate worker pool */
typedef std::function<void(dcgm_connection_id_t, std::unique_ptr<DcgmMessage>, void *userData)>
    DcgmIpcProcessMessageFunc_f;

/* Callback function to pass to DcgmIpc::Init that will handle when a connection is lost
   This will be invoked on a separate worker pool */
typedef std::function<void(dcgm_connection_id_t, void *userData)> DcgmIpcProcessDisconnectFunc_f;

class DcgmIpcConnection
{
private:
    struct bufferevent *m_bev;
    DcgmIpcConnectionState_t m_connectionState;
    bool m_shouldReadHeader;            /* Should we read the message header next (true) or the message body (false) */
    dcgm_message_header_t m_readHeader; /* Header of the message we are currently reading. This gets updated
                                           by ReadMessages */

public:
    /* Promise used for async connect. Making this public for ease of use as a private class */
    std::promise<dcgmReturn_t> m_connectPromise;

    /* Constructor and destructor */
    DcgmIpcConnection(struct bufferevent *bev,
                      DcgmIpcConnectionState_t connectionState,
                      std::promise<dcgmReturn_t> &&connectPromise);
    ~DcgmIpcConnection();

    dcgmReturn_t SendMessage(std::unique_ptr<DcgmMessage> dcgmMessage);
    void SetConnectionState(DcgmIpcConnectionState_t state);
    dcgmReturn_t ReadMessages(struct bufferevent *bev, std::vector<std::unique_ptr<DcgmMessage>> &messages);
};

class DcgmIpc : public DcgmThread
{
private:
    /* libevent base class instance */
    event_base *m_eventBase;
    evdns_base *m_dnsBase;
    struct event *m_tcpListenEvent;
    struct event *m_domainListenEvent;

    /* Optional parameters for TCP/IP and domain socket listener sockets.
       If these are not set, then don't start a listening server */
    std::optional<DcgmIpcTcpServerParams_t> m_tcpParameters;
    std::optional<DcgmIpcDomainServerParams_t> m_domainParameters;

    /* Worker threads where cllbacks like
       ProcessMessage() and OnClientDisconnect() are called from */
    DcgmNs::ThreadPool m_workersPool;

    /* State of this instance's event base thread, including our server listeners */
    std::atomic<DcgmIpcState_t> m_state = DCGM_IPC_STATE_NOT_STARTED;

    /* Listening socket file descriptors (FDs) */
    int m_tcpListenSocketFd;
    int m_domainListenSocketFd;

    pthread_t m_ipcThreadId; /* ID of the IPC thread */

    /* Callback to call when processing messages from the worker thread */
    DcgmIpcProcessMessageFunc_f m_processMessageFunc;
    void *m_processMessageData;

    /* callback to call when clients disconnect */
    DcgmIpcProcessDisconnectFunc_f m_processDisconnectFunc;
    void *m_processDisconnectData;

    /* This tracks the next connectionId that will be allocated to a client. Use
       GetNextConnectionId() to access this */
    std::atomic<dcgm_connection_id_t> m_connectionId = DCGM_CONNECTION_ID_NONE;

    /* Tracking of connections - These should only be changed from the IPC thread.
       Use the ASSERT_IS_IPC_THREAD macro to verify you are in the IPC thread before
       reading or writing any of the following. */
    std::unordered_map<struct bufferevent *, dcgm_connection_id_t> m_bevToConnectionId;
    std::unordered_map<dcgm_connection_id_t, std::unique_ptr<DcgmIpcConnection>> m_connections;

    /* Start-up promise. gets set by worker thread after init finishes or fails */
    std::promise<dcgmReturn_t> m_initPromise;

public:
    /* How many un-accepted connections to allow at a time. This value has worked
       since the beginning of DCGM */
    static const int DCGM_IPC_CONNECTION_BACKLOG = 6;

    /*************************************************************************/
    explicit DcgmIpc(int numWorkerThreads);
    ~DcgmIpc();

    /*************************************************************************/
    /* Inherited from DcgmThread() */
    void OnStop() override;
    void run() override;

    /*************************************************************************/
    /* Soft constructor. Returns DCGM_ST_OK on success */
    dcgmReturn_t Init(std::optional<DcgmIpcTcpServerParams_t> tcpParameters,
                      std::optional<DcgmIpcDomainServerParams_t> domainParameters,
                      DcgmIpcProcessMessageFunc_f processMessageFunc,
                      void *processMessageData,
                      DcgmIpcProcessDisconnectFunc_f processDisconnectFunc,
                      void *processDisconnectData);

    /*************************************************************************/
    /* Connect to a TCP/IP Host
     *
     * hostname      IN: DNS name or IP address
     * port          IN: TCP port to connect to
     * connectionId OUT: Connection ID that was allocated for this
     * timeoutMs     IN: How long to wait for this connection to establish in ms
     *
     * Returns: DCGM_ST_OK if the request was successful.
     *          DCGM_ST_CONNECTION_NOT_VALID if the connection failed
     *
     */
    dcgmReturn_t ConnectTcp(std::string hostname, int port, dcgm_connection_id_t &connectionId, unsigned int timeoutMs);

    /*************************************************************************/
    /* Connect to a domain socket
     *
     * path          IN: Domain socket to connect to
     * connectionId OUT: Connection ID that was allocated for this
     * timeoutMs     IN: How long to wait for this connection to establish in ms
     *
     * Returns: DCGM_ST_OK if the request was successful.
     *          DCGM_ST_CONNECTION_NOT_VALID if the connection failed
     *
     */
    dcgmReturn_t ConnectDomain(std::string path, dcgm_connection_id_t &connectionId, unsigned int timeoutMs);

    /*************************************************************************/
    /* Monitor a socket file descriptor (fd) that has already been opened
     * elsewhere. This could be from a socketpair() or even an existing domain
     * or TCP/IP fd.
     *
     * fd            IN: Open file descriptor to monitor
     * connectionId OUT: Connection ID that was allocated for this
     *
     * Returns: DCGM_ST_OK if the request was successful.
     *          Any nonzero DCGM_ST_? error code on failure.
     *
     */
    dcgmReturn_t MonitorSocketFd(int fd, dcgm_connection_id_t &connectionId);

    /*************************************************************************/
    /* Send a message to a given connectionId. Note that this returns once the
     * message has been queued to be sent.
     *
     * connectionId  IN: Connection to send to
     * message       IN: Message to send to the connection
     * waitForSend   IN: Should we wait for the send to complete. This allows
     *                   us to detect broken connections..etc. Pass false if you
     *                   don't care if connectionId is even valid anymore
     *
     * Returns: DCGM_ST_OK if the send was successful.
     *          DCGM_ST_? on error
     *
     */
    dcgmReturn_t SendMessage(dcgm_connection_id_t connectionId, std::unique_ptr<DcgmMessage> message, bool waitForSend);

    /*************************************************************************/
    /* Request that a connection be closed
     *
     * connectionId  IN: Connection to close
     *
     * Returns: DCGM_ST_OK if the send was successful.
     *          DCGM_ST_? on error
     *
     */
    dcgmReturn_t CloseConnection(dcgm_connection_id_t connectionId);

private:
    /*************************************************************************/
    /* Helpers to start listening sockets */
    dcgmReturn_t InitTCPListenerSocket();
    dcgmReturn_t InitUnixListenerSocket();

    /*************************************************************************/
    dcgm_connection_id_t GetNextConnectionId();

    /*************************************************************************/
    /* Track and untrack connections */
    dcgmReturn_t AddConnection(struct bufferevent *bev,
                               dcgm_connection_id_t connectionId,
                               DcgmIpcConnectionState_t initialConnState,
                               std::promise<dcgmReturn_t> connectPromise);
    dcgmReturn_t RemoveConnectionByBev(struct bufferevent *bev);
    dcgmReturn_t RemoveConnectionById(dcgm_connection_id_t connectionId);
    dcgmReturn_t SetConnectionState(dcgm_connection_id_t connectionId, DcgmIpcConnectionState_t state);

    /*****************************************************************************/
    class DcgmIpcConnectTcp
    {
    public:
        DcgmIpc *m_ipc;                       /* Instance of DcgmIpc this is associated with. Not owned here */
        std::string m_hostname;               /* Hostname to connect to */
        int m_port;                           /* Port to connect to */
        dcgm_connection_id_t m_connectionId;  /* Connection ID that was assigned to this
                                             pending connect */
        std::promise<dcgmReturn_t> m_promise; /* Promise used to return if we connected or not */

        DcgmIpcConnectTcp(DcgmIpc *ipc, std::string hostname, int port, dcgm_connection_id_t connectionId)
            : m_ipc(ipc)
            , m_hostname(hostname)
            , m_port(port)
            , m_connectionId(connectionId)
        {}
    };

    static void ConnectTcpAsyncImplCB(evutil_socket_t, short, void *data);
    void ConnectTcpAsyncImpl(DcgmIpcConnectTcp &tcpConnect);

    /*****************************************************************************/
    class DcgmIpcConnectDomain
    {
    public:
        DcgmIpc *m_ipc;                       /* Instance of DcgmIpc this is associated with. Not owned here */
        std::string m_path;                   /* Path to the unix socket to open */
        dcgm_connection_id_t m_connectionId;  /* Connection ID that was assigned to this
                                             pending connect */
        std::promise<dcgmReturn_t> m_promise; /* Promise used to return if we connected or not */

        DcgmIpcConnectDomain(DcgmIpc *ipc, std::string path, dcgm_connection_id_t connectionId)
            : m_ipc(ipc)
            , m_path(path)
            , m_connectionId(connectionId)
        {}
    };

    static void ConnectDomainAsyncImplCB(evutil_socket_t, short, void *data);
    void ConnectDomainAsyncImpl(DcgmIpcConnectDomain &domainConnect);

    /*****************************************************************************/
    struct DcgmIpcMonitorSocketFd
    {
    public:
        DcgmIpc *m_ipc;                       /* Instance of DcgmIpc this is associated with. Not owned here */
        int m_fd;                             /* File descriptor to monitor */
        dcgm_connection_id_t m_connectionId;  /* Connection ID that was assigned to this
                                             pending connect */
        std::promise<dcgmReturn_t> m_promise; /* Promise used to return if we connected or not */

        DcgmIpcMonitorSocketFd(DcgmIpc *ipc, int fd, dcgm_connection_id_t connectionId)
            : m_ipc(ipc)
            , m_fd(fd)
            , m_connectionId(connectionId)
        {}
    };

    static void MonitorSocketFdAsyncImplCB(evutil_socket_t, short, void *data);
    void MonitorSocketFdAsyncImpl(DcgmIpcMonitorSocketFd &monitorFd);

    /*****************************************************************************/
    class DcgmIpcSendMessage
    {
    public:
        DcgmIpc *m_ipc;                         /* Instance of DcgmIpc this is associated with. Not owned here */
        dcgm_connection_id_t m_connectionId;    /* Connection to send the message to */
        std::unique_ptr<DcgmMessage> m_message; /* Message to send */
        std::promise<dcgmReturn_t> m_promise;   /* Promise used to return if we connected or not */

        DcgmIpcSendMessage(DcgmIpc *ipc, dcgm_connection_id_t connectionId, std::unique_ptr<DcgmMessage> message)
            : m_ipc(ipc)
            , m_connectionId(connectionId)
            , m_message(std::move(message))
        {}
    };

    static void SendMessageImplCB(evutil_socket_t, short, void *data);
    void SendMessageImpl(DcgmIpcSendMessage &sendMessage);

    /*****************************************************************************/
    class DcgmIpcCloseConnection
    {
    public:
        DcgmIpc *m_ipc;                      /* Instance of DcgmIpc this is associated with. Not owned here */
        dcgm_connection_id_t m_connectionId; /* Connection to close */

        DcgmIpcCloseConnection(DcgmIpc *ipc, dcgm_connection_id_t connectionId)
            : m_ipc(ipc)
            , m_connectionId(connectionId)
        {}
    };

    static void CloseConnectionImplCB(evutil_socket_t, short, void *data);
    void CloseConnectionImpl(DcgmIpcCloseConnection &closeConnection);

    /*************************************************************************/
    /* Libevent eventCB. Called on connect/disconnect */
    static void StaticEventCB(struct bufferevent *bev, short events, void *ptr);
    void EventCB(struct bufferevent *bev, short events);

    /*************************************************************************/
    /* Libevent readCB. Called when data is read from a socket */
    static void StaticReadCB(struct bufferevent *bev, void *ptr);
    void ReadCB(bufferevent *bev);

    /*************************************************************************/
    /* Libevent acceptCB. Called when a new connection is ready to accept */
    static void StaticOnAccept(int fd, short /*ev*/, void *userData);
    void OnAccept(int fd);

    /*************************************************************************/
    /* Helper methods for converting a bev or connectionId to a pointer to a DcgmIpcConnection object
       These are only safe to call from the IPC main thread */
    dcgm_connection_id_t BevToConnectionId(struct bufferevent *bev);
    DcgmIpcConnection *ConnectionIdToPtr(dcgm_connection_id_t connectionId);

    /*************************************************************************/
    /* Function to call to process a message in our worker pool. We queue this outside
       of the IPC thread to avoid deadlocks and keep sockets responsive */
    typedef struct
    {
        dcgm_connection_id_t connectionId;          /* Connection this message came from */
        DcgmMessage *dcgmMessage;                   /* Message to process */
        DcgmIpcProcessMessageFunc_f processMessage; /* Function to call with the message */
        void *userData;                             /* User context pointer that was passed to Init() */
    } DcgmIpcProcessMessage_t;

    static void ProcessMessageInPool(DcgmIpcProcessMessage_t &processMe);

    /*************************************************************************/
    /* Function to call to process a client disconnect in our worker pool. We queue this outside
       of the IPC thread to avoid deadlocks and keep sockets responsive */
    typedef struct
    {
        dcgm_connection_id_t connectionId;                /* Connection this message came from */
        DcgmIpcProcessDisconnectFunc_f processDisconnect; /* Function to call */
        void *userData;                                   /* User context pointer that was passed to Init() */
    } DcgmIpcProcessDisconnect_t;

    static void ProcessDisconnectInPool(DcgmIpcProcessDisconnect_t &processMe);

    /*************************************************************************/
    /* Helper method to wait for a connection future for a given timeout */
    dcgmReturn_t WaitForConnectHelper(dcgm_connection_id_t connectionId,
                                      std::future<dcgmReturn_t> &fut,
                                      unsigned int timeoutMs);

    /*************************************************************************/
};
