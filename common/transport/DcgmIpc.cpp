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


#include "DcgmIpc.h"
#include <DcgmLogging.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <optional>
#include <sys/socket.h>
#include <sys/un.h>

/*****************************************************************************/
/* The previous IPC library had serious threading issues. Adding this macro
   to validate that we're indeed in the correct thread */
#define ASSERT_IS_IPC_THREAD assert(pthread_equal(pthread_self(), m_ipcThreadId))

/*****************************************************************************/
DcgmIpc::DcgmIpc(int numWorkerThreads)
    : DcgmThread(false, "dcgm_ipc")
    , m_workersPool(numWorkerThreads)
{
    m_tcpParameters         = std::nullopt;
    m_domainParameters      = std::nullopt;
    m_tcpListenSocketFd     = -1;
    m_domainListenSocketFd  = -1;
    m_ipcThreadId           = 0; /* Any valid value is nonzero */
    m_eventBase             = nullptr;
    m_dnsBase               = nullptr;
    m_processDisconnectData = nullptr;
    m_tcpListenEvent        = nullptr;
    m_domainListenEvent     = nullptr;
    m_processMessageData    = nullptr;
}

/*****************************************************************************/
static void DcgmIpcEventLogCB(int severity, const char *msg)
{
    switch (severity)
    {
        case EVENT_LOG_DEBUG:
            DCGM_LOG_DEBUG << "libevent: " << msg;
            break;
        case EVENT_LOG_MSG:
            DCGM_LOG_DEBUG << "libevent: " << msg;
            break;
        case EVENT_LOG_WARN:
            DCGM_LOG_WARNING << "libevent: " << msg;
            break;
        default:
        case EVENT_LOG_ERR:
            DCGM_LOG_ERROR << "libevent: " << msg;
            break;
    }
}

/*****************************************************************************/
dcgmReturn_t DcgmIpc::Init(std::optional<DcgmIpcTcpServerParams_t> tcpParameters,
                           std::optional<DcgmIpcDomainServerParams_t> domainParameters,
                           DcgmIpcProcessMessageFunc_f processMessageFunc,
                           void *processMessageData,
                           DcgmIpcProcessDisconnectFunc_f processDisconnectFunc,
                           void *processDisconnectData)
{
    (void)evthread_use_pthreads();

    /* Enable libevent logging if we're at debug or higher */
    IF_LOG_(BASE_LOGGER, plog::verbose)
    {
        event_set_log_callback(DcgmIpcEventLogCB);
        event_enable_debug_logging(EVENT_DBG_ALL);
    }

    m_eventBase = event_base_new();
    if (m_eventBase == nullptr)
    {
        DCGM_LOG_ERROR << "Failed to open event base";
        return DCGM_ST_GENERIC_ERROR;
    }

#if 1
    m_dnsBase = nullptr; /* Libevent async DNS doesn't work for some reason. The old DNS
                            resolution was sync'd, and the caller
                            is blocked on our future anyway. */
#else
    m_dnsBase = evdns_base_new(m_eventBase, 0);
    if (m_dnsBase == nullptr)
    {
        DCGM_LOG_ERROR << "Failed to open DNS event base";
        return DCGM_ST_GENERIC_ERROR;
    }
#endif

    m_tcpParameters    = tcpParameters;
    m_domainParameters = domainParameters;

    m_processMessageFunc = processMessageFunc;
    m_processMessageData = processMessageData;

    m_processDisconnectFunc = processDisconnectFunc;
    m_processDisconnectData = processDisconnectData;

    m_initPromise = {};

    auto initFuture = m_initPromise.get_future();

    int st = Start();
    if (st != 0)
    {
        DCGM_LOG_ERROR << "Start() returned " << st;
        return DCGM_ST_GENERIC_ERROR;
    }

    /* This has the desired side effect of waiting for start-up */
    dcgmReturn_t dcgmReturn = initFuture.get();
    if (dcgmReturn != DCGM_ST_OK)
    {
        DCGM_LOG_ERROR << "initFuture returned " << errorString(dcgmReturn);
    }

    return dcgmReturn;
}

/*****************************************************************************/
DcgmIpc::~DcgmIpc()
{
    int st = StopAndWait(60000);
    if (st)
    {
        DCGM_LOG_ERROR << "Killing DcgmIpc thread that is still running.";
        Kill();
    }

    if (m_tcpListenEvent != nullptr)
    {
        event_free(m_tcpListenEvent);
    }

    if (m_domainListenEvent != nullptr)
    {
        event_free(m_domainListenEvent);
    }

    /* Free mpBase after any libevent workers are gone */
    if (m_dnsBase)
    {
        evdns_base_free(m_dnsBase, 1);
        m_dnsBase = nullptr;
    }
    if (m_eventBase)
    {
        event_base_free(m_eventBase);
        m_eventBase = nullptr;
    }

    libevent_global_shutdown();
}

/*****************************************************************************/
void DcgmIpc::run()
{
    dcgmReturn_t dcgmReturn;

    m_ipcThreadId = pthread_self();

    ASSERT_IS_IPC_THREAD; /* Make sure the macro works */

    dcgmReturn = InitTCPListenerSocket();
    if (dcgmReturn != DCGM_ST_OK)
    {
        DCGM_LOG_ERROR << "InitTCPListenerSocket() returned " << errorString(dcgmReturn);
        m_state = DCGM_IPC_STATE_FAILED;
        m_initPromise.set_value(dcgmReturn);
        return;
    }

    dcgmReturn = InitUnixListenerSocket();
    if (dcgmReturn != DCGM_ST_OK)
    {
        DCGM_LOG_ERROR << "InitUnixListenerSocket() returned " << errorString(dcgmReturn);
        m_state = DCGM_IPC_STATE_FAILED;
        m_initPromise.set_value(dcgmReturn);
        return;
    }

    m_state = DCGM_IPC_STATE_RUNNING;
    m_initPromise.set_value(DCGM_ST_OK);

    DCGM_LOG_DEBUG << "starting event_base_loop()";

    /* Run until we're told to stop */
    event_base_loop(m_eventBase, EVLOOP_NO_EXIT_ON_EMPTY);

    DCGM_LOG_DEBUG << "event_base_loop() ended. Closing connections.";

    /* Clear out our structures from this thread since this thread owns them */
    m_bevToConnectionId.clear();
    m_connections.clear();

    m_state = DCGM_IPC_STATE_STOPPED;

    DCGM_LOG_DEBUG << "dcgmipc thread exiting.";
}

/*****************************************************************************/
void DcgmIpc::OnStop()
{
    /* This method is called by the caller of DcgmIpc::StopAndWait(). We must not
       modify any of the data structures that are owned by the IPC thread. Those
       will be cleaned up by that thread once event_base_loopexit() causes it to
       break out of event_base_loop().
       There should be a really good reason why the caller of DcgmIpc::StopAndWait()
       isn't the main thread from the destructor of the owner of this class instance. */

    DCGM_LOG_DEBUG << "OnStop()";

    /* Stop the workers before we stop the event loop. The event loop can leak
       callbacks, but the worker pool queue won't leak */
    m_workersPool.StopAndWait();

    if (m_eventBase)
    {
        DCGM_LOG_DEBUG << "Requesting loop exit";
        event_base_loopexit(m_eventBase, nullptr);
    }
}

/*****************************************************************************/
static int SetNonBlocking(int fd)
{
    int flags;

    /* Get the current file descriptor flags. This will only fail if fd is invalid */
    flags = fcntl(fd, F_GETFL);
    if (flags < 0)
    {
        DCGM_LOG_ERROR << "fcntl failed for fd " << fd;
        return -1;
    }

    /* Add the Non-blocking flag to our FD */
    flags |= O_NONBLOCK;
    if (fcntl(fd, F_SETFL, flags) < 0)
    {
        DCGM_LOG_ERROR << "fcntl failed for fd " << fd << " flags 0x" << std::hex << flags;
        return -1;
    }
    return 0;
}

/*****************************************************************************/
dcgmReturn_t DcgmIpc::InitTCPListenerSocket()
{
    struct sockaddr_in listenAddr;
    int reuseAddrOn;

    ASSERT_IS_IPC_THREAD;

    if (!m_tcpParameters.has_value())
    {
        DCGM_LOG_DEBUG << "m_tcpParameters was not set.";
        return DCGM_ST_OK;
    }

    /* Create our listening socket. */
    m_tcpListenSocketFd = socket(AF_INET, SOCK_STREAM, 0);
    if (m_tcpListenSocketFd < 0)
    {
        DCGM_LOG_ERROR << "ERROR: socket creation failed";
        return DCGM_ST_GENERIC_ERROR;
    }

    reuseAddrOn = 1;
    if (setsockopt(m_tcpListenSocketFd, SOL_SOCKET, SO_REUSEADDR, &reuseAddrOn, sizeof(reuseAddrOn)))
    {
        DCGM_LOG_ERROR << "ERROR: setsockopt(SO_REUSEADDR) failed. errno " << errno;
        close(m_tcpListenSocketFd);
        m_tcpListenSocketFd = -1;
        return DCGM_ST_GENERIC_ERROR;
    }

    memset(&listenAddr, 0, sizeof(listenAddr));
    listenAddr.sin_family      = AF_INET;
    listenAddr.sin_addr.s_addr = INADDR_ANY;

    if (m_tcpParameters.value().bindIPAddress.size() > 0)
    {
        /* Convert mSocketPath to a number in network byte order */
        if (!inet_aton(m_tcpParameters.value().bindIPAddress.c_str(), &listenAddr.sin_addr))
        {
            DCGM_LOG_ERROR << "Unable to convert \"" << m_tcpParameters.value().bindIPAddress
                           << "\" to a network address.";
            close(m_tcpListenSocketFd);
            m_tcpListenSocketFd = -1;
            return DCGM_ST_GENERIC_ERROR;
        }
    }

    listenAddr.sin_port = htons(m_tcpParameters.value().port);
    if (bind(m_tcpListenSocketFd, (struct sockaddr *)&listenAddr, sizeof(listenAddr)) < 0)
    {
        DCGM_LOG_ERROR << "bind failed. port " << m_tcpParameters.value().port << ", address "
                       << m_tcpParameters.value().bindIPAddress.c_str() << ", errno " << errno;
        close(m_tcpListenSocketFd);
        m_tcpListenSocketFd = -1;
        return DCGM_ST_GENERIC_ERROR;
    }

    if (listen(m_tcpListenSocketFd, DCGM_IPC_CONNECTION_BACKLOG) < 0)
    {
        DCGM_LOG_ERROR << "TCP listen failed";
        close(m_tcpListenSocketFd);
        m_tcpListenSocketFd = -1;
        return DCGM_ST_IN_USE;
    }

    if (SetNonBlocking(m_tcpListenSocketFd))
    {
        DCGM_LOG_ERROR << "SetNonBlocking failed";
        close(m_tcpListenSocketFd);
        m_tcpListenSocketFd = -1;
        return DCGM_ST_GENERIC_ERROR;
    }

    m_tcpListenEvent = event_new(m_eventBase, m_tcpListenSocketFd, EV_READ | EV_PERSIST, DcgmIpc::StaticOnAccept, this);
    if (m_tcpListenEvent == nullptr)
    {
        DCGM_LOG_ERROR << "event_new() failed for TCP listener";
        close(m_tcpListenSocketFd);
        m_tcpListenSocketFd = -1;
        return DCGM_ST_GENERIC_ERROR;
    }

    if (event_add(m_tcpListenEvent, nullptr))
    {
        DCGM_LOG_ERROR << "event_add() failed for TCP listener";
        close(m_tcpListenSocketFd);
        m_tcpListenSocketFd = -1;
        return DCGM_ST_GENERIC_ERROR;
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmIpc::InitUnixListenerSocket()
{
    struct sockaddr_un listenAddr;
    int reuseAddrOn;

    ASSERT_IS_IPC_THREAD;

    if (!m_domainParameters.has_value())
    {
        DCGM_LOG_DEBUG << "m_domainParameters was not set.";
        return DCGM_ST_OK;
    }

    /* Create our listening socket. */
    m_domainListenSocketFd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (m_domainListenSocketFd < 0)
    {
        DCGM_LOG_ERROR << "socket creation failed";
        return DCGM_ST_GENERIC_ERROR;
    }

    reuseAddrOn = 1;
    if (setsockopt(m_domainListenSocketFd, SOL_SOCKET, SO_REUSEADDR, &reuseAddrOn, sizeof(reuseAddrOn)))
    {
        DCGM_LOG_ERROR << "ERROR: set socket option failed";
        close(m_domainListenSocketFd);
        m_domainListenSocketFd = -1;
        return DCGM_ST_GENERIC_ERROR;
    }

    memset(&listenAddr, 0, sizeof(listenAddr));
    listenAddr.sun_family = AF_UNIX;
    strncpy(listenAddr.sun_path, m_domainParameters.value().domainSocketPath.c_str(), sizeof(listenAddr.sun_path));
    listenAddr.sun_path[sizeof(listenAddr.sun_path) - 1] = '\0';
    unlink(
        m_domainParameters.value().domainSocketPath.c_str()); /* Make sure the path doesn't exist or bind will fail */
    if (bind(m_domainListenSocketFd, (struct sockaddr *)&listenAddr, sizeof(listenAddr)) < 0)
    {
        DCGM_LOG_ERROR << "ERROR: domain socket bind failed for " << m_domainParameters.value().domainSocketPath;
        close(m_domainListenSocketFd);
        m_domainListenSocketFd = -1;
        return DCGM_ST_GENERIC_ERROR;
    }

    if (listen(m_domainListenSocketFd, DCGM_IPC_CONNECTION_BACKLOG) < 0)
    {
        DCGM_LOG_ERROR << "Domain socket listen failed";
        close(m_domainListenSocketFd);
        m_domainListenSocketFd = -1;
        return DCGM_ST_IN_USE;
    }

    if (SetNonBlocking(m_domainListenSocketFd))
    {
        DCGM_LOG_ERROR << "SetNonBlocking failed";
        close(m_domainListenSocketFd);
        m_domainListenSocketFd = -1;
        return DCGM_ST_GENERIC_ERROR;
    }

    m_domainListenEvent
        = event_new(m_eventBase, m_domainListenSocketFd, EV_READ | EV_PERSIST, DcgmIpc::StaticOnAccept, this);
    if (m_domainListenEvent == nullptr)
    {
        DCGM_LOG_ERROR << "event_new() failed for domain listener";
        close(m_domainListenSocketFd);
        m_domainListenSocketFd = -1;
        return DCGM_ST_GENERIC_ERROR;
    }

    if (event_add(m_domainListenEvent, nullptr))
    {
        DCGM_LOG_ERROR << "event_add() failed for domain listener";
        close(m_domainListenSocketFd);
        m_domainListenSocketFd = -1;
        return DCGM_ST_GENERIC_ERROR;
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmIpc::AddConnection(struct bufferevent *bev,
                                    dcgm_connection_id_t connectionId,
                                    DcgmIpcConnectionState_t initialConnState,
                                    std::promise<dcgmReturn_t> connectPromise)
{
    ASSERT_IS_IPC_THREAD;

    if (bev == nullptr || connectionId == DCGM_CONNECTION_ID_NONE)
    {
        DCGM_LOG_ERROR << "Bad parameter";
        return DCGM_ST_BADPARAM;
    }

    m_bevToConnectionId[bev]    = connectionId;
    m_connections[connectionId] = std::make_unique<DcgmIpcConnection>(bev, initialConnState, std::move(connectPromise));

    DCGM_LOG_DEBUG << "Added connectionId " << connectionId << " bev " << bev << " ics " << initialConnState;
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmIpc::RemoveConnectionById(dcgm_connection_id_t connectionId)
{
    ASSERT_IS_IPC_THREAD;

    auto connectionIt = m_connections.find(connectionId);
    if (connectionIt == m_connections.end())
    {
        DCGM_LOG_DEBUG << "connectionId " << connectionId << " did not exist.";
        return DCGM_ST_NO_DATA;
    }

    /* Now look up the bev (linear search) */
    struct bufferevent *bev = nullptr;
    for (auto &bevIt : m_bevToConnectionId)
    {
        if (bevIt.second == connectionId)
        {
            bev = bevIt.first;
            break;
        }
    }

    if (bev == nullptr)
    {
        DCGM_LOG_ERROR << "bev -> connectionId did not exist but connectionId -> object did for connectionId "
                       << connectionId;
        m_connections.erase(connectionIt);
        return DCGM_ST_GENERIC_ERROR;
    }

    /* Use the helper API since it calls callbacks */
    return RemoveConnectionByBev(bev);
}

/*****************************************************************************/
dcgmReturn_t DcgmIpc::RemoveConnectionByBev(struct bufferevent *bev)
{
    ASSERT_IS_IPC_THREAD;

    if (bev == nullptr)
    {
        DCGM_LOG_ERROR << "Bad parameter";
        return DCGM_ST_BADPARAM;
    }

    auto conIdIt = m_bevToConnectionId.find(bev);
    if (conIdIt == m_bevToConnectionId.end())
    {
        DCGM_LOG_DEBUG << "bev " << bev << " was already gone.";
        return DCGM_ST_OK;
    }

    /* Save the connectionId for our callback later */
    dcgm_connection_id_t connectionId = conIdIt->second;

    auto connectionIt = m_connections.find(connectionId);
    if (connectionIt == m_connections.end())
    {
        DCGM_LOG_DEBUG << "m_connections entry missing for connectionId " << connectionId << " bev " << bev;
        m_bevToConnectionId.erase(conIdIt);
        return DCGM_ST_OK;
    }

    /* Found both. Erase them */
    DCGM_LOG_DEBUG << "Removing bev " << bev << ", connectionId " << connectionId;
    m_bevToConnectionId.erase(conIdIt);
    m_connections.erase(connectionIt);

    /* Notify our parent that we got a disconnect */
    DcgmIpcProcessDisconnect_t pd {};
    pd.connectionId      = connectionId;
    pd.processDisconnect = m_processDisconnectFunc;
    pd.userData          = m_processDisconnectData;

    m_workersPool.Enqueue([pd]() mutable { DcgmIpc::ProcessDisconnectInPool(pd); });
    return DCGM_ST_OK;
}

/*****************************************************************************/
void DcgmIpc::ConnectTcpAsyncImpl(DcgmIpcConnectTcp &tcpConnect)
{
    ASSERT_IS_IPC_THREAD;

    /* TCP/IP */
    DCGM_LOG_DEBUG << "Client trying to connect to " << tcpConnect.m_hostname << ":" << tcpConnect.m_port;

    struct bufferevent *bev = bufferevent_socket_new(m_eventBase, -1, BEV_OPT_CLOSE_ON_FREE);
    if (bev == nullptr)
    {
        DCGM_LOG_ERROR << "Failed to create socket";
        tcpConnect.m_promise.set_value(DCGM_ST_GENERIC_ERROR);
        return;
    }

    /* Add a tracked connection and remove our pending status */
    dcgmReturn_t dcgmReturn
        = AddConnection(bev, tcpConnect.m_connectionId, DCGM_IPC_CS_PENDING, std::move(tcpConnect.m_promise));
    if (dcgmReturn != DCGM_ST_OK)
    {
        DCGM_LOG_ERROR << "Failed to AddConnection";
        bufferevent_free(bev);
        tcpConnect.m_promise.set_value(DCGM_ST_GENERIC_ERROR);
        return;
    }

    /* Track our event before callbacks could be invoked */
    bufferevent_setcb(bev, DcgmIpc::StaticReadCB, NULL, DcgmIpc::StaticEventCB, this);
    bufferevent_enable(bev, EV_READ | EV_WRITE);

    int ret = bufferevent_socket_connect_hostname(
        bev, m_dnsBase, AF_INET, tcpConnect.m_hostname.c_str(), tcpConnect.m_port);
    if (0 != ret)
    {
        RemoveConnectionByBev(bev);
        DCGM_LOG_ERROR << "Failed to connect to Host engine running at IP " << tcpConnect.m_hostname << " port "
                       << tcpConnect.m_port;
        return;
    }

    DCGM_LOG_DEBUG << "connectionId " << tcpConnect.m_connectionId << " connection in progress to "
                   << tcpConnect.m_hostname << " port " << tcpConnect.m_port;
    return;
}

/*****************************************************************************/
void DcgmIpc::ConnectTcpAsyncImplCB(evutil_socket_t, short, void *data)
{
    std::unique_ptr<DcgmIpcConnectTcp> tcpConnect((DcgmIpcConnectTcp *)data);

    tcpConnect->m_ipc->ConnectTcpAsyncImpl(*tcpConnect);
}

/*****************************************************************************/
dcgmReturn_t DcgmIpc::ConnectTcp(std::string hostname,
                                 int port,
                                 dcgm_connection_id_t &connectionId,
                                 unsigned int timeoutMs)
{
    connectionId = GetNextConnectionId();

    /* Using new here because we're transferring it through a C callback. The callback will
       assign this to a unique_ptr and then free it automatically */
    DcgmIpcConnectTcp *connectTcp = new DcgmIpcConnectTcp(this, hostname, port, connectionId);

    std::future<dcgmReturn_t> connectReturn = connectTcp->m_promise.get_future();

    int st = event_base_once(m_eventBase, -1, EV_TIMEOUT, DcgmIpc::ConnectTcpAsyncImplCB, connectTcp, 0);
    if (st)
    {
        DCGM_LOG_ERROR << "Got error " << st << " from event_base_once";
        return DCGM_ST_GENERIC_ERROR;
    }

    return WaitForConnectHelper(connectionId, connectReturn, timeoutMs);
}

/*****************************************************************************/
dcgmReturn_t DcgmIpc::WaitForConnectHelper(dcgm_connection_id_t connectionId,
                                           std::future<dcgmReturn_t> &fut,
                                           unsigned int timeoutMs)
{
    try
    {
    auto futStatus = fut.wait_for(std::chrono::milliseconds(timeoutMs));
    if (futStatus != std::future_status::ready)
    {
        DCGM_LOG_ERROR << "connectionId " << connectionId << " timed out after " << timeoutMs << " ms.";
        /* Close the connection, which will handle both:
           1. If the connection was established right after the timeout
           2. If the connection was still pending and just blocked on libevent's ~2 minute timeout */
        CloseConnection(connectionId);
        return DCGM_ST_CONNECTION_NOT_VALID;
    }

    return fut.get();
}
    catch (const std::future_error &e)
    {
        DCGM_LOG_ERROR << "Caught future error for connectionId " << connectionId << ", what: " << e.what();
        return DCGM_ST_GENERIC_ERROR;
    }
}

/*****************************************************************************/
void DcgmIpc::ConnectDomainAsyncImpl(DcgmIpcConnectDomain &domainConnect)
{
    ASSERT_IS_IPC_THREAD;

    /* Domain socket */
    DCGM_LOG_DEBUG << "Client trying to connect to " << domainConnect.m_path;

    struct bufferevent *bev = bufferevent_socket_new(m_eventBase, -1, BEV_OPT_CLOSE_ON_FREE);
    if (bev == nullptr)
    {
        DCGM_LOG_ERROR << "Failed to create socket";
        domainConnect.m_promise.set_value(DCGM_ST_GENERIC_ERROR);
        return;
    }

    /* Add a tracked connection and remove our pending status */
    dcgmReturn_t dcgmReturn
        = AddConnection(bev, domainConnect.m_connectionId, DCGM_IPC_CS_PENDING, std::move(domainConnect.m_promise));
    if (dcgmReturn != DCGM_ST_OK)
    {
        DCGM_LOG_ERROR << "Failed to AddConnection";
        bufferevent_free(bev);
        /* Not setting promise here since it's owned by AddConnection or was destructed there */
        return;
    }

    /* Track our event before callbacks could be invoked */
    bufferevent_setcb(bev, DcgmIpc::StaticReadCB, NULL, DcgmIpc::StaticEventCB, this);
    bufferevent_enable(bev, EV_READ | EV_WRITE);

    struct sockaddr_un unixDomainAddr; /* Unix domain socket address used for specifying connection details */
    memset(&unixDomainAddr, 0, sizeof(unixDomainAddr));
    unixDomainAddr.sun_family = AF_UNIX;
    strncpy(unixDomainAddr.sun_path, domainConnect.m_path.c_str(), sizeof(unixDomainAddr.sun_path) - 1);

    int ret = bufferevent_socket_connect(bev, (struct sockaddr *)&unixDomainAddr, sizeof(unixDomainAddr));
    if (0 != ret)
    {
        RemoveConnectionByBev(bev);
        DCGM_LOG_ERROR << "Failed to connect to Host engine running at IP " << domainConnect.m_path;
        return;
    }

    DCGM_LOG_DEBUG << "connectionId " << domainConnect.m_connectionId << " connection in progress to "
                   << domainConnect.m_path;
    return;
}

/*****************************************************************************/
void DcgmIpc::ConnectDomainAsyncImplCB(evutil_socket_t, short, void *data)
{
    std::unique_ptr<DcgmIpcConnectDomain> domainConnect((DcgmIpcConnectDomain *)data);

    domainConnect->m_ipc->ConnectDomainAsyncImpl(*domainConnect);
}

/*****************************************************************************/
dcgmReturn_t DcgmIpc::ConnectDomain(std::string path, dcgm_connection_id_t &connectionId, unsigned int timeoutMs)
{
    connectionId = GetNextConnectionId();

    /* Using new here because we're transferring it through a C callback. The callback will
       assign this to a unique_ptr and then free it automatically */
    DcgmIpcConnectDomain *connectDomain = new DcgmIpcConnectDomain(this, path, connectionId);

    std::future<dcgmReturn_t> connectReturn = connectDomain->m_promise.get_future();

    int st = event_base_once(m_eventBase, -1, EV_TIMEOUT, DcgmIpc::ConnectDomainAsyncImplCB, connectDomain, 0);
    if (st)
    {
        DCGM_LOG_ERROR << "Got error " << st << " from event_base_once";
        return DCGM_ST_GENERIC_ERROR;
    }

    return WaitForConnectHelper(connectionId, connectReturn, timeoutMs);
}

/*****************************************************************************/
void DcgmIpc::MonitorSocketFdAsyncImpl(DcgmIpcMonitorSocketFd &monitorSocketFd)
{
    ASSERT_IS_IPC_THREAD;

    /* Domain socket */
    DCGM_LOG_DEBUG << "Client trying to monitor socket fd " << monitorSocketFd.m_fd;

    struct bufferevent *bev = bufferevent_socket_new(m_eventBase, monitorSocketFd.m_fd, BEV_OPT_CLOSE_ON_FREE);
    if (bev == nullptr)
    {
        DCGM_LOG_ERROR << "Failed to create bufferevent for fd " << monitorSocketFd.m_fd;
        monitorSocketFd.m_promise.set_value(DCGM_ST_GENERIC_ERROR);
        return;
    }

    /* Track our event before callbacks could be invoked */
    bufferevent_setcb(bev, DcgmIpc::StaticReadCB, NULL, DcgmIpc::StaticEventCB, this);
    bufferevent_enable(bev, EV_READ | EV_WRITE);

    /* Add a tracked connection and remove our pending status */
    dcgmReturn_t dcgmReturn
        = AddConnection(bev, monitorSocketFd.m_connectionId, DCGM_IPC_CS_PENDING, std::move(monitorSocketFd.m_promise));
    if (dcgmReturn != DCGM_ST_OK)
    {
        DCGM_LOG_ERROR << "Failed to AddConnection";
        bufferevent_free(bev);
        monitorSocketFd.m_promise.set_value(DCGM_ST_GENERIC_ERROR);
        return;
    }

    /* Setting this to active will set the future */
    SetConnectionState(monitorSocketFd.m_connectionId, DCGM_IPC_CS_ACTIVE);

    DCGM_LOG_DEBUG << "connectionId " << monitorSocketFd.m_connectionId << " connection to fd " << monitorSocketFd.m_fd
                   << " is now actively monitored.";
    return;
}

/*****************************************************************************/
void DcgmIpc::MonitorSocketFdAsyncImplCB(evutil_socket_t, short, void *data)
{
    std::unique_ptr<DcgmIpcMonitorSocketFd> monitorSocketFd((DcgmIpcMonitorSocketFd *)data);

    monitorSocketFd->m_ipc->MonitorSocketFdAsyncImpl(*monitorSocketFd);
}

/*****************************************************************************/
dcgmReturn_t DcgmIpc::MonitorSocketFd(int fd, dcgm_connection_id_t &connectionId)
{
    if (fd < 0)
    {
        DCGM_LOG_ERROR << "Invalid fd: " << fd;
        return DCGM_ST_BADPARAM;
    }

    if (SetNonBlocking(fd))
    {
        DCGM_LOG_ERROR << "failed to set client socket to non-blocking";
        return DCGM_ST_GENERIC_ERROR;
    }

    connectionId = GetNextConnectionId();

    /* Using new here because we're transferring it through a C callback. The callback will
       assign this to a unique_ptr and then free it automatically */
    auto monitorSocketFd = std::make_unique<DcgmIpcMonitorSocketFd>(this, fd, connectionId);

    std::future<dcgmReturn_t> connectReturn = monitorSocketFd->m_promise.get_future();

    int st
        = event_base_once(m_eventBase, -1, EV_TIMEOUT, DcgmIpc::MonitorSocketFdAsyncImplCB, monitorSocketFd.get(), 0);
    if (st)
    {
        DCGM_LOG_ERROR << "Got error " << st << " from event_base_once. Closing fd " << fd;
        close(fd);
        return DCGM_ST_GENERIC_ERROR;
    }
    else
    {
        /* No longer owned by us on success */
        monitorSocketFd.release();
    }

    /* Passing 30 seconds to wait here just to give a reasonable value that isn't infinity but gives
       us enough time to continue our debugger. We aren't actually connecting so we're giving the
       IPC thread enough time to wake up and set our future. */
    return WaitForConnectHelper(connectionId, connectReturn, 30000);
}

/*****************************************************************************/
dcgm_connection_id_t DcgmIpc::GetNextConnectionId()
{
    dcgm_connection_id_t newId = m_connectionId++;

    /* Don't allocate a connection as id DCGM_CONNECTION_ID_NONE. In practice,
       this will only happen after 2^32 connections */
    if (newId == DCGM_CONNECTION_ID_NONE)
    {
        newId = m_connectionId++;
    }

    return newId;
}

/*****************************************************************************/
void DcgmIpc::EventCB(struct bufferevent *bev, short events)
{
    dcgm_connection_id_t connectionId = BevToConnectionId(bev);
    if (connectionId == DCGM_CONNECTION_ID_NONE)
    {
        DCGM_LOG_ERROR << "Unknown bev " << bev << " got events x" << std::hex << events;
        return;
    }

    DCGM_LOG_DEBUG << "bev " << bev << " got events " << std::hex << events;

    if (events & BEV_EVENT_CONNECTED)
    {
        /* Connected */
        DCGM_LOG_DEBUG << "Got connected event for connectionId " << connectionId << " bev " << bev;
        SetConnectionState(connectionId, DCGM_IPC_CS_ACTIVE);
    }
    else if ((events & BEV_EVENT_ERROR) || (events & BEV_EVENT_EOF))
    {
        DCGM_LOG_DEBUG << "Got connection error for bev " << bev << " connectionId " << connectionId << " events "
                       << std::hex << events;
        RemoveConnectionByBev(bev);
    }
}

/*****************************************************************************/
void DcgmIpc::StaticEventCB(struct bufferevent *bev, short events, void *ptr)
{
    DcgmIpc *dcgmIpc = (DcgmIpc *)ptr;
    dcgmIpc->EventCB(bev, events);
}

/*****************************************************************************/
void DcgmIpc::StaticReadCB(struct bufferevent *bev, void *ptr)
{
    DcgmIpc *dcgmIpc = (DcgmIpc *)ptr;
    dcgmIpc->ReadCB(bev);
}

/*****************************************************************************/
dcgmReturn_t DcgmIpc::SetConnectionState(dcgm_connection_id_t connectionId, DcgmIpcConnectionState_t state)
{
    ASSERT_IS_IPC_THREAD;

    DcgmIpcConnection *connection = ConnectionIdToPtr(connectionId);
    if (connection == nullptr)
    {
        DCGM_LOG_ERROR << "SetConnectionState got unknown connectionId " << connectionId;
        return DCGM_ST_CONNECTION_NOT_VALID;
    }

    connection->SetConnectionState(state);
    return DCGM_ST_OK;
}

/*****************************************************************************/
void DcgmIpc::ProcessMessageInPool(DcgmIpcProcessMessage_t &processMe)
{
    std::unique_ptr<DcgmMessage> dcgmMessage(processMe.dcgmMessage);

    processMe.processMessage(processMe.connectionId, std::move(dcgmMessage), processMe.userData);
}

/*****************************************************************************/
void DcgmIpc::ProcessDisconnectInPool(DcgmIpcProcessDisconnect_t &processMe)
{
    processMe.processDisconnect(processMe.connectionId, processMe.userData);
}

/*****************************************************************************/
DcgmIpcConnection *DcgmIpc::ConnectionIdToPtr(dcgm_connection_id_t connectionId)
{
    auto connectionIt = m_connections.find(connectionId);
    if (connectionIt == m_connections.end())
    {
        DCGM_LOG_DEBUG << "Unknown connectionId " << connectionId;
        return nullptr;
    }
    else
    {
        return connectionIt->second.get();
    }
}

/*****************************************************************************/
void DcgmIpc::ReadCB(bufferevent *bev)
{
    dcgm_connection_id_t connectionId = BevToConnectionId(bev);
    if (connectionId == DCGM_CONNECTION_ID_NONE)
    {
        DCGM_LOG_ERROR << "Unknown bev " << bev << " got ReadCB";
        return;
    }

    DcgmIpcConnection *connection = ConnectionIdToPtr(connectionId);
    if (connection == nullptr)
    {
        DCGM_LOG_ERROR << "Unknown connectionId " << connectionId << " got ReadCB for bev " << bev;
        return;
    }

    std::vector<std::unique_ptr<DcgmMessage>> messages;

    dcgmReturn_t dcgmReturn = connection->ReadMessages(bev, messages);
    if (dcgmReturn == DCGM_ST_NO_DATA)
    {
        DCGM_LOG_VERBOSE << "connectionId " << connectionId << " got no complete messages.";
        /* No complete messages were read */
        return;
    }
    else if (dcgmReturn != DCGM_ST_OK)
    {
        DCGM_LOG_ERROR << "Got error " << errorString(dcgmReturn) << " from ReadMessages";
        /* Assume the connection is broken */
        RemoveConnectionByBev(bev);
        return;
    }

    DcgmIpcProcessMessage_t processMessage {};

    processMessage.connectionId   = connectionId;
    processMessage.processMessage = m_processMessageFunc;
    processMessage.userData       = m_processMessageData;

    for (auto &&dcgmMessage : messages)
    {
        processMessage.dcgmMessage = dcgmMessage.release();

        m_workersPool.Enqueue([processMessage]() mutable { DcgmIpc::ProcessMessageInPool(processMessage); });
    }
}

/*****************************************************************************/
dcgm_connection_id_t DcgmIpc::BevToConnectionId(struct bufferevent *bev)
{
    auto it = m_bevToConnectionId.find(bev);
    if (it == m_bevToConnectionId.end())
    {
        return DCGM_CONNECTION_ID_NONE;
    }
    else
    {
        return it->second;
    }
}

/*****************************************************************************/
DcgmIpcConnection::DcgmIpcConnection(struct bufferevent *bev,
                                     DcgmIpcConnectionState_t connectionState,
                                     std::promise<dcgmReturn_t> &&connectPromise)
    : m_bev(bev)
    , m_connectionState(connectionState)
    , m_shouldReadHeader(true)
    , m_readHeader({})
    , m_connectPromise(std::move(connectPromise))
{
    DCGM_LOG_DEBUG << "DcgmIpcConnection constructor for bev " << m_bev;
}

/*****************************************************************************/
DcgmIpcConnection::~DcgmIpcConnection()
{
    DCGM_LOG_DEBUG << "DcgmIpcConnection destructor for bev " << m_bev;
    /* Set this connection to closed, possibly triggering the m_connectPromise-linked future */
    SetConnectionState(DCGM_IPC_CS_CLOSED);

    if (m_bev != nullptr)
    {
        DCGM_LOG_DEBUG << "bufferevent_free " << m_bev;
        bufferevent_free(m_bev);
        m_bev = nullptr;
    }
}

/*****************************************************************************/
void DcgmIpcConnection::SetConnectionState(DcgmIpcConnectionState_t state)
{
    DcgmIpcConnectionState_t oldState = m_connectionState;
    m_connectionState                 = state;

    /* Signal the connection future if we were pending */
    if (oldState == DCGM_IPC_CS_PENDING)
    {
        if (state == DCGM_IPC_CS_ACTIVE)
        {
            m_connectPromise.set_value(DCGM_ST_OK);
        }
        else
        {
            m_connectPromise.set_value(DCGM_ST_CONNECTION_NOT_VALID);
        }
    }

    DCGM_LOG_DEBUG << "SetConnectionState bev" << m_bev << " " << oldState << " -> " << state;
}

/*****************************************************************************/
dcgmReturn_t DcgmIpcConnection::ReadMessages(struct bufferevent *bev,
                                             std::vector<std::unique_ptr<DcgmMessage>> &messages)
{
    dcgmReturn_t retSt = DCGM_ST_NO_DATA;
    size_t numBytes;

    assert(bev == m_bev);

    struct evbuffer *inputEvBuf = bufferevent_get_input(bev);

    size_t inputBufRemaining = evbuffer_get_length(inputEvBuf);

    while (inputBufRemaining > 0)
    {
        if (m_shouldReadHeader)
        {
            /* Do we have an entire header yet? If not, return without action */
            if (inputBufRemaining < sizeof(m_readHeader))
            {
                DCGM_LOG_DEBUG << "Have " << inputBufRemaining << "/" << sizeof(m_readHeader)
                               << " bytes of msg header.";
                return retSt;
            }

            /* Read the message header first and get the message type and the
                size of message to be received */
            numBytes = bufferevent_read(bev, &m_readHeader, sizeof(m_readHeader));
            if (numBytes != sizeof(m_readHeader))
            {
                DCGM_LOG_ERROR << "Got " << numBytes << " instead of header of size " << sizeof(m_readHeader);
                return DCGM_ST_GENERIC_ERROR;
            }

            /* Adjust the Buf length available to be read */
            inputBufRemaining -= numBytes;

            if (m_readHeader.length < 0 || m_readHeader.length > DCGM_PROTO_MAX_MESSAGE_SIZE)
            {
                DCGM_LOG_ERROR << "Got bad message size " << m_readHeader.length << ". Closing connection.";
                return DCGM_ST_CONNECTION_NOT_VALID;
            }

            if (m_readHeader.msgId != DCGM_PROTO_MAGIC)
            {
                DCGM_LOG_ERROR << "Unexpected DCGM Proto ID " << std::hex << m_readHeader.msgId;
                return DCGM_ST_CONNECTION_NOT_VALID;
            }

            m_shouldReadHeader = false;

            /* Fall through to read the message body */
        }

        /* Read the message body. Do we have enough input data? */
        if (inputBufRemaining < m_readHeader.length)
        {
            DCGM_LOG_DEBUG << "Have " << inputBufRemaining << "/" << m_readHeader.length << " bytes of msgType 0x"
                           << std::hex << m_readHeader.msgType;
            return retSt;
        }

        /* Allocate a new DCGM Message */
        std::unique_ptr<DcgmMessage> dcgmMessage = std::make_unique<DcgmMessage>(&m_readHeader);
        dcgmMessage->SetRequestId(m_readHeader.requestId);

        auto msgBytes = dcgmMessage->GetMsgBytesPtr();

        /* Read Length of message. Make sure the complete message is received */
        msgBytes->resize(m_readHeader.length);
        numBytes = bufferevent_read(bev, msgBytes->data(), m_readHeader.length);
        if (numBytes != m_readHeader.length)
        {
            DCGM_LOG_ERROR << "Unexpected numBytes " << numBytes << " != " << m_readHeader.length;
            /* Try to read a header after this. We're really in a bad state now, so we'll probably
            end up disconnecting */
            m_shouldReadHeader = true;
            return DCGM_ST_GENERIC_ERROR;
        }

        inputBufRemaining -= numBytes;

        /* We read an entire message. We should read a header next */
        m_shouldReadHeader = true;

        messages.push_back(std::move(dcgmMessage));
        retSt = DCGM_ST_OK; /* We read at least one complete message */
    }

    return retSt;
}

/*****************************************************************************/
void DcgmIpc::OnAccept(int listenerFd)
{
    ASSERT_IS_IPC_THREAD;

    int clientFd = -1; /* FD of the new client connection we're accepting */
    struct sockaddr_in clientAddr;
    socklen_t clientLen = sizeof(clientAddr);

    clientFd = accept(listenerFd, (struct sockaddr *)&clientAddr, &clientLen);
    if (clientFd < 0)
    {
        DCGM_LOG_ERROR << "accept failed";
        return;
    }

    /* Set the client socket to non-blocking mode. */
    if (SetNonBlocking(clientFd))
    {
        DCGM_LOG_ERROR << "failed to set client socket to non-blocking";
        close(clientFd);
        return;
    }

    /* TCP/IP */
    struct bufferevent *bev = bufferevent_socket_new(m_eventBase, clientFd, BEV_OPT_CLOSE_ON_FREE);
    if (bev == nullptr)
    {
        DCGM_LOG_ERROR << "Failed to create socket for fd " << clientFd;
        close(clientFd);
        return;
    }

    dcgm_connection_id_t connectionId = GetNextConnectionId();
    /* Add a tracked connection and remove our pending status */
    dcgmReturn_t dcgmReturn = AddConnection(bev, connectionId, DCGM_IPC_CS_ACTIVE, std::promise<dcgmReturn_t>());
    if (dcgmReturn != DCGM_ST_OK)
    {
        DCGM_LOG_ERROR << "Failed to AddConnection connectionId" << connectionId;
        bufferevent_free(bev); /* This auto closes clientFd */
        return;
    }

    /* Track our event before callbacks could be invoked */
    bufferevent_setcb(bev, DcgmIpc::StaticReadCB, NULL, DcgmIpc::StaticEventCB, this);
    bufferevent_enable(bev, EV_READ | EV_WRITE);

    DCGM_LOG_DEBUG << "Server connection accepted with connectionId " << connectionId << " bev " << bev;
}

/*****************************************************************************/
void DcgmIpc::StaticOnAccept(int listenFd, short /*ev*/, void *userData)
{
    DcgmIpc *dcgmIpc = (DcgmIpc *)userData;

    dcgmIpc->OnAccept(listenFd);
}

/*****************************************************************************/
void DcgmIpc::SendMessageImpl(DcgmIpcSendMessage &sendMessage)
{
    ASSERT_IS_IPC_THREAD;

    /* TCP/IP */
    DCGM_LOG_DEBUG << "Sending message to " << sendMessage.m_connectionId;

    DcgmIpcConnection *connection = ConnectionIdToPtr(sendMessage.m_connectionId);
    if (connection == nullptr)
    {
        DCGM_LOG_ERROR << "Couldn't find connectionId " << sendMessage.m_connectionId << " for SendMessage()";
        sendMessage.m_promise.set_value(DCGM_ST_CONNECTION_NOT_VALID);
        return;
    }

    dcgmReturn_t dcgmReturn = connection->SendMessage(std::move(sendMessage.m_message));
    sendMessage.m_promise.set_value(dcgmReturn);
}

/*****************************************************************************/
void DcgmIpc::SendMessageImplCB(evutil_socket_t, short, void *data)
{
    std::unique_ptr<DcgmIpcSendMessage> sendMessage((DcgmIpcSendMessage *)data);

    sendMessage->m_ipc->SendMessageImpl(*sendMessage);
}

/*****************************************************************************/
dcgmReturn_t DcgmIpc::SendMessage(dcgm_connection_id_t connectionId,
                                  std::unique_ptr<DcgmMessage> message,
                                  bool waitForSend)
{
    /* Using new here because we're transferring it through a C callback */
    DcgmIpcSendMessage *sendMessage = new DcgmIpcSendMessage(this, connectionId, std::move(message));

    auto sendMessageFuture = sendMessage->m_promise.get_future();

    int st = event_base_once(m_eventBase, -1, EV_TIMEOUT, DcgmIpc::SendMessageImplCB, sendMessage, 0);
    if (st)
    {
        DCGM_LOG_ERROR << "Got error " << st << " from event_base_once";
        return DCGM_ST_GENERIC_ERROR;
    }

    if (waitForSend)
    {
        dcgmReturn_t dcgmReturn = sendMessageFuture.get();
        if (dcgmReturn != DCGM_ST_OK)
        {
            DCGM_LOG_ERROR << "Async SendMessage returned " << errorString(dcgmReturn);
            return dcgmReturn;
        }
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmIpcConnection::SendMessage(std::unique_ptr<DcgmMessage> dcgmMessage)
{
    if (m_bev == nullptr)
    {
        DCGM_LOG_ERROR << "Tried to send to a connection with a null m_bev";
        return DCGM_ST_CONNECTION_NOT_VALID;
    }

    auto msgHdr   = dcgmMessage->GetMessageHdr();
    auto msgBytes = dcgmMessage->GetMsgBytesPtr();


    /* Note that we're only able to do these calls in succession because
       we only write to connections from a single thread. Otherwise, we'd have
       to stage the entire message in an evbuffer and call bufferevent_write_buffer */
    int st  = bufferevent_write(m_bev, msgHdr, sizeof(*msgHdr));
    int st2 = bufferevent_write(m_bev, msgBytes->data(), msgBytes->size());
    if (st || st2)
    {
        DCGM_LOG_ERROR << "Got error from first or second write " << st << ", " << st2;
        return DCGM_ST_CONNECTION_NOT_VALID;
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
void DcgmIpc::CloseConnectionImpl(DcgmIpcCloseConnection &closeConnection)
{
    ASSERT_IS_IPC_THREAD;

    dcgmReturn_t dcgmReturn = RemoveConnectionById(closeConnection.m_connectionId);
    if (dcgmReturn != DCGM_ST_OK)
    {
        DCGM_LOG_ERROR << "Got error " << errorString(dcgmReturn) << " for RemoveConnectionById of connectionId "
                       << closeConnection.m_connectionId;
    }
    else
    {
        DCGM_LOG_DEBUG << "connectionId " << closeConnection.m_connectionId << " was successfully closed.";
    }
}

/*****************************************************************************/
void DcgmIpc::CloseConnectionImplCB(evutil_socket_t, short, void *data)
{
    std::unique_ptr<DcgmIpcCloseConnection> closeConnection((DcgmIpcCloseConnection *)data);

    closeConnection->m_ipc->CloseConnectionImpl(*closeConnection);
}

/*****************************************************************************/
dcgmReturn_t DcgmIpc::CloseConnection(dcgm_connection_id_t connectionId)
{
    /* Using new here because we're transferring it through a C callback. The callback will
       assign this to a unique_ptr and then free it automatically */
    DcgmIpcCloseConnection *closeConnection = new DcgmIpcCloseConnection(this, connectionId);

    int st = event_base_once(m_eventBase, -1, EV_TIMEOUT, DcgmIpc::CloseConnectionImplCB, closeConnection, 0);
    if (st)
    {
        DCGM_LOG_ERROR << "Got error " << st << " from event_base_once";
        return DCGM_ST_GENERIC_ERROR;
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/