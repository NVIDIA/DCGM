//
// Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.
//

#ifndef __UTILS_H__
#define __UTILS_H__

#include <unordered_map>
#include <string>
#include <iostream>
#include <optional>
#include <vector>
#include <sstream>

#include "nvsdm.h"
#ifdef NV_MODS
#include "../mods/mods_utils.h"
#endif

#define NVSDM_CALL_FUNC_AND_CHECK(func, ...) do { \
	nvsdmRet_t _ret = func(__VA_ARGS__); \
	if (_ret != NVSDM_SUCCESS) { \
		return _ret; \
	} \
} while(0)

#ifdef DEBUG
#define TRACE_FUNC_LINE() \
    NVSDM_DEBUG_MSG(__PRETTY_FUNCTION__, "::", __LINE__)
#else
#define TRACE_FUNC_LINE()
#endif

/*
 * Convenience typedefs
 */
typedef enum nvsdmDevType nvsdmDevType_t;
typedef enum nvsdmLogLevel nvsdmLogLevel_t;

namespace nvsdm
{
namespace utils
{
    inline nvsdmLogLevel_t g_logLevel = NVSDM_LOG_LEVEL_FATAL; /* Print out fatal messages by default */

    inline std::ostream *g_stream = nullptr;

    inline nvsdmRet_t setLogLevel(nvsdmLogLevel_t lvl)
    {
        if (size_t(lvl) > NVSDM_LOG_LEVEL_INFO)
        {
            /* Invalid log level */
            return NVSDM_ERROR_INVALID_ARG;
        }
        g_logLevel = lvl;
        return NVSDM_SUCCESS;
    }

    inline nvsdmLogLevel_t getLogLevel()
    {
        return g_logLevel;
    }

    nvsdmRet_t setLogFile(std::string const& logFile);

	inline nvsdmRet_t checkNULLArgs()
	{
		/* Default case */
		return NVSDM_SUCCESS;
	}

	template <typename T, typename...Ts> nvsdmRet_t checkNULLArgs(T arg, Ts... tsub)
	{
		if (arg == nullptr)
		{
			return NVSDM_ERROR_INVALID_ARG;
		}

		return checkNULLArgs(tsub...);
	}

	inline char const *getErrorString(nvsdmRet_t ret)
	{
        /*
         * Map of error code to string value
         * Must be kept in sync with the 'nvsdmRet_t' definition
         */
		static std::unordered_map <nvsdmRet_t, char const *, std::hash<int>> const s_errorStrings = {
			{NVSDM_SUCCESS,                             "Success"},
			{NVSDM_ERROR_UNINITIALIZED,                 "Uninitialized"},
			{NVSDM_ERROR_NOT_SUPPORTED,                 "Not Supported"},
			{NVSDM_ERROR_INVALID_ARG,                   "Invalid Argument"},
			{NVSDM_ERROR_INSUFFICIENT_SIZE,             "Insufficient Size"},
			{NVSDM_ERROR_VERSION_NOT_SUPPORTED,         "Version Not Supported"},
			{NVSDM_ERROR_MEMORY,                        "Insufficient Memory"},
            {NVSDM_ERROR_DEVICE_DISCOVERY_FAILURE,      "Device Discovery Failure"},
            {NVSDM_ERROR_LIBRARY_LOAD,                  "Error Loading Library"},
            {NVSDM_ERROR_FUNCTION_NOT_FOUND,            "Function Not Found"},
            {NVSDM_ERROR_INVALID_CTR,                   "Invalid Telemetry Counter"},
            {NVSDM_ERROR_TELEMETRY_READ,                "Telemetry Read Error"},
            {NVSDM_ERROR_DEVICE_NOT_FOUND,              "Device Not Found"},
            {NVSDM_ERROR_UMAD_INIT,                     "UMAD Init Error"},
            {NVSDM_ERROR_UMAD_LIB_CALL,                 "UMAD Library Call Error"},
            {NVSDM_ERROR_MAD_LIB_CALL,                  "MAD Library Call Error"},
            {NVSDM_ERROR_NLSOCKET_OPEN_FAILED,          "Netlink Socket Open Error"},
            {NVSDM_ERROR_NLSOCKET_BIND_FAILED,          "Netlink Socket Bind Error"},
            {NVSDM_ERROR_NLSOCKET_SEND_FAILED,          "Netlink Socket Send Error"},
            {NVSDM_ERROR_NLSOCKET_RECV_FAILED,          "Netlink Socket Recv Error"},
            {NVSDM_ERROR_FILE_OPEN_FAILED,              "Error Opening File"},
			{NVSDM_ERROR_UNKNOWN,                       "Unknown Error"},
		};
        char const *str = "N/A";
		auto it = s_errorStrings.find(ret);
		if (it != s_errorStrings.end())
		{
			str = it->second;
		}
		return str;
	}

    inline void nvsdmMsgHelper(std::ostream& stream)
    {
        stream << std::endl;
    }

    template <typename HeadType, typename...Args> void nvsdmMsgHelper(std::ostream& stream, HeadType head, Args... tail)
    {
        stream << " " << head;
        nvsdmMsgHelper(stream, tail...);
    }

    template <typename...Args> void nvsdmMsg(nvsdmLogLevel_t lvl, std::string&& str, Args...args)
    {
        if (g_logLevel < lvl)
        {
            /* Priority not high enough */
            return;
        }

        if (g_stream == nullptr)
        {
            /* No existing stream so default to std::cerr */
            g_stream = &std::cerr;
        }

        std::ostream& stream = *g_stream;
        stream << str << ":";
        nvsdmMsgHelper(stream, args...);
    }

    /*
     * Retrieve counter name for the given counter type.
     */
    std::optional<std::string> getCounterName(uint16_t type, uint16_t ctr);

    /*
     * Retrieve a {counter type, counter ID} pair for the given counter name.
     */
    std::optional<std::pair<uint16_t, uint16_t>> getCounterGivenName(std::string const& name);

    /*
     * Retrieve a list of supported counters of the given type.
     */
    std::vector<uint16_t> getSupportedCounters(nvsdmTelemType_t type);

    /*
     * Generic hex dump function; dumps to stdout via the "NVSDM_LOG" framework
     */
    void hexDump(uint8_t const *data, size_t size);

    /*
     * Wrapper around "be32toh"; allows it to be passed as an "std::function"
     */
    uint32_t nvsdmNetToHost32(uint32_t net);
    /*
     * Wrapper around "be64toh"; allows it to be passed as an "std::function"
     */
    uint64_t nvsdmNetToHost64(uint64_t net);

    /*
     * Wrapper around "htobe32"; allows it to be passed as an "std::function"
     */
    uint32_t nvsdmHostToNet32(uint32_t net);
    /*
     * Wrapper around "htobe64"; allows it to be passed as an "std::function"
     */
    uint64_t nvsdmHostToNet64(uint64_t net);

    /*
     * Formatting function to convert a number to a hex string
     */
    template <typename T> std::string toHexString(T t)
    {
        std::stringstream _str;
        _str << "0x" << std::hex << t;
        return _str.str();
    }

} // utils
} // nvsdm

/*
 * Helper macros
 */
#ifndef NVSDM_LOG_MSG
#define NVSDM_LOG_MSG(lvl, ...) \
    nvsdm::utils::nvsdmMsg(NVSDM_LOG_LEVEL_ ## lvl, #lvl, __VA_ARGS__)
#endif

#define NVSDM_FATAL_MSG(...)   NVSDM_LOG_MSG(FATAL, __VA_ARGS__)
#define NVSDM_ERROR_MSG(...)   NVSDM_LOG_MSG(ERROR, __VA_ARGS__)
#define NVSDM_WARN_MSG(...)    NVSDM_LOG_MSG(WARN, __VA_ARGS__)
#define NVSDM_DEBUG_MSG(...)   NVSDM_LOG_MSG(DEBUG, __VA_ARGS__)
#define NVSDM_INFO_MSG(...)    NVSDM_LOG_MSG(INFO, __VA_ARGS__)
/*
 * Special macro for "todo" messages -- always prints at 'FATAL' level
 * (but without the 'FATAL' prefix)
 */
#define NVSDM_TODO_MSG(...)    nvsdm::utils::nvsdmMsg(NVSDM_LOG_LEVEL_FATAL, "TODO", __VA_ARGS__)
/*
 * Special macro to always print out a message -- always prints at 'FATAL level
 * to ensure it is always printed (no 'FATAL prefix is printed)
 */
#define NVSDM_FORCE_MSG(...)   nvsdm::utils::nvsdmMsg(NVSDM_LOG_LEVEL_FATAL, "FORCE", __VA_ARGS__)
/*
 * Special macro to print out a message without the standard level prefix
 */
#define NVSDM_MSG(lvl, ...) \
    nvsdm::utils::nvsdmMsg(NVSDM_LOG_LEVEL_ ## lvl, __VA_ARGS__)

#endif // __UTILS_H__
