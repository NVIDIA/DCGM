//
// Copyright 2021 NVIDIA Corporation.  All rights reserved.
//
// NOTICE TO USER:
//
// This source code is subject to NVIDIA ownership rights under U.S. and
// international Copyright laws.  Users and possessors of this source code
// are hereby granted a nonexclusive, royalty-free license to use this code
// in individual and commercial software.
//
// NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
// CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
// IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
// REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
// MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
// IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
// OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
// OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
// OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
// OR PERFORMANCE OF THIS SOURCE CODE.
//
// U.S. Government End Users.   This source code is a "commercial item" as
// that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
// "commercial computer  software"  and "commercial computer software
// documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
// and is provided to the U.S. Government only as a commercial end item.
// Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
// 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
// source code with only those rights set forth herein.
//
// Any use of this source code in individual and commercial software must
// include, in the user documentation and internal comments to the code,
// the above Disclaimer and U.S. Government End Users Notice.
//

#ifndef _NSCQ_H_
#define _NSCQ_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stdint.h>

#define NSCQ_API_VERSION(major, minor, patch)                                       \
    (((((uint32_t)major) & 0xFFu) << 24u) | ((((uint32_t)minor) & 0xFFFu) << 12u) | \
     ((((uint32_t)patch) & 0xFFFu) << 0u))
#define NSCQ_API_VERSION_CODE_MAJOR(code) (((code) >> 24u) & 0xFFu)
#define NSCQ_API_VERSION_CODE_MINOR(code) (((code) >> 12u) & 0xFFFu)
#define NSCQ_API_VERSION_CODE_PATCH(code) (((code) >> 0u) & 0xFFFu)

#define NSCQ_API_VERSION_CODE \
    NSCQ_API_VERSION(1, 2, 0)
#define NSCQ_API_VERSION_DEVEL "g1747f35"

extern const uint32_t nscq_api_version;
extern const char nscq_api_version_devel[];

// nscq_rc_t value ranges:
//  0          : success
//  1 to 127   : warnings (success, but with caveats)
//  -128 to -1 : errors
#define NSCQ_RC_SUCCESS                      (0)
#define NSCQ_RC_WARNING_RDT_INIT_FAILURE     (1)
#define NSCQ_RC_ERROR_NOT_IMPLEMENTED        (-1)
#define NSCQ_RC_ERROR_INVALID_UUID           (-2)
#define NSCQ_RC_ERROR_RESOURCE_NOT_MOUNTABLE (-3)
#define NSCQ_RC_ERROR_OVERFLOW               (-4)
#define NSCQ_RC_ERROR_UNEXPECTED_VALUE       (-5)
#define NSCQ_RC_ERROR_UNSUPPORTED_DRV        (-6)
#define NSCQ_RC_ERROR_DRV                    (-7)
#define NSCQ_RC_ERROR_EXT                    (-127)
#define NSCQ_RC_ERROR_UNSPECIFIED            (-128)

// The pointer-cast-dereference is done so that these macros can also be used
// with the nscq_*_result_t types, which embed the result code as the first
// member of the result struct.
#ifdef __cplusplus
#define NSCQ_SUCCESS(result) (*(reinterpret_cast<nscq_rc_t*>(&(result))) == NSCQ_RC_SUCCESS)
#define NSCQ_WARNING(result) (*(reinterpret_cast<nscq_rc_t*>(&(result))) > NSCQ_RC_SUCCESS)
#define NSCQ_ERROR(result)   (*(reinterpret_cast<nscq_rc_t*>(&(result))) < NSCQ_RC_SUCCESS)
#else
#define NSCQ_SUCCESS(result) (*((nscq_rc_t*)&(result)) == NSCQ_RC_SUCCESS)
#define NSCQ_WARNING(result) (*((nscq_rc_t*)&(result)) > NSCQ_RC_SUCCESS)
#define NSCQ_ERROR(result)   (*((nscq_rc_t*)&(result)) < NSCQ_RC_SUCCESS)
#endif

#define _NSCQ_RESULT_TYPE(t, m) \
    typedef struct {            \
        nscq_rc_t rc;           \
        t m;                    \
    } nscq_##m##_result_t

typedef int8_t nscq_rc_t;
typedef struct nscq_session_st* nscq_session_t;
typedef struct nscq_observer_st* nscq_observer_t;

// All function callbacks (e.g., used for path observers) are passed using a single type.
// These are cast internally to the appropriate function types internally before use.
typedef void (*nscq_fn_t)(void);

// Convenience macro for casting a function pointer to the common nscq_fn_t type.
#ifdef __cplusplus
#define NSCQ_FN(fn) reinterpret_cast<nscq_fn_t>(fn)
#else
#define NSCQ_FN(fn) ((nscq_fn_t)&fn)
#endif

typedef struct {
    uint8_t bytes[16];
} nscq_uuid_t;

#define NSCQ_DRIVER_FABRIC_STATE_UNKNOWN         (-1) // unknown state
#define NSCQ_DRIVER_FABRIC_STATE_OFFLINE         (0)  // offline (No driver loaded)
#define NSCQ_DRIVER_FABRIC_STATE_STANDBY         (1)  // driver up, no FM
#define NSCQ_DRIVER_FABRIC_STATE_CONFIGURED      (2)  // driver up, FM up
#define NSCQ_DRIVER_FABRIC_STATE_MANAGER_TIMEOUT (3)  // driver up, FM timed out
#define NSCQ_DRIVER_FABRIC_STATE_MANAGER_ERROR   (4)  // driver up, FM error state

#define NSCQ_DEVICE_FABRIC_STATE_UNKNOWN     (-1) // unknown state
#define NSCQ_DEVICE_FABRIC_STATE_OFFLINE     (0)  // offline: No driver, no FM
#define NSCQ_DEVICE_FABRIC_STATE_STANDBY     (1)  // driver up, no FM, !blacklisted
#define NSCQ_DEVICE_FABRIC_STATE_CONFIGURED  (2)  // driver up, FM up, !blacklisted
#define NSCQ_DEVICE_FABRIC_STATE_BLACKLISTED (3)  // device is blacklisted

#define NSCQ_NVLINK_STATE_UNKNOWN (-1)
#define NSCQ_NVLINK_STATE_OFF     (0)
#define NSCQ_NVLINK_STATE_SAFE    (1)
#define NSCQ_NVLINK_STATE_ACTIVE  (2)
#define NSCQ_NVLINK_STATE_ERROR   (3)

typedef int8_t nscq_nvlink_state_t;

typedef struct {
    int16_t driver;
    int16_t device;
} nscq_fabric_state_t;

#define NSCQ_DEVICE_BLACKLIST_REASON_UNKNOWN                    (-1)
#define NSCQ_DEVICE_BLACKLIST_REASON_NONE                       (0)
#define NSCQ_DEVICE_BLACKLIST_REASON_MANUAL_OUT_OF_BAND         (1)
#define NSCQ_DEVICE_BLACKLIST_REASON_MANUAL_IN_BAND             (2)
#define NSCQ_DEVICE_BLACKLIST_REASON_MANUAL_PEER                (3)
#define NSCQ_DEVICE_BLACKLIST_REASON_TRUNK_LINK_FAILURE         (4)
#define NSCQ_DEVICE_BLACKLIST_REASON_TRUNK_LINK_FAILURE_PEER    (5)
#define NSCQ_DEVICE_BLACKLIST_REASON_ACCESS_LINK_FAILURE        (6)
#define NSCQ_DEVICE_BLACKLIST_REASON_ACCESS_LINK_FAILURE_PEER   (7)
#define NSCQ_DEVICE_BLACKLIST_REASON_UNSPEC_DEVICE_FAILURE      (8)
#define NSCQ_DEVICE_BLACKLIST_REASON_UNSPEC_DEVICE_FAILURE_PEER (9)

typedef int8_t nscq_blacklist_reason_t;

typedef struct {
    uint32_t domain;
    uint8_t bus;
    uint8_t device;
    uint8_t function;
} nscq_pcie_location_t;

typedef struct {
    char data[64];
} nscq_label_t;

typedef struct {
    uint8_t protocol_version;
    uint8_t link_width;
    uint32_t bandwidth; // Mibps
} nscq_link_caps_t;

typedef struct {
    uint64_t rx; // Mibits
    uint64_t tx; // Mibits
} nscq_link_throughput_t;

typedef struct
{
    uint32_t error_value;
    uint64_t time;
} nscq_error_t;

_NSCQ_RESULT_TYPE(nscq_session_t, session);
_NSCQ_RESULT_TYPE(nscq_observer_t, observer);

nscq_rc_t nscq_uuid_to_label(const nscq_uuid_t*, nscq_label_t*, uint32_t);

#define NSCQ_SESSION_CREATE_MOUNT_DEVICES (0x1u)

nscq_session_result_t nscq_session_create(uint32_t);
void nscq_session_destroy(nscq_session_t);

nscq_rc_t nscq_session_mount(nscq_session_t, const nscq_uuid_t*, uint32_t);
void nscq_session_unmount(nscq_session_t, const nscq_uuid_t*);

nscq_rc_t nscq_session_path_observe(nscq_session_t, const char*, nscq_fn_t, void*, uint32_t);
nscq_observer_result_t nscq_session_path_register_observer(nscq_session_t, const char*, nscq_fn_t,
                                                           void*, uint32_t);
void nscq_observer_deregister(nscq_observer_t);
nscq_rc_t nscq_observer_observe(nscq_observer_t, uint32_t);
nscq_rc_t nscq_session_observe(nscq_session_t, uint32_t);

#ifdef __cplusplus
}
#endif

#endif // _NSCQ_H_
