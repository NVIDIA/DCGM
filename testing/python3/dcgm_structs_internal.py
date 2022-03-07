# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##
# Python bindings for "dcgm_structs.h"
##

from ctypes import *
from ctypes.util import find_library
import sys
import os
import threading
import string
#import cuda
import dcgm_structs
from dcgm_fields import _PrintableStructure

# Max length of the DCGM string field
DCGM_MAX_STR_LENGTH = 256
DCGM_MAX_BLOB_LENGTH = 3200


class c_dcgmGpuInfo(dcgm_structs._PrintableStructure):
    _fields_ = [
        ('gpuId', c_uint),
        ('uuid', c_char * DCGM_MAX_STR_LENGTH)
    ]

class value(Union):
    _fields_ = [
        ('i64', c_int64),
        ('dbl', c_double),
        ('str', c_char * DCGM_MAX_STR_LENGTH)
    ]

# Below is a test API simply to make sure versioning is working correctly
class c_dcgmVersionTest_v1(dcgm_structs._PrintableStructure):
    _fields_ = [
        # version must always be first
        ('version', c_uint),
        ('a', c_uint)
    ]

class c_dcgmVersionTest_v2(dcgm_structs._PrintableStructure):
    _fields_ = [
        # version must always be first
        ('version', c_uint),
        ('a', c_uint),
        ('b', c_uint)
    ]

dcgmVersionTest_version1 = dcgm_structs.make_dcgm_version(c_dcgmVersionTest_v1, 1)
dcgmVersionTest_version2 = dcgm_structs.make_dcgm_version(c_dcgmVersionTest_v2, 2)
dcgmVersionTest_version3 = dcgm_structs.make_dcgm_version(c_dcgmVersionTest_v2, 3)

# Represents a command to save or load a JSON file to/from the DcgmCacheManager
_dcgmStatsFileType_t = c_uint
DCGM_STATS_FILE_TYPE_JSON = 0

class c_dcgmCacheManagerSave_v1(dcgm_structs._PrintableStructure):
    _fields_ = [
        # version must always be first
        ('version', c_uint),
        ('fileType', _dcgmStatsFileType_t),
        ('filename', c_char * 256)
    ]

class c_dcgmCacheManagerLoad_v1(dcgm_structs._PrintableStructure):
    _fields_ = [
        ('version', c_uint),
    ]

dcgmCacheManagerSave_version1 = dcgm_structs.make_dcgm_version(c_dcgmCacheManagerSave_v1, 1)
dcgmCacheManagerLoad_version1 = dcgm_structs.make_dcgm_version(c_dcgmCacheManagerLoad_v1, 1)

class c_dcgmInjectFieldValue_v1(dcgm_structs._PrintableStructure):
    _fields_ = [
        ('version', c_uint),
        ('fieldId', c_short),
        ('fieldType', c_short),
        ('status', c_uint),
        ('ts', c_int64),
        ('value', dcgm_structs.c_dcgmFieldValue_v1_value)
    ]

# This structure is used to represent a field value to be injected into the cache manager
dcgmInjectFieldValue_version1  = dcgm_structs.make_dcgm_version(c_dcgmInjectFieldValue_v1, 1)

#Cache Manager Info flags
DCGM_CMI_F_WATCHED = 0x00000001

#Watcher types
DcgmWatcherTypeClient           = 0 # Embedded or remote client via external APIs
DcgmWatcherTypeHostEngine       = 1 # Watcher is NvcmHostEngineHandler
DcgmWatcherTypeHealthWatch      = 2 # Watcher is NvcmHealthWatch
DcgmWatcherTypePolicyManager    = 3 # Watcher is NvcmPolicyMgr
DcgmWatcherTypeCacheManager     = 4 # Watcher is DcgmCacheManager
DcgmWatcherTypeConfigManager    = 5 # Watcher is NvcmConfigMgr
DcgmWatcherTypeNvSwitchManager  = 6 # Watcher is NvSwitchManager


# ID of a remote client connection within the host engine
dcgm_connection_id_t = c_uint32

# Special constant for not connected
DCGM_CONNECTION_ID_NONE = 0

DCGM_CM_FIELD_INFO_NUM_WATCHERS = 10

class c_dcgm_cm_field_info_watcher_t(dcgm_structs._PrintableStructure):
    _fields_ = [
        ('watcherType', c_uint),
        ('connectionId', dcgm_connection_id_t),
        ('monitorFrequencyUsec', c_int64),
        ('maxAgeUsec', c_int64)
    ]

class dcgmCacheManagerFieldInfo_v3(dcgm_structs._PrintableStructure):
    _fields_ = [
        ('version', c_uint32),
        ('flags', c_uint32),
        ('gpuId', c_uint32),
        ('fieldId', c_uint16),
        ('lastStatus', c_int16),
        ('oldestTimestamp', c_int64),
        ('newestTimestamp', c_int64),
        ('monitorFrequencyUsec', c_int64),
        ('maxAgeUsec', c_int64),
        ('execTimeUsec', c_int64),
        ('fetchCount', c_int64),
        ('numSamples', c_int32),
        ('numWatchers', c_int32),
        ('watchers', c_dcgm_cm_field_info_watcher_t * DCGM_CM_FIELD_INFO_NUM_WATCHERS)
    ]

dcgmCacheManagerFieldInfo_version3 = dcgm_structs.make_dcgm_version(dcgmCacheManagerFieldInfo_v3, 3)

class c_dcgmCreateFakeEntities_v2(dcgm_structs._PrintableStructure):
    _fields_ = [
        ('version', c_uint32),
        ('numToCreate', c_uint32),
        ('entityList', dcgm_structs.DCGM_MAX_HIERARCHY_INFO * dcgm_structs.c_dcgmMigHierarchyInfo_t),
    ]

dcgmCreateFakeEntities_version2 = dcgm_structs.make_dcgm_version(c_dcgmCreateFakeEntities_v2, 2)

class c_dcgmSetNvLinkLinkState_v1(dcgm_structs._PrintableStructure):
    _fields_ = [
        ('version', c_uint32),       # Version. Should be dcgmSetNvLinkLinkState_version1 
        ('entityGroupId', c_uint32), # Entity group of the entity to set the link state of
        ('entityId', c_uint32),      # ID of the entity to set the link state of
        ('linkId', c_uint32),        # Link (or portId) of the link to set the state of
        ('linkState', c_uint32),     # State to set the link to
        ('unused', c_uint32)         # Not used for now. Set to 0
    ]

dcgmSetNvLinkLinkState_version1 = dcgm_structs.make_dcgm_version(c_dcgmSetNvLinkLinkState_v1, 1)
