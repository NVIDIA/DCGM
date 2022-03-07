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
#ifndef DCGMWATCHER_H
#define DCGMWATCHER_H

#include <dcgm_structs_internal.h>

/* DcgmWatcherType is defined in dcgm_structs_internal.h */

/*****************************************************************************/
class DcgmWatcher
{
public:
    /* Constructor */
    explicit DcgmWatcher(DcgmWatcherType_t watcherType, dcgm_connection_id_t connectionId = DCGM_CONNECTION_ID_NONE)
        : watcherType(watcherType)
        , connectionId(connectionId)
    {}

    DcgmWatcher()
        : DcgmWatcher(DcgmWatcherTypeClient, DCGM_CONNECTION_ID_NONE)
    {}

    DcgmWatcher(DcgmWatcher const &) = default;
    DcgmWatcher(DcgmWatcher &&)      = default;

    DcgmWatcher &operator=(DcgmWatcher const &) = default;
    DcgmWatcher &operator=(DcgmWatcher &&) = default;

    DcgmWatcherType_t watcherType;     /*!< Watcher type */
    dcgm_connection_id_t connectionId; /*!< Connection associated with this watcher */

    /* Operators */
    bool operator==(const DcgmWatcher &other) const;
    bool operator!=(const DcgmWatcher &other) const;
};


/*****************************************************************************/

#endif // DCGMWATCHER_H
