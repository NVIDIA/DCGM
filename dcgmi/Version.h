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
/*
 * File:   Version.h
 */

#ifndef VERSION_H
#define VERSION_H

#include "Command.h"


/*****************************************************************************
 * Define classes to extend commands
 ****************************************************************************/

/**
 * Set Version Invoker class
 */
class VersionInfo : public Command
{
public:
    VersionInfo(std::string hostname);

protected:
    /*****************************************************************************
     * Override the Execute method for getting version info
     *****************************************************************************/
    dcgmReturn_t DoExecuteConnected() override;
};

#endif /* VERSION_H */