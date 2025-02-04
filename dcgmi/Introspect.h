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
#ifndef INTROSPECT_H_
#define INTROSPECT_H_

#include "Command.h"
#include "CommandOutputController.h"
#include <string.h>

using std::string;

/*****************************************************************************
 * Define classes to extend commands
 ****************************************************************************/

class Introspect
{
public:
    Introspect();
    virtual ~Introspect();

    dcgmReturn_t DisplayStats(dcgmHandle_t handle, bool forHostengine);

private:
    string readableMemory(long long bytes);
    string readablePercent(double p);

    template <typename T>
    string readableTime(T usec);
};

/**
 * Display a summary of introspection information
 */
class DisplayIntrospectSummary : public Command
{
public:
    DisplayIntrospectSummary(string hostname, bool forHostengine);

protected:
    dcgmReturn_t DoExecuteConnected() override;

private:
    Introspect introspectObj;
    bool forHostengine;
};


#endif /* INTROSPECT_H_ */
