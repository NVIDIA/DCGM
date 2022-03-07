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
 *  Base/common class for all enumerated devices in a system
 */

#ifndef _NVVS_NVVS_DEVICE_H
#define _NVVS_NVVS_DEVICE_H

#include <string>

class Device
{
    /***************************PUBLIC***********************************/
public:
    Device() {};
    ~Device() {};

    void setDeviceName(std::string name)
    {
        this->name = name;
    }
    std::string getDeviceName()
    {
        return name;
    }

    /***************************PRIVATE**********************************/
private:
    /***************************PROTECTED********************************/
protected:
    std::string name;
};

#endif //_NVVS_NVVS_DEVICE_H
