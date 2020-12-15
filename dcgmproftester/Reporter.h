/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
#include "DcgmLogging.h"
#include <iostream>
#include <sstream>
#include <thread>

namespace DcgmNs::ProfTester
{
/**
 * This class is used as a wrapper to log to the console and DCGM_LOG_INFO.
 */
class ReporterBase
{
private:
    static void checkThread(void);

public:
    enum Flags
    {
        new_line
    };

protected:
    ReporterBase()
    {}

    template <typename T>
    void log(std::stringstream &buffer, T const &value)
    {
        checkThread();
        std::cout << value; // output this right away
        buffer << value;    // logger works best with whole lines
    }

    void log(std::stringstream &buffer, plog::Severity severity, Flags flag);
};

template <plog::Severity Severity>
class Reporter : public ReporterBase
{
private:
    static inline std::stringstream m_buffer = std::stringstream("");

public:
    Reporter()
        : ReporterBase()
    {}

    template <typename T>
    Reporter<Severity> &operator<<(T const &value)
    {
        log(m_buffer, value);

        return *this;
    }

    Reporter<Severity> &operator<<(Flags flag)
    {
        log(m_buffer, Severity, flag);

        return *this;
    }
};

} // namespace DcgmNs::ProfTester
