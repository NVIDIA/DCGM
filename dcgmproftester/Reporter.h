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
#include "DcgmLogging.h"
#include <iostream>
#include <sstream>
#include <thread>
#include <type_traits>

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

class Reporter : public ReporterBase
{
private:
    std::stringstream m_buffer;
    plog::Severity m_severity;

public:
    Reporter(plog::Severity severity)
        : ReporterBase()
        , m_buffer(std::stringstream(""))
        , m_severity(severity)
    {}

    /*
     * This rather convoluted template exists to limit generation of individual
     * template instances for each of the integral types and types convertible
     * to const char * (such as const char (&)[N], for each N).
     *
     * We only accept the char and long long integral types, and const char *
     * C-style string types, as well as other non-integral and non-string
     * types.
     *
     * To convert other integral and string-convertible types, we take advantage
     * of explicit parameter conversion permitted for fully-specialized template
     * instantiations and provide the necessary functions below. These simply
     * trampoline back into the template instantiations for the integral or
     * string types that it accepts. Technically, these are different functions.
     *
     * This leverages the SINFAE ("Substitution Failure Is Not An Error")
     * principle.
     *
     * Note that it is not enough to partially specialize the function
     * template as private as then a match WILL be found to generate the
     * function, which will be non-accessable, resulting in a compile-time
     * error.
     *
     * Note also, that although we lose type information converting const
     * char (&)[N} to const char * (namely the length of the string), the
     * underlying logging routine interprets const char * as a C-string and
     * we chose to preserve these semantics instead of introducing a length-
     * saving conversion into a string_view.
     */
    template <
        typename T,
        typename std::enable_if<(!std::is_convertible<T, const char *>::value || std::is_same<T, const char *>::value)
                                    && (!std::is_convertible<T, long long>::value || std::is_same<T, long long>::value
                                        || std::is_same<T, char>::value
                                        || (!std::is_integral<T>::value && !std::is_same<T, dcgmReturn_enum>::value
                                            && std::is_convertible<T, double>::value)),
                                bool>::type * = nullptr>
    Reporter &operator<<(T value)
    {
        log(m_buffer, value);

        return *this;
    }

    /*
     * Here, we explicitly generate the integral and C-string templates that
     * accept implicit conversions from other integral types or fixed, known
     * size character arrays.
     */
    Reporter &operator<<(const char *value)
    {
        return operator<<<const char *>(value);
    }

    Reporter &operator<<(long long value)
    {
        return operator<<<long long>(value);
    }

    /*
     * Here, we explicitly generate the Flag handling template.
     */
    Reporter &operator<<(Flags flag)
    {
        log(m_buffer, m_severity, flag);

        return *this;
    }
};

extern Reporter info_reporter;
extern Reporter warn_reporter;
extern Reporter error_reporter;

} // namespace DcgmNs::ProfTester

//  LocalWords:  instantiations
