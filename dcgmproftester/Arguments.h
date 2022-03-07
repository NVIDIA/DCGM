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
#pragma once

#include "DcgmLogging.h"
#include "dcgm_structs.h"

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <tclap/Arg.h>
#include <tclap/CmdLine.h>
#include <tclap/SwitchArg.h>
#include <tclap/ValueArg.h>
#include <tclap/ValuesConstraint.h>
#include <vector>

namespace DcgmNs::ProfTester
{
struct Arguments_t
{
public:
    struct Parameters
    {
        bool m_valueValid; // whether values are specified.
        double m_minValue; // <value> --- ---       minimum match value
        double m_maxValue; // <value> --- ---       maximum match value

        bool m_percentTolerance; // if tolerance is a percentage (or absolute)
        double m_tolerance;      // tolerance (if m_valueValid is false)

        uint8_t m_waitToCheck;       // [0-100] 0 10      wait a percentage of the tests before comparing
        uint8_t m_maxGpusInParallel; // [0-255] 4 all  max GPU proc. in parallel
        double m_duration;           // <value> 30.0 2.00 duration of the test in seconds
        double m_reportInterval;     // <value> 1.00 0.01 rate of report gathering in seconds
        unsigned int m_syncCount;    // maximum activity synchronization count
        bool m_targetMaxValue;       // bool    false false   target maximum value
        bool m_noDcgmValidation;     // bool    false false   if set, we will NOT self-validate DCGM metrics.
        bool m_dvsOutput;            // bool    false false   if set, we will append DVS tags to our stdout.

        // Log Control
        std::string m_logFile;            // string  dcgmproftester.log --- log file
        DcgmLoggingSeverity_t m_logLevel; // enum    warning warning log severity

        // Operational flags.

        bool m_generate { true };  // whether to generate load
        bool m_report { true };    // whether to produce report
        bool m_validate { false }; // whether to validate
        bool m_fast { false };     // whether to finish as soon as possible

        unsigned int m_fieldId; // <value> --- ---  profiling FieldId
    } m_parameters;

    std::vector<unsigned int> m_gpuIds; // GPU ids for this run
};

template <typename T>
class Argument_t
{
private:
    using ThisType = Argument_t<T>;

    class Constraint_t : public TCLAP::Constraint<T>
    {
    private:
        ThisType &m_parent;
        std::string m_shortId;
        std::string m_constraintDescription;

    public:
        Constraint_t(ThisType &parent, const std::string &shortId, const std::string &constraintDescription)
            : m_parent(parent)
            , m_shortId(shortId)
            , m_constraintDescription(constraintDescription)
        {}

        std::string description() const override
        {
            return m_constraintDescription;
        }

        std::string shortID() const override
        {
            return m_shortId;
        }

        bool check(T const &value) const override
        {
            return m_parent.Check(value);
        }
    };

    T m_defaultValue;
    T m_value {};
    Constraint_t m_constraint;
    TCLAP::ValueArg<T> m_valueArg;
    std::function<bool(ThisType &, const T &)> m_checkFn;
    bool m_isDefault { true };

public:
    typedef T ArgType;

    Argument_t(TCLAP::CmdLine &cmd,
               T defaultValue,
               bool required,
               const std::string &shortName,
               const std::string &longName,
               const std::string &description,
               const std::string &shortId,
               const std::string &constraintDescription,
               std::function<bool(ThisType &, const T &)> checkFn)
        : m_defaultValue(defaultValue)
        , m_constraint(*this, shortId, constraintDescription)
        ,

        m_valueArg(shortName, longName, description, required, defaultValue, &m_constraint, cmd)
        ,

        m_checkFn(checkFn)
    {}

    bool Check(const T &value)
    {
        auto result = m_checkFn(*this, value);

        if (result)
        {
            m_isDefault = false;
            m_value     = value;
            m_valueArg.reset();
        }

        return result;
    }

    T Value(void)
    {
        return m_isDefault ? m_defaultValue : m_value;
    }

    bool IsDefault(void) const
    {
        return m_isDefault;
    }

    void Reset(void)
    {
        m_isDefault = true;
    }

    void ArgReset(void)
    {
        m_valueArg.reset();
    }
};

template <>
class Argument_t<bool>
{
private:
    using ThisType = Argument_t<bool>;

    struct Visitor : public TCLAP::Visitor
    {
    private:
        ThisType &m_parent;

    public:
        Visitor(ThisType &parent)
            : TCLAP::Visitor()
            , m_parent(parent)
        {}

        void visit() override
        {
            m_parent.Visit();
        }
    };


    bool m_defaultValue;
    std::function<void(ThisType &)> m_visitFn;
    bool m_value {};
    Visitor m_visitor;
    TCLAP::SwitchArg m_switchArg;
    bool m_isDefault { true };

public:
    typedef bool ArgType;

    Argument_t(TCLAP::CmdLine &cmd,
               bool defaultValue,
               bool required,
               const std::string &shortName,
               const std::string &longName,
               const std::string &description,
               std::map<unsigned char, bool> &shortMap,
               std::map<std::string, bool> &longMap,
               std::function<void(ThisType &)> visitFn)
        : m_defaultValue(defaultValue)
        , m_visitFn(visitFn)
        , m_visitor(*this)
        , m_switchArg(shortName, longName, description, cmd, defaultValue, &m_visitor)
    {
        if (shortName.length() > 0)
        {
            shortMap[shortName[0]] = true;
        }

        if (longName.length() > 0)
        {
            longMap[longName] = true;
        }
    }

    bool Value(void)
    {
        return m_isDefault ? m_defaultValue : m_value;
    }

    bool IsDefault(void) const
    {
        return m_isDefault;
    }

    void Reset(void)
    {
        m_isDefault = true;
        m_value     = m_defaultValue;
    }

    void ArgReset(void)
    {
        m_switchArg.reset();
    }

    void Visit(void)
    {
        m_switchArg.reset();
        m_isDefault = false;
        m_value     = true;
        m_visitFn(*this);
    }
};

struct ValueRange_t
{
    double m_min { 0.0 };
    double m_max { 0.0 };

    ValueRange_t() = default;

    ValueRange_t(double min, double max)
        : m_min(min)
        , m_max(max)
    {}
};

class ArgumentSet_t
{
private:
    std::map<unsigned int, ValueRange_t> m_defaults;
    std::map<unsigned char, bool> m_shortMap;
    std::map<std::string, bool> m_longMap;

    TCLAP::CmdLine m_cmd;

    Argument_t<double> m_waitToCheck;
    Argument_t<double> m_maxGpusInParallel;
    Argument_t<double> m_percentTolerance;
    Argument_t<double> m_absoluteTolerance;
    Argument_t<double> m_minValue;
    Argument_t<double> m_matchValue;
    Argument_t<double> m_maxValue;
    Argument_t<double> m_duration;
    Argument_t<double> m_reportInterval;
    Argument_t<bool> m_targetMaxValue;
    Argument_t<bool> m_noDcgmValidation;
    Argument_t<bool> m_dvsOutput;
    Argument_t<std::string> m_fieldIds;
    Argument_t<std::string> m_gpuIdString;
    Argument_t<bool> m_reset;
    Argument_t<std::string> m_modeString;
    Argument_t<unsigned int> m_syncCount;
    Argument_t<std::string> m_logFileString;
    Argument_t<std::string> m_logLevelString;
    Argument_t<std::string> m_configFile;

    std::vector<std::shared_ptr<Arguments_t>> m_arguments;
    bool m_snapshotReady { false };
    std::string m_lastFieldIds { "all" };

    dcgmReturn_t ProcessIntegerList(
        std::string,
        std::shared_ptr<Arguments_t>,
        bool,
        dcgmReturn_t (*callback)(ArgumentSet_t &, std::shared_ptr<Arguments_t>, bool, unsigned int arg));

    dcgmReturn_t ProcessStringList(
        std::string,
        std::shared_ptr<Arguments_t>,
        bool,
        dcgmReturn_t (*callback)(ArgumentSet_t &, std::shared_ptr<Arguments_t>, bool, const std::string &arg));

    dcgmReturn_t ProcessFieldIds(void);

public:
    ArgumentSet_t(const std::string &message, const std::string &version)
        : m_cmd(message, ' ', version)

        , m_waitToCheck(
              m_cmd,
              0.0,
              false,
              std::string("w"),
              std::string("wait-to-check"),
              std::string("percentage of run to wait before checking value"),
              std::string("wait to check percentage"),
              std::string("wait-to-check should be between 0 and 100"),

              [](decltype(m_waitToCheck) &arg, const decltype(m_waitToCheck)::ArgType &value) { return value < 100; })
        ,

        m_maxGpusInParallel(m_cmd,
                            4.0,
                            false,
                            std::string(""),
                            std::string("max-processes"),
                            std::string("maximum simultaneous GPUs tested (0=all)"),
                            std::string("maximum simultaneous GPUs tested"),
                            std::string("max-processes should be between 0 and 255"),

                            [](decltype(m_maxGpusInParallel) &arg,
                               const decltype(m_maxGpusInParallel)::ArgType &value) { return value <= 255; })
        ,

        m_percentTolerance(
            m_cmd,
            10.0,
            false,
            std::string(""),
            std::string("percent-tolerance"),
            std::string("percentage tolerance when checking value"),
            std::string("percentage tolerance"),
            std::string("percentage tolerance should be between 0 and 100"),

            [this](decltype(m_percentTolerance) &arg, const decltype(m_percentTolerance)::ArgType &value) {
                if (value > 100)
                {
                    return false;
                }

                m_absoluteTolerance.Reset();

                return value < 100;
            })
        ,

        m_absoluteTolerance(
            m_cmd,
            0.10,
            false,
            std::string("a"),
            std::string("absolute-tolerance"),
            std::string("absolute value match tolerance"),
            std::string("absolute tolerance"),
            std::string("absolute match tolerance must be non-negative"),

            [this](decltype(m_absoluteTolerance) &arg, const decltype(m_absoluteTolerance)::ArgType &value) {
                if (value < 0.0)
                {
                    return false;
                }

                m_percentTolerance.Reset();

                return true;
            })
        ,

        m_minValue(m_cmd,
                   0.0,
                   false,
                   std::string(""),
                   std::string("min-value"),
                   std::string("minimum value"),
                   std::string("minimum value for validation"),
                   std::string("minimum value must be less than or equal to maximum value"),

                   [this](decltype(m_minValue) &arg, const decltype(m_minValue)::ArgType &value) {
                       m_matchValue.Reset();

                       return true;
                   })
        ,

        m_matchValue(m_cmd,
                     0.0,
                     false,
                     std::string("m"),
                     std::string("match-value"),
                     std::string("value to match"),
                     std::string("mean value to match for validation"),
                     std::string("mean value should be midpoint of acceptable values"),

                     [this](decltype(m_matchValue) &arg, const decltype(m_matchValue)::ArgType &value) {
                         m_minValue.Reset();
                         m_maxValue.Reset();

                         return true;
                     })
        ,

        m_maxValue(m_cmd,
                   0.0,
                   false,
                   std::string(""),
                   std::string("max-value"),
                   std::string("maximum value"),
                   std::string("maximum value to match for validation"),
                   std::string("maximum value should be greater than or equal to minimum value"),

                   [this](decltype(m_maxValue) &arg, const decltype(m_maxValue)::ArgType &value) {
                       m_matchValue.Reset();

                       return true;
                   })
        ,

        m_duration(m_cmd,
                   30.0,
                   false,
                   std::string("d"),
                   std::string("duration"),
                   std::string("duration of test"),
                   std::string("Duration in seconds"),
                   std::string("Duration should be at least 1 second"),

                   [](decltype(m_duration) &arg, const decltype(m_duration)::ArgType &value) { return value >= 1.0; })
        ,

        m_reportInterval(m_cmd,
                         1.0,
                         false,
                         std::string("r"),
                         std::string("report"),
                         std::string("report gathering rate"),
                         std::string("Rate of report gathering in seconds"),
                         std::string("Reporting interval should not be more than 5 seconds"),

                         [](decltype(m_reportInterval) &arg, const decltype(m_reportInterval)::ArgType &value) {
                             return value <= 5.0;
                         })
        ,

        m_targetMaxValue(m_cmd,
                         false,
                         false,
                         std::string(""),
                         std::string("target-max-value"),
                         std::string("Run only at the target maximum value"),
                         m_shortMap,
                         m_longMap,
                         [](decltype(m_targetMaxValue) &arg) { return; })
        , m_noDcgmValidation(m_cmd,
                             false,
                             false,
                             std::string(""),
                             std::string("no-dcgm-validation"),
                             std::string("Do not do validation"),
                             m_shortMap,
                             m_longMap,
                             [](decltype(m_noDcgmValidation) &arg) { return; })
        , m_dvsOutput(m_cmd,
                      false,
                      false,
                      std::string(""),
                      std::string("dvs"),
                      std::string("Do not collect dvs output"),
                      m_shortMap,
                      m_longMap,
                      [](decltype(m_dvsOutput) &arg) { return; })
        ,

#define xstr(s) str(s)
#define str(s)  #s

        m_fieldIds(m_cmd,
                   std::string("all"),
                   false,
                   std::string("t"),
                   std::string("fieldId"),
                   std::string("Profiling FieldId"),
                   std::string("Valid value for the FieldId is a comma-separated list of"),
                   std::string("integers between " xstr(DCGM_FI_PROF_FIRST_ID) " and " xstr(
                       DCGM_FI_PROF_LAST_ID) " or 'all' or 'nvlink' or 'nonvlink'"),

                   [this](decltype(m_fieldIds) &arg, const decltype(m_fieldIds)::ArgType &value) {
                       if (!m_snapshotReady)
                       {
                           m_snapshotReady = true;
                           m_lastFieldIds  = value;

                           return true;
                       }

                       dcgmReturn_t rv = ProcessFieldIds();

                       m_lastFieldIds = value;

                       return rv == DCGM_ST_OK;
                   })
        ,

#undef str
#undef xstr

        m_gpuIdString(m_cmd,
                      std::string("all"),
                      false,
                      std::string("i"),
                      std::string("gpuIds"),
                      std::string("gpu IDs"),
                      std::string("List of GPU IDs to run on"),
                      std::string("GPU IDs to run on. Use dcgmi discovery -l to list valid gpuIds"),

                      [](decltype(m_gpuIdString) &arg, const decltype(m_gpuIdString)::ArgType &value) { return true; })
        ,

        m_reset(m_cmd,
                false,
                false,
                std::string(""),
                std::string("reset"),
                std::string("Reset switch arguments"),
                m_shortMap,
                m_longMap,
                [this](decltype(m_reset) &arg) {
                    m_targetMaxValue.Reset();
                    m_noDcgmValidation.Reset();
                    m_dvsOutput.Reset();
                })
        ,

        m_modeString(
            m_cmd,
            std::string("generateload,report,novalidate,nofast"),
            false,
            std::string(""),
            std::string("mode"),
            std::string("operational mode"),
            std::string("operational mode: fast, generate load, report, validate"),
            std::string("operational mode must be one of [no]fast, [no]generateload, [no]report, [no]validate"),
            [](decltype(m_modeString) &arg, const decltype(m_modeString)::ArgType &value) { return true; })
        ,

        m_syncCount(m_cmd,
                    0,
                    false,
                    std::string(""),
                    std::string("sync-count"),
                    std::string("max synchronous measurement attempt count"),
                    std::string("Number of attempts to reach target activity levels"),
                    std::string("Sync count should not be more than 10"),

                    [](decltype(m_syncCount) &arg, const decltype(m_syncCount)::ArgType &value) { return value <= 10; })
        ,

        m_logFileString(
            m_cmd,
            std::string("dcgmproftester.log"),
            false,
            std::string(""),
            std::string("log-file"),
            std::string("log file"),
            std::string("Log file name"),
            std::string("Log file must name a writable file"),

            [](decltype(m_logFileString) &arg, const decltype(m_logFileString)::ArgType &value) { return true; })
        ,

        m_logLevelString(
            m_cmd,
            std::string("info"),
            false,
            std::string(""),
            std::string("log-level"),
            std::string("log level"),
            std::string("Log severity level"),
            std::string("Log severity level must be one of" DCGM_LOGGING_SEVERITY_OPTIONS),

            [](decltype(m_logLevelString) &arg, const decltype(m_logLevelString)::ArgType &value) { return true; })
        ,

        m_configFile(m_cmd,
                     std::string(""),
                     false,
                     std::string("c"),
                     std::string("config"),
                     std::string("configuration file"),
                     std::string("Configuration file (YAML or JSON) to include"),
                     std::string("Configuration file must be in YAML or JSON format"),

                     [](decltype(m_configFile) &arg, const decltype(m_configFile)::ArgType &value) { return true; })

            {};

    void AddDefault(unsigned int fieldId, ValueRange_t valueRange);

    dcgmReturn_t Parse(int argc, char *argv[]);

    dcgmReturn_t Process(std::function<dcgmReturn_t(std::shared_ptr<Arguments_t> arguments)>);
};

} // namespace DcgmNs::ProfTester
