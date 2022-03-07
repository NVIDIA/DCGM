/*
 * Copyright(c) 2022, NVIDIA CORPORATION.   All rights reserved.
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
#include "Arguments.h"

#include "DcgmLogging.h"
#include "dcgm_fields_internal.h"

#include <map>
#include <string>
#include <vector>

namespace DcgmNs::ProfTester
{
dcgmReturn_t ArgumentSet_t::ProcessIntegerList(std::string argList,
                                               std::shared_ptr<Arguments_t> arguments,
                                               bool useDefaults,
                                               dcgmReturn_t (*callback)(ArgumentSet_t &argumentSet,
                                                                        std::shared_ptr<Arguments_t> arguments,
                                                                        bool useDefaults,
                                                                        unsigned int arg))
{
    unsigned int argument { 0 };
    bool parsing { false };
    dcgmReturn_t rv;

    for (unsigned int i = 0; i < argList.length(); i++)
    {
        unsigned char c = argList[i];

        if ((c >= '0') && (c <= '9'))
        {
            argument *= 10;
            argument += c - '0';
            parsing = true;
        }
        else if (c == ',')
        {
            rv = callback(*this, arguments, useDefaults, argument);

            if (rv != DCGM_ST_OK)
            {
                return rv;
            }

            argument = 0;
            parsing  = false;
        }
        else
        {
            DCGM_LOG_ERROR << "Arguments -- bad numeric character '" << c << "'";

            return DCGM_ST_BADPARAM;
        }
    }

    return parsing ? callback(*this, arguments, useDefaults, argument) : DCGM_ST_OK;
}


dcgmReturn_t ArgumentSet_t::ProcessStringList(std::string argList,
                                              std::shared_ptr<Arguments_t> arguments,
                                              bool useDefaults,
                                              dcgmReturn_t (*callback)(ArgumentSet_t &argumentSet,
                                                                       std::shared_ptr<Arguments_t> arguments,
                                                                       bool useDefaults,
                                                                       const std::string &arg))

{
    std::string argument;
    unsigned int pos { 0 };
    bool parsing { false };
    dcgmReturn_t rv { DCGM_ST_OK };

    for (unsigned int i = 0; i < argList.length(); i++)
    {
        if (argList[i] == ',')
        {
            argument = std::string(argList, pos, i - pos);
            rv       = (*callback)(*this, arguments, useDefaults, argument);

            if (rv != DCGM_ST_OK)
            {
                return rv;
            }

            pos     = i + 1;
            parsing = false;
        }
        else
        {
            parsing = true;
        }
    }

    if (parsing)
    {
        argument = std::string(argList, pos, argList.length() - pos);
        rv       = (*callback)(*this, arguments, useDefaults, argument);
    }

    return rv;
}


dcgmReturn_t ArgumentSet_t::ProcessFieldIds(void)
{
    auto arguments = std::make_shared<Arguments_t>();

    arguments->m_parameters.m_waitToCheck       = m_waitToCheck.Value();
    arguments->m_parameters.m_maxGpusInParallel = m_maxGpusInParallel.Value();
    arguments->m_parameters.m_duration          = m_duration.Value();
    arguments->m_parameters.m_reportInterval    = m_reportInterval.Value();
    arguments->m_parameters.m_syncCount         = m_syncCount.Value();
    arguments->m_parameters.m_targetMaxValue    = m_targetMaxValue.Value();
    arguments->m_parameters.m_noDcgmValidation  = m_noDcgmValidation.Value();
    arguments->m_parameters.m_dvsOutput         = m_dvsOutput.Value();

    double minValue;
    double maxValue;
    bool useDefaults { false };

    if (!m_minValue.IsDefault())
    {
        minValue = m_minValue.Value();

        if (!m_maxValue.IsDefault())
        {
            maxValue = m_maxValue.Value();
        }
        else if (!m_percentTolerance.IsDefault())
        {
            maxValue
                = minValue * (1.0 + m_percentTolerance.Value() / 100.0) / (1.0 - m_percentTolerance.Value() / 100.0);
        }
        else if (!m_absoluteTolerance.IsDefault())
        {
            maxValue = minValue + m_absoluteTolerance.Value() * 2.0;
        }
        else
        {
            maxValue = minValue;
        }
    }
    else if (!m_maxValue.IsDefault())
    {
        maxValue = m_maxValue.Value();

        if (!m_percentTolerance.IsDefault())
        {
            minValue
                = maxValue * (1.0 - m_percentTolerance.Value() / 100.0) / (1.0 + m_percentTolerance.Value() / 100.0);
        }
        else if (!m_absoluteTolerance.IsDefault())
        {
            minValue = maxValue - m_absoluteTolerance.Value() * 2.0;
        }
        else
        {
            minValue = maxValue;
        }
    }
    else if (!m_matchValue.IsDefault())
    {
        if (!m_percentTolerance.IsDefault())
        {
            minValue = m_matchValue.Value() * (1.0 - m_percentTolerance.Value() / 100.0);

            maxValue = m_matchValue.Value() * (1.0 + m_percentTolerance.Value() / 100.0);
        }
        else if (!m_absoluteTolerance.IsDefault())
        {
            minValue = m_matchValue.Value() + m_absoluteTolerance.Value();
            maxValue = m_matchValue.Value() - m_absoluteTolerance.Value();
        }
        else
        {
            minValue = m_matchValue.Value();
            maxValue = m_matchValue.Value();
        }
    }
    else
    {
        useDefaults = true;
    }

    if (!useDefaults)
    {
        arguments->m_parameters.m_minValue = minValue;
        arguments->m_parameters.m_maxValue = maxValue;
    }

    if (!m_gpuIdString.IsDefault()) // Some were specified: we need to get the list.
    {
        ProcessIntegerList(m_gpuIdString.Value(),
                           arguments,
                           false,
                           [](ArgumentSet_t &self,
                              std::shared_ptr<Arguments_t> arguments,
                              bool useDefaults,
                              unsigned int gpuId) mutable -> dcgmReturn_t {
                               arguments->m_gpuIds.push_back(gpuId);

                               return DCGM_ST_OK;
                           });
    }

    ProcessStringList(m_modeString.Value(),
                      arguments,
                      false,
                      [](ArgumentSet_t &self,
                         std::shared_ptr<Arguments_t> arguments,
                         bool useDefaults,
                         const std::string &mode) mutable -> dcgmReturn_t {
                          bool setting { true };
                          unsigned int pos { 0 };

                          if (mode.compare(pos, 2, "no") == 0)
                          {
                              setting = false;
                              pos += 2;
                          }

                          if (mode.compare(pos, std::string::npos, "generateload") == 0)
                          {
                              arguments->m_parameters.m_generate = setting;
                          }
                          else if (mode.compare(pos, std::string::npos, "report") == 0)
                          {
                              arguments->m_parameters.m_report = setting;
                          }
                          else if (mode.compare(pos, std::string::npos, "validate") == 0)
                          {
                              arguments->m_parameters.m_validate = setting;
                          }
                          else if (mode.compare(pos, std::string::npos, "fast") == 0)
                          {
                              arguments->m_parameters.m_fast = setting;
                          }
                          else
                          {
                              DCGM_LOG_ERROR << "Arguments -- bad mode " << mode;

                              return DCGM_ST_BADPARAM;
                          }

                          return DCGM_ST_OK;
                      });

    arguments->m_parameters.m_logFile = m_logFileString.Value();

    arguments->m_parameters.m_logLevel
        = DcgmLogging::dcgmSeverityFromString(m_logLevelString.Value().c_str(), DcgmLoggingSeverityInfo);

    bool limitedFields { false };
    unsigned int firstFieldId { DCGM_FI_PROF_FIRST_ID };
    unsigned int lastFieldId { DCGM_FI_PROF_LAST_ID };

    if (m_lastFieldIds == std::string("all"))
    {
        limitedFields = true;
    }
    else if (m_lastFieldIds == std::string("nvlink"))
    {
        limitedFields = true;

        firstFieldId = DCGM_FI_PROF_NVLINK_TX_BYTES;
        lastFieldId  = DCGM_FI_PROF_NVLINK_RX_BYTES;
    }
    else if (m_lastFieldIds == std::string("nonvlink"))
    {
        limitedFields = true;

        lastFieldId = DCGM_FI_PROF_PCIE_RX_BYTES;
    }

    if (!m_percentTolerance.IsDefault())
    {
        arguments->m_parameters.m_percentTolerance = true;
        arguments->m_parameters.m_tolerance        = m_percentTolerance.Value();
    }
    else if (!m_absoluteTolerance.IsDefault())
    {
        arguments->m_parameters.m_percentTolerance = false;
        arguments->m_parameters.m_tolerance        = m_absoluteTolerance.Value();
    }
    else
    { // hard default.
        arguments->m_parameters.m_percentTolerance = true;
        arguments->m_parameters.m_tolerance        = 10.0;
    }

    if (limitedFields)
    {
        for (unsigned int fieldId = firstFieldId; fieldId <= lastFieldId; fieldId++)
        {
            if (useDefaults)
            {
                if (m_defaults.find(fieldId) != m_defaults.end())
                {
                    arguments->m_parameters.m_minValue   = m_defaults[fieldId].m_min;
                    arguments->m_parameters.m_maxValue   = m_defaults[fieldId].m_max;
                    arguments->m_parameters.m_valueValid = true;
                }
                else
                {
                    arguments->m_parameters.m_valueValid = false;
                }
            }
            else
            {
                arguments->m_parameters.m_valueValid = false;
            }

            arguments->m_parameters.m_fieldId = fieldId;
            m_arguments.push_back(arguments);

            // We make a copy for the next arguments with a different field.
            if (fieldId != lastFieldId)
            {
                auto arguments2 = std::make_shared<Arguments_t>();

                *arguments2 = *arguments;

                // Just make these consistent.
                if (arguments->m_parameters.m_valueValid)
                {
                    arguments->m_parameters.m_percentTolerance = false;
                }

                arguments = arguments2;
            }
        }
    }
    else
    {
        return ProcessIntegerList(m_lastFieldIds,
                                  arguments,
                                  useDefaults,
                                  [](ArgumentSet_t &self,
                                     std::shared_ptr<Arguments_t> arguments,
                                     bool useDefaults,
                                     unsigned int fieldId) mutable -> dcgmReturn_t {
                                      if ((fieldId < DCGM_FI_PROF_FIRST_ID) || (fieldId > DCGM_FI_PROF_LAST_ID))
                                      {
                                          DCGM_LOG_ERROR << "Arguments -- bad fieldId " << fieldId;

                                          return DCGM_ST_BADPARAM;
                                      }

                                      if (useDefaults)
                                      {
                                          if (self.m_defaults.find(fieldId) != self.m_defaults.end())
                                          {
                                              arguments->m_parameters.m_minValue   = self.m_defaults[fieldId].m_min;
                                              arguments->m_parameters.m_maxValue   = self.m_defaults[fieldId].m_max;
                                              arguments->m_parameters.m_valueValid = true;
                                          }
                                          else
                                          {
                                              arguments->m_parameters.m_valueValid = false;
                                          }
                                      }
                                      else
                                      {
                                          arguments->m_parameters.m_valueValid = false;
                                      }

                                      /**
                                       * We make a copy of the arguments with
                                       * each different field.
                                       */
                                      auto arguments2                    = std::make_shared<Arguments_t>();
                                      *arguments2                        = *arguments;
                                      arguments2->m_parameters.m_fieldId = fieldId;

                                      // Just make these consistent.
                                      if (arguments2->m_parameters.m_valueValid)
                                      {
                                          arguments2->m_parameters.m_percentTolerance = false;
                                      }

                                      self.m_arguments.push_back(arguments2);

                                      return DCGM_ST_OK;
                                  });
    }

    return DCGM_ST_OK;
}


void ArgumentSet_t::AddDefault(unsigned int fieldId, ValueRange_t valueRange)
{
    m_defaults[fieldId] = valueRange;
}


dcgmReturn_t ArgumentSet_t::Parse(int argc, char *argv[])
{
    unsigned int count;
    for (unsigned int argPos = 0; argPos < (argc - 1); argPos += count)
    {
        count = 2; // Presume a Value argument

        /**
         * This long and painful test is to see if we have a non-valued (flag)
         * argument, and to parse it as such. This is because TCLAP does not
         * offer a means to throw an exception on parse error and let the caller
         * decide how to recover.
         */
        if (argPos == (argc - 2))
        {
            count = 1;
        }
        else if (argv[argPos + 1][0] == '-')
        {
            if (argv[argPos + 1][1] == '-')
            {
                if (m_longMap.find(std::string(argv[argPos + 1] + 2)) != m_longMap.end())
                {
                    count = 1;
                }
            }
            else if (m_shortMap.find(argv[argPos + 1][1]) != m_shortMap.end())
            {
                count = 1;
            }
        }

        /**
         * We mangle and restore this to get the command line displayed on
         * error.
         */

        auto argPtr = argv[argPos];

        try
        {
            argv[argPos] = argv[0];
            m_cmd.parse(count + 1, argv + argPos);

            /**
             * We need to reset these so that additional parser calls work
             * for arguments already parsed.
             *
             * Currently, the log file can only be set once, so subsequent
             * changes are ignored (and this includes not being able to set it
             * after the first set of field IDs are specified, as the first
             * set will have a default of dcgnproftester.log if not explicitly
             * set -- so you have to set any non-default value BEFORE the first
             * set of field IDs is specified). However, this will only cause
             * a DEBUG log to be made. Eventually, changing the log file between
             * field ID sets may be supported.
             */

            m_waitToCheck.ArgReset();
            m_maxGpusInParallel.ArgReset();
            m_percentTolerance.ArgReset();
            m_absoluteTolerance.ArgReset();
            m_minValue.ArgReset();
            m_matchValue.ArgReset();
            m_maxValue.ArgReset();
            m_duration.ArgReset();
            m_reportInterval.ArgReset();
            m_targetMaxValue.ArgReset();
            m_noDcgmValidation.ArgReset();
            m_dvsOutput.ArgReset();
            m_fieldIds.ArgReset();
            m_gpuIdString.ArgReset();
            m_modeString.ArgReset();
            m_reset.ArgReset();
            m_logFileString.ArgReset();
            m_logLevelString.ArgReset();
            m_configFile.ArgReset();
        }
        catch (TCLAP::ArgException const &ex)
        {
            argv[argPos] = argPtr;

            DCGM_LOG_ERROR << "Arguments -- " << ex.argId() << " " << ex.error();

            return DCGM_ST_BADPARAM;
        }

        argv[argPos] = argPtr;
    }

    // This handles the case if we did not specify -t <fieldId> at all.

    if (!m_snapshotReady)
    {
        m_lastFieldIds = "all";
    }

    return ProcessFieldIds();
}

dcgmReturn_t ArgumentSet_t::Process(std::function<dcgmReturn_t(std::shared_ptr<Arguments_t> arguments)> processor)
{
    dcgmReturn_t retVal { DCGM_ST_OK };

    for (auto arguments : m_arguments)
    {
        dcgmReturn_t rv = processor(arguments);

        if (rv != DCGM_ST_OK)
        {
            retVal = DCGM_ST_GENERIC_ERROR;
        }
    }

    return retVal;
}

} // namespace DcgmNs::ProfTester
