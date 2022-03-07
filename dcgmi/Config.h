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
 * File:   Config.h
 */

#ifndef CONFIG_H
#define CONFIG_H

#include "Command.h"


/**
 * Receiver Class
 */
class Config
{
public:
    Config()                   = default;
    Config(const Config &orig) = default;
    Config(Config &&)          = default;
    Config &operator=(Config const &) = default;
    Config &operator=(Config &&) = default;

    virtual ~Config() = default;

    /*****************************************************************************
     * This method is used to run GetConfig on the host-engine represented by the
     * DCGM handle
     *****************************************************************************/
    dcgmReturn_t RunGetConfig(dcgmHandle_t pNvcmHandle, bool verbose, bool json);

    /*****************************************************************************
     * This method is used to run SetConfig on the host-engine represented by the
     * DCGM handle
     *****************************************************************************/
    dcgmReturn_t RunSetConfig(dcgmHandle_t pNvcmHandle);

    /*****************************************************************************
     * This method is used to Enforce configuration on the host-engine represented
     * by the DCGM handle
     *****************************************************************************/
    dcgmReturn_t RunEnforceConfig(dcgmHandle_t pNvcmHandle);

    /*****************************************************************************
     * This method is used to set args for the Config object
     *****************************************************************************/
    int SetArgs(unsigned int groupId, dcgmConfig_t *pConfigVal);

private:
    /*****************************************************************************
     * Helper method to give proper output to compute mode values
     *****************************************************************************/
    std::string HelperDisplayComputeMode(unsigned int val);
    /*****************************************************************************
     * Helper method to give proper output to current sync boost
     *****************************************************************************/
    std::string HelperDisplayCurrentSyncBoost(unsigned int val);
    /*****************************************************************************
     * Helper method to give proper output to current sync boost
     *****************************************************************************/
    std::string HelperDisplayBool(unsigned int val);

    /*****************************************************************************
     * Helper method returning true if all configurations have the same setting for
     * the member parameter
     *****************************************************************************/
    template <typename TMember>
    bool HelperCheckIfAllTheSameBoost(dcgmConfig_t *configs, TMember member, unsigned int numGpus);
    template <typename TMember>
    bool HelperCheckIfAllTheSameMode(dcgmConfig_t *configs, TMember member, unsigned int numGpus);
    template <typename TMember>
    bool HelperCheckIfAllTheSameClock(dcgmConfig_t *configs, TMember member, unsigned int numGpus);

    bool HelperCheckIfAllTheSamePowerLim(dcgmConfig_t *configs, unsigned int numGpus);

private:
    dcgmGpuGrp_t mGroupId;
    dcgmConfig_t mConfigVal;
};

/*****************************************************************************
 * Define classes to extend commands
 ****************************************************************************/

/**
 * Set Config Invoker class
 */
class SetConfig : public Command
{
public:
    SetConfig(std::string hostname, Config obj);

protected:
    /*****************************************************************************
     * Override the Execute method for Setting configuration
     *****************************************************************************/
    dcgmReturn_t DoExecuteConnected() override;

private:
    Config configObj;
};

/**
 * Get Config Invoker class
 */
class GetConfig : public Command
{
public:
    GetConfig(std::string hostname, Config obj, bool verbose, bool json);

protected:
    /*****************************************************************************
     * Override the Execute method for Getting configuration
     *****************************************************************************/
    dcgmReturn_t DoExecuteConnected() override;

private:
    Config configObj;
    bool verbose;
};


/**
 * Enforce Config Invoker class
 */
class EnforceConfig : public Command
{
public:
    EnforceConfig(std::string hostname, Config obj);

protected:
    /*****************************************************************************
     * Override the Execute method to Enforce configuration
     *****************************************************************************/
    dcgmReturn_t DoExecuteConnected() override;

private:
    Config configObj;
};


#endif /* CONFIG_H */
