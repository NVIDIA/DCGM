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
 * CommandOutputController.h
 *
 *  Created on: Oct 23, 2015
 *      Author: chris
 */

#ifndef COMMANDOUTPUTCONTROLLER_H_
#define COMMANDOUTPUTCONTROLLER_H_

#include <DcgmLogging.h>
#include <dcgm_structs.h>
#include <iostream>
#include <vector>

typedef struct
{
    std::string tag; //!<  Tag to be parsed for and replaced with value
    std::string val; //!<  Value to replaced parsed tag
} dcgmDisplayParameter_t;

class CommandOutputController
{
public:
    CommandOutputController()          = default;
    virtual ~CommandOutputController() = default;

    /* Clear all display parameters in memory. This is automatically called from display(). */
    void clearDisplayParameters();

    /* Setter for display stencil. The stencil is the base that will be parsed for tags which
     * will be replaced with values from the display parameters */
    void setDisplayStencil(char const stencil[]);

    /* This displays the current stencil, with all stored display parameter tags swapped for
     * their values  */
    virtual void display();

    /* Add display parameter to list. toReplace is the tag to be parsed for and it is
     * replaced with replacedWith. See helper functions for formatting. */
    template <typename T>
    void addDisplayParameter(std::string toReplace, T replaceWith)
    {
        dcgmDisplayParameter_t value { std::move(toReplace), HelperDisplayValue(replaceWith) };
        displayParams.push_back(std::move(value));
    }

    /*****************************************************************************
     * Helper method to give correct output for 32 bit integers
     *****************************************************************************/
    std::string HelperDisplayValue(int val);

    /*****************************************************************************
     * Helper method to give correct output for 32 bit unsigned integers
     *****************************************************************************/
    std::string HelperDisplayValue(unsigned int val);

    /*****************************************************************************
     * Helper method to give correct output for 64 bit integers
     *****************************************************************************/
    std::string HelperDisplayValue(long long val);

    /*****************************************************************************
     * Helper method to give correct output for doubles
     *****************************************************************************/
    std::string HelperDisplayValue(double val);

    /*****************************************************************************
     * Helper method to give proper output to strings
     *****************************************************************************/
    std::string HelperDisplayValue(std::string val);

    /*****************************************************************************
     * Helper method to give proper output Enabled/Disabled Values
     *****************************************************************************/
    std::string HelperDisplayValue(char const *val);

    /*****************************************************************************
     * Replaces the tag with the information given
     *****************************************************************************/
    static void ReplaceTag(std::string &buff, std::string_view tag, char const *fmt, ...);

private:
    /*****************************************************************************
     * Returns the number of consecutive spaces from start to end
     *****************************************************************************/
    int OnlySpacesBetween(char *start, char *end);

    /*****************************************************************************
     * Removes the tabs and newlines from str
     *****************************************************************************/
    void RemoveTabsAndNewlines(std::string &str);

    std::string displayStencil;
    std::vector<dcgmDisplayParameter_t> displayParams;
};

typedef struct
{
    std::string overrideString;
    short fieldId;
    dcgmReturn_t errorCode;
} dcgmErrorStringOverride_t;

// Class that is used to display GPU errors.
class GPUErrorOutputController : CommandOutputController
{
public:
    GPUErrorOutputController();
    virtual ~GPUErrorOutputController();

    /* Display a preset error display containing all the error codes and error
     * strings given by the error previously added */
    void display();

    /* A handle used to get all of the errors and fill display parameters with
     *  their corresponding information */
    void addError(dcgmStatus_t errHandle);

    /* Adds an string to override an error message in the error output
     * before an error is displayed, the field id and error code are checked
     * and if it matches an override it will use the replacement string*/
    void addErrorStringOverride(short fieldId, dcgmReturn_t errorCode, std::string replacement);

private:
    dcgmStatus_t mErrHandle {};
    std::vector<dcgmErrorStringOverride_t> mStringOverriders;
};

#endif /* COMMANDOUTPUTCONTROLLER_H_ */
