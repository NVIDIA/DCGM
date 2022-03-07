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
#include <iostream>

#include "CommandOutputController.h"
#include "TestCommandOutputController.h"
#include "dcgm_structs.h"

TestCommandOutputController::TestCommandOutputController()
{}

TestCommandOutputController::~TestCommandOutputController()
{}

std::string TestCommandOutputController::GetTag()
{
    return "outputcontroller";
}

int TestCommandOutputController::Run()
{
    int st;
    int numFailed = 0;

    st = TestHelperDisplayValue();
    if (st)
    {
        numFailed++;
        fprintf(stderr, "TestCommandOutputContoller::TestHelperDisplayValue FAILED with %d.\n", st);
    }
    else
        printf("TestCommandOutputContoller::TestHelperDisplayValue PASSED\n");

    return numFailed;
}

int TestCommandOutputController::TestHelperDisplayValue()
{
    CommandOutputController coc;
    int ret = 0;
    char buf[1024];

    snprintf(buf, sizeof(buf), "%s", DCGM_STR_BLANK);
    std::string retd = coc.HelperDisplayValue(buf);
    if (retd != "Not Specified")
    {
        ret = -1;
        fprintf(stderr,
                "'%s' should have been transformed into 'Not Specified', but was '%s'.\n",
                DCGM_STR_BLANK,
                retd.c_str());
    }

    snprintf(buf, sizeof(buf), "%s", DCGM_STR_NOT_FOUND);
    retd = coc.HelperDisplayValue(buf);
    if (retd != "Not Found")
    {
        ret = -1;
        fprintf(stderr,
                "'%s' should have been transformed into 'Not Found', but was '%s'.\n",
                DCGM_STR_NOT_FOUND,
                retd.c_str());
    }

    snprintf(buf, sizeof(buf), "%s", DCGM_STR_NOT_SUPPORTED);
    retd = coc.HelperDisplayValue(buf);
    if (retd != "Not Supported")
    {
        ret = -1;
        fprintf(stderr,
                "'%s' should have been transformed into 'Not Supported', but was '%s'.\n",
                DCGM_STR_NOT_SUPPORTED,
                retd.c_str());
    }

    snprintf(buf, sizeof(buf), "%s", DCGM_STR_NOT_PERMISSIONED);
    retd = coc.HelperDisplayValue(buf);
    if (retd != "Insf. Permission")
    {
        ret = -1;
        fprintf(stderr,
                "'%s' should have been transformed into 'Insf. Permission', but was '%s'.\n",
                DCGM_STR_NOT_PERMISSIONED,
                retd.c_str());
    }

    snprintf(buf, sizeof(buf), "Brando\tSando");
    retd = coc.HelperDisplayValue(buf);
    if (retd != "Brando Sando")
    {
        ret = -1;
        fprintf(stderr, "'%s' should have been transformed into 'Brando Sando', but was '%s'.\n", buf, retd.c_str());
    }

    snprintf(buf, sizeof(buf), "There's weird\nspacing\n\there.");
    retd = coc.HelperDisplayValue(buf);
    if (retd != "There's weird spacing  here.")
    {
        ret = -1;
        fprintf(stderr,
                "'%s' should have been transformed into 'There's weird spacing  here', but was '%s'.\n",
                buf,
                retd.c_str());
    }

    return ret;
}
