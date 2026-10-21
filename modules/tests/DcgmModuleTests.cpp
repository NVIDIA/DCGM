/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

#include <DcgmModule.h>

#include <catch2/catch_test_macros.hpp>

#include <stdexcept>

namespace
{
class FakeModule : public DcgmModule
{
public:
    explicit FakeModule(dcgmReturn_t ret)
        : m_ret(ret)
    {}

    dcgmReturn_t ProcessMessage(dcgm_module_command_header_t *moduleCommand) override
    {
        m_lastCommand = moduleCommand;
        return m_ret;
    }

    dcgm_module_command_header_t *m_lastCommand = nullptr;

private:
    dcgmReturn_t m_ret;
};

class ThrowingModule : public DcgmModule
{
public:
    enum class ThrowType
    {
        RuntimeError,
        StandardException,
        Unknown,
    };

    explicit ThrowingModule(ThrowType throwType)
        : m_throwType(throwType)
    {}

    dcgmReturn_t ProcessMessage(dcgm_module_command_header_t *) override
    {
        switch (m_throwType)
        {
            case ThrowType::RuntimeError:
                throw std::runtime_error("runtime error");
            case ThrowType::StandardException:
                throw std::logic_error("logic error");
            case ThrowType::Unknown:
                throw 1;
        }

        return DCGM_ST_OK;
    }

private:
    ThrowType m_throwType;
};
} // namespace

TEST_CASE("DcgmModule::CheckVersion validates command versions")
{
    dcgm_module_command_header_t header {};
    header.version = 0x1234;

    WHEN("the command pointer is null")
    {
        auto const result = DcgmModule::CheckVersion(nullptr, header.version);

        THEN("bad parameter is returned")
        {
            CHECK(result == DCGM_ST_BADPARAM);
        }
    }

    WHEN("the command version does not match the expected version")
    {
        auto const result = DcgmModule::CheckVersion(&header, header.version + 1);

        THEN("version mismatch is returned")
        {
            CHECK(result == DCGM_ST_VER_MISMATCH);
        }
    }

    WHEN("the command version matches the expected version")
    {
        auto const result = DcgmModule::CheckVersion(&header, header.version);

        THEN("success is returned")
        {
            CHECK(result == DCGM_ST_OK);
        }
    }
}

TEST_CASE("PassMessageToModule rejects bad inputs")
{
    dcgm_module_command_header_t header {};
    FakeModule module(DCGM_ST_OK);

    WHEN("the module pointer is null")
    {
        auto const result = PassMessageToModule(nullptr, &header);

        THEN("bad parameter is returned")
        {
            CHECK(result == DCGM_ST_BADPARAM);
        }
    }

    WHEN("the command pointer is null")
    {
        auto const result = PassMessageToModule(&module, nullptr);

        THEN("bad parameter is returned")
        {
            CHECK(result == DCGM_ST_BADPARAM);
        }
    }
}

TEST_CASE("PassMessageToModule forwards successful messages")
{
    dcgm_module_command_header_t header {};
    FakeModule module(DCGM_ST_NOT_SUPPORTED);

    WHEN("a valid module and command are passed")
    {
        auto const result = PassMessageToModule(&module, &header);

        THEN("the module result is returned and the command is forwarded")
        {
            CHECK(result == DCGM_ST_NOT_SUPPORTED);
            CHECK(module.m_lastCommand == &header);
        }
    }
}

TEST_CASE("PassMessageToModule converts module exceptions to bad parameter")
{
    dcgm_module_command_header_t header {};

    WHEN("the module throws a runtime error")
    {
        ThrowingModule module(ThrowingModule::ThrowType::RuntimeError);
        auto const result = PassMessageToModule(&module, &header);

        THEN("bad parameter is returned")
        {
            CHECK(result == DCGM_ST_BADPARAM);
        }
    }

    WHEN("the module throws a standard exception")
    {
        ThrowingModule module(ThrowingModule::ThrowType::StandardException);
        auto const result = PassMessageToModule(&module, &header);

        THEN("bad parameter is returned")
        {
            CHECK(result == DCGM_ST_BADPARAM);
        }
    }

    WHEN("the module throws an unknown exception")
    {
        ThrowingModule module(ThrowingModule::ThrowType::Unknown);
        auto const result = PassMessageToModule(&module, &header);

        THEN("bad parameter is returned")
        {
            CHECK(result == DCGM_ST_BADPARAM);
        }
    }
}
