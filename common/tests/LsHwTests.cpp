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

#include <catch2/catch_all.hpp>

#include <DcgmUtilities.h>
#include <LsHw.h>

class MockRunningUserChecker : public DcgmNs::Utils::RunningUserChecker
{
public:
    bool IsRoot() const override
    {
        return m_isRoot;
    }

    void MockIsRoot(bool isRoot)
    {
        m_isRoot = isRoot;
    }

private:
    bool m_isRoot = true;
};

class MockRunCmdHelper : public DcgmNs::Utils::RunCmdHelper
{
public:
    dcgmReturn_t RunCmdAndGetOutput(std::string const &cmd, std::string &output) const override
    {
        if (!m_mockCmdOutput.contains(cmd))
        {
            return DCGM_ST_GENERIC_ERROR;
        }
        dcgmReturn_t ret;
        tie(ret, output) = m_mockCmdOutput.at(cmd);
        return ret;
    }

    void MockCmdOutput(std::string const &cmd, dcgmReturn_t ret, std::string const &output)
    {
        m_mockCmdOutput[cmd] = { ret, output };
    }

private:
    std::unordered_map<std::string, std::pair<dcgmReturn_t, std::string>> m_mockCmdOutput;
};

std::string lshwMultipleCpusAbridgedValidJson = R""(
{
  "id" : "localhost",
  "class" : "system",
  "claimed" : true,
  "handle" : "DMI:0002",
  "description" : "Rack Mount Chassis",
  "product" : "Grace Hopper x4 P4496 (675-23187-0070-EV1)",
  "vendor" : "NVIDIA",
  "version" : "00",
  "serial" : "1640323000029",
  "width" : 64,
  "configuration" : {
    "boot" : "normal",
    "chassis" : "rackmount",
    "family" : "HGX",
    "sku" : "675-23187-0070-EV1",
    "uuid" : "d05fed57-fe8e-8124-d954-3aae63dc8210"
  },
  "capabilities" : {
    "smbios-3.6.0" : "SMBIOS version 3.6.0",
    "dmi-3.6.0" : "DMI version 3.6.0",
    "smp" : "Symmetric Multi-Processing",
    "sve_default_vector_length" : true,
    "tagged_addr_disabled" : true
  },
  "children" : [    {
      "id" : "core",
      "class" : "bus",
      "claimed" : true,
      "handle" : "DMI:0037",
      "description" : "Motherboard",
      "product" : "UT2.1 DP Chassis",
      "vendor" : "NVIDIA",
      "physid" : "0",
      "version" : "H.1",
      "serial" : "1640523001044",
      "slot" : "Management module",
      "children" : [        {
          "id" : "firmware",
          "class" : "memory",
          "claimed" : true,
          "description" : "BIOS",
          "vendor" : "NVIDIA",
          "physid" : "1",
          "version" : "01.02.01",
          "date" : "20240207",
          "units" : "bytes",
          "size" : 1048576,
          "capacity" : 67108864,
          "capabilities" : {
            "pci" : "PCI bus",
            "pnp" : "Plug-and-Play",
            "upgrade" : "BIOS EEPROM can be upgraded",
            "shadowing" : "BIOS shadowing",
            "cdboot" : "Booting from CD-ROM/DVD",
            "bootselect" : "Selectable boot path",
            "int14serial" : "INT14 serial line control",
            "acpi" : "ACPI",
            "uefi" : "UEFI specification is supported"
          }
        },
        {
          "id" : "cache:0",
          "class" : "memory",
          "claimed" : true,
          "handle" : "DMI:0004",
          "description" : "L1 cache",
          "physid" : "4",
          "slot" : "L1 Instruction Cache",
          "units" : "bytes",
          "size" : 4718592,
          "capacity" : 4718592,
          "configuration" : {
            "level" : "1"
          },
          "capabilities" : {
            "instruction" : "Instruction cache"
          }
        },
        {
          "id" : "cache:15",
          "class" : "memory",
          "claimed" : true,
          "handle" : "DMI:0013",
          "description" : "L3 cache",
          "physid" : "13",
          "slot" : "L3 Cache",
          "units" : "bytes",
          "size" : 119537664,
          "capacity" : 119537664,
          "configuration" : {
            "level" : "3"
          },
          "capabilities" : {
            "unified" : "Unified cache"
          }
        },
        {
          "id" : "cpu:0",
          "class" : "processor",
          "claimed" : true,
          "handle" : "DMI:0014",
          "description" : "CPU",
          "product" : "ARMv8 (900-2G530-0000-000)",
          "vendor" : "NVIDIA",
          "physid" : "14",
          "businfo" : "cpu@0",
          "version" : "Grace A02",
          "serial" : "0x000000017820B1C80400000015FF81C0",
          "slot" : "G1:0.0",
          "units" : "Hz",
          "size" : 4122000000,
          "clock" : 1000000000,
          "configuration" : {
            "cores" : "72",
            "enabledcores" : "72",
            "threads" : "72"
          },
          "capabilities" : {
            "lm" : "64-bit capable",
            "cpufreq" : "CPU Frequency scaling"
          }
        },
        {
          "id" : "cpu:1",
          "class" : "processor",
          "claimed" : true,
          "handle" : "DMI:0015",
          "description" : "CPU",
          "product" : "ARMv8 (900-2G530-0000-000)",
          "vendor" : "NVIDIA",
          "physid" : "15",
          "businfo" : "cpu@1",
          "version" : "Grace A02",
          "serial" : "0x000000017820B1C8040000000A0200C0",
          "slot" : "G1:1.0",
          "units" : "Hz",
          "size" : 2961000000,
          "capacity" : 3357000000,
          "clock" : 1000000000,
          "configuration" : {
            "cores" : "72",
            "enabledcores" : "72",
            "threads" : "72"
          },
          "capabilities" : {
            "lm" : "64-bit capable",
            "cpufreq" : "CPU Frequency scaling"
          }
        },
        {
          "id" : "memory",
          "class" : "memory",
          "claimed" : true,
          "handle" : "DMI:002E",
          "description" : "System Memory",
          "physid" : "2e",
          "slot" : "System board or motherboard",
          "units" : "bytes",
          "size" : 515396075520,
          "configuration" : {
            "errordetection" : "ecc"
          },
          "capabilities" : {
            "ecc" : "Single-bit error-correcting code (ECC)"
          }
        }]
    }]
}  
)"";

std::string lshwSingleCpuAbridgedValidJson = R""(
{
  "id" : "lego-cg1-qs-41",
  "class" : "system",
  "claimed" : true,
  "handle" : "DMI:0002",
  "description" : "Computer",
  "product" : "GH200 P5042 (940-23880-0000-EVT)",
  "vendor" : "NVIDIA",
  "version" : "E.1",
  "serial" : "Unknown",
  "width" : 64,
  "configuration" : {
    "boot" : "normal",
    "sku" : "940-23880-0000-EVT",
    "uuid" : "fb0dec54-1b3e-a522-9f53-cb334c15a331"
  },
  "capabilities" : {
    "smbios-3.6.0" : "SMBIOS version 3.6.0",
    "dmi-3.6.0" : "DMI version 3.6.0",
    "smp" : "Symmetric Multi-Processing",
    "cp15_barrier" : true,
    "setend" : true,
    "sve_default_vector_length" : true,
    "swp" : true,
    "tagged_addr_disabled" : true
  },
  "children" : [    {
      "id" : "core",
      "class" : "bus",
      "claimed" : true,
      "handle" : "DMI:0015",
      "description" : "Motherboard",
      "product" : "P3880",
      "vendor" : "NVIDIA",
      "physid" : "0",
      "version" : "A.2",
      "serial" : "1582823700115",
      "slot" : "Management module",
      "children" : [       
        {
          "id" : "cpu",
          "class" : "processor",
          "claimed" : true,
          "handle" : "DMI:0008",
          "description" : "CPU",
          "product" : "ARMv8 (900-2G530-0000-000)",
          "vendor" : "NVIDIA",
          "physid" : "8",
          "businfo" : "cpu@0",
          "version" : "Grace A01",
          "serial" : "0x000000017820B1C80400000015FF81C0",
          "slot" : "G1:0.0",
          "units" : "Hz",
          "size" : 4275000000,
          "clock" : 1000000000,
          "configuration" : {
            "cores" : "72",
            "enabledcores" : "72",
            "threads" : "72"
          },
          "capabilities" : {
            "lm" : "64-bit capable",
            "cpufreq" : "CPU Frequency scaling"
          }
        },
        {
          "id" : "memory",
          "class" : "memory",
          "claimed" : true,
          "handle" : "DMI:0012",
          "description" : "System Memory",
          "physid" : "12",
          "slot" : "System board or motherboard",
          "units" : "bytes",
          "size" : 617401548800,
          "configuration" : {
            "errordetection" : "ecc"
          },
          "capabilities" : {
            "ecc" : "Single-bit error-correcting code (ECC)"
          }
        }]
    }]
}    
)"";

std::string lshwSingleCpuNoSerialNumberAbridgedValidJson = R""(
{
  "id" : "lego-cg1-qs-41",
  "class" : "system",
  "claimed" : true,
  "handle" : "DMI:0002",
  "description" : "Computer",
  "product" : "GH200 P5042 (940-23880-0000-EVT)",
  "vendor" : "NVIDIA",
  "version" : "E.1",
  "serial" : "Unknown",
  "width" : 64,
  "configuration" : {
    "boot" : "normal",
    "sku" : "940-23880-0000-EVT",
    "uuid" : "fb0dec54-1b3e-a522-9f53-cb334c15a331"
  },
  "children" : [    {
      "id" : "core",
      "class" : "bus",
      "claimed" : true,
      "handle" : "DMI:0015",
      "description" : "Motherboard",
      "product" : "P3880",
      "vendor" : "NVIDIA",
      "physid" : "0",
      "version" : "A.2",
      "serial" : "1582823700115",
      "slot" : "Management module",
      "children" : [      
        {
          "id" : "cpu",
          "class" : "processor",
          "claimed" : true,
          "handle" : "DMI:0008",
          "description" : "CPU",
          "product" : "ARMv8 (900-2G530-0000-000)",
          "vendor" : "NVIDIA",
          "physid" : "8",
          "businfo" : "cpu@0",
          "version" : "Grace A01",
          "slot" : "G1:0.0",
          "units" : "Hz",
          "size" : 4275000000,
          "clock" : 1000000000,
          "configuration" : {
            "cores" : "72",
            "enabledcores" : "72",
            "threads" : "72"
          },
          "capabilities" : {
            "lm" : "64-bit capable",
            "cpufreq" : "CPU Frequency scaling"
          }
        },
        {
          "id" : "memory",
          "class" : "memory",
          "claimed" : true,
          "handle" : "DMI:0012",
          "description" : "System Memory",
          "physid" : "12",
          "slot" : "System board or motherboard",
          "units" : "bytes",
          "size" : 617401548800,
          "configuration" : {
            "errordetection" : "ecc"
          },
          "capabilities" : {
            "ecc" : "Single-bit error-correcting code (ECC)"
          }
        }]
    }]
}    
)"";

std::string lshwSingleNonNvidiaCpuAbridgedValidJson = R""(
{
  "id" : "lego-cg1-qs-41",
  "class" : "system",
  "claimed" : true,
  "handle" : "DMI:0002",
  "description" : "Computer",
  "product" : "GH200 P5042 (940-23880-0000-EVT)",
  "vendor" : "NVIDIA",
  "version" : "E.1",
  "serial" : "Unknown",
  "width" : 64,
  "configuration" : {
    "boot" : "normal",
    "sku" : "940-23880-0000-EVT",
    "uuid" : "fb0dec54-1b3e-a522-9f53-cb334c15a331"
  },
  "capabilities" : {
    "smbios-3.6.0" : "SMBIOS version 3.6.0",
    "dmi-3.6.0" : "DMI version 3.6.0",
    "smp" : "Symmetric Multi-Processing",
    "cp15_barrier" : true,
    "setend" : true,
    "sve_default_vector_length" : true,
    "swp" : true,
    "tagged_addr_disabled" : true
  },
  "children" : [    {
      "id" : "core",
      "class" : "bus",
      "claimed" : true,
      "handle" : "DMI:0015",
      "description" : "Motherboard",
      "product" : "P3880",
      "vendor" : "NVIDIA",
      "physid" : "0",
      "version" : "A.2",
      "serial" : "1582823700115",
      "slot" : "Management module",
      "children" : [       
        {
          "id" : "cpu",
          "class" : "processor",
          "claimed" : true,
          "handle" : "DMI:0008",
          "description" : "CPU",
          "product" : "13th Gen Intel(R) Core(TM) i9-13900K",
          "vendor" : "Intel Corp.",
          "physid" : "8",
          "businfo" : "cpu@0",
          "version" : "Grace A01",
          "serial" : "0x000000017820B1C80400000015FF81C0",
          "slot" : "G1:0.0",
          "units" : "Hz",
          "size" : 4275000000,
          "clock" : 1000000000,
          "configuration" : {
            "cores" : "72",
            "enabledcores" : "72",
            "threads" : "72"
          },
          "capabilities" : {
            "lm" : "64-bit capable",
            "cpufreq" : "CPU Frequency scaling"
          }
        },
        {
          "id" : "memory",
          "class" : "memory",
          "claimed" : true,
          "handle" : "DMI:0012",
          "description" : "System Memory",
          "physid" : "12",
          "slot" : "System board or motherboard",
          "units" : "bytes",
          "size" : 617401548800,
          "configuration" : {
            "errordetection" : "ecc"
          },
          "capabilities" : {
            "ecc" : "Single-bit error-correcting code (ECC)"
          }
        }]
    }]
}    
)"";

std::string lshwIncorrectIdValueTypeJson = R""(
{
  "id" : "lego-cg1-qs-41",
  "class" : "system",
  "claimed" : true,
  "handle" : "DMI:0002",
  "description" : "Computer",
  "product" : "GH200 P5042 (940-23880-0000-EVT)",
  "vendor" : "NVIDIA",
  "version" : "E.1",
  "serial" : "Unknown",
  "width" : 64,
  "configuration" : {
    "boot" : "normal",
    "sku" : "940-23880-0000-EVT",
    "uuid" : "fb0dec54-1b3e-a522-9f53-cb334c15a331"
  },
  "capabilities" : {
    "smbios-3.6.0" : "SMBIOS version 3.6.0",
    "dmi-3.6.0" : "DMI version 3.6.0",
    "smp" : "Symmetric Multi-Processing",
    "cp15_barrier" : true,
    "setend" : true,
    "sve_default_vector_length" : true,
    "swp" : true,
    "tagged_addr_disabled" : true
  },
  "children" : [    {
      "id" : "core",
      "class" : "bus",
      "claimed" : true,
      "handle" : "DMI:0015",
      "description" : "Motherboard",
      "product" : "P3880",
      "vendor" : "NVIDIA",
      "physid" : "0",
      "version" : "A.2",
      "serial" : "1582823700115",
      "slot" : "Management module",
      "children" : [       
        {
          "id" : [],
          "class" : "processor",
          "claimed" : true,
          "handle" : "DMI:0008",
          "description" : "CPU",
          "product" : "ARMv8 (900-2G530-0000-000)",
          "vendor" : "NVIDIA",
          "physid" : "8",
          "businfo" : "cpu@0",
          "version" : "Grace A01",
          "serial" : "0x000000017820B1C80400000015FF81C0",
          "slot" : "G1:0.0",
          "units" : "Hz",
          "size" : 4275000000,
          "clock" : 1000000000,
          "configuration" : {
            "cores" : "72",
            "enabledcores" : "72",
            "threads" : "72"
          },
          "capabilities" : {
            "lm" : "64-bit capable",
            "cpufreq" : "CPU Frequency scaling"
          }
        },
        {
          "id" : "memory",
          "class" : "memory",
          "claimed" : true,
          "handle" : "DMI:0012",
          "description" : "System Memory",
          "physid" : "12",
          "slot" : "System board or motherboard",
          "units" : "bytes",
          "size" : 617401548800,
          "configuration" : {
            "errordetection" : "ecc"
          },
          "capabilities" : {
            "ecc" : "Single-bit error-correcting code (ECC)"
          }
        }]
    }]
}    
)"";

std::string lshwMissingCpuJson = R""(
{
  "id" : "localhost",
  "class" : "system",
  "claimed" : true,
  "handle" : "DMI:0002",
  "description" : "Rack Mount Chassis",
  "product" : "Grace Hopper x4 P4496 (675-23187-0070-EV1)",
  "vendor" : "NVIDIA",
  "version" : "00",
  "serial" : "1640323000029",
  "width" : 64,
  "configuration" : {
    "boot" : "normal",
    "chassis" : "rackmount",
    "family" : "HGX",
    "sku" : "675-23187-0070-EV1",
    "uuid" : "d05fed57-fe8e-8124-d954-3aae63dc8210"
  }
}
)"";

std::string lshwBadSyntaxJson = "oops";

TEST_CASE("LsHw::GetCpuSerials")
{
    SECTION("Non Root")
    {
        std::unique_ptr<MockRunningUserChecker> checker = std::make_unique<MockRunningUserChecker>();
        checker->MockIsRoot(false);

        LsHw lshw;
        lshw.SetChecker(std::move(checker));
        REQUIRE(!lshw.GetCpuSerials().has_value());
    }

    SECTION("Multiple CPUs")
    {
        std::unique_ptr<MockRunningUserChecker> checker = std::make_unique<MockRunningUserChecker>();
        checker->MockIsRoot(true);
        std::unique_ptr<MockRunCmdHelper> runCmdHelper = std::make_unique<MockRunCmdHelper>();
        runCmdHelper->MockCmdOutput("/usr/bin/lshw -json", DCGM_ST_OK, lshwMultipleCpusAbridgedValidJson);

        LsHw lshw;
        lshw.SetChecker(std::move(checker));
        lshw.SetRunCmdHelper(std::move(runCmdHelper));
        auto cpuSerials = lshw.GetCpuSerials();
        REQUIRE(cpuSerials.has_value());
        REQUIRE(cpuSerials->size() == 2);
        REQUIRE(cpuSerials.value()[0] == "0x000000017820B1C80400000015FF81C0");
        REQUIRE(cpuSerials.value()[1] == "0x000000017820B1C8040000000A0200C0");
    }

    SECTION("Single CPU")
    {
        std::unique_ptr<MockRunningUserChecker> checker = std::make_unique<MockRunningUserChecker>();
        checker->MockIsRoot(true);
        std::unique_ptr<MockRunCmdHelper> runCmdHelper = std::make_unique<MockRunCmdHelper>();
        runCmdHelper->MockCmdOutput("/usr/bin/lshw -json", DCGM_ST_OK, lshwSingleCpuAbridgedValidJson);

        LsHw lshw;
        lshw.SetChecker(std::move(checker));
        lshw.SetRunCmdHelper(std::move(runCmdHelper));
        auto cpuSerials = lshw.GetCpuSerials();
        REQUIRE(cpuSerials.has_value());
        REQUIRE(cpuSerials->size() == 1);
        REQUIRE(cpuSerials.value()[0] == "0x000000017820B1C80400000015FF81C0");
    }

    SECTION("Single No-Serial CPU")
    {
        std::unique_ptr<MockRunningUserChecker> checker = std::make_unique<MockRunningUserChecker>();
        checker->MockIsRoot(true);
        std::unique_ptr<MockRunCmdHelper> runCmdHelper = std::make_unique<MockRunCmdHelper>();
        runCmdHelper->MockCmdOutput("/usr/bin/lshw -json", DCGM_ST_OK, lshwSingleCpuNoSerialNumberAbridgedValidJson);

        LsHw lshw;
        lshw.SetChecker(std::move(checker));
        lshw.SetRunCmdHelper(std::move(runCmdHelper));
        auto cpuSerials = lshw.GetCpuSerials();
        REQUIRE(cpuSerials.has_value());
        REQUIRE(cpuSerials->size() == 1);
        REQUIRE(cpuSerials.value()[0].empty());
    }

    SECTION("Single Non-NVIDIA CPU")
    {
        std::unique_ptr<MockRunningUserChecker> checker = std::make_unique<MockRunningUserChecker>();
        checker->MockIsRoot(true);
        std::unique_ptr<MockRunCmdHelper> runCmdHelper = std::make_unique<MockRunCmdHelper>();
        runCmdHelper->MockCmdOutput("/usr/bin/lshw -json", DCGM_ST_OK, lshwSingleNonNvidiaCpuAbridgedValidJson);

        LsHw lshw;
        lshw.SetChecker(std::move(checker));
        lshw.SetRunCmdHelper(std::move(runCmdHelper));
        auto cpuSerials = lshw.GetCpuSerials();
        REQUIRE(cpuSerials.has_value());
        REQUIRE(cpuSerials->size() == 0);
    }

    SECTION("Incorrect Id")
    {
        std::unique_ptr<MockRunningUserChecker> checker = std::make_unique<MockRunningUserChecker>();
        checker->MockIsRoot(true);
        std::unique_ptr<MockRunCmdHelper> runCmdHelper = std::make_unique<MockRunCmdHelper>();
        runCmdHelper->MockCmdOutput("/usr/bin/lshw -json", DCGM_ST_OK, lshwIncorrectIdValueTypeJson);

        LsHw lshw;
        lshw.SetChecker(std::move(checker));
        lshw.SetRunCmdHelper(std::move(runCmdHelper));
        auto cpuSerials = lshw.GetCpuSerials();
        REQUIRE(!cpuSerials.has_value());
    }

    SECTION("Missing CPU")
    {
        std::unique_ptr<MockRunningUserChecker> checker = std::make_unique<MockRunningUserChecker>();
        checker->MockIsRoot(true);
        std::unique_ptr<MockRunCmdHelper> runCmdHelper = std::make_unique<MockRunCmdHelper>();
        runCmdHelper->MockCmdOutput("/usr/bin/lshw -json", DCGM_ST_OK, lshwMissingCpuJson);

        LsHw lshw;
        lshw.SetChecker(std::move(checker));
        lshw.SetRunCmdHelper(std::move(runCmdHelper));
        auto cpuSerials = lshw.GetCpuSerials();
        REQUIRE(cpuSerials.has_value());
        REQUIRE(cpuSerials->size() == 0);
    }

    SECTION("Bad JSON Syntax")
    {
        std::unique_ptr<MockRunningUserChecker> checker = std::make_unique<MockRunningUserChecker>();
        checker->MockIsRoot(true);
        std::unique_ptr<MockRunCmdHelper> runCmdHelper = std::make_unique<MockRunCmdHelper>();
        runCmdHelper->MockCmdOutput("/usr/bin/lshw -json", DCGM_ST_OK, lshwBadSyntaxJson);

        LsHw lshw;
        lshw.SetChecker(std::move(checker));
        lshw.SetRunCmdHelper(std::move(runCmdHelper));
        auto cpuSerials = lshw.GetCpuSerials();
        REQUIRE(!cpuSerials.has_value());
    }
}

TEST_CASE("LsHw::Exec")
{
    SECTION("Will try /usr/sbin/")
    {
        std::unique_ptr<MockRunningUserChecker> checker = std::make_unique<MockRunningUserChecker>();
        checker->MockIsRoot(true);
        std::unique_ptr<MockRunCmdHelper> runCmdHelper = std::make_unique<MockRunCmdHelper>();
        runCmdHelper->MockCmdOutput("/usr/sbin/lshw -json", DCGM_ST_OK, lshwMultipleCpusAbridgedValidJson);

        LsHw lshw;
        lshw.SetChecker(std::move(checker));
        lshw.SetRunCmdHelper(std::move(runCmdHelper));
        auto cpuSerials = lshw.GetCpuSerials();
        REQUIRE(cpuSerials.has_value());
        REQUIRE(cpuSerials->size() == 2);
        REQUIRE(cpuSerials.value()[0] == "0x000000017820B1C80400000015FF81C0");
        REQUIRE(cpuSerials.value()[1] == "0x000000017820B1C8040000000A0200C0");
    }
}