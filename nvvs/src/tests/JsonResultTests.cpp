/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
#include <catch2/catch.hpp>

#include <JsonResult.hpp>

TEST_CASE("JsonResult: JsonSerialize Info")
{
    using namespace DcgmNs::Nvvs::Json;
    using namespace DcgmNs::JsonSerialize;
    Info const info {
        .messages = { "message1", "message2" },
    };

    auto json = Serialize(info);

    auto const info2 = Deserialize<Info>(json);
    REQUIRE(info == info2);
}

TEST_CASE("JsonResult: JsonSerialize Warning")
{
    using namespace DcgmNs::Nvvs::Json;
    using namespace DcgmNs::JsonSerialize;
    Warning const warning {
        .message        = "message",
        .error_code     = 1,
        .error_category = 1,
        .error_severity = 1,
    };

    auto json = Serialize(warning);

    auto const warning2 = Deserialize<Warning>(json);
    REQUIRE(warning == warning2);
}

TEST_CASE("JsonResult: JsonSerialize GpuIds")
{
    using namespace DcgmNs::Nvvs::Json;
    using namespace DcgmNs::JsonSerialize;
    GpuIds const gpuIds {
        .ids = { 1, 2, 3 },
    };

    auto json = Serialize(gpuIds);
    REQUIRE(json.asString() == "1,2,3");

    auto const gpuIds2 = Deserialize<GpuIds>(json);
    REQUIRE(gpuIds == gpuIds2);
}

TEST_CASE("JsonResult: JsonSerialize Single Result")
{
    using namespace DcgmNs::Nvvs::Json;
    using namespace DcgmNs::JsonSerialize;

    Result const result {
        .gpuIds   = { .ids = { 1, 2, 3 } },
        .status   = { .result = NVVS_RESULT_PASS },
        .warnings = std::vector<Warning> { {
                                               .message        = "message",
                                               .error_code     = 1,
                                               .error_category = 1,
                                               .error_severity = 1,
                                           },
                                           {
                                               .message        = "message2",
                                               .error_code     = 2,
                                               .error_category = 1,
                                               .error_severity = 1,
                                           } },
        .info     = Info { .messages = { "message1", "message2" } },
        .auxData  = ::Json::Value::nullSingleton(),
    };

    auto json = Serialize(result);

    auto const result2 = Deserialize<Result>(json);
    REQUIRE(result == result2);
}

static nvvsPluginResult_t RandomPluginResult()
{
    return static_cast<nvvsPluginResult_t>(rand() % 3);
}

DcgmNs::Nvvs::Json::Result GenerateRandomResult()
{
    using namespace DcgmNs::Nvvs::Json;
    using namespace DcgmNs::JsonSerialize;

    Result result {
        .gpuIds   = { .ids = {} },
        .status   = { .result = RandomPluginResult() },
        .warnings = std::vector<Warning> { {
                                               .message        = "warn message 1",
                                               .error_code     = rand(),
                                               .error_category = rand(),
                                               .error_severity = rand(),
                                           },
                                           {
                                               .message        = "warn message 2",
                                               .error_code     = rand(),
                                               .error_category = rand(),
                                               .error_severity = rand(),
                                           } },
        .info     = Info { .messages = { "info message 1", "info message 2" } },
        .auxData  = ::Json::Value::nullSingleton(),
    };

    for (int i = 0; i < 10; i++)
    {
        result.gpuIds.ids.insert(rand() % 100);
    }

    return result;
}

DcgmNs::Nvvs::Json::Test GenerateRandomTest()
{
    using namespace DcgmNs::Nvvs::Json;
    using namespace DcgmNs::JsonSerialize;

    Test test {
        .name    = "test name",
        .results = { GenerateRandomResult(), GenerateRandomResult(), GenerateRandomResult() },
    };

    return test;
}

TEST_CASE("JsonResult: JsonSerialize Single Test")
{
    using namespace DcgmNs::Nvvs::Json;
    using namespace DcgmNs::JsonSerialize;

    auto const test = GenerateRandomTest();

    auto json = Serialize(test);

    auto const test2 = Deserialize<Test>(json);
    REQUIRE(test == test2);
    REQUIRE_FALSE(test < test2);
    REQUIRE_FALSE(test > test2);
}


TEST_CASE("JsonResult: JsonSerialize Single Category")
{
    using namespace DcgmNs::Nvvs::Json;
    using namespace DcgmNs::JsonSerialize;

    Category const category =
    {
        .category  = "software",
        .tests = {
            GenerateRandomTest(),
            GenerateRandomTest(),
            GenerateRandomTest(),
        },
    };

    auto json = Serialize(category);

    auto const category2 = Deserialize<Category>(json);
    REQUIRE(category == category2);
    REQUIRE_FALSE(category < category2);
    REQUIRE_FALSE(category > category2);
}

TEST_CASE("JsonResult: JsonSerialize DiagnosticResults")
{
    using namespace DcgmNs::Nvvs::Json;
    using namespace DcgmNs::JsonSerialize;

    DiagnosticResults const diagnosticResults {
        .version       = "3.1.2",
        .runtimeError  = "Runtime error message",
        .warning       = "Deprecation message",
        .driverVersion = "525.75",
        .categories = std::vector<Category>{
            {
                .category  = "software",
                .tests = {
                    GenerateRandomTest(),
                    GenerateRandomTest(),
                    GenerateRandomTest(),
                },
            },
            {
                .category  = "hardware",
                .tests = {
                    GenerateRandomTest(),
                    GenerateRandomTest(),
                    GenerateRandomTest(),
                },
            },
        },
        .devIds = std::nullopt,
        .devSerials = std::nullopt,
        .errorCode = NVVS_ST_GENERIC_ERROR,
    };

    auto json = Serialize(diagnosticResults);

    auto const diagnosticResults2 = Deserialize<DiagnosticResults>(json);
    REQUIRE(diagnosticResults == diagnosticResults2);
}

TEST_CASE("JsonResult: JsonSerialize Diagnostic Runtime Error")
{
    using namespace DcgmNs::Nvvs::Json;
    using namespace DcgmNs::JsonSerialize;

    DiagnosticResults const diagnosticResults {
        .version       = "1.7.1",
        .runtimeError  = "runtime error",
        .warning       = std::nullopt,
        .driverVersion = "525.75",
        .categories    = std::nullopt,
        .devIds        = std::nullopt,
        .devSerials    = std::nullopt,
        .errorCode     = NVVS_ST_GENERIC_ERROR,
    };

    auto json = Serialize(diagnosticResults);

    auto const diagnosticResults2 = Deserialize<DiagnosticResults>(json);
    REQUIRE(diagnosticResults == diagnosticResults2);
}

TEST_CASE("JsonResult: Bad Diagnostic Result")
{
    using namespace DcgmNs::Nvvs::Json;
    using namespace DcgmNs::JsonSerialize;

    DiagnosticResults const diagnosticResults {
        .version       = "3.2.0",
        .runtimeError  = std::nullopt,
        .warning       = std::nullopt,
        .driverVersion = "525.75",
        .categories    = std::nullopt,
        .devIds        = std::nullopt,
        .devSerials    = std::nullopt,
        .errorCode     = std::nullopt,
    };

    auto json = Serialize(diagnosticResults);

    auto const diagnosticResults2 = TryDeserialize<DiagnosticResults>(json);
    REQUIRE_FALSE(diagnosticResults2.has_value());
}

TEST_CASE("JsonResult: Deserialize real output")
{
    char const jsonStr[] = R"(
	WARNING: THIS SHOULD NOT HAPPEN
	{
	"DCGM GPU Diagnostic" :
	{
		"test_categories" :
		[
			{
				"category" : "Deployment",
				"tests" :
				[
					{
						"name" : "Denylist",
						"results" :
						[
							{
								"gpu_ids" : "0,1,2,3,4,5,6,7",
								"status" : "PASS"
							}
						]
					},
					null,
					{
						"name" : "NVML Library",
						"results" :
						[
							{
								"gpu_ids" : "0,1,2,3,4,5,6,7",
								"status" : "PASS"
							}
						]
					},
					null,
					{
						"name" : "CUDA Main Library",
						"results" :
						[
							{
								"gpu_ids" : "0,1,2,3,4,5,6,7",
								"status" : "PASS"
							}
						]
					},
					null,
					{
						"name" : "Permissions and OS-related Blocks",
						"results" :
						[
							{
								"gpu_ids" : "0,1,2,3,4,5,6,7",
								"status" : "PASS"
							}
						]
					},
					null,
					{
						"name" : "Persistence Mode",
						"results" :
						[
							{
								"gpu_ids" : "0,1,2,3,4,5,6,7",
								"status" : "PASS"
							}
						]
					},
					null,
					{
						"name" : "Environmental Variables",
						"results" :
						[
							{
								"gpu_ids" : "0,1,2,3,4,5,6,7",
								"status" : "PASS"
							}
						]
					},
					null,
					{
						"name" : "Page Retirement/Row Remap",
						"results" :
						[
							{
								"gpu_ids" : "0,1,2,3,4,5,6,7",
								"status" : "PASS"
							}
						]
					},
					null,
					{
						"name" : "Graphics Processes",
						"results" :
						[
							{
								"gpu_ids" : "0,1,2,3,4,5,6,7",
								"status" : "PASS"
							}
						]
					},
					null,
					{
						"name" : "Inforom",
						"results" :
						[
							{
								"gpu_ids" : "0,1,2,3,4,5,6,7",
								"status" : "PASS"
							}
						]
					}
				]
			},
			{
				"category" : "Custom",
				"tests" :
				[
					{
						"name" : "software",
						"results" :
						[
							{
								"gpu_ids" : 0,
								"status" : "PASS"
							},
							{
								"gpu_ids" : 1,
								"status" : "PASS"
							},
							{
								"gpu_ids" : 2,
								"status" : "PASS"
							},
							{
								"gpu_ids" : 3,
								"status" : "PASS"
							},
							{
								"gpu_ids" : 4,
								"status" : "PASS"
							},
							{
								"gpu_ids" : 5,
								"status" : "PASS"
							},
							{
								"gpu_ids" : 6,
								"status" : "PASS"
							},
							{
								"gpu_ids" : 7,
								"status" : "PASS"
							}
						]
					}
				]
			}
		],
		"version" : "3.1.0"
	}
}
)";

    using namespace DcgmNs::Nvvs::Json;
    using namespace DcgmNs::JsonSerialize;

    std::string_view json = std::string_view { jsonStr, sizeof(jsonStr) };
    json.remove_prefix(std::min(json.find_first_of('{'), json.size()));
    REQUIRE(json[0] == '{');

    DiagnosticResults results = Deserialize<DiagnosticResults>(json);
    REQUIRE(results.version == "3.1.0");
}
