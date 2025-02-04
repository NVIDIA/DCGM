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
#include <string>

const char *verboseFullOutput = R"({
	"nvbandwidth" :
	{
		"CUDA Runtime Version" : 12070,
		"Driver Version" : "565.02",
		"git_version" : "",
		"testcases" :
		[
			{
				"average" : 12.396361291250001,
				"bandwidth_description" : "memcpy CE CPU(row) -> GPU(column) bandwidth (GB/s) ",
				"bandwidth_matrix" :
				[
					[
						"12.3964"
					]
				],
				"max" : 12.396361291250001,
				"min" : 12.396361291250001,
				"name" : "host_to_device_memcpy_ce",
				"status" : "Passed",
				"sum" : 12.396361291250001
			},
			{
				"average" : 13.200093726,
				"bandwidth_description" : "memcpy CE CPU(row) <- GPU(column) bandwidth (GB/s) ",
				"bandwidth_matrix" :
				[
					[
						"13.2001"
					]
				],
				"max" : 13.200093726,
				"min" : 13.200093726,
				"name" : "device_to_host_memcpy_ce",
				"status" : "Passed",
				"sum" : 13.200093726
			},
			{
				"average" : 11.503123744750001,
				"bandwidth_description" : "memcpy CE CPU(row) <-> GPU(column) bandwidth (GB/s) ",
				"bandwidth_matrix" :
				[
					[
						"11.5031"
					]
				],
				"max" : 11.503123744750001,
				"min" : 11.503123744750001,
				"name" : "host_to_device_bidirectional_memcpy_ce",
				"status" : "Passed",
				"sum" : 11.503123744750001
			},
			{
				"average" : 11.32898976375,
				"bandwidth_description" : "memcpy CE CPU(row) <-> GPU(column) bandwidth (GB/s) ",
				"bandwidth_matrix" :
				[
					[
						"11.329"
					]
				],
				"max" : 11.32898976375,
				"min" : 11.32898976375,
				"name" : "device_to_host_bidirectional_memcpy_ce",
				"status" : "Passed",
				"sum" : 11.32898976375
			},
			{
				"name" : "device_to_device_memcpy_read_ce",
				"status" : "Waived"
			},
			{
				"name" : "device_to_device_memcpy_write_ce",
				"status" : "Waived"
			},
			{
				"name" : "device_to_device_bidirectional_memcpy_read_ce",
				"status" : "Waived"
			},
			{
				"name" : "device_to_device_bidirectional_memcpy_write_ce",
				"status" : "Waived"
			},
			{
				"average" : 13.199809800000001,
				"bandwidth_description" : "memcpy CE CPU(row) <- GPU(column) bandwidth (GB/s) ",
				"bandwidth_matrix" :
				[
					[
						"13.1998"
					]
				],
				"max" : 13.199809800000001,
				"min" : 13.199809800000001,
				"name" : "all_to_host_memcpy_ce",
				"status" : "Passed",
				"sum" : 13.199809800000001
			},
			{
				"average" : 11.3292587155,
				"bandwidth_description" : "memcpy CE CPU(row) <-> GPU(column) bandwidth (GB/s) ",
				"bandwidth_matrix" :
				[
					[
						"11.3293"
					]
				],
				"max" : 11.3292587155,
				"min" : 11.3292587155,
				"name" : "all_to_host_bidirectional_memcpy_ce",
				"status" : "Passed",
				"sum" : 11.3292587155
			},
			{
				"average" : 12.39628974875,
				"bandwidth_description" : "memcpy CE CPU(row) -> GPU(column) bandwidth (GB/s) ",
				"bandwidth_matrix" :
				[
					[
						"12.3963"
					]
				],
				"max" : 12.39628974875,
				"min" : 12.39628974875,
				"name" : "host_to_all_memcpy_ce",
				"status" : "Passed",
				"sum" : 12.39628974875
			},
			{
				"average" : 11.503524198000001,
				"bandwidth_description" : "memcpy CE CPU(row) <-> GPU(column) bandwidth (GB/s) ",
				"bandwidth_matrix" :
				[
					[
						"11.5035"
					]
				],
				"max" : 11.503524198000001,
				"min" : 11.503524198000001,
				"name" : "host_to_all_bidirectional_memcpy_ce",
				"status" : "Passed",
				"sum" : 11.503524198000001
			},
			{
				"name" : "all_to_one_write_ce",
				"status" : "Waived"
			},
			{
				"name" : "all_to_one_read_ce",
				"status" : "Waived"
			},
			{
				"name" : "one_to_all_write_ce",
				"status" : "Waived"
			},
			{
				"name" : "one_to_all_read_ce",
				"status" : "Waived"
			},
			{
				"average" : 12.39958829875,
				"bandwidth_description" : "memcpy SM CPU(row) -> GPU(column) bandwidth (GB/s) ",
				"bandwidth_matrix" :
				[
					[
						"12.3996"
					]
				],
				"max" : 12.39958829875,
				"min" : 12.39958829875,
				"name" : "host_to_device_memcpy_sm",
				"status" : "Passed",
				"sum" : 12.39958829875
			},
			{
				"average" : 13.202739736750001,
				"bandwidth_description" : "memcpy SM CPU(row) <- GPU(column) bandwidth (GB/s) ",
				"bandwidth_matrix" :
				[
					[
						"13.2027"
					]
				],
				"max" : 13.202739736750001,
				"min" : 13.202739736750001,
				"name" : "device_to_host_memcpy_sm",
				"status" : "Passed",
				"sum" : 13.202739736750001
			},
			{
				"name" : "device_to_device_memcpy_read_sm",
				"status" : "Waived"
			},
			{
				"name" : "device_to_device_memcpy_write_sm",
				"status" : "Waived"
			},
			{
				"name" : "device_to_device_bidirectional_memcpy_read_sm",
				"status" : "Waived"
			},
			{
				"name" : "device_to_device_bidirectional_memcpy_write_sm",
				"status" : "Waived"
			},
			{
				"average" : 13.202739736750001,
				"bandwidth_description" : "memcpy SM CPU(row) <- GPU(column) bandwidth (GB/s) ",
				"bandwidth_matrix" :
				[
					[
						"13.2027"
					]
				],
				"max" : 13.202739736750001,
				"min" : 13.202739736750001,
				"name" : "all_to_host_memcpy_sm",
				"status" : "Passed",
				"sum" : 13.202739736750001
			},
			{
				"average" : 9.8367895670000003,
				"bandwidth_description" : "memcpy SM CPU(row) <-> GPU(column) bandwidth (GB/s) ",
				"bandwidth_matrix" :
				[
					[
						"9.83679"
					]
				],
				"max" : 9.8367895670000003,
				"min" : 9.8367895670000003,
				"name" : "all_to_host_bidirectional_memcpy_sm",
				"status" : "Passed",
				"sum" : 9.8367895670000003
			},
			{
				"average" : 12.399516219250001,
				"bandwidth_description" : "memcpy SM CPU(row) -> GPU(column) bandwidth (GB/s) ",
				"bandwidth_matrix" :
				[
					[
						"12.3995"
					]
				],
				"max" : 12.399516219250001,
				"min" : 12.399516219250001,
				"name" : "host_to_all_memcpy_sm",
				"status" : "Passed",
				"sum" : 12.399516219250001
			},
			{
				"average" : 10.12349464775,
				"bandwidth_description" : "memcpy SM CPU(row) <-> GPU(column) bandwidth (GB/s) ",
				"bandwidth_matrix" :
				[
					[
						"10.1235"
					]
				],
				"max" : 10.12349464775,
				"min" : 10.12349464775,
				"name" : "host_to_all_bidirectional_memcpy_sm",
				"status" : "Passed",
				"sum" : 10.12349464775
			},
			{
				"name" : "all_to_one_write_sm",
				"status" : "Waived"
			},
			{
				"name" : "all_to_one_read_sm",
				"status" : "Waived"
			},
			{
				"name" : "one_to_all_write_sm",
				"status" : "Waived"
			},
			{
				"name" : "one_to_all_read_sm",
				"status" : "Waived"
			},
			{
				"average" : 649.87858038348088,
				"bandwidth_description" : "memory latency SM CPU(row) <-> GPU(column) (ns) ",
				"bandwidth_matrix" :
				[
					[
						"649.879"
					]
				],
				"max" : 649.87858038348088,
				"min" : 649.87858038348088,
				"name" : "host_device_latency_sm",
				"status" : "Passed",
				"sum" : 649.87858038348088
			},
			{
				"name" : "device_to_device_latency_sm",
				"status" : "Waived"
			}
		],
		"version" : "v0.5"
	}
})";


const char *verboseFullOutputWithErrors
    = R"(WARNING: Failed to acquire log file lock. File is in use by a different instance
WARNING: garbage messages {
WARNING: another garbage messages {{{}}}}}
{
	"nvbandwidth" :
	{
		"CUDA Runtime Version" : 12070,
		"Driver Version" : "565.02",
		"git_version" : "",
		"testcases" :
		[
			{
				"average" : 12.396361291250001,
				"bandwidth_description" : "memcpy CE CPU(row) -> GPU(column) bandwidth (GB/s) ",
				"bandwidth_matrix" :
				[
					[
						"12.3964"
					]
				],
				"max" : 12.396361291250001,
				"min" : 12.396361291250001,
				"name" : "host_to_device_memcpy_ce",
				"status" : "Passed",
				"sum" : 12.396361291250001
			},
			{
				"average" : 13.200093726,
				"bandwidth_description" : "memcpy CE CPU(row) <- GPU(column) bandwidth (GB/s) ",
				"bandwidth_matrix" :
				[
					[
						"13.2001"
					]
				],
				"max" : 13.200093726,
				"min" : 13.200093726,
				"name" : "device_to_host_memcpy_ce",
				"status" : "Passed",
				"sum" : 13.200093726
			},
			{
				"average" : 11.503123744750001,
				"bandwidth_description" : "memcpy CE CPU(row) <-> GPU(column) bandwidth (GB/s) ",
				"bandwidth_matrix" :
				[
					[
						"11.5031"
					]
				],
				"max" : 11.503123744750001,
				"min" : 11.503123744750001,
				"name" : "host_to_device_bidirectional_memcpy_ce",
				"status" : "Passed",
				"sum" : 11.503123744750001
			},
			{
				"average" : 11.32898976375,
				"bandwidth_description" : "memcpy CE CPU(row) <-> GPU(column) bandwidth (GB/s) ",
				"bandwidth_matrix" :
				[
					[
						"11.329"
					]
				],
				"max" : 11.32898976375,
				"min" : 11.32898976375,
				"name" : "device_to_host_bidirectional_memcpy_ce",
				"status" : "Passed",
				"sum" : 11.32898976375
			},
			{
				"name" : "device_to_device_memcpy_read_ce",
				"status" : "Waived"
			},
			{
				"name" : "device_to_device_memcpy_write_ce",
				"status" : "Waived"
			},
			{
				"name" : "device_to_device_bidirectional_memcpy_read_ce",
				"status" : "Waived"
			},
			{
				"name" : "device_to_device_bidirectional_memcpy_write_ce",
				"status" : "Waived"
			},
			{
				"average" : 13.199809800000001,
				"bandwidth_description" : "memcpy CE CPU(row) <- GPU(column) bandwidth (GB/s) ",
				"bandwidth_matrix" :
				[
					[
						"13.1998"
					]
				],
				"max" : 13.199809800000001,
				"min" : 13.199809800000001,
				"name" : "all_to_host_memcpy_ce",
				"status" : "Passed",
				"sum" : 13.199809800000001
			},
			{
				"average" : 11.3292587155,
				"bandwidth_description" : "memcpy CE CPU(row) <-> GPU(column) bandwidth (GB/s) ",
				"bandwidth_matrix" :
				[
					[
						"11.3293"
					]
				],
				"max" : 11.3292587155,
				"min" : 11.3292587155,
				"name" : "all_to_host_bidirectional_memcpy_ce",
				"status" : "Passed",
				"sum" : 11.3292587155
			},
			{
				"average" : 12.39628974875,
				"bandwidth_description" : "memcpy CE CPU(row) -> GPU(column) bandwidth (GB/s) ",
				"bandwidth_matrix" :
				[
					[
						"12.3963"
					]
				],
				"max" : 12.39628974875,
				"min" : 12.39628974875,
				"name" : "host_to_all_memcpy_ce",
				"status" : "Passed",
				"sum" : 12.39628974875
			},
			{
				"average" : 11.503524198000001,
				"bandwidth_description" : "memcpy CE CPU(row) <-> GPU(column) bandwidth (GB/s) ",
				"bandwidth_matrix" :
				[
					[
						"11.5035"
					]
				],
				"max" : 11.503524198000001,
				"min" : 11.503524198000001,
				"name" : "host_to_all_bidirectional_memcpy_ce",
				"status" : "Passed",
				"sum" : 11.503524198000001
			},
			{
				"name" : "all_to_one_write_ce",
				"status" : "Waived"
			},
			{
				"name" : "all_to_one_read_ce",
				"status" : "Waived"
			},
			{
				"name" : "one_to_all_write_ce",
				"status" : "Waived"
			},
			{
				"name" : "one_to_all_read_ce",
				"status" : "Waived"
			},
			{
				"average" : 12.39958829875,
				"bandwidth_description" : "memcpy SM CPU(row) -> GPU(column) bandwidth (GB/s) ",
				"bandwidth_matrix" :
				[
					[
						"12.3996"
					]
				],
				"max" : 12.39958829875,
				"min" : 12.39958829875,
				"name" : "host_to_device_memcpy_sm",
				"status" : "Passed",
				"sum" : 12.39958829875
			},
			{
				"average" : 13.202739736750001,
				"bandwidth_description" : "memcpy SM CPU(row) <- GPU(column) bandwidth (GB/s) ",
				"bandwidth_matrix" :
				[
					[
						"13.2027"
					]
				],
				"max" : 13.202739736750001,
				"min" : 13.202739736750001,
				"name" : "device_to_host_memcpy_sm",
				"status" : "Passed",
				"sum" : 13.202739736750001
			},
			{
				"name" : "device_to_device_memcpy_read_sm",
				"status" : "Waived"
			},
			{
				"name" : "device_to_device_memcpy_write_sm",
				"status" : "Waived"
			},
			{
				"name" : "device_to_device_bidirectional_memcpy_read_sm",
				"status" : "Waived"
			},
			{
				"name" : "device_to_device_bidirectional_memcpy_write_sm",
				"status" : "Waived"
			},
			{
				"average" : 13.202739736750001,
				"bandwidth_description" : "memcpy SM CPU(row) <- GPU(column) bandwidth (GB/s) ",
				"bandwidth_matrix" :
				[
					[
						"13.2027"
					]
				],
				"max" : 13.202739736750001,
				"min" : 13.202739736750001,
				"name" : "all_to_host_memcpy_sm",
				"status" : "Passed",
				"sum" : 13.202739736750001
			},
			{
				"average" : 9.8367895670000003,
				"bandwidth_description" : "memcpy SM CPU(row) <-> GPU(column) bandwidth (GB/s) ",
				"bandwidth_matrix" :
				[
					[
						"9.83679"
					]
				],
				"max" : 9.8367895670000003,
				"min" : 9.8367895670000003,
				"name" : "all_to_host_bidirectional_memcpy_sm",
				"status" : "Passed",
				"sum" : 9.8367895670000003
			},
			{
				"average" : 12.399516219250001,
				"bandwidth_description" : "memcpy SM CPU(row) -> GPU(column) bandwidth (GB/s) ",
				"bandwidth_matrix" :
				[
					[
						"12.3995"
					]
				],
				"max" : 12.399516219250001,
				"min" : 12.399516219250001,
				"name" : "host_to_all_memcpy_sm",
				"status" : "Passed",
				"sum" : 12.399516219250001
			},
			{
				"average" : 10.12349464775,
				"bandwidth_description" : "memcpy SM CPU(row) <-> GPU(column) bandwidth (GB/s) ",
				"bandwidth_matrix" :
				[
					[
						"10.1235"
					]
				],
				"max" : 10.12349464775,
				"min" : 10.12349464775,
				"name" : "host_to_all_bidirectional_memcpy_sm",
				"status" : "Passed",
				"sum" : 10.12349464775
			},
			{
				"name" : "all_to_one_write_sm",
				"status" : "Waived"
			},
			{
				"name" : "all_to_one_read_sm",
				"status" : "Waived"
			},
			{
				"name" : "one_to_all_write_sm",
				"status" : "Waived"
			},
			{
				"name" : "one_to_all_read_sm",
				"status" : "Waived"
			},
			{
				"average" : 649.87858038348088,
				"bandwidth_description" : "memory latency SM CPU(row) <-> GPU(column) (ns) ",
				"bandwidth_matrix" :
				[
					[
						"649.879"
					]
				],
				"max" : 649.87858038348088,
				"min" : 649.87858038348088,
				"name" : "host_device_latency_sm",
				"status" : "Passed",
				"sum" : 649.87858038348088
			},
			{
				"name" : "device_to_device_latency_sm",
				"status" : "Waived"
			}
		],
		"version" : "v0.5"
	}
}
WARNING: suffix warning messages here
WARNING: garbage messages }}}
WARNING: another garbage messages {{{}}}}}
)";