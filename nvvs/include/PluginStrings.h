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
#ifndef PLUGINSTRINGS_H
#define PLUGINSTRINGS_H

/*****************************************************************************/
/********************* NOTE **************************************************/
/* All parameters added here need to be added to ParameterValidator.cpp or you
 * will get an error if you attempt to use them. *****************************/
/*****************************************************************************/

/*****************************************************************************/
/* Parameters common for all tests */
#define PS_PLUGIN_NAME        "name"
#define PS_TEST_NAME          "test_name"
#define PS_LOGFILE            "logfile"
#define PS_LOGFILE_TYPE       "logfile_type"
#define PS_SUITE_LEVEL        "suite_level"
#define PS_IGNORE_ERROR_CODES "ignore_error_codes"

#define PS_RUN_IF_GOM_ENABLED "run_if_gom_enabled" /* Should this plugin run if GOM mode is enabled */

/*****************************************************************************/
/* Categories */

#define PLUGIN_CATEGORY_DEPLOYMENT  "Deployment"
#define PLUGIN_CATEGORY_HW          "Hardware"
#define PLUGIN_CATEGORY_STRESS      "Stress"
#define PLUGIN_CATEGORY_INTEGRATION "Integration"

/******************************************************************************
 * SOFTWARE PLUGIN
 *****************************************************************************/
#define SW_STR_DO_TEST             "do_test"
#define SW_STR_REQUIRE_PERSISTENCE "require_persistence_mode"
#define SW_PLUGIN_NAME             "software"
#define SW_PLUGIN_CATEGORY         PLUGIN_CATEGORY_DEPLOYMENT
#define SW_STR_CHECK_FILE_CREATION "check_file_creation"
#define SW_STR_SKIP_DEVICE_TEST    "skip_device_test"

/******************************************************************************
 * PCIE PLUGIN
 *****************************************************************************/
#define PCIE_PLUGIN_NAME     "pcie"
#define PCIE_PLUGIN_CATEGORY PLUGIN_CATEGORY_INTEGRATION

/* Public parameters - we expect users to change these */
#define PCIE_STR_TEST_PINNED              "test_pinned"
#define PCIE_STR_TEST_UNPINNED            "test_unpinned"
#define PCIE_STR_TEST_P2P_ON              "test_p2p_on"
#define PCIE_STR_TEST_P2P_OFF             "test_p2p_off"
#define PCIE_STR_NVSWITCH_NON_FATAL_CHECK "check_non_fatal"
#define PCIE_STR_TEST_BROKEN_P2P          "test_broken_p2p"
#define PCIE_STR_TEST_NVLINK_STATUS       "test_nvlink_status"
#define PCIE_STR_TEST_WITH_GEMM           "test_with_gemm"
#define PCIE_STR_DISABLE_TESTS            "disable_tests"
#define PCIE_STR_AER_THRESHOLD            "aer_threshold" /* max pci aer errors before test fails */
#define PCIE_STR_DONT_BIND_NUMA           "dont_bind_numa"

/* Private parameters */
#define PCIE_STR_IS_ALLOWED "is_allowed" /* Is the busgrind plugin allowed to run? */

/* Private sub-test parameters. These apply to some sub tests and not others */
#define PCIE_STR_INTS_PER_COPY "num_ints_per_copy"
#define PCIE_STR_ITERATIONS    "iterations"

/* Public sub-test parameters. These apply to some sub tests and not others */
#define PCIE_STR_MIN_BANDWIDTH                                                      \
    "min_bandwidth" /* Bandwidth below this in GB/s is                              \
                                                   considered a failure for a given \
                                                   sub test */
#define PCIE_STR_MAX_LATENCY                                                 \
    "max_latency" /* Latency above this in microseconds is                   \
                                            considered a failure for a given \
                                            sub test */
#define PCIE_STR_MIN_PCI_GEN                                                      \
    "min_pci_generation" /* Minimum PCI generation allowed.                       \
                                                   PCI generation below this will \
                                                   cause a sub test failure */
#define PCIE_STR_MIN_PCI_WIDTH                                                    \
    "min_pci_width" /* Minimum PCI width allowed. 16x = 16 etc                    \
                                                PCI width below this will cause a \
                                                sub test failure */

#define PCIE_STR_MAX_PCIE_REPLAYS                                                            \
    "max_pcie_replays" /* Maximum PCIe replays allowed per device                            \
                                                      while the plugin runs. If more replays \
                                                      occur than this threshold, this plugin \
                                                      will fail */

/* The maximum number of recovery errors that can happen before it's considered an error */
#define PCIE_STR_MAX_NVLINK_RECOVERY_ERRORS "max_nvlink_recovery_errors"

#define PCIE_STR_MAX_MEMORY_CLOCK                                                              \
    "max_memory_clock" /* Maximum memory clock in MHZ to use when locking                      \
                                                      application clocks to max while busgrind \
                                                      runs. */
#define PCIE_STR_MAX_GRAPHICS_CLOCK                                                               \
    "max_graphics_clock" /* Maximum graphics clock in MHZ to use when                             \
                                                          locking application clocks to max while \
                                                          busgrind runs */

#define PCIE_STR_CRC_ERROR_THRESHOLD                                             \
    "nvlink_crc_error_threshold" /* threshold at which CRC errors should cause a \
                                                                   failure */
/* time in seconds to run the parallel bw check */
#define PCIE_STR_PARALLEL_BW_CHECK_DURATION "parallel_bw_check_duration"

#define PCIE_STR_GPU_NVLINKS_EXPECTED_UP "gpu_nvlinks_expected_up" /* number of nvlinks expected up for each GPU */
#define PCIE_STR_NVSWITCH_NVLINKS_EXPECTED_UP \
    "nvswitch_nvlinks_expected_up" /* number of nvlinks expected up for each NvSwitch */

/* Sub tests tags */
#define PCIE_SUBTEST_H2D_D2H_SINGLE_PINNED   "h2d_d2h_single_pinned"
#define PCIE_SUBTEST_H2D_D2H_SINGLE_UNPINNED "h2d_d2h_single_unpinned"

#define PCIE_SUBTEST_H2D_D2H_CONCURRENT_PINNED   "h2d_d2h_concurrent_pinned"
#define PCIE_SUBTEST_H2D_D2H_CONCURRENT_UNPINNED "h2d_d2h_concurrent_unpinned"

#define PCIE_SUBTEST_H2D_D2H_LATENCY_PINNED   "h2d_d2h_latency_pinned"
#define PCIE_SUBTEST_H2D_D2H_LATENCY_UNPINNED "h2d_d2h_latency_unpinned"

#define PCIE_SUBTEST_P2P_BW_P2P_ENABLED  "p2p_bw_p2p_enabled"
#define PCIE_SUBTEST_P2P_BW_P2P_DISABLED "p2p_bw_p2p_disabled"

#define PCIE_SUBTEST_P2P_BW_CONCURRENT_P2P_ENABLED  "p2p_bw_concurrent_p2p_enabled"
#define PCIE_SUBTEST_P2P_BW_CONCURRENT_P2P_DISABLED "p2p_bw_concurrent_p2p_disabled"

#define PCIE_SUBTEST_1D_EXCH_BW_P2P_ENABLED  "1d_exch_bw_p2p_enabled"
#define PCIE_SUBTEST_1D_EXCH_BW_P2P_DISABLED "1d_exch_bw_p2p_disabled"

#define PCIE_SUBTEST_P2P_LATENCY_P2P_ENABLED  "p2p_latency_p2p_enabled"
#define PCIE_SUBTEST_P2P_LATENCY_P2P_DISABLED "p2p_latency_p2p_disabled"

#define PCIE_SUBTEST_BROKEN_P2P            "broken_p2p"
#define PCIE_SUBTEST_BROKEN_P2P_SIZE_IN_KB "broken_p2p_size_in_kb"

/******************************************************************************
 * TARGETED POWER PLUGIN
 *****************************************************************************/
#define TP_PLUGIN_NAME     "targeted_power"
#define TP_PLUGIN_CATEGORY PLUGIN_CATEGORY_STRESS

/* Public parameters - we expect users to change these */
#define TP_STR_TEST_DURATION      "test_duration"
#define TP_STR_TARGET_POWER       "target_power"
#define TP_STR_TEMPERATURE_MAX    "temperature_max"
#define TP_STR_FAIL_ON_CLOCK_DROP "fail_on_clock_drop"

/* Private parameters - we can have users change these but we don't need to
 * document them until users will change them
 */
#define TP_STR_USE_DGEMM               "use_dgemm"
#define TP_STR_CUDA_STREAMS_PER_GPU    "cuda_streams_per_gpu"
#define TP_STR_READJUST_INTERVAL       "readjust_interval"
#define TP_STR_PRINT_INTERVAL          "print_interval"
#define TP_STR_TARGET_POWER_MIN_RATIO  "target_power_min_ratio"
#define TP_STR_TARGET_POWER_MAX_RATIO  "target_power_max_ratio"
#define TP_STR_MOV_AVG_PERIODS         "moving_average_periods"
#define TP_STR_TARGET_MOVAVG_MIN_RATIO "target_movavg_min_ratio"
#define TP_STR_TARGET_MOVAVG_MAX_RATIO "target_movavg_max_ratio"
#define TP_STR_ENFORCED_POWER_LIMIT    "enforced_power_limit"

#define TP_STR_MAX_MEMORY_CLOCK                                                                      \
    "max_memory_clock" /* Maximum memory clock in MHZ to use when locking                            \
                                                      application clocks to max while targeted power \
                                                      runs. */
#define TP_STR_MAX_GRAPHICS_CLOCK                                                                 \
    "max_graphics_clock" /* Maximum graphics clock in MHZ to use when                             \
                                                          locking application clocks to max while \
                                                          targeted power runs */
#define TP_STR_OPS_PER_REQUEUE                                                                      \
    "ops_per_requeue" /* How many matrix multiplication operations should                           \
                                                    we queue every time the stream is idle. Setting \
                                                    this higher overcomes the kernel launch latency */
#define TP_STR_STARTING_MATRIX_DIM                                                                        \
    "starting_matrix_dim"                           /* Starting dimension N in NxN for our matrix         \
                                                            to start ramping up from. Setting this higher \
                                                            decreases the ramp-up time needed to hit      \
                                                            our power target. */
#define TP_STR_IS_ALLOWED          "is_allowed"     /* Is the targeted power plugin allowed to run? */
#define TP_STR_SBE_ERROR_THRESHOLD "max_sbe_errors" /* Threshold beyond which sbe's are treated as errors */
#define TP_STR_USE_DGEMV           "use_dgemv"      /* Use cuBLAS level-2 DGEMV instead of level-3 DEMM */
#define TP_STR_MAX_MATRIX_DIM      "max_matrix_dim" /* Maximum single dimension */
/******************************************************************************
 * TARGETED STRESS PLUGIN
 *****************************************************************************/
#define TS_PLUGIN_NAME     "targeted_stress"
#define TS_PLUGIN_CATEGORY PLUGIN_CATEGORY_STRESS

/* Public parameters - we expect users to change these */
#define TS_STR_TEST_DURATION         "test_duration"
#define TS_STR_TARGET_PERF           "target_stress"
#define TS_STR_TARGET_PERF_MIN_RATIO "target_perf_min_ratio"
#define TS_STR_TEMPERATURE_MAX       "temperature_max"

/* Private parameters - we can have users change these but we don't need to
 * document them until users will change them
 */
#define TS_STR_IS_ALLOWED           "is_allowed" /* Is the targeted stress plugin allowed to run? */
#define TS_STR_USE_DGEMM            "use_dgemm"
#define TS_STR_CUDA_STREAMS_PER_GPU "cuda_streams_per_gpu"
#define TS_STR_CUDA_OPS_PER_STREAM  "ops_per_stream_queue"

#define TS_STR_MAX_PCIE_REPLAYS                                                                                     \
    "max_pcie_replays" /* Maximum PCIe replays allowed per device while the plugin runs. If more replays occur than \
                          this threshold, this plugin will fail */

#define TS_STR_MAX_MEMORY_CLOCK                                                                                    \
    "max_memory_clock" /* Maximum memory clock in MHZ to use when locking application clocks to max while targeted \
                          perf runs. */
#define TS_STR_MAX_GRAPHICS_CLOCK                                                                                      \
    "max_graphics_clock" /* Maximum graphics clock in MHZ to use when locking application clocks to max while targeted \
                            perf runs */
#define TS_STR_SBE_ERROR_THRESHOLD "max_sbe_errors" /* Threshold beyond which sbe's are treated as errors */

/******************************************************************************
 * MEMORY PLUGIN
 *****************************************************************************/
#define MEMORY_PLUGIN_NAME     "memory"
#define MEMORY_PLUGIN_CATEGORY PLUGIN_CATEGORY_HW

#define MEMORY_STR_IS_ALLOWED                "is_allowed" /* Is the memory plugin allowed to run? */
#define MEMORY_STR_MIN_ALLOCATION_PERCENTAGE "minimum_allocation_percentage"

// Parameters controlling the cache subtest
#define MEMORY_SUBTEST_L1TAG                     "gpu_memory_cache"
#define MEMORY_L1TAG_STR_IS_ALLOWED              "l1_is_allowed" /* Is the l1tag subtest allowed to run? */
#define MEMORY_L1TAG_STR_TEST_DURATION           "test_duration"
#define MEMORY_L1TAG_STR_TEST_LOOPS              "test_loops"
#define MEMORY_L1TAG_STR_INNER_ITERATIONS        "inner_iterations"
#define MEMORY_L1TAG_STR_ERROR_LOG_LEN           "log_len"
#define MEMORY_L1TAG_STR_DUMP_MISCOMPARES        "dump_miscompares"
#define MEMORY_L1TAG_STR_L1_CACHE_SIZE_KB_PER_SM "l1cache_size_kb_per_sm"

/******************************************************************************
 * MEMTEST PLUGIN
 *****************************************************************************/
#define MEMTEST_PLUGIN_NAME     "memtest"
#define MEMTEST_PLUGIN_CATEGORY PLUGIN_CATEGORY_STRESS

#define MEMTEST_STR_IS_ALLOWED     "is_allowed"     /* Is the memory plugin allowed to run? */
#define MEMTEST_STR_PATTERN        "pattern"        /* test pattern for test4/test8/test10 */
#define MEMTEST_STR_TEST_DURATION  "test_duration"  /* target test time */
#define MEMTEST_STR_USE_MAPPED_MEM "use_mapped_mem" /* use cuda mapped memory instead of native device memory */
#define MEMTEST_STR_TEST0          "test0"          /* whether to enable test0 */
#define MEMTEST_STR_TEST1          "test1"          /* whether to enable test1 */
#define MEMTEST_STR_TEST2          "test2"          /* whether to enable test2 */
#define MEMTEST_STR_TEST3          "test3"          /* whether to enable test3 */
#define MEMTEST_STR_TEST4          "test4"          /* whether to enable test4 */
#define MEMTEST_STR_TEST5          "test5"          /* whether to enable test5 */
#define MEMTEST_STR_TEST6          "test6"          /* whether to enable test6 */
#define MEMTEST_STR_TEST7          "test7"          /* whether to enable test7 */
#define MEMTEST_STR_TEST8          "test8"          /* whether to enable test8 */
#define MEMTEST_STR_TEST9          "test9"          /* whether to enable test9 */
#define MEMTEST_STR_TEST10         "test10"         /* whether to enable test10 */

/******************************************************************************
 * HARDWARE PLUGIN
 *****************************************************************************/
#define HARDWARE_PLUGIN_INTERNAL_NAME "hardware"

/******************************************************************************
 * SM STRESS PLUGIN - included as part of PCIe
 *****************************************************************************/
#define SMSTRESS_PLUGIN_NAME     "sm_stress"
#define SMSTRESS_PLUGIN_CATEGORY PCIE_PLUGIN_CATEGORY

/* Public parameters - we expect users to change these */
#define SMSTRESS_STR_TEST_DURATION   "test_duration"
#define SMSTRESS_STR_TARGET_PERF     "target_stress"
#define SMSTRESS_STR_TEMPERATURE_MAX "temperature_max"

/* Private parameters - we can have users change these but we don't need to
 * document them until users will change them
 */
#define SMSTRESS_STR_USE_DGEMM  "use_dgemm"
#define SMSTRESS_STR_MATRIX_DIM "matrix_dim" /* The dimension of the matrix used for S/Dgemm */

/****************************************************************************
 * GPU BURN PLUGIN
 ***************************************************************************/
#define DIAGNOSTIC_PLUGIN_NAME     "diagnostic"
#define DIAGNOSTIC_PLUGIN_CATEGORY PLUGIN_CATEGORY_HW

/* Public parameters - we expect users to change these */
#define DIAGNOSTIC_STR_SBE_ERROR_THRESHOLD "max_sbe_errors"  /* Threshold beyond which sbe's are treated as errors */
#define DIAGNOSTIC_STR_TEST_DURATION       "test_duration"   /* Length of the test */
#define DIAGNOSTIC_STR_USE_DOUBLES         "use_doubles"     /* Use doubles instead of floating point */
#define DIAGNOSTIC_STR_TEMPERATURE_MAX     "temperature_max" /* Max temperature allowed during test */
#define DIAGNOSTIC_STR_IS_ALLOWED          "is_allowed"      /* Is this plugin allowed to run? */
#define DIAGNOSTIC_STR_MATRIX_DIM          "matrix_dim"      /* The starting dimension of the matrix used for S/Dgemm */
#define DIAGNOSTIC_STR_PRECISION           "precision"       /* The precision to use: half, single, or double */
#define DIAGNOSTIC_STR_GFLOPS_TOLERANCE_PCNT \
    "gflops_tolerance_pcnt" /* % of mean below which gflops are treated as errors */

/****************************************************************************
 * CONTEXT CREATE PLUGIN
 ***************************************************************************/
#define CTXCREATE_PLUGIN_NAME     "context_create"
#define CTXCREATE_PLUGIN_CATEGORY PLUGIN_CATEGORY_HW

/* Private parameters */
#define CTXCREATE_IS_ALLOWED       "is_allowed"       /* Is this plugin allowed to run */
#define CTXCREATE_IGNORE_EXCLUSIVE "ignore_exclusive" /* Attempt the test even if exlusive mode is set */

/****************************************************************************
 * MEMORY BANDWIDTH PLUGIN
 ***************************************************************************/
#define MEMBW_PLUGIN_NAME     "memory_bandwidth"
#define MEMBW_PLUGIN_CATEGORY PLUGIN_CATEGORY_STRESS

#define MEMBW_STR_MINIMUM_BANDWIDTH "minimum_bandwidth" /* minimum bandwidth in MB / s */

#define MEMBW_STR_IS_ALLOWED          "is_allowed"     /* Is the memory bandwidth plugin allowed to run? */
#define MEMBW_STR_SBE_ERROR_THRESHOLD "max_sbe_errors" /* Threshold beyond which sbe's are treated as errors */

/****************************************************************************
 * PULSE TEST PLUGIN
 ***************************************************************************/
#define PULSE_TEST_PLUGIN_NAME     "pulse_test"
#define PULSE_TEST_PLUGIN_CATEGORY PLUGIN_CATEGORY_HW

#define PULSE_TEST_STR_IS_ALLOWED        "is_allowed" /* Is the plugin allowed to run */
#define PULSE_TEST_STR_CURRENT_ITERATION "current_iteration"
#define PULSE_TEST_STR_TOTAL_ITERATIONS  "total_iterations"
#define PULSE_TEST_STR_TEST_DURATION     "test_duration"
#define PULSE_TEST_STR_DISPLAY_PATTERNS  "display_patterns"
#define PULSE_TEST_STR_PATTERNS          "patterns"

/****************************************************************************
 * EUD PLUGIN
 ***************************************************************************/
#define EUD_PLUGIN_NAME      "eud"
#define EUD_PLUGIN_CATEGORY  PLUGIN_CATEGORY_STRESS
#define EUD_FULL_PROFILE     "full_profile"     /*!< Run the full MODS profile or not */
#define EUD_PASSTHROUGH_ARGS "passthrough_args" /*!< Arguments to pass to the EUD binary */
#define EUD_MLE_PARSE        "parse_mle"        /*!< Parse the MLE or not */
#define EUD_STR_IS_ALLOWED   "is_allowed"       /* Is the EUD plugin allowed to run? */
#define CPU_EUD_TEST_NAME    "cpu_eud"

/****************************************************************************
 * NVBandwidth PLUGIN
 ***************************************************************************/
#define NVBANDWIDTH_PLUGIN_NAME     "nvbandwidth"
#define NVBANDWIDTH_PLUGIN_CATEGORY PLUGIN_CATEGORY_HW

#define NVBANDWIDTH_STR_IS_ALLOWED "is_allowed" /* Is the plugin allowed to run */
#define NVBANDWIDTH_STR_TESTCASES  "testcases"  /* test case separated by , e.g, 0,1,2 */

/*****************************************************************************
 * PER PLUGIN ERROR DEFINITIONS AND THEIR BITMASKS
 *****************************************************************************/

/******************************************************************************
 * SM STRESS PLUGIN
 *****************************************************************************/
#define SMSTRESS_ERR_GENERIC 0x0000100000000000ULL

/******************************************************************************
 * DIAGNOSTIC PLUGIN
 *****************************************************************************/

#define DIAGNOSTIC_ERR_GENERIC 0x0001000000000000ULL

/******************************************************************************
 * PCIE PLUGIN
 *****************************************************************************/
#define PCIE_ERR_PEER_ACCESS_DENIED 0x0000080000000000ULL
#define PCIE_ERR_CUDA_ALLOC_FAIL    0x0000040000000000ULL
#define PCIE_ERR_CUDA_GENERAL_FAIL  0x0000020000000000ULL
#define PCIE_ERR_NVLINK_ERROR       0x0000010000000000ULL
#define PCIE_ERR_PCIE_REPLAY_ERROR  0x0000008000000000ULL
#define PCIE_ERR_CUDA_SYNC_FAIL     0x0000004000000000ULL
#define PCIE_ERR_BW_TOO_LOW         0x0000002000000000ULL
#define PCIE_ERR_LATENCY_TOO_HIGH   0x0000001000000000ULL

/* Old constants that have been consolidated */
#define PCIE_ERR_CUDA_EVENT_FAIL  PCIE_ERR_CUDA_GENERAL_FAIL
#define PCIE_ERR_CUDA_STREAM_FAIL PCIE_ERR_CUDA_GENERAL_FAIL

/******************************************************************************
 * TARGETED POWER PLUGIN
 *****************************************************************************/
#define TP_ERR_BIT_ERROR                    0x0000000100000000ULL
#define TP_ERR_WORKER_BEGIN_FAIL            0x0000000080000000ULL
#define TP_ERR_CUDA_MEMCPY_FAIL             0x0000000040000000ULL
#define TP_ERR_CUDA_ALLOC_FAIL              0x0000000020000000ULL
#define TP_ERR_CUDA_STREAM_FAIL             0x0000000010000000ULL
#define TP_ERR_NVML_FAIL                    0x0000000008000000ULL
#define TP_ERR_CUBLAS_FAIL                  0x0000000004000000ULL
#define TP_ERR_GPU_POWER_TOO_LOW            0x0000000002000000ULL
#define TP_ERR_GPU_POWER_TOO_HIGH           0x0000000001000000ULL
#define TP_ERR_GPU_TEMP_TOO_HIGH            0x0000000000800000ULL
#define TP_ERR_GPU_POWER_VIOLATION_COUNTERS 0x0000000000400000ULL
#define TP_ERR_GPU_THERM_VIOLATION_COUNTERS 0x0000000000200000ULL
#define TP_ERR_GPU_CLOCKS_DROPPED           0x0000000000100000ULL

/******************************************************************************
 * TARGETED STRESS PLUGIN
 *****************************************************************************/
#define TS_ERR_PCIE_REPLAY_ERROR            0x0000000000040000ULL
#define TS_ERR_BIT_ERROR                    0x0000000000020000ULL
#define TS_ERR_WORKER_BEGIN_FAIL            0x0000000000010000ULL
#define TS_ERR_CUDA_MEMCPY_FAIL             0x0000000000008000ULL
#define TS_ERR_CUDA_ALLOC_FAIL              0x0000000000004000ULL
#define TS_ERR_CUDA_STREAM_FAIL             0x0000000000002000ULL
#define TS_ERR_NVML_FAIL                    0x0000000000001000ULL
#define TS_ERR_CUBLAS_FAIL                  0x0000000000000800ULL
#define TS_ERR_GPU_PERF_TOO_LOW             0x0000000000000400ULL
#define TS_ERR_GPU_TEMP_TOO_HIGH            0x0000000000000200ULL
#define TS_ERR_GPU_THERM_VIOLATION_COUNTERS 0x0000000000000100ULL

/******************************************************************************
 * MEMORY PLUGIN
 *****************************************************************************/
#define MEM_ERR_CUDA_ALLOC_FAIL 0x0000000000000020ULL
#define MEM_ERR_DBE_FAIL        0x0000000000000010ULL

/******************************************************************************
 * SOFTWARE PLUGIN
 *****************************************************************************/
#define SW_ERR_FAIL 0x0000000000000001ULL

/******************************************************************************
 * COMMON TO ALL PLUGINS
 *****************************************************************************/
#define SUPPRESSED_ERROR_STR "Suppressed error: "

#endif // PLUGINSTRINGS_H
