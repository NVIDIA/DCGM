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
#include <cuda.h>
#include <errno.h>
#include <stdio.h>
#include <sys/types.h>

char VERSION[] = "2.0.1";

int verboseLogging = 1;
#define CU_CHECK_ERROR(command, string, ...)                                                    \
    do                                                                                          \
    {                                                                                           \
        CUresult cceResult = (command);                                                         \
        if (verboseLogging)                                                                     \
            printf("%s: line %d: Calling %s\n", getCurrentDateAndTimeMs(), __LINE__, #command); \
        if (cceResult != CUDA_SUCCESS)                                                          \
        {                                                                                       \
            fprintf(stderr, "ERROR %d: " string "\n", __LINE__, ##__VA_ARGS__);                 \
            fprintf(stderr, " '%s' returned %d\n", #command, cceResult);                        \
            return 1;                                                                           \
        }                                                                                       \
    } while (0)

#define MIN(a, b)      ((a) < (b) ? (a) : (b))
#define ROUND_UP(x, n) (((x) + ((n)-1)) & ~((n)-1))
#define MAX_GPUS       64

#define ASSERT(cond, string, ...)                                                                            \
    do                                                                                                       \
    {                                                                                                        \
        if (!(cond))                                                                                         \
        {                                                                                                    \
            fprintf(stderr, "ERROR %d: Condition %s failed : " string "\n", __LINE__, #cond, ##__VA_ARGS__); \
            exit(1);                                                                                         \
        }                                                                                                    \
    } while (0)

typedef struct
{
    unsigned int year;
    unsigned int month;      // 1-12
    unsigned int dayOfMonth; // 1-31
    unsigned int dayOfWeek;  // 0-6
    unsigned int hour;       // 0-23
    unsigned int min;        // 0-59
    unsigned int sec;        // 0-59
    unsigned int msec;       // 0-999
} localTime_t;

#if defined(_WIN32)
#include <windows.h>
static double second(void)
{
    LARGE_INTEGER t;
    static double oofreq;
    static int checkedForHighResTimer;
    static BOOL hasHighResTimer;

    if (!checkedForHighResTimer)
    {
        hasHighResTimer        = QueryPerformanceFrequency(&t);
        oofreq                 = 1.0 / (double)t.QuadPart;
        checkedForHighResTimer = 1;
    }
    if (hasHighResTimer)
    {
        QueryPerformanceCounter(&t);
        return (double)t.QuadPart * oofreq;
    }
    else
    {
        return (double)GetTickCount() / 1000.0;
    }
}
void getLocalTime(localTime_t *localTime)
{
    SYSTEMTIME winTime;
    GetLocalTime(&winTime);
    localTime->year       = winTime.wYear;
    localTime->month      = winTime.wMonth;
    localTime->dayOfMonth = winTime.wDay;
    localTime->dayOfWeek  = winTime.wDayOfWeek;
    localTime->hour       = winTime.wHour;
    localTime->min        = winTime.wMinute;
    localTime->sec        = winTime.wSecond;
    localTime->msec       = winTime.wMilliseconds;
}
void mySleep(unsigned int msec)
{
    Sleep(msec);
}

#elif defined(__linux__) || defined(__APPLE__)
#include <stddef.h>
#include <sys/time.h>
#include <time.h>
static double second(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
}
void getLocalTime(localTime_t *localTime)
{
    struct tm ttm;
    struct timeval timev;

    gettimeofday(&timev, NULL);

    // This is a hack which "just works": cast the seconds field of the timeval
    // struct to a time_t
    localtime_r((time_t *)&timev.tv_sec, &ttm);

    localTime->year       = ttm.tm_year + 1900;
    localTime->month      = ttm.tm_mon + 1;
    localTime->dayOfMonth = ttm.tm_mday;
    localTime->dayOfWeek  = ttm.tm_wday;
    localTime->hour       = ttm.tm_hour;
    localTime->min        = ttm.tm_min;
    localTime->sec        = ttm.tm_sec;
    localTime->msec       = timev.tv_usec / 1000;
}
void mySleep(unsigned int msec)
{
    struct timespec t_req, t_rem;
    int ret = 0;

    t_req.tv_sec  = msec / 1000;
    t_req.tv_nsec = (msec % 1000) * 1000000;

    ret = nanosleep(&t_req, &t_rem);

    // if interrupted by a non-blocked signal
    // copy remaining time to the requested time
    while (ret != 0 && errno == EINTR)
    {
        t_req = t_rem;
        ret   = nanosleep(&t_req, &t_rem);
    }
}

#else
#error unsupported platform
#endif

char *getCurrentDateAndTimeMs(void)
{
    localTime_t time;
    static char buff[128];

    // Get the local time
    getLocalTime(&time);

    sprintf(buff,
            "%u/%02u/%02u %02u:%02u:%02u.%03u",
            time.year,
            time.month,
            time.dayOfMonth,
            time.hour,
            time.min,
            time.sec,
            time.msec);

    return buff;
}

void usage(int code, char *name)
{
    printf(
        "\n"
        "\n Version %s"
        "\n"
        "\n Usage %s [-h | --help] [list of commands]"
        "\n  --ctxCreate,-i <pci bus id>"
        "\n  --ctxDestroy <pci bus id>"
        "\n"
        "\n  --cuMemAlloc <pci bus id> <size in MB>"
        "\n         Also frees previous memory buffer before allocating new one."
        "\n         Requires ctxCreate"
        "\n  --cuMemFree <pci bus id>"
        "\n"
        "\n  --busyGpu <pci bus id> <ms>"
        "\n         Runs kernel that causes 100%% gpu & memory utilization (allocates some memory)"
        "\n         Requires ctxCreate"
        "\n"
        "\n  --assertGpu <pci bus id> <ms>"
        "\n         Runs kernel and intentionally asserts it as false to generate and XID 43 error (allocates some memory)"
        "\n         Requires ctxCreate"
        "\n"
        "\n  --busyIter <pci bus id> <num iterations>"
        "\n         Runs kernel for specified number of iterations that causes 100%% gpu & memory utilization (allocates some memory)"
        "\n         Requires ctxCreate"
        "\n"
        "\n  --getchar"
        "\n     pauses application till <enter> is passed"
        "\n  --sleep <ms>"
        "\n"
        "\n Notes:"
        "\n    Order of flags matter!"
        "\n    Flags are executed from left to right and can be repeat."
        "\n"
        "\n EXAMPLE:"
        "\n ./cuda_ctx_create_64bit --ctxCreate 0:6:0 --cuMemAlloc 0:6:0 100 --cuMemFree 0:6:0 --ctxDestroy 0:6:0"
        "\n                         --ctxCreate 0:6:0 --cuMemAlloc 0:6:0 200 --cuMemFree 0:6:0 --busyGpu 0:6:0 10000"
        "\n                         --ctxCreate 0:6:0 --cuMemAlloc 0:6:0 200 --cuMemFree 0:6:0 --busyIter 0:6:0 100"
        "\n                         --sleep 1000 --ctxDestroy 0:6:0"
        "\n"
        "\n",
        VERSION,
        name);

    exit(code);
}

int main(int argc, char **argv)
{
    CUcontext ctx[MAX_GPUS]   = { 0 };
    CUdeviceptr ptr[MAX_GPUS] = { 0 };
    int i;

    // set unbuffered output, no flushing required
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);

    printf("My PID: %u\n", getpid());
    CU_CHECK_ERROR(cuInit(0), "cuInit failed");

    for (i = 1; i < argc; i++)
    {
        printf("%s: Processing %s\n", getCurrentDateAndTimeMs(), argv[i]);

        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0)
            usage(0, argv[0]);
        else if (strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--ctxCreate") == 0)
        {
            int device;
            i++;
            CU_CHECK_ERROR(cuDeviceGetByPCIBusId(&device, argv[i]), "Bus id %s not matched", argv[i]);
            ASSERT(ctx[device] == NULL, "Previous ctx hasn't been destroyed on this device");
            CU_CHECK_ERROR(cuCtxCreate(&ctx[device], 0, device), "could not create context on CUDA device %d", device);
            printf("Context created\n");
        }
        else if (strcmp(argv[i], "--ctxDestroy") == 0)
        {
            int device;
            i++;
            CU_CHECK_ERROR(cuDeviceGetByPCIBusId(&device, argv[i]), "Bus id %s not matched", argv[i]);
            ASSERT(ctx[device] != NULL, "No ctx was created on this device");
            CU_CHECK_ERROR(cuCtxDestroy(ctx[device]), "couldn't destroy context on CUDA device %d", device);
            ctx[device] = NULL;
            ASSERT(!ptr[device], "There's unfreed memory");
        }
        else if (strcmp(argv[i], "--busyGpu") == 0)
        {
            CUfunction kernel;
            CUmodule mod;
            int device;
            CUdeviceptr ptr;
            size_t items   = 1024 * 1024 * 10;
            int iterations = 100;
            float ms;
            double start, stop, lastSyncSec, prevLastSyncSec;
            int pass    = 0;
            int passMod = 1; /* Maximum of 128 */

            CU_CHECK_ERROR(cuDeviceGetByPCIBusId(&device, argv[++i]), "Bus id %s not matched", argv[i]);
            ms = atoi(argv[++i]);

            ASSERT(ctx[device] != NULL, "No ctx was created on this device");
            CU_CHECK_ERROR(cuCtxPushCurrent(ctx[device]), "");
#if _WIN64 || __amd64__
            CU_CHECK_ERROR(cuModuleLoad(&mod, "busy_gpu64.ptx"), "couldn't load busy_gpu.ptx module");
#elif __powerpc64__
            CU_CHECK_ERROR(cuModuleLoad(&mod, "busy_gpu_ppc64le.ptx"), "couldn't load busy_gpu.ptx module");
#elif __aarch64__
            CU_CHECK_ERROR(cuModuleLoad(&mod, "busy_gpu_aarch64.ptx"), "couldn't load busy_gpu.ptx module");
#else
            CU_CHECK_ERROR(cuModuleLoad(&mod, "busy_gpu32.ptx"), "couldn't load busy_gpu.ptx module");
#endif
            CU_CHECK_ERROR(cuModuleGetFunction(&kernel, mod, "make_gpu_busy"), "couldn't load busy_gpu.ptx module");
            CU_CHECK_ERROR(cuMemAlloc(&ptr, items * sizeof(int)), "Failed to allocate memory");
            CU_CHECK_ERROR(cuMemsetD32(ptr, 12345, items), "");

            {
                int offset = 0;
                CU_CHECK_ERROR(cuParamSetv(kernel, offset, &ptr, sizeof(void *)), "");
                offset = ROUND_UP(offset + sizeof(void *), sizeof(void *));
                CU_CHECK_ERROR(cuParamSetv(kernel, offset, &items, sizeof(size_t)), "");
                offset = ROUND_UP(offset + sizeof(size_t), sizeof(size_t));
                CU_CHECK_ERROR(cuParamSetv(kernel, offset, &iterations, sizeof(int)), "");
                offset += sizeof(int);
                CU_CHECK_ERROR(cuParamSetSize(kernel, offset), "");
                CU_CHECK_ERROR(cuFuncSetBlockShape(kernel, 256, 1, 1), "");
            }

            start           = second();
            prevLastSyncSec = start;
            do
            {
                CU_CHECK_ERROR(cuLaunchGridAsync(kernel, 1024, 1, 0), "");
                pass++;
                if (pass % passMod == 0) // Synchronize every passMod passes
                {
                    CU_CHECK_ERROR(cuCtxSynchronize(), "");
                    lastSyncSec = second();

                    /* We want passes before synchronization to take between .5 and 1 second, so stop doubling after .25
                     * seconds */
                    if (lastSyncSec - prevLastSyncSec < 0.25)
                    {
                        passMod = MIN(128, passMod * 2);
                        // printf("diff: %f, passMod %d\n", lastSyncSec - prevLastSyncSec, passMod);
                    }
                    prevLastSyncSec = lastSyncSec;
                }
                stop           = second();
                verboseLogging = 0;
            } while ((stop - start) < ms / 1000);
            verboseLogging = 1;
            CU_CHECK_ERROR(cuCtxSynchronize(), "");
            printf("passes %d\n", pass);

            CU_CHECK_ERROR(cuMemFree(ptr), "");
            CU_CHECK_ERROR(cuModuleUnload(mod), "");
            {
                CUcontext tmp;
                CU_CHECK_ERROR(cuCtxPopCurrent(&tmp), "");
            }
        }
        else if (strcmp(argv[i], "--assertGpu") == 0)
        {
            CUfunction kernel;
            CUmodule mod;
            int device;
            CUdeviceptr ptr;
            size_t items   = 1024 * 1024 * 10;
            int iterations = 100;
            int pass       = 0;

            CU_CHECK_ERROR(cuDeviceGetByPCIBusId(&device, argv[++i]), "Bus id %s not matched", argv[i]);

            ASSERT(ctx[device] != NULL, "No ctx was created on this device");
            CU_CHECK_ERROR(cuCtxPushCurrent(ctx[device]), "");
            CU_CHECK_ERROR(cuModuleLoad(&mod, "cuda_assert.ptx"), "couldn't load cuda_assert.ptx module");
            CU_CHECK_ERROR(cuModuleGetFunction(&kernel, mod, "make_assert"), "couldn't load busy_gpu.ptx module");
            CU_CHECK_ERROR(cuMemAlloc(&ptr, items * sizeof(int)), "Failed to allocate memory");
            CU_CHECK_ERROR(cuMemsetD32(ptr, 12345, items), "");

            {
                int offset = 0;
                CU_CHECK_ERROR(cuParamSetv(kernel, offset, &ptr, sizeof(void *)), "");
                offset = ROUND_UP(offset + sizeof(void *), sizeof(void *));
                CU_CHECK_ERROR(cuParamSetv(kernel, offset, &items, sizeof(size_t)), "");
                offset = ROUND_UP(offset + sizeof(size_t), sizeof(size_t));
                CU_CHECK_ERROR(cuParamSetv(kernel, offset, &iterations, sizeof(int)), "");
                offset += sizeof(int);
                CU_CHECK_ERROR(cuParamSetSize(kernel, offset), "");
                CU_CHECK_ERROR(cuFuncSetBlockShape(kernel, 256, 1, 1), "");
            }

            CU_CHECK_ERROR(cuLaunchGridAsync(kernel, 1, 1, 0), "");

            verboseLogging = 1;
            CU_CHECK_ERROR(cuCtxSynchronize(), "");
            printf("passes %d\n", pass);

            CU_CHECK_ERROR(cuMemFree(ptr), "");
            CU_CHECK_ERROR(cuModuleUnload(mod), "");
            {
                CUcontext tmp;
                CU_CHECK_ERROR(cuCtxPopCurrent(&tmp), "");
            }
        }
        else if (strcmp(argv[i], "--busyIter") == 0)
        {
            CUfunction kernel;
            CUmodule mod;
            int device;
            CUdeviceptr ptr;
            size_t items = 1024 * 1024 * 10;
            int iterations;
            int pass    = 0;
            int passMod = 2; // Synchronize every 2 iterations to avoid any issues with slower GPUs

            CU_CHECK_ERROR(cuDeviceGetByPCIBusId(&device, argv[++i]), "Bus id %s not matched", argv[i]);
            iterations = atoi(argv[++i]);

            ASSERT(ctx[device] != NULL, "No ctx was created on this device");
            CU_CHECK_ERROR(cuCtxPushCurrent(ctx[device]), "");
#if _WIN64 || __amd64__
            CU_CHECK_ERROR(cuModuleLoad(&mod, "busy_gpu64.ptx"), "couldn't load busy_gpu.ptx module");
#elif __powerpc64__
            CU_CHECK_ERROR(cuModuleLoad(&mod, "busy_gpu_ppc64le.ptx"), "couldn't load busy_gpu.ptx module");
#else
            CU_CHECK_ERROR(cuModuleLoad(&mod, "busy_gpu32.ptx"), "couldn't load busy_gpu.ptx module");
#endif
            CU_CHECK_ERROR(cuModuleGetFunction(&kernel, mod, "make_gpu_busy"), "couldn't load busy_gpu.ptx module");
            CU_CHECK_ERROR(cuMemAlloc(&ptr, items * sizeof(int)), "Failed to allocate memory");
            CU_CHECK_ERROR(cuMemsetD32(ptr, 12345, items), "");

            {
                int offset = 0;
                CU_CHECK_ERROR(cuParamSetv(kernel, offset, &ptr, sizeof(void *)), "");
                offset = ROUND_UP(offset + sizeof(void *), sizeof(void *));
                CU_CHECK_ERROR(cuParamSetv(kernel, offset, &items, sizeof(size_t)), "");
                offset = ROUND_UP(offset + sizeof(size_t), sizeof(size_t));
                CU_CHECK_ERROR(cuParamSetv(kernel, offset, &iterations, sizeof(int)), "");
                offset += sizeof(int);
                CU_CHECK_ERROR(cuParamSetSize(kernel, offset), "");
                CU_CHECK_ERROR(cuFuncSetBlockShape(kernel, 256, 1, 1), "");
            }

            do
            {
                CU_CHECK_ERROR(cuLaunchGridAsync(kernel, 1024, 1, 0), "");
                pass++;
                if (pass % passMod == 0) // Synchronize every passMod passes
                {
                    CU_CHECK_ERROR(cuCtxSynchronize(), "");
                }

                verboseLogging = 0;
            } while (pass < iterations);
            verboseLogging = 1;
            CU_CHECK_ERROR(cuCtxSynchronize(), "");
            printf("passes %d\n", pass);

            CU_CHECK_ERROR(cuMemFree(ptr), "");
            CU_CHECK_ERROR(cuModuleUnload(mod), "");
            {
                CUcontext tmp;
                CU_CHECK_ERROR(cuCtxPopCurrent(&tmp), "");
            }
        }
        else if (strcmp(argv[i], "--cuMemAlloc") == 0)
        {
            int device;
            size_t size;

            CU_CHECK_ERROR(cuDeviceGetByPCIBusId(&device, argv[++i]), "Bus id %s not matched", argv[i]);
            size = atoi(argv[++i]) * 1024 * 1024;

            ASSERT(ctx[device] != NULL, "No ctx was created on this device");
            CU_CHECK_ERROR(cuCtxPushCurrent(ctx[device]), "");

            if (ptr[device])
                CU_CHECK_ERROR(cuMemFree(ptr[device]), "");
            CU_CHECK_ERROR(cuMemAlloc(&ptr[device], size), "Failed to allocate memory");
            {
                CUcontext tmp;
                CU_CHECK_ERROR(cuCtxPopCurrent(&tmp), "");
            }
        }
        else if (strcmp(argv[i], "--cuMemFree") == 0)
        {
            int device;

            CU_CHECK_ERROR(cuDeviceGetByPCIBusId(&device, argv[++i]), "Bus id %s not matched", argv[i]);

            ASSERT(ctx[device] != NULL, "No ctx was created on this device");
            CU_CHECK_ERROR(cuCtxPushCurrent(ctx[device]), "");

            CU_CHECK_ERROR(cuMemFree(ptr[device]), "");
            ptr[device] = 0;
            {
                CUcontext tmp;
                CU_CHECK_ERROR(cuCtxPopCurrent(&tmp), "");
            }
        }
        else if (strcmp(argv[i], "--sleep") == 0)
        {
            int ms = atoi(argv[++i]);
            printf("Sleeping for %d ms\n", ms);
            mySleep(ms);
        }
        else if (strcmp(argv[i], "--getchar") == 0)
        {
            printf("Waiting for new line char\n");
            getchar();
        }
        else
        {
            ASSERT(0, "Unrecognized command %s", argv[i]);
        }
    }

    printf("%s: Terminating application.\n", getCurrentDateAndTimeMs());
    return 0;
}
