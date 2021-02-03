#ifndef NVML_BENCHMARK_H
#define NVML_BENCHMARK_H

#include <dcgm_nvml.h>

/*****************************************************************************/
/**
 *  Callback return codes.
 *
 * Return codes > 0 are reserved for NVML return codes. See nvmlReturn_enum in nvml.h
 **/
#define NB_ST_OK 0             /* OK. Same as NVML_SUCCESS */
#define NB_ST_UNKNOWN_ERROR -1 /* Unspecified error */

/*****************************************************************************/
#define NB_MAX_CLOCKS 64 /* maximum number of clock speed entries in various arrays */

/*****************************************************************************/
/* Holds all of the context information for a given NVML device. This can be
 * used to cache information about a device or restore state
 */
typedef struct nb_device_t
{
    char printHandle[32];    /* Handle to print for this device */
    int nvmlDeviceIndex;     /* nvml device index of this device */
    nvmlDevice_t nvmlDevice; /* nvml handle to this device */

    /* Supported graphics clock speeds */
    unsigned int NgraphicsClock;
    unsigned int graphicsClock[NB_MAX_CLOCKS];

    /* Supported memory clocks speeds */
    unsigned int NmemoryClock;
    unsigned int memoryClock[NB_MAX_CLOCKS];

} nb_device_t, *nb_device_p;

#define NB_MAX_DEVICES 16 /* Maximum number of devices to run against */

/*****************************************************************************/
/**
 * Prototype for an API callback that takes no NVML parameters such as
 * nvmlSystemGetDriverVersion
 *
 * Returns: NB_ST_OK if OK
 *          Other NB_ST_? status code on error
 */
typedef int (*nb_no_parameters_f)(void);

/*****************************************************************************/
/**
 * Prototype for an API callback that only takes a NVML device as an input
 * parameter such as nvmlDeviceGetName
 *
 * Returns: NB_ST_OK if OK
 *          Other NB_ST_? status code on error
 */
typedef int (*nb_device_f)(nb_device_p nbDevice);

/*****************************************************************************/
/* Struct to represent one API to test */
typedef struct nb_api_t
{
    char *apiName; /* Name of the API. Ex nvmlSystemGetDriverVersion */

    /* Callback functions. Only one of the below should be set */
    nb_no_parameters_f noParametersCB;
    nb_device_f deviceCB;
} nb_api_t, *nb_api_p;

/*****************************************************************************/
/* Holds all of the context information for a given worker thread */
typedef struct nb_thread_t
{
} nb_thread_t, *nb_thread_p;

/*****************************************************************************/
typedef struct nb_globals_t
{
    int nvmlInitialized; /* Has NVML been successfully initialized */

    /* Test Settings */
    int callsPerApi; /* How many times we should call each API to get timing */


    int Ndevice; /* Number of entries that are valid in device[] */
    nb_device_t device[NB_MAX_DEVICES];
} nb_globals_t, *nb_globals_p;

/*****************************************************************************/

#endif //NVML_BENCHMARK_H
