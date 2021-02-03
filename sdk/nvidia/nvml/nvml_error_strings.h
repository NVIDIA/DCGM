#ifndef __nvml_error_strings_h__
#define __nvml_error_strings_h__

#include <dcgm_nvml.h>
static const char *errorString(nvmlReturn_t result)
{
    switch (result)
    {
        case NVML_SUCCESS:
            return "Success";
        case NVML_ERROR_UNINITIALIZED:
            return "Uninitialized";
        case NVML_ERROR_INVALID_ARGUMENT:
            return "Invalid Argument";
        case NVML_ERROR_NOT_SUPPORTED:
            return "Not Supported";
        case NVML_ERROR_NO_PERMISSION:
            return "Insufficient Permissions";
        case NVML_ERROR_ALREADY_INITIALIZED:
            return "Already Initialized";
        case NVML_ERROR_NOT_FOUND:
            return "Not Found";
        case NVML_ERROR_INSUFFICIENT_SIZE:
            return "Insufficient Size";
        case NVML_ERROR_INSUFFICIENT_POWER:
            return "Insufficient External Power";
        case NVML_ERROR_DRIVER_NOT_LOADED:
            return "Driver Not Loaded";
        case NVML_ERROR_TIMEOUT:
            return "Timeout";
        case NVML_ERROR_IRQ_ISSUE:
            return "Interrupt Request Issue";
        case NVML_ERROR_LIBRARY_NOT_FOUND:
            return "NVML Shared Library Not Found";
        case NVML_ERROR_FUNCTION_NOT_FOUND:
            return "Function Not Found";
        case NVML_ERROR_CORRUPTED_INFOROM:
            return "Corrupted infoROM";
        case NVML_ERROR_GPU_IS_LOST:
            return "GPU is lost";
        case NVML_ERROR_RESET_REQUIRED:
            return "GPU requires reset";
        case NVML_ERROR_OPERATING_SYSTEM:
            return "GPU access blocked by the operating system";
        case NVML_ERROR_LIB_RM_VERSION_MISMATCH:
            return "Driver/library version mismatch";
        case NVML_ERROR_IN_USE:
            return "In use by another client";
        case NVML_ERROR_MEMORY:
            return "Insufficient Memory";
        case NVML_ERROR_INSUFFICIENT_RESOURCES:
            return "Insufficient Resources";
        case NVML_ERROR_UNKNOWN:
            return "Unknown Error";

        default:
            // Wrong error codes should be handled by the caller
            return 0;
    }
}


#endif // __nvml_error_strings_h__
