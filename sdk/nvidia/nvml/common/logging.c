#include "logging.h"


/* This is used by old C modules to determine the current logging level. 
   See nvmlDebugingLevel constants in logging.h */
int loggingDebugLevel = 0;


/* This is used by old C modules to write log messages. Discard these messages for now.
   loggingDebugLevel is 0 (NVML_DBG_DISABLED) anyway */
int loggingPrintf(const char *fmt, ...)
{
    return 0;
}
