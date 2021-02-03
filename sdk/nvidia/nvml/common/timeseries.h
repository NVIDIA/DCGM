/*
 * timeseries.h
 *
 */

#ifndef TIMESERIES_H
#define TIMESERIES_H

#include "keyedvector.h"
#include "timelib.h"
#include <float.h>  //DBL_MAX
#include <limits.h> //LLONG_MAX

#ifdef __cplusplus
extern "C"
{
#endif

/*****************************************************************************/
/* Error codes */
#define TS_ST_OK 0        /* Success */
#define TS_ST_BADPARAM -1 /* Bad parameter to function call */
#define TS_ST_MEMORY -2   /* Out of memory */
#define TS_ST_WRONGTYPE \
    -3 /* Tried to add an entry of a different type than
                                  what is stored in */
#define TS_ST_DUPETIMESTAMP \
    -4 /* Found a duplicate timestamped entry and might
                                  have even tried to resolve it but was ultimately
                                  unsuccessful in getting a unique key */
#define TS_ST_UNKNOWN \
    -5                  /* Unknown error to timeseries, most likely an
                                  error in keyedvector */
#define TS_ST_NODATA -6 /* Not enough data to do requested calculation */

/*****************************************************************************/
/* "Empty" or NULL values */
#define TS_EMPTY_INT64 9223372036854775807LL
#define TS_EMPTY_DOUBLE DBL_MAX

/* Get empty value based on type. */
#define TS_EMPTY(type) ((type == TS_TYPE_DOUBLE) ? TS_EMPTY_DOUBLE : TS_EMPTY_INT64)

/*****************************************************************************/
/* Types of data stored in a timeseries */
#define TS_TYPE_UNKNOWN 0 /* Uninitialized */
#define TS_TYPE_INT64 1   /* Signed 64-bit integer */
#define TS_TYPE_DOUBLE 2  /* Double-precision floating point */
#define TS_TYPE_STRING 3  /* Null-terminated string */
#define TS_TYPE_BLOB 4    /* Anonymous binary blob with size provided */
#define TS_TYPE_MAX 4     /* Same as above. Update as add new types */
    /* If you add a type here, make sure to add it to TS_EMPTY(type) above */

#define TS_LGE_EQUAL KV_LGE_EQUAL           /* return nearest == time (or none) */
#define TS_LGE_LESSEQUAL KV_LGE_LESSEQUAL   /* return nearest <= time */
#define TS_LGE_GREATEQUAL KV_LGE_GREATEQUAL /* return nearest >= time */
#define TS_LGE_LESS KV_LGE_LESS             /* return nearest < time */
#define TS_LGE_GREATER KV_LGE_GREATER       /* return nearest > time */

    /*****************************************************************************/
    /* Handle to a timeseries structure */
    typedef struct timeseries_t
    {
        int tsType;                /* TS_TYPE_? #define of the type of value stored
                                  in keyedVector */
        keyedvector_p keyedVector; /* Data structure to hold the time series */
    } timeseries_t, *timeseries_p;

    /* Cursor into a timeseries enumeration. Note that these cursors are only
 * valid as long as the timeseries is not modified (inserting or removing elements).
 */
    typedef kv_cursor_t timeseries_cursor_t;
    typedef kv_cursor_p timeseries_cursor_p;

    /* Entry stored in keyed vector */
    typedef struct timeseries_entry_t
    {
        timelib64_t usecSince1970; /* Key field. When did this entry happen */
        union
        {
            double dbl;
            long long i64;
            void *ptr; /* Element is a pointer. See val2.ptrSize for allocated size.
                             Note that ptrSize MUST include terminating NULL for strings */
        } val;         /* Measurement at time usecSince1970 */
        union
        {
            double dbl;
            long long i64;
            long long ptrSize; /* Size of memory that val.ptr points at */
        } val2;                /* To store any additional Information at time usecSince1970 */
    } timeseries_entry_t, *timeseries_entry_p;

    /*****************************************************************************/
    /*
 * Allocate a timeseries collection
 *
 * tsType    IN: TS_TYPE_? #define of the type of value to store
 * errorSt  OUT: Where to store the error
 *
 */
    timeseries_p timeseries_alloc(int tsType, int *errorSt);

    /*****************************************************************************/
    /*
 * Destroy an allocated timeseries collection
 */
    void timeseries_destroy(timeseries_p ts);

    /*****************************************************************************/
    /*
 * Count the number of entries in a time series
 *
 */
    int timeseries_size(timeseries_p ts);

    /*****************************************************************************/
    /*
 * Insert two double values into a time series. The series must be a 
 * TS_TYPE_DOUBLE or an error will be returned
 *
 * A timestamp of 0 means use the current time. If a timestamp already exists
 * in the data structure, timestamp will be incremented until no collision occurs
 *
 * Returns: 0 if OK
 *         <0 TS_ST_? #define on error
 *
 */
    int timeseries_insert_double(timeseries_p ts, timelib64_t timestamp, double value1, double value2);

    /*****************************************************************************/
    /*
 * Insert two int64 values into a time series. The series must be a 
 * TS_TYPE_INT64 or an error will be returned
 *
 * A timestamp of 0 means use the current time. If a timestamp already exists
 * in the data structure, timestamp will be incremented until no collision occurs
 *
 * Returns: 0 if OK
 *         <0 TS_ST_? #define on error
 *
 */
    int timeseries_insert_int64(timeseries_p ts, timelib64_t timestamp, long long value1, long long value2);

    /*****************************************************************************/
    /*
 * Insert a string value into a time series. The series must be a TS_TYPE_STRING
 * or an error will be returned
 *
 * A timestamp of 0 means use the current time. If a timestamp already exists
 * in the data structure, timestamp will be incremented until no collision occurs
 *
 * Returns: 0 if OK
 *         <0 TS_ST_? #define on error
 *
 */
    int timeseries_insert_string(timeseries_p ts, timelib64_t timestamp, char *value);

    /*****************************************************************************/
    /*
 * Insert a blob value into a time series. The series must be a TS_TYPE_BLOB
 * or an error will be returned
 *
 * A timestamp of 0 means use the current time. If a timestamp already exists
 * in the data structure, timestamp will be incremented until no collision occurs
 *
 * valueSize is the size of the value in bytes. The inserted value will be a
 * copy of value of size valueSize.
 *
 * Returns: 0 if OK
 *         <0 TS_ST_? #define on error
 *
 */
    int timeseries_insert_blob(timeseries_p ts, timelib64_t timestamp, void *value, int valueSize);

    /*****************************************************************************/
    /*
 * Enforce a quota on this time series on both number of records kept and
 * timestamp of the oldest record to keep
 *
 * oldestKeepTimestamp IN: Oldest usec since 1970 to to keep. Records older
 *                         than this timestamp will be deleted. 0 = don't enforce
 * maxKeepEntries      IN: Count of the maximum number of entries to keep.
 *                         0 = don't enforce
 *
 * Returns: 0 if OK
 *         <0 TS_ST_? #define on error
 */
    int timeseries_enforce_quota(timeseries_p ts, timelib64_t oldestKeepTimestamp, int maxKeepEntries);

    /*****************************************************************************/
    /*
 * Coersion versions of above functions to insert an int64 or double into a time
 * series that will automatically convert the input type into the type of the
 * time series
 *
 * A timestamp of 0 means use the current time. If a timestamp already exists
 * in the data structure, timestamp will be incremented until no collision occurs
 *
 * Returns: 0 if OK
 *         <0 TS_ST_? #define on error
 *
 */
    int timeseries_insert_double_coerce(timeseries_p ts, timelib64_t timestamp, double value1, double value2);
    int timeseries_insert_int64_coerce(timeseries_p ts, timelib64_t timestamp, long long value1, long long value2);

    /*****************************************************************************/
    /*
 * Calculate the sum from startTime to endTime. If startTime is 0, start from
 * the beginning of the time series. If endTime is 0, go until the end of the
 * series
 *
 * Returns: Sum of values (val) between time range
 *          TS_EMPTY_(DOUBLE|INT64) #define on error. Check errorSt if this is returned
 *
 */
    long long timeseries_sum_int64(timeseries_p ts, timelib64_t startTime, timelib64_t endTime, int *errorSt);
    double timeseries_sum_double(timeseries_p ts, timelib64_t startTime, timelib64_t endTime, int *errorSt);

    /*****************************************************************************/
    /*
 * Calculate the average from startTime to endTime. If startTime is 0, start from
 * the beginning of the time series. If endTime is 0, go until the end of the
 * series
 *
 * Returns: Average of values between time range
 *          TS_EMPTY_(DOUBLE|INT64) #define on error. Check errorSt if this is returned
 *
 */
    double timeseries_average(timeseries_p ts, timelib64_t startTime, timelib64_t endTime, int *errorSt);

    /*****************************************************************************/
    /*
 * Calculate the average from startTime to endTime. If startTime is 0, start from
 * the beginning of the time series. If endTime is 0, go until the end of the
 * series
 *
 * Returns: Average of values between time range
 *          TS_EMPTY_DOUBLE #define on error. Check errorSt if this is returned
 *
 */
    double timeseries_average(timeseries_p ts, timelib64_t startTime, timelib64_t endTime, int *errorSt);

    /*****************************************************************************/
    /*
 * Calculate the moving average starting from endTime and going backward a
 * maximum of maxSamples samples. If endTime is 0, start from the end of the
 * series
 *
 * Returns: Average of values between time range
 *          TS_EMPTY_DOUBLE #define on error. Check errorSt if this is returned
 *
 */
    double timeseries_moving_average(timeseries_p ts, timelib64_t endTime, int maxSamples, int *errorSt);

    /*****************************************************************************/
    /*
 * Calculate the minimum value of a time series across a time range of values
 *
 * Returns: Minimum value between startTIme and endTime. Specify startTime as
 *          0 to start at the beginning of the series and endTime as 0 to end
 *          at the end of the series.
 *
 */
    long long timeseries_min_int64(timeseries_p ts, timelib64_t startTime, timelib64_t endTime, int *errorSt);
    double timeseries_min_double(timeseries_p ts, timelib64_t startTime, timelib64_t endTime, int *errorSt);

    /*****************************************************************************/
    /*
 * Calculate the maximum value of a time series across a time range of values
 *
 * Returns: Maximum value between startTime and endTime. Specify startTime as
 *          0 to start at the beginning of the series and endTime as 0 to end
 *          at the end of the series.
 *
 */
    long long timeseries_max_int64(timeseries_p ts, timelib64_t startTime, timelib64_t endTime, int *errorSt);
    double timeseries_max_double(timeseries_p ts, timelib64_t startTime, timelib64_t endTime, int *errorSt);

/*****************************************************************************/
/* Relation operators for matching relative to a base value */
#define TS_REL_EQUAL 0      /* Matches == value (or none) */
#define TS_REL_LESSEQUAL 1  /* Matches <= value */
#define TS_REL_GREATEQUAL 2 /* Matches >= value */
#define TS_REL_LESS 3       /* Matches < value */
#define TS_REL_GREATER 4    /* Matches > value */

    /*****************************************************************************/
    /*
 * Calculate the number of values in a time series above or below a threshold
 *
 * startTime  IN: Earliest time of values to consider. 0=start at beginning
 * endTime    IN: Latest time of values to consider. 0=start at end
 * threshold  IN: Base value to compare values against
 * relation   IN: How to compare values against threshold.
 *                Example: TS_REL_GREATEQUAL, threshold=3.0 increments the counter
 *                for every value encountered that is >= 3.0
 *
 * Returns: >= 0 Number of matched samples
 *           < 0 on error.
 *
 */
    int timeseries_threshold_count(timeseries_p ts,
                                   timelib64_t startTime,
                                   timelib64_t endTime,
                                   double threshold,
                                   int relation);

    /*****************************************************************************/
    /* Calculate (roughly) the number of bytes in memory that this timeseries takes up.
 *
 * Returns >= 0 Number of bytes
 */
    long long timeseries_bytes_used(timeseries_p ts);

    /*************************************************************************/
    /*
 * Get the first element in the timeseries and a cursor that can be used to get the
 * next or previous value.  cursor can be NULL if you don't want it.
 */
    timeseries_entry_p timeseries_first(timeseries_p ts, timeseries_cursor_p cursor);

    /*************************************************************************/
    /*
 * Get the last element in the timeseries and a cursor that can be used to get the
 * next or previous value.  cursor can be NULL if you don't want it.
 */
    timeseries_entry_p timeseries_last(timeseries_p ts, timeseries_cursor_p cursor);

    /*************************************************************************/
    /*
Get the next/previous element.

NULL if there are no more elements or if cursor is invalid
*/
    timeseries_entry_p timeseries_next(timeseries_p ts, timeseries_cursor_p cursor);
    timeseries_entry_p timeseries_prev(timeseries_p ts, timeseries_cursor_p cursor);

    /*************************************************************************/
    /*
Locate the timeseries value that is close to the given time.

cursor will contain the position of the element so that timeseries_next() and
       timeseries_prev() can be called on it.
findOp is the relative operator used to find the element. See TS_LGE_? #defines
       TS_LGE_EQUAL will only match exactly. Other operators will return the
       nearest match in the direction the operator specifies

Returns NULL if not found.
        Pointer to element if found. Do not change key field information
            of the returned pointer or you will corrupt the data structure
*/
    timeseries_entry_p timeseries_find(timeseries_p ts, timelib64_t time, int findOp, timeseries_cursor_p cursor);

#ifdef __cplusplus
}
#endif

#endif /* TIMESERIES_H */
