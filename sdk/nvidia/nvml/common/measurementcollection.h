
#ifndef MEASUREMENTCOLLECTION_H
#define MEASUREMENTCOLLECTION_H

#include "hashtable.h"
#include "timelib.h"
#include "timeseries.h"

#ifdef __cplusplus
extern "C"
{
#endif

#ifdef _WINDOWS
// Silence window's error
// C4996: 'strdup': The POSIX name for this item is deprecated. Instead, use the
//        ISO C++ conformant name: _strdup. See online help for details.
#define strdup _strdup

// Windows strtok_r is under different name strtok_s (interface and behavior is the same)
#define strtok_r strtok_s

#endif // _WINDOWS


/*****************************************************************************/
/* Status codes < 0 are errors. > 0 are informational */
#define MCOLLECT_ST_EXISTS 1 /* Element exists, nonfatal version for add+set */
#define MCOLLECT_ST_OK 0
#define MCOLLECT_ST_BADPARAM -1 /* Bad parameter provided to function */
#define MCOLLECT_ST_MEMORY -2   /* Out of memory */
#define MCOLLECT_ST_OPNOTSUPPORTED \
    -3 /* The provided operation is not supported
                                         for the operation in question */
#define MCOLLECT_ST_NOTFOUND \
    -4 /* The given measurement collection was not
                                         found */

/*****************************************************************************/
/* Types of measurement values */
#define MC_TYPE_UNKNOWN 0           /* Bad value/uninitialized */
#define MC_TYPE_INT64 1             /* Signed 64-bit integer */
#define MC_TYPE_DOUBLE 2            /* Double precision string */
#define MC_TYPE_STRING 3            /* C char * string */
#define MC_TYPE_TIMESTAMP 4         /* timelib64_t, microseconds since 1970 */
#define MC_TYPE_TIMESERIES_DOUBLE 5 /* A series of timelib64_t, double */
#define MC_TYPE_TIMESERIES_INT64 6  /* A series of timelib64_t, int64 */
#define MC_TYPE_TIMESERIES_STRING 7 /* A series of timelib64_t, char *  int size */
#define MC_TYPE_TIMESERIES_BLOB 8   /* A series of timelib64_t, void *, int size */

    /* Note: if you add a time series type value here, make sure to update
 *       mcollect_type_is_timeseries
 */

#define MC_TYPE_COUNT 9 /* 1 greater than max value above */


    /*****************************************************************************/

    /* Public handle to a measurement collection. Call mcollect_alloc */
    typedef struct mcollect_t
    {
        hashtable_t hashTable; /* Hash table to manage this collection. keys are
                              char * and values are mcollect_measurement_t's */
    } mcollect_t, *mcollect_p;

    /* Public handle to an individual measurement of a measurement collection */
    typedef struct mcollect_value_t
    {
        int type; /* MC_TYPE_? #define of the measurement type */
        union
        {
            double dbl;
            long long i64;
            char *str;
            timelib64_t tstamp;
            timeseries_p tseries;
        } val;
    } mcollect_value_t, *mcollect_value_p;

    /*****************************************************************************/
    /*
 * Allocate a measurement collection and return it
 *
 * Returns: Pointer to measurement collection on success. Free with mcollect_destroy()
 *          NULL on error.
 *
 */
    mcollect_p mcollect_alloc(void);

    /*****************************************************************************/
    /*
 * Destroy a measurement collection that was allocated with mcollect_alloc
 *
 */
    void mcollect_destroy(mcollect_p mcollect);

    /*****************************************************************************/
    /*
 * Remove an entry from a measurement collection, freeing any resources it was
 * using
 *
 * Returns: MCOLLECT_ST_OK if removed
 *          MCOLLECT_ST_NOTFOUND if key not found
 *          MCOLLECT_ST_? #define on other error
 *
 */
    int mcollect_remove(mcollect_p mcollect, char *key);


    /*****************************************************************************/
    void mcollect_value_free(mcollect_value_p value);
    /*
 * Free a pointer to an allocated mcollect value, doing any cleanup related
 * to individual value types
 *
 * NOTE: Do NOT call this on a value if it is still in a mcollect_p structure
 *       mcollect will automatically free values that are known internally
 *
 */

    /*****************************************************************************/
    /*
 * mcollect_value_add_? variants
 *
 * Add a new value to the collection with "key" as the key. If the value at 'key'
 * does not exist yet, a new value is allocated, and it is set to 'defaultValue'
 * where applicable
 *
 * In the case of _string, defaultValue will be strdup'd, so you don't have to
 * pre-copy it
 *
 */
    mcollect_value_p mcollect_value_add_double(mcollect_p mcollect, char *key, double defaultValue);
    mcollect_value_p mcollect_value_add_int64(mcollect_p mcollect, char *key, long long defaultValue);
    mcollect_value_p mcollect_value_add_string(mcollect_p mcollect, char *key, char *defaultValue);
    mcollect_value_p mcollect_value_add_timestamp(mcollect_p mcollect, char *key, timelib64_t defaultValue);
    mcollect_value_p mcollect_value_add_timeseries_double(mcollect_p mcollect, char *key);
    mcollect_value_p mcollect_value_add_timeseries_int64(mcollect_p mcollect, char *key);
    mcollect_value_p mcollect_value_add_timeseries_string(mcollect_p mcollect, char *key);
    mcollect_value_p mcollect_value_add_timeseries_blob(mcollect_p mcollect, char *key);

    /*****************************************************************************/
    /*
 * Get a value from a collection
 *
 * Note that the mcollect_value_add_? variants already check for a value existing
 * and return it to you if they do, so this API is really only here for
 * completion
 *
 * Returns: Pointer to value if it exists
 *          NULL if value doesn't exist
 *
 */
    mcollect_value_p mcollect_value_get(mcollect_p mcollect, char *key);


    /*****************************************************************************/
    /*
 * Enumeration function to enumerate a measurement collection, calling this 
 * callback on every key value pair in the collection.
 *
 * Returns: 0 if ok. Keep enumerating
 *         >0 user provided error to return from mcollect_iterate
 *         <0 MCOLLECT_ST_? #define
 */
    typedef int (*mcollect_iterate_f)(char *key, mcollect_value_p value, void *userData);

    /*****************************************************************************/
    /*
 * Iterate over all of the key value pairs in a measurement collection, calling
 * the provided callback on each key value pair found. userData will be passed
 * to each callback call.
 *
 * Returns: 0 if OK
 *         >0 if the user callback returned an error
 *         <0 MCOLLECT_ST_? #define
 */
    int mcollect_iterate(mcollect_p mcollect, mcollect_iterate_f iterateCB, void *userData);

    /*****************************************************************************/
    int mcollect_type_is_timeseries(int mcType);
    /*
 * Returns whether or not the given MC_TYPE_? #define is a time series type
 * or not.
 *
 * Returns: 1 = is a timeseries type
 *          0 = is not a timeseries type
 *
 */

    /*****************************************************************************/
    /*
 * Get the number of measurements stored in this collection.
 *
 * Returns: >=0 for number of measurements currently stored in the collection
 */
    unsigned int mcollect_size(mcollect_p mcollect);

    /*****************************************************************************/
    /*
 * Calculate (roughly) the number of bytes in memory that this mcollect takes up.
 * This includes the size of any base structures and the size
 * of each of the values that are stored even if they are not in contiguous memory.
 */
    long long mcollect_key_bytes_used(mcollect_p mcollect, char *key);

    /*****************************************************************************/

#ifdef __cplusplus
}
#endif

#endif /* MEASUREMENTCOLLECTION_H */
