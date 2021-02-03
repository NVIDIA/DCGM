
#include "timeseries.h"
#include "logging.h"
#include "nvcmvalue.h"
#include <stdlib.h>
#include <string.h>

#ifdef _WINDOWS
// Silence window's error
// C4996: 'strdup': The POSIX name for this item is deprecated. Instead, use the
//        ISO C++ conformant name: _strdup. See online help for details.
#define strdup _strdup

// Windows strtok_r is under different name strtok_s (interface and behavior is the same)
#define strtok_r strtok_s

#endif // _WINDOWS

/*****************************************************************************/
static int timeseries_compareCB(timeseries_entry_p elem1, timeseries_entry_p elem2)
{
    /* Ascending time order */
    if (elem1->usecSince1970 < elem2->usecSince1970)
        return -1;
    else if (elem1->usecSince1970 > elem2->usecSince1970)
        return 1;
    return 0;
}

/*****************************************************************************/
static int timeseries_mergeCB(timeseries_entry_p current, timeseries_entry_p inserting, void *user)
{
    return KV_ST_DUPLICATE; /* Need to resolve */
}

/*****************************************************************************/
static void timeseries_freeCB(timeseries_entry_p elem, timeseries_p ts)
{
    /* Free any allocated memory within the element that is being freed */
    if (ts->tsType == TS_TYPE_STRING || ts->tsType == TS_TYPE_BLOB)
    {
        if (elem->val.ptr)
            free(elem->val.ptr);
        elem->val.ptr      = 0;
        elem->val2.ptrSize = 0;
    }
}

/*****************************************************************************/
void timeseries_destroy(timeseries_p ts)
{
    if (!ts)
        return;

    if (ts->keyedVector)
    {
        keyedvector_destroy(ts->keyedVector);
        ts->keyedVector = 0;
    }

    free(ts);
}

/*****************************************************************************/
timeseries_p timeseries_alloc(int tsType, int *errorSt)
{
    timeseries_p ts = 0;
    int kvErrorSt;

    if (!errorSt)
        return NULL;

    *errorSt = TS_ST_OK;

    if (tsType <= TS_TYPE_UNKNOWN || tsType > TS_TYPE_MAX)
    {
        *errorSt = TS_ST_BADPARAM;
        return NULL;
    }

    ts = (timeseries_p)malloc(sizeof(*ts));
    if (!ts)
    {
        *errorSt = TS_ST_MEMORY;
        return NULL;
    }
    memset(ts, 0, sizeof(*ts));

    ts->tsType = tsType;

    /* allocate the keyedvector */
    ts->keyedVector = keyedvector_alloc(sizeof(timeseries_entry_t),
                                        0,
                                        (kv_compare_f)timeseries_compareCB,
                                        (kv_merge_f)timeseries_mergeCB,
                                        (kv_free_f)timeseries_freeCB,
                                        ts,
                                        &kvErrorSt);
    if (!ts->keyedVector)
    {
        PRINT_ERROR("%d", "Error %d from keyedvector_alloc\n", kvErrorSt);
        timeseries_destroy(ts);
        ts = 0;
    }

    return ts;
}

/*****************************************************************************/
int timeseries_size(timeseries_p ts)
{
    if (!ts || !ts->keyedVector)
        return 0;

    return keyedvector_size(ts->keyedVector);
}

/*****************************************************************************/
static int timeseries_insert(timeseries_p ts, timeseries_entry_p entry)
{
    int tries;
    int maxTries = 10000; /* infinite loops are bad */
    int insertSt;
    kv_cursor_t cursor;

    if (!entry->usecSince1970)
        entry->usecSince1970 = timelib_usecSince1970();

    for (tries = 0; tries < maxTries; tries++)
    {
        insertSt = keyedvector_insert(ts->keyedVector, entry, &cursor);
        if (!insertSt)
            return TS_ST_OK; /* Success */
        else if (insertSt != KV_ST_DUPLICATE)
        {
            PRINT_ERROR("%d %ld",
                        "Error %d from keyedvector_insert ts %ld\n",
                        insertSt,
                        entry->usecSince1970);
            return TS_ST_UNKNOWN;
        }

        /* Was a duplicate. Try again */
        entry->usecSince1970++;
    }

    return TS_ST_UNKNOWN;
}

/*****************************************************************************/
int timeseries_insert_int64(timeseries_p ts, timelib64_t timestamp, long long value1, long long value2)
{
    int retSt;
    timeseries_entry_t entry;

    if (!ts)
        return TS_ST_BADPARAM;
    if (ts->tsType != TS_TYPE_INT64)
        return TS_ST_WRONGTYPE;

    entry.usecSince1970 = timestamp;
    entry.val.i64       = value1;
    entry.val2.i64      = value2;
    retSt               = timeseries_insert(ts, &entry);
    return retSt;
}

/*****************************************************************************/
int timeseries_insert_string(timeseries_p ts, timelib64_t timestamp, char *value)
{
    int retSt;
    timeseries_entry_t entry;

    if (!ts || !value)
        return TS_ST_BADPARAM;
    if (ts->tsType != TS_TYPE_STRING)
        return TS_ST_WRONGTYPE;

    entry.usecSince1970 = timestamp;
    entry.val.ptr       = strdup(value);
    entry.val2.ptrSize  = strlen(value) + 1;
    retSt               = timeseries_insert(ts, &entry);
    return retSt;
}

/*****************************************************************************/
int timeseries_insert_blob(timeseries_p ts, timelib64_t timestamp, void *value, int valueSize)
{
    int retSt;
    timeseries_entry_t entry;

    if (!ts || !value || valueSize < 1)
        return TS_ST_BADPARAM;
    if (ts->tsType != TS_TYPE_BLOB)
        return TS_ST_WRONGTYPE;

    entry.usecSince1970 = timestamp;
    entry.val.ptr       = malloc(valueSize);
    if (!entry.val.ptr)
        return TS_ST_MEMORY;

    memcpy(entry.val.ptr, value, valueSize);

    entry.val2.ptrSize = valueSize;
    retSt              = timeseries_insert(ts, &entry);
    return retSt;
}

/*****************************************************************************/
int timeseries_insert_int64_coerce(timeseries_p ts, timelib64_t timestamp, long long value1, long long value2)
{
    int retSt;
    timeseries_entry_t entry;

    if (!ts)
        return TS_ST_BADPARAM;

    switch (ts->tsType)
    {
        case TS_TYPE_INT64:
            break; /* Fall through */

        case TS_TYPE_DOUBLE: {
            double doubleValue1, doubleValue2;

            doubleValue1 = nvcmvalue_int64_to_double(value1);
            doubleValue2 = nvcmvalue_int64_to_double(value2);
            return timeseries_insert_double(ts, timestamp, doubleValue1, doubleValue2);
        }

        default:
            return TS_ST_WRONGTYPE; /* No coercion available */
    }

    entry.usecSince1970 = timestamp;
    entry.val.i64       = value1;
    entry.val2.i64      = value2;
    retSt               = timeseries_insert(ts, &entry);
    return retSt;
}

/*****************************************************************************/
int timeseries_insert_double(timeseries_p ts, timelib64_t timestamp, double value1, double value2)
{
    int retSt;
    timeseries_entry_t entry;

    if (!ts || !ts->keyedVector)
        return TS_ST_BADPARAM;
    if (ts->tsType != TS_TYPE_DOUBLE)
        return TS_ST_WRONGTYPE;

    entry.usecSince1970 = timestamp;
    entry.val.dbl       = value1;
    entry.val2.dbl      = value2;
    retSt               = timeseries_insert(ts, &entry);
    return retSt;
}

/*****************************************************************************/
int timeseries_insert_double_coerce(timeseries_p ts, timelib64_t timestamp, double value1, double value2)
{
    int retSt;
    timeseries_entry_t entry;

    if (!ts || !ts->keyedVector)
        return TS_ST_BADPARAM;

    switch (ts->tsType)
    {
        case TS_TYPE_DOUBLE:
            break; /* Fall through */

        case TS_TYPE_INT64: {
            long long i64Value1 = nvcmvalue_double_to_int64(value1);
            long long i64Value2 = nvcmvalue_double_to_int64(value2);
            return timeseries_insert_int64(ts, timestamp, i64Value1, i64Value2);
        }

        default:
            return TS_ST_WRONGTYPE; /* No coercion available */
    }

    entry.usecSince1970 = timestamp;
    entry.val.dbl       = value1;
    entry.val2.dbl      = value2;
    retSt               = timeseries_insert(ts, &entry);
    return retSt;
}

/*****************************************************************************/
int timeseries_enforce_quota(timeseries_p ts, timelib64_t oldestKeepTimestamp, int maxKeepEntries)
{
    timeseries_entry_t key, *elem;
    kv_cursor_t firstElemCursor;
    kv_cursor_t lastDeleteCursor;
    int st;
    int currentCount, NtoDelete;

    if (!ts || !ts->keyedVector)
        return TS_ST_BADPARAM;

    memset(&key, 0, sizeof(key));

    elem = (timeseries_entry_t *)keyedvector_first(ts->keyedVector, &firstElemCursor);
    if (!elem)
        return TS_ST_OK; /* Nothing to do */

    /* Timestamp enforcement */
    if (oldestKeepTimestamp && elem->usecSince1970 < oldestKeepTimestamp)
    {
        key.usecSince1970 = oldestKeepTimestamp;
        elem = (timeseries_entry_t *)keyedvector_find_by_key(ts->keyedVector, &key, KV_LGE_LESS, &lastDeleteCursor);
        /* If elem is null, there is nothing to free less than our timestamp */
        if (elem)
        {
            st = keyedvector_remove_range_by_cursor(ts->keyedVector, &firstElemCursor, &lastDeleteCursor);
            if (st)
                return st;
        }
    }

    /* Check (or recheck) the size to make sure we're under our quota */

    if (!maxKeepEntries)
        return TS_ST_OK;

    /* Under our quota? */
    currentCount = keyedvector_size(ts->keyedVector);
    if (currentCount <= maxKeepEntries)
        return TS_ST_OK;

    NtoDelete = currentCount - maxKeepEntries;

    /* Refind the first element in case elements were deleted above */
    elem = (timeseries_entry_t *)keyedvector_first(ts->keyedVector, &firstElemCursor);
    if (!elem)
        return TS_ST_OK; /* Nothing to do */

    elem = (timeseries_entry_t *)keyedvector_find_by_index(ts->keyedVector, NtoDelete - 1, &lastDeleteCursor);
    if (!elem)
        return TS_ST_OK; /* Nothing to do */

    /* Delete the elements from the first to elem[NtoDelete] */
    st = keyedvector_remove_range_by_cursor(ts->keyedVector, &firstElemCursor, &lastDeleteCursor);
    if (st)
        return st;

    return TS_ST_OK;
}

/*****************************************************************************/
/* Simple one-return calculations from startTime to endTime */
#define TS_CALC_SUM 0
#define TS_CALC_MIN 1
#define TS_CALC_MAX 2

/*****************************************************************************/
static long long
timeseries_simple_calc_int64(timeseries_p ts, int tsCalc, timelib64_t startTime, timelib64_t endTime, int *errorSt)
{
    long long retVal = TS_EMPTY_INT64;
    int Nsamples     = 0;
    kv_cursor_t cursor;
    timeseries_entry_t searchEntry;
    timeseries_entry_p elem;

    if (!errorSt)
        return TS_EMPTY_INT64;
    if (!ts || !ts->keyedVector)
    {
        *errorSt = TS_ST_BADPARAM;
        return TS_EMPTY_INT64;
    }
    if (ts->tsType != TS_TYPE_INT64)
    {
        *errorSt = TS_ST_WRONGTYPE;
        return TS_EMPTY_INT64;
    }

    /* Get the starting iteration point */
    if (startTime)
    {
        searchEntry.usecSince1970 = startTime;
        elem = (timeseries_entry_p)keyedvector_find_by_key(ts->keyedVector, &searchEntry, KV_LGE_GREATEQUAL, &cursor);
    }
    else
    {
        elem = (timeseries_entry_p)keyedvector_first(ts->keyedVector, &cursor);
    }

    if (!elem)
    {
        *errorSt = TS_ST_NODATA;
        return retVal; /* No records >= start time. Easy enough */
    }

    for (; elem; elem = (timeseries_entry_p)keyedvector_next(ts->keyedVector, &cursor))
    {
        /* Past end of our time searching range? */
        if (endTime && elem->usecSince1970 > endTime)
            break;

        /* Ignore blank values */
        if (DCGM_INT64_IS_BLANK(elem->val.i64))
            continue;

        Nsamples++;

        switch (tsCalc)
        {
            case TS_CALC_SUM:
                if (retVal == TS_EMPTY_INT64)
                    retVal = 0;
                retVal += elem->val.i64;
                break;
            case TS_CALC_MIN:
                if (retVal == TS_EMPTY_INT64 || elem->val.i64 < retVal)
                    retVal = elem->val.i64;
                break;
            case TS_CALC_MAX:
                if (retVal == TS_EMPTY_INT64 || elem->val.i64 > retVal)
                    retVal = elem->val.i64;
                break;
            default:
                *errorSt = TS_ST_BADPARAM;
        }
    }

    return retVal;
}

/*****************************************************************************/
static double
timeseries_simple_calc_double(timeseries_p ts, int tsCalc, timelib64_t startTime, timelib64_t endTime, int *errorSt)
{
    double retVal = TS_EMPTY_DOUBLE;
    int Nsamples  = 0;
    kv_cursor_t cursor;
    timeseries_entry_t searchEntry;
    timeseries_entry_p elem;

    if (!errorSt)
        return TS_EMPTY_DOUBLE;
    if (!ts || !ts->keyedVector)
    {
        *errorSt = TS_ST_BADPARAM;
        return TS_EMPTY_DOUBLE;
    }
    if (ts->tsType != TS_TYPE_DOUBLE)
    {
        *errorSt = TS_ST_WRONGTYPE;
        return TS_EMPTY_DOUBLE;
    }

    /* Get the starting iteration point */
    if (startTime)
    {
        searchEntry.usecSince1970 = startTime;
        elem = (timeseries_entry_p)keyedvector_find_by_key(ts->keyedVector, &searchEntry, KV_LGE_GREATEQUAL, &cursor);
    }
    else
    {
        elem = (timeseries_entry_p)keyedvector_first(ts->keyedVector, &cursor);
    }

    if (!elem)
    {
        *errorSt = TS_ST_NODATA;
        return retVal; /* No records >= start time. Easy enough */
    }

    for (; elem; elem = (timeseries_entry_p)keyedvector_next(ts->keyedVector, &cursor))
    {
        /* Past end of our time searching range? */
        if (endTime && elem->usecSince1970 > endTime)
            break;

        /* Ignore blank values */
        if (DCGM_FP64_IS_BLANK(elem->val.dbl))
            continue;

        Nsamples++;

        switch (tsCalc)
        {
            case TS_CALC_SUM:
                if (retVal == TS_EMPTY_DOUBLE)
                    retVal = 0.0;
                retVal += elem->val.dbl;
                break;
            case TS_CALC_MIN:
                if (retVal == TS_EMPTY_DOUBLE || elem->val.dbl < retVal)
                    retVal = elem->val.dbl;
                break;
            case TS_CALC_MAX:
                if (retVal == TS_EMPTY_DOUBLE || elem->val.dbl > retVal)
                    retVal = elem->val.dbl;
                break;
            default:
                *errorSt = TS_ST_BADPARAM;
        }
    }

    return retVal;
}

/*****************************************************************************/
long long timeseries_sum_int64(timeseries_p ts, timelib64_t startTime, timelib64_t endTime, int *errorSt)
{
    return timeseries_simple_calc_int64(ts, TS_CALC_SUM, startTime, endTime, errorSt);
}

/*****************************************************************************/
double timeseries_sum_double(timeseries_p ts, timelib64_t startTime, timelib64_t endTime, int *errorSt)
{
    return timeseries_simple_calc_double(ts, TS_CALC_SUM, startTime, endTime, errorSt);
}

/*****************************************************************************/
double timeseries_average(timeseries_p ts, timelib64_t startTime, timelib64_t endTime, int *errorSt)
{
    double retVal = 0;
    int Nsamples  = 0;
    kv_cursor_t cursor;
    timeseries_entry_t searchEntry;
    timeseries_entry_p elem;

    if (!errorSt)
        return TS_EMPTY_DOUBLE;
    if (!ts || !ts->keyedVector)
    {
        *errorSt = TS_ST_BADPARAM;
        return TS_EMPTY_DOUBLE;
    }
    if (ts->tsType != TS_TYPE_INT64 && ts->tsType != TS_TYPE_DOUBLE)
    {
        *errorSt = TS_ST_WRONGTYPE;
        return TS_EMPTY_DOUBLE;
    }

    /* Get the starting iteration point */
    if (startTime)
    {
        searchEntry.usecSince1970 = startTime;
        elem = (timeseries_entry_p)keyedvector_find_by_key(ts->keyedVector, &searchEntry, KV_LGE_GREATEQUAL, &cursor);
    }
    else
    {
        elem = (timeseries_entry_p)keyedvector_first(ts->keyedVector, &cursor);
    }

    if (!elem)
    {
        *errorSt = TS_ST_NODATA;
        return TS_EMPTY_DOUBLE; /* Undefined if no samples. Beats dividing by 0 */
    }
    for (; elem; elem = (timeseries_entry_p)keyedvector_next(ts->keyedVector, &cursor))
    {
        /* Past end of our time searching range? */
        if (endTime && elem->usecSince1970 > endTime)
            break;

        if (ts->tsType == TS_TYPE_DOUBLE)
        {
            /* Ignore blank values */
            if (DCGM_FP64_IS_BLANK(elem->val.dbl))
                continue;
            retVal += elem->val.dbl;
        }
        else if (ts->tsType == TS_TYPE_INT64)
        {
            /* Ignore blank values */
            if (DCGM_INT64_IS_BLANK(elem->val.i64))
                continue;
            retVal += ((double)elem->val.i64);
        }

        Nsamples++;
    }

    if (!Nsamples)
    {
        *errorSt = TS_ST_NODATA;
        return TS_EMPTY_DOUBLE; /* Undefined if no samples. Beats dividing by 0 */
    }

    retVal /= ((double)Nsamples);
    return retVal;
}

/*****************************************************************************/
double timeseries_moving_average(timeseries_p ts, timelib64_t endTime, int maxSamples, int *errorSt)
{
    double retVal = 0;
    int Nsamples  = 0;
    kv_cursor_t cursor;
    timeseries_entry_t searchEntry;
    timeseries_entry_p elem;

    if (!errorSt)
        return TS_EMPTY_DOUBLE;
    if (!ts || !ts->keyedVector || maxSamples < 0)
    {
        *errorSt = TS_ST_BADPARAM;
        return TS_EMPTY_DOUBLE;
    }
    if (ts->tsType != TS_TYPE_INT64 && ts->tsType != TS_TYPE_DOUBLE)
    {
        *errorSt = TS_ST_WRONGTYPE;
        return TS_EMPTY_DOUBLE;
    }

    /* Get the starting iteration point */
    if (endTime)
    {
        searchEntry.usecSince1970 = endTime;
        elem = (timeseries_entry_p)keyedvector_find_by_key(ts->keyedVector, &searchEntry, KV_LGE_LESSEQUAL, &cursor);
    }
    else
    {
        elem = (timeseries_entry_p)keyedvector_last(ts->keyedVector, &cursor);
    }

    if (!elem)
    {
        *errorSt = TS_ST_NODATA;
        return TS_EMPTY_DOUBLE; /* Undefined if no samples. Beats dividing by 0 */
    }

    for (; elem; elem = (timeseries_entry_p)keyedvector_prev(ts->keyedVector, &cursor))
    {
        /* Collected enough samples yet? */
        if (Nsamples >= maxSamples)
            break;

        if (ts->tsType == TS_TYPE_DOUBLE)
        {
            /* Ignore blank values */
            if (DCGM_FP64_IS_BLANK(elem->val.dbl))
                continue;
            retVal += elem->val.dbl;
        }
        else if (ts->tsType == TS_TYPE_INT64)
        {
            /* Ignore blank values */
            if (DCGM_INT64_IS_BLANK(elem->val.i64))
                continue;
            retVal += ((double)elem->val.i64);
        }

        Nsamples++;
    }

    if (!Nsamples)
    {
        *errorSt = TS_ST_NODATA;
        return TS_EMPTY_DOUBLE; /* Undefined if no samples. Beats dividing by 0 */
    }

    retVal /= ((double)Nsamples);
    return retVal;
}

/*****************************************************************************/
long long timeseries_min_int64(timeseries_p ts, timelib64_t startTime, timelib64_t endTime, int *errorSt)
{
    return timeseries_simple_calc_int64(ts, TS_CALC_MIN, startTime, endTime, errorSt);
}

/*****************************************************************************/
double timeseries_min_double(timeseries_p ts, timelib64_t startTime, timelib64_t endTime, int *errorSt)
{
    return timeseries_simple_calc_double(ts, TS_CALC_MIN, startTime, endTime, errorSt);
}

/*****************************************************************************/
long long timeseries_max_int64(timeseries_p ts, timelib64_t startTime, timelib64_t endTime, int *errorSt)
{
    return timeseries_simple_calc_int64(ts, TS_CALC_MAX, startTime, endTime, errorSt);
}

/*****************************************************************************/
double timeseries_max_double(timeseries_p ts, timelib64_t startTime, timelib64_t endTime, int *errorSt)
{
    return timeseries_simple_calc_double(ts, TS_CALC_MAX, startTime, endTime, errorSt);
}

/*****************************************************************************/
int timeseries_threshold_count(timeseries_p ts,
                               timelib64_t startTime,
                               timelib64_t endTime,
                               double threshold,
                               int relation)
{
    double val          = 0;
    int Nsamples        = 0;
    int NmatchedSamples = 0;
    kv_cursor_t cursor;
    timeseries_entry_t searchEntry;
    timeseries_entry_p elem;

    if (!ts || !ts->keyedVector)
        return TS_ST_BADPARAM;

    /* Get the starting iteration point */
    if (startTime)
    {
        searchEntry.usecSince1970 = startTime;
        elem = (timeseries_entry_p)keyedvector_find_by_key(ts->keyedVector, &searchEntry, KV_LGE_GREATEQUAL, &cursor);
    }
    else
    {
        elem = (timeseries_entry_p)keyedvector_first(ts->keyedVector, &cursor);
    }

    if (!elem)
        return 0; /* No data */

    for (; elem; elem = (timeseries_entry_p)keyedvector_next(ts->keyedVector, &cursor))
    {
        /* Past end of our time searching range? */
        if (endTime && elem->usecSince1970 > endTime)
            break;

        if (ts->tsType == TS_TYPE_DOUBLE)
        {
            /* Ignore blank values */
            if (DCGM_FP64_IS_BLANK(elem->val.dbl))
                continue;
            val = elem->val.dbl;
        }
        else if (ts->tsType == TS_TYPE_INT64)
        {
            /* Ignore blank values */
            if (DCGM_INT64_IS_BLANK(elem->val.i64))
                continue;
            val = ((double)elem->val.i64);
        }

        Nsamples++;

        switch (relation)
        {
            default:
            case TS_REL_EQUAL:
                if (val == threshold)
                    NmatchedSamples++;
                break;
            case TS_REL_LESSEQUAL:
                if (val <= threshold)
                    NmatchedSamples++;
                break;
            case TS_REL_LESS:
                if (val < threshold)
                    NmatchedSamples++;
                break;
            case TS_REL_GREATEQUAL:
                if (val >= threshold)
                    NmatchedSamples++;
                break;
            case TS_REL_GREATER:
                if (val > threshold)
                    NmatchedSamples++;
                break;
        }
    }

    return NmatchedSamples;
}

/*****************************************************************************/
long long timeseries_bytes_used(timeseries_p ts)
{
    if (!ts)
        return 0;

    long long bytesUsed = sizeof(*ts) + keyedvector_bytes_used(ts->keyedVector);

    return bytesUsed;
}

/*****************************************************************************/
timeseries_entry_p timeseries_first(timeseries_p ts, timeseries_cursor_p cursor)
{
    timeseries_cursor_t tempCursor;
    if (NULL == cursor)
        cursor = &tempCursor;
    return (timeseries_entry_p)keyedvector_first(ts->keyedVector, cursor);
}

/*****************************************************************************/
timeseries_entry_p timeseries_last(timeseries_p ts, timeseries_cursor_p cursor)
{
    timeseries_cursor_t tempCursor;
    if (NULL == cursor)
        cursor = &tempCursor;
    return (timeseries_entry_p)keyedvector_last(ts->keyedVector, cursor);
}

/*****************************************************************************/
timeseries_entry_p timeseries_next(timeseries_p ts, timeseries_cursor_p cursor)
{
    return (timeseries_entry_p)keyedvector_next(ts->keyedVector, cursor);
}

/*****************************************************************************/
timeseries_entry_p timeseries_prev(timeseries_p ts, timeseries_cursor_p cursor)
{
    return (timeseries_entry_p)keyedvector_prev(ts->keyedVector, cursor);
}

/*****************************************************************************/
timeseries_entry_p timeseries_find(timeseries_p ts, timelib64_t time, int findOp, timeseries_cursor_p cursor)
{
    timeseries_cursor_t tempCursor;
    if (NULL == cursor)
        cursor = &tempCursor;
    return (timeseries_entry_p)keyedvector_find_by_key(ts->keyedVector, &time, findOp, cursor);
}
