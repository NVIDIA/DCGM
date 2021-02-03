
#include "measurementcollection.h"
#include "MurmurHash3.h"
#include "logging.h"
#include <stdlib.h>
#include <string.h>

/*****************************************************************************/
/*
 * Hash function for measurementcollection keys (char *)
 *
 */
static unsigned int mc_keyHashCB(const void *key)
{
    char *val = (char *)key;
    int len   = (int)strlen(val);
    unsigned int retVal;

    MurmurHash3_x86_32(val, len, 0, &retVal);
    return retVal;
}

/*****************************************************************************/
/*
 * Key compare function
 *
 * Hashtable wants a non-zero return for equal keys, so invert strcmp()'s return
 */
static int mc_keyCmpCB(const void *key1, const void *key2)
{
    int st = strcmp((const char *)key1, (const char *)key2); /* Case sensitive, just like the hash function */

    return (!st); /* Inverted on purpose */
}

/*****************************************************************************/
/*
 * Key free function
 *
 */
static void mc_keyFreeCB(void *key)
{
    /* Since this is a malloc'd string, free it */
    free(key);
}

/*****************************************************************************/
/*
 * Value free function
 *
 */
static void mc_valueFreeCB(void *value)
{
    mcollect_value_free((mcollect_value_p)value);
}

/*****************************************************************************/
mcollect_p mcollect_alloc(void)
{
    mcollect_p mcollect = NULL;
    int st;

    mcollect = (mcollect_p)malloc(sizeof(*mcollect));
    if (!mcollect)
        return NULL;

    memset(mcollect, 0, sizeof(*mcollect));

    st = hashtable_init(&mcollect->hashTable, mc_keyHashCB, mc_keyCmpCB, mc_keyFreeCB, mc_valueFreeCB);
    if (st)
    {
        PRINT_ERROR("%d", "error %d from hashtable_init()\n", st);
        free(mcollect);
        mcollect = NULL;
        return NULL;
    }

    return mcollect;
}

/*****************************************************************************/
void mcollect_destroy(mcollect_p mcollect)
{
    if (!mcollect)
        return;

    hashtable_close(&mcollect->hashTable);

    free(mcollect);
}

/*****************************************************************************/
void mcollect_value_free(mcollect_value_p value)
{
    /* Free values */

    /* Do special per-value freeing */
    switch (value->type)
    {
        case MC_TYPE_STRING:
            if (value->val.str)
                free(value->val.str);
            value->val.str = 0;
            break;

        case MC_TYPE_TIMESERIES_DOUBLE:
        case MC_TYPE_TIMESERIES_INT64:
        case MC_TYPE_TIMESERIES_STRING:
        case MC_TYPE_TIMESERIES_BLOB:
            if (value->val.tseries)
                timeseries_destroy(value->val.tseries);
            value->val.tseries = 0;
            break;
        default:
            break;
    }

    free(value);
}

/*****************************************************************************/
mcollect_value_p mcollect_value_get(mcollect_p mcollect, char *key)
{
    if (!mcollect || !key)
        return NULL;

    return (mcollect_value_p)hashtable_get(&mcollect->hashTable, key);
}

/*****************************************************************************/
/*
 * Add or get a value
 *
 */
static mcollect_value_p mcollect_add_or_get(mcollect_p mcollect, char *key, int type, int *errorSt)
{
    mcollect_value_p value = 0;
    int st;

    if (!errorSt)
        return NULL;
    if (!mcollect || !key || type == MC_TYPE_UNKNOWN)
    {
        *errorSt = MCOLLECT_ST_BADPARAM;
        return NULL;
    }
    *errorSt = MCOLLECT_ST_OK;

    value = (mcollect_value_p)hashtable_get(&mcollect->hashTable, key);
    if (value)
    {
        /* Already exists. Return the current value */
        *errorSt = MCOLLECT_ST_EXISTS;
        return value;
    }

    /* Need to copy the key since the hashtable just contains a pointer to it */
    key = strdup(key);
    if (!key)
    {
        *errorSt = MCOLLECT_ST_MEMORY;
        return 0;
    }

    value = (mcollect_value_p)malloc(sizeof(*value));
    if (!value)
    {
        free(key);
        *errorSt = MCOLLECT_ST_MEMORY;
        return 0;
    }

    memset(value, 0, sizeof(*value));
    value->type = type;
    st          = hashtable_set(&mcollect->hashTable, key, value);
    if (st)
    {
        PRINT_ERROR("%d %s", "Error %d from hashtable_set key %s\n", st, key);
        free(value);
        value    = 0;
        *errorSt = MCOLLECT_ST_MEMORY;
        return 0;
    }

    return value;
}

/*****************************************************************************/
mcollect_value_p mcollect_value_add_double(mcollect_p mcollect, char *key, double defaultValue)
{
    mcollect_value_p value;
    int errorSt;


    value = mcollect_add_or_get(mcollect, key, MC_TYPE_DOUBLE, &errorSt);
    if (!value)
    {
        PRINT_ERROR("%d %s", "Error %d from mcollect_add_or_get of %s\n", errorSt, key);
        return NULL;
    }

    if (errorSt == MCOLLECT_ST_EXISTS)
        return value;

    /* New value. Set to default */
    value->val.dbl = defaultValue;
    return value;
}

/*****************************************************************************/
mcollect_value_p mcollect_value_add_int64(mcollect_p mcollect, char *key, long long defaultValue)
{
    mcollect_value_p value;
    int errorSt;


    value = mcollect_add_or_get(mcollect, key, MC_TYPE_INT64, &errorSt);
    if (!value)
    {
        PRINT_ERROR("%d %s", "Error %d from mcollect_add_or_get of %s\n", errorSt, key);
        return NULL;
    }

    if (errorSt == MCOLLECT_ST_EXISTS)
        return value;

    /* New value. Set to default */
    value->val.i64 = defaultValue;
    return value;
}

/*****************************************************************************/
mcollect_value_p mcollect_value_add_string(mcollect_p mcollect, char *key, char *defaultValue)
{
    mcollect_value_p value;
    int errorSt;

    value = mcollect_add_or_get(mcollect, key, MC_TYPE_STRING, &errorSt);
    if (!value)
    {
        PRINT_ERROR("%d %s", "Error %d from mcollect_add_or_get of %s\n", errorSt, key);
        return NULL;
    }

    if (errorSt == MCOLLECT_ST_EXISTS)
        return value;

    /* New value. Set to default */
    if (defaultValue)
    {
        value->val.str = strdup(defaultValue);
        if (!value->val.str)
        {
            PRINT_ERROR("%s %s", "Unable to copy defaultValue string %s for key %s\n", defaultValue, key);
            /* We could delete the key again, but I don't think that's what
             * we want considering the user can provide a null default value
             * in the first place
             */
        }
    }

    return value;
}

/*****************************************************************************/
mcollect_value_p mcollect_value_add_timestamp(mcollect_p mcollect, char *key, timelib64_t defaultValue)
{
    mcollect_value_p value;
    int errorSt;


    value = mcollect_add_or_get(mcollect, key, MC_TYPE_TIMESTAMP, &errorSt);
    if (!value)
    {
        PRINT_ERROR("%d %s", "Error %d from mcollect_add_or_get of %s\n", errorSt, key);
        return NULL;
    }

    if (errorSt == MCOLLECT_ST_EXISTS)
        return value;

    /* New value. Set to default */
    value->val.tstamp = defaultValue;
    return value;
}

/*****************************************************************************/
mcollect_value_p mcollect_value_add_timeseries_double(mcollect_p mcollect, char *key)
{
    mcollect_value_p value;
    int errorSt;
    int tsError;

    value = mcollect_add_or_get(mcollect, key, MC_TYPE_TIMESERIES_DOUBLE, &errorSt);
    if (!value)
    {
        PRINT_ERROR("%d %s", "Error %d from mcollect_add_or_get of %s\n", errorSt, key);
        return NULL;
    }

    if (errorSt == MCOLLECT_ST_EXISTS)
        return value;

    /* New value. allocate the keyedvector */
    value->val.tseries = timeseries_alloc(TS_TYPE_DOUBLE, &tsError);
    if (!value->val.tseries)
    {
        mcollect_remove(mcollect, key);
        PRINT_ERROR("%d", "Error %d from timeseries_alloc\n", tsError);
        return NULL;
    }
    return value;
}

/*****************************************************************************/
mcollect_value_p mcollect_value_add_timeseries_int64(mcollect_p mcollect, char *key)
{
    mcollect_value_p value;
    int errorSt;
    int tsError;

    value = mcollect_add_or_get(mcollect, key, MC_TYPE_TIMESERIES_INT64, &errorSt);
    if (!value)
    {
        PRINT_ERROR("%d %s", "Error %d from mcollect_add_or_get of %s\n", errorSt, key);
        return NULL;
    }

    if (errorSt == MCOLLECT_ST_EXISTS)
        return value;

    /* New value. allocate the keyedvector */
    value->val.tseries = timeseries_alloc(TS_TYPE_INT64, &tsError);
    if (!value->val.tseries)
    {
        mcollect_remove(mcollect, key);
        PRINT_ERROR("%d", "Error %d from timeseries_alloc\n", tsError);
        return NULL;
    }
    return value;
}

/*****************************************************************************/
mcollect_value_p mcollect_value_add_timeseries_string(mcollect_p mcollect, char *key)
{
    mcollect_value_p value;
    int errorSt;
    int tsError;

    value = mcollect_add_or_get(mcollect, key, MC_TYPE_TIMESERIES_STRING, &errorSt);
    if (!value)
    {
        PRINT_ERROR("%d %s", "Error %d from mcollect_add_or_get of %s\n", errorSt, key);
        return NULL;
    }

    if (errorSt == MCOLLECT_ST_EXISTS)
        return value;

    /* New value. allocate the keyedvector */
    value->val.tseries = timeseries_alloc(TS_TYPE_STRING, &tsError);
    if (!value->val.tseries)
    {
        mcollect_remove(mcollect, key);
        PRINT_ERROR("%d", "Error %d from timeseries_alloc\n", tsError);
        return NULL;
    }
    return value;
}

/*****************************************************************************/
mcollect_value_p mcollect_value_add_timeseries_blob(mcollect_p mcollect, char *key)
{
    mcollect_value_p value;
    int errorSt;
    int tsError;

    value = mcollect_add_or_get(mcollect, key, MC_TYPE_TIMESERIES_BLOB, &errorSt);
    if (!value)
    {
        PRINT_ERROR("%d %s", "Error %d from mcollect_add_or_get of %s\n", errorSt, key);
        return NULL;
    }

    if (errorSt == MCOLLECT_ST_EXISTS)
        return value;

    /* New value. allocate the keyedvector */
    value->val.tseries = timeseries_alloc(TS_TYPE_BLOB, &tsError);
    if (!value->val.tseries)
    {
        mcollect_remove(mcollect, key);
        PRINT_ERROR("%d", "Error %d from timeseries_alloc\n", tsError);
        return NULL;
    }
    return value;
}

/*****************************************************************************/
int mcollect_remove(mcollect_p mcollect, char *key)
{
    int hashSt;

    if (!mcollect)
        return MCOLLECT_ST_BADPARAM;

    hashSt = hashtable_del(&mcollect->hashTable, key);
    if (hashSt)
        return MCOLLECT_ST_NOTFOUND;

    return MCOLLECT_ST_OK;
}

/*****************************************************************************/
int mcollect_iterate(mcollect_p mcollect, mcollect_iterate_f iterateCB, void *userData)
{
    void *hashtableIter;
    char *key;
    int cbSt;
    mcollect_value_p value;

    if (!mcollect)
        return MCOLLECT_ST_BADPARAM;

    for (hashtableIter = hashtable_iter(&mcollect->hashTable); hashtableIter;
         hashtableIter = hashtable_iter_next(&mcollect->hashTable, hashtableIter))
    {
        key   = (char *)hashtable_iter_key(hashtableIter);
        value = (mcollect_value_p)hashtable_iter_value(hashtableIter);

        cbSt = iterateCB(key, value, userData);
        if (cbSt)
            return cbSt; /* Callback requested exit */
    }

    return MCOLLECT_ST_OK;
}

/*****************************************************************************/
int mcollect_type_is_timeseries(int mcType)
{
    switch (mcType)
    {
        case MC_TYPE_TIMESERIES_DOUBLE: /* intentional fall-through */
        case MC_TYPE_TIMESERIES_INT64:
        case MC_TYPE_TIMESERIES_STRING:
        case MC_TYPE_TIMESERIES_BLOB:
            return 1;

        case MC_TYPE_INT64: /* intentional fall-through */
        case MC_TYPE_DOUBLE:
        case MC_TYPE_STRING:
        case MC_TYPE_TIMESTAMP:
        default:
            return 0;
    }

    return 0; /* Shouldn't get here */
}

/*****************************************************************************/
unsigned int mcollect_size(mcollect_p mcollect)
{
    return mcollect->hashTable.size;
}

static int mcollect_timeseries_size_cb(char *key, mcollect_value_p value, void *userData)
{
    long long *bytesUsed = (long long *)userData;

    if (value && mcollect_type_is_timeseries(value->type))
    {
        *bytesUsed += timeseries_bytes_used(value->val.tseries);
    }

    return 0;
}

/*****************************************************************************/
long long mcollect_key_bytes_used(mcollect_p mcollect, char *key)
{
    long long bytesUsed = 0;

    mcollect_value_p value = mcollect_value_get(mcollect, key);

    // key is not present in this mcollect
    if (!value)
        return 0;

    bytesUsed += sizeof(mcollect_value_t);

    if (mcollect_type_is_timeseries(value->type))
        bytesUsed += timeseries_bytes_used(value->val.tseries);

    return bytesUsed;
}

/*****************************************************************************/
