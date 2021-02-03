
#include "nvcmvalue.h"

/*****************************************************************************/
int nvcmvalue_int64_to_int32(long long int64Value)
{
    if (!DCGM_INT64_IS_BLANK(int64Value))
        return (int)int64Value;

    switch (int64Value)
    {
        case DCGM_INT64_NOT_FOUND:
            return DCGM_INT32_NOT_FOUND;

        case DCGM_INT64_NOT_SUPPORTED:
            return DCGM_INT32_NOT_SUPPORTED;

        case DCGM_INT64_NOT_PERMISSIONED:
            return DCGM_INT32_NOT_PERMISSIONED;

        case DCGM_INT64_BLANK:
        /* Intentional fall-through to handle new values added later */
        default:
            return DCGM_INT32_BLANK;
    }

    /* Shouldn't get here */
    return DCGM_INT32_BLANK;
}

/*****************************************************************************/
long long nvcmvalue_int32_to_int64(int int32Value)
{
    if (!DCGM_INT32_IS_BLANK(int32Value))
        return (long long)int32Value;

    switch (int32Value)
    {
        case DCGM_INT32_NOT_FOUND:
            return DCGM_INT64_NOT_FOUND;

        case DCGM_INT32_NOT_SUPPORTED:
            return DCGM_INT64_NOT_SUPPORTED;

        case DCGM_INT32_NOT_PERMISSIONED:
            return DCGM_INT64_NOT_PERMISSIONED;

        case DCGM_INT32_BLANK:
        /* Intentional fall-through to handle new values added later */
        default:
            return DCGM_INT64_BLANK;
    }

    /* Shouldn't get here */
    return DCGM_INT64_BLANK;
}

/*****************************************************************************/
double nvcmvalue_int64_to_double(long long int64Value)
{
    if (!DCGM_INT64_IS_BLANK(int64Value))
        return (double)int64Value;

    switch (int64Value)
    {
        case DCGM_INT64_NOT_FOUND:
            return DCGM_FP64_NOT_FOUND;

        case DCGM_INT64_NOT_SUPPORTED:
            return DCGM_FP64_NOT_SUPPORTED;

        case DCGM_INT64_NOT_PERMISSIONED:
            return DCGM_FP64_NOT_PERMISSIONED;

        case DCGM_INT64_BLANK:
        /* Intentional fall-through to handle new values added later */
        default:
            return DCGM_FP64_BLANK;
    }
}

/*****************************************************************************/
long long nvcmvalue_double_to_int64(double doubleValue)
{
    if (!DCGM_FP64_IS_BLANK(doubleValue))
        return (long long)doubleValue;

    if (doubleValue == DCGM_FP64_NOT_FOUND)
        return DCGM_INT64_NOT_FOUND;
    else if (doubleValue == DCGM_FP64_NOT_SUPPORTED)
        return DCGM_INT64_NOT_SUPPORTED;
    else if (doubleValue == DCGM_FP64_NOT_PERMISSIONED)
        return DCGM_INT64_NOT_PERMISSIONED;
    else
        return DCGM_INT64_BLANK;
}

/*****************************************************************************/
int nvcmvalue_double_to_int32(double doubleValue)
{
    if (!DCGM_FP64_IS_BLANK(doubleValue))
        return (int)doubleValue;

    if (doubleValue == DCGM_FP64_NOT_FOUND)
        return DCGM_INT32_NOT_FOUND;
    else if (doubleValue == DCGM_FP64_NOT_SUPPORTED)
        return DCGM_INT32_NOT_SUPPORTED;
    else if (doubleValue == DCGM_FP64_NOT_PERMISSIONED)
        return DCGM_INT32_NOT_PERMISSIONED;
    else
        return DCGM_INT32_BLANK;
}

/*****************************************************************************/
double nvcmvalue_int32_to_double(int int32Value)
{
    if (!DCGM_INT32_IS_BLANK(int32Value))
        return (double)int32Value;

    switch (int32Value)
    {
        case DCGM_INT32_NOT_FOUND:
            return DCGM_FP64_NOT_FOUND;

        case DCGM_INT32_NOT_SUPPORTED:
            return DCGM_FP64_NOT_SUPPORTED;

        case DCGM_INT32_NOT_PERMISSIONED:
            return DCGM_FP64_NOT_PERMISSIONED;

        case DCGM_INT32_BLANK:
        /* Intentional fall-through to handle new values added later */
        default:
            return DCGM_FP64_BLANK;
    }
}

/*****************************************************************************/
