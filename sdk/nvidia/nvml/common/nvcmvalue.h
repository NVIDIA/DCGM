#ifndef NVCMVALUE_H
#define NVCMVALUE_H

/* These are also defined in nvcm_structs.h for use in external applications */
#ifndef DCGM_BLANK_VALUES
#define DCGM_BLANK_VALUES

/* Base value for integer blank. can be used as an unspecified blank */
#define DCGM_INT32_BLANK 0x7ffffff0
#define DCGM_INT64_BLANK 0x7ffffffffffffff0ll

/* Base value for double blank. 2 ** 47. FP 64 has 52 bits of mantissa,
 * so 47 bits can still increment by 1 and represent each value from 0-15 */
#define DCGM_FP64_BLANK 140737488355328.0

#define DCGM_STR_BLANK "<<<NULL>>>"

/* Represents an error where data was not found */
#define DCGM_INT32_NOT_FOUND (DCGM_INT32_BLANK + 1)
#define DCGM_INT64_NOT_FOUND (DCGM_INT64_BLANK + 1)
#define DCGM_FP64_NOT_FOUND (DCGM_FP64_BLANK + 1.0)
#define DCGM_STR_NOT_FOUND "<<<NOT_FOUND>>>"

/* Represents an error where fetching the value is not supported */
#define DCGM_INT32_NOT_SUPPORTED (DCGM_INT32_BLANK + 2)
#define DCGM_INT64_NOT_SUPPORTED (DCGM_INT64_BLANK + 2)
#define DCGM_FP64_NOT_SUPPORTED (DCGM_FP64_BLANK + 2.0)
#define DCGM_STR_NOT_SUPPORTED "<<<NOT_SUPPORTED>>>"

/* Represents and error where fetching the value is not allowed with our current credentials */
#define DCGM_INT32_NOT_PERMISSIONED (DCGM_INT32_BLANK + 3)
#define DCGM_INT64_NOT_PERMISSIONED (DCGM_INT64_BLANK + 3)
#define DCGM_FP64_NOT_PERMISSIONED (DCGM_FP64_BLANK + 3.0)
#define DCGM_STR_NOT_PERMISSIONED "<<<NOT_PERM>>>"


/* Macro to check if a value is blank or not */
#define DCGM_INT32_IS_BLANK(val) (((val) >= DCGM_INT32_BLANK) ? 1 : 0)
#define DCGM_INT64_IS_BLANK(val) (((val) >= DCGM_INT64_BLANK) ? 1 : 0)
#define DCGM_FP64_IS_BLANK(val) (((val) >= DCGM_FP64_BLANK ? 1 : 0))
/* Works on (char *). Looks for <<< at first position and >>> inside string */
#define DCGM_STR_IS_BLANK(val) (val == strstr(val, "<<<") && strstr(val, ">>>"))

#endif //DCGM_BLANK_VALUES


#ifdef __cplusplus
extern "C"
{
#endif


    /*****************************************************************************/
    /*
 * Value to value conversion functions. These take special blank values into
 * account
 *
 */
    int nvcmvalue_int64_to_int32(long long int64Value);
    long long nvcmvalue_int32_to_int64(int int32Value);
    double nvcmvalue_int64_to_double(long long int64Value);
    long long nvcmvalue_double_to_int64(double doubleValue);
    int nvcmvalue_double_to_int32(double doubleValue);
    double nvcmvalue_int32_to_double(int int32Value);

    /*****************************************************************************/

#ifdef __cplusplus
}
#endif

#endif //NVCMVALUE_H
