#ifndef KEYEDVECTOR_H
#define KEYEDVECTOR_H


/* 
Efficient key'd vector container that minimizes allocations and frees

NOTE: This class is NOT thread safe. It is up to the caller to be thread safe 
*/

#ifdef __cplusplus
extern "C"
{
#endif

/* Use macros for alloc/free to allow redefining for other projects. 
   Redefine all malloc/frees if you redefine one */
#ifndef kv_malloc
#define kv_malloc malloc
#define kv_free free
#define kv_realloc realloc
#endif //kv_malloc

/* Status codes */
#define KV_ST_OK 0        /* Success */
#define KV_ST_BADPARAM -1 /* A bad parameter was passed to a function */
#define KV_ST_MEMORY -2   /* Out of memory */
#define KV_ST_DUPLICATE \
    -3                    /* An attempt to insert a duplicate element
                                 resulted in an error due to the merge callback
                                 saying this is not allowed */
#define KV_ST_NOTFOUND -4 /* The element in question was not found */
#define KV_ST_CORRUPT -5  /* Unrecoverable corruption has been detected */
#define KV_ST_INSERTED \
    -6 /* An internal-only return code that signifies
                                 that the record being inserted was already 
                                 inserted into the data structure and doesn't
                                 need to be inserted by the caller. */

/* Cursor to use for all operations into this keyed vector. A null cursor
   has -1 in both blockIndex and subIndex */
#define KV_CURSOR_NULL -1   /* Cursor is NULL */
#define KV_CURSOR_BEFORE -2 /* Cursor points before first element of data structure */
#define KV_CURSOR_AFTER -3  /* Cursor points after last element of data structure */


    /* Cursor into a KeyedVector enumeration. Note that these cursors are only
 * valid as long as the KeyedVector is not modified via Insert or Delete
 */
    typedef struct kv_cursor_t
    {
        int blockIndex; /* Index of this element's block into m_blocks */
        int subIndex;   /* Index of this element within its block */
    } kv_cursor_t, *kv_cursor_p;

    /*****************************************************************************/
    /*
Callback function for comparing two elements. This should return:
-1 if elem1 comes before elem2
 0 if elem1 and elem2 are the same
 1 if elem1 comes after elem2
*/
    typedef int (*kv_compare_f)(void *elem1, void *elem2);

    /*****************************************************************************/
    /* Callback function for merging two elements. This will be called if elements
   with duplicate keys are inserted. 
   This should return:
   KV_ST_OK if OK, meaning you have applied "inserting"s data to "current"
   KV_ST_DUPLICATE if duplicates are not allowed

   If you want the information from inserting to be applied to current, it is
   up to this callback to do so. Do not change key information
*/
    typedef int (*kv_merge_f)(void *current, void *inserting, void *user);

    /*****************************************************************************/
    /* Callback for freeing the contents of an element. This is an optional
   parameter to keyedvector_alloc and is not required if your elements
   don't alloc other memory.

   Returns: None
*/
    typedef void (*kv_free_f)(void *elem, void *user);

/*****************************************************************************/
/* Operations for searching */
#define KV_LGE_EQUAL 0      /* return nearest == key (or none) */
#define KV_LGE_LESSEQUAL 1  /* return nearest <= key */
#define KV_LGE_GREATEQUAL 2 /* return nearest >= key */
#define KV_LGE_LESS 3       /* return nearest < key */
#define KV_LGE_GREATER 4    /* return nearest > key */

    /*****************************************************************************/
    /* A keyed vector object handle returned from keyedvector_alloc() */
    typedef struct keyedvector_t
    {
        int subBlockSize; /* Size of a subblock in bytes */

        int Nblocks;     /* Current used space in m_blocks */
        int maxBlocks;   /* Current maximum capacity in m_Mblocks */
        void **blocks;   /* Array of pointers to sub blocks */
        int *blockNelem; /* For each corresponding index to m_blocks, how many
                           elements are in the block. Note that this structure
                           is made to handle inserting elements or even
                           blocks of elements in the middle */

        int elemSize; /* Size of an element in bytes */

        void *user; /* User-supplied data pointer */
        int Nelem;  /* Cached number of elements that are currently
                           stored in the keyed vector. This should match
                           taking the sum of all of the blockNelem entries */

        kv_compare_f compareCB;
        kv_merge_f mergeCB;
        kv_free_f freeCB; /* Optional element free callback. Can be null */
    } keyedvector_t, *keyedvector_p;

    /*************************************************************************/
    /*
Allocate an initialize this structure

elemSize     IN: The size of a raw element structure including key and
                    value
subBlockSize IN: Size of a subblock in bytes. 0=use default. Making this
                    larger incurs more overhead for smaller KeyedVectors
                    but results in less memory allocations occuring
compareCB    IN: Comparison callback
mergeCB      IN: Merge callback
freeCB       IN: Optional element free callback.
user         IN: User supplied context pointer for passing to relevant
                 functions like mergeCB
errorSt     OUT: KV_ST_? #define of the error that occured if NULL Is returned

Returns: Pointer to allocated keyedvector object
*/
    keyedvector_p keyedvector_alloc(int elemSize,
                                    int subBlockSize,
                                    kv_compare_f compareCB,
                                    kv_merge_f mergeCB,
                                    kv_free_f freeCB,
                                    void *user,
                                    int *errorSt);

    /*************************************************************************/
    /*
Locate an element by searching for an element by matching key fields

cursor will contain the position of the element so that keyedvector_next() and
       keyedvector_prev() can be called on it.
findOp is the relative operator to use to find the element. See KV_LGE_? #defines
       KV_LGE_EQUAL will only match exactly. Other operators will return the
       nearest match in the direction the operator specifies

Returns 0 if not found.
        Pointer to element if found. Do not change key field information
            of the returned pointer or you will corrupt the data structure
*/
    void *keyedvector_find_by_key(keyedvector_p kv, void *key, int findOp, kv_cursor_p cursor);

    /*************************************************************************/
    void *keyedvector_find_by_index(keyedvector_p kv, int index, kv_cursor_p cursor);
    /*
Locate an element by its index into the keyed vector

Note that this api is here for completeness and simplifying automated tests.
It is preferred to use keyedvector_find_by_key() and then keyedvector_next()
and keyedvecotr_prev() to walk elements

cursor will contain the position of the element so that keyedvector_next() and
       keyedvector_prev() can be called on it.

Returns 0 if index out of range of elements
        Pointer to element if found. Do not change key field information
            of the returned pointer or you will corrupt the data structure
 */

    /*************************************************************************/
    /*
Returns the first/last element in this data structure.

NULL if the structure is empty.
ptr to the element if there are any. Use cursor to advance via Next() and Prev()
*/
    void *keyedvector_first(keyedvector_p kv, kv_cursor_p cursor);
    void *keyedvector_last(keyedvector_p kv, kv_cursor_p cursor);

    /*************************************************************************/
    /*
Get the next/previous element.

NULL if there are no more elements or if cursor is invalid

*/
    void *keyedvector_next(keyedvector_p kv, kv_cursor_p cursor);
    void *keyedvector_prev(keyedvector_p kv, kv_cursor_p cursor);

    /*************************************************************************/
    /*
Insert an element, returning the element's position in cursor

Note that this operation invalidates other cursors

Returns: 0 if OK (including if a successful merge occurred via mergeCB)
         KV_ST_DUPLICATE if an element with the same key already existed and
                         mergeCB said to return an error
         <= 0 KV_ST_? #define on error
*/
    int keyedvector_insert(keyedvector_p kv, void *element, kv_cursor_p cursor);

    /*************************************************************************/
    /*
Remove an element by key

Returns: 0 if OK
         KV_ST_NOTFOUND If element was not found by key
       <=0 KV_ST_? #define on other error
*/
    int keyedvector_remove(keyedvector_p kv, void *key);

    /*************************************************************************/
    /*
Remove an element by cursor

Note that cursor is not updated to point at a valid element. If you were
iterating backward, then Prev() can be used. It is safest to use FindByKey
after a delete to maintain cursor consistency

Returns: 0 if OK
         KV_ST_NOTFOUND If element was not found by cursor (cursor was invalid)
        <0 KV_ST_? #define on other error
*/
    int keyedvector_remove_by_cursor(keyedvector_p kv, kv_cursor_p cursor);

    /*************************************************************************/
    int keyedvector_remove_range_by_cursor(keyedvector_p kv, kv_cursor_p startCursor, kv_cursor_p endCursor);
    /*
Remove a range of elements by a pair of cursors. All values between and
including the two cursors will be removed.

startCursor     IN: Starting position to delete from. NULL=beginning of dataset
endCursor       IN: Ending position to delete until. NULL=end of dataset

Returns: 0 if OK
        <0 KV_ST_? #define on error
*/

    /*************************************************************************/
    /*
Return the number of elements in the collection
*/
    int keyedvector_size(keyedvector_p kv);

    /*************************************************************************/
    /*
Return the number of elements in the collection the slow way, adding up
the number of elements in each of the collection's sub blocks. This function
is here to sanity check keyedvector_size().
*/
    int keyedvector_size_slow(keyedvector_p kv);

    /*************************************************************************/
    /*
Empty this collection, leaving it useable (unlike Destroy())
*/
    void keyedvector_empty(keyedvector_p kv);

    /*************************************************************************/
    /*
Destroy this collection. Called by destructor
*/
    void keyedvector_destroy(keyedvector_p kv);

    /*************************************************************************/
    /*
Calculate (roughly) the number of bytes in memory that this keyedvector takes up.
 */
    long long keyedvector_bytes_used(keyedvector_p kv);

    /*************************************************************************/

#ifdef __cplusplus
}
#endif

#endif //KEYEDVECTOR_H
