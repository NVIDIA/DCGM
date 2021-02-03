
#include "keyedvector.h"
#include "logging.h"
#include <stdlib.h> //malloc/free
#include <string.h> //memset

#define KV_DEFAULT_BLOCK_SIZE 1       /* Start with capacity for N subblocks */
#define KV_DEFAULT_SUBBLOCK_SIZE 1024 /* Size of a subblock in bytes */

/*****************************************************************************/
/* Stubs for local functions */
static int keyedvector_which_block(keyedvector_p kv, void *key);
static int keyedvector_grow_blocks(keyedvector_p kv, int newSize);
static int keyedvector_delete_block_range(keyedvector_p kv, int startBlockIdx, int endBlockIdx);
static void keyedvector_call_freeCB_range(keyedvector_p kv, kv_cursor_p startCursor, kv_cursor_p endCursor);

/*****************************************************************************/
keyedvector_p keyedvector_alloc(int elemSize,
                                int subBlockSize,
                                kv_compare_f compareCB,
                                kv_merge_f mergeCB,
                                kv_free_f freeCB,
                                void *user,
                                int *errorSt)
{
    int st;
    keyedvector_p kv = NULL;

    if (!compareCB || !mergeCB || !errorSt)
    {
        if (errorSt)
        {
            *errorSt = KV_ST_BADPARAM;
        }
        return NULL;
    }

    *errorSt = KV_ST_OK;

    kv = (keyedvector_p)kv_malloc(sizeof(*kv));
    if (!kv)
        return NULL;

    memset(kv, 0, sizeof(*kv));
    kv->subBlockSize = KV_DEFAULT_SUBBLOCK_SIZE;
    kv->compareCB    = compareCB;
    kv->mergeCB      = mergeCB;
    kv->freeCB       = freeCB;
    kv->user         = user;

    if (subBlockSize < 0)
    {
        keyedvector_destroy(kv);
        kv       = 0;
        *errorSt = KV_ST_BADPARAM;
        return NULL;
    }
    else if (subBlockSize != 0)
        kv->subBlockSize = subBlockSize;

    if (elemSize > kv->subBlockSize || elemSize < 0)
    {
        keyedvector_destroy(kv);
        kv       = 0;
        *errorSt = KV_ST_BADPARAM;
        return NULL;
    }
    else if (elemSize != 0)
    {
        kv->elemSize = elemSize;
    }

    /* Allocate initial structures */
    kv->Nblocks   = 0;
    kv->maxBlocks = 0;

    st = keyedvector_grow_blocks(kv, KV_DEFAULT_BLOCK_SIZE);
    if (st != KV_ST_OK)
    {
        keyedvector_destroy(kv);
        kv       = 0;
        *errorSt = st;
        return NULL;
    }

    /* Allocate the first block as empty. It is the only block that's allowed
     * to be empty
     */
    kv->blocks[0] = kv_malloc(kv->subBlockSize);
    if (!kv->blocks[0])
    {
        keyedvector_destroy(kv);
        kv       = 0;
        *errorSt = KV_ST_MEMORY;
        return NULL;
    }
    kv->Nblocks = 1;

    return kv;
}

/*****************************************************************************/
void keyedvector_destroy(keyedvector_p kv)
{
    int i;
    void *elem, *elem1;
    kv_cursor_t startCursor, endCursor;

    if (!kv)
        return;

    /* Call the freeCB on every element */
    if (kv->freeCB)
    {
        elem  = keyedvector_first(kv, &startCursor);
        elem1 = keyedvector_last(kv, &endCursor);
        if (elem && elem1)
        {
            keyedvector_call_freeCB_range(kv, &startCursor, &endCursor);
        }
    }

    if (kv->blocks)
    {
        /* Free sub blocks */
        for (i = 0; i < kv->Nblocks; i++)
        {
            if (kv->blocks[i])
            {
                kv_free(kv->blocks[i]);
                kv->blocks[i] = 0;
            }
        }

        kv->Nblocks = 0;

        /* Now free the parent block */
        kv_free(kv->blocks);
        kv->blocks = 0;
    }

    if (kv->blockNelem)
    {
        free(kv->blockNelem);
        kv->blockNelem = 0;
    }

    kv->Nelem = 0;

    kv_free(kv);
}

/*****************************************************************************/
/*
Grow kv->blocks by newSize

Returns:  0 KV_ST_OK if grown OK
         <0 KV_ST_? #define on error
*/
static int keyedvector_grow_blocks(keyedvector_p kv, int newSize)
{
    int oldSize;
    void *tmpBlock;

    if (!kv)
        return KV_ST_BADPARAM;

    oldSize = kv->maxBlocks;

    if (newSize <= kv->maxBlocks)
        return KV_ST_OK; /* Nothing to do */

    kv->maxBlocks = newSize;

    tmpBlock = kv_realloc(kv->blocks, sizeof(void *) * kv->maxBlocks);
    if (!tmpBlock)
    {
      return KV_ST_MEMORY;
    }
    kv->blocks = (void**)tmpBlock;

    tmpBlock = kv_realloc(kv->blockNelem, sizeof(int) * kv->maxBlocks);
    if (!tmpBlock)
    {
      return KV_ST_MEMORY;
    }
    kv->blockNelem = (int*)tmpBlock;

    /* Zero out new elements */
    memset(&kv->blocks[oldSize], 0, sizeof(void *) * (newSize - oldSize));
    memset(&kv->blockNelem[oldSize], 0, sizeof(int) * (newSize - oldSize));
    return KV_ST_OK;
}

/*****************************************************************************/
void *keyedvector_find_by_index(keyedvector_p kv, int index, kv_cursor_p cursor)
{
    int soFarIndex;
    int blockIndex;
    char *block;

    if (!kv || index < 0 || !cursor)
        return NULL;

    cursor->blockIndex = KV_CURSOR_NULL;
    cursor->subIndex   = KV_CURSOR_NULL;

    if (!kv->Nblocks)
        return NULL;

    soFarIndex = 0;
    for (blockIndex = 0; blockIndex < kv->Nblocks; blockIndex++)
    {
        if (soFarIndex + kv->blockNelem[blockIndex] <= index)
        {
            soFarIndex += kv->blockNelem[blockIndex];
            continue;
        }

        /* In this block. Get a pointer to the element in question */
        block              = (char *)kv->blocks[blockIndex];
        cursor->blockIndex = blockIndex;
        cursor->subIndex   = index - soFarIndex;
        return &block[cursor->subIndex * kv->elemSize];
    }

    return NULL; /* Got past end of last block */
}

/*****************************************************************************/
void *keyedvector_find_by_key(keyedvector_p kv, void *key, int findOp, kv_cursor_p cursor)
{
    int blockIndex;
    int compareSt;
    int searchFromSubIndex;
    int blockMin, blockMax, blockMid;
    char *block;

    if (!kv || !key || !kv->Nblocks || !kv->blocks || !kv->blocks[0])
    {
        return NULL;
    }

    if (!cursor)
    {
        PRINT_ERROR("", "cursor was null");
        return NULL;
    }

    cursor->blockIndex = -1;
    cursor->subIndex   = -1;

    blockIndex = keyedvector_which_block(kv, key);
    if (blockIndex < 0)
    {
        PRINT_ERROR("%d", "Error %d from WhichBlock()\n", blockIndex);
        return NULL;
    }

    block = (char *)kv->blocks[blockIndex];

    searchFromSubIndex = 0;

    blockMin = 0;
    blockMax = kv->blockNelem[blockIndex] - 1;

    while (blockMax >= blockMin)
    {
        blockMid = (blockMin + blockMax) / 2;

        compareSt = kv->compareCB(key, &block[kv->elemSize * blockMid]);
        if (!compareSt)
        {
            /* Exact match. Prepare cursor for return or further search */
            cursor->blockIndex = blockIndex;
            cursor->subIndex   = blockMid;

            switch (findOp)
            {
                case KV_LGE_EQUAL:
                case KV_LGE_LESSEQUAL:
                case KV_LGE_GREATEQUAL:
                    return &block[kv->elemSize * blockMid];

                case KV_LGE_LESS:
                    return keyedvector_prev(kv, cursor);

                case KV_LGE_GREATER:
                    return keyedvector_next(kv, cursor);

                default:
                    PRINT_ERROR("%d", "Unknown findOp: %d\n", findOp);
                    return NULL;
            }
        }
        else if (compareSt < 0)
        {
            blockMax           = blockMid - 1; /* Before */
            searchFromSubIndex = blockMin;
        }
        else
        {
            blockMin           = blockMid + 1; /* After */
            searchFromSubIndex = blockMin;
        }
    }

    if (searchFromSubIndex < 0)
        searchFromSubIndex = 0;

    /* No exact match so exact match fails */
    if (findOp == KV_LGE_EQUAL)
    {
        return NULL;
    }

    cursor->blockIndex = blockIndex;
    cursor->subIndex   = searchFromSubIndex;

    switch (findOp)
    {
        case KV_LGE_LESSEQUAL:
        case KV_LGE_LESS:
            return keyedvector_prev(kv, cursor);

        case KV_LGE_GREATEQUAL:
        case KV_LGE_GREATER:
            /* Cursor is already pointing at matching element. Make sure index is inside block (not past end) */
            if (cursor->subIndex < kv->blockNelem[cursor->blockIndex])
                return &block[kv->elemSize * cursor->subIndex];
            else
                return keyedvector_next(kv, cursor);
            break;

        default:
            PRINT_ERROR("%d", "Unknown findOp: %d\n", findOp);
            return NULL;
    }

    return NULL; /* Not reachable but here for compiler warning */
}

/*****************************************************************************/
void *keyedvector_next(keyedvector_p kv, kv_cursor_p cursor)
{
    int blockIndex, subIndex;
    char *block;

    if (!kv || !cursor || cursor->blockIndex < 0 || cursor->subIndex < 0 || cursor->blockIndex >= kv->Nblocks)
        return NULL;

    blockIndex = cursor->blockIndex;
    subIndex   = cursor->subIndex + 1;

    if (subIndex >= kv->blockNelem[blockIndex])
    {
        blockIndex++;
        subIndex = 0;
    }

    if (blockIndex >= kv->Nblocks || !kv->blocks[blockIndex] || kv->blockNelem[blockIndex] < 1)
    {
        /* Past end of structure */
        cursor->blockIndex = KV_CURSOR_AFTER;
        cursor->subIndex   = KV_CURSOR_AFTER;
        return NULL;
    }

    cursor->blockIndex = blockIndex;
    cursor->subIndex   = subIndex;
    block              = (char *)kv->blocks[blockIndex];
    return &block[kv->elemSize * subIndex];
}

/*****************************************************************************/
/*
 * Find the place in the KV where element should be inserted based on comparing
 * elem against other elements in the KV. If an element with the same key already
 * exists, this function will attempt to call the KV's mergeCB to merge the records
 * together. 
 * 
 * Note that this function may insert the record for us if it's advantageous and will
 * return KV_ST_INSERTED in that case.
 * 
 * kv                 IN: Keyed vector to look in
 * element            IN: Element that is being inserted
 * cursor            OUT: Contains the location of where the element should be inserted.
 *                        Note that this is the location of the existing element in the case that
 *                        KV_ST_MERGED is returned.
 * 
 * Returns: KV_ST_OK on success
 *          KV_ST_INSERTED if the record being inserted was inserted by this function
 *          KV_ST_DUPLICATE if the record being inserted's key already exists in the KV and the
 *                       KV's mergeCB does not allow merging records.
 *          Other KV_ST_? #define on failure
 */
static int keyedvector_insert_helper(keyedvector_p kv, void *element, kv_cursor_p cursor)
{
    int blockMid, blockMin, blockMax; /* Binary search variables */
    int blockIndex, insertAt = 0;
    int compareSt;
    char *block = NULL;

    blockIndex = keyedvector_which_block(kv, element);
    if (blockIndex < 0)
        return blockIndex; /* Error */

    /* Has the block in question been allocated yet? */
    if (!kv->blocks[blockIndex])
    {
        /* Unexpected null block */
        return KV_ST_CORRUPT;
    }

    /* Find where in the block in question we should insert */
    block = (char *)kv->blocks[blockIndex];

    /* Does the block have any elements (check only really matters for first block) */
    if (!kv->blockNelem[blockIndex])
    {
        cursor->blockIndex = blockIndex;
        cursor->subIndex   = 0;
        memmove(block, element, kv->elemSize);
        kv->blockNelem[blockIndex]++;
        kv->Nelem++;
        return KV_ST_INSERTED;
    }

    blockMin = 0;
    blockMax = kv->blockNelem[blockIndex] - 1;

    while (blockMax >= blockMin)
    {
        blockMid = (blockMin + blockMax) / 2;

        compareSt = kv->compareCB(element, &block[kv->elemSize * blockMid]);
        if (!compareSt)
        {
            int mergeSt;
            /* Exact match */
            cursor->blockIndex = blockIndex;
            cursor->subIndex   = blockMid;
            mergeSt            = kv->mergeCB(&block[kv->elemSize * blockMid], element, kv->user);
            if (mergeSt < 0)
                return mergeSt;
            else
                return KV_ST_INSERTED; /* Successfully merged */
        }
        else if (compareSt < 0)
        {
            blockMax = blockMid - 1; /* Before */
            insertAt = blockMin;
        }
        else
        {
            blockMin = blockMid + 1; /* After */
            insertAt = blockMin;
        }
    }

    if (insertAt < 0)
        insertAt = 0;

    cursor->blockIndex = blockIndex;
    cursor->subIndex   = insertAt;
    return KV_ST_OK;
}

/*****************************************************************************/
int keyedvector_insert(keyedvector_p kv, void *element, kv_cursor_p cursor)
{
    int blockIndex;
    int st;
    int elementsPerBlock, spaceLeft;
    char *block = NULL, *block1 = NULL;
    int insertAt = 0;
    void *lastElement;

    if (!kv || !element || !cursor)
        return KV_ST_BADPARAM;

    /* See if our element is past the entire keyedvector, which is a common usecase
       when records are being appended to the end, like in timeseries. Handle this
       as a fast path */
    lastElement = keyedvector_last(kv, cursor);
    if (lastElement && 0 < kv->compareCB(element, lastElement))
    {
        /* The element being inserted is past the end. Just increment the subIndex, 
           which will be handled below */
        cursor->subIndex++;
    }
    else
    {
        /* Search for where element should be inserted */
        st = keyedvector_insert_helper(kv, element, cursor);
        if (st == KV_ST_INSERTED)
            return KV_ST_OK; /* Was already inserted */
        else if (st != KV_ST_OK)
            return st;
    }

    /* Cursor contains where we should insert the element */
    blockIndex = cursor->blockIndex;
    insertAt   = cursor->subIndex;

    /* Exact matches were taken care of above. blockIndex, insertAt now points at where to
       insert.
    */
    block            = (char *)kv->blocks[blockIndex];
    elementsPerBlock = kv->subBlockSize / kv->elemSize;
    spaceLeft        = elementsPerBlock - kv->blockNelem[blockIndex];

    if (spaceLeft > 0)
    {
        /* Have room for the element. Shift the elements of the block to
           make room for the new one */
        if (insertAt < kv->blockNelem[blockIndex])
        {
            memmove(&block[(insertAt + 1) * kv->elemSize],
                    &block[(insertAt)*kv->elemSize],
                    (kv->blockNelem[blockIndex] - insertAt) * kv->elemSize);
        }

        memmove(&block[insertAt * kv->elemSize], element, kv->elemSize);
        kv->blockNelem[blockIndex]++;
        kv->Nelem++;

        cursor->blockIndex = blockIndex;
        cursor->subIndex   = insertAt;
        return KV_ST_OK;
    }

    /* No space for the element in question. We will need another block.
       Room for one? */
    if (kv->Nblocks >= kv->maxBlocks)
    {
        st = keyedvector_grow_blocks(kv, kv->maxBlocks * 2);
        if (st)
            return st;
    }

    /* Move blocks after this block back since we're not the last block */
    if (blockIndex < kv->Nblocks - 1)
    {
        memmove(
            &kv->blocks[blockIndex + 1], &kv->blocks[blockIndex], sizeof(kv->blocks[0]) * (kv->Nblocks - blockIndex));
        memmove(&kv->blockNelem[blockIndex + 1],
                &kv->blockNelem[blockIndex],
                sizeof(kv->blockNelem[0]) * (kv->Nblocks - blockIndex));
        /* kv->blocks[blockIndex] and kv->blocks[blockIndex+1] now contain the same pointer
         * overwrite blockIndex+1 with the new block */
    }

    kv->Nblocks++;
    kv->blockNelem[blockIndex + 1] = 0;
    kv->blocks[blockIndex + 1]     = kv_malloc(kv->subBlockSize);
    if (!kv->blocks[blockIndex + 1])
        return KV_ST_MEMORY;

    /* blockIndex+1 has an empty block. If the record being inserted is supposed
     * to be the very last record, just put it by itself in the new block,
     * otherwise, split around the new record
     */
    if (insertAt >= kv->blockNelem[blockIndex]) /* Logic intentional for clarity */
    {
        cursor->blockIndex = blockIndex + 1;
        cursor->subIndex   = 0;
        memmove(kv->blocks[cursor->blockIndex], element, kv->elemSize);
        kv->blockNelem[cursor->blockIndex]++;
        kv->Nelem++;
        return KV_ST_OK;
    }

    /* Split the new block around where the new element will be inserted.
     * the new element will be at its own index of block[] */
    block                          = (char *)kv->blocks[blockIndex];
    block1                         = (char *)kv->blocks[blockIndex + 1];
    kv->blockNelem[blockIndex + 1] = kv->blockNelem[blockIndex] - insertAt;
    memmove(&block1[0], &block[insertAt * kv->elemSize], kv->elemSize * kv->blockNelem[blockIndex + 1]);
    kv->blockNelem[blockIndex] = insertAt;

    /* Append new element to block[] */
    memmove(&block[insertAt * kv->elemSize], element, kv->elemSize);
    kv->blockNelem[blockIndex]++;
    kv->Nelem++;
    cursor->blockIndex = blockIndex;
    cursor->subIndex   = insertAt;
    return KV_ST_OK;
}

/*****************************************************************************/
void *keyedvector_prev(keyedvector_p kv, kv_cursor_p cursor)
{
    int blockIndex, subIndex;
    char *block;

    if (!kv || !cursor || cursor->blockIndex < 0 || cursor->subIndex < 0)
        return NULL;

    blockIndex = cursor->blockIndex;
    subIndex   = cursor->subIndex - 1;

    /* Cursor could be past elements in block if an element was deleted */
    if (subIndex >= kv->blockNelem[blockIndex])
    {
        subIndex = kv->blockNelem[blockIndex] - 1;
        /* Note that the check below handles if this underflows */
    }

    if (subIndex < 0)
    {
        if (!blockIndex)
        {
            cursor->blockIndex = KV_CURSOR_BEFORE;
            cursor->subIndex   = KV_CURSOR_BEFORE;
            return NULL;
        }

        blockIndex--;
        subIndex = kv->blockNelem[blockIndex] - 1;

        if (subIndex < 0)
        {
            PRINT_ERROR("%d", "Unexpected empty block at blockIndex %d\n", blockIndex);
            cursor->blockIndex = KV_CURSOR_BEFORE;
            cursor->subIndex   = KV_CURSOR_BEFORE;
            return NULL;
        }
    }

    cursor->blockIndex = blockIndex;
    cursor->subIndex   = subIndex;
    block              = (char *)kv->blocks[blockIndex];
    return &block[kv->elemSize * subIndex];
}

/*****************************************************************************/
void *keyedvector_first(keyedvector_p kv, kv_cursor_p cursor)
{
    if (!kv || !kv->blocks || !kv->Nblocks || !kv->blocks[0] || kv->blockNelem[0] < 1)
    {
        cursor->blockIndex = KV_CURSOR_NULL;
        cursor->subIndex   = KV_CURSOR_NULL;
        return NULL;
    }

    cursor->blockIndex = 0;
    cursor->subIndex   = 0;
    return kv->blocks[0];
}

/*****************************************************************************/
void *keyedvector_last(keyedvector_p kv, kv_cursor_p cursor)
{
    int lastBlock;
    char *block;

    if (!kv || !kv->blocks || !kv->Nblocks)
    {
        cursor->blockIndex = KV_CURSOR_NULL;
        cursor->subIndex   = KV_CURSOR_NULL;
        return NULL;
    }

    lastBlock = kv->Nblocks - 1;
    if (!kv->blocks[lastBlock] || kv->blockNelem[lastBlock] < 1)
    {
        cursor->blockIndex = KV_CURSOR_NULL;
        cursor->subIndex   = KV_CURSOR_NULL;
        return NULL;
    }

    cursor->blockIndex = lastBlock;
    cursor->subIndex   = kv->blockNelem[lastBlock] - 1;
    block              = (char *)kv->blocks[lastBlock];
    return &block[cursor->subIndex * kv->elemSize];
}

/*****************************************************************************/
/*
Return which block a key belongs in.

Returns: >= 0 block key belongs in
         <0 KV_ST_? #define on error
*/
static int keyedvector_which_block(keyedvector_p kv, void *key)
{
    int blkMin, blkMax, blkMid, compareSt;
    char *midSubBlock; /* Use a char ptr so byte math can be done */

    if (!kv || !key)
        return KV_ST_BADPARAM;

    if (kv->Nblocks <= 1)
        return 0; /* Goes in first block if there is only one */

    blkMin = 0;
    blkMax = kv->Nblocks - 1;

    while (blkMax >= blkMin)
    {
        blkMid = (blkMax + blkMin) / 2;

        midSubBlock = (char *)kv->blocks[blkMid];

        if (!kv->blockNelem[blkMid] || !midSubBlock)
        {
            PRINT_ERROR("%d %p", "Unexpected empty block %d (%p)\n", blkMid, midSubBlock);
            /* Can't recover in middle of binary search */
            return KV_ST_CORRUPT;
        }

        /* Compare with first element */
        compareSt = kv->compareCB(key, midSubBlock);
        if (!compareSt)
            return blkMid; /* Matches first element. This is the block for sure */
        else if (compareSt < 0)
        {
            blkMax = blkMid - 1; /* Before first element. Search blocks before this one */
        }
        else
            blkMin = blkMid + 1; /* After first element. Search blocks after this one */
    }

    if (blkMax <= 0)
        return 0;

    return blkMax; /* Insert into the leftmost block */
}

/*****************************************************************************/
void keyedvector_empty(keyedvector_p kv)
{
    int i;

    if (!kv || !kv->blocks || !kv->blockNelem)
        return;

    /* Free every block but first one */
    for (i = 1; i < kv->maxBlocks; i++)
    {
        if (kv->blocks[i])
        {
            kv_free(kv->blocks[i]);
            kv->blocks[i] = NULL;
        }
        kv->blockNelem[i] = 0;
    }

    kv->blockNelem[0] = 0;
    kv->Nelem         = 0;
}

/*****************************************************************************/
int keyedvector_remove_by_cursor(keyedvector_p kv, kv_cursor_p cursor)
{
    char *block;
    int blockIndex, subIndex;

    if (!kv || !cursor)
        return KV_ST_BADPARAM;

    /* Cache locally for better readability */
    blockIndex = cursor->blockIndex;
    subIndex   = cursor->subIndex;

    if (blockIndex < 0 || blockIndex >= kv->Nblocks)
        return KV_ST_NOTFOUND;
    if (subIndex < 0 || subIndex >= kv->blockNelem[blockIndex])
        return KV_ST_NOTFOUND;

    /* Cursor points at a valid element at this point. */

    block = (char *)kv->blocks[blockIndex];
    if (!block)
    {
        PRINT_ERROR("%d", "Unexpected null block %d\n", blockIndex);
        return KV_ST_CORRUPT;
    }

    /* Call freeCB on the element if freeCB exists */
    if (kv->freeCB)
    {
        kv->freeCB(&block[subIndex * kv->elemSize], kv->user);
    }

    /* Last element of subblock? */
    if (subIndex == kv->blockNelem[blockIndex] - 1)
    {
        kv->blockNelem[blockIndex]--;
        kv->Nelem--;

        /* Last element of block and not the very first block? */
        if ((blockIndex > 0 || kv->Nblocks > 1) && kv->blockNelem[blockIndex] < 1)
            return keyedvector_delete_block_range(kv, blockIndex, blockIndex);
        else
            return KV_ST_OK;
    }

    /* Move the elements after the one being deleted forward */
    kv->blockNelem[blockIndex]--;
    memmove(&block[subIndex * kv->elemSize],
            &block[(subIndex + 1) * kv->elemSize],
            kv->elemSize * kv->blockNelem[blockIndex]);
    kv->Nelem--;
    return KV_ST_OK;
}

/*****************************************************************************/
/*
Delete and free a range of blocks

To delete a single block, set startBlockIdx and endBlockIdx to the same value

startBlockIdx  IN: Starting block index to delete (inclusive)
endBlockIdx    IN: Ending block index to delete (inclusive)
*/
static int keyedvector_delete_block_range(keyedvector_p kv, int startBlockIdx, int endBlockIdx)
{
    int blockIndex;
    int Ndeleted;

    if (!kv)
        return KV_ST_BADPARAM;

    if (startBlockIdx < 0 || startBlockIdx >= kv->Nblocks || startBlockIdx > endBlockIdx || endBlockIdx < 0
        || endBlockIdx >= kv->Nblocks)
    {
        PRINT_ERROR("%d %d %d",
                    "DeleteBlockRange() Bad startBlockIdx %d or endBlockIdx %d. kv->Nblocks %d\n",
                    startBlockIdx,
                    endBlockIdx,
                    kv->Nblocks);
        return KV_ST_BADPARAM;
    }

    /* Block index 0 doesn't get deleted if it's the last one. it can get zeroed though */
    if (startBlockIdx < 1 && endBlockIdx >= kv->Nblocks - 1)
    {
        kv->Nelem -= kv->blockNelem[0];
        kv->blockNelem[0] = 0;

        startBlockIdx = 1;
        if (endBlockIdx < 1)
            return KV_ST_OK; /* Nothing left to do */
    }

    Ndeleted = (endBlockIdx - startBlockIdx) + 1;

    /* First, free and zero the blocks being removed */
    for (blockIndex = startBlockIdx; blockIndex <= endBlockIdx; blockIndex++)
    {
        kv->Nelem -= kv->blockNelem[blockIndex];
        kv->blockNelem[blockIndex] = 0;
        if (!kv->blocks[blockIndex])
        {
            PRINT_ERROR("%d", "Unexpected null block at %d\n", blockIndex);
        }
        else
        {
            kv_free(kv->blocks[blockIndex]);
            kv->blocks[blockIndex] = NULL;
        }
    }

    if (endBlockIdx >= kv->Nblocks - 1)
    {
        kv->Nblocks -= Ndeleted;
        return KV_ST_OK;
    }

    /* If there are blocks after endBlockIdx, move them forward */
    memmove(&kv->blocks[startBlockIdx],
            &kv->blocks[endBlockIdx + 1],
            sizeof(kv->blocks[0]) * ((kv->Nblocks - endBlockIdx) - 1));
    memmove(&kv->blockNelem[startBlockIdx],
            &kv->blockNelem[endBlockIdx + 1],
            sizeof(kv->blockNelem[0]) * ((kv->Nblocks - endBlockIdx) - 1));


    kv->Nblocks -= Ndeleted;
    return KV_ST_OK;
}

/*****************************************************************************/
/*
Local helper to call freeCB on all of the values between two cursors. The
cursors are assumed to be pre-validated since this is a helper.
 */
static void keyedvector_call_freeCB_range(keyedvector_p kv, kv_cursor_p startCursor, kv_cursor_p endCursor)
{
    int blockIdx, subBlockIdx;
    int startSubIdx, endSubIdx;
    char *block;

    for (blockIdx = startCursor->blockIndex; blockIdx <= endCursor->blockIndex; blockIdx++)
    {
        /* Determine starting and ending sub indexes for this block */

        startSubIdx = 0;
        if (blockIdx == startCursor->blockIndex)
            startSubIdx = startCursor->subIndex;

        endSubIdx = kv->blockNelem[blockIdx] - 1;
        if (blockIdx == endCursor->blockIndex)
            endSubIdx = endCursor->subIndex;

        block = (char *)kv->blocks[blockIdx];

        /* Walk the range of sub blocks, calling freeCB on each element */
        for (subBlockIdx = startSubIdx; subBlockIdx <= endSubIdx; subBlockIdx++)
        {
            kv->freeCB(&block[subBlockIdx * kv->elemSize], kv->user);
        }
    }
}

/*****************************************************************************/
int keyedvector_remove_range_by_cursor(keyedvector_p kv, kv_cursor_p startCursor, kv_cursor_p endCursor)
{
    kv_cursor_t startCursorLocal;
    kv_cursor_t endCursorLocal;
    void *elem;
    int startBlock, endBlock; /* Range of entire blocks to delete */
    int NelemToDelete;
    char *block;
    int st;

    if (!kv)
        return KV_ST_BADPARAM;

    /* Validate start cursor */
    if (startCursor)
    {
        if (startCursor->blockIndex < 0 || startCursor->blockIndex >= kv->Nblocks || startCursor->subIndex < 0
            || startCursor->subIndex >= kv->blockNelem[startCursor->blockIndex])
            return KV_ST_BADPARAM;
    }
    else
    {
        /* Set to first element */
        startCursor = &startCursorLocal;
        elem        = keyedvector_first(kv, startCursor);
        if (!elem)
            return KV_ST_OK; /* No elements */
    }

    /* Validate end cursor */
    if (endCursor)
    {
        if (endCursor->blockIndex < 0 || endCursor->blockIndex >= kv->Nblocks || endCursor->subIndex < 0
            || endCursor->subIndex >= kv->blockNelem[endCursor->blockIndex])
            return KV_ST_BADPARAM;
    }
    else
    {
        /* Set to last element */
        endCursor = &endCursorLocal;
        elem      = keyedvector_last(kv, endCursor);
        if (!elem)
            return KV_ST_OK; /* No elements */
    }

    /* Call freeCB on all elements before we start shuffling memory around */
    if (kv->freeCB)
    {
        keyedvector_call_freeCB_range(kv, startCursor, endCursor);
    }

    /* Starting and ending element in same block? */
    if (startCursor->blockIndex == endCursor->blockIndex)
    {
        /* Delete element range in question */
        NelemToDelete = (endCursor->subIndex + 1) - startCursor->subIndex;

        /* Are we deleting the entire block?. If so, delete that one block */
        if (NelemToDelete == kv->blockNelem[endCursor->blockIndex])
            return keyedvector_delete_block_range(kv, startCursor->blockIndex, startCursor->blockIndex);

        if (endCursor->subIndex < kv->blockNelem[endCursor->blockIndex] - 1)
        {
            /* There are elements after endCursor in the same block, move them up */
            int NtoMove = kv->blockNelem[endCursor->blockIndex] - (endCursor->subIndex + 1);
            block       = (char *)kv->blocks[endCursor->blockIndex];
            memmove(&block[startCursor->subIndex * kv->elemSize],
                    &block[(endCursor->subIndex + 1) * kv->elemSize],
                    NtoMove * kv->elemSize);
        }
        kv->blockNelem[endCursor->blockIndex] -= NelemToDelete;
        kv->Nelem -= NelemToDelete;

        return KV_ST_OK;
    }
    /* Start and end are in different blocks */

    /* Set range of blocks to delete. Default to one block inside of range to delete */
    startBlock = startCursor->blockIndex + 1;
    endBlock   = endCursor->blockIndex - 1;

    if (startCursor->subIndex == 0)
        startBlock--; /* At beginning of block. Include this one in range of block delete */
    else
    {
        /* We don't have to copy sub-elements forward since we're deleting to the
         * end of the first block. Just lower the element count */
        NelemToDelete = kv->blockNelem[startCursor->blockIndex] - startCursor->subIndex;
        kv->blockNelem[startCursor->blockIndex] -= NelemToDelete;
        kv->Nelem -= NelemToDelete;
    }

    if (endCursor->subIndex >= kv->blockNelem[endCursor->blockIndex] - 1)
        endBlock++; /* At end of block. Include this one in range of block delete */
    else
    {
        int NtoMove;
        NelemToDelete = endCursor->subIndex + 1;
        /* There are elements after endCursor in the same block, move them up */
        NtoMove = kv->blockNelem[endCursor->blockIndex] - (endCursor->subIndex + 1);
        block   = (char *)kv->blocks[endCursor->blockIndex];
        memmove(block, &block[(endCursor->subIndex + 1) * kv->elemSize], NtoMove * kv->elemSize);

        kv->blockNelem[endCursor->blockIndex] -= NelemToDelete;
        kv->Nelem -= NelemToDelete;
    }

    /* Any whole blocks to delete? */
    if (endBlock >= startBlock)
    {
        st = keyedvector_delete_block_range(kv, startBlock, endBlock);
        if (st)
        {
            PRINT_ERROR(
                "%d %d %d", "Unexpected st %d from keyedvector_delete_block_range %d %d\n", st, startBlock, endBlock);
            return st;
        }
    }

    return KV_ST_OK;
}

/*****************************************************************************/
int keyedvector_remove(keyedvector_p kv, void *key)
{
    kv_cursor_t cursor;
    void *foundKey;

    if (!kv || !key)
        return KV_ST_BADPARAM;

    foundKey = keyedvector_find_by_key(kv, key, KV_LGE_EQUAL, &cursor);
    if (!foundKey)
        return KV_ST_NOTFOUND;

    /* Assumes freeCB will be called by keyedvector_remove_by_cursor */

    return keyedvector_remove_by_cursor(kv, &cursor);
}

/*****************************************************************************/
int keyedvector_size_slow(keyedvector_p kv)
{
    int i;
    int totalSize = 0;

    if (!kv || !kv->Nblocks || !kv->blockNelem)
        return 0;

    for (i = 0; i < kv->Nblocks; i++)
        totalSize += kv->blockNelem[i];

    return totalSize;
}

/*****************************************************************************/
int keyedvector_size(keyedvector_p kv)
{
    if (!kv)
        return 0;

    return kv->Nelem;
}

/*****************************************************************************/
long long keyedvector_bytes_used(keyedvector_p kv)
{
    long long bytesUsed = 0;
    if (!kv)
        return 0;

    // space used by outer struct
    bytesUsed += sizeof(*kv);

    // space used by blockNelem array
    if (kv->blockNelem)
        bytesUsed += sizeof(kv->blockNelem[0]) * kv->maxBlocks;

    // space used by pointers to sub-blocks
    if (kv->blocks)
        bytesUsed += sizeof(kv->blocks[0]) * kv->maxBlocks;

    // space used by actual item storage (total sub-block size)
    bytesUsed += kv->subBlockSize * kv->maxBlocks;

    // the only thing not possible to know is the size of the user-supplied data pointer
    return bytesUsed;
}

/*****************************************************************************/
