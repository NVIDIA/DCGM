/*
 * Illinois Open Source License
 *
 * University of Illinois/NCSA
 * Open Source License
 *
 * Copyright 2009,    University of Illinois.  All rights reserved.
 *
 * Developed by:
 *
 * Innovative Systems Lab
 * National Center for Supercomputing Applications
 * http://www.ncsa.uiuc.edu/AboutUs/Directorates/ISL.html
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal with
 * the Software without restriction, including without limitation the rights to use,
 * copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
 * Software, and to permit persons to whom the Software is furnished to do so, subject
 * to the following conditions:
 *
 * * Redistributions of source code must retain the above copyright notice, this list
 * of conditions and the following disclaimers.
 *
 * * Redistributions in binary form must reproduce the above copyright notice, this list
 * of conditions and the following disclaimers in the documentation and/or other materials
 * provided with the distribution.
 *
 * * Neither the names of the Innovative Systems Lab, the National Center for Supercomputing
 * Applications, nor the names of its contributors may be used to endorse or promote products
 * derived from this Software without specific prior written permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
 * PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE
 * LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT
 * OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS WITH THE SOFTWARE.
 */

#include "misc.h"
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <unistd.h>

#define RECORD_ERR(err, p, expect, current) do{		\
	unsigned int idx = atomicAdd(err, 1);		\
	idx = idx % MAX_ERR_RECORD_COUNT;		\
	err_addr[idx] = (unsigned long)p;		\
	err_expect[idx] = (unsigned long)expect;	\
	err_current[idx] = (unsigned long)current;	\
	err_second_read[idx] = (unsigned long)(*p);	\
}while(0)

extern "C"
__global__ void
move_inv_write(char* _ptr, char* end_ptr, unsigned int pattern)
{
    unsigned int i;
    unsigned int* ptr = (unsigned int*) (_ptr + blockIdx.x*BLOCKSIZE);
    if (ptr >= (unsigned int*) end_ptr) {
	return;
    }

    for (i = 0;i < BLOCKSIZE/sizeof(unsigned int); i++){
	ptr[i] = pattern;
    }

    return;
}


extern "C"
__global__ void
move_inv_readwrite(char* _ptr, char* end_ptr, unsigned int p1, unsigned int p2, unsigned int* err,
			  unsigned long* err_addr, unsigned long* err_expect, unsigned long* err_current, unsigned long* err_second_read)
{
    unsigned int i;
    unsigned int* ptr = (unsigned int*) (_ptr + blockIdx.x*BLOCKSIZE);
    if (ptr >= (unsigned int*) end_ptr) {
	return;
    }

    for (i = 0;i < BLOCKSIZE/sizeof(unsigned int); i++){
	if (ptr[i] != p1){
	    RECORD_ERR(err, &ptr[i], p1, ptr[i]);
	}
	ptr[i] = p2;

    }

    return;
}

extern "C" __global__ void
move_inv_read(char* _ptr, char* end_ptr,  unsigned int pattern, unsigned int* err,
		     unsigned long* err_addr, unsigned long* err_expect, unsigned long* err_current, unsigned long* err_second_read )
{
    unsigned int i;
    unsigned int* ptr = (unsigned int*) (_ptr + blockIdx.x*BLOCKSIZE);
    if (ptr >= (unsigned int*) end_ptr) {
	return;
    }

    for (i = 0;i < BLOCKSIZE/sizeof(unsigned int); i++){
	if (ptr[i] != pattern){
	    RECORD_ERR(err, &ptr[i], pattern, ptr[i]);
	}
    }

    return;
}

extern "C" __global__ void
test0_global_write(char *_ptr, char *_end_ptr)
{
    unsigned int *ptr      = (unsigned int *)_ptr;
    unsigned int *end_ptr  = (unsigned int *)_end_ptr;
    unsigned int *orig_ptr = ptr;

    unsigned int pattern = 1;

    unsigned long mask = 4;

    *ptr = pattern;

    while (ptr < end_ptr)
    {
        ptr = (unsigned int *)(((unsigned long)orig_ptr) | mask);
        if (ptr == orig_ptr)
        {
            mask = mask << 1;
            continue;
        }
        if (ptr >= end_ptr)
        {
            break;
        }

        *ptr = pattern;

        pattern = pattern << 1;
        mask    = mask << 1;
    }

    return;
}

extern "C"
__global__ void
test0_global_read(char* _ptr, char* _end_ptr, unsigned int* err, unsigned long* err_addr,
			 unsigned long* err_expect, unsigned long* err_current, unsigned long* err_second_read)
{
    unsigned int* ptr = (unsigned int*)_ptr;
    unsigned int* end_ptr = (unsigned int*)_end_ptr;
    unsigned int* orig_ptr = ptr;

    unsigned int pattern = 1;

    unsigned long mask = 4;

    if (*ptr != pattern){
	RECORD_ERR(err, ptr, pattern, *ptr);
    }

    while(ptr < end_ptr){

	ptr = (unsigned int*) ( ((unsigned long)orig_ptr) | mask);
	if (ptr == orig_ptr){
	    mask = mask <<1;
	    continue;
	}
	if (ptr >= end_ptr){
	    break;
	}

	if (*ptr != pattern){
	    RECORD_ERR(err, ptr, pattern, *ptr);
	}

	pattern = pattern << 1;
	mask = mask << 1;
    }

    return;
}

extern "C" __global__ void
test0_write(char* _ptr, char* end_ptr)
{
    unsigned int* orig_ptr = (unsigned int*) (_ptr + blockIdx.x*BLOCKSIZE);;
    unsigned int* ptr = orig_ptr;
    if (ptr >= (unsigned int*) end_ptr) {
	return;
    }

    unsigned int* block_end = orig_ptr + BLOCKSIZE/sizeof(unsigned int);

    unsigned int pattern = 1;

    unsigned long mask = 4;

    *ptr = pattern;

    while(ptr < block_end){

	ptr = (unsigned int*) ( ((unsigned long)orig_ptr) | mask);
	if (ptr == orig_ptr){
	    mask = mask <<1;
	    continue;
	}
	if (ptr >= block_end){
	    break;
	}

	*ptr = pattern;

	pattern = pattern << 1;
	mask = mask << 1;
    }

    return;
}

extern "C" __global__ void
test0_read(char* _ptr, char* end_ptr, unsigned int* err, unsigned long* err_addr,
		  unsigned long* err_expect, unsigned long* err_current, unsigned long* err_second_read)
{
    unsigned int* orig_ptr = (unsigned int*) (_ptr + blockIdx.x*BLOCKSIZE);;
    unsigned int* ptr = orig_ptr;
    if (ptr >= (unsigned int*) end_ptr) {
	return;
    }

    unsigned int* block_end = orig_ptr + BLOCKSIZE/sizeof(unsigned int);

    unsigned int pattern = 1;

    unsigned long mask = 4;
    if (*ptr != pattern){
	RECORD_ERR(err, ptr, pattern, *ptr);
    }

    while(ptr < block_end){

	ptr = (unsigned int*) ( ((unsigned long)orig_ptr) | mask);
	if (ptr == orig_ptr){
	    mask = mask <<1;
	    continue;
	}
	if (ptr >= block_end){
	    break;
	}

	if (*ptr != pattern){
	    RECORD_ERR(err, ptr, pattern, *ptr);
	}

	pattern = pattern << 1;
	mask = mask << 1;
    }
    return;
}

extern "C" __global__ void
test1_write(char* _ptr, char* end_ptr, unsigned int* err)
{
    unsigned int i;
    unsigned long* ptr = (unsigned long*) (_ptr + blockIdx.x*BLOCKSIZE);

    if (ptr >= (unsigned long*) end_ptr) {
	return;
    }


    for (i = 0;i < BLOCKSIZE/sizeof(unsigned long); i++){
	ptr[i] =(unsigned long) & ptr[i];
    }
    return;
}

extern "C" __global__ void
test1_read(char* _ptr, char* end_ptr, unsigned int* err, unsigned long* err_addr,
		  unsigned long* err_expect, unsigned long* err_current, unsigned long* err_second_read)
{
    unsigned int i;
    unsigned long* ptr = (unsigned long*) (_ptr + blockIdx.x*BLOCKSIZE);

    if (ptr >= (unsigned long*) end_ptr) {
	return;
    }


    for (i = 0;i < BLOCKSIZE/sizeof(unsigned long); i++){
	if (ptr[i] != (unsigned long)& ptr[i]){
	    RECORD_ERR(err, &ptr[i], (unsigned long)&ptr[i], ptr[i]);
	}
    }
    return;
}

extern "C" __global__ void
test5_init(char* _ptr, char* end_ptr)
{
    unsigned int i;
    unsigned int* ptr = (unsigned int*) (_ptr + blockIdx.x*BLOCKSIZE);

    if (ptr >= (unsigned int*) end_ptr) {
	return;
    }

    unsigned int p1 = 1;
    for (i = 0;i < BLOCKSIZE/sizeof(unsigned int); i+=16){
	unsigned int p2 = ~p1;

	ptr[i] = p1;
	ptr[i+1] = p1;
	ptr[i+2] = p2;
	ptr[i+3] = p2;
	ptr[i+4] = p1;
	ptr[i+5] = p1;
	ptr[i+6] = p2;
	ptr[i+7] = p2;
	ptr[i+8] = p1;
	ptr[i+9] = p1;
	ptr[i+10] = p2;
	ptr[i+11] = p2;
	ptr[i+12] = p1;
	ptr[i+13] = p1;
	ptr[i+14] = p2;
	ptr[i+15] = p2;

	p1 = p1<<1;
	if (p1 == 0){
	    p1 = 1;
	}
    }
    return;
}

extern "C" __global__ void
test5_move(char* _ptr, char* end_ptr)
{
    unsigned int i;
    unsigned int* ptr = (unsigned int*) (_ptr + blockIdx.x*BLOCKSIZE);

    if (ptr >= (unsigned int*) end_ptr) {
	return;
    }

    unsigned int half_count = BLOCKSIZE/sizeof(unsigned int)/2;
    unsigned int* ptr_mid = ptr + half_count;

    for (i = 0;i < half_count; i++){
	ptr_mid[i] = ptr[i];
    }

    for (i=0;i < half_count - 8; i++){
	ptr[i + 8] = ptr_mid[i];
    }

    for (i=0;i < 8; i++){
	ptr[i] = ptr_mid[half_count - 8 + i];
    }

    return;
}

extern "C" __global__ void
test5_check(char* _ptr, char* end_ptr, unsigned int* err, unsigned long* err_addr,
		   unsigned long* err_expect, unsigned long* err_current, unsigned long* err_second_read)
{
    unsigned int i;
    unsigned int* ptr = (unsigned int*) (_ptr + blockIdx.x*BLOCKSIZE);

    if (ptr >= (unsigned int*) end_ptr) {
	return;
    }

    for (i=0;i < BLOCKSIZE/sizeof(unsigned int); i+=2){
	if (ptr[i] != ptr[i+1]){
	    RECORD_ERR(err, &ptr[i], ptr[i+1], ptr[i]);
	}
    }

    return;
}

extern "C" __global__ void
movinv32_write(char* _ptr, char* end_ptr, unsigned int pattern,
		unsigned int lb, unsigned int sval, unsigned int offset)
{

    unsigned int i;
    unsigned int* ptr = (unsigned int*) (_ptr + blockIdx.x*BLOCKSIZE);

    if (ptr >= (unsigned int*) end_ptr) {
	return;
    }

    unsigned int k = offset;
    unsigned pat = pattern;
    for (i = 0;i < BLOCKSIZE/sizeof(unsigned int); i++){
	ptr[i] = pat;
	k++;
	if (k >= 32){
	    k=0;
	    pat = lb;
	}else{
	    pat = pat << 1;
	    pat |= sval;
	}
    }

    return;
}

extern "C" __global__ void
movinv32_readwrite(char* _ptr, char* end_ptr, unsigned int pattern,
			  unsigned int lb, unsigned int sval, unsigned int offset, unsigned int * err,
			  unsigned long* err_addr, unsigned long* err_expect, unsigned long* err_current, unsigned long* err_second_read)
{
    unsigned int i;
    unsigned int* ptr = (unsigned int*) (_ptr + blockIdx.x*BLOCKSIZE);

    if (ptr >= (unsigned int*) end_ptr) {
	return;
    }

    unsigned int k = offset;
    unsigned pat = pattern;
    for (i = 0;i < BLOCKSIZE/sizeof(unsigned int); i++){
	if (ptr[i] != pat){
	    RECORD_ERR(err, &ptr[i], pat, ptr[i]);
	}

	ptr[i] = ~pat;

	k++;
	if (k >= 32){
	    k=0;
	    pat = lb;
	}else{
	    pat = pat << 1;
	    pat |= sval;
	}
    }

    return;
}

extern "C" __global__ void
movinv32_read(char* _ptr, char* end_ptr, unsigned int pattern,
		     unsigned int lb, unsigned int sval, unsigned int offset, unsigned int * err,
		     unsigned long* err_addr, unsigned long* err_expect, unsigned long* err_current, unsigned long* err_second_read)
{
    unsigned int i;
    unsigned int* ptr = (unsigned int*) (_ptr + blockIdx.x*BLOCKSIZE);

    if (ptr >= (unsigned int*) end_ptr) {
	return;
    }

    unsigned int k = offset;
    unsigned pat = pattern;
    for (i = 0;i < BLOCKSIZE/sizeof(unsigned int); i++){
	if (ptr[i] != ~pat){
	    RECORD_ERR(err, &ptr[i], ~pat, ptr[i]);
	}

	k++;
	if (k >= 32){
	    k=0;
	    pat = lb;
	}else{
	    pat = pat << 1;
	    pat |= sval;
	}
    }

    return;
}

extern "C" __global__ void
test7_write(char* _ptr, char* end_ptr, char* _start_ptr, unsigned int* err)
{
    unsigned int i;
    unsigned int* ptr = (unsigned int*) (_ptr + blockIdx.x*BLOCKSIZE);
    unsigned int* start_ptr = (unsigned int*) _start_ptr;

    if (ptr >= (unsigned int*) end_ptr) {
	return;
    }


    for (i = 0;i < BLOCKSIZE/sizeof(unsigned int); i++){
	ptr[i] = start_ptr[i];
    }

    return;
}


extern "C" __global__ void
test7_readwrite(char* _ptr, char* end_ptr, char* _start_ptr, unsigned int* err,
		       unsigned long* err_addr, unsigned long* err_expect, unsigned long* err_current, unsigned long* err_second_read)
{
    unsigned int i;
    unsigned int* ptr = (unsigned int*) (_ptr + blockIdx.x*BLOCKSIZE);
    unsigned int* start_ptr = (unsigned int*) _start_ptr;

    if (ptr >= (unsigned int*) end_ptr) {
	return;
    }


    for (i = 0;i < BLOCKSIZE/sizeof(unsigned int); i++){
	if (ptr[i] != start_ptr[i]){
	    RECORD_ERR(err, &ptr[i], start_ptr[i], ptr[i]);
	}
	ptr[i] = ~(start_ptr[i]);
    }

    return;
}

extern "C" __global__ void
test7_read(char* _ptr, char* end_ptr, char* _start_ptr, unsigned int* err, unsigned long* err_addr,
		  unsigned long* err_expect, unsigned long* err_current, unsigned long* err_second_read)
{
    unsigned int i;
    unsigned int* ptr = (unsigned int*) (_ptr + blockIdx.x*BLOCKSIZE);
    unsigned int* start_ptr = (unsigned int*) _start_ptr;

    if (ptr >= (unsigned int*) end_ptr) {
	return;
    }


    for (i = 0;i < BLOCKSIZE/sizeof(unsigned int); i++){
	if (ptr[i] != ~(start_ptr[i])){
	    RECORD_ERR(err, &ptr[i], ~(start_ptr[i]), ptr[i]);
	}
    }

    return;
}


extern "C" __global__ void
modtest_write(char* _ptr, char* end_ptr, unsigned int offset, unsigned int p1, unsigned int p2)
{
    unsigned int i;
    unsigned int* ptr = (unsigned int*) (_ptr + blockIdx.x*BLOCKSIZE);

    if (ptr >= (unsigned int*) end_ptr) {
	return;
    }

    for (i = offset;i < BLOCKSIZE/sizeof(unsigned int); i+=MOD_SZ){
	ptr[i] =p1;
    }

    for (i = 0;i < BLOCKSIZE/sizeof(unsigned int); i++){
	if (i % MOD_SZ != offset){
	    ptr[i] =p2;
	}
    }

    return;
}


extern "C" __global__ void
modtest_read(char* _ptr, char* end_ptr, unsigned int offset, unsigned int p1, unsigned int* err,
		    unsigned long* err_addr, unsigned long* err_expect, unsigned long* err_current, unsigned long* err_second_read)
{
    unsigned int i;
    unsigned int* ptr = (unsigned int*) (_ptr + blockIdx.x*BLOCKSIZE);

    if (ptr >= (unsigned int*) end_ptr) {
	return;
    }

    for (i = offset;i < BLOCKSIZE/sizeof(unsigned int); i+=MOD_SZ){
	if (ptr[i] !=p1){
	    RECORD_ERR(err, &ptr[i], p1, ptr[i]);
	}
    }

    return;
}

extern "C" __global__ void
test10_write(char* ptr, unsigned long long memsize, unsigned long p1)
{
    unsigned int i;
    unsigned long avenumber = memsize/(gridDim.x*gridDim.y);
    unsigned long* mybuf = (unsigned long*)(ptr + blockIdx.x* avenumber);
    unsigned int n = avenumber/(blockDim.x*sizeof(unsigned long));

    for(i=0;i < n;i++){
        unsigned int index = i*blockDim.x + threadIdx.x;
        mybuf[index]= p1;
    }
    unsigned int index = n*blockDim.x + threadIdx.x;
    if (index*sizeof(unsigned long) < avenumber){
        mybuf[index] = p1;
    }

    return;
}

extern "C" __global__ void
test10_readwrite(char* ptr, unsigned long long memsize, unsigned long p1, unsigned long p2,  unsigned int* err,
					unsigned long* err_addr, unsigned long* err_expect, unsigned long* err_current, unsigned long* err_second_read)
{
    unsigned int i;
    unsigned long avenumber = memsize/(gridDim.x*gridDim.y);
    unsigned long* mybuf = (unsigned long*)(ptr + blockIdx.x* avenumber);
    unsigned int n = avenumber/(blockDim.x*sizeof(unsigned long));
    unsigned long localp;

    for(i=0;i < n;i++){
        unsigned int index = i*blockDim.x + threadIdx.x;
        localp = mybuf[index];
        if (localp != p1){
	    RECORD_ERR(err, &mybuf[index], p1, localp);
	}
	mybuf[index] = p2;
    }
    unsigned int index = n*blockDim.x + threadIdx.x;
    if (index*sizeof(unsigned long) < avenumber){
	localp = mybuf[index];
	if (localp!= p1){
	    RECORD_ERR(err, &mybuf[index], p1, localp);
	}
	mybuf[index] = p2;
    }

    return;
}
