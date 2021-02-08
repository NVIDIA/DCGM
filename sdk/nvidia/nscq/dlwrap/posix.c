/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
//
// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
//

#include "os.h"

#include "nscq.h"

#include <dlfcn.h>
#include <errno.h>
#include <pthread.h>
#include <stdio.h>

// Dynamic library state specific to the POSIX implementation.
//  rwlock - reader-writer lock protecting the nscq_dl state
//  handle - library handle (as returned by dlopen())
//  cycle  - a count of which nscq_dl_load()/nscq_dl_unload() cycle we're on
//
// The cycle tracking is done so that we can detect and handle sequences like
// the following:
//  1. call nscq_dl_load()
//  2. call wrapped API nscq_xyz()
//  3. call nscq_dl_unload()
//  4. call nscq_dl_load()
//  5. call wrapped API nscq_xyz() again
//
// In this scenario, we need to detect that the library was unloaded in between steps 2 and 5, and
// so we need to look up the symbol again in case the library code was mapped to a different
// location in the process address space. We do this by acquiring a read-lock on the nscq_dl state
// ensuring it remains loaded while we look up the symbol, and return the cycle count from
// nscq_dl_get(). The wrapper layer can save off this cycle count along with the symbol it looks up
// while under the read lock. For future calls, if the cycle count is different than what is saved
// for the symbol, the wrapper layer can look up the symbol again.
static struct {
    pthread_rwlock_t rwlock;
    void* handle;
    int cycle;
} nscq_dl = {PTHREAD_RWLOCK_INITIALIZER, NULL, 0};

// Helper for nscq_dl_load()
int nscq_dl_load_helper(char *dlname) {
    int ret;

    ret = pthread_rwlock_wrlock(&nscq_dl.rwlock);
    if (ret != 0) {
        return -ret;
    }

    if (nscq_dl.handle == NULL) {
        nscq_dl.handle = dlopen(dlname, RTLD_NOW);
        if (nscq_dl.handle == NULL) {
            ret = -ELIBACC;
        } else if (++nscq_dl.cycle < 0) {
            nscq_dl.cycle = 1;
        }
    }

    pthread_rwlock_unlock(&nscq_dl.rwlock);

    return ret;
}

// Loads the requested major version of the library using dlopen() or the
// DCGM-provided library if standalone is not found on the system.
int nscq_dl_load() {
    int ret;

    ret = nscq_dl_load_helper("libnvidia-nscq.so");

    if (ret == -ELIBACC) {
        ret = nscq_dl_load_helper("libnvidia-nscq-dcgm.so");
    }

    return ret;
}

// Unloads the NSCQ library loaded by nscq_dl_load(), using dlclose().
void nscq_dl_unload(void) {
    if (pthread_rwlock_wrlock(&nscq_dl.rwlock)) {
        return;
    }

    if (nscq_dl.handle != NULL) {
        dlclose(nscq_dl.handle);
        nscq_dl.handle = NULL;
    }

    pthread_rwlock_unlock(&nscq_dl.rwlock);
}

// Make sure the library is loaded, and return the current cycle with the read lock held.
// This must be followed by a call to nscq_dl_put() when the caller is finished with the atomic
// section it required the library to remain loaded for.
int nscq_dl_get(void) {
    if (pthread_rwlock_rdlock(&nscq_dl.rwlock) == 0) {
        if (nscq_dl.handle != NULL) {
            // coverity[missing_unlock] - We are intentionally leaving nscq_dl.rwlock locked
            return nscq_dl.cycle;
        }

        pthread_rwlock_unlock(&nscq_dl.rwlock);
    }

    return -ELIBACC;
}

// Releases a reference to the loaded library by unlocking the rwlock which was acquired by
// nscq_dl_get().
void nscq_dl_put() {
    pthread_rwlock_unlock(&nscq_dl.rwlock);
}

// Load a symbol address from the library by name. This assumes that the caller has an outstanding
// reference to the library acquired with nscq_dl_get().
void* nscq_dl_symbol(const char* sym_name) {
    return dlsym(nscq_dl.handle, sym_name);
}
