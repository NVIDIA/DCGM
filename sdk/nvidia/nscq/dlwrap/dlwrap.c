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

#include "dlwrap.h"

#include "os.h"

#include <stdatomic.h>
#include <stddef.h>

int nscq_dlwrap_attach(void) {
    return nscq_dl_load();
}

void nscq_dlwrap_detach(void) {
    nscq_dl_unload();
}

nscq_rc_t nscq_dlwrap_api_version(uint32_t* version, const char** devel) {
    static atomic_int sym_cycle;
    static uint32_t* version_sym;
    static const char* devel_sym;
    int dl_cycle;
    nscq_rc_t rc = NSCQ_RC_ERROR_EXT;

    if ((dl_cycle = nscq_dl_get()) <= 0) {
        return NSCQ_RC_ERROR_EXT;
    }

    if (sym_cycle != dl_cycle) {
        version_sym = nscq_dl_symbol("nscq_api_version");
        devel_sym = nscq_dl_symbol("nscq_api_version_devel");
        sym_cycle = dl_cycle;
    }

    if (version_sym != NULL && devel_sym != NULL) {
        if (version) {
            *version = *version_sym;
        }
        if (devel) {
            *devel = devel_sym;
        }
        rc = NSCQ_RC_SUCCESS;
    }

    nscq_dl_put();
    return rc;
}

#define NSCQ_DLWRAP_FUNC_IMPL(func_name, return_type, params, args, result_var_type, result_var,   \
                              result_var_assign, ...)                                              \
    return_type func_name params {                                                                 \
        static atomic_int sym_cycle;                                                               \
        static void* _Atomic sym;                                                                  \
        int dl_cycle;                                                                              \
        typedef return_type(*nscq_wrapped_func_t) params;                                          \
        nscq_wrapped_func_t func;                                                                  \
        /* The preprocessor splits macro parameters in a brace initializer like "{x, y}"           \
           as two parameters, "{x" and "y}". Putting this at the end of the argument list and      \
           reattaching them using __VA_ARGS__ is a hack to work around that. */                    \
        result_var_type result_var_assign __VA_ARGS__;                                             \
        if ((dl_cycle = nscq_dl_get()) <= 0) {                                                     \
            return result_var;                                                                     \
        }                                                                                          \
        if (sym_cycle != dl_cycle) {                                                               \
            /* We need to load the symbol. We don't synchronize as it doesn't matter if multiple   \
               threads try to do this at once, since they will all be setting it to the same value \
               since the library handle is locked for reading. It's enough to make the pointer     \
               and cycle atomic, and only update the cycle number after updating the pointer, to   \
               keep it sequentially consistent. */                                                 \
            sym = nscq_dl_symbol(#func_name);                                                      \
            sym_cycle = dl_cycle;                                                                  \
        }                                                                                          \
        /* Casting hack to avoid pedantic warnings from GCC that are basically incompatible with   \
           how nscq_dl_symbol() (by proxy, dlsym()) is designed:                                   \
             warning: ISO C forbids conversion of object pointer to function pointer type */       \
        *(void**)(&func) = sym;                                                                    \
        if (func != NULL) {                                                                        \
            result_var_assign(*func) args;                                                         \
        }                                                                                          \
        nscq_dl_put();                                                                             \
        return result_var;                                                                         \
    }

#define NSCQ_DLWRAP_FUNC(func_name, return_type, params, args, ...)                            \
    NSCQ_DLWRAP_FUNC_IMPL(func_name, return_type, params, args, return_type, result, result =, \
                          __VA_ARGS__)
#define NSCQ_DLWRAP_FUNC_VOID(func_name, params, args)                                        \
    NSCQ_DLWRAP_FUNC_IMPL(func_name, void, params, args, /* result_var_type */, /* result */, \
                          /* result_var_assign */,                                            \
                          /* __VA_ARGS__ */)

// clang-format off
NSCQ_DLWRAP_FUNC     (nscq_uuid_to_label, nscq_rc_t,
                      (const nscq_uuid_t* uuid, nscq_label_t* label, uint32_t flags),
                      (uuid, label, flags),
                      NSCQ_RC_ERROR_EXT)
NSCQ_DLWRAP_FUNC     (nscq_session_create, nscq_session_result_t,
                      (uint32_t flags),
                      (flags),
                      {NSCQ_RC_ERROR_EXT, NULL})
NSCQ_DLWRAP_FUNC_VOID(nscq_session_destroy,
                      (nscq_session_t session),
                      (session))
NSCQ_DLWRAP_FUNC     (nscq_session_mount, nscq_rc_t,
                      (nscq_session_t session, const nscq_uuid_t* uuid, uint32_t flags),
                      (session, uuid, flags),
                      NSCQ_RC_ERROR_EXT)
NSCQ_DLWRAP_FUNC_VOID(nscq_session_unmount,
                      (nscq_session_t session, const nscq_uuid_t* uuid),
                      (session, uuid))
NSCQ_DLWRAP_FUNC     (nscq_session_path_observe, nscq_rc_t,
                      (nscq_session_t session, const char* path, nscq_fn_t callback, void* data, uint32_t flags),
                      (session, path, callback, data, flags),
                      NSCQ_RC_ERROR_EXT)
NSCQ_DLWRAP_FUNC     (nscq_session_path_register_observer, nscq_observer_result_t,
                      (nscq_session_t session, const char* path, nscq_fn_t callback, void* data, uint32_t flags),
                      (session, path, callback, data, flags), {NSCQ_RC_ERROR_EXT, NULL})
NSCQ_DLWRAP_FUNC_VOID(nscq_observer_deregister,
                      (nscq_observer_t observer),
                      (observer))
NSCQ_DLWRAP_FUNC     (nscq_observer_observe, nscq_rc_t,
                      (nscq_observer_t observer, uint32_t flags),
                      (observer, flags),
                      NSCQ_RC_ERROR_EXT)
NSCQ_DLWRAP_FUNC     (nscq_session_observe, nscq_rc_t,
                      (nscq_session_t session, uint32_t flags),
                      (session, flags),
                      NSCQ_RC_ERROR_EXT)
// clang-format on
