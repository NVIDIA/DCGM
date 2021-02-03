#ifndef PROC_H
#define PROC_H

//Proccess and thread basic functions

#ifdef NV_UNIX

#include <pthread.h>
#include <unistd.h>

#if defined(NV_LINUX) || defined(NV_VMWARE)

#include <sys/syscall.h>

/* The following calls gettid() directly via a syscall, because
     * the function itself can be found on none of the known Linux
     * installations (as of Aug 2007).  Note that SYS_gettid has been
     * introduced halfway the 2.4 kernel series, so might not be
     * available on all platforms.
     * We already require Linux 2.6 so this is a safe workaround for
     * when this is compiled against older headers. */
#if !defined(SYS_gettid) && (defined(i386) || defined(__arm__))
#define SYS_gettid 224
#endif

#define getCurrentThreadId() ((unsigned long long)syscall(SYS_gettid))

#elif defined(NV_BSD)

#include <string.h>

#ifndef MIN
#define MIN(X_, Y_) ((X_ < Y_) ? (X_) : (Y_))
#endif

static inline unsigned long long getCurrentThreadId(void)
{
    unsigned long long myTid = 0;
    pthread_t tid            = pthread_self();
    memcpy(&myTid, &tid, MIN(sizeof(myTid), sizeof(tid)));
    return myTid;
}

#elif defined(NV_SUNOS)
#define getCurrentThreadId() ((unsigned long long)pthread_self())
#else
#error "Unrecognized UNIX platform!"
#endif

#else // Windows

#include <windows.h>
#define getCurrentThreadId() (GetCurrentThreadId())

#endif //NV_UNIX

#endif //PROC_H
