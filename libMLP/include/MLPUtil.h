/*
 *  COPYRIGHT:  Copyright (c) 2014 Advanced Micro Devices, Inc.  All rights reserved
 *
 *   Written by Qianfeng Zhang@amd.com ( March 2014 )
 *
 */


#ifndef _MLP_UTIL_H_
#define _MLP_UTIL_H_

#ifdef _WIN32
#include <windows.h>
#else
#include <pthread.h>
#include <signal.h>
#include <unistd.h>
#endif

#include <sstream>
#include <exception>
#include <stdexcept>

#include "MLPApiExport.h"

using namespace std;

struct mlp_tv {
   long tv_sec;
   long tv_usec;
};

LIBMLPAPI extern void getCurrentTime(struct mlp_tv *tv);
LIBMLPAPI extern long diff_msec(struct mlp_tv *stv, struct mlp_tv *etv);
LIBMLPAPI extern long diff_usec(struct mlp_tv *stv, struct mlp_tv *etv);

extern int getLogicCoreNum();

extern int read_srcfile(const char *filename, char * &src_str);

LIBMLPAPI extern void mlp_log(const char *header, const char *content);
LIBMLPAPI extern void mlp_log_retval(const char *header, int retVal);

#define MLP_Exception(info) throw runtime_error(#info)
#define MLP_BadAlloc(info)  throw bad_alloc(#info)


#define MLP_CHECK(flag)                                                                                           \
	do  {                                                                                                         \
        int _tmpVal;                                                                                              \
	    if ( (_tmpVal = flag) != 0) {                                                                             \
	        ostringstream mystream;                                                                               \
		    mystream << "MLP Function Failed (" << __FILE__ << "," << __LINE__ << "), Error Code:" << _tmpVal;    \
		    mlp_log("MLP", mystream.str().c_str());                                                               \
			MLP_Exception("MLP function call returned unexpected value");                                         \
	    }                                                                                                         \
	}                                                                                                             \
    while (0)

#define CL_CHECK(flag)                                                                                            \
	do  {                                                                                                         \
	    int _tmpVal;                                                                                              \
	    if ( (_tmpVal = flag) != 0) {                                                                             \
	        ostringstream mystream;                                                                               \
		    mystream << "OpenCL Function Failed (" << __FILE__ << "," << __LINE__ << "), Error Code:" << _tmpVal; \
		    mlp_log("MLP", mystream.str().c_str());                                                               \
			MLP_Exception("OpenCL function call returned unexpected value");                                      \
	    }                                                                                                         \
	}                                                                                                             \
    while (0)

#define AMDBLAS_CHECK(flag)                                                                                         \
	do  {                                                                                                           \
	    int _tmpVal;                                                                                                \
	    if ( (_tmpVal = flag) != 0) {                                                                               \
	        ostringstream mystream;                                                                                 \
		    mystream << "AmdBlas Function Failed (" << __FILE__ << "," << __LINE__ << "), Error Code:" << _tmpVal;  \
		    mlp_log("MLP", mystream.str().c_str());                                                                 \
            MLP_Exception("clAmdBlas function call returned unexpected value");                                     \
	    }                                                                                                           \
	}                                                                                                               \
	while (0)


#ifdef _WIN32
#define  MLP_CREATE_THREAD(threadStruct_p,threadFun,threadArg_p)                                                                         \
	do  {                                                                                                                                \
         if ( (*threadStruct_p = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)threadFun, (LPVOID)threadArg_p, 0 ,NULL)) == NULL ) {     \
               mlp_log("MLP", "Failed to create threads");                                                         \
               MLP_Exception("Failed to create OS thread");                                                        \
		 }                                                                                                         \
	}                                                                                                              \
	while (0)
#else
#define  MLP_CREATE_THREAD(threadStruct_p,threadFun,threadArg_p)                                                   \
	do  {                                                                                                          \
         if ( pthread_create(threadStruct_p, NULL, threadFun, (void*)threadArg_p) )  {                            \
              mlp_log("MLP", "Failed to create threads");                                                          \
              MLP_Exception("");                                                                                   \
		 }                                                                                                         \
	}                                                                                                              \
    while (0)
#endif

#ifdef _WIN32
#define MLP_JOIN_THREAD(threadStruct)                                   \
    do  {                                                               \
            WaitForSingleObject(threadStruct, INFINITE);                \
            CloseHandle(threadStruct);                                  \
	}                                                                   \
    while (0)
#else
#define MLP_JOIN_THREAD(threadStruct)                                   \
    do  {                                                               \
          pthread_join(threadStruct, NULL);                             \
	}                                                                   \
	while (0)
#endif

#ifdef _WIN32
#define MLP_KILL_THREAD(threadStruct)                                   \
    do  {                                                               \
        (void) TerminateThread(threadStruct, 9);                        \
	}                                                                   \
    while (0)
#else
#define MLP_KILL_THREAD(threadStruct)                                   \
    do  {                                                               \
          pthread_cancel(threadStruct);                                 \
	}                                                                   \
	while (0)
#endif

#ifdef _WIN32
#define MLP_SLEEP(seconds)                                              \
	do {                                                                \
	    Sleep(seconds*1000);                                            \
	}                                                                   \
	while (0)
#else
#define MLP_SLEEP(seconds)                                              \
	do {                                                                \
	    sleep(seconds);                                                 \
	}                                                                   \
	while (0)
#endif

#ifdef _WIN32
#define MLP_LOCK_INIT(lockp)                                            \
	do {                                                                \
        InitializeCriticalSection(lockp);                               \
	}                                                                   \
	while (0)
#else
#define MLP_LOCK_INIT(lockp)                                            \
	do {                                                                \
	    pthread_mutex_init((lockp),NULL);                               \
	}                                                                   \
	while (0)
#endif


#ifdef _WIN32
#define MLP_LOCK(lockp)                                                 \
	do {                                                                \
        EnterCriticalSection(lockp);                                    \
	}                                                                   \
	while (0)
#else
#define MLP_LOCK(lockp)                                                 \
	do {                                                                \
	    pthread_mutex_lock(lockp);                                      \
	}                                                                   \
	while (0)
#endif

#ifdef _WIN32
#define MLP_UNLOCK(lockp)                                               \
	do {                                                                \
        LeaveCriticalSection(lockp);                                    \
	}                                                                   \
	while (0)
#else
#define MLP_UNLOCK(lockp)                                               \
	do {                                                                \
	    pthread_mutex_unlock(lockp);                                    \
	}                                                                   \
	while (0)
#endif


// conversion of word type data between host and assigned endian
static inline void LEtoLEHosts(unsigned short &x)
{
};

static inline void LEtoHosts(unsigned short &x)
{
    LEtoLEHosts(x);       // for x86 and x86-64
};

static inline void LEHostToLEs(unsigned short &x)
{
};

static inline void HostToLEs(unsigned short &x)
{
   	LEHostToLEs(x);  // for x86 architecture
	                 // ToDO: other architecture
};

static inline void BEtoLEHosts(unsigned short &x)
{
    x = ( (x >> 8) & 0x00ff ) | ( (x << 8) & 0xff00 );
};

static inline void BEtoHosts(unsigned short &x)
{
    BEtoLEHosts(x);      // for x86 and x86-64
};

static inline void LEHostToBEs(unsigned short &x)
{
    x = ( (x >> 8) & 0x00ff ) | ( (x << 8) & 0xff00 );
};

static inline void HostToBEs(unsigned short &x)
{
   	LEHostToBEs(x);  // for x86 architecture
	                 // ToDO: other architecture
};

// conversion of dword type data between host and assigned endian

static void BEtoLEHostl(unsigned int &x)
{
  x = ((x>>24) & 0x000000FF) | ((x<<8) & 0x00FF0000) | ((x>>8) & 0x0000FF00) | ((x<<24) & 0xFF000000);
};

static void BEtoHostl(unsigned int &x)
{
	BEtoLEHostl(x);  // for x86 architecture
}; 

static inline void LEHostToBEl(unsigned int &x)
{
  x = ((x>>24) & 0x000000FF) | ((x<<8) & 0x00FF0000) | ((x>>8) & 0x0000FF00) | ((x<<24) & 0xFF000000);
};

static inline void HostToBEl(unsigned int &x)
{
	LEHostToBEl(x);   // for x86 architecture
	                  // TODO: other architectures
};

static inline void LEHostToLEl(unsigned int &x)
{
};

static inline void HostToLEl(unsigned int &x)
{
   	LEHostToLEl(x);  // for x86 architecture
	                 // ToDO: other architecture
};

static inline void LEtoLEHostl(unsigned int &x)
{
};

static inline void LEtoHostl(unsigned int &x)
{
	LEtoLEHostl(x);  // for x86 architecture
                     // ToDO: other architecture
};


#endif   // end of _MLP_UTIL_H


