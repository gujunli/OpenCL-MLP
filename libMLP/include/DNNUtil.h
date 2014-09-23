/*
 *  COPYRIGHT:  Copyright (c) 2014 Advanced Micro Devices, Inc.  All rights reserved
 *
 *   Written by Qianfeng Zhang@amd.com ( March 2014 )
 *
 */


#ifndef _DNN_UTIL_H_
#define _DNN_UTIL_H_

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

#include "DNNApiExport.h"

using namespace std;

struct dnn_tv {
   long tv_sec;
   long tv_usec;
};

LIBDNNAPI extern void getCurrentTime(struct dnn_tv *tv);
LIBDNNAPI extern long diff_msec(struct dnn_tv *stv, struct dnn_tv *etv);
LIBDNNAPI extern long diff_usec(struct dnn_tv *stv, struct dnn_tv *etv);

extern int getLogicCoreNum();

extern int read_srcfile(const char *filename, char * &src_str);

LIBDNNAPI extern void dnn_log(const char *header, const char *content);
LIBDNNAPI extern void dnn_log_retval(const char *header, int retVal);

#define DNN_Exception(info) throw runtime_error(#info)
#define DNN_BadAlloc(info)  throw bad_alloc(#info)


#define DNN_CHECK(flag)                                                                                           \
	do  {                                                                                                         \
        int _tmpVal;                                                                                              \
	    if ( (_tmpVal = flag) != 0) {                                                                             \
	        ostringstream mystream;                                                                               \
		    mystream << "DNN Function Failed (" << __FILE__ << "," << __LINE__ << "), Error Code:" << _tmpVal;    \
		    dnn_log("DNN", mystream.str().c_str());                                                               \
			DNN_Exception("DNN function call returned unexpected value");                                         \
	    }                                                                                                         \
	}                                                                                                             \
    while (0)


#ifdef _WIN32
#define  DNN_CREATE_THREAD(threadStruct_p,threadFun,threadArg_p)                                                                         \
	do  {                                                                                                                                \
         if ( (*threadStruct_p = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)threadFun, (LPVOID)threadArg_p, 0 ,NULL)) == NULL ) {     \
               dnn_log("DNN", "Failed to create threads");                                                         \
               DNN_Exception("Failed to create OS thread");                                                        \
		 }                                                                                                         \
	}                                                                                                              \
	while (0)
#else
#define  DNN_CREATE_THREAD(threadStruct_p,threadFun,threadArg_p)                                                   \
	do  {                                                                                                          \
         if ( pthread_create(threadStruct_p, NULL, threadFun, (void*)threadArg_p) )  {                            \
              dnn_log("DNN", "Failed to create threads");                                                          \
              DNN_Exception("");                                                                                   \
		 }                                                                                                         \
	}                                                                                                              \
    while (0)
#endif

#ifdef _WIN32
#define DNN_JOIN_THREAD(threadStruct)                                   \
    do  {                                                               \
            WaitForSingleObject(threadStruct, INFINITE);                \
            CloseHandle(threadStruct);                                  \
	}                                                                   \
    while (0)
#else
#define DNN_JOIN_THREAD(threadStruct)                                   \
    do  {                                                               \
          pthread_join(threadStruct, NULL);                             \
	}                                                                   \
	while (0)
#endif

#ifdef _WIN32
#define DNN_KILL_THREAD(threadStruct)                                   \
    do  {                                                               \
        (void) TerminateThread(threadStruct, 9);                        \
	}                                                                   \
    while (0)
#else
#define DNN_KILL_THREAD(threadStruct)                                   \
    do  {                                                               \
          pthread_cancel(threadStruct);                                 \
	}                                                                   \
	while (0)
#endif

#ifdef _WIN32
#define DNN_SLEEP(seconds)                                              \
	do {                                                                \
	    Sleep(seconds*1000);                                            \
	}                                                                   \
	while (0)
#else
#define DNN_SLEEP(seconds)                                              \
	do {                                                                \
	    sleep(seconds);                                                 \
	}                                                                   \
	while (0)
#endif

#ifdef _WIN32
#define DNN_LOCK_INIT(lockp)                                            \
	do {                                                                \
        InitializeCriticalSection(lockp);                               \
	}                                                                   \
	while (0)
#else
#define DNN_LOCK_INIT(lockp)                                            \
	do {                                                                \
	    pthread_mutex_init((lockp),NULL);                               \
	}                                                                   \
	while (0)
#endif


#ifdef _WIN32
#define DNN_LOCK(lockp)                                                 \
	do {                                                                \
        EnterCriticalSection(lockp);                                    \
	}                                                                   \
	while (0)
#else
#define DNN_LOCK(lockp)                                                 \
	do {                                                                \
	    pthread_mutex_lock(lockp);                                      \
	}                                                                   \
	while (0)
#endif

#ifdef _WIN32
#define DNN_UNLOCK(lockp)                                               \
	do {                                                                \
        LeaveCriticalSection(lockp);                                    \
	}                                                                   \
	while (0)
#else
#define DNN_UNLOCK(lockp)                                               \
	do {                                                                \
	    pthread_mutex_unlock(lockp);                                    \
	}                                                                   \
	while (0)
#endif


#endif   // end of _DNN_UTIL_H


