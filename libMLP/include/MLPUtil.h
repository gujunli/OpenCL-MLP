/*
 *  COPYRIGHT:  Copyright (c) 2014 Advanced Micro Devices, Inc.  All rights reserved
 *
 *   Written by Qianfeng Zhang@amd.com ( March 2014 )
 *
 */


#ifndef _MLP_UTIL_H_
#define _MLP_UTIL_H_

#include "DNNUtil.h"

#define MLP_CHECK(flag)                                                                                           \
	do  {                                                                                                         \
        int _tmpVal;                                                                                              \
	    if ( (_tmpVal = flag) != 0) {                                                                             \
	        ostringstream mystream;                                                                               \
		    mystream << "MLP Function Failed (" << __FILE__ << "," << __LINE__ << "), Error Code:" << _tmpVal;    \
		    dnn_log("MLP", mystream.str().c_str());                                                               \
			DNN_Exception("MLP function call returned unexpected value");                                         \
	    }                                                                                                         \
	}                                                                                                             \
    while (0)

#define CL_CHECK(flag)                                                                                            \
	do  {                                                                                                         \
	    int _tmpVal;                                                                                              \
	    if ( (_tmpVal = flag) != 0) {                                                                             \
	        ostringstream mystream;                                                                               \
		    mystream << "OpenCL Function Failed (" << __FILE__ << "," << __LINE__ << "), Error Code:" << _tmpVal; \
		    dnn_log("MLP", mystream.str().c_str());                                                               \
			DNN_Exception("OpenCL function call returned unexpected value");                                      \
	    }                                                                                                         \
	}                                                                                                             \
    while (0)

#define AMDBLAS_CHECK(flag)                                                                                         \
	do  {                                                                                                           \
	    int _tmpVal;                                                                                                \
	    if ( (_tmpVal = flag) != 0) {                                                                               \
	        ostringstream mystream;                                                                                 \
		    mystream << "AmdBlas Function Failed (" << __FILE__ << "," << __LINE__ << "), Error Code:" << _tmpVal;  \
		    dnn_log("MLP", mystream.str().c_str());                                                                 \
            DNN_Exception("clAmdBlas function call returned unexpected value");                                     \
	    }                                                                                                           \
	}                                                                                                               \
	while (0)


#define mlp_log(header,content)         dnn_log(header,content)
#define mlp_log_retval(header,retval)   dnn_log_retval(header,retval)

#define MLP_Exception(info) DNN_Exception(info)
#define MLP_BadAlloc(info)  DNN_BadAlloc(info)

#endif   // end of _MLP_UTIL_H


