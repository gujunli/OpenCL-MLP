/*
 *  COPYRIGHT:  Copyright (c) 2014 Advanced Micro Devices, Inc.  All rights reserved
 *
 *   Written by Qianfeng Zhang@amd.com ( March 2014 )
 *
 */

#ifndef _DNN_CONSTANTS_H
#define _DNN_CONSTANTS_H

#define ROUNDK(val,K)  ((((val)+K-1)/(K))*(K))
#define DIVUPK(val,K) (((val)+K-1)/(K))

enum  DNN_DATA_MODE
{
    DNN_DATAMODE_SP_TRAIN,        // for supervised training
	DNN_DATAMODE_TEST,
	DNN_DATAMODE_PREDICT,
	DNN_DATAMODE_US_TRAIN,        // for unsupervised training
	DNN_DATAMODE_ERROR
};

#endif

