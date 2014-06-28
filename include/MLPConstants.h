/*
 *  COPYRIGHT:  Copyright (c) 2014 Advanced Micro Devices, Inc.  All rights reserved
 *
 *   Written by Qianfeng Zhang@amd.com ( March 2014 )
 *
 */

#ifndef _MLP_CONSTANTS_H
#define _MLP_CONSTANTS_H

#define ROUNDK(val,K)  ((((val)+K-1)/(K))*(K))
#define DIVUPK(val,K) (((val)+K-1)/(K))

enum  MLP_DATA_MODE 
{
    MLP_DATAMODE_TRAIN,
	MLP_DATAMODE_TEST,
	MLP_DATAMODE_PREDICT,
	MLP_DATAMODE_ERROR
}; 

#endif

