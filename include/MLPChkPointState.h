/*
 *  COPYRIGHT:  Copyright (c) 2014 Advanced Micro Devices, Inc.  All rights reserved
 *
 *   Written by Qianfeng Zhang@amd.com ( March 2014 )
 *
 */

#ifndef _MLP_CHKPOINT_STATE_H_
#define _MLP_CHKPOINT_STATE_H_

struct MLPCheckPointState {
	unsigned int chkPointID;                   // a number increased sequently when new checkpoint is produced 

	// for the MLPNetProvider
	char netConfPath[256];
	char netConfArchFileName[32];
	char netConfDataFileName[32]; 

	// for the MLPTrainer
	unsigned int cpBatchNo;                   // the batch recorded by the MLPTrainer at which it is expected to get data at

	// for the MLPDataProvider
	unsigned int cpFrameNo;                   // the frame of the data provider that it should start providing data from  
}; 

#endif