/*
 *  COPYRIGHT:  Copyright (c) 2014 Advanced Micro Devices, Inc.  All rights reserved
 *
 *   Written by Qianfeng Zhang@amd.com ( March 2014 )
 *
 */

#ifndef _MLP_CHKPOINT_STATE_H_
#define _MLP_CHKPOINT_STATE_H_

struct MLPCheckPointState {
	unsigned int chkPointID;           // a number increased sequently when new checkpoint is produced

	// for the MLPNetProvider
	char netConfPath[256];
	char ncTrainingConfigFname[32];
	char ncNNetDataFname[32];

	// for the MLPTrainer
	unsigned int cpBatchNo;            // the batch recorded by the MLPTrainer at which it is expected to get data
	unsigned int cpEpoch;              // the epoch recorded by the MLPTrainer, which indicates which round the training at

	// for the MLPDataProvider
	unsigned int cpFrameNo;            // the frame of the data provider that we should start providing data from 
                                           // when recovering from the checkpoint state
};

#endif
