/*
 *  COPYRIGHT:  Copyright (c) 2013 Advanced Micro Devices, Inc.  All rights reserved
 *
 *   Written by  Qianfeng Zhang@amd.com ( March 2014 )
 *
 *   Written by  Junli Gu @amd.com ( Dec 2013 )
 */


#ifndef _MLP_TRAINER_BASE_H_
#define _MLP_TRAINER_BASE_H_

#ifdef _WIN32
#include <Windows.h>
#else
#include <pthread.h>
#endif


#include "DNNApiExport.h"
#include "DNNConstants.h"
#include "DNNDataProvider.h"
#include "MLPConfigProvider.h"
#include "MLPChkPointState.h"

// Implement the interfaces for training the MLP network
class MLPTrainerBase
{
protected:
	bool initialized;

	MLP_NETTYPE netType;
	int   nLayers;               // Number of layers
	int   minibatch;             // Size of minibatch -- number of input frames received and processed by the MLP network in batches
	int  *dimensions;            // Dimensions of all layers, first layer is the input and last layer is the output
    float *etas;                 // Learning rate of all layers
	float momentum;              // Used to indicate the importance of historic variance of weights
	ACT_FUNC *actFuncs;          // Activation function used by all layers,  usually all hidden layers use same activation function, the output uses different one
	COST_FUNC costFunc;          // Cost function used to measure the error value of the input batch got on the current MLP network
	int epochs; 

	DNNDataProvider *dataProviderp;

	int currBatchNo;             // Indicate the current batchNo the training is on, need be saved when doing checkpointing
	int currEpoch;               // Indicate the current epoch the training is on, need be saved when doing checkpointing

#ifdef WIN32                       // for Windows
	CRITICAL_SECTION chkPointingLock;
#else                              // for Linux
	pthread_mutex_t chkPointingLock;
#endif

protected:
	void _initialize(MLPConfigProvider & NetProvider, int minibatch);

private:
	void _dispose();

public:
	LIBDNNAPI MLPTrainerBase();
	LIBDNNAPI virtual ~MLPTrainerBase()=0;

public:
	LIBDNNAPI virtual void setupMLP(MLPConfigProvider & configProvider, DNNDataProvider & dataProvider, int minipatch)=0;

	LIBDNNAPI virtual void synchronizeNetConfig(MLPConfigProvider &configProvider)=0;

	LIBDNNAPI virtual int batchTrainingWithCheckPointing(int maxBatches, int startBatch, int startEpoch, bool doChkPointing)=0;

	LIBDNNAPI int batchTraining(int maxBatches);

	LIBDNNAPI void saveNetConfig(const char *configPath);
	LIBDNNAPI void showNetConfig();

	LIBDNNAPI DNNDataProvider *getDataProvider();

	LIBDNNAPI int getEpochs(); 

	void checkPointing(struct MLPCheckPointState &state);
};


#endif // __MPL_TRAINER_BASE_H_
