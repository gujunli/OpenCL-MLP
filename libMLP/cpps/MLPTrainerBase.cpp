/*
 *  COPYRIGHT:  Copyright (c) 2013 Advanced Micro Devices, Inc.  All rights reserved
 *
 *   Written by  Qianfeng Zhang@amd.com ( March 2014)
 *
 *   Written by  Junli Gu@amd.com ( Dec 2013 )
 */

#include "MLPUtil.h"
#include "MLPTrainerBase.h"
#include "MLPChkPointState.h"

const char MLP_version[] = "MLP version 1.6.0 developed by AMD China DNN Team";

MLPTrainerBase::MLPTrainerBase()
{
	this->dimensions = NULL;
	this->nLayers = 0;
	this->minibatch = 0;

	this->etas = NULL;
	this->actFuncs = NULL;

	this->currBatchNo = 0;

	DNN_LOCK_INIT(&this->chkPointingLock);

	this->dataProviderp = NULL;

	this->initialized = false;
};


MLPTrainerBase::~MLPTrainerBase()
{
    this->_dispose();
};

// only called by the constructor and setupMLP()
void MLPTrainerBase::_initialize(MLPConfigProvider & provider, int _minibatch)
{
	this->netType = provider.netType;
	this->nLayers = provider.nLayers;
	this->minibatch = _minibatch;
	this->dimensions = new int[this->nLayers];

	this->etas = new float[this->nLayers];            // learning rate for each layer
	this->actFuncs = new ACT_FUNC[this->nLayers];     // activating function for each layer

	for ( int i = 0; i < this->nLayers; i++ )
		this->dimensions[i] = provider.dimensions[i];

	for ( int i = 0; i < this->nLayers; i++ )
		this->etas[i] = provider.etas[i];

	for ( int i = 0; i < this->nLayers; i++ )
		this->actFuncs[i] = provider.actFuncs[i];

	this->costFunc = provider.costFunc;
	this->momentum = provider.momentum;
	this->epochs = provider.epochs; 
}

// only called by the destructor
void MLPTrainerBase::_dispose()
{
	if ( this->dimensions )
		delete [] this->dimensions;
	if ( this->etas )
		delete [] this->etas;
	if ( this->actFuncs )
		delete [] this->actFuncs;
}


DNNDataProvider *MLPTrainerBase::getDataProvider()
{
	return(this->dataProviderp);
};

void MLPTrainerBase::saveNetConfig(const char *configPath)
{
	MLPConfigProvider  netProvider(this->nLayers,this->dimensions,false);

	this->synchronizeNetConfig(netProvider);

	netProvider.saveConfig(configPath, MLP_CP_TRAINING_CONF_NEW, MLP_CP_NNET_DATA_NEW);
};


void MLPTrainerBase::showNetConfig()
{
	MLPConfigProvider  netProvider(this->nLayers,this->dimensions,false);

	this->synchronizeNetConfig(netProvider);

	netProvider.showConfig();
};


void MLPTrainerBase::checkPointing(struct MLPCheckPointState &cpState)
{

	 DNN_LOCK(&this->chkPointingLock);          // need be lock protected from the Training of MLPTrainer

     // Snapshot one value of BatchNo as the state checkpointed from the MLPTrainer
     cpState.cpBatchNo = (unsigned int) this->currBatchNo;
	 cpState.cpEpoch = (unsigned int) this->currEpoch;

     // Snapshot one value of FrameNo as the state checkpointed from the MLPDataProvider
	 int frameNo;
     this->dataProviderp->getCheckPointFrame(frameNo);
	 cpState.cpFrameNo = (unsigned int) frameNo;

     // Snapshot one state of network configuration from the MLPTrainer, and save it to the files
     MLPConfigProvider  netProvider(this->nLayers,this->dimensions,false);
     this->synchronizeNetConfig(netProvider);
     netProvider.saveConfig(cpState.netConfPath, cpState.ncTrainingConfigFname, cpState.ncNNetDataFname);

     DNN_UNLOCK(&this->chkPointingLock);
}

int MLPTrainerBase::batchTraining(int maxBatches)
{
	return(this->batchTrainingWithCheckPointing(maxBatches, 0, 0, NULL));
};

int MLPTrainerBase::getEpochs()
{
	return(this->epochs); 
}; 