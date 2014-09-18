/*
 *  COPYRIGHT:  Copyright (c) 2013 Advanced Micro Devices, Inc.  All rights reserved
 *
 *  Written by  Qianfeng Zhang@amd.com ( March 2014 )
 *
 *  Written by  Junli Gu@amd.com ( Dec 2013 )
 */


#include <algorithm>
#include <clAmdBlas.h>

#include "MLPUtil.h"
#include "MLPOclCommon.h"
#include "MLPPredictorBase.h"


MLPPredictorBase::MLPPredictorBase()
{
	this->dimensions = NULL;
	this->nLayers = 0;
	this->batchSize = 0;

	this->actFuncs = NULL;

	this->initialized = false;
}

MLPPredictorBase::~MLPPredictorBase()
{
    this->_dispose();
};

// only called by the constructor and setupMLP()
void MLPPredictorBase::_initialize(MLPNetProvider & provider, int _batchSize)
{
	this->nLayers = provider.nLayers;
	this->batchSize = _batchSize;
	this->dimensions = new int[this->nLayers];
	this->actFuncs = new ACT_FUNC[this->nLayers];     // activating function for each layer

	for ( int i = 0; i < this->nLayers; i++ )
		this->dimensions[i] = provider.dimensions[i];

	for ( int i = 0; i < this->nLayers; i++ )
		this->actFuncs[i] = provider.actFuncs[i];
}

// only called by the destructor
void MLPPredictorBase::_dispose()
{
	if ( this->dimensions )
		delete [] this->dimensions;
	if ( this->actFuncs )
		delete [] this->actFuncs;
}

int MLPPredictorBase::getInputVectorSize()
{
	return(this->dimensions[0]);
};

int MLPPredictorBase::getOutputVectorSize()
{
	return(this->dimensions[this->nLayers-1]);
};

int MLPPredictorBase::getBatchSize()
{
	return(this->batchSize);
};

