/*
 *  COPYRIGHT:  Copyright (c) 2013 Advanced Micro Devices, Inc.  All rights reserved
 *
 *  Changed by  Qianfeng Zhang@amd.com ( March 2014 )
 *
 *  Written by  Junli Gu@amd.com ( Dec 2013 )
 */


#include "MLPTesterBase.h"


MLPTesterBase::MLPTesterBase()
{
	this->dimensions = NULL;
	this->nLayers = 0;
	this->batchSize = 0;

	this->actFuncs = NULL;

	this->dataProviderp = NULL;
	this->initialized = false;
}

MLPTesterBase::~MLPTesterBase()
{
    this->_dispose();
};

void MLPTesterBase::_initialize(MLPNetProvider & provider, int _batchSize)
{
    this->netType = provider.netType;
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
void MLPTesterBase::_dispose()
{
	if ( this->dimensions )
		delete [] this->dimensions;
	if ( this->actFuncs )
		delete [] this->actFuncs;
}


DNNDataProvider *MLPTesterBase::getDataProvider()
{
	return(this->dataProviderp);
};

void MLPTesterBase::getTestingStats(int &totalFrames, int &succFrames)
{
	totalFrames = this->totalTestFrames;
	succFrames = this->succTestFrames;
};

int MLPTesterBase::getInputVectorSize()
{
	return(this->dimensions[0]);
};

int MLPTesterBase::getOutputVectorSize()
{
	return(this->dimensions[this->nLayers-1]);
};

int MLPTesterBase::getBatchSize()
{
	return(this->batchSize);
};

