/*
 *  COPYRIGHT:  Copyright (c) 2014 Advanced Micro Devices, Inc.  All rights reserved
 *
 *   Written by Qianfeng Zhang@amd.com ( March 2014 )
 *
 */

#include <algorithm>

#include "DNNUtil.h"
#include "DNNSimpleDataProvider.h"

//////////////////////////////////////////////////////////////////////////////////////
////                          constructors and destructor                         ////
//////////////////////////////////////////////////////////////////////////////////////

DNNSimpleDataProvider::DNNSimpleDataProvider()
{
	this->dataMode = DNN_DATAMODE_SP_TRAIN;
	this->haveLabel = ( (this->dataMode == DNN_DATAMODE_SP_TRAIN) || (this->dataMode == DNN_DATAMODE_TEST) )? true:false;
	this->haveLabel = true;
	this->m_dataFeatureSize = 429;
	this->m_dataLabelSize = 8991;

	this->m_batchSize = 1024;
	this->m_shuffleBatches = 50;

	this->total_batches =this->m_shuffleBatches*10;
};

DNNSimpleDataProvider::DNNSimpleDataProvider(DNN_DATA_MODE mode, int dataFeatureSize, int dataLabelSize, int batchSize, int shuffleBatches)
{
	if ( (mode < 0) || (mode >= DNN_DATAMODE_ERROR) ) {
		  dnn_log("DNNSimpleDataProvider", "Data mode for constructing DNNSimpleDataProvider is not correct");
		  DNN_Exception("");
	};

	this->dataMode = mode;
	this->haveLabel = ( (this->dataMode == DNN_DATAMODE_SP_TRAIN) || (this->dataMode == DNN_DATAMODE_TEST) )? true:false;
	this->m_dataFeatureSize = dataFeatureSize;
	this->m_dataLabelSize = dataLabelSize;

	this->m_batchSize = batchSize;
	this->m_shuffleBatches = shuffleBatches;

	this->total_batches = this->m_shuffleBatches*10;
};

DNNSimpleDataProvider::~DNNSimpleDataProvider()
{
    DNN_CHECK(this->shutdown_worker());

	this->release_io_buffers();
	this->release_transfer_buffers();
};

void DNNSimpleDataProvider::setup_first_data_batches()
{
	this->stageBatchNo = 0;
	this->setup_cont_data_batches();

    if ( this->batches_loaded ) {
	     this->shuffle_data(this->permutations, this->m_batchSize * this->batches_loaded );
    };
};

void DNNSimpleDataProvider::setup_cont_data_batches()
{
	struct dnn_tv tv;

	getCurrentTime(&tv);
	srand(tv.tv_usec); // use current time as random seed

	for (int k=0; k < this->m_batchSize * this->m_shuffleBatches; k++ ) {
		int pos;
		float max_val;

		for (int i=0; i < this->m_dataFeatureSize; i++)
		    this->featureData[k*this->m_dataFeatureSize+i] = ((float)rand()/((float)RAND_MAX+1.0f)-0.5f)*14.0f;

		pos = 0;
		max_val = -14.0f;

		for (int i=0; i < this->m_dataFeatureSize; i++)
			 if ( max_val < this->featureData[k*this->m_dataFeatureSize+i] ) {
				  max_val =  this->featureData[k*this->m_dataFeatureSize+i];
				  pos = i;
			 };

	    if ( this->haveLabel ){
             for (int i=0; i < this->m_dataLabelSize; i++)
				 this->labelData[k*this->m_dataLabelSize+i] = 0.0f;
			 //this->labelData[k*this->m_dataLabelSize + rand() % this->m_dataLabelSize] = 1.0f;
             this->labelData[k*this->m_dataLabelSize + pos] = 1.0f;   // use the position of the maximum value in the input vector as the label of the vector
		 };
	};

	this->batches_loaded = this->m_shuffleBatches;
	this->batchNo += this->batches_loaded;

	if ( this->batchNo == this->total_batches )
		 this->endOfDataSource = true;
};


//////////////////////////////////////////////////////////////////////////////////////
////                          public member functions                             ////
//////////////////////////////////////////////////////////////////////////////////////

void DNNSimpleDataProvider::setupBackendDataProvider()
{
	this->batchNo = 0;

	this->setup_first_data_batches();
};

void DNNSimpleDataProvider::resetBackendDataProvider()
{
	this->batchNo = 0;

	this->setup_first_data_batches();
};


// if the output for the frame matches its label, return true to indicate a successful mapping of this
// frame by the neural network.  This interface will be called by the DNNTester class when calculating
// the success ratio of the neural network on this type of data
bool DNNSimpleDataProvider::frameMatching(const float *frameOutput, const float *frameLabel, int len)
{
	float element;

	for (int i=0; i< len; i++) {
		 element = (frameOutput[i]<0.5)?0.0f:1.0f;

		 if ( element != frameLabel[i] )
			  return(false);
	};

	return(true);
};


