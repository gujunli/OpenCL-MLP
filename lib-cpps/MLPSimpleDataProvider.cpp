/*
 *  COPYRIGHT:  Copyright (c) 2014 Advanced Micro Devices, Inc.  All rights reserved
 *
 *   Written by Qianfeng Zhang@amd.com ( March 2014 )
 *
 */

#include <algorithm>

#include "MLPUtil.h"
#include "MLPSimpleDataProvider.h"

//////////////////////////////////////////////////////////////////////////////////////
////                          constructors and destructor                         ////
//////////////////////////////////////////////////////////////////////////////////////

MLPSimpleDataProvider::MLPSimpleDataProvider()
{
	this->dataMode = MLP_DATAMODE_TRAIN; 
	this->haveLabel = true; 
	this->m_dataFeatureSize = 429; 
	this->m_dataLabelSize = 8991; 

	this->m_batchSize = 1024;
	this->m_shuffleBatches = 50; 

	this->batchNo = 0; 
	this->rounds = 4; 

	this->total_batches = this->m_batchSize * this->m_shuffleBatches * this->rounds; 

    this->setup_data_source(); 
}; 

MLPSimpleDataProvider::MLPSimpleDataProvider(MLP_DATA_MODE mode, int dataFeatureSize, int dataLabelSize, int batchSize, int shuffleBatches)
{
	if ( (mode < 0) || (mode >= MLP_DATAMODE_ERROR) ) {
		  mlp_log("MLPSimpleDataProvider", "Data mode for constructing MLPSimpleDataProvider is not correct"); 
		  MLP_Exception(""); 
	}; 

	this->dataMode = mode; 
	this->haveLabel = (mode==MLP_DATAMODE_PREDICT)?false:true; 
	this->m_dataFeatureSize = dataFeatureSize; 
	this->m_dataLabelSize = dataLabelSize; 

	this->m_batchSize = batchSize; 
	this->m_shuffleBatches = shuffleBatches; 

	this->batchNo = 0; 
	this->rounds = 4; 

	this->total_batches = this->m_shuffleBatches * this->rounds;

	this->setup_data_source(); 
}; 

MLPSimpleDataProvider::~MLPSimpleDataProvider()
{
	this->shutdown_worker(); 

	delete [] this->permutations; 
	delete [] this->featureData; 

	if ( this->haveLabel )
		delete [] this->labelData;

	this->release_buffers(); 
};


//////////////////////////////////////////////////////////////////////////////////////
////                          private member functions                            ////
//////////////////////////////////////////////////////////////////////////////////////


// fetch a batch of data from the source and get it ready on the data buffer for reading
void MLPSimpleDataProvider::prepare_batch_data()
{
	this->load_feature_batch(this->featureData,this->permutations,this->m_batchSize*(this->batchNo % this->m_shuffleBatches)); 
	this->load_label_batch(this->labelData,this->permutations,this->m_batchSize*(this->batchNo % this->m_shuffleBatches));

	this->batchNo++; 

	if ( (this->batchNo % this->m_shuffleBatches) == 0 ) {
		  this->shuffle_data(this->permutations, this->m_batchSize * this->m_shuffleBatches ); 
	}; 
}; 

bool MLPSimpleDataProvider::haveBatchToProvide()
{
	if ( this->batchNo == this->total_batches ) 
		 return(false); 
	else 
	     return(true); 
}; 


// set up the data source of MLPSimpleDataProvider
void MLPSimpleDataProvider::setup_data_source()
{
	this->permutations = new int[this->m_batchSize * this->m_shuffleBatches]; 
	this->featureData  = new float[this->m_batchSize * this->m_shuffleBatches * this->m_dataFeatureSize]; 
	if ( this->haveLabel )
		 this->labelData = new float[this->m_batchSize * this->m_shuffleBatches * this->m_dataLabelSize]; 

	// initial permutations, permutated each epoch
	for (int k=0; k < this->m_batchSize * this->m_shuffleBatches; k++) 
		    this->permutations[k] = k; 

	struct mlp_tv tv; 

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
}; 

void MLPSimpleDataProvider::shuffle_data(int *index, int len)
{	
	std::random_shuffle(index, index+len);
};

//////////////////////////////////////////////////////////////////////////////////////
////                          public member functions                             ////
//////////////////////////////////////////////////////////////////////////////////////

void MLPSimpleDataProvider::setupDataProvider()
{
	this->create_buffers(this->m_batchSize); 

	this->initialized = true; 

	this->startup_worker(); 
};

void MLPSimpleDataProvider::resetDataProvider()
{
	if ( !this->initialized ) {
		 mlp_log("MLPSimpleDataProvider", "The DataProvider is still not started yet, no reset should be called"); 
		 MLP_Exception("");
	}; 
	MLP_CHECK(this->shutdown_worker()); 
	
	this->batchNo = 0; 

	// initial permutations, permutated each epoch
	for (int k=0; k < this->m_batchSize * this->m_shuffleBatches; k++) 
		    this->permutations[k] = k; 	

	MLP_CHECK(this->startup_worker()); 
}; 

bool MLPSimpleDataProvider::endofInputBatches()
{
	if ( this->batchNo < this->total_batches ) 
		 return(false); 
	else 
		 return(true); 
}; 

// if the output for the frame matches its label, return true to indicate a successful mapping of this
// frame by the neural network.  This interface will be called by the MLPTester class when calculating 
// the success ratio of the neural network on this type of data 
bool MLPSimpleDataProvider::frameMatching(float *frameOutput, float *frameLabel, int len)
{
	float element; 

	for (int i=0; i< len; i++) {
		 element = (frameOutput[i]<0.5)?0.0f:1.0f; 
		
		 if ( element != frameLabel[i] )
			  return(false); 
	}; 

	return(true); 
};

 




