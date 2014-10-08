/*
 *  COPYRIGHT:  Copyright (c) 2014 Advanced Micro Devices, Inc.  All rights reserved
 *
 *   Written by Qianfeng Zhang@amd.com ( March 2014 )
 *
 */

#include <algorithm>
#include <fstream> 

#include "DNNUtil.h"
#include "DNNDataProvider.h"
#include "conv_endian.h"
#include "stats_info.h"

using namespace std; 

DNNDataProvider::DNNDataProvider()
{
	this->haveLabel = false;
	this->m_dataFeatureSize = 0;
	this->m_dataLabelSize = 0;
	this->m_batchSize = 0;

	for (int i=0; i< DNN_BATCH_RING_SIZE; i++) {
	     this->features[i] = NULL;
	     this->labels[i] = NULL;
	};

	this->use_stats = false; 
	this->meanvalues = NULL; 
	this->stddevs = NULL; 

	this->running = false;

	this->initialized = false;
};

DNNDataProvider::~DNNDataProvider()
{
	if ( this->meanvalues ) 
		 delete [] this->meanvalues; 
	if ( this->stddevs )
		 delete [] this->stddevs;  
};

// create the data buffers and initialize the locks and condition variable
void DNNDataProvider::create_transfer_buffers(int batchSize)
{

	for (int i=0; i< DNN_BATCH_RING_SIZE; i++) {

        this->features[i] = new float[batchSize*this->m_dataFeatureSize*sizeof(float)];

	    if ( this->haveLabel )
		     this->labels[i] = new float[batchSize*this->m_dataLabelSize*sizeof(float)];
	    else
		     this->labels[i] = NULL;
	};

	this->m_batchSize = batchSize;

	this->rbuf_count = 0;
	this->wbuf_count = DNN_BATCH_RING_SIZE;

#ifdef _WIN32      // for Windows
	InitializeConditionVariable(&this->readReady);
	InitializeConditionVariable(&this->writeReady);
#else             // for Linux
	pthread_cond_init(&this->readReady, NULL);
	pthread_cond_init(&this->writeReady, NULL);
#endif

	DNN_LOCK_INIT(&this->bufferLock);
};

// release the data buffers
void DNNDataProvider::release_transfer_buffers()
{

	for (int i=0; i< DNN_BATCH_RING_SIZE; i++) {
	     delete [] this->features[i];

	     if ( this->haveLabel )
		      delete [] this->labels[i];
	};
};

void DNNDataProvider::reset_transfer_buffers()
{
	this->rbuf_count = 0;
	this->wbuf_count = DNN_BATCH_RING_SIZE;

#ifdef _WIN32      // for Windows
	InitializeConditionVariable(&this->readReady);
	InitializeConditionVariable(&this->writeReady);
#else             // for Linux
	pthread_cond_init(&this->readReady, NULL);
	pthread_cond_init(&this->writeReady, NULL);
#endif

	DNN_LOCK_INIT(&this->bufferLock);
};

void DNNDataProvider::create_io_buffers()
{
    // allocate batches IO buffers used by the backend data provider
	this->permutations = new int[this->m_batchSize * this->m_shuffleBatches];
	this->featureData  = new float[this->m_batchSize * this->m_shuffleBatches * this->m_dataFeatureSize];
	if ( this->haveLabel )
         this->labelData = new float[this->m_batchSize * this->m_shuffleBatches * this->m_dataLabelSize];

    for (int k=0; k < this->m_batchSize * this->m_shuffleBatches; k++)
	     this->permutations[k] = k;

	this->batches_loaded  = 0;
};

void DNNDataProvider::release_io_buffers()
{
	delete [] this->featureData;
	if ( this->haveLabel )
		delete [] this->labelData;
	delete [] this->permutations;
};

void DNNDataProvider::reset_io_buffers()
{
	this->batches_loaded = 0;

    for (int k=0; k < this->m_batchSize * this->m_shuffleBatches; k++)
	     this->permutations[k] = k;
};


// create and start the worker thread
int DNNDataProvider::startup_worker()
{
	if ( !this->initialized )
		 return(-1);

	if ( this->running )
		 return(-2);

	this->running = true;
	DNN_CREATE_THREAD(&this->worker,DNNDataProvider::worker_fun,(void*)this);

	return(0);
};

// shutdown the worker thread
int DNNDataProvider::shutdown_worker()
{
    DNN_KILL_THREAD(this->worker);
	DNN_JOIN_THREAD(this->worker);
	this->running = false;

	return(0);
};

int DNNDataProvider::getTotalBatches()
{
	return(this->total_batches);
};

int DNNDataProvider::getFeatureSize()
{
	return(this->m_dataFeatureSize);
};

int DNNDataProvider::getLabelSize()
{
	return(this->m_dataLabelSize);
};



#ifdef _WIN32             // for Windows

// get the pointer of the new batch of data
int DNNDataProvider::getBatchData(int batchSize, float * & pFeatures, bool blocking)
{
	if ( !this->running )
		 return(-1);

	if ( batchSize != this->m_batchSize )
		 return(-2);

	while (1) {
	      EnterCriticalSection(&this->bufferLock);

          if ( this->rbuf_count > 0 ) {
		       pFeatures = this->features[this->rbuf_index];
	           LeaveCriticalSection(&this->bufferLock);
			   return(0);
	      }
	      else {
			   if ( blocking ) {
                    SleepConditionVariableCS(&this->readReady, &this->bufferLock, INFINITE);
			        LeaveCriticalSection(&this->bufferLock);
			   }
			   else {
	                LeaveCriticalSection(&this->bufferLock);
					return(-3);
			   };
		  };
	};

	return(0);
};

// get the pointer of the new batch of data
int DNNDataProvider::getBatchData(int batchSize, float * & pFeatures, float * & pLabels, bool blocking)
{
	if ( !this->running )
		 return(-1);

	if ( batchSize != this->m_batchSize )
		 return(-2);

	while (1) {
	      EnterCriticalSection(&this->bufferLock);

          if ( this->rbuf_count > 0 ) {
		       pFeatures = this->features[this->rbuf_index];
			   if (this->haveLabel)
			       pLabels = this->labels[this->rbuf_index];
	           LeaveCriticalSection(&this->bufferLock);
			   return(0);
	      }
	      else {
			   if ( blocking ) {
                    SleepConditionVariableCS(&this->readReady, &this->bufferLock, INFINITE);
	                LeaveCriticalSection(&this->bufferLock);
			   }
			   else {
	                LeaveCriticalSection(&this->bufferLock);
					return(-3);
			   };
		  };
	};

	return(0);
};

// tell the worker thread that the using of the current batch of data is finished
int DNNDataProvider::nextBatch()
{
	 EnterCriticalSection(&this->bufferLock);

	 this->rbuf_count--;
	 this->wbuf_count++;
	 this->rbuf_index = (this->rbuf_index+1) % DNN_BATCH_RING_SIZE;

	 WakeConditionVariable(&this->writeReady);

	 LeaveCriticalSection(&this->bufferLock);

	 return(0);
};

// the worker thread which read batches of data asynchronously
void *DNNDataProvider::worker_fun(void *argp)
{
	 DNNDataProvider *objp;

	 objp = (DNNDataProvider *)argp;
	 objp->rbuf_index = 0;
	 objp->wbuf_index = 0;

	 objp->prepare_batch_data_top_half();    // first time to prepare the data onto the buffer

	 EnterCriticalSection(&objp->bufferLock);

     objp->wbuf_index = (objp->wbuf_index+1) % DNN_BATCH_RING_SIZE;

	 objp->rbuf_count = 1;
	 objp->wbuf_count = DNN_BATCH_RING_SIZE-1;

	 WakeConditionVariable(&objp->readReady);

	 LeaveCriticalSection(&objp->bufferLock);

     objp->prepare_batch_data_bottom_half();

	 while ( objp->running ) {
            EnterCriticalSection(&objp->bufferLock);

		    if ( objp->wbuf_count > 0 ) {

                 LeaveCriticalSection(&objp->bufferLock);

				 if ( ! objp->haveBatchToProvide() )
						break;

				 objp->prepare_batch_data_top_half();   // load a batch of data

	             EnterCriticalSection(&objp->bufferLock);

                 objp->wbuf_index = (objp->wbuf_index+1) % DNN_BATCH_RING_SIZE;
				 objp->wbuf_count--;
				 objp->rbuf_count++;

			     WakeConditionVariable(&objp->readReady);

                 LeaveCriticalSection(&objp->bufferLock);

				 objp->prepare_batch_data_bottom_half();   // load a batch of data
			}
			else {
			     SleepConditionVariableCS(&objp->writeReady, &objp->bufferLock, INFINITE);
                 LeaveCriticalSection(&objp->bufferLock);
			};
	 };

	 return(0);
};

#else            // for linux

// get the pointer of the new batch of data
int DNNDataProvider::getBatchData(int batchSize, float * & pFeatures, bool blocking)
{
	if ( !this->running )
		 return(-1);

	if ( batchSize != this->m_batchSize )
		 return(-2);

	while (1) {
	      pthread_mutex_lock(&this->bufferLock);

          if ( this->rbuf_count > 0 ) {
		       pFeatures = this->features[this->rbuf_index];
	           pthread_mutex_unlock(&this->bufferLock);
			   return(0);
	      }
	      else {
			   if ( blocking ) {
                    pthread_cond_wait(&this->readReady, &this->bufferLock);
			        pthread_mutex_unlock(&this->bufferLock);
			   }
			   else {
	                pthread_mutex_unlock(&this->bufferLock);
					return(-3);
			   };
		  };
	};

	return(0);
};

// get the pointer of the new batch of data
int DNNDataProvider::getBatchData(int batchSize, float * & pFeatures, float * & pLabels, bool blocking)
{
	if ( !this->running )
		 return(-1);

	if ( batchSize != this->m_batchSize )
		 return(-2);

	while (1) {
	      pthread_mutex_lock(&this->bufferLock);

          if ( this->rbuf_count > 0 ) {
		       pFeatures = this->features[this->rbuf_index];
			   pLabels = this->labels[this->rbuf_index];
	           pthread_mutex_unlock(&this->bufferLock);
			   return(0);
	      }
	      else {
			   if ( blocking ) {
                    pthread_cond_wait(&this->readReady, &this->bufferLock);
	                pthread_mutex_unlock(&this->bufferLock);
			   }
			   else {
	                pthread_mutex_unlock(&this->bufferLock);
					return(-3);
			   };
		  };
	};

	return(0);
};

// tell the worker thread that the using of the current batch of data is finished
int DNNDataProvider::nextBatch()
{
	 pthread_mutex_lock(&this->bufferLock);

	 this->rbuf_count--;
	 this->wbuf_count++;
	 this->rbuf_index = (this->rbuf_index+1) % DNN_BATCH_RING_SIZE;

	 pthread_cond_signal(&this->writeReady);

	 pthread_mutex_unlock(&this->bufferLock);

	 return(0);
};


// the worker thread which read batches of data asynchronously
void *DNNDataProvider::worker_fun(void *argp)
{
	 DNNDataProvider *objp;

	 objp = (DNNDataProvider *)argp;
	 objp->rbuf_index = 0;
	 objp->wbuf_index = 0;

	 objp->prepare_batch_data_top_half();    // first time to prepare the data onto the buffer

	 pthread_mutex_lock(&objp->bufferLock);

     objp->wbuf_index = (objp->wbuf_index+1) % DNN_BATCH_RING_SIZE;

	 objp->rbuf_count = 1;
	 objp->wbuf_count = DNN_BATCH_RING_SIZE-1;

	 pthread_cond_signal(&objp->readReady);

	 pthread_mutex_unlock(&objp->bufferLock);

	 objp->prepare_batch_data_bottom_half();

	 while ( objp->running ) {
            pthread_mutex_lock(&objp->bufferLock);

		    if ( objp->wbuf_count > 0 ) {

                 pthread_mutex_unlock(&objp->bufferLock);

				 if ( ! objp->haveBatchToProvide() )
						break;

				 objp->prepare_batch_data_top_half();   // load a batch of data

	             pthread_mutex_lock(&objp->bufferLock);

                 objp->wbuf_index = (objp->wbuf_index+1) % DNN_BATCH_RING_SIZE;
				 objp->wbuf_count--;
				 objp->rbuf_count++;

			     pthread_cond_signal(&objp->readReady);

                 pthread_mutex_unlock(&objp->bufferLock);

				 objp->prepare_batch_data_bottom_half();
			}
			else {
			     pthread_cond_wait(&objp->writeReady, &objp->bufferLock);
                 pthread_mutex_unlock(&objp->bufferLock);
			};
	 };

	 return(0);
};

#endif


// load one batch of features data from source to data buffer
void DNNDataProvider::load_feature_batch(float *srcp, int *indexBase, int indexOffset)
{
     for(int i = 0; i < this->m_batchSize; i++)
		for(int j = 0; j < this->m_dataFeatureSize; j++)
			this->features[this->wbuf_index][i*this->m_dataFeatureSize+j] = srcp[indexBase[indexOffset+i]*this->m_dataFeatureSize+j];
};

// load one batch of labels data from source to data buffer
void DNNDataProvider::load_label_batch(float *srcp, int *indexBase, int indexOffset)
{
     for(int i = 0; i < this->m_batchSize; i++)
		for(int j = 0; j < this->m_dataLabelSize; j++)
			this->labels[this->wbuf_index][i*this->m_dataLabelSize+j] = srcp[indexBase[indexOffset+i]*this->m_dataLabelSize+j];
};


void DNNDataProvider::prepare_batch_data_top_half()
{
	// get one batch from the io buffer to the transfer buffer
	this->load_feature_batch(this->featureData,this->permutations,this->m_batchSize*(this->stageBatchNo % this->m_shuffleBatches));
	if ( this->haveLabel )
	     this->load_label_batch(this->labelData,this->permutations,this->m_batchSize*(this->stageBatchNo % this->m_shuffleBatches));
};

void DNNDataProvider::prepare_batch_data_bottom_half()
{
	if ( this->supportChkPointing)
	     DNN_LOCK(&this->chkPointingLock);

	this->stageBatchNo++;

	if ( this->stageBatchNo == this->batches_loaded ) {

		  this->batches_loaded = 0;

		  if ( !this->endOfDataSource )
		        this->setup_cont_data_batches();

		  if ( this->batches_loaded ) {
			  	// initial permutations, permutated each round
	            for (int k=0; k < this->m_batchSize * this->m_shuffleBatches; k++)
		             this->permutations[k] = k;

		        this->shuffle_data(this->permutations, this->m_batchSize * this->batches_loaded );
		        // cout << "Load new data from files and shuffling the frame sequences" << endl;
		  };

		  this->stageBatchNo = 0;
	};

	if ( this->supportChkPointing )
		 DNN_UNLOCK(&this->chkPointingLock);
};

void DNNDataProvider::shuffle_data(int *index, int len)
{
	std::random_shuffle(index, index+len);
};


bool DNNDataProvider::haveBatchToProvide()
{
	if ( !this->endOfDataSource )
		 return(true);

    if ( this->batches_loaded )
		 return(true);

	return(false);
};


bool DNNDataProvider::batchAvailable()
{
	if ( this->haveBatchToProvide() )
		 return(true);

	if ( this->rbuf_count > 0 )
		 return(true);

	return(false);
};

void DNNDataProvider::setupDataProvider()
{
	this->supportChkPointing = false;
	this->create_transfer_buffers(this->m_batchSize);
	this->create_io_buffers();

	this->endOfDataSource = false;

    this->setupBackendDataProvider();   // call the back-end setup

	this->initialized = true;

	DNN_CHECK(this->startup_worker());
};

void DNNDataProvider::setupDataProvider(int startFrameNo, bool doChkPointing)
{
 	if ( (this->dataMode != DNN_DATAMODE_SP_TRAIN) && (this->dataMode != DNN_DATAMODE_US_TRAIN) ) {
		 dnn_log("DNNDataProvider", "This interface can only be called with the TRAIN mode");
		 DNN_Exception("");
	};

	// For CheckPoint
	this->supportChkPointing = doChkPointing;
	if ( this->supportChkPointing )
		 DNN_LOCK_INIT(&this->chkPointingLock);

	this->create_transfer_buffers(this->m_batchSize);
	this->create_io_buffers();

	this->endOfDataSource = false;

    this->setupBackendDataProvider(startFrameNo, doChkPointing);

	this->initialized = true;

	DNN_CHECK(this->startup_worker());
};


void DNNDataProvider::resetDataProvider()
{
 	if ( !this->initialized ) {
		 dnn_log("DNNDataProvider", "The DataProvider is still not started yet, no reset should be called");
		 DNN_Exception("");
	};
	DNN_CHECK(this->shutdown_worker());

	this->reset_transfer_buffers();
	this->reset_io_buffers();

	this->endOfDataSource = false;

	if ( this->supportChkPointing )
		 DNN_LOCK_INIT(&this->chkPointingLock);

	this->resetBackendDataProvider();

	DNN_CHECK(this->startup_worker());
};

int DNNDataProvider::getBatchSize()
{
    return(this->m_batchSize);
};

DNN_DATA_MODE DNNDataProvider::getDataMode()
{
    return(this->dataMode);
};

void DNNDataProvider::load_stats_info(const char *filePath)
{
	this->meanvalues = new float[this->m_dataFeatureSize]; 
	this->stddevs = new float[this->m_dataFeatureSize]; 
    
	read_stats_info(filePath, this->m_dataFeatureSize, this->meanvalues, this->stddevs); 
}; 