/*
 *  COPYRIGHT:  Copyright (c) 2014 Advanced Micro Devices, Inc.  All rights reserved
 *
 *   Written by Qianfeng Zhang@amd.com ( March 2014 )
 *
 */

#include "MLPUtil.h"
#include "MLPDataProvider.h"

MLPDataProvider::MLPDataProvider()
{
	this->haveLabel = false;
	this->m_dataFeatureSize = 0;
	this->m_dataLabelSize = 0;
	this->m_batchSize = 0;

	for (int i=0; i< MLP_BATCH_RING_SIZE; i++) {
	     this->features[i] = NULL;
	     this->labels[i] = NULL;
	};

	this->running = false;

	this->initialized = false;
};

MLPDataProvider::~MLPDataProvider()
{
};

// create the data buffers and initialize the locks and condition variable
void MLPDataProvider::create_buffers(int batchSize)
{

	for (int i=0; i< MLP_BATCH_RING_SIZE; i++) {

        this->features[i] = new float[batchSize*this->m_dataFeatureSize*sizeof(float)];

	    if ( this->haveLabel )
		     this->labels[i] = new float[batchSize*this->m_dataLabelSize*sizeof(float)];
	    else
		     this->labels[i] = NULL;
	};

	this->m_batchSize = batchSize;

	this->rbuf_count = 0;
	this->wbuf_count = MLP_BATCH_RING_SIZE;

#ifdef _WIN32      // for Windows
	InitializeConditionVariable(&this->readReady);
	InitializeConditionVariable(&this->writeReady);
#else             // for Linux
	pthread_cond_init(&this->readReady, NULL);
	pthread_cond_init(&this->writeReady, NULL);
#endif

	MLP_LOCK_INIT(&this->bufferLock);
};

// release the data buffers
void MLPDataProvider::release_buffers()
{

	for (int i=0; i< MLP_BATCH_RING_SIZE; i++) {
	     delete [] this->features[i];

	     if ( this->haveLabel )
		      delete [] this->labels[i];
	};
};

void MLPDataProvider::reset_buffers()
{
	this->rbuf_count = 0;
	this->wbuf_count = MLP_BATCH_RING_SIZE;

#ifdef _WIN32      // for Windows
	InitializeConditionVariable(&this->readReady);
	InitializeConditionVariable(&this->writeReady);
#else             // for Linux
	pthread_cond_init(&this->readReady, NULL);
	pthread_cond_init(&this->writeReady, NULL);
#endif

	MLP_LOCK_INIT(&this->bufferLock);
};

// create and start the worker thread
int MLPDataProvider::startup_worker()
{
	if ( !this->initialized )
		 return(-1);

	if ( this->running )
		 return(-2);

	this->running = true;
	MLP_CREATE_THREAD(&this->worker,MLPDataProvider::worker_fun,(void*)this);

	return(0);
};

// shutdown the worker thread
int MLPDataProvider::shutdown_worker()
{
    MLP_KILL_THREAD(this->worker);
	MLP_JOIN_THREAD(this->worker);
	this->running = false;

	return(0);
};

int MLPDataProvider::getTotalBatches()
{
	return(this->total_batches);
};

int MLPDataProvider::getFeatureSize()
{
	return(this->m_dataFeatureSize);
};

int MLPDataProvider::getLabelSize()
{
	return(this->m_dataLabelSize);
};



#ifdef _WIN32             // for Windows

// get the pointer of the new batch of data
int MLPDataProvider::getBatchData(int batchSize, float * & pFeatures, bool blocking)
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
int MLPDataProvider::getBatchData(int batchSize, float * & pFeatures, float * & pLabels, bool blocking)
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
int MLPDataProvider::nextBatch()
{
	 EnterCriticalSection(&this->bufferLock);

	 this->rbuf_count--;
	 this->wbuf_count++;
	 this->rbuf_index = (this->rbuf_index+1) % MLP_BATCH_RING_SIZE;

	 WakeConditionVariable(&this->writeReady);

	 LeaveCriticalSection(&this->bufferLock);

	 return(0);
};

// the worker thread which read batches of data asynchronously
void *MLPDataProvider::worker_fun(void *argp)
{
	 MLPDataProvider *objp;

	 objp = (MLPDataProvider *)argp;
	 objp->rbuf_index = 0;
	 objp->wbuf_index = 0;

	 objp->prepare_batch_data();    // first time to prepare the data onto the buffer

	 EnterCriticalSection(&objp->bufferLock);

     objp->wbuf_index = (objp->wbuf_index+1) % MLP_BATCH_RING_SIZE;

	 objp->rbuf_count = 1;
	 objp->wbuf_count = MLP_BATCH_RING_SIZE-1;

	 WakeConditionVariable(&objp->readReady);

	 LeaveCriticalSection(&objp->bufferLock);

	 while ( objp->running ) {
            EnterCriticalSection(&objp->bufferLock);

		    if ( objp->wbuf_count > 0 ) {

                 LeaveCriticalSection(&objp->bufferLock);

				 if ( ! objp->haveBatchToProvide() )
						break;

				 objp->prepare_batch_data();   // load a batch of data

	             EnterCriticalSection(&objp->bufferLock);

                 objp->wbuf_index = (objp->wbuf_index+1) % MLP_BATCH_RING_SIZE;
				 objp->wbuf_count--;
				 objp->rbuf_count++;

			     WakeConditionVariable(&objp->readReady);

                 LeaveCriticalSection(&objp->bufferLock);
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
int MLPDataProvider::getBatchData(int batchSize, float * & pFeatures, bool blocking)
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
int MLPDataProvider::getBatchData(int batchSize, float * & pFeatures, float * & pLabels, bool blocking)
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
int MLPDataProvider::nextBatch()
{
	 pthread_mutex_lock(&this->bufferLock);

	 this->rbuf_count--;
	 this->wbuf_count++;
	 this->rbuf_index = (this->rbuf_index+1) % MLP_BATCH_RING_SIZE;

	 pthread_cond_signal(&this->writeReady);

	 pthread_mutex_unlock(&this->bufferLock);

	 return(0);
};


// the worker thread which read batches of data asynchronously
void *MLPDataProvider::worker_fun(void *argp)
{
	 MLPDataProvider *objp;

	 objp = (MLPDataProvider *)argp;
	 objp->rbuf_index = 0;
	 objp->wbuf_index = 0;

	 objp->prepare_batch_data();    // first time to prepare the data onto the buffer

	 pthread_mutex_lock(&objp->bufferLock);

     objp->wbuf_index = (objp->wbuf_index+1) % MLP_BATCH_RING_SIZE;

	 objp->rbuf_count = 1;
	 objp->wbuf_count = MLP_BATCH_RING_SIZE-1;

	 pthread_cond_signal(&objp->readReady);

	 pthread_mutex_unlock(&objp->bufferLock);

	 while ( objp->running ) {
            pthread_mutex_lock(&objp->bufferLock);

		    if ( objp->wbuf_count > 0 ) {

                 pthread_mutex_unlock(&objp->bufferLock);

				 if ( ! objp->haveBatchToProvide() )
						break;

				 objp->prepare_batch_data();   // load a batch of data

	             pthread_mutex_lock(&objp->bufferLock);

                 objp->wbuf_index = (objp->wbuf_index+1) % MLP_BATCH_RING_SIZE;
				 objp->wbuf_count--;
				 objp->rbuf_count++;

			     pthread_cond_signal(&objp->readReady);

                 pthread_mutex_unlock(&objp->bufferLock);
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
void MLPDataProvider::load_feature_batch(float *srcp, int *indexBase, int indexOffset)
{
     for(int i = 0; i < this->m_batchSize; i++)
		for(int j = 0; j < this->m_dataFeatureSize; j++)
			this->features[this->wbuf_index][i*this->m_dataFeatureSize+j] = srcp[indexBase[indexOffset+i]*this->m_dataFeatureSize+j];
};

// load one batch of labels data from source to data buffer
void MLPDataProvider::load_label_batch(float *srcp, int *indexBase, int indexOffset)
{
     for(int i = 0; i < this->m_batchSize; i++)
		for(int j = 0; j < this->m_dataLabelSize; j++)
			this->labels[this->wbuf_index][i*this->m_dataLabelSize+j] = srcp[indexBase[indexOffset+i]*this->m_dataLabelSize+j];
};


bool MLPDataProvider::batchAvailable()
{
	if ( this->haveBatchToProvide() )
		 return(true);

	if ( this->rbuf_count > 0 )
		 return(true);

	return(false);
};

void MLPDataProvider::setupDataProvider()
{
	this->supportChkPointing = false;

	this->create_buffers(this->m_batchSize);

    this->setupBackendDataProvider();

	this->initialized = true;

	MLP_CHECK(this->startup_worker());
};

void MLPDataProvider::setupDataProvider(int startFrameNo, bool doChkPointing)
{
 	if ( this->dataMode != MLP_DATAMODE_TRAIN ) {
		 mlp_log("MLPDataProvider", "This interface can only be called with the TRAIN mode");
		 MLP_Exception("");
	};

	// For CheckPoint
	this->supportChkPointing = doChkPointing;
	if ( this->supportChkPointing )
		 MLP_LOCK_INIT(&this->chkPointingLock);

	this->create_buffers(this->m_batchSize);

    this->setupBackendDataProvider(startFrameNo, doChkPointing);

	this->initialized = true;

	MLP_CHECK(this->startup_worker());
};


void MLPDataProvider::resetDataProvider()
{
 	if ( !this->initialized ) {
		 mlp_log("MLPDataProvider", "The DataProvider is still not started yet, no reset should be called");
		 MLP_Exception("");
	};
	MLP_CHECK(this->shutdown_worker());

	this->reset_buffers();

	if ( this->supportChkPointing )
		 MLP_LOCK_INIT(&this->chkPointingLock);

	this->resetBackendDataProvider();

	MLP_CHECK(this->startup_worker());
};

