/*
 *  COPYRIGHT:  Copyright (c) 2014 Advanced Micro Devices, Inc.  All rights reserved
 *
 *   Written by Qianfeng Zhang@amd.com ( March 2014 )
 *
 */

#ifndef _MLP_DATA_PROVIDER_H_
#define _MLP_DATA_PROVIDER_H_

#ifdef _WIN32
#include <Windows.h>
#else
#include <pthread.h>
#endif

#include "MLPApiExport.h"
#include "MLPConstants.h"

#define MLP_BATCH_RING_SIZE 8

class MLPDataProvider
{
	friend class MLPTrainer;
	friend class MLPTester;
	friend class MLPPredictor;
protected:
	MLP_DATA_MODE  dataMode;     // TRAIN, TEST, PREDICT standing for three different usages of the neural network
	int  total_batches;          // Total number of batches provided by the data source and provider (shuffled batches also counted)

	int m_batchSize;             // Number of input frames in each minibatch
	int m_shuffleBatches;        // Number of minibatches of input frames for shuffling

	int m_dataFeatureSize;       // Size of feature frame as input to neural network, in units of float,  same as the dimension of the input layer
	int m_dataLabelSize;         // Size of label frame as input to neural network, in units of float, same as the dimension of the output layer
	bool haveLabel;              // Indicates if we need use label frames, label frames are needed for training and testing, but not for predicting

	float *features[MLP_BATCH_RING_SIZE];      // ring buffer for feature frames batches, which will be directly delivered to the neural network
	float *labels[MLP_BATCH_RING_SIZE];        // ring buffer for label frames batches, which will be directly delivered to the neural network
	int rbuf_index,wbuf_index;

	int rbuf_count;              // Used to implement a producer-consumer like synchronization between the neural network side and the data provider side
	int wbuf_count;

	bool supportChkPointing;     // Whether this Data Provider supports CheckPointing
#ifdef _WIN32                       // for Windows
	CRITICAL_SECTION chkPointingLock;
#else                               // for Linux
	pthread_mutex_t chkPointingLock;
#endif

	bool initialized;

#ifdef _WIN32                       // for Windows
    CONDITION_VARIABLE readReady;
    CONDITION_VARIABLE writeReady;
	CRITICAL_SECTION bufferLock;
#else                               // for Linux
    pthread_cond_t readReady;
	pthread_cond_t writeReady;
	pthread_mutex_t bufferLock;
#endif

#ifdef  _WIN32
    HANDLE worker;
#else
	pthread_t worker;
#endif
	bool running;                  // Indicate the worker thread is running

	int stageBatchNo;          // batch number inside each loaded batches (eg. inside each [m_shufflebatches * rounds] batches
    int batches_loaded;        // the number of batches that were just loaded from the file to the io buffers

	bool endOfDataSource;      // indicates the end of the data source, updated by the backend data provider codes

	float *featureData;
	float *labelData;
	int *permutations;

private:
	void create_transfer_buffers(int batchSize);
	void create_io_buffers(); 
	void reset_transfer_buffers();
	void reset_io_buffers(); 

	// Load one batch of feature frames from the backend io buffer to the front-end transfer buffer
    void load_feature_batch(float *srcp, int *indexBase, int indexOffset); 
	// Load one batch of label frames from the backend io buffer to the front-end transfer buffer
    void load_label_batch(float *srcp, int *indexBase, int indexOffset);      

	int startup_worker();

    void prepare_batch_data();

	bool haveBatchToProvide();

	virtual void setupBackendDataProvider()=0;
    virtual void resetBackendDataProvider()=0;
    virtual void setupBackendDataProvider(int startFrameNo, bool doChkPointing)=0;

	virtual void setup_cont_data_batches()=0;             // read group of batches from data source to the io buffers

protected:
	LIBMLPAPI int shutdown_worker(); 
	LIBMLPAPI void release_transfer_buffers();
	LIBMLPAPI void release_io_buffers(); 
  
    LIBMLPAPI void shuffle_data(int *index, int len);

private:
	static void * worker_fun(void *argp);

public:
	LIBMLPAPI MLPDataProvider();

	LIBMLPAPI int getBatchData(int batchSize, float * & pFeatures, bool blocking);
	LIBMLPAPI int getBatchData(int batchSize, float * & pFeatures, float * & pLabels, bool blocking);
	LIBMLPAPI int nextBatch();

	LIBMLPAPI int getFeatureSize();
	LIBMLPAPI int getLabelSize();
	LIBMLPAPI int getTotalBatches();

	LIBMLPAPI void setupDataProvider();
	LIBMLPAPI void resetDataProvider();

	LIBMLPAPI virtual bool frameMatching(const float *frameOutput, const float *frameLabel, int len)=0;

    // The following two interfaces are only used by the CheckPointing Function
    LIBMLPAPI virtual void getCheckPointFrame(int & frameNo)=0;                   // Use to get the Frame Position the DataProvider should start from
    LIBMLPAPI void setupDataProvider(int startFrameNo, bool doChkPointing);       // Setup the DataProvider to provide data start from this Frame Position

	LIBMLPAPI bool batchAvailable();

	LIBMLPAPI virtual ~MLPDataProvider()=0;
};


#endif