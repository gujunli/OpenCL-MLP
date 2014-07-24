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
	int  batchNo;                // Current batchNo of data it is providing
	int  total_batches;          // Total number of batches provided by the data source and provider (shuffled batches also counted)

	int m_batchSize;             // Number of input frames in each minibatch
	int m_shuffleBatches;        // Number of minibatches of input frames for shuffling

	int m_dataFeatureSize;       // Size of feature frame as input to neural network, in units of float,  same as the dimension of the input layer
	int m_dataLabelSize;         // Size of label frame as input to neural network, in units of float, same as the dimension of the output layer
	bool haveLabel;              // Indicates if we need use label frames, label frames are needed for training and testing, but not for predicting

	float *features[MLP_BATCH_RING_SIZE];       // ring buffer for feature frames batches, which will be directly delivered to the neural network
	float *labels[MLP_BATCH_RING_SIZE];         // ring buffer for label frames batches, which will be directly delivered to the neural network
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

protected:
	void create_buffers(int batchSize);
	void release_buffers();
	void reset_buffers();
    void load_feature_batch(float *srcp, int *indexBase, int indexOffset);      // Load one batch of feature frames from the provider's internal buffer to the double-buffer
    void load_label_batch(float *srcp, int *indexBase, int indexOffset);        // Load one batch of feature frames from the provider's internal buffer to the double-buffer

	int startup_worker();
	int shutdown_worker();

private:
	virtual void prepare_batch_data()=0;
	virtual bool haveBatchToProvide()=0;

	static void * worker_fun(void *argp);

public:
	LIBMLPAPI MLPDataProvider();

	LIBMLPAPI int getBatchData(int batchSize, float * & pFeatures, bool blocking);
	LIBMLPAPI int getBatchData(int batchSize, float * & pFeatures, float * & pLabels, bool blocking);
	LIBMLPAPI int nextBatch();

	LIBMLPAPI int getFeatureSize();
	LIBMLPAPI int getLabelSize();
	LIBMLPAPI int getTotalBatches();

	LIBMLPAPI virtual ~MLPDataProvider()=0;
	LIBMLPAPI virtual void setupDataProvider()=0;
	LIBMLPAPI virtual void resetDataProvider()=0;
	LIBMLPAPI virtual bool frameMatching(const float *frameOutput, const float *frameLabel, int len)=0;

    // The following two interfaces are only used by the CheckPointing Function
    virtual void getCheckPointFrame(int & frameNo)=0;                           // Use to get the Frame Position the DataProvider should start from
    virtual void setupDataProvider(int startFrameNo, bool doChkPointing)=0;     // Setup the DataProvider to provide data start from this Frame Position

	LIBMLPAPI bool batchAvailable();
};


#endif
