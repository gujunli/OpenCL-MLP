/*
 *  COPYRIGHT:  Copyright (c) 2013 Advanced Micro Devices, Inc.  All rights reserved
 *
 *   Changed by  Qianfeng Zhang@amd.com ( March 2014 )
 * 
 *   Written by  Junli Gu @amd.com ( Dec 2013 )
 */


#ifndef _MLP_TRAINER_H_
#define _MLP_TRAINER_H_

#ifdef _WIN32
#include <Windows.h>
#else
#include <pthread.h>
#endif

#include <CL/cl.h>
#include <CL/cl_ext.h>

#include "MLPApiExport.h"
#include "MLPConstants.h"
#include "MLPCommon.h"
#include "SingleDevClass.h"
#include "MLPNetProvider.h"
#include "MLPDataProvider.h"
#include "MLPChkPointState.h"

// Implement the interfaces for training the MLP network
class MLPTrainer
{
private:
	bool initialized;   
private:                
	MLP_OCL_DEVTYPE devType; 

	MLP_NETTYPE netType; 
	int   nLayers;               // Number of layers
	int   minibatch;             // Size of minibatch -- number of input frames received and processed by the MLP network in batches
	int  *dimensions;            // Dimensions of all layers, first layer is the input and last layer is the output
    float *etas;                 // Learning rate of all layers
	float momentum;              // Used to indicate the importance of historic variance of weights
	ACT_FUNC *actFuncs;          // Activation function used by all layers,  usually all hidden layers use same activation function, the output uses different one
	COST_FUNC costFunc;          // Cost function used to measure the error value of the input batch got on the current MLP network 

	cl_mem *inputs;              // Device buffers to store input/output data calculated on various layers of the MLP network
	cl_mem *weightT;             // Device buffers to store weights matrix of various layers of the MLP network
	cl_mem *biases;              // Device buffers to store biases vector of various layers of the MLP network
	cl_mem output;               // Device buffer to store input/output data of the output
	cl_mem target;               // Device buffer to store label data provided to the MLP network
	cl_mem *delta;	             // Device buffer to store delta of output data calculated on various layers of the MLP network
	
	float *reduceBuff;           // Dynamically allocated host buffer used by some reducing operations (eg.  calculateError )
	cl_mem reduceMem;            // Dynamically allocated device memory used by some reducing operations (eg. calculateError )

	MLPDataProvider *dataProviderp; 

	int currBatchNo;             // Indicate the current batchNo the training is on, need be saved when doing checkpointing 
	int currEpoch;               // Indicate the current epoch the training is on, need be saved when doing checkpointing

#ifdef WIN32                       // for Windows
	CRITICAL_SECTION chkPointingLock; 
#else                              // for Linux
	pthread_mutex_t chkPointingLock; 
#endif


private: 
	static MLP_Kerns mykerns; 

	static SingleDevClass * CLContext; 
	static int nInstances;

private:
	void setDefault();
	void _initialize(MLPNetProvider & NetProvider, int minibatch);
	void _dispose(); 

private:
	void transpose_float_matrix(cl_mem src, cl_mem dst, cl_int width, cl_int height);          // helper 
    void expandFloatVectorToMatrix(cl_mem  myVector, cl_mem myMatrix, int width, int height);  // helper
	void activate(int layer, cl_mem x, cl_mem y, int width, int height);	
	void calculateError(cl_mem output, cl_mem target, int width, int height, float &ret); 
	void calculateDelta(cl_mem output, cl_mem target, cl_mem delta, int width, int height); 
	void derivative(int layer, cl_mem delta1, cl_mem y, cl_mem delta2, int width, int height);	

public:
	LIBMLPAPI MLPTrainer();
	LIBMLPAPI MLPTrainer(MLPNetProvider & netProvider, MLPDataProvider & dataProvider, MLP_OCL_DEVTYPE devType, int minibatch);
	LIBMLPAPI ~MLPTrainer();

public:	 	 
	LIBMLPAPI static SingleDevClass* getCLContext()
	{
		return CLContext;
	}
	LIBMLPAPI void setupMLP(MLPNetProvider & netProvider, MLPDataProvider & dataProvider, int minipatch);	

	LIBMLPAPI MLPDataProvider *getDataProvider(); 

	LIBMLPAPI int batchTraining(int maxBatches, int epoches);	
	LIBMLPAPI int batchTrainingWithCheckPointing(int maxBatches, int epoches, int startBatch, int startEpoch, bool doChkPointing); 

	LIBMLPAPI void saveNetConfig(const char *configPath); 
	LIBMLPAPI void synchronizeNetConfig(MLPNetProvider &netProvider); 
	LIBMLPAPI void showNetConfig(); 

	void checkPointing(struct MLPCheckPointState &state);
};


#endif // __MPL_TRAINER_H_
