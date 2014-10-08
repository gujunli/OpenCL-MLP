/*
 *  COPYRIGHT:  Copyright (c) 2013 Advanced Micro Devices, Inc.  All rights reserved
 *
 *   Written by  Qianfeng Zhang@amd.com ( March 2014 )
 *
 *   Written by  Junli Gu @amd.com ( Dec 2013 )
 */


#ifndef _MLP_TRAINER_OCL_H_
#define _MLP_TRAINER_OCL_H_

#include <CL/cl.h>
#include <CL/cl_ext.h>

#include "DNNApiExport.h"
#include "DNNConstants.h"
#include "DNNDataProvider.h"

#include "MLPOclCommon.h"
#include "SingleDevClass.h"
#include "MLPConfigProvider.h"
#include "MLPChkPointState.h"
#include "MLPTrainerBase.h"

// Implement the interfaces for training the MLP network
class MLPTrainerOCL:public MLPTrainerBase
{
private:
	DNN_OCL_DEVTYPE devType;

	cl_mem *inputs;              // Device buffers to store input/output data calculated on various layers of the MLP network
	cl_mem *weightT;             // Device buffers to store weights matrix of various layers of the MLP network
	cl_mem *biases;              // Device buffers to store biases vector of various layers of the MLP network
	cl_mem output;               // Device buffer to store input/output data of the output
	cl_mem target;               // Device buffer to store label data provided to the MLP network
	cl_mem *delta;	             // Device buffer to store delta of output data calculated on various layers of the MLP network

	float *reduceBuff;           // Dynamically allocated host buffer used by some reducing operations (eg.  calculateError )
	cl_mem reduceMem;            // Dynamically allocated device memory used by some reducing operations (eg. calculateError )


private:
	static MLP_Kerns mykerns;

	static SingleDevClass * CLCtx;
	static int nInstances;

private:
	void setup_ocl_kernels();
	void destroy_ocl_kernels();
	void create_ocl_buffers(MLPConfigProvider &provider);
	void release_ocl_buffers();

private:
	void transpose_float_matrix(cl_mem src, cl_mem dst, cl_int width, cl_int height);          // helper
    void expandFloatVectorToMatrix(cl_mem  myVector, cl_mem myMatrix, int width, int height);  // helper
	void activate(int layer, cl_mem x, cl_mem y, int width, int height);
	void calculateError(cl_mem output, cl_mem target, int width, int height, float &ret);
	void calculateDelta(cl_mem output, cl_mem target, cl_mem delta, int width, int height);
	void derivative(int layer, cl_mem delta1, cl_mem y, cl_mem delta2, int width, int height);

public:
	LIBDNNAPI MLPTrainerOCL();
	LIBDNNAPI MLPTrainerOCL(MLPConfigProvider & configProvider, DNNDataProvider & dataProvider, DNN_OCL_DEVTYPE devType, int _minibatch);
    ~MLPTrainerOCL();

public:
	void setupMLP(MLPConfigProvider & configProvider, DNNDataProvider & dataProvider, int _minipatch);

	int batchTrainingWithCheckPointing(int maxBatches, int startBatch, int startEpoch, bool doChkPointing);
	void synchronizeNetConfig(MLPConfigProvider &configProvider);

};


#endif // __MPL_TRAINER_OCL_H_
