/*
 *  COPYRIGHT:  Copyright (c) 2014 Advanced Micro Devices, Inc.  All rights reserved
 *
 *   Written by  Qianfeng Zhang@amd.com ( March 2014 )
 *
 *   Written by  Junli Gu @amd.com ( Dec 2013 )
 */


#ifndef _MPL_TESTER_OCL_H_
#define _MPL_TESTER_OCL_H_

#include <CL/cl.h>
#include <CL/cl_ext.h>

#include "DNNApiExport.h"
#include "DNNConstants.h"
#include "DNNDataProvider.h"
#include "MLPOclCommon.h"
#include "SingleDevClass.h"
#include "MLPConfigProvider.h"
#include "MLPTesterBase.h"


class MLPTesterOCL:public MLPTesterBase
{
private:
	DNN_OCL_DEVTYPE devType;

	cl_mem *inputs;
	cl_mem *weights;
	cl_mem *biases;
	cl_mem target;
	cl_mem output;

	cl_mem *biasMatrixes;

private:
    static MLP_Kerns mykerns;

	static SingleDevClass * CLCtx;
	static int nInstances ;

private:
	void setup_ocl_kernels();
	void destroy_ocl_kernels();
	void create_ocl_buffers(MLPConfigProvider & configProvider);
	void release_ocl_buffers();

private:
    void expandFloatVectorToMatrix(cl_mem  myVector, cl_mem myMatrix, int width, int height);  // helper
	void activate(int layer, cl_mem x, cl_mem y, int width, int height);

public:
	LIBDNNAPI MLPTesterOCL();
	LIBDNNAPI MLPTesterOCL(MLPConfigProvider & configProvider, DNNDataProvider & dataProvider, DNN_OCL_DEVTYPE devType, int _batchSize);
	~MLPTesterOCL();

public:
	void setupMLP(MLPConfigProvider & configProvider, DNNDataProvider & dataProvider, int minipatch);
	void batchTesting(int maxBatches);
	bool singleTesting(float *inputVector, float *labelVector, VECTOR_MATCH matchFunc);
};


#endif // __MPL_TESTER_OCL_H_
