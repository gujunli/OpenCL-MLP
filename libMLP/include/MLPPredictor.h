/*
 *  COPYRIGHT:  Copyright (c) 2013 Advanced Micro Devices, Inc.  All rights reserved
 *
 *   Changed by  Qianfeng Zhang@amd.com ( March 2014 )
 *
 *   Written by  Junli Gu @amd.com ( Dec 2013 )
 */


#ifndef _MPL_PREDICTOR_H_
#define _MPL_PREDICTOR_H_

#include <CL/cl.h>
#include <CL/cl_ext.h>

#include "MLPApiExport.h"
#include "MLPCommon.h"
#include "MLPConstants.h"
#include "SingleDevClass.h"
#include "MLPNetProvider.h"
#include "MLPDataProvider.h"


class MLPPredictor
{
private:
	bool initialized;
private:
	MLP_OCL_DEVTYPE devType;

	MLP_NETTYPE netType;
	int   nLayers;
	int   batchSize;
	int  *dimensions;
	ACT_FUNC *actFuncs;

	cl_mem *inputs;
	cl_mem *weights;
	cl_mem *biases;
	cl_mem output;

	cl_mem *biasMatrixes;

private:
	static MLP_Kerns mykerns;

	static SingleDevClass * CLContext;
	static int nInstances ;

private:
	void setDefault();
	void _initialize(MLPNetProvider & NetProvider, int minibatch);
	void _dispose();

private:
    void expandFloatVectorToMatrix(cl_mem  myVector, cl_mem myMatrix, int width, int height);  // helper
	void activate(int layer, cl_mem x, cl_mem y, int width, int height);

public:
	LIBMLPAPI MLPPredictor();
	LIBMLPAPI MLPPredictor(MLPNetProvider & netProvider, MLP_OCL_DEVTYPE devType, int batchSize);
	LIBMLPAPI ~MLPPredictor();

public:
	LIBMLPAPI static SingleDevClass* getCLContext()
	{
		return CLContext;
	}
	LIBMLPAPI void setupMLP(MLPNetProvider & netProvider, int batchSize);

	LIBMLPAPI void batchPredicting(float *inVectors, float *outVectors);
	LIBMLPAPI void singlePredicting(float *inVector, float *outVector);

	LIBMLPAPI int getInputVectorSize();
	LIBMLPAPI int getOutputVectorSize();
	LIBMLPAPI int getBatchSize();
};


#endif // __MPL_PREDICTOR_H
