/*
 *  COPYRIGHT:  Copyright (c) 2013 Advanced Micro Devices, Inc.  All rights reserved
 *
 *   Written by  Qianfeng Zhang@amd.com ( March 2014 )
 *
 *   Written by  Junli Gu @amd.com ( Dec 2013 )
 */


#ifndef _MPL_PREDICTOR_BASE_H_
#define _MPL_PREDICTOR_BASE_H_

#include "DNNApiExport.h"
#include "DNNConstants.h"
#include "DNNDataProvider.h"
#include "MLPNetProvider.h"


class MLPPredictorBase
{
protected:
	bool initialized;

	MLP_NETTYPE netType;
	int   nLayers;
	int   batchSize;
	int  *dimensions;
	ACT_FUNC *actFuncs;

protected:
	void _initialize(MLPNetProvider & NetProvider, int minibatch);

private:
	void _dispose();

public:
	LIBDNNAPI MLPPredictorBase();
	LIBDNNAPI virtual ~MLPPredictorBase()=0;

public:
	LIBDNNAPI virtual void setupMLP(MLPNetProvider & netProvider, int batchSize)=0;

	LIBDNNAPI virtual void batchPredicting(float *inVectors, float *outVectors)=0;
	LIBDNNAPI virtual void singlePredicting(float *inVector, float *outVector)=0;

	LIBDNNAPI int getInputVectorSize();
	LIBDNNAPI int getOutputVectorSize();
	LIBDNNAPI int getBatchSize();
};


#endif // __MPL_PREDICTOR_BASE_H
