/*
 *  COPYRIGHT:  Copyright (c) 2014 Advanced Micro Devices, Inc.  All rights reserved
 *
 *   Written by  Qianfeng Zhang@amd.com ( March 2014 )
 *
 *   Written by  Junli Gu @amd.com ( Dec 2013 )
 */


#ifndef _MPL_TESTER_BASE_H_
#define _MPL_TESTER_BASE_H_

#include "DNNApiExport.h"
#include "DNNConstants.h"
#include "DNNDataProvider.h"
#include "MLPNetProvider.h"


typedef bool (*VECTOR_MATCH)(float *inVector, float *labelVector, int len);

class MLPTesterBase
{
protected:
	bool initialized;

	MLP_NETTYPE netType;
	int   nLayers;
	int   batchSize;
	int  *dimensions;
	ACT_FUNC *actFuncs;

	DNNDataProvider *dataProviderp;

	int succTestFrames;
	int totalTestFrames;

protected:
	void _initialize(MLPNetProvider & NetProvider, int minibatch);

private:
	void _dispose();

public:
	LIBDNNAPI MLPTesterBase();
	LIBDNNAPI virtual ~MLPTesterBase()=0;

public:
	LIBDNNAPI virtual void setupMLP(MLPNetProvider & netProvider, DNNDataProvider & dataProvider, int minipatch)=0;

	LIBDNNAPI virtual void batchTesting(int maxBatches)=0;
	LIBDNNAPI virtual bool singleTesting(float *inputVector, float *labelVector, VECTOR_MATCH matchFunc)=0;

	LIBDNNAPI DNNDataProvider *getDataProvider();

	LIBDNNAPI void getTestingStats(int &totalFrames, int &succFrames);

    LIBDNNAPI int getInputVectorSize();
    LIBDNNAPI int getOutputVectorSize();
	LIBDNNAPI int getBatchSize();
};


#endif // __MPL_TESTER_BASE_H_
