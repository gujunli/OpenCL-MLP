/*
 *  COPYRIGHT:  Copyright (c) 2014 Advanced Micro Devices, Inc.  All rights reserved
 *
 *   Written by  Qianfeng Zhang@amd.com ( March 2014 )
 *
 *   Written by  Junli Gu @amd.com ( Dec 2013 )
 */


#ifndef _MPL_TESTER_BASE_H_
#define _MPL_TESTER_BASE_H_

#include "MLPApiExport.h"
#include "MLPConstants.h"
#include "MLPNetProvider.h"
#include "MLPDataProvider.h"


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

	MLPDataProvider *dataProviderp;

	int succTestFrames;
	int totalTestFrames;

protected:
	void _initialize(MLPNetProvider & NetProvider, int minibatch);

private:
	void _dispose();

public:
	LIBMLPAPI MLPTesterBase();
	LIBMLPAPI virtual ~MLPTesterBase()=0;

public:
	LIBMLPAPI virtual void setupMLP(MLPNetProvider & netProvider, MLPDataProvider & dataProvider, int minipatch)=0;

	LIBMLPAPI virtual void batchTesting(int maxBatches)=0;
	LIBMLPAPI virtual bool singleTesting(float *inputVector, float *labelVector, VECTOR_MATCH matchFunc)=0;

	LIBMLPAPI MLPDataProvider *getDataProvider();

	LIBMLPAPI void getTestingStats(int &totalFrames, int &succFrames);

    LIBMLPAPI int getInputVectorSize();
    LIBMLPAPI int getOutputVectorSize();
	LIBMLPAPI int getBatchSize();
};


#endif // __MPL_TESTER_BASE_H_
