/*
 *  COPYRIGHT:  Copyright (c) 2014 Advanced Micro Devices, Inc.  All rights reserved
 *
 *   Changed by  Qianfeng Zhang@amd.com ( March 2014 )
 *
 *   Written by  Junli Gu @amd.com ( Dec 2013 )
 */


#ifndef _MPL_TESTER_H_
#define _MPL_TESTER_H_

#include <CL/cl.h>
#include <CL/cl_ext.h>

#include "MLPApiExport.h"
#include "MLPConstants.h"
#include "MLPCommon.h"
#include "SingleDevClass.h"
#include "MLPNetProvider.h"
#include "MLPDataProvider.h"

class MLPTester
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
	cl_mem target;
	cl_mem output;

	MLPDataProvider *dataProviderp; 

	int succTestFrames; 
	int totalTestFrames;

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
	LIBMLPAPI MLPTester();
	LIBMLPAPI MLPTester(MLPNetProvider & netProvider, MLPDataProvider & dataProvider, MLP_OCL_DEVTYPE devType, int minipatch);
	LIBMLPAPI ~MLPTester();

public:	 	 
	LIBMLPAPI static SingleDevClass* getCLContext()
	{
		return CLContext;
	}
	LIBMLPAPI void setupMLP(MLPNetProvider & netProvider, MLPDataProvider & dataProvider, int minipatch);	

	LIBMLPAPI MLPDataProvider *getDataProvider(); 

	LIBMLPAPI void batchTesting(int maxBatches); 

	LIBMLPAPI void getTestingStats(int &totalFrames, int &succFrames); 
};


#endif // __MPL_TESTER_H_
