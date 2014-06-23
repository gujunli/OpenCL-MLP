/*
 *  COPYRIGHT:  Copyright (c) 2014 Advanced Micro Devices, Inc.  All rights reserved
 *
 *   Written by Qianfeng Zhang@amd.com ( March 2014 )
 *
 */


#ifndef _MLP_SIMPLE_DATA_PROVIDER_H_
#define _MLP_SIMPLE_DATA_PROVIDER_H_

#include "MLPApiExport.h"
#include "MLPDataProvider.h"

// For simple data randomly produced on the memory buffer
class MLPSimpleDataProvider:public MLPDataProvider
{
private:
	float *featureData; 
	float *labelData; 
	
	int *permutations; 

public:
	LIBMLPAPI MLPSimpleDataProvider(); 
	LIBMLPAPI MLPSimpleDataProvider(MLP_DATA_MODE mode, int dataFeatureSize, int dataLabelSize, int batchSize, int shuffleBatches);

    ~MLPSimpleDataProvider(); 

    void setupDataProvider(); 
	void resetDataProvider(); 
    bool endofInputBatches(); 
    bool frameMatching(float *frameOutput, float *frameLabel, int len); 

    // The following two interfaces are only used by the CheckPointing Function
	void getCheckPointFrame(int & frameNo) {};                            // Use to get the Frame Position the DataProvider should start from
	void setupDataProvider(int startFrameNo, bool doChkPointing) {};      // Setup the DataProvider to provide data starting from this Frame Position

private:
	 void prepare_batch_data();          // implementation of base class virtual interface
	 bool haveBatchToProvide();          // implementation of base class virtual interface

	 void setup_data_source(); 
	 void shuffle_data(int *index, int len); 

	 friend class MLPTrainer; 
}; 

#endif 
