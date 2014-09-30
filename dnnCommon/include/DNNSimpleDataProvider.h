/*
 *  COPYRIGHT:  Copyright (c) 2014 Advanced Micro Devices, Inc.  All rights reserved
 *
 *   Written by Qianfeng Zhang@amd.com ( March 2014 )
 *
 */


#ifndef _DNN_SIMPLE_DATA_PROVIDER_H_
#define _DNN_SIMPLE_DATA_PROVIDER_H_

#include "DNNApiExport.h"
#include "DNNDataProvider.h"

// For simple data randomly produced on the memory buffer
class DNNSimpleDataProvider:public DNNDataProvider
{
private:
	int  batchNo;              // Accumulated number of batches that have been read from the data source, presenting the latest batch to see
	                           // this is mainly used to determine a checkpointing location

public:
	LIBDNNAPI DNNSimpleDataProvider();
	LIBDNNAPI DNNSimpleDataProvider(DNN_DATA_MODE mode, int dataFeatureSize, int dataLabelSize, int batchSize, int shuffleBatches);

    ~DNNSimpleDataProvider();

    void setupBackendDataProvider();
	void resetBackendDataProvider();

    bool frameMatching(const float *frameOutput, const float *frameLabel, int len);

    // The following two interfaces are only used by the CheckPointing Function
	void getCheckPointFrame(int & frameNo) {};                                   // Use to get the Frame Position the DataProvider should start from
	void setupBackendDataProvider(int startFrameNo, bool doChkPointing) {};      // Setup the DataProvider to provide data starting from this Frame Position

private:
	 void setup_first_data_batches();            // first time read a group of batches from the source and setup them on the io buffers
	 void setup_cont_data_batches();             // read a group of batches from the source and setup them on the io buffers
};

#endif
