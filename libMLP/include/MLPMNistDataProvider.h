/*
 *  COPYRIGHT:  Copyright (c) 2014 Advanced Micro Devices, Inc.  All rights reserved
 *
 *   Written by Qianfeng Zhang@amd.com ( March 2014 )
 *
 */

#ifndef _MLP_MNIST_DATA_PROVIDER_H_
#define _MLP_MNIST_DATA_PROVIDER_H_

#include <fstream>

#include "MLPApiExport.h"
#include "MLPDataProvider.h"

using namespace std;

// for MNIST image data
class MLPMNistDataProvider:public MLPDataProvider
{
private:
	ifstream dataFile;
	ifstream labelFile;

	int  batchNo;              // Accumulated number of batches that have been read from the data source, presenting the latest batch to see
	                           // this is mainly used to determine a checkpointing location

	int num_frames;            // total number of data frames

    int imageWidth;            // width of the input image, only used by distorting_frame()
	int imageHeight;           // height of the input image, only used by distorting_frame()

public:
	LIBMLPAPI MLPMNistDataProvider();
	LIBMLPAPI MLPMNistDataProvider(const char *dataPath, MLP_DATA_MODE mode, int batchSize, int shuffleBatches);

    ~MLPMNistDataProvider();

    void setupBackendDataProvider();                                                      // implementation of public base class virtual interface
	void resetBackendDataProvider();                                                      // implementation of public base class virtual interface
    bool frameMatching(const float *frameOutput, const float *frameLabel, int len);       // implementation of public base class virtual interface

    // The following two interfaces are only used by the CheckPointing Function
    void getCheckPointFrame(int & frameNo);                                   // implementation of public base class virtual interface
    void setupBackendDataProvider(int startFrameNo, bool doChkPointing);      // implementation of public base class virtual interface

private:
	 void setup_first_data_batches();            // first time read a group of batches from the source and setup them on the io buffers
	 void setup_cont_data_batches();             // read a group of batches from the source and setup them on the io buffers

     void InitializeFromMNistSource(const char *dataPath);
	 void gotoDataFrame(int frameNo);
	 void gotoLabelFrame(int frameNo);
};

#ifdef  _WIN32
#define MNIST_PATH "../../MNIST/"
#else
#define MNIST_PATH "../../MNIST/"
#endif

struct header_imagefile {
	unsigned int magicNum;
	unsigned int numImages;
	unsigned int imageHeight;
	unsigned int imageWidth;
};

struct header_labelfile {
	unsigned int magicNum;
	unsigned int numLabels;
};

#endif
