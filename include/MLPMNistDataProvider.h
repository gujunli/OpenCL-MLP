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
	float *featureData;
	float *labelData;

	int *permutations;

	ifstream dataFile;
	ifstream labelFile;

	int num_frames;            // total number of data frames
	int stageBatchNo;          // batch number inside each loaded batches (eg. inside each [m_shufflebatches * rounds] batches

	bool endOfDataSource;
    bool batches_loaded;       // the batches of data just were loaded from the file to the buffer

    int imageWidth;            // width of the input image, only used by distorting_frame()
	int imageHeight;           // height of the input image, only used by distorting_frame()

public:
	LIBMLPAPI MLPMNistDataProvider();
	LIBMLPAPI MLPMNistDataProvider(const char *dataPath, MLP_DATA_MODE mode, int batchSize, int shuffleBatches);

    ~MLPMNistDataProvider();

    void setupDataProvider();                                                 // implementation of public base class virtual interface
	void resetDataProvider();                                                 // implementation of public base class virtual interface
    bool frameMatching(float *frameOutput, float *frameLabel, int len);       // implementation of public base class virtual interface

    // The following two interfaces are only used by the CheckPointing Function
    void getCheckPointFrame(int & frameNo);                                   // implementation of public base class virtual interface
    void setupDataProvider(int startFrameNo, bool doChkPointing);             // implementation of public base class virtual interface

private:
	 void prepare_batch_data();                 // implementation of private base class virtual interface
	 bool haveBatchToProvide();                 // implementation of private base class virtual interface

	 void setup_first_data_batches();            // first time read data from the file and setup them on the memory
	 void setup_cont_data_batches();             // continue to read data from the file and setup them on the memory
	 void shuffle_data(int *index, int len);

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
