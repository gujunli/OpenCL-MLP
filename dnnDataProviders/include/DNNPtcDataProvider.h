/*
 *  COPYRIGHT:  Copyright (c) 2014 Advanced Micro Devices, Inc.  All rights reserved
 *
 *   Written by Qianfeng Zhang@amd.com ( March 2014 )
 *
 */

#ifndef _DNN_PTC_DATA_PROVIDER_H_
#define _DNN_PTC_DATA_PROVIDER_H_

#include <fstream>

#include "DNNApiExport.h"
#include "DNNDataProvider.h"

using namespace std;

// for Printed-Text Characters data set
class DNNPtcDataProvider:public DNNDataProvider
{
private:
	ifstream dataFile;

	int  batchNo;              // Current batchNo of data it is providing

	int num_frames;            // total number of data frames

    int imageWidth;            // width of the input image
	int imageHeight;           // height of the input image

public:
	LIBDNNAPI DNNPtcDataProvider();
	LIBDNNAPI DNNPtcDataProvider(const char *dataPath, DNN_DATA_MODE mode, int batchSize, int shuffleBatches);

    ~DNNPtcDataProvider();

    void setupBackendDataProvider();                                                      // implementation of public base class virtual interface
	void resetBackendDataProvider();                                                      // implementation of public base class virtual interface
    bool frameMatching(const float *frameOutput, const float *frameLabel, int len);       // implementation of public base class virtual interface

    // The following two interfaces are only used by the CheckPointing Function
    void getCheckPointFrame(int & frameNo);                                   // implementation of public base class virtual interface
    void setupBackendDataProvider(int startFrameNo, bool doChkPointing);      // implementation of public base class virtual interface

private:
	 void setup_first_data_batches();            // first time read a group of batches from the source and setup them on the io buffers
	 void setup_cont_data_batches();             // read a group of batches from the source and setup them on the io buffers

     void InitializeFromPtcSource(const char *dataPath);
	 void gotoDataFrame(int frameNo);
};


#define PTC_DB_PATH "../../ptc_dataset/English-ptc/"

struct ptc_sample_header {
    unsigned short wCode;
    unsigned short index;
};

struct ptc_db_header {
    unsigned char tag[4];        // tag of the PTC dataset file, should be "PTC!"
    unsigned int numChars;       // number of characters described by this database of printed-text character samples
    unsigned int numSamples;     // number of samples included by this database of printed-text characters
    unsigned int sWidth;         // pixmap width of each sample
    unsigned int sHeight;        // pixmap height of each sample
};

#endif
