/*
 *  COPYRIGHT:  Copyright (c) 2014 Advanced Micro Devices, Inc.  All rights reserved
 *
 *   Written by Qianfeng Zhang@amd.com ( March 2014 )
 *
 */

#ifndef _MLP_PTC_DATA_PROVIDER_H_
#define _MLP_PTC_DATA_PROVIDER_H_

#include <fstream>

#include "MLPApiExport.h"
#include "MLPDataProvider.h"

using namespace std;

// for Printed-Text Characters data set
class MLPPtcDataProvider:public MLPDataProvider
{
private:
	float *featureData;
	float *labelData;

	int *permutations;

	ifstream dataFile;

	int  batchNo;              // Current batchNo of data it is providing

	int num_frames;            // total number of data frames
	int stageBatchNo;          // batch number inside each loaded batches (eg. inside each [m_shufflebatches * rounds] batches

	bool endOfDataSource;
    bool batches_loaded;       // the batches of data just were loaded from the file to the buffer

    int imageWidth;            // width of the input image
	int imageHeight;           // height of the input image

public:
	LIBMLPAPI MLPPtcDataProvider();
	LIBMLPAPI MLPPtcDataProvider(const char *dataPath, MLP_DATA_MODE mode, int batchSize, int shuffleBatches);

    ~MLPPtcDataProvider();

    void setupBackendDataProvider();                                                      // implementation of public base class virtual interface
	void resetBackendDataProvider();                                                      // implementation of public base class virtual interface
    bool frameMatching(const float *frameOutput, const float *frameLabel, int len);       // implementation of public base class virtual interface

    // The following two interfaces are only used by the CheckPointing Function
    void getCheckPointFrame(int & frameNo);                                   // implementation of public base class virtual interface
    void setupBackendDataProvider(int startFrameNo, bool doChkPointing);      // implementation of public base class virtual interface

private:
	 void prepare_batch_data();                 // implementation of private base class virtual interface
	 bool haveBatchToProvide();                 // implementation of private base class virtual interface

	 void setup_first_data_batches();            // first time read data from the file and setup them on the memory
	 void setup_cont_data_batches();             // continue to read data from the file and setup them on the memory
	 void shuffle_data(int *index, int len);

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
