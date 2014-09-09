/*
 *  COPYRIGHT:  Copyright (c) 2014 Advanced Micro Devices, Inc.  All rights reserved
 *
 *   Written by Qianfeng Zhang@amd.com ( March 2014 )
 *
 */


#ifndef _MLP_IFLY_DATA_PROVIDER_H_
#define _MLP_IFLY_DATA_PROVIDER_H_

#include <vector>
#include <fstream>

#include "MLPApiExport.h"
#include "MLPDataProvider.h"

using namespace std;

// for voice data from Iflytek
class MLPIFlyDataProvider:public MLPDataProvider
{
private:
	ifstream dataFile;
	ifstream labelFile;

private:
	int dataFrameLen;          // Length of the raw data frame,  eg. 39 floats
	int labelFrameLen;         // Length of the raw label frame, eg. 1  int

	int numSentences;          // Number of sentences available in the total dataset file
	int numFrames;             // Number of frames available in the total dataset file
	int TestSetStart;          // Starting frame number of the TestDataSet, TestDataSet takes appr. 5% of the total dataset, it is also the end of the TrainDataSet

    int mySetStart;            // Starting frame number of the currently used dataset (TrainDataSet or TestDataSet)
	int mySetFrames;           // Total number of frames for TrainDataSet or TestDataSet we are current using, one file is splitted into training set and testing set

	int curSentence;           // Sentence ID of the frames we are currently accessing
	int curFrame;              // Global frame sequence number of the frame we are currently accessin
	int curStartFrame;         // First frame of the sentence currently accessed

	// The following three members are only used by the CheckPointing Function
	int curChkPointFrame;      // First frame of the sentence when we call setup_cont_data_source() currently, this is used as a checkpointing position
	int lastChkPointFrame;     // First frame of the sentence when we call setup_cont_data_source() last time, this is used as a checkpointing position

	vector<float *> sDataFrames;       //  Vector to store all data frames of one sentence read from the file
	vector<int> sLabelFrames;          //  Vector to store all label frames of one sentence read from the file
	int frameIndex;                    //  Index in the vector, of the frame we are going to working on

	vector<float> mean_v;              //  Vector to store the means calculated from all frames of the dataset,  used to normalize the raw input frames
	vector<float> covariance_v;        //  Vector to store the covariance calculated from all frames of the dataset, used to normalize the raw input frames

public:
	LIBMLPAPI MLPIFlyDataProvider();
	LIBMLPAPI MLPIFlyDataProvider(const char *dataPath, MLP_DATA_MODE mode, int batchSize, int shuffleBatches);

    ~MLPIFlyDataProvider();

    void setupBackendDataProvider();                                                   // Implementation of public base class virtual interface
	void resetBackendDataProvider();                                                   // Implementation of public base class virtual interface
    bool frameMatching(const float *frameOutput, const float *frameLabel, int len);    // Implementation of public base class virtual interface

    // The following two interfaces are only used by the CheckPointing Function
    void getCheckPointFrame(int & frameNo);                                // Implementation of public base class virtual interface
    void setupBackendDataProvider(int startFrameNo, bool doChkPointing);   // Implementation of public base class virtual interface

private:
	 void setup_first_data_batches();            // first time read a group of batches from the source and setup them on the io buffers
	 void setup_cont_data_batches();             // read a group of batches from the source and setup them on the io buffers

     void InitializeFromIFlySource(const char *dataPath);
	 void gotoDataFrame(int frameNo);
	 void gotoLabelFrame(int frameNo);
	 void readOneSentence();
};

#ifdef  _WIN32
#define IFLY_PATH "../../IFLYTEK/87/config/"
#else
#define IFLY_PATH "../../IFLYTEK/87/config/"
#endif

#define PHEADER_SIZE 32768
#define PCHKSUM_LEN  4

#endif
