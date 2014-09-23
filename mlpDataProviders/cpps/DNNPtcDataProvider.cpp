/*
 *  COPYRIGHT:  Copyright (c) 2014 Advanced Micro Devices, Inc.  All rights reserved
 *
 *   Written by Qianfeng Zhang@amd.com ( March 2014 )
 *
 */

#include <iostream>

#include "DNNUtil.h"
#include "DNNPtcDataProvider.h"
#include "conv_endian.h"

//////////////////////////////////////////////////////////////////////////////////////
////                          constructors and destructor                         ////
//////////////////////////////////////////////////////////////////////////////////////

// only called by the constructors
void DNNPtcDataProvider::InitializeFromPtcSource(const char *dataPath)
{
	unsigned long filelen;
	unsigned long expectlen;

	string datafname(dataPath);

	if ( this->dataMode == DNN_DATAMODE_TRAIN )
 	     datafname.append("ptc_training_db.dat");
	else
		 datafname.append("ptc_testing_db.dat");

	this->dataFile.open(datafname.c_str(),ios_base::in|ios_base::binary);

	if ( ! this->dataFile.is_open() ) {
		   dnn_log("DNNPtc", "Failed to open PTC data file");
		   DNN_Exception("");
	};

	struct ptc_db_header header;

	this->dataFile.seekg(16);    // 16 bytes reserved for checksum
	this->dataFile.read(reinterpret_cast<char*>(&header),sizeof(header));

	LEtoHostl(header.numChars);
	LEtoHostl(header.numSamples);
	LEtoHostl(header.sWidth);
	LEtoHostl(header.sHeight);

	if ( header.tag[0] != 'P' || header.tag[1] != 'T' || header.tag[2] != 'C' || header.tag[3] != '!' ) {
		 dnn_log("DNNPtc", "Incorrect PTC data set file tag is detected, the data set file may be not correct one");
		 DNN_Exception("");
	};
	this->m_dataFeatureSize = header.sWidth * header.sHeight;

	this->dataFile.seekg(0,ios_base::end);
	filelen = (unsigned long)dataFile.tellg();
    dataFile.seekg(16+sizeof(header));

	expectlen = 16 + sizeof(struct ptc_db_header) + header.numSamples * (sizeof(struct ptc_sample_header) + header.sWidth*header.sHeight);
	if ( filelen != expectlen )  {
		 dnn_log("DNNPtc", "The ptc file length is inconsistent with the number of samples recorded in file header");
		 DNN_Exception("");
	};

	this->m_dataLabelSize = header.numChars;

	this->num_frames = header.numSamples;

	this->imageWidth = header.sWidth;
	this->imageHeight = header.sHeight;
};

DNNPtcDataProvider::DNNPtcDataProvider()
{
	this->dataMode = DNN_DATAMODE_TRAIN;
	this->haveLabel = true;
 	this->m_batchSize = 512;

	if ( this->dataMode == DNN_DATAMODE_TRAIN )
		this->m_shuffleBatches = 10;          // for testing and predicting, we don't need to shuffle the data
	else
	    this->m_shuffleBatches = 1;

    this->InitializeFromPtcSource(PTC_DB_PATH);

	this->total_batches = DIVUPK(this->num_frames,this->m_batchSize);
};

DNNPtcDataProvider::DNNPtcDataProvider(const char *dataPath, DNN_DATA_MODE mode, int batchSize, int shuffleBatches)
{
	if ( (mode < 0) || (mode >= DNN_DATAMODE_ERROR) ) {
		  dnn_log("DNNPtcDataProvider", "Data mode for constructing DNNPtcDataProvider is not correct");
		  DNN_Exception("");
	};

	this->dataMode = mode;
	this->haveLabel = (mode==DNN_DATAMODE_PREDICT)?false:true;
	this->m_batchSize = batchSize;

	if ( this->dataMode == DNN_DATAMODE_TRAIN )
		 this->m_shuffleBatches = shuffleBatches;    // for testing and predicting, we don't need to shuffle the data
	else
	     this->m_shuffleBatches = 1;

	this->InitializeFromPtcSource(dataPath);

	this->total_batches = DIVUPK(this->num_frames,this->m_batchSize);
};

DNNPtcDataProvider::~DNNPtcDataProvider()
{
    DNN_CHECK(this->shutdown_worker());

	if ( this->dataFile.is_open() )
	     this->dataFile.close();

	this->release_io_buffers();
	this->release_transfer_buffers();
};


// set up the data source of DNNPtcDataProvider
void DNNPtcDataProvider::setup_first_data_batches()
{
	this->stageBatchNo = 0;
	this->setup_cont_data_batches();

    if ( this->batches_loaded ) {
	     this->shuffle_data(this->permutations, this->m_batchSize * this->batches_loaded );
    };
};

void DNNPtcDataProvider::setup_cont_data_batches()
{
	int readCount=0;
	int frame;

	unsigned char *imagebuf;
	unsigned short label;

	imagebuf = new unsigned char[this->m_dataFeatureSize];
    for (frame=0; frame < this->m_batchSize * this->m_shuffleBatches; frame++) {  // read the data frame by frame
		 struct ptc_sample_header sheader;

		 this->dataFile.read(reinterpret_cast<char*>(&sheader),sizeof(struct ptc_sample_header));
         this->dataFile.read(reinterpret_cast<char*>(imagebuf),this->m_dataFeatureSize);

		 if ( this->dataFile.eof() ) {
			  readCount = frame;
		      this->endOfDataSource = true;   // no data can read from the data source any more
			  goto endf;
		 };

		 if ( this->dataFile.fail() ) {
			  dnn_log("DNNPtcDataProvider", "Failed to access data set file");
			  DNN_Exception("");
		 };

		 LEtoHosts(sheader.wCode);
		 LEtoHosts(sheader.index);
		 label = sheader.index;

		 if ( this->haveLabel ) {
		      if ( ! ( label>=0 && label < this->m_dataLabelSize ) ) {
				   dnn_log("DNNPtcDataProvider", "label value from the label file is not correct");
				   DNN_Exception("");
			  };
		 };

		 for (int i=0; i < this->m_dataFeatureSize; i++)
			  this->featureData[frame*this->m_dataFeatureSize+i] = (float)imagebuf[i]/255.0f-0.5f;

		 if ( this->haveLabel ) {
		      for (int i=0; i < this->m_dataLabelSize; i++)
		           this->labelData[frame*this->m_dataLabelSize+i] = 0.0f;
	          this->labelData[frame*this->m_dataLabelSize+(int)label] = 1.0f;
		 };
	};

endf:

	if ( readCount % this->m_batchSize > 0 ) {  // not one complete batch of frames are loaded
	     int dst=readCount;
		 int batches;
		 int src=0;

		 batches = readCount/this->m_batchSize + 1;

		 // replicate to fill the left frame in last batch
		 while ( dst < this->m_batchSize * batches ) {
				src = dst % readCount;

		        for (int i=0; i < this->m_dataFeatureSize; i++)
			           this->featureData[dst*this->m_dataFeatureSize+i] = this->featureData[src*this->m_dataFeatureSize+i];

				if ( this->haveLabel )
		             for (int i=0; i < this->m_dataLabelSize; i++)
				        this->labelData[dst*this->m_dataLabelSize+i] = this->labelData[src*this->m_dataLabelSize+i];

				dst++;
		  };
		  this->batches_loaded = batches;
	 }
 	 else {
		  this->batches_loaded = (readCount == 0)? this->m_shuffleBatches: (readCount/this->m_batchSize);
	 };

	 this->batchNo += this->batches_loaded;

	 delete [] imagebuf;
};


void DNNPtcDataProvider::gotoDataFrame(int frameNo)
{
	this->dataFile.seekg(16 + sizeof(struct ptc_db_header));

	for (int i=0; i< frameNo/400000; i++)
		this->dataFile.seekg(400000*(sizeof(struct ptc_sample_header) + this->m_dataFeatureSize ), ios_base::cur);

	this->dataFile.seekg((frameNo%400000)*(sizeof(struct ptc_sample_header) + this->m_dataFeatureSize ), ios_base::cur);
};


//////////////////////////////////////////////////////////////////////////////////////
////                          public member functions                             ////
//////////////////////////////////////////////////////////////////////////////////////

void DNNPtcDataProvider::setupBackendDataProvider()
{
	this->gotoDataFrame(0);

	this->batchNo = 0;

	this->setup_first_data_batches();
};

void DNNPtcDataProvider::setupBackendDataProvider(int startFrameNo, bool doChkPointing)
{
	this->gotoDataFrame((startFrameNo/this->m_batchSize)*this->m_batchSize);

	this->batchNo = (startFrameNo / this->m_batchSize);  // The new "batchNo" will start from this one

	this->setup_first_data_batches();
};


void DNNPtcDataProvider::resetBackendDataProvider()
{
	this->batchNo = 0;

    this->dataFile.clear();
	this->gotoDataFrame(0);

	this->setup_first_data_batches();
};


void DNNPtcDataProvider::getCheckPointFrame(int & frameNo)
{
    int batch;

	DNN_LOCK(&this->chkPointingLock);

    // get the latest batch for which we are sure having been processed,  Consider there are batches on
	// the transfer and io buffer that may not be processed by the Trainer
	batch = this->batchNo - (this->batches_loaded - this->stageBatchNo) - DNN_BATCH_RING_SIZE;

	// We will start from the first frame of the "stage", since frames before this "stage" have been processed
	frameNo = batch * this->m_batchSize;

	DNN_UNLOCK(&this->chkPointingLock);
};



// if the output for the frame matches its label, return true to indicate a successful mapping of this
// frame by the neural network.  This interface will be called by the DNNTester class when calculating
// the success ratio of the neural network on this type of data
bool DNNPtcDataProvider::frameMatching(const float *frameOutput, const float *frameLabel, int len)
{
	float element;

	for (int i=0; i< len; i++) {
		 element = (frameOutput[i]<0.5)?0.0f:1.0f;

		 if ( element != frameLabel[i] )
			  return(false);
	};

	return(true);
};
