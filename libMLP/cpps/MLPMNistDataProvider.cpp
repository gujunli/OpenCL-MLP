/*
 *  COPYRIGHT:  Copyright (c) 2014 Advanced Micro Devices, Inc.  All rights reserved
 *
 *   Written by Qianfeng Zhang@amd.com ( March 2014 )
 *
 */

#include <iostream>

#include "MLPUtil.h"
#include "MLPMNistDataProvider.h"

//////////////////////////////////////////////////////////////////////////////////////
////                          constructors and destructor                         ////
//////////////////////////////////////////////////////////////////////////////////////

// only called by the constructors
void MLPMNistDataProvider::InitializeFromMNistSource(const char *dataPath)
{
	int filelen;
	int records;

	string datafname(dataPath);
    string labelfname(dataPath);

 	if ( this->dataMode == MLP_DATAMODE_TRAIN )
 	     datafname.append("train-images.idx3-ubyte");
	else
         datafname.append("t10k-images.idx3-ubyte");         // for TEST or PREDICT

	this->dataFile.open(datafname.c_str(),ios_base::in|ios_base::binary);

	if ( ! this->dataFile.is_open() ) {
		   mlp_log("MLPMNist", "Failed to open MNIST data file");
		   MLP_Exception("");
	};

	struct header_imagefile header1;

	this->dataFile.read(reinterpret_cast<char*>(&header1),sizeof(header1));

	BEtoHostl(header1.magicNum);
	BEtoHostl(header1.numImages);
	BEtoHostl(header1.imageHeight);
	BEtoHostl(header1.imageWidth);

	if ( header1.magicNum != 0x0803 ) {
		 mlp_log("MLPMNist", "MNIST data file may be not correct");
		 MLP_Exception("");
	};
	this->m_dataFeatureSize = header1.imageWidth * header1.imageHeight;

	this->dataFile.seekg(0,ios_base::end);
	filelen = (int)dataFile.tellg();
    dataFile.seekg(sizeof(header1));

	records = ((filelen-sizeof(header1))/sizeof(unsigned char))/(header1.imageWidth*header1.imageHeight);
	if ( records < (int) header1.numImages )  {
		 mlp_log("MLPMNist", "MNIST TrainData file may be not correct");
		 MLP_Exception("");
	};

	if ( !this->haveLabel )
		 goto finish;

 	if ( this->dataMode == MLP_DATAMODE_TRAIN )
 	     labelfname.append("train-labels.idx1-ubyte");
	else
         labelfname.append("t10k-labels.idx1-ubyte");         // for TEST or PREDICT

	this->labelFile.open(labelfname.c_str(),ios_base::in|ios_base::binary);
	if ( ! this->labelFile.is_open() ) {
		   mlp_log("MLPMNist", "Failed to open MNIST label file");
		   MLP_Exception("");
	};

	struct header_labelfile header2;

    this->labelFile.read(reinterpret_cast<char*>(&header2),sizeof(header2));

	BEtoHostl(header2.magicNum);
	BEtoHostl(header2.numLabels);

	if ( header2.magicNum != 0x0801 ) {
		 mlp_log("MLPMNist", "MNIST label file may be not correct");
		 MLP_Exception("");
	};
	this->m_dataLabelSize = 10;

	this->labelFile.seekg(0,ios_base::end);
	filelen = (int)labelFile.tellg();
    labelFile.seekg(sizeof(header2));

	records = (filelen-sizeof(header2))/sizeof(unsigned char);
    if (  records < (int)header2.numLabels ) {
		 mlp_log("MLPMNist", "MNIST TrainLabel file may be not correct");
		 MLP_Exception("");
	};

	if ( header1.numImages != header2.numLabels ) {
		 mlp_log("MLPMNist", "MNIST data is not correct");
		 MLP_Exception("");
	};

finish:
	this->num_frames = header1.numImages;

	this->imageWidth = header1.imageWidth;
	this->imageHeight = header1.imageHeight;
};

MLPMNistDataProvider::MLPMNistDataProvider()
{
	this->dataMode = MLP_DATAMODE_TRAIN;
	this->haveLabel = true;
 	this->m_batchSize = 512;

	if ( this->dataMode == MLP_DATAMODE_TRAIN )
		this->m_shuffleBatches = 10;          // for testing and predicting, we don't need to shuffle the data
	else
	    this->m_shuffleBatches = 1;

    this->InitializeFromMNistSource(MNIST_PATH);

	this->total_batches = DIVUPK(this->num_frames,this->m_batchSize);
};

MLPMNistDataProvider::MLPMNistDataProvider(const char *dataPath, MLP_DATA_MODE mode, int batchSize, int shuffleBatches)
{
	if ( (mode < 0) || (mode >= MLP_DATAMODE_ERROR) ) {
		  mlp_log("MLPMNistDataProvider", "Data mode for constructing MLPMNistDataProvider is not correct");
		  MLP_Exception("");
	};

	this->dataMode = mode;
	this->haveLabel = (mode==MLP_DATAMODE_PREDICT)?false:true;
	this->m_batchSize = batchSize;

	if ( this->dataMode == MLP_DATAMODE_TRAIN )
		 this->m_shuffleBatches = shuffleBatches;    // for testing and predicting, we don't need to shuffle the data
	else
	     this->m_shuffleBatches = 1;

	this->InitializeFromMNistSource(dataPath);

	this->total_batches = DIVUPK(this->num_frames,this->m_batchSize);
};

MLPMNistDataProvider::~MLPMNistDataProvider()
{
    MLP_CHECK(this->shutdown_worker());

	if ( this->dataFile.is_open() )
	     this->dataFile.close();
	if ( this->labelFile.is_open() )
	     this->labelFile.close();

	this->release_io_buffers(); 
	this->release_transfer_buffers();
};


// set up the data source of MLPMNistDataProvider
void MLPMNistDataProvider::setup_first_data_batches()
{
	this->stageBatchNo = 0;
	this->setup_cont_data_batches();

    if ( this->batches_loaded ) {
	     this->shuffle_data(this->permutations, this->m_batchSize * this->batches_loaded );
    };
};

void MLPMNistDataProvider::setup_cont_data_batches()
{
	int readCount=0;
	int frame;

	unsigned char *imagebuf;
	unsigned char label;

	imagebuf = new unsigned char[this->m_dataFeatureSize];
    for (frame=0; frame < this->m_batchSize * this->m_shuffleBatches; frame++) {  // read the data frame by frame

         this->dataFile.read(reinterpret_cast<char*>(imagebuf),this->m_dataFeatureSize);
		 if ( this->haveLabel )
	          this->labelFile.read((char*)&label,1);

		 if ( this->dataFile.eof() || this->labelFile.eof() ) {
			  readCount = frame;
		      this->endOfDataSource = true;   // no data can read from the data source any more
			  goto endf;
		 };

		 if ( this->dataFile.fail() ) {
			  mlp_log("MLPMNistDataProvider", "Failed to access feature data");
			  MLP_Exception("");
		 };

		 if ( this->haveLabel ) {
		      if ( this->labelFile.fail() ) {
			       mlp_log("MLPMNistDataProvider", "Failed to access feature data");
			       MLP_Exception("");
		      };

		      if ( ! (label>=0 && label <=9 ) ) {
				   mlp_log("MLPMNistDataProvider", "label value from the label file is not correct");
				   MLP_Exception("");
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


void MLPMNistDataProvider::gotoDataFrame(int frameNo)
{
	this->dataFile.seekg(sizeof(struct header_imagefile));
	this->dataFile.seekg(frameNo * this->m_dataFeatureSize, ios_base::cur);
};


void MLPMNistDataProvider::gotoLabelFrame(int frameNo)
{
	this->labelFile.seekg(sizeof(struct header_labelfile));
	this->labelFile.seekg(frameNo * 1, ios_base::cur);
};

//////////////////////////////////////////////////////////////////////////////////////
////                          public member functions                             ////
//////////////////////////////////////////////////////////////////////////////////////

void MLPMNistDataProvider::setupBackendDataProvider()
{
	this->gotoDataFrame(0);
	if ( this->haveLabel )
		 this->gotoLabelFrame(0);

	this->batchNo = 0;

	this->setup_first_data_batches();
};

void MLPMNistDataProvider::setupBackendDataProvider(int startFrameNo, bool doChkPointing)
{
	this->gotoDataFrame((startFrameNo/this->m_batchSize)*this->m_batchSize);
	this->gotoLabelFrame((startFrameNo/this->m_batchSize)*this->m_batchSize);

	this->batchNo = (startFrameNo / this->m_batchSize);  

	this->setup_first_data_batches();
};


void MLPMNistDataProvider::resetBackendDataProvider()
{
	this->batchNo = 0;

    this->dataFile.clear();
	this->gotoDataFrame(0);
	if ( this->haveLabel ) {
	     this->labelFile.clear();
		 this->gotoLabelFrame(0);
    };

	this->setup_first_data_batches();
};


void MLPMNistDataProvider::getCheckPointFrame(int & frameNo)
{
    int batch;

	MLP_LOCK(&this->chkPointingLock);

    // get the latest batch for which we are sure having been processed,  Consider there are batches on
	// the transfer and io buffer that may not be processed by the Trainer
	batch = this->batchNo - (this->batches_loaded - this->stageBatchNo) - MLP_BATCH_RING_SIZE;    

	// We will start from the first frame of the "stage", since frames before this "stage" have been processed
	frameNo = batch * this->m_batchSize;

	MLP_UNLOCK(&this->chkPointingLock);
};



// if the output for the frame matches its label, return true to indicate a successful mapping of this
// frame by the neural network.  This interface will be called by the MLPTester class when calculating
// the success ratio of the neural network on this type of data
bool MLPMNistDataProvider::frameMatching(const float *frameOutput, const float *frameLabel, int len)
{
	float element;

	for (int i=0; i< len; i++) {
		 element = (frameOutput[i]<0.5)?0.0f:1.0f;

		 if ( element != frameLabel[i] )
			  return(false);
	};

	return(true);
};
