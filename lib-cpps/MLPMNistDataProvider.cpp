/*
 *  COPYRIGHT:  Copyright (c) 2014 Advanced Micro Devices, Inc.  All rights reserved
 *
 *   Written by Qianfeng Zhang@amd.com ( March 2014 )
 *
 */

#include <algorithm>
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

	string trainfname(dataPath);
    string labelfname(dataPath);

 	if ( this->dataMode == MLP_DATAMODE_TRAIN )
 	     trainfname.append("train-images.idx3-ubyte");
	else
         trainfname.append("t10k-images.idx3-ubyte");         // for TEST or PREDICT

	this->dataFile.open(trainfname.c_str(),ios_base::in|ios_base::binary);

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
		this->m_shuffleBatches = 50;          // for testing and predicting, we don't need to shuffle the data
	else
	    this->m_shuffleBatches = 1;

    this->InitializeFromMNistSource(MNIST_PATH);

	this->total_batches = ROUNDK(DIVUPK(this->num_frames,this->m_batchSize),this->m_shuffleBatches);
	this->endOfDataSource = false;
	this->batches_loaded = false;
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

	this->total_batches = ROUNDK(DIVUPK(this->num_frames,this->m_batchSize),this->m_shuffleBatches);
	this->endOfDataSource = false;
	this->batches_loaded  = false;
};

MLPMNistDataProvider::~MLPMNistDataProvider()
{
    MLP_CHECK(this->shutdown_worker());

	if ( this->dataFile.is_open() )
	     this->dataFile.close();
	if ( this->labelFile.is_open() )
	     this->labelFile.close();

	delete [] this->permutations;
	delete [] this->featureData;
	if ( this->haveLabel )
	     delete [] this->labelData;

	this->release_buffers();
};


//////////////////////////////////////////////////////////////////////////////////////
////                          private member functions                            ////
//////////////////////////////////////////////////////////////////////////////////////


void MLPMNistDataProvider::prepare_batch_data()
{
	if ( this->supportChkPointing)
	     MLP_LOCK(&this->chkPointingLock);

	// transfer one batch from the data source to the Double-buffer
	this->load_feature_batch(this->featureData,this->permutations,this->m_batchSize*(this->batchNo % this->m_shuffleBatches));
	if ( this->haveLabel )
	     this->load_label_batch(this->labelData,this->permutations,this->m_batchSize*(this->batchNo % this->m_shuffleBatches));

	this->batchNo++;
	this->stageBatchNo++;

	if ( this->supportChkPointing )
		 MLP_UNLOCK(&this->chkPointingLock);

	if ( this->stageBatchNo == this->m_shuffleBatches ) {   

		  this->batches_loaded = false; 

		  if ( !this->endOfDataSource ) 
		        this->setup_cont_data_batches();

		  if ( this->batches_loaded ) {
		       this->shuffle_data(this->permutations, this->m_batchSize * this->m_shuffleBatches );
		       // cout << "Load new data from files and shuffling the frame sequences" << endl;
		  }; 

		  this->stageBatchNo = 0;
	};
};

bool MLPMNistDataProvider::haveBatchToProvide()
{
	if ( !this->endOfDataSource )
		 return(true);

    if ( this->batches_loaded )
		 return(true);

	return(false);
};


// set up the data source of MLPMNistDataProvider
void MLPMNistDataProvider::setup_first_data_batches()
{
	this->stageBatchNo = 0;
	this->setup_cont_data_batches();

    if ( this->batches_loaded ) {
	     this->shuffle_data(this->permutations, this->m_batchSize * this->m_shuffleBatches );
	     //cout << "Load new data from files and shuffling the frame sequences" << endl;
    }; 
};

void MLPMNistDataProvider::setup_cont_data_batches()
{
	int readCount=0;
	int frame; 

	// initial permutations, permutated each round
	for (int k=0; k < this->m_batchSize * this->m_shuffleBatches; k++)
		    this->permutations[k] = k;

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

endf:    // duplicate the first "readCount" records to minibatch*shuffleBatches records

    if ( readCount > 0 ) {
	     int dst=readCount;
		 int src=0;

		 // setup a new stage of batches using the left frames
		 while ( dst < this->m_batchSize * this->m_shuffleBatches ) {
				src = dst % readCount;

		        for (int i=0; i < this->m_dataFeatureSize; i++)
			           this->featureData[dst*this->m_dataFeatureSize+i] = this->featureData[src*this->m_dataFeatureSize+i];

				if ( this->haveLabel )
		             for (int i=0; i < this->m_dataLabelSize; i++)
				        this->labelData[dst*this->m_dataLabelSize+i] = this->labelData[src*this->m_dataLabelSize+i];

				dst++;
		  };
	 }

	 if ( frame > 0  || readCount > 0 ) 
		  this->batches_loaded = true; 

	 delete [] imagebuf;
};

void MLPMNistDataProvider::shuffle_data(int *index, int len)
{
	std::random_shuffle(index, index+len);
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

void MLPMNistDataProvider::setupDataProvider()
{
	this->gotoDataFrame(0);
	if ( this->haveLabel )
		 this->gotoLabelFrame(0);

	this->batchNo = 0;

	this->supportChkPointing = false;

    // allocate batches buffer
	this->permutations = new int[this->m_batchSize * this->m_shuffleBatches];
	this->featureData  = new float[this->m_batchSize * this->m_shuffleBatches * this->m_dataFeatureSize];
	if ( this->haveLabel )
         this->labelData = new float[this->m_batchSize * this->m_shuffleBatches * this->m_dataLabelSize];

	this->setup_first_data_batches();

	this->create_buffers(this->m_batchSize);

	this->initialized = true;

	MLP_CHECK(this->startup_worker());
};

void MLPMNistDataProvider::setupDataProvider(int startFrameNo, bool doChkPointing)
{
	if ( this->dataMode != MLP_DATAMODE_TRAIN ) {
		 mlp_log("MLPMNistDataProvider", "This interface can only be called with the TRAIN mode");
		 MLP_Exception("");
	};

	this->gotoDataFrame(startFrameNo);
    this->gotoLabelFrame(startFrameNo);

	this->batchNo = (startFrameNo / this->m_batchSize);  // The new "batchNo" will start from this one

	// For CheckPoint
	this->supportChkPointing = doChkPointing;
	if ( this->supportChkPointing )
		 MLP_LOCK_INIT(&this->chkPointingLock);

    // allocate batches buffer
	this->permutations = new int[this->m_batchSize * this->m_shuffleBatches];
	this->featureData  = new float[this->m_batchSize * this->m_shuffleBatches * this->m_dataFeatureSize];
	if ( this->haveLabel )
         this->labelData = new float[this->m_batchSize * this->m_shuffleBatches * this->m_dataLabelSize];

	this->setup_first_data_batches();

	this->create_buffers(this->m_batchSize);

	this->initialized = true;

	MLP_CHECK(this->startup_worker());
};


void MLPMNistDataProvider::resetDataProvider()
{
	if ( !this->initialized ) {
		 mlp_log("MLPMNistDataProvider", "The DataProvider is still not started yet, no reset should be called");
		 MLP_Exception("");
	};
	MLP_CHECK(this->shutdown_worker());

	this->reset_buffers();

    this->endOfDataSource = false;
	this->batchNo = 0;
	this->batches_loaded = false;

    this->dataFile.clear();
	this->gotoDataFrame(0);
	if ( this->haveLabel ) {
	     this->labelFile.clear();
		 this->gotoLabelFrame(0);
    };

	if ( this->supportChkPointing )
		 MLP_LOCK_INIT(&this->chkPointingLock);

	this->setup_first_data_batches();

	MLP_CHECK(this->startup_worker());
};


void MLPMNistDataProvider::getCheckPointFrame(int & frameNo)
{
    int batch;
	int stage;

	MLP_LOCK(&this->chkPointingLock);

	batch = this->batchNo - MLP_BATCH_RING_SIZE;    // This is the batchNo for the sequence of shuffled batche,  Consider there are batches on the buffer ring (may not be processed by the Trainer)

	stage = batch / this->m_shuffleBatches;

	// We will start from the first frame of the "stage", since frames before this "stage" have been processed
	frameNo = stage * this->m_shuffleBatches * this->m_batchSize;

	MLP_UNLOCK(&this->chkPointingLock);
};



// if the output for the frame matches its label, return true to indicate a successful mapping of this
// frame by the neural network.  This interface will be called by the MLPTester class when calculating
// the success ratio of the neural network on this type of data
bool MLPMNistDataProvider::frameMatching(float *frameOutput, float *frameLabel, int len)
{
	float element;

	for (int i=0; i< len; i++) {
		 element = (frameOutput[i]<0.5)?0.0f:1.0f;

		 if ( element != frameLabel[i] )
			  return(false);
	};

	return(true);
};
