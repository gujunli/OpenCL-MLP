/*
 *  COPYRIGHT:  Copyright (c) 2014 Advanced Micro Devices, Inc.  All rights reserved
 *
 *   Written by Qianfeng Zhang@amd.com ( March 2014 )
 *
 */


#include <algorithm>
#include <iostream>

#include "MLPUtil.h"
#include "MLPIFlyDataProvider.h"

using namespace std;

//////////////////////////////////////////////////////////////////////////////////////
////                          constructors and destructor                         ////
//////////////////////////////////////////////////////////////////////////////////////

// only called by the constructors
static void check_pfile_header(ifstream &pfile, int &numSentences, int &numFrames, int &lenFeature, int &lenLabel)
{
    string *lines = new string[30];
    int lineCount=0;

	numSentences = 0;
	numFrames = 0;
	lenFeature = 0;
	lenLabel = 0;

	pfile.seekg(0,ios_base::beg);
    while ( (!pfile.eof()) && ((int)pfile.tellg() < 1024) ) {
          string myline;

          getline(pfile,myline);

          if ( !myline.empty() ) {
               lines[lineCount] = myline;    // assignment of string
               lineCount++;
          }
		  else
			   break;
    };

	string key1("-num_sentences");
    for (int i=0; i< lineCount; i++) {
          if ( lines[i].compare(0,key1.length(),key1) == 0 ) {
                istringstream mystream(lines[i].substr(key1.length()+1));

                mystream >> numSentences;
                break;
          };
    };

	string key2("-num_frames");
    for (int i=0; i< lineCount; i++) {
          if ( lines[i].compare(0,key2.length(),key2) == 0 ) {
                istringstream mystream(lines[i].substr(key2.length()+1));

                mystream >> numFrames;
                break;
          };
    };

	string key3("-num_features");
    for (int i=0; i< lineCount; i++) {
          if ( lines[i].compare(0,key3.length(),key3) == 0 ) {
                istringstream mystream(lines[i].substr(key3.length()+1));

                mystream >> lenFeature;
                break;
          };
    };

	string key4("-num_labels");
    for (int i=0; i< lineCount; i++) {
          if ( lines[i].compare(0,key4.length(),key4) == 0 ) {
                istringstream mystream(lines[i].substr(key4.length()+1));

                mystream >> lenLabel;
                break;
          };
    };

	delete [] lines;
};

// only called by the constructor

/*
static void getMaxLabelValue(ifstream &labfile, int numFrames, unsigned int &maxSentence, unsigned int &minSentence, unsigned int &maxFrame,
	                        unsigned int &minFrame, unsigned int &maxLabel, unsigned int &minLabel)
{
	unsigned int frame[3];

	maxSentence = 0;
	minSentence = 20000000;
	maxFrame = 0;
	minFrame = 20000000;
	maxLabel = 0;
	minLabel = 20000000;

	labfile.seekg(PHEADER_SIZE);
	for (int i=0; i< numFrames; i++) {
	     labfile.read(reinterpret_cast<char*>(&frame[0]),sizeof(frame));

		 BEtoHostl(frame[0]);
		 BEtoHostl(frame[1]);
		 BEtoHostl(frame[2]);

		 if ( frame[0] > maxSentence )
			  maxSentence = frame[0];
		 if ( frame[0] < minSentence )
			  minSentence = frame[0];

		 if ( frame[1] > maxFrame )
			  maxFrame = frame[1];
		 if ( frame[1] < minFrame )
			  minFrame = frame[1];

		 if ( frame[2] > maxLabel )
			  maxLabel = frame[2];
		 if ( frame[2] < minLabel )
			  minLabel = frame[2];
	};
};
*/

// only called by the constructor
void MLPIFlyDataProvider::InitializeFromIFlySource(const char *dataPath)
{
	unsigned long filelen;
	unsigned long expectlen;    // expected len of the file

	string trainfname(dataPath);
    string labelfname(dataPath);

	// read information from the header of "plp.file" and check

 	trainfname.append("plp.pfile");

	this->dataFile.open(trainfname.c_str(),fstream::in|fstream::binary);
    if ( ! this->dataFile.is_open() ) {
		   mlp_log("MLPIFlyDataProvider", "Failed to open the file plp.pfile");
		   MLP_Exception("");
	};
	int numSentences1, numFrames1, lenFeature1, lenLabel1;

	check_pfile_header(this->dataFile, numSentences1, numFrames1, lenFeature1, lenLabel1);

	this->dataFile.seekg(0,fstream::end);
	filelen = (unsigned long)this->dataFile.tellg();
	expectlen = PHEADER_SIZE + numFrames1*(lenFeature1+2)*sizeof(float) + PCHKSUM_LEN + numSentences1*sizeof(int);
	if ( (filelen <= PHEADER_SIZE) || (filelen != expectlen) || (lenLabel1 !=0)  ) {
		 mlp_log("MLPIFlyDataProvider", "The plp.file seems not correct");
		 MLP_Exception("");
	};

	this->numSentences = numSentences1;
	this->numFrames = numFrames1;
	this->dataFrameLen = lenFeature1;

    // read information from the header of "lab.file" and check

    labelfname.append("lab.pfile");

	this->labelFile.open(labelfname.c_str(),fstream::in|fstream::binary);
    if ( ! this->dataFile.is_open() ) {
		   mlp_log("MLPIFlyDataProvider", "Failed to open the file lab.pfile");
		   MLP_Exception("");
	};

	if ( this->haveLabel ) {
	     int numSentences2, numFrames2, lenFeature2, lenLabel2;

	     check_pfile_header(this->labelFile, numSentences2, numFrames2, lenFeature2, lenLabel2);
	     this->labelFile.seekg(0,fstream::end);
	     filelen = (unsigned long)this->labelFile.tellg();
	     expectlen = PHEADER_SIZE + numFrames1*(lenLabel2+2)*sizeof(int) + PCHKSUM_LEN + numSentences2*sizeof(int);
         if ( (filelen <= PHEADER_SIZE) || (filelen != expectlen) || (lenFeature2 !=0)  ) {
		       mlp_log("MLPIFlyDataProvider", "The lab.file seems not correct");
		       MLP_Exception("");
	     };

	     // combined check
	     if ( (numSentences1 != numSentences2) || (numFrames1 != numFrames2) ) {
		       mlp_log("MLPIFlyDataProvider", "The IFly data files seem not correct");
		       MLP_Exception("");
	     };

	     this->labelFrameLen = lenLabel2;
	};

	/*
    unsigned int maxSentence, minSentence, maxFrame, minFrame, maxLabel, minLabel;
	getMaxLabelValue(this->labelFile, this->numFrames, maxSentence, minSentence, maxFrame, minFrame, maxLabel, minLabel);
	*/

	this->m_dataFeatureSize = this->dataFrameLen * 11;
	this->m_dataLabelSize = 8991;     // maxValue=8990 detected

	int frameNo;
	unsigned int frameHeader[2];
	unsigned int sentence1, sentence2;

	frameNo = (int) ( (float)this->numFrames * 0.95f );   // estimated starting frame number of the testing dataset

	this->gotoDataFrame(frameNo);
    if ( this->haveLabel )
		this->gotoLabelFrame(frameNo);

    this->dataFile.read(reinterpret_cast<char*>(&frameHeader[0]),sizeof(frameHeader));
	BEtoHostl(frameHeader[0]);
	sentence1 = frameHeader[0];

	// position the end frame of the training dataset so that all frames of a sentence are included into the dataset
	while (1) {
         frameNo++;
		 this->dataFile.seekg(this->dataFrameLen*sizeof(float), ios_base::cur);
         this->dataFile.read(reinterpret_cast<char*>(&frameHeader[0]),sizeof(frameHeader));
	     BEtoHostl(frameHeader[0]);
	     sentence2 = frameHeader[0];
		 if ( sentence1 != sentence2 )
			  break;
	};
	this->TestSetStart = frameNo;

	if ( this->dataMode == MLP_DATAMODE_TRAIN )    {        // first 95% frames as training set
		 this->mySetFrames = this->TestSetStart;
		 this->mySetStart = 0;
	}
	else    {                                               // last 5% frames as testing set
	     this->mySetFrames = this->numFrames - this->TestSetStart;
	     this->mySetStart = this->TestSetStart;
	};

	// read the mean and covariance data from the "plp.norm" file
	string normfname(dataPath);
	ifstream normFile;

	normfname.append("plp.norm");
	normFile.open(normfname.c_str(), fstream::in);
    if ( ! normFile.is_open() ) {
		   mlp_log("MLPIFlyDataProvider", "Failed to open the file plp.norm");
		   MLP_Exception("");
	};

	string myline;

	// header for mean data, skipped, no check
	getline(normFile, myline);

	for (int i=0; i< this->dataFrameLen; i++) {
		 getline(normFile, myline);

		 istringstream mystream(myline);
		 float fval;

		 mystream >> fval;
		 this->mean_v.push_back(fval);
	};

	// header for covariance data, skipped, no check
	getline(normFile, myline);

	for (int i=0; i< this->dataFrameLen; i++) {
		 getline(normFile, myline);

		 istringstream mystream(myline);
		 float fval;

		 mystream >> fval;
		 this->covariance_v.push_back(fval);
	};

	normFile.close();
};

MLPIFlyDataProvider::MLPIFlyDataProvider()
{
	this->dataMode = MLP_DATAMODE_TRAIN;
	this->haveLabel = true;
 	this->m_batchSize = 512;

	if ( this->dataMode == MLP_DATAMODE_TRAIN )
		this->m_shuffleBatches = 10;          // for testing and predicting, we don't need to shuffle the data
	else
	    this->m_shuffleBatches = 1;

    this->InitializeFromIFlySource(IFLY_PATH);

	this->total_batches = ROUNDK(DIVUPK(this->mySetFrames,this->m_batchSize),this->m_shuffleBatches);
	this->endOfDataSource = false;
	this->batches_loaded = false;
	this->curSentence = -1;
	this->batchNo = 0;
};

MLPIFlyDataProvider::MLPIFlyDataProvider(const char *dataPath, MLP_DATA_MODE mode, int batchSize, int shuffleBatches)
{
	if ( (mode < 0) || (mode >= MLP_DATAMODE_ERROR) ) {
		  mlp_log("MLPIFlyDataProvider", "Data mode for constructing MLPIFlyDataProvider is not correct");
		  MLP_Exception("");
	};

	this->dataMode = mode;
	this->haveLabel = (mode==MLP_DATAMODE_PREDICT)?false:true;
	this->m_batchSize = batchSize;

	if ( this->dataMode == MLP_DATAMODE_TRAIN )
		 this->m_shuffleBatches = shuffleBatches;    // for testing and predicting, we don't need to shuffle the data
	else
	     this->m_shuffleBatches = 1;

	this->InitializeFromIFlySource(dataPath);

	this->total_batches = ROUNDK(DIVUPK(this->mySetFrames,this->m_batchSize),this->m_shuffleBatches);
	this->endOfDataSource = false;
	this->batches_loaded = false;
	this->curSentence = -1;
	this->batchNo = 0;
};

MLPIFlyDataProvider::~MLPIFlyDataProvider()
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

	for (int i=0; i< (int)this->sDataFrames.size(); i++)
		 delete [] this->sDataFrames[i];

	this->sDataFrames.clear();
	this->sLabelFrames.clear();

	this->release_buffers();
};


//////////////////////////////////////////////////////////////////////////////////////
////                          private member functions                            ////
//////////////////////////////////////////////////////////////////////////////////////


void MLPIFlyDataProvider::prepare_batch_data()
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

bool MLPIFlyDataProvider::haveBatchToProvide()
{
	if ( ! this->endOfDataSource )
		 return(true);

    if ( this->batches_loaded )
		 return(true);

	return(false);
};

void MLPIFlyDataProvider::readOneSentence()
{
	float *dataRecord=NULL;
	unsigned int *frameHeader=NULL;
	unsigned int *labelRecord=NULL;
	unsigned int sentenceID;
	int label;

	this->curStartFrame = this->curFrame;    // Save the first frame of the sentence to ->curStartFrame

	// read the first frame of the sentence
	dataRecord = new float[2+this->dataFrameLen];
	this->dataFile.read(reinterpret_cast<char*>(&dataRecord[0]),sizeof(float)*(2+this->dataFrameLen));
	frameHeader = (unsigned int *)dataRecord;
	BEtoHostl(frameHeader[0]);
	sentenceID = frameHeader[0];
	BEtoHostl(frameHeader[1]);
	//frameID = frameHeader[1];

    for (int i=0; i< this->dataFrameLen; i++)
		 BEtoHostl(*(unsigned int*)&dataRecord[2+i]);

	if ( this->haveLabel ) {
		 labelRecord = new unsigned int[2+this->labelFrameLen];
	     this->labelFile.read(reinterpret_cast<char*>(&labelRecord[0]),sizeof(unsigned int)*(2+this->labelFrameLen));
	     BEtoHostl(labelRecord[2]);
		 label = labelRecord[2];
	};

	this->sDataFrames.push_back(dataRecord);
	if ( this->haveLabel ) {
		 this->sLabelFrames.push_back(label);
		 delete [] labelRecord;
	};

	this->curFrame++;

	// read the other frames of the sentence
	while ( this->curFrame < this->mySetStart + this->mySetFrames ) {
          dataRecord = new float[2+this->dataFrameLen];
	      this->dataFile.read(reinterpret_cast<char*>(&dataRecord[0]),sizeof(float)*(2+this->dataFrameLen));
		  frameHeader = (unsigned int *)dataRecord;
	      BEtoHostl(frameHeader[0]);
	      BEtoHostl(frameHeader[1]);
	      //frameID = frameHeader[1];

		  for (int i=0; i< this->dataFrameLen; i++)
			   BEtoHostl(*(unsigned int*)&dataRecord[2+i]);

		  if ( this->haveLabel ) {
		       labelRecord = new unsigned int[2+this->labelFrameLen];
	           this->labelFile.read(reinterpret_cast<char*>(&labelRecord[0]),sizeof(unsigned int)*(2+this->labelFrameLen));
	           BEtoHostl(labelRecord[2]);
		       label = labelRecord[2];
		  };

		  if ( sentenceID ==  frameHeader[0] ) {
			   this->sDataFrames.push_back(dataRecord);
			   if ( this->haveLabel ) {
		            this->sLabelFrames.push_back(label);
		            delete [] labelRecord;
			   };
		  }
		  else {
			   delete [] dataRecord;
			   //this->dataFile.seekg(-(2+this->dataFrameLen)*sizeof(float), ios_base::cur);  // move back by one frame since we need not read it now
			   if ( this->haveLabel )  {
				   //this->labelFile.seekg(-(2+this->labelFrameLen)*sizeof(int), ios_base::cur);
			       delete [] labelRecord;
			   };

			   break;
		  };
		  this->curFrame++;
	};

	this->gotoDataFrame(this->curFrame);
	if ( this->haveLabel )
	     this->gotoLabelFrame(this->curFrame);
};

// set up the data source of MLPIFlyDataProvider
void MLPIFlyDataProvider::setup_first_data_batches()
{
	this->stageBatchNo = 0;
	this->setup_cont_data_batches();

    if ( this->batches_loaded ) {
	     this->shuffle_data(this->permutations, this->m_batchSize * this->m_shuffleBatches );
	     //cout << "Load new data from files and shuffling the frame sequences" << endl;
    };
};

void MLPIFlyDataProvider::setup_cont_data_batches()
{
	int readCount=0;
	int frame;

	// Initial permutations, permutated each round
	for (int k=0; k < this->m_batchSize * this->m_shuffleBatches; k++)
		    this->permutations[k] = k;

	// For CheckPoint

	if ( this->supportChkPointing ) {
		 MLP_LOCK(&this->chkPointingLock);

	     this->lastChkPointFrame =  this->curChkPointFrame;
	     if ( this->sDataFrames.empty() )
		      this->curChkPointFrame = this->curFrame;        // The start of the new stage coincide with the start of a sentence
	     else
		      this->curChkPointFrame = this->curStartFrame;   // The current sentence extended to the new stage of batches

		 MLP_UNLOCK(&this->chkPointingLock);
	};

	float *tmpFeature;
	int tmpLabel;
	float *fvals;

	tmpFeature = new float[this->dataFrameLen*11];

    for (frame=0; frame < this->m_batchSize * this->m_shuffleBatches; frame++) {  // read the data frame by frame
		 if ( this->sDataFrames.empty() ) {
			  this->readOneSentence();
			  this->frameIndex=0;
		 };

		 // build a  11*39 sized feature data from a  39 sized raw frame
		 for (int ind=1; ind<= 5; ind++) {    // 5 raw frames before the current frames
			  int pos;

			  pos =  this->frameIndex-ind;
			  pos = (int) max(pos,0);

		      fvals = this->sDataFrames[pos];
		      for (int i=0; i< this->dataFrameLen; i++)
			       tmpFeature[(5-ind)*this->dataFrameLen+i] = fvals[2+i];
		 };

		 fvals = this->sDataFrames[this->frameIndex];
		 for (int i=0; i< this->dataFrameLen; i++)
			  tmpFeature[5*this->dataFrameLen+i] = fvals[2+i];

		 for (int ind=1; ind<= 5; ind++) {    // 5 raw frames after the current frames
			  int pos;

			  pos =  this->frameIndex+ind;
			  pos = (int) min<int>(pos,(int)this->sDataFrames.size()-1);

		      fvals = this->sDataFrames[pos];
		      for (int i=0; i< this->dataFrameLen; i++)
			       tmpFeature[(5+ind)*this->dataFrameLen+i] = fvals[2+i];
		 };

		 if ( this->haveLabel )
			  tmpLabel = this->sLabelFrames[this->frameIndex];


		 // put the created feature and label into the batch buffer of the DataProvider
		 for (int i=0; i < this->m_dataFeatureSize; i++)
			  this->featureData[frame*this->m_dataFeatureSize+i] = (float)tmpFeature[i];  // need normalization ?

		 if ( this->haveLabel ) {
		      for (int i=0; i < this->m_dataLabelSize; i++)
		           this->labelData[frame*this->m_dataLabelSize+i] = 0.0f;
	          this->labelData[frame*this->m_dataLabelSize+(int)tmpLabel] = 1.0f;
		 };

		 this->frameIndex++;

		 // when all frames from the current sentence have been processed
		 if ( this->frameIndex >= (int) this->sDataFrames.size() ) {

			  // clean the vector representing the sentence frames
			  for (int i=0; i < (int) this->sDataFrames.size(); i++)
				   delete [] this->sDataFrames[i];
			  this->sDataFrames.clear();
			  if ( this->haveLabel )
				   this->sLabelFrames.clear();

			  if ( this->curFrame >= this->mySetStart + this->mySetFrames ) {  // all the frames of the dataset has been processed
				   this->endOfDataSource = true;
			       readCount = frame+1;
			       goto endf;
		      };
		 };
	};

endf:    // duplicate the first "readCount" records to minibatch*shuffleBatches records

    if ( readCount > 0 ) {
	     int dst=readCount;
		 int src=0;
		 while ( dst < this->m_batchSize * this->m_shuffleBatches ) {
				src = dst % readCount;

		        for (int i=0; i < this->m_dataFeatureSize; i++)
			           this->featureData[dst*this->m_dataFeatureSize+i] = this->featureData[src*this->m_dataFeatureSize+i];

				if ( this->haveLabel)
		             for (int i=0; i < this->m_dataLabelSize; i++)
				          this->labelData[dst*this->m_dataLabelSize+i] = this->labelData[src*this->m_dataLabelSize+i];

				dst++;
		  };
	 };

	 if ( frame > 0  || readCount > 0 )
		  this->batches_loaded = true;

	 delete [] tmpFeature;
};

void MLPIFlyDataProvider::shuffle_data(int *index, int len)
{
	std::random_shuffle(index, index+len);
};

void MLPIFlyDataProvider::gotoDataFrame(int frameNo)
{
	//this->dataFile.seekg(PHEADER_SIZE+frameNo*(2+this->dataFrameLen)*sizeof(float));

	//using relative seeking since direct positioning above 4G causes issue
	this->dataFile.seekg(PHEADER_SIZE);
	for (int i=0; i< frameNo/4000000; i++)
		this->dataFile.seekg(4000000*(2+this->dataFrameLen)*sizeof(float), ios_base::cur);

	this->dataFile.seekg((frameNo%4000000)*(2+this->dataFrameLen)*sizeof(float), ios_base::cur);
};


void MLPIFlyDataProvider::gotoLabelFrame(int frameNo)
{
	//this->labelFile.seekg(PHEADER_SIZE+frameNo*(2+this->labelFrameLen)*sizeof(int));

	//using relative seeking since direct positioning above 4G causes issue
	this->labelFile.seekg(PHEADER_SIZE);
	for (int i=0; i< frameNo/4000000; i++)
	     this->labelFile.seekg(4000000*(2+this->labelFrameLen)*sizeof(int), ios_base::cur);

	this->labelFile.seekg((frameNo%4000000)*(2+this->labelFrameLen)*sizeof(int), ios_base::cur);
};


//////////////////////////////////////////////////////////////////////////////////////
////                          public member functions                             ////
//////////////////////////////////////////////////////////////////////////////////////


void MLPIFlyDataProvider::setupDataProvider()
{
	this->gotoDataFrame(this->mySetStart);
	if ( this->haveLabel )
		this->gotoLabelFrame(this->mySetStart);

	this->curFrame = this->mySetStart;

	this->curStartFrame = this->curFrame;

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

// This one is called when we want the data provider first to recover from a checkpointed start, and the may start checkpointing or not
void MLPIFlyDataProvider::setupDataProvider(int startFrameNo, bool doChkPointing)
{
	if ( this->dataMode != MLP_DATAMODE_TRAIN ) {
		 mlp_log("MLPIFlyDataProvider", "This interface can only be called with the TRAIN mode");
		 MLP_Exception("");
	};

	this->gotoDataFrame(startFrameNo);
	this->gotoLabelFrame(startFrameNo);

	this->curFrame = startFrameNo;
	this->curStartFrame = this->curFrame;

	this->curChkPointFrame = this->lastChkPointFrame = this->curFrame;

	// For CheckPoint
	this->supportChkPointing = doChkPointing;
	if ( this->supportChkPointing )
		 MLP_LOCK_INIT(&this->chkPointingLock);

	this->permutations = new int[this->m_batchSize * this->m_shuffleBatches];
	this->featureData  = new float[this->m_batchSize * this->m_shuffleBatches * this->m_dataFeatureSize];
	if ( this->haveLabel )
         this->labelData = new float[this->m_batchSize * this->m_shuffleBatches * this->m_dataLabelSize];

	this->setup_first_data_batches();

	this->create_buffers(this->m_batchSize);

	this->initialized = true;

	MLP_CHECK(this->startup_worker());
};

void MLPIFlyDataProvider::resetDataProvider()
{
	if ( !this->initialized ) {
		 mlp_log("MLPIFlyDataProvider", "The DataProvider is still not started yet, no reset should be called");
		 MLP_Exception("");
	};
	MLP_CHECK(this->shutdown_worker());

	this->reset_buffers();

    this->endOfDataSource = false;
	this->batchNo = 0;
	this->batches_loaded = false;

	this->dataFile.clear();
	this->gotoDataFrame(this->mySetStart);
	if ( this->haveLabel ) {
		this->labelFile.clear();
		this->gotoLabelFrame(this->mySetStart);
	};

	this->curFrame = this->mySetStart;

	this->curChkPointFrame = this->lastChkPointFrame = this->curFrame;

	for (int i=0; i< (int)this->sDataFrames.size(); i++)
		 delete [] this->sDataFrames[i];

	this->sDataFrames.clear();
	this->sLabelFrames.clear();

	if ( this->supportChkPointing )
		 MLP_LOCK_INIT(&this->chkPointingLock);

	this->setup_first_data_batches();

	MLP_CHECK(this->startup_worker());
};

void MLPIFlyDataProvider::getCheckPointFrame(int & frameNo)
{
	 int stageBatch;

	 MLP_LOCK(&this->chkPointingLock);

	 stageBatch = this->stageBatchNo;
	 if ( stageBatch > MLP_BATCH_RING_SIZE )
		  frameNo = this->curChkPointFrame;   // Even with the batches on buffer considered considered, this position still ensure no frame being skipped by the MLPTrainer
	 else
	      frameNo = this->lastChkPointFrame;  // Use "lastChkPointFrame" as checkpoint position to ensure no frame will be skipped for processing

	 MLP_UNLOCK(&this->chkPointingLock);
};


// if the output for the frame matches its label, return true to indicate a successful mapping of this
// frame by the neural network.  This interface will be called by the MLPTester class when calculating
// the success ratio of the neural network on this type of data
bool MLPIFlyDataProvider::frameMatching(const float *frameOutput, const float *frameLabel, int len)
{
	float element;

	for (int i=0; i< len; i++) {
		 element = (frameOutput[i]<0.5)?0.0f:1.0f;

		 if ( element != frameLabel[i] )
			  return(false);
	};

	return(true);
};
