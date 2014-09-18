/*
 *  COPYRIGHT:  Copyright (c) 2013 Advanced Micro Devices, Inc.  All rights reserved
 *
 *   Changed by:         Qianfeng Zhang@amd.com ( March 2014 )
 *
 *   Written By:               Junli Gu@amd.com ( Dec   2013 )
 */

#include <iostream>

#include "MLPUtil.h"
#include "MLPTrainerOCL.h"
#include "MLPTesterOCL.h"
#include "MLPPredictorOCL.h"
#include "MLPNetProvider.h"
#include "MLPSimpleDataProvider.h"
#include "MLPMNistDataProvider.h"
#include "MLPIFlyDataProvider.h"
#include "MLPChkPointingMgr.h"

using namespace std;


void simple_training();
void simple_batch_testing();
void simple_predicting();


void simple_training()
{
	struct mlp_tv startv, endv;

	MLP_NETTYPE nettype;
	const int nLayers = 8;
	int dimensions[nLayers] = {429,2048,2048,2048,2048,2048,2048,8991};
	float etas[nLayers] = {0.0f, 0.0001f, 0.0001f, 0.0001f, 0.0001f, 0.00005f, 0.00005f, 0.00005f};
	float momentum = 0.3f;
	ACT_FUNC actFuncs[nLayers] = {ANOFUNC, AFUNC_SIGMOID,AFUNC_SIGMOID,AFUNC_SIGMOID,AFUNC_SIGMOID,AFUNC_SIGMOID,AFUNC_SIGMOID, AFUNC_SOFTMAX};
	COST_FUNC costFunc = CFUNC_CE;

	int minibatch = 1024;
	int shuffleBatches = 10;
	int batches;
	int totalbatches;
	int epoches = 200;

	MLPNetProvider *netProviderp=NULL;
    MLPDataProvider *dataProviderp=NULL;


    // Training the neural network using Simple labelled dataset
	MLPTrainerBase *trainerp=NULL;

	nettype = NETTYPE_MULTI_CLASSIFICATION;
	netProviderp = new MLPNetProvider(nettype,nLayers,dimensions,etas, momentum, actFuncs,costFunc, true);
	dataProviderp =	new MLPSimpleDataProvider(MLP_DATAMODE_TRAIN,dimensions[0],dimensions[nLayers-1],minibatch,shuffleBatches);
	dataProviderp->setupDataProvider();                            // set up the data provider
    totalbatches = dataProviderp->getTotalBatches();

    trainerp = new MLPTrainerOCL();
	trainerp->setupMLP(*netProviderp,*dataProviderp,minibatch);    // set up the trainer

	cout << totalbatches << " batches of data to be trained with " << epoches << " epoches, just waiting..." << endl;

	getCurrentTime(&startv);
	batches = trainerp->batchTraining(0,epoches);               // do the training
	getCurrentTime(&endv);

	cout << batches << " batches of data were trained actually" << endl;
    cout << "Training duration: " << diff_msec(&startv, &endv) << " mill-seconds" << endl;

    //save the result from the training work, so that the Tester or Predictor can be set up based on it
	trainerp->saveNetConfig("./");

	delete netProviderp;
	delete dataProviderp;
	delete trainerp;
}

void simple_batch_testing()
{
	struct mlp_tv startv, endv;

	const int nLayers = 8;
	int dimensions[nLayers] = {429,2048,2048,2048,2048,2048,2048,8991};

	int minibatch = 512;
	int shuffleBatches = 1;
	// int totalbatches;

	MLPNetProvider *netProviderp=NULL;
    MLPDataProvider *dataProviderp=NULL;

	// Testing the Simple labelled dataset on the trained neural network
	MLPTesterBase *testerp=NULL;

	netProviderp = new MLPNetProvider("./", MLP_NP_NNET_DATA_NEW);
	dataProviderp =	new MLPSimpleDataProvider(MLP_DATAMODE_TEST,dimensions[0],dimensions[nLayers-1],minibatch,shuffleBatches);
	dataProviderp->setupDataProvider();                              // set up the data provider

	testerp = new MLPTesterOCL(*netProviderp,*dataProviderp,MLP_OCL_DI_GPU, minibatch);

	// totalbatches = dataProviderp->getTotalBatches();

	getCurrentTime(&startv);
	testerp->batchTesting(100);
	getCurrentTime(&endv);

    cout << "Testing duration: " << diff_msec(&startv, &endv) << " mill-seconds" << endl;

	int totalNum, succNum;
	testerp->getTestingStats(totalNum, succNum);
	cout << totalNum << " frames tested," << succNum << " frames succeed, success ratio is " << ((float)succNum)*100.0f/((float)totalNum) << "%" << endl;

	delete netProviderp;
	delete dataProviderp;
	delete testerp;
}

void simple_predicting()
{
	struct mlp_tv startv, endv;

	const int nLayers = 3;
	int dimensions[nLayers] = {429,2048,8991};

	int minibatch = 512;
	int batches;
	int totalbatches;
	int frames;

	MLPNetProvider *netProviderp=NULL;
    MLPDataProvider *dataProviderp=NULL;

	// Using Simple testing dataset to do batch predicting on the trained neural network
	MLPPredictorBase *predictorp=NULL;
	float *inputVectors;
	float *outputVectors;

	netProviderp = new MLPNetProvider("./", MLP_NP_NNET_DATA_NEW);
    dataProviderp =	new MLPSimpleDataProvider(MLP_DATAMODE_PREDICT,dimensions[0],dimensions[nLayers-1],minibatch,0);
	dataProviderp->setupDataProvider();                               // set up the data provider

	predictorp = new MLPPredictorOCL(*netProviderp,MLP_OCL_DI_GPU, minibatch);
	outputVectors = new float[predictorp->getOutputVectorSize() * minibatch];

	getCurrentTime(&startv);

    batches=0;
	totalbatches = dataProviderp->getTotalBatches();
	while ( ( batches < totalbatches ) && ( batches < 100 ) ) {

		    MLP_CHECK(dataProviderp->getBatchData(predictorp->getBatchSize(),inputVectors,true));

			predictorp->batchPredicting(inputVectors,outputVectors);

            // tell the data provider that I have done with current batch of data, want next batch of data
			MLP_CHECK(dataProviderp->nextBatch());

			batches++;
	}

    getCurrentTime(&endv);

	cout << "Predicting duration: " << diff_msec(&startv, &endv) << " mill-seconds" << endl;
	cout << "Batch size:" << predictorp->getBatchSize() << " " << batches << " predicted" << endl;

	delete [] outputVectors;

	dataProviderp->resetDataProvider();

	// Using Simple testing dataset to do single predicting on the trained neural network
	float *inputVector;
	float *outputVector;

	outputVector = new float[predictorp->getOutputVectorSize()];

	getCurrentTime(&startv);

	batches=0;
	totalbatches = dataProviderp->getTotalBatches();
	frames=0;
	while ( ( batches < totalbatches ) && ( batches < 100 ) ) {

		    MLP_CHECK(dataProviderp->getBatchData(predictorp->getBatchSize(),inputVector,true));

			for (int i=0; i< predictorp->getBatchSize(); i++) {
				 predictorp->singlePredicting(inputVector,outputVector);
				 frames++;
			};

            // tell the data provider that I have done with current batch of data, want next batch of data
		 	MLP_CHECK(dataProviderp->nextBatch());

			batches++;
	}

    getCurrentTime(&endv);

	cout << "Predicting duration: " << diff_msec(&startv, &endv) << " mill-seconds" << endl;
    cout << frames << " frames of data predicted" << endl;

	delete [] outputVectors;

	delete netProviderp;
	delete dataProviderp;
	delete predictorp;
}

