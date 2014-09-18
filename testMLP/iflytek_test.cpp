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
#include "MLPIFlyDataProvider.h"
#include "MLPChkPointingMgr.h"

using namespace std;

void iflytek_training();
void iflytek_training2();   // training using pretrained weight
void iflytek_training3();   // training with checkpointing support
void iflytek_batch_testing();
void iflytek_predicting();

void iflytek_training()
{
	struct mlp_tv startv, endv;

	MLP_NETTYPE nettype;
	const int nLayers = 8;
	int dimensions[nLayers] = {429, 2048, 2048, 2048, 2048, 2048, 2048, 8991};
	float etas[nLayers] = {0.0f, 0.0001f, 0.0001f, 0.0001f, 0.0001f, 0.00005f, 0.00005f, 0.00005f};
	float momentum = 0.3f;
	ACT_FUNC actFuncs[nLayers] = {ANOFUNC,  AFUNC_SIGMOID, AFUNC_SIGMOID, AFUNC_SIGMOID, AFUNC_SIGMOID, AFUNC_SIGMOID, AFUNC_SIGMOID, AFUNC_SOFTMAX};
	COST_FUNC costFunc = CFUNC_CE;

	int minibatch = 1024;
	int shuffleBatches = 10;
	int batches;
	int totalbatches;
	int epoches=2;

	MLPNetProvider *netProviderp=NULL;
    MLPDataProvider *dataProviderp=NULL;

	// Training the neural network using MNist labelled dataset
    MLPTrainerBase *trainerp;

	dataProviderp = new MLPIFlyDataProvider(IFLY_PATH, MLP_DATAMODE_TRAIN, minibatch, shuffleBatches);
	dataProviderp->setupDataProvider();                            // set up the data provider
	dimensions[0] = dataProviderp->getFeatureSize();
	dimensions[nLayers-1] = dataProviderp->getLabelSize();
	totalbatches = dataProviderp->getTotalBatches();

	nettype = NETTYPE_MULTI_CLASSIFICATION;
    netProviderp = new MLPNetProvider(nettype, nLayers, dimensions, etas, momentum, actFuncs, costFunc, true);

    trainerp = new MLPTrainerOCL(*netProviderp,*dataProviderp, MLP_OCL_DI_GPU, minibatch);    // set up the trainer

	cout << totalbatches << " batches of data to be trained with " << epoches << " epoches, just waiting..." << endl;

	getCurrentTime(&startv);
	batches = trainerp->batchTraining(0,epoches);                                       // do the training
	getCurrentTime(&endv);

	cout << batches << " batches of data were trained actually" << endl;
    cout << "Training duration: " << diff_msec(&startv, &endv) << " mill-seconds" << endl;

    // Finalize the result from the training work, so that the Tester or Predictor can be set up based on it
	trainerp->saveNetConfig("./");

	delete netProviderp;
	delete dataProviderp;
	delete trainerp;
};

// For training using pretrained weights
void iflytek_training2()
{
	struct mlp_tv startv, endv;

	int minibatch = 1024;
	int shuffleBatches = 10;
	int batches;
	int totalbatches;
	int epoches=16;
	int startBatch;
	int startEpoch;

	MLPCheckPointManager cpManager;
	MLPNetProvider *netProviderp=NULL;
    MLPDataProvider *dataProviderp=NULL;
    MLPTrainerBase *trainerp;

	cpManager.cpFindAndLoad("./tmp/");
	if ( cpManager.cpAvailable() ) {
		 struct MLPCheckPointState *statep;

	     cout << "Valid checkpoint found, recover and start new checkpointing from this one"  << endl;

		 statep = cpManager.getChkPointState();
		 netProviderp = new MLPNetProvider(statep->netConfPath, statep->ncTrainingConfigFname, statep->ncNNetDataFname);

         dataProviderp = new MLPIFlyDataProvider(IFLY_PATH, MLP_DATAMODE_TRAIN, minibatch, shuffleBatches);
		 dataProviderp->setupDataProvider(statep->cpFrameNo, true);

		 cout << "The MLPDataProvider start from Frame " << statep->cpFrameNo << endl;

		 startBatch = statep->cpBatchNo;
		 startEpoch = statep->cpEpoch;

		 cout << "The Trainer start from batch " << statep->cpBatchNo << " of Epoch " << statep->cpEpoch << endl;

		 cpManager.cpUnload();
	}
	else {
		 cout << "No old checkpoint found, start new checkpointing any way" << endl;

	     netProviderp = new MLPNetProvider("./", "mlp_training_init.conf", "mlp_nnet_init.dat");
         dataProviderp = new MLPIFlyDataProvider(IFLY_PATH, MLP_DATAMODE_TRAIN, minibatch, shuffleBatches);
	     dataProviderp->setupDataProvider(0, true);

		 startBatch = 0;
		 startEpoch = 0;
	};

    trainerp = new MLPTrainerOCL(*netProviderp,*dataProviderp, MLP_OCL_DI_GPU, minibatch);

	cpManager.enableCheckPointing(*trainerp, "./tmp/");
	MLP_CHECK( cpManager.startCheckPointing() );

	totalbatches = dataProviderp->getTotalBatches();

	cout << totalbatches << " batches of data to be trained with " << epoches << " epoches, just waiting..." << endl;

	getCurrentTime(&startv);
	batches = trainerp->batchTrainingWithCheckPointing(0, epoches, startBatch, startEpoch,  true);
	getCurrentTime(&endv);

	MLP_CHECK( cpManager.endCheckPointing() );

	cout << batches << " batches of data were trained actually" << endl;
    cout << "Training duration: " << diff_msec(&startv, &endv) << " mill-seconds" << endl;

    // Finalize the result from the training, so that the Tester or Predictor can be set up based on it
	trainerp->saveNetConfig("./");

	cpManager.cpCleanUp("./tmp/");

	delete netProviderp;
	delete dataProviderp;
	delete trainerp;
};


// For testing the MLPTrainer with CheckPointing support
void iflytek_training3()
{
	struct mlp_tv startv, endv;

	MLP_NETTYPE nettype = NETTYPE_MULTI_CLASSIFICATION;
	const int nLayers = 8;
	int dimensions[nLayers] = {429, 2048, 2048, 2048, 2048, 2048, 2048, 8991};
	float etas[nLayers] = {0.0f, 0.0001f, 0.0001f, 0.0001f, 0.0001f, 0.00005f, 0.00005f, 0.00005f};
	float momentum = 0.3f;
	ACT_FUNC actFuncs[nLayers] = {ANOFUNC,  AFUNC_SIGMOID, AFUNC_SIGMOID, AFUNC_SIGMOID, AFUNC_SIGMOID, AFUNC_SIGMOID, AFUNC_SIGMOID, AFUNC_SOFTMAX};
	COST_FUNC costFunc = CFUNC_CE;

	int minibatch = 1024;
	int shuffleBatches = 10;
	int batches;
	int totalbatches;
	int epoches=2;
	int startBatch;
	int startEpoch;

	MLPCheckPointManager cpManager;
	MLPNetProvider *netProviderp=NULL;
    MLPDataProvider *dataProviderp=NULL;
    MLPTrainerBase *trainerp;

	cpManager.cpFindAndLoad("./tmp/");
	if ( cpManager.cpAvailable() ) {
		 struct MLPCheckPointState *statep;

	     cout << "Valid checkpoint found, recover and start new checkpointing from this one"  << endl;

		 statep = cpManager.getChkPointState();
		 netProviderp = new MLPNetProvider(statep->netConfPath, statep->ncTrainingConfigFname, statep->ncNNetDataFname);

         dataProviderp = new MLPIFlyDataProvider(IFLY_PATH, MLP_DATAMODE_TRAIN, minibatch, shuffleBatches);
		 dataProviderp->setupDataProvider(statep->cpFrameNo, true);

		 cout << "The MLPDataProvider start from Frame " << statep->cpFrameNo << endl;

		 startBatch = statep->cpBatchNo;
		 startEpoch = statep->cpEpoch;

		 cout << "The Trainer start from batch " << statep->cpBatchNo << " of Epoch " << statep->cpEpoch << endl;

		 cpManager.cpUnload();
	}
	else {
		 cout << "No old checkpoint found, start new checkpointing any way" << endl;

         netProviderp = new MLPNetProvider(nettype, nLayers, dimensions, etas, momentum, actFuncs, costFunc, true);
         dataProviderp = new MLPIFlyDataProvider(IFLY_PATH, MLP_DATAMODE_TRAIN, minibatch, shuffleBatches);
	     dataProviderp->setupDataProvider(0, true);

		 startBatch = 0;
		 startEpoch = 0;
	};

    trainerp = new MLPTrainerOCL(*netProviderp,*dataProviderp, MLP_OCL_DI_GPU, minibatch);

	cpManager.enableCheckPointing(*trainerp, "./tmp/");
	MLP_CHECK( cpManager.startCheckPointing() );

	totalbatches = dataProviderp->getTotalBatches();

	cout << totalbatches << " batches of data to be trained with " << epoches << " epoches, just waiting..." << endl;

	getCurrentTime(&startv);
	batches = trainerp->batchTrainingWithCheckPointing(0, epoches, startBatch, startEpoch,  true);
	getCurrentTime(&endv);

	MLP_CHECK( cpManager.endCheckPointing() );

	cout << batches << " batches of data were trained actually" << endl;
    cout << "Training duration: " << diff_msec(&startv, &endv) << " mill-seconds" << endl;

    // Finalize the result from the training, so that the Tester or Predictor can be set up based on it
	trainerp->saveNetConfig("./");

	cpManager.cpCleanUp("./tmp/");

	delete netProviderp;
	delete dataProviderp;
	delete trainerp;
};


void iflytek_batch_testing()
{
	struct mlp_tv startv, endv;

	int minibatch = 1024;
	int shuffleBatches = 4;
	int totalbatches;

	MLPNetProvider *netProviderp=NULL;
    MLPDataProvider *dataProviderp=NULL;

	// Testing the MNist labelled dataset on the trained neural network
	MLPTesterBase *testerp=NULL;

	netProviderp = new MLPNetProvider("./", MLP_NP_NNET_DATA_NEW);
	dataProviderp =	new MLPIFlyDataProvider(IFLY_PATH, MLP_DATAMODE_TEST, minibatch, shuffleBatches);
	dataProviderp->setupDataProvider();                              // set up the data provider

	testerp = new MLPTesterOCL(*netProviderp,*dataProviderp,MLP_OCL_DI_GPU, minibatch);

    totalbatches = dataProviderp->getTotalBatches();

    cout << totalbatches << " batches of data to be tested, just waiting ..." << endl;

	getCurrentTime(&startv);
	testerp->batchTesting(totalbatches);
	getCurrentTime(&endv);

    cout << "Testing duration: " << diff_msec(&startv, &endv) << " mill-seconds" << endl;

	int totalNum, succNum;
	testerp->getTestingStats(totalNum, succNum);
	cout << totalNum << " frames tested," << succNum << " frames succeed, success ratio is " << ((float)succNum)*100.0f/((float)totalNum) << "%" << endl;

    delete dataProviderp;
	delete netProviderp;
	delete testerp;
};

void iflytek_predicting()
{
	struct mlp_tv startv, endv;

	int minibatch = 1024;
	int batches;
	int totalbatches;
	int frames;

	MLPNetProvider *netProviderp=NULL;
    MLPDataProvider *dataProviderp=NULL;

	// Using MNist testing dataset to do batch predicting on the trained neural network
	MLPPredictorBase *predictorp=NULL;
	float *inputVectors;
	float *outputVectors;

	netProviderp = new MLPNetProvider("./", MLP_NP_NNET_DATA_NEW);
	dataProviderp =	new MLPIFlyDataProvider(".", MLP_DATAMODE_PREDICT, minibatch, 0);
	dataProviderp->setupDataProvider();                              // set up the data provider

	predictorp = new MLPPredictorOCL(*netProviderp,MLP_OCL_DI_GPU, minibatch);
	outputVectors = new float[predictorp->getOutputVectorSize()*minibatch];

    totalbatches = dataProviderp->getTotalBatches();
	totalbatches = min<int>(totalbatches, 10);

    cout << totalbatches << " batches of data to be predicted, just waiting ..." << endl;

	getCurrentTime(&startv);

    batches=0;

	while ( dataProviderp->batchAvailable() && (batches < totalbatches) ) {

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

    // Using MNist testing dataset to do single predicting on the trained neural network
	float *inputVector;
	float *outputVector;

	outputVector = new float[predictorp->getOutputVectorSize()];

	getCurrentTime(&startv);

	batches=0;
    frames=0;
	while (  dataProviderp->batchAvailable() ) {

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


	delete [] outputVector;

	delete netProviderp;
	delete dataProviderp;
	delete predictorp;
};

