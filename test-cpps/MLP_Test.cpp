/*
 *  COPYRIGHT:  Copyright (c) 2013 Advanced Micro Devices, Inc.  All rights reserved
 *
 *   Changed by:         Qianfeng Zhang@amd.com ( March 2014 )
 *
 *   Written By:               Junli Gu@amd.com ( Dec   2013 )
 */

#include <iostream>

#include "MLPUtil.h"
#include "MLPTrainer.h"
#include "MLPTester.h"
#include "MLPPredictor.h"
#include "MLPNetProvider.h"
#include "MLPSimpleDataProvider.h"
#include "MLPMNistDataProvider.h"
#include "MLPIFlyDataProvider.h"
#include "MLPChkPointingMgr.h"

using namespace std;

#define MNIST_PATH2 "../../MNIST2/"

static void simple_training();
static void simple_testing();
static void simple_predicting();

static void mnist_training();
static void mnist_training2();
static void mnist_training3();     // training with checkpointing support
static void mnist_testing();
static void mnist_predicting();

static void iflytek_training();
static void iflytek_training3();   // training with checkpointing support
static void iflytek_testing();
static void iflytek_predicting();

static void test_cp_cleanup();

int main()
{
	char anykey;

    //iflytek_training3();
	//mnist_training2();
	mnist_training();
	//simple_training();

	//cout << "Press any key to continue ..." << endl;

	//cin >> anykey;

	//iflytek_testing();
	//iflytek_predicting();
	mnist_testing();
	//simple_testing();
	//mnist_predicting();

	cout << "Press any key to end ..." << endl;

	cin >> anykey;

	return(0);
};

static void test_cp_cleanup()
{
	MLPCheckPointManager cpManager;

	cpManager.cpCleanUp("./tmp/");

};

static void mnist_training()
{
	struct mlp_tv startv, endv;

	MLP_NETTYPE nettype = NETTYPE_MULTI_CLASSIFICATION;
	const int nLayers = 3;
	int dimensions[nLayers] = {784, 800, 10};
	float etas[nLayers] = {0.0f, 0.0002f, 0.0001f};
	float momentum = 0.4f;
	ACT_FUNC actFuncs[nLayers] = {ANOFUNC, AFUNC_SIGMOID, AFUNC_SOFTMAX};
	COST_FUNC costFunc = CFUNC_CE;

	int minibatch = 1200;
	int shuffleBatches = 50;
	int batches;
	int totalbatches;
	int epoches = 200;

	MLPNetProvider *netProviderp=NULL;
    MLPDataProvider *dataProviderp=NULL;


	// Training the neural network using MNist labelled dataset
    MLPTrainer *trainerp;

	dataProviderp = new MLPMNistDataProvider(MNIST_PATH, MLP_DATAMODE_TRAIN, minibatch, shuffleBatches);
	//dataProviderp = new MLPMNistDataProvider(MNIST_PATH2, MLP_DATAMODE_TRAIN, minibatch, shuffleBatches);
	dataProviderp->setupDataProvider();                            // set up the data provider
	dimensions[0] = dataProviderp->getFeatureSize();
	dimensions[nLayers-1] = dataProviderp->getLabelSize();
	totalbatches = dataProviderp->getTotalBatches();


    netProviderp = new MLPNetProvider(nettype, nLayers, dimensions, etas, momentum, actFuncs, costFunc, true);

    trainerp = new MLPTrainer(*netProviderp,*dataProviderp, MLP_OCL_DI_GPU, minibatch);    // set up the trainer

	cout << totalbatches << " batches of data to be trained with " << epoches << " epoches, just waiting..." << endl;

	getCurrentTime(&startv);
	batches = trainerp->batchTraining(0, epoches);                                       // do the training
	getCurrentTime(&endv);

	cout << batches << " batches of data were trained actually" << endl;
    cout << "Training duration: " << diff_msec(&startv, &endv) << " mill-seconds" << endl;


    // Finalize the result from the training work, so that the Tester or Predictor can be set up based on it
	trainerp->saveNetConfig("./");

	delete netProviderp;
	delete dataProviderp;
	delete trainerp;
};

// doing MNIST training based on pre-trained weights
static void mnist_training2()
{
	struct mlp_tv startv, endv;

	int minibatch = 1024;
	int shuffleBatches = 59;
	int batches;
	int totalbatches;
	int epoches=200;

	MLPNetProvider *netProviderp=NULL;
    MLPDataProvider *dataProviderp=NULL;


	// Training the neural network using MNist labelled dataset
    MLPTrainer *trainerp;

	dataProviderp = new MLPMNistDataProvider(MNIST_PATH2, MLP_DATAMODE_TRAIN, minibatch, shuffleBatches);
	dataProviderp->setupDataProvider();                            // set up the data provider
	totalbatches = dataProviderp->getTotalBatches();

	netProviderp = new MLPNetProvider("./", "mlp_netarch_init.conf", "mlp_netweights_init.dat");

    trainerp = new MLPTrainer(*netProviderp,*dataProviderp, MLP_OCL_DI_GPU, minibatch);    // set up the trainer

	cout << totalbatches << " batches of data to be trained with " << epoches << " epoches, just waiting..." << endl;

	getCurrentTime(&startv);
	batches = trainerp->batchTraining(0, epoches);                                       // do the training
	getCurrentTime(&endv);

	cout << batches << " batches of data were trained actually" << endl;
    cout << "Training duration: " << diff_msec(&startv, &endv) << " mill-seconds" << endl;


    // Finalize the result from the training work, so that the Tester or Predictor can be set up based on it
	trainerp->saveNetConfig("./");

	delete netProviderp;
	delete dataProviderp;
	delete trainerp;
};

// doing MNIST training with checkpointing support
static void mnist_training3()
{
	struct mlp_tv startv, endv;

	MLP_NETTYPE nettype = NETTYPE_MULTI_CLASSIFICATION;
	const int nLayers = 3;
	int dimensions[nLayers] = {784, 800, 10};
	float etas[nLayers] = {0.0f, 0.0002f, 0.0001f};
	float momentum = 0.4f;
	ACT_FUNC actFuncs[nLayers] = {ANOFUNC, AFUNC_SIGMOID, AFUNC_SOFTMAX};
	COST_FUNC costFunc = CFUNC_CE;

	int minibatch = 1200;
	int shuffleBatches = 25;
	int batches;
	int totalbatches;
	int epoches = 200;
	int startBatch;
	int startEpoch;


	MLPCheckPointManager cpManager;
	MLPNetProvider *netProviderp=NULL;
    MLPDataProvider *dataProviderp=NULL;

	cpManager.cpFindAndLoad("./tmp/");
	if ( cpManager.cpAvailable() ) {
		 struct MLPCheckPointState *statep;

		 cout << "Valid checkpoint found, recover and start new checkpointing from this one" << endl;

		 statep = cpManager.getChkPointState();
		 netProviderp = new MLPNetProvider(statep->netConfPath, statep->netConfArchFileName, statep->netConfDataFileName);

	     dataProviderp = new MLPMNistDataProvider(MNIST_PATH, MLP_DATAMODE_TRAIN, minibatch, shuffleBatches);
		 dataProviderp->setupDataProvider(statep->cpFrameNo, true);

		 startBatch = statep->cpBatchNo;
		 startEpoch = statep->cpEpoch;

		 cpManager.cpUnload();
	}
	else {
		 cout << "No old checkpoint found, start new checkpointing any way" << endl;

         netProviderp = new MLPNetProvider(nettype, nLayers, dimensions, etas, momentum, actFuncs, costFunc, true);

	     dataProviderp = new MLPMNistDataProvider(MNIST_PATH, MLP_DATAMODE_TRAIN, minibatch, shuffleBatches);
	     dataProviderp->setupDataProvider(0, true);

		 startBatch = 0;
		 startEpoch = 0;
	};

	// Training the neural network using MNist labelled dataset
    MLPTrainer *trainerp;

    trainerp = new MLPTrainer(*netProviderp,*dataProviderp, MLP_OCL_DI_GPU, minibatch);

	cpManager.enableCheckPointing(*trainerp, "./tmp/");
	MLP_CHECK( cpManager.startCheckPointing() );

	totalbatches = dataProviderp->getTotalBatches();

	cout << totalbatches << " batches of data to be trained with " << epoches << " epoches, just waiting..." << endl;

	getCurrentTime(&startv);
	batches = trainerp->batchTrainingWithCheckPointing(0, epoches,  startBatch, startEpoch, true);
	getCurrentTime(&endv);

	MLP_CHECK( cpManager.endCheckPointing() );

	cout << batches << " batches of data were trained actually" << endl;
    cout << "Training duration: " << diff_msec(&startv, &endv) << " mill-seconds" << endl;

    // Finalize the result from the training work, so that the Tester or Predictor can be set up based on it
	trainerp->saveNetConfig("./");

	cpManager.cpCleanUp("./tmp/");

	delete netProviderp;
	delete dataProviderp;
	delete trainerp;
};

static void mnist_testing()
{
	struct mlp_tv startv, endv;

	int minibatch = 500;
	int shuffleBatches = 21;
	int totalbatches;

	MLPNetProvider *netProviderp=NULL;
    MLPDataProvider *dataProviderp=NULL;

	// Testing the MNist labelled dataset on the trained neural network
	MLPTester *testerp=NULL;

	netProviderp = new MLPNetProvider("./", MLP_NC_ARCH_NEW, MLP_NC_DATA_NEW);
	dataProviderp =	new MLPMNistDataProvider(MNIST_PATH, MLP_DATAMODE_TEST, minibatch, shuffleBatches);
	dataProviderp->setupDataProvider();                              // set up the data provider

	testerp = new MLPTester(*netProviderp,*dataProviderp,MLP_OCL_DI_GPU,minibatch);

    totalbatches = dataProviderp->getTotalBatches();

    cout << totalbatches << " batches of data to be tested just waiting ..." << endl;

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


static void mnist_predicting()
{
	struct mlp_tv startv, endv;

	int minibatch = 1024;
    int shuffleBatches = 10;
	int batches;
	int totalbatches;
	int frames;

	MLPNetProvider *netProviderp=NULL;
    MLPDataProvider *dataProviderp=NULL;

	// Using MNist testing dataset to do batch predicting on the trained neural network
	MLPPredictor *predictorp=NULL;
	float *inputVectors;
	float *outputVectors;

	netProviderp = new MLPNetProvider("./", MLP_NC_ARCH, MLP_NC_DATA);
	dataProviderp =	new MLPMNistDataProvider(MNIST_PATH, MLP_DATAMODE_PREDICT, minibatch, shuffleBatches);
	dataProviderp->setupDataProvider();                              // set up the data provider

	predictorp = new MLPPredictor(*netProviderp,MLP_OCL_DI_GPU, minibatch);
	outputVectors = new float[predictorp->getOutputVectorSize()*minibatch];

    totalbatches = dataProviderp->getTotalBatches();
    cout << totalbatches << " batches of data to be predicted just waiting ..." << endl;

	getCurrentTime(&startv);

    batches=0;
	while ( dataProviderp->batchAvailable() ) {

		    MLP_CHECK(dataProviderp->getBatchData(predictorp->getBatchSize(),inputVectors,true));

			predictorp->batchPredicting(inputVectors,outputVectors);

            // tell the data provider that I have done with current batch of data, want next batch of data
			MLP_CHECK(dataProviderp->nextBatch());

			batches++;
	}

    getCurrentTime(&endv);

	cout << "Predicting duration: " << diff_msec(&startv, &endv) << " mill-seconds" << endl;
	cout << "Batch size:" << predictorp->getBatchSize() << ", " << batches << " batches predicted" << endl;

	delete [] outputVectors;


	dataProviderp->resetDataProvider();

    // Using MNist testing dataset to do single predicting on the trained neural network
	float *inputVector;
	float *outputVector;

	outputVector = new float[predictorp->getOutputVectorSize()];

    cout << "Please Wait 5 seconds, the single frame predicting will start soon" << endl;
    MLP_SLEEP(5);

	getCurrentTime(&startv);

	totalbatches = dataProviderp->getTotalBatches();

	totalbatches = min<int>(totalbatches, 20);

	batches=0;
    frames=0;
	while ( dataProviderp->batchAvailable() && (batches < totalbatches) ) {

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
    cout << frames <<  " frames of data predicted" << endl;

	delete [] outputVector;

	delete netProviderp;
	delete dataProviderp;
	delete predictorp;
};


static void simple_training()
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
	int shuffleBatches = 50;
	int batches;
	int totalbatches;
	int epoches = 200;

	MLPNetProvider *netProviderp=NULL;
    MLPDataProvider *dataProviderp=NULL;


    // Training the neural network using Simple labelled dataset
	MLPTrainer *trainerp=NULL;

	nettype = NETTYPE_MULTI_CLASSIFICATION;
	netProviderp = new MLPNetProvider(nettype,nLayers,dimensions,etas, momentum, actFuncs,costFunc, true);
	dataProviderp =	new MLPSimpleDataProvider(MLP_DATAMODE_TRAIN,dimensions[0],dimensions[nLayers-1],minibatch,shuffleBatches);
	dataProviderp->setupDataProvider();                            // set up the data provider
    totalbatches = dataProviderp->getTotalBatches();

    trainerp = new MLPTrainer;
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

static void simple_testing()
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
	MLPTester *testerp=NULL;

	netProviderp = new MLPNetProvider("./", MLP_NC_ARCH_NEW, MLP_NC_DATA_NEW);
	dataProviderp =	new MLPSimpleDataProvider(MLP_DATAMODE_TEST,dimensions[0],dimensions[nLayers-1],minibatch,shuffleBatches);
	dataProviderp->setupDataProvider();                              // set up the data provider

	testerp = new MLPTester(*netProviderp,*dataProviderp,MLP_OCL_DI_GPU, minibatch);

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

static void simple_predicting()
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
	MLPPredictor *predictorp=NULL;
	float *inputVectors;
	float *outputVectors;

	netProviderp = new MLPNetProvider("./", MLP_NC_ARCH, MLP_NC_DATA);
    dataProviderp =	new MLPSimpleDataProvider(MLP_DATAMODE_PREDICT,dimensions[0],dimensions[nLayers-1],minibatch,0);
	dataProviderp->setupDataProvider();                               // set up the data provider

	predictorp = new MLPPredictor(*netProviderp,MLP_OCL_DI_GPU, minibatch);
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

static void iflytek_training()
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
	int shuffleBatches = 50;
	int batches;
	int totalbatches;
	int epoches=8;

	MLPNetProvider *netProviderp=NULL;
    MLPDataProvider *dataProviderp=NULL;

	// Training the neural network using MNist labelled dataset
    MLPTrainer *trainerp;

	dataProviderp = new MLPIFlyDataProvider(IFLY_PATH, MLP_DATAMODE_TRAIN, minibatch, shuffleBatches);
	dataProviderp->setupDataProvider();                            // set up the data provider
	dimensions[0] = dataProviderp->getFeatureSize();
	dimensions[nLayers-1] = dataProviderp->getLabelSize();
	totalbatches = dataProviderp->getTotalBatches();

	nettype = NETTYPE_MULTI_CLASSIFICATION;
    netProviderp = new MLPNetProvider(nettype, nLayers, dimensions, etas, momentum, actFuncs, costFunc, true);

    trainerp = new MLPTrainer(*netProviderp,*dataProviderp, MLP_OCL_DI_GPU, minibatch);    // set up the trainer

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

// For testing the MLPTrainer with CheckPointing support
static void iflytek_training3()
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
	int shuffleBatches = 50;
	int batches;
	int totalbatches;
	int epoches=8;
	int startBatch;
	int startEpoch;

	MLPCheckPointManager cpManager;
	MLPNetProvider *netProviderp=NULL;
    MLPDataProvider *dataProviderp=NULL;
    MLPTrainer *trainerp;

	cpManager.cpFindAndLoad("./tmp/");
	if ( cpManager.cpAvailable() ) {
		 struct MLPCheckPointState *statep;

	     cout << "Valid checkpoint found, recover and start new checkpointing from this one"  << endl;

		 statep = cpManager.getChkPointState();
		 netProviderp = new MLPNetProvider(statep->netConfPath, statep->netConfArchFileName, statep->netConfDataFileName);

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

    trainerp = new MLPTrainer(*netProviderp,*dataProviderp, MLP_OCL_DI_GPU, minibatch);

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



static void iflytek_testing()
{
	struct mlp_tv startv, endv;

	int minibatch = 1024;
	int shuffleBatches = 4;
	int totalbatches;

	MLPNetProvider *netProviderp=NULL;
    MLPDataProvider *dataProviderp=NULL;

	// Testing the MNist labelled dataset on the trained neural network
	MLPTester *testerp=NULL;

	netProviderp = new MLPNetProvider("./", MLP_NC_ARCH_NEW, MLP_NC_DATA_NEW);
	dataProviderp =	new MLPIFlyDataProvider(IFLY_PATH, MLP_DATAMODE_TEST, minibatch, shuffleBatches);
	dataProviderp->setupDataProvider();                              // set up the data provider

	testerp = new MLPTester(*netProviderp,*dataProviderp,MLP_OCL_DI_GPU, minibatch);

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

static void iflytek_predicting()
{
	struct mlp_tv startv, endv;

	int minibatch = 1024;
	int batches;
	int totalbatches;
	int frames;

	MLPNetProvider *netProviderp=NULL;
    MLPDataProvider *dataProviderp=NULL;

	// Using MNist testing dataset to do batch predicting on the trained neural network
	MLPPredictor *predictorp=NULL;
	float *inputVectors;
	float *outputVectors;

	netProviderp = new MLPNetProvider("./", MLP_NC_ARCH, MLP_NC_DATA);
	dataProviderp =	new MLPIFlyDataProvider(".", MLP_DATAMODE_PREDICT, minibatch, 0);
	dataProviderp->setupDataProvider();                              // set up the data provider

	predictorp = new MLPPredictor(*netProviderp,MLP_OCL_DI_GPU, minibatch);
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

