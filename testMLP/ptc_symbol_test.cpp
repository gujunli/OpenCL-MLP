/*
 *  COPYRIGHT:  Copyright (c) 2013 Advanced Micro Devices, Inc.  All rights reserved
 *
 *   Changed by:         Qianfeng Zhang@amd.com ( March 2014 )
 *
 */

#include <iostream>

#include "MLPUtil.h"
#include "MLPTrainerOCL.h"
#include "MLPTesterOCL.h"
#include "MLPPredictorOCL.h"
#include "MLPConfigProvider.h"
#include "MLPChkPointingMgr.h"
#include "DNNPtcDataProvider.h"

using namespace std;

#define PTC_SYMBOL_DB_PATH "../../ptc_dataset/Symbol-ptc/"

void ptc_symbol_training();
void ptc_symbol_training2();
void ptc_symbol_training3();     // training with checkpointing support
void ptc_symbol_batch_testing();
void ptc_symbol_single_testing();
void ptc_symbol_predicting();

static bool use_stats = false;

// do ptc_en training with randomly initialized weights
void ptc_symbol_training()
{
	struct dnn_tv startv, endv;

	int minibatch = 1024;
	int shuffleBatches = 10;
	int batches;
	int totalbatches;

	MLPConfigProvider *configProviderp=NULL;
    DNNDataProvider *dataProviderp=NULL;

    MLPTrainerBase *trainerp;

	dataProviderp = new DNNPtcDataProvider(PTC_SYMBOL_DB_PATH, use_stats, DNN_DATAMODE_SP_TRAIN, minibatch, shuffleBatches);
	dataProviderp->setupDataProvider();                            // set up the data provider

	totalbatches = dataProviderp->getTotalBatches();

    configProviderp = new MLPConfigProvider("./", "mlp_training_init.conf", true);

    trainerp = new MLPTrainerOCL(*configProviderp,*dataProviderp, DNN_OCL_DI_GPU, minibatch);    // set up the trainer

	cout << totalbatches << " batches of data to be trained with " << trainerp->getEpochs() << " epoches, just waiting..." << endl;

	getCurrentTime(&startv);
	batches = trainerp->batchTraining(0);                                       // do the training
	getCurrentTime(&endv);

	cout << batches << " batches of data were trained actually" << endl;
    cout << "Training duration: " << diff_msec(&startv, &endv) << " mill-seconds" << endl;


    // Finalize the result from the training work, so that the Tester or Predictor can be set up based on it
	trainerp->saveNetConfig("./");

	delete configProviderp;
	delete dataProviderp;
	delete trainerp;
};

// doing ptc_en training based on pre-trained weights
void ptc_symbol_training2()
{
	struct dnn_tv startv, endv;

	int minibatch = 1024;
	int shuffleBatches = 10;
	int batches;
	int totalbatches;

	MLPConfigProvider *configProviderp=NULL;
    DNNDataProvider *dataProviderp=NULL;


	// Training the neural network using ptc_en labelled dataset
    MLPTrainerBase *trainerp;

	dataProviderp = new DNNPtcDataProvider(PTC_SYMBOL_DB_PATH,  use_stats, DNN_DATAMODE_SP_TRAIN, minibatch, shuffleBatches);
	dataProviderp->setupDataProvider();                            // set up the data provider
	totalbatches = dataProviderp->getTotalBatches();

	configProviderp = new MLPConfigProvider("./", "mlp_training_init.conf", "mlp_nnet_init.dat");

    trainerp = new MLPTrainerOCL(*configProviderp,*dataProviderp, DNN_OCL_DI_GPU, minibatch);

	cout << totalbatches << " batches of data to be trained with " << trainerp->getEpochs() << " epoches, just waiting..." << endl;

	getCurrentTime(&startv);
	batches = trainerp->batchTraining(0);
	getCurrentTime(&endv);

	cout << batches << " batches of data were trained actually" << endl;
    cout << "Training duration: " << diff_msec(&startv, &endv) << " mill-seconds" << endl;


    // Finalize the result from the training work, so that the Tester or Predictor can be set up based on it
	trainerp->saveNetConfig("./");

	delete configProviderp;
	delete dataProviderp;
	delete trainerp;
};

// doing ptc_en training with checkpointing support
void ptc_symbol_training3()
{
	struct dnn_tv startv, endv;

	int minibatch = 1024;
	int shuffleBatches = 10;
	int batches;
	int totalbatches;
	int startBatch;
	int startEpoch;


	MLPCheckPointManager cpManager;
	MLPConfigProvider *configProviderp=NULL;
    DNNDataProvider *dataProviderp=NULL;

	cpManager.cpFindAndLoad("./tmp/");
	if ( cpManager.cpAvailable() ) {
		 struct MLPCheckPointState *statep;

		 cout << "Valid checkpoint found, recover and start new checkpointing from this one" << endl;

		 statep = cpManager.getChkPointState();
		 configProviderp = new MLPConfigProvider(statep->netConfPath, statep->ncTrainingConfigFname, statep->ncNNetDataFname);

         dataProviderp = new DNNPtcDataProvider(PTC_SYMBOL_DB_PATH, use_stats, DNN_DATAMODE_SP_TRAIN, minibatch, shuffleBatches);
		 dataProviderp->setupDataProvider(statep->cpFrameNo, true);

		 startBatch = statep->cpBatchNo;
		 startEpoch = statep->cpEpoch;

		 cpManager.cpUnload();
	}
	else {
		 cout << "No old checkpoint found, start new checkpointing any way" << endl;

          configProviderp = new MLPConfigProvider("./", "mlp_training_init.conf", "mlp_nnet_init.dat");

	     dataProviderp = new DNNPtcDataProvider(PTC_SYMBOL_DB_PATH, use_stats, DNN_DATAMODE_SP_TRAIN, minibatch, shuffleBatches);
	     dataProviderp->setupDataProvider(0, true);

		 startBatch = 0;
		 startEpoch = 0;
	};

	// Training the neural network using ptc_en labelled dataset
    MLPTrainerBase *trainerp;

    trainerp = new MLPTrainerOCL(*configProviderp,*dataProviderp, DNN_OCL_DI_GPU, minibatch);

	cpManager.enableCheckPointing(*trainerp, "./tmp/");
	MLP_CHECK( cpManager.startCheckPointing() );

	totalbatches = dataProviderp->getTotalBatches();

	cout << totalbatches << " batches of data to be trained with " << trainerp->getEpochs() << " epoches, just waiting..." << endl;

	getCurrentTime(&startv);
	batches = trainerp->batchTrainingWithCheckPointing(0, startBatch, startEpoch, true);
	getCurrentTime(&endv);

	MLP_CHECK( cpManager.endCheckPointing() );

	cout << batches << " batches of data were trained actually" << endl;
    cout << "Training duration: " << diff_msec(&startv, &endv) << " mill-seconds" << endl;

    // Finalize the result from the training work, so that the Tester or Predictor can be set up based on it
	trainerp->saveNetConfig("./");

	cpManager.cpCleanUp("./tmp/");

	delete configProviderp;
	delete dataProviderp;
	delete trainerp;
};

void ptc_symbol_batch_testing()
{
	struct dnn_tv startv, endv;

	// 285 * 5 = 1425 to match the number of samples in the testing set
	int minibatch = 285;
	int shuffleBatches = 5;
	int totalbatches;

	MLPConfigProvider *configProviderp=NULL;
    DNNDataProvider *dataProviderp=NULL;

	// Testing the ptc_en labelled dataset on the trained neural network
	MLPTesterBase *testerp=NULL;

	configProviderp = new MLPConfigProvider("./", MLP_CP_NNET_DATA_NEW);
	dataProviderp =	new DNNPtcDataProvider(PTC_SYMBOL_DB_PATH, use_stats, DNN_DATAMODE_TEST, minibatch, shuffleBatches);
	dataProviderp->setupDataProvider();                              // set up the data provider

	testerp = new MLPTesterOCL(*configProviderp,*dataProviderp,DNN_OCL_DI_GPU,minibatch);

    totalbatches = dataProviderp->getTotalBatches();

    cout << "batch testing, batches of data to be tested just waiting ..." << endl;

	getCurrentTime(&startv);
	testerp->batchTesting(totalbatches);
	getCurrentTime(&endv);

    cout << "Testing duration: " << diff_msec(&startv, &endv) << " mill-seconds" << endl;

	int totalNum, succNum;
	testerp->getTestingStats(totalNum, succNum);
	cout << totalNum << " frames tested," << succNum << " frames succeed, success ratio is " << ((float)succNum)*100.0f/((float)totalNum) << "%" << endl;

    delete dataProviderp;
	delete configProviderp;
	delete testerp;
};

static bool ptc_symbol_output_matching(float *frameOutput, float *frameLabel, int len)
{
	float element;

	for (int i=0; i< len; i++) {
		 element = (frameOutput[i]<0.5)?0.0f:1.0f;

		 if ( element != frameLabel[i] )
			  return(false);
	};

	return(true);
};

void ptc_symbol_single_testing()
{
	struct dnn_tv startv, endv;

	int minibatch = 500;
	int shuffleBatches = 10;
	int totalbatches, batches;
	int inVecLen, outVecLen;
	int frames, succNum;

	MLPConfigProvider *configProviderp=NULL;
    DNNDataProvider *dataProviderp=NULL;

	// Testing the ptc_en labelled dataset on the trained neural network
	MLPTesterBase *testerp=NULL;

	configProviderp = new MLPConfigProvider("./", MLP_CP_NNET_DATA_NEW);
	dataProviderp =	new DNNPtcDataProvider(PTC_SYMBOL_DB_PATH,  use_stats, DNN_DATAMODE_TEST, minibatch, shuffleBatches);
	dataProviderp->setupDataProvider();                              // set up the data provider

	testerp = new MLPTesterOCL(*configProviderp,*dataProviderp,DNN_OCL_DI_GPU,minibatch);

    totalbatches = dataProviderp->getTotalBatches();
    totalbatches = min<int>(totalbatches, 2);

    inVecLen = testerp->getInputVectorSize();
    outVecLen = testerp->getOutputVectorSize();

    // Using ptc_en testing dataset to do single testing on the trained neural network
	float *inputVectors;
	float *labelVectors;

    cout << "Single testing, batches of data to be tested just waiting ..." << endl;

	getCurrentTime(&startv);

	batches=0;
    frames=0;
    succNum=0;

	while ( dataProviderp->batchAvailable() && (batches < totalbatches) ) {

		    MLP_CHECK(dataProviderp->getBatchData(testerp->getBatchSize(),inputVectors, labelVectors, true));

			for (int i=0; i< testerp->getBatchSize(); i++) {
				 if ( testerp->singleTesting(&inputVectors[i*inVecLen], &labelVectors[i*outVecLen], ptc_symbol_output_matching) )
                       succNum++;
				 frames++;
			};

            // tell the data provider that I have done with current batch of data, want next batch of data
		 	MLP_CHECK(dataProviderp->nextBatch());

			batches++;
	}

    getCurrentTime(&endv);

	cout << "Testing duration: " << diff_msec(&startv, &endv) << " mill-seconds" << endl;
	cout << frames << " frames tested," << succNum << " frames succeed, success ratio is " << ((float)succNum)*100.0f/((float)frames) << "%" << endl;

	delete configProviderp;
	delete dataProviderp;
	delete testerp;
};


void ptc_symbol_predicting()
{
	struct dnn_tv startv, endv;

	int minibatch = 1024;
    int shuffleBatches = 4;
	int batches;
	int totalbatches;
	int frames;

	MLPConfigProvider *configProviderp=NULL;
    DNNDataProvider *dataProviderp=NULL;

	// Using ptc_en testing dataset to do batch predicting on the trained neural network
	MLPPredictorBase *predictorp=NULL;
	float *inputVectors;
	float *outputVectors;

	configProviderp = new MLPConfigProvider("./", MLP_CP_NNET_DATA_NEW);
	dataProviderp =	new DNNPtcDataProvider(PTC_SYMBOL_DB_PATH,  use_stats, DNN_DATAMODE_PREDICT, minibatch, shuffleBatches);
	dataProviderp->setupDataProvider();                              // set up the data provider

	predictorp = new MLPPredictorOCL(*configProviderp,DNN_OCL_DI_GPU, minibatch);
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

    // Using ptc_en testing dataset to do single predicting on the trained neural network
	float *inputVector;
	float *outputVector;

	outputVector = new float[predictorp->getOutputVectorSize()];

    cout << "Please Wait 5 seconds, the single frame predicting will start soon" << endl;
    DNN_SLEEP(5);

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

	delete configProviderp;
	delete dataProviderp;
	delete predictorp;
};


