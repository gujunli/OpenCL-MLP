/*
 *  COPYRIGHT:  Copyright (c) 2014 Advanced Micro Devices, Inc.  All rights reserved
 *
 *   Written by Qianfeng Zhang@amd.com ( March 2014 )
 *
 */

#ifndef _MLP_CONFIG_PROVIDER_H_
#define _MLP_CONFIG_PROVIDER_H_

#include "DNNApiExport.h"

enum  MLP_NETTYPE
{
   NETTYPE_BIN_CLASSIFICATION,
   NETTYPE_MULTI_CLASSIFICATION,
   NETTYPE_LINEAR_REGRESSION,
   NETNOTYPE
};

enum  ACT_FUNC
{
   AFUNC_SIGMOID,
   AFUNC_TANH,              // hyperbolic tangent
   AFUNC_RELU,
   AFUNC_SOFTMAX,
   AFUNC_IDENTITY,          // only meaningful for the output layer
   ANOFUNC
};

enum COST_FUNC
{
   CFUNC_SSE,
   CFUNC_CE,
   CNOFUNC
};

#define MLP_NNET_MAX_LAYERS 16

struct mlp_nnet_layer_desc {
    unsigned int dimension;         // the number of neurons of this layer
    char activation[16];            // the name of the activation function used by this layer
};

struct mlp_nnet_data_header {
    char  nnet_type[32];                               // a typename describing the neural network, might not be used right now, but reserved
    unsigned int  nLayers;                             // the total number of layers of the neural network including the input layer
    struct mlp_nnet_layer_desc layers[MLP_NNET_MAX_LAYERS];
    unsigned int weight_offsets[MLP_NNET_MAX_LAYERS];  // offset of the weights matrix and bias vector for each layer(at 1024-byte boundary)
};


// Implements the MLP neural network configuration used by the MLPTrainer/MLPTester/MLPPredictor
class MLPConfigProvider
{
	friend class MLPTrainerBase;   // add more friend class here
	friend class MLPTesterBase;
	friend class MLPPredictorBase;
	friend class MLPTrainerOCL;
	friend class MLPTesterOCL;
	friend class MLPPredictorOCL;
private:
	MLP_NETTYPE netType;
	int nLayers;
	int epochs; 
	int *dimensions;
	float *etas;
	float **biases;
	float **weights;
	float momentum;
	ACT_FUNC *actFuncs;
	COST_FUNC costFunc;

private:
	void biasesInitialize();
	void weightsInitialize();
	void etasInitialize();
	void actFuncsInitialize();

public:
	LIBDNNAPI MLPConfigProvider();
    LIBDNNAPI MLPConfigProvider(int layers, int dimensions[], bool DoInitialize=false);
    LIBDNNAPI MLPConfigProvider(MLP_NETTYPE type, int layers, int dimensions_[], float etas_[], float momentum_, ACT_FUNC actFuncs_[], COST_FUNC costFunc_, int _epochs, bool DoInitialize);
    LIBDNNAPI MLPConfigProvider(const char *dir, const char *trainingConfigFile, const char *nnetDataFile);   // initialized weights from the file
	LIBDNNAPI MLPConfigProvider(const char *dir, const char *trainingConfigFile, bool DoInitialize);         // using randomly initialized weights

	LIBDNNAPI MLPConfigProvider(const char *dir, const char *nnetDataFile);   // used for the MLPTester and MLPPredictor

	LIBDNNAPI ~MLPConfigProvider();

	LIBDNNAPI void saveConfig(const char *dir, const char *trainingConfigFile, const char *nnetDataFile);

	LIBDNNAPI void showConfig();

	LIBDNNAPI int getInputLayerSize();
	LIBDNNAPI int getOutputLayerSize();
	LIBDNNAPI int getLayerSize(int layer); 
	LIBDNNAPI void setLayerWeights(int layer, int layerSize, int lowerLayerSize, float *weights); 
	LIBDNNAPI void setLayerBiases(int layer, int layerSize, float *biases); 
};

#define MLP_CP_TRAINING_CONF "mlp_training.conf"
#define MLP_CP_NNET_DATA "mlp_nnet.dat"
#define MLP_CP_TRAINING_CONF_NEW "mlp_training_new.conf"
#define MLP_CP_NNET_DATA_NEW "mlp_nnet_new.dat"


#endif

