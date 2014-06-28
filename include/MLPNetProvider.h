/*
 *  COPYRIGHT:  Copyright (c) 2014 Advanced Micro Devices, Inc.  All rights reserved
 *
 *   Written by Qianfeng Zhang@amd.com ( March 2014 )
 *
 */

#ifndef _MLP_NET_PROVIDER_H_
#define _MLP_NET_PROVIDER_H_

#include "MLPApiExport.h"

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

// Implements the MLP network used by the MLPTrainer/MLPTester/MLPPredictor 
class MLPNetProvider 
{
	friend class MLPTrainer;   // add more friend class here
	friend class MLPTester; 
	friend class MLPPredictor; 
private:
	MLP_NETTYPE netType; 
	int nLayers; 
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
	LIBMLPAPI MLPNetProvider();  
    LIBMLPAPI MLPNetProvider(int layers, int dimensions[], bool DoInitialize=false);
    LIBMLPAPI MLPNetProvider(MLP_NETTYPE type, int layers, int dimensions_[], float etas_[], float momentum_, ACT_FUNC actFuncs_[], COST_FUNC costFunc_, bool DoInitialize);
	LIBMLPAPI MLPNetProvider(const char *configPath, const char *archFilename, const char *weightFilename);

	LIBMLPAPI ~MLPNetProvider(); 

	void saveConfig(const char *configPath, const char *archFilename, const char *weightFilename); 
	void showConfig(); 
};

#define MLP_NC_ARCH  "mlp_netarch.conf"           // Text file to define the MLP neural network architecture, this file is modifiable manually
#define MLP_NC_DATA  "mlp_netweights.dat"         // Binary file to save the weights and biases of the MLP net, not modifiable manually
#define MLP_NC_ARCH_NEW "mlp_netarch_new.conf"
#define MLP_NC_DATA_NEW "mlp_netweights_new.dat"  

#endif

