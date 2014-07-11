/*
 *  COPYRIGHT:  Copyright (c) 2013 Advanced Micro Devices, Inc.  All rights reserved
 *
 *   Changed by  Qianfeng Zhang@amd.com ( March 2014)
 *
 *   Written by  Junli Gu@amd.com ( Dec 2013 )
 */

#include <algorithm>
#include <clAmdBlas.h>

#include "MLPUtil.h"
#include "MLPCommon.h"
#include "MLPTrainer.h"
#include "MLPChkPointState.h"

//Class specific member shared by all instances
SingleDevClass *MLPTrainer::CLContext = NULL;
int MLPTrainer::nInstances = 0;

MLP_Kerns MLPTrainer::mykerns;

MLPTrainer::MLPTrainer()
{
	this->devType = MLP_OCL_DI_GPU;
	this->setDefault();
	this->dataProviderp = NULL;
	this->initialized = false;
}

MLPTrainer::MLPTrainer(MLPNetProvider & netProvider, MLPDataProvider & dataProvider, MLP_OCL_DEVTYPE dType, int minibatch)
{
	if (  (minibatch != dataProvider.m_batchSize) || dataProvider.dataMode != MLP_DATAMODE_TRAIN ) {
		   mlp_log("MLPTrainer", "The setting of the MLPDataProvider doesn't match the need of the MLPTrainer");
		   MLP_Exception("");
	};

	this->devType = dType;
	this->setDefault();
	this->_initialize(netProvider, minibatch);
	this->dataProviderp = &dataProvider;
	this->initialized = true;
}

MLPTrainer::~MLPTrainer()
{
	if ( --this->nInstances == 0 )
	{
		CL_CHECK( clReleaseKernel(this->mykerns.activate_sigmoid_kernel) );
	    CL_CHECK( clReleaseKernel(this->mykerns.activate_softmax_kernel1) );
        CL_CHECK( clReleaseKernel(this->mykerns.activate_softmax_kernel2) );
	    CL_CHECK( clReleaseKernel(this->mykerns.activate_tanh_kernel) );

		CL_CHECK( clReleaseKernel(this->mykerns.derivative_sigmoid_kernel) );
		CL_CHECK( clReleaseKernel(this->mykerns.derivative_tanh_kernel) );

		CL_CHECK( clReleaseKernel(this->mykerns.calculateError_SSE_kernel1) );
	    CL_CHECK( clReleaseKernel(this->mykerns.calculateError_SSE_kernel2) );
	    CL_CHECK( clReleaseKernel(this->mykerns.calculateError_CE_kernel1) );
        CL_CHECK( clReleaseKernel(this->mykerns.calculateError_CE_kernel2) );

		CL_CHECK( clReleaseKernel(this->mykerns.calculateDelta_SSE_Sigmoid_kernel) );
	    CL_CHECK( clReleaseKernel(this->mykerns.calculateDelta_CE_Softmax_kernel) );

	    CL_CHECK( clReleaseKernel(this->mykerns.transpose_sim_kernel) );
	    CL_CHECK( clReleaseKernel(this->mykerns.transpose_kernel4) );
	    CL_CHECK( clReleaseKernel(this->mykerns.transpose_kernel32) );

		CL_CHECK( clReleaseKernel(this->mykerns.expandMatrix_kernel) );

		CL_CHECK( clReleaseProgram(this->CLContext->m_program) );

		delete this->CLContext;

		clAmdBlasTeardown();

	}

	this->_dispose();
}

// only called by the constructor
void MLPTrainer::setDefault()
{
	cl_int status;

	// class wide setting up
	if ( this->nInstances++ == 0 )  // first instance
	{

        this->CLContext = new SingleDevClass(this->devType);

		clAmdBlasSetup();

		char *kernel_src;
	    if ( read_srcfile("kernels.cl", kernel_src) < 0 ) {
		     mlp_log("MLPTrainer", "Failed to read kernel source file\n");
		     MLP_Exception("");
		     return;
	    };

		this->CLContext->m_program = clCreateProgramWithSource(this->CLContext->m_context, 1, (const char**)&kernel_src,NULL,&status);
		CL_CHECK( status );

		CL_CHECK( clBuildProgram(this->CLContext->m_program,1,&this->CLContext->m_device, NULL, NULL, NULL) );

		delete [] kernel_src;

		this->mykerns.activate_sigmoid_kernel = clCreateKernel(this->CLContext->m_program,"activate_sigmoid",&status);
		CL_CHECK( status );
		this->mykerns.activate_softmax_kernel1 = clCreateKernel(this->CLContext->m_program,"activate_softmax1",&status);
		CL_CHECK( status );
		this->mykerns.activate_softmax_kernel2 = clCreateKernel(this->CLContext->m_program,"activate_softmax2",&status);
		CL_CHECK( status );
		this->mykerns.activate_tanh_kernel = clCreateKernel(this->CLContext->m_program,"activate_tanh",&status);
		CL_CHECK( status );

		this->mykerns.derivative_sigmoid_kernel = clCreateKernel(this->CLContext->m_program,"derivative_sigmoid",&status);
		CL_CHECK( status );
		this->mykerns.derivative_tanh_kernel = clCreateKernel(this->CLContext->m_program,"derivative_tanh",&status);
		CL_CHECK( status );

		this->mykerns.calculateError_SSE_kernel1 = clCreateKernel(this->CLContext->m_program,"calculateError_SSE1",&status);
		CL_CHECK( status );
		this->mykerns.calculateError_SSE_kernel2 = clCreateKernel(this->CLContext->m_program,"calculateError_SSE2",&status);
		CL_CHECK( status );
		this->mykerns.calculateError_CE_kernel1 = clCreateKernel(this->CLContext->m_program,"calculateError_CE1",&status);
		CL_CHECK( status );
		this->mykerns.calculateError_CE_kernel2 = clCreateKernel(this->CLContext->m_program,"calculateError_CE2",&status);
		CL_CHECK( status );

		this->mykerns.calculateDelta_SSE_Sigmoid_kernel = clCreateKernel(this->CLContext->m_program,"calculateDelta_SSE_Sigmoid",&status);
		CL_CHECK( status );
		this->mykerns.calculateDelta_CE_Softmax_kernel = clCreateKernel(this->CLContext->m_program,"calculateDelta_CE_Softmax",&status);
		CL_CHECK( status );

        this->mykerns.transpose_sim_kernel = clCreateKernel(this->CLContext->m_program,"transpose_simple",&status);
	    CL_CHECK( status );
        this->mykerns.transpose_kernel4 = clCreateKernel(this->CLContext->m_program,"transpose_f4",&status);
	    CL_CHECK( status );
        this->mykerns.transpose_kernel32 = clCreateKernel(this->CLContext->m_program,"transpose_32x32",&status);
	    CL_CHECK( status );

        this->mykerns.expandMatrix_kernel = clCreateKernel(this->CLContext->m_program,"expandVectorToMatrix",&status);
	    CL_CHECK( status );
	}

	// setting up needed by the instance only
	this->dimensions = NULL;
	this->nLayers = 0;
	this->minibatch = 0;

	this->inputs = NULL;
	this->weightT = NULL;
	this->output = NULL;
	this->target = NULL;
	this->delta = NULL;

	this->etas = NULL;
	this->actFuncs = NULL;

	this->currBatchNo = 0;

	MLP_LOCK_INIT(&this->chkPointingLock);
}

// only called by the constructor and setupMLP()
void MLPTrainer::_initialize(MLPNetProvider & provider, int _minibatch)
{
	cl_int status;

	this->netType = provider.netType;
	this->nLayers = provider.nLayers;
	this->minibatch = _minibatch;
	this->dimensions = new int[this->nLayers];
	this->inputs = new cl_mem[this->nLayers];         // buffers for storing the input/output for each layers
	this->weightT = new cl_mem[this->nLayers];        // weights for connecting the previous layer and current layer
	this->biases =  new cl_mem[this->nLayers];        // bias for each layer, added to the input of each layer
	this->delta = new cl_mem[this->nLayers];          // delta for each layer, used by back propagation
	this->etas = new cl_float[this->nLayers];         // learning rate for each layer
	this->actFuncs = new ACT_FUNC[this->nLayers];     // activating function for each layer

	for ( int i = 0; i < this->nLayers; i++ )
		this->dimensions[i] = provider.dimensions[i];

	for ( int i = 0; i < this->nLayers; i++ )
		this->etas[i] = provider.etas[i];

	for ( int i = 0; i < this->nLayers; i++ )
		this->actFuncs[i] = provider.actFuncs[i];

	this->costFunc = provider.costFunc;
	this->momentum = provider.momentum;

	// The Input/Output of layer i is stored in this->inputs[i+1], so this->inputs[1] is for the input layer, this->inputs[2] is for
	// the first hidden layer, this->inputs[0] is for the output layer
	for (int i = 1; i < this->nLayers; i++ )
	{
		this->inputs[i] = clCreateBuffer(this->CLContext->m_context, CL_MEM_READ_WRITE, sizeof(cl_float)*this->dimensions[i-1]*this->minibatch,NULL,&status);
		CL_CHECK(status);

		this->weightT[i] = clCreateBuffer(this->CLContext->m_context, CL_MEM_READ_WRITE, sizeof(cl_float)*this->dimensions[i-1]*this->dimensions[i], NULL,&status);
		CL_CHECK(status);

		cl_mem tmpBuff;

		tmpBuff = clCreateBuffer(this->CLContext->m_context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, sizeof(cl_float)*this->dimensions[i-1]*this->dimensions[i],
			                               provider.weights[i],&status);
		CL_CHECK(status);
		transpose_float_matrix(tmpBuff, this->weightT[i], this->dimensions[i], this->dimensions[i-1]);  // make this->weightT[i] in transposed format
		clReleaseMemObject(tmpBuff);


		this->biases[i] = clCreateBuffer(this->CLContext->m_context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, sizeof(cl_float)*this->dimensions[i],
			                               provider.biases[i],&status);
		CL_CHECK(status);

		this->delta[i] = clCreateBuffer(this->CLContext->m_context, CL_MEM_READ_WRITE, sizeof(cl_float)*this->dimensions[i]*this->minibatch,NULL,&status);
		CL_CHECK(status);
	}

	// for output layer
	this->output = clCreateBuffer(this->CLContext->m_context, CL_MEM_READ_WRITE, sizeof(cl_float)*(this->dimensions[this->nLayers-1])*this->minibatch,NULL,&status);
	CL_CHECK(status);

	this->inputs[0] = this->output;

	this->target = clCreateBuffer(this->CLContext->m_context, CL_MEM_READ_WRITE, sizeof(cl_float)*this->dimensions[this->nLayers-1]*this->minibatch,NULL,&status);
	CL_CHECK(status);
}

// only called by the destructor
void MLPTrainer::_dispose()
{
	if ( this->nLayers == 0 )    // nothing to be released
		return;

	for (int i = 1; i < this->nLayers; i++ )
	{
		CL_CHECK( clReleaseMemObject(this->inputs[i]) );
		CL_CHECK( clReleaseMemObject(this->weightT[i]) );
		CL_CHECK( clReleaseMemObject(this->biases[i]) );
		CL_CHECK( clReleaseMemObject(this->delta[i]) );
	}

	CL_CHECK( clReleaseMemObject(this->output) );
	CL_CHECK( clReleaseMemObject(this->target) );

	if ( this->dimensions )
		delete [] this->dimensions;
	if ( this->etas )
		delete [] this->etas;
	if ( this->actFuncs )
		delete [] this->actFuncs;
	if ( this->inputs )
		delete [] this->inputs;
	if ( this->weightT )
		delete [] this->weightT;
	if ( this->delta )
		delete [] this->delta;
	if ( this->biases)
		delete [] this->biases;
}

void MLPTrainer::setupMLP(MLPNetProvider & netProvider, MLPDataProvider & dataProvider, int minibatch)
{
	if (  (minibatch != dataProvider.m_batchSize) || dataProvider.dataMode != MLP_DATAMODE_TRAIN ) {
		   mlp_log("MLPTrainer", "The setting of the MLPDataProvider doesn't match the need of the MLPTrainer");
		   MLP_Exception("");
	};

	this->_initialize(netProvider, minibatch);
	this->dataProviderp = &dataProvider;
	this->initialized = true;
}

MLPDataProvider *MLPTrainer::getDataProvider()
{
	return(this->dataProviderp);
};

void MLPTrainer::saveNetConfig(const char *configPath)
{
	MLPNetProvider  netProvider(this->nLayers,this->dimensions,false);

	this->synchronizeNetConfig(netProvider);

	netProvider.saveConfig(configPath, MLP_NC_ARCH_NEW, MLP_NC_DATA_NEW);
};

void MLPTrainer::synchronizeNetConfig(MLPNetProvider &netProvider)
{
	cl_int status;

	for ( int i = 0; i < this->nLayers; i++ )
		 netProvider.etas[i] = this->etas[i];

	for ( int i = 0; i < this->nLayers; i++ )
		 netProvider.actFuncs[i] = this->actFuncs[i];

	netProvider.netType = this->netType;
	netProvider.costFunc = this->costFunc;
	netProvider.momentum = this->momentum;

	// The Input/Output of layer i is stored in this->inputs[i+1], so this->inputs[1] is for the input layer, this->inputs[2] is for
	// the first hidden layer, this->inputs[0] is for the output layer
	for (int i = 1; i < this->nLayers; i++ )
	{
		cl_mem tmpBuff;

	    tmpBuff = clCreateBuffer(this->CLContext->m_context, CL_MEM_READ_WRITE, sizeof(cl_float)*this->dimensions[i-1]*this->dimensions[i], NULL,&status);
		this->transpose_float_matrix(this->weightT[i], tmpBuff, this->dimensions[i-1], this->dimensions[i]);    // make tmpBuff in original format
		status = clEnqueueReadBuffer(this->CLContext->m_cmd_queues[0], tmpBuff, CL_TRUE, 0,sizeof(cl_float)*this->dimensions[i-1]*this->dimensions[i],
			                         netProvider.weights[i], 0, NULL, NULL);
		CL_CHECK(status);
        clReleaseMemObject(tmpBuff);

		status = clEnqueueReadBuffer(this->CLContext->m_cmd_queues[0], this->biases[i], CL_TRUE, 0,sizeof(cl_float)*this->dimensions[i],
			                         netProvider.biases[i], 0, NULL, NULL);
		CL_CHECK(status);
	}
};

void MLPTrainer::showNetConfig()
{
	cl_int status;

	MLPNetProvider  netProvider(this->nLayers,this->dimensions,false);

    netProvider.netType = this->netType;
	netProvider.costFunc = this->costFunc;
	netProvider.momentum = this->momentum;

	for ( int i = 0; i < this->nLayers; i++ )
		 netProvider.etas[i] = this->etas[i];

	for ( int i = 0; i < this->nLayers; i++ )
		 netProvider.actFuncs[i] = this->actFuncs[i];

	// The Input/Output of layer i is stored in this->inputs[i+1], so this->inputs[1] is for the input layer, this->inputs[2] is for
	// the first hidden layer, this->inputs[0] is for the output layer
	for (int i = 1; i < this->nLayers; i++ )
	{
		cl_mem tmpBuff;

	    tmpBuff = clCreateBuffer(this->CLContext->m_context, CL_MEM_READ_WRITE, sizeof(cl_float)*this->dimensions[i-1]*this->dimensions[i], NULL,&status);
		this->transpose_float_matrix(this->weightT[i], tmpBuff, this->dimensions[i-1], this->dimensions[i]);    // make tmpBuff in original format
		status = clEnqueueReadBuffer(this->CLContext->m_cmd_queues[0], tmpBuff, CL_TRUE, 0,sizeof(cl_float)*this->dimensions[i-1]*this->dimensions[i],
			                         netProvider.weights[i], 0, NULL, NULL);
		CL_CHECK(status);
        clReleaseMemObject(tmpBuff);


		status = clEnqueueReadBuffer(this->CLContext->m_cmd_queues[0], this->biases[i], CL_TRUE, 0,sizeof(cl_float)*this->dimensions[i],
			                         netProvider.biases[i], 0, NULL, NULL);
		CL_CHECK(status);
	}

	netProvider.showConfig();
};

// the following interfaces make calls to OpenCL kernels

void MLPTrainer::expandFloatVectorToMatrix(cl_mem  myVector, cl_mem myMatrix, int width, int height)
{
	cmn_expandFloatVectorToMatrix(this->CLContext->m_cmd_queues[0],this->mykerns, myVector, myMatrix, width, height);
};

// use three different method to implement transposition depending on the size of the width and height
void MLPTrainer::transpose_float_matrix(cl_mem src, cl_mem dst, cl_int width, cl_int height)
{
	if  ( (width % 32 == 0) && (height % 32 == 0) )
            cmn_transpose_matrix_32x32(this->CLContext->m_cmd_queues[0],this->mykerns,src,dst,width,height);
	else
	    if (width % 4 == 0)
		    cmn_transpose_matrix_f4(this->CLContext->m_cmd_queues[0],this->mykerns,src,dst,width,height);
		else
			cmn_transpose_matrix_simple(this->CLContext->m_cmd_queues[0],this->mykerns,src,dst,width,height);
};


void MLPTrainer::activate(int layer, cl_mem x, cl_mem y, int width, int height )
{
	switch (this->actFuncs[layer] ) {
	case AFUNC_SIGMOID:
		cmn_activate_sigmoid(this->CLContext->m_cmd_queues[0],this->mykerns,x,y,width,height);
		return;
	case AFUNC_SOFTMAX:
	    cmn_activate_softmax(this->CLContext->m_cmd_queues[0],this->mykerns,x,y,width,height);
		return;
	case AFUNC_TANH:
	    cmn_activate_tanh(this->CLContext->m_cmd_queues[0],this->mykerns,x,y,width,height);
		return;
	case AFUNC_IDENTITY:
	    cmn_activate_identity(this->CLContext->m_cmd_queues[0],this->mykerns,x,y,width,height);
        return;
	default:
		mlp_log("MLPTrainer", "The assigned activation function for this layer is not supported.");
		MLP_Exception("");
	};
};


void MLPTrainer::calculateError(cl_mem output, cl_mem target, int width, int height, float &ret )
{
	switch (this->costFunc) {
	case CFUNC_SSE:
		cmn_calculateError_SSE(this->CLContext->m_cmd_queues[0],this->mykerns,output,target,this->reduceMem,this->reduceBuff,width,height,ret);
		return;
	case CFUNC_CE:
		cmn_calculateError_CE(this->CLContext->m_cmd_queues[0],this->mykerns,output,target,this->reduceMem,this->reduceBuff,width,height,ret);
		return;
	default:
		mlp_log("MLPTrainer", "The assigned cost function for this neural network is not supported.");
		MLP_Exception("");
	};
	return;
};


void MLPTrainer::calculateDelta(cl_mem output, cl_mem target, cl_mem delta, int width, int height)
{
	if ( (this->costFunc == CFUNC_CE) && (this->actFuncs[this->nLayers-1] == AFUNC_SOFTMAX) ) {
		 cmn_calculateDelta_CE_Softmax(this->CLContext->m_cmd_queues[0],this->mykerns,output,target,delta,width,height);
		 return;
	};
	if ( (this->costFunc == CFUNC_SSE) && (this->actFuncs[this->nLayers-1] == AFUNC_SIGMOID) ) {
		 cmn_calculateDelta_SSE_Sigmoid(this->CLContext->m_cmd_queues[0],this->mykerns,output,target,delta,width,height);
		 return;
	};

	mlp_log("MLPTrainer", "The configuration for this neural network is not supported");
	MLP_Exception("");
	return;
};


void MLPTrainer::derivative(int layer, cl_mem delta1, cl_mem y, cl_mem delta2, int width, int height )
{
	switch (this->actFuncs[layer] ) {
	case AFUNC_SIGMOID:
		cmn_derivative_sigmoid(this->CLContext->m_cmd_queues[0],this->mykerns,delta1,y,delta2,width,height);
		return;
	case AFUNC_TANH:
		cmn_derivative_tanh(this->CLContext->m_cmd_queues[0],this->mykerns,delta1,y,delta2,width,height);
		return;
	default:
		mlp_log("MLPTrainer", "The assigned activation function for this layer is not supported.");
		MLP_Exception("");
	};
};

void MLPTrainer::checkPointing(struct MLPCheckPointState &cpState)
{

	 MLP_LOCK(&this->chkPointingLock);          // need be lock protected from the Training of MLPTrainer

     // Snapshot one value of BatchNo as the state checkpointed from the MLPTrainer
     cpState.cpBatchNo = (unsigned int) this->currBatchNo;
	 cpState.cpEpoch = (unsigned int) this->currEpoch;

     // Snapshot one value of FrameNo as the state checkpointed from the MLPDataProvider
	 int frameNo;
     this->dataProviderp->getCheckPointFrame(frameNo);
	 cpState.cpFrameNo = (unsigned int) frameNo;

     // Snapshot one state of network configuration from the MLPTrainer, and save it to the files
     MLPNetProvider  netProvider(this->nLayers,this->dimensions,false);
     this->synchronizeNetConfig(netProvider);
     netProvider.saveConfig(cpState.netConfPath, cpState.netConfArchFileName, cpState.netConfDataFileName);

     MLP_UNLOCK(&this->chkPointingLock);
}

