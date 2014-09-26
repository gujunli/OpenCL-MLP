/*
 *  COPYRIGHT:  Copyright (c) 2013 Advanced Micro Devices, Inc.  All rights reserved
 *
 *   Written by  Qianfeng Zhang@amd.com ( March 2014)
 *
 *   Written by  Junli Gu@amd.com ( Dec 2013 )
 */

#include <algorithm>
#include <clAmdBlas.h>

#include "MLPUtil.h"
#include "MLPOclCommon.h"
#include "MLPTrainerOCL.h"
#include "MLPChkPointState.h"

//Class specific member shared by all instances
SingleDevClass *MLPTrainerOCL::CLCtx = NULL;
int MLPTrainerOCL::nInstances = 0;

MLP_Kerns MLPTrainerOCL::mykerns;

MLPTrainerOCL::MLPTrainerOCL()
{
	// class wide set up
	if ( this->nInstances++ == 0 )  // first instance
	{

        this->CLCtx = new SingleDevClass(this->devType);

		clAmdBlasSetup();

		this->setup_ocl_kernels();

	}

	this->devType = DNN_OCL_DI_GPU;    // default OpenCL device

	this->inputs = NULL;
	this->weightT = NULL;
	this->output = NULL;
	this->target = NULL;

	this->initialized = false;
};

MLPTrainerOCL::MLPTrainerOCL(MLPNetProvider & netProvider, DNNDataProvider & dataProvider, DNN_OCL_DEVTYPE dType, int _minibatch)
{
   	this->devType = dType;

	// class wide set up
	if ( this->nInstances++ == 0 )  // first instance
	{

        this->CLCtx = new SingleDevClass(this->devType);

		clAmdBlasSetup();

		this->setup_ocl_kernels();
	}

	this->setupMLP(netProvider, dataProvider, _minibatch);
}


void MLPTrainerOCL::setupMLP(MLPNetProvider & netProvider, DNNDataProvider & dataProvider, int _minibatch)
{
 	if (  ( netProvider.getInputLayerSize() != dataProvider.getFeatureSize() ) ||
		  ( netProvider.getOutputLayerSize() != dataProvider.getLabelSize() )   ) {
		   mlp_log("MLPTrainer", "The setting provided from MLPDataProvider doesn't match those of the MLPNetProvider");
		   MLP_Exception("");
	};

	if (  (_minibatch != dataProvider.getBatchSize()) || dataProvider.getDataMode() != DNN_DATAMODE_SP_TRAIN) {
		   mlp_log("MLPTrainer", "The setting of the MLPDataProvider doesn't match the need of the MLPTrainer");
		   MLP_Exception("");
	};

	this->_initialize(netProvider, _minibatch);

	this->create_ocl_buffers(netProvider);

    this->dataProviderp = &dataProvider;

	this->initialized = true;
}


MLPTrainerOCL::~MLPTrainerOCL()
{
	if ( --this->nInstances == 0 )  {
        this->destroy_ocl_kernels();

		delete this->CLCtx;

		clAmdBlasTeardown();

	}

    this->release_ocl_buffers();
}

void MLPTrainerOCL::setup_ocl_kernels()
{
        cl_int status;

 		char *kernel_src;
	    if ( read_srcfile("kernels.cl", kernel_src) < 0 ) {
		     mlp_log("MLPTrainerOCL", "Failed to read kernel source file\n");
		     MLP_Exception("");
		     return;
	    };

		this->CLCtx->m_program = clCreateProgramWithSource(this->CLCtx->m_context, 1, (const char**)&kernel_src,NULL,&status);
		CL_CHECK( status );

		CL_CHECK( clBuildProgram(this->CLCtx->m_program,1,&this->CLCtx->m_device, NULL, NULL, NULL) );

		delete [] kernel_src;

		this->mykerns.activate_sigmoid_kernel = clCreateKernel(this->CLCtx->m_program,"activate_sigmoid",&status);
		CL_CHECK( status );
		this->mykerns.activate_softmax_kernel1 = clCreateKernel(this->CLCtx->m_program,"activate_softmax1",&status);
		CL_CHECK( status );
		this->mykerns.activate_softmax_kernel2 = clCreateKernel(this->CLCtx->m_program,"activate_softmax2",&status);
		CL_CHECK( status );
		this->mykerns.activate_tanh_kernel = clCreateKernel(this->CLCtx->m_program,"activate_tanh",&status);
		CL_CHECK( status );

		this->mykerns.derivative_sigmoid_kernel = clCreateKernel(this->CLCtx->m_program,"derivative_sigmoid",&status);
		CL_CHECK( status );
		this->mykerns.derivative_tanh_kernel = clCreateKernel(this->CLCtx->m_program,"derivative_tanh",&status);
		CL_CHECK( status );

		this->mykerns.calculateError_SSE_kernel1 = clCreateKernel(this->CLCtx->m_program,"calculateError_SSE1",&status);
		CL_CHECK( status );
		this->mykerns.calculateError_SSE_kernel2 = clCreateKernel(this->CLCtx->m_program,"calculateError_SSE2",&status);
		CL_CHECK( status );
		this->mykerns.calculateError_CE_kernel1 = clCreateKernel(this->CLCtx->m_program,"calculateError_CE1",&status);
		CL_CHECK( status );
		this->mykerns.calculateError_CE_kernel2 = clCreateKernel(this->CLCtx->m_program,"calculateError_CE2",&status);
		CL_CHECK( status );

		this->mykerns.calculateDelta_SSE_Sigmoid_kernel = clCreateKernel(this->CLCtx->m_program,"calculateDelta_SSE_Sigmoid",&status);
		CL_CHECK( status );
		this->mykerns.calculateDelta_CE_Softmax_kernel = clCreateKernel(this->CLCtx->m_program,"calculateDelta_CE_Softmax",&status);
		CL_CHECK( status );

        this->mykerns.transpose_sim_kernel = clCreateKernel(this->CLCtx->m_program,"transpose_simple",&status);
	    CL_CHECK( status );
        this->mykerns.transpose_kernel4 = clCreateKernel(this->CLCtx->m_program,"transpose_f4",&status);
	    CL_CHECK( status );
        this->mykerns.transpose_kernel32 = clCreateKernel(this->CLCtx->m_program,"transpose_32x32",&status);
	    CL_CHECK( status );

        this->mykerns.expandMatrix_kernel = clCreateKernel(this->CLCtx->m_program,"expandVectorToMatrix",&status);
	    CL_CHECK( status );

};

void MLPTrainerOCL::destroy_ocl_kernels()
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

		CL_CHECK( clReleaseProgram(this->CLCtx->m_program) );
};

void MLPTrainerOCL::create_ocl_buffers(MLPNetProvider &provider)
{
	cl_int status;

	this->inputs = new cl_mem[this->nLayers];         // buffers for storing the input/output for each layers
	this->weightT = new cl_mem[this->nLayers];        // weights for connecting the previous layer and current layer
	this->biases =  new cl_mem[this->nLayers];        // bias for each layer, added to the input of each layer
	this->delta = new cl_mem[this->nLayers];          // delta for each layer, used by back propagation

	// The Input/Output of layer i is stored in this->inputs[i+1], so this->inputs[1] is for the input layer, this->inputs[2] is for
	// the first hidden layer, this->inputs[0] is for the output layer
	for (int i = 1; i < this->nLayers; i++ )
	{
		this->inputs[i] = clCreateBuffer(this->CLCtx->m_context, CL_MEM_READ_WRITE, sizeof(cl_float)*this->dimensions[i-1]*this->minibatch,NULL,&status);
		CL_CHECK(status);

		this->weightT[i] = clCreateBuffer(this->CLCtx->m_context, CL_MEM_READ_WRITE, sizeof(cl_float)*this->dimensions[i-1]*this->dimensions[i], NULL,&status);
		CL_CHECK(status);

		cl_mem tmpBuff;

		tmpBuff = clCreateBuffer(this->CLCtx->m_context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, sizeof(cl_float)*this->dimensions[i-1]*this->dimensions[i],
			                               provider.weights[i],&status);
		CL_CHECK(status);
		transpose_float_matrix(tmpBuff, this->weightT[i], this->dimensions[i], this->dimensions[i-1]);  // make this->weightT[i] in transposed format
		clReleaseMemObject(tmpBuff);


		this->biases[i] = clCreateBuffer(this->CLCtx->m_context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, sizeof(cl_float)*this->dimensions[i],
			                               provider.biases[i],&status);
		CL_CHECK(status);

		this->delta[i] = clCreateBuffer(this->CLCtx->m_context, CL_MEM_READ_WRITE, sizeof(cl_float)*this->dimensions[i]*this->minibatch,NULL,&status);
		CL_CHECK(status);
	}

	// for output layer
	this->output = clCreateBuffer(this->CLCtx->m_context, CL_MEM_READ_WRITE, sizeof(cl_float)*(this->dimensions[this->nLayers-1])*this->minibatch,NULL,&status);
	CL_CHECK(status);

	this->inputs[0] = this->output;

	this->target = clCreateBuffer(this->CLCtx->m_context, CL_MEM_READ_WRITE, sizeof(cl_float)*this->dimensions[this->nLayers-1]*this->minibatch,NULL,&status);
	CL_CHECK(status);
}

void MLPTrainerOCL::release_ocl_buffers()
{
	for (int i = 1; i < this->nLayers; i++ )
	{
		CL_CHECK( clReleaseMemObject(this->inputs[i]) );
		CL_CHECK( clReleaseMemObject(this->weightT[i]) );
		CL_CHECK( clReleaseMemObject(this->biases[i]) );
		CL_CHECK( clReleaseMemObject(this->delta[i]) );
	}

	CL_CHECK( clReleaseMemObject(this->output) );
	CL_CHECK( clReleaseMemObject(this->target) );

	if ( this->inputs )
		delete [] this->inputs;
	if ( this->weightT )
		delete [] this->weightT;
	if ( this->delta )
		delete [] this->delta;
	if ( this->biases)
		delete [] this->biases;
};

void MLPTrainerOCL::synchronizeNetConfig(MLPNetProvider &netProvider)
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

	    tmpBuff = clCreateBuffer(this->CLCtx->m_context, CL_MEM_READ_WRITE, sizeof(cl_float)*this->dimensions[i-1]*this->dimensions[i], NULL,&status);
		this->transpose_float_matrix(this->weightT[i], tmpBuff, this->dimensions[i-1], this->dimensions[i]);    // make tmpBuff in original format
		status = clEnqueueReadBuffer(this->CLCtx->m_queues[0], tmpBuff, CL_TRUE, 0,sizeof(cl_float)*this->dimensions[i-1]*this->dimensions[i],
			                         netProvider.weights[i], 0, NULL, NULL);
		CL_CHECK(status);
        clReleaseMemObject(tmpBuff);

		status = clEnqueueReadBuffer(this->CLCtx->m_queues[0], this->biases[i], CL_TRUE, 0,sizeof(cl_float)*this->dimensions[i],
			                         netProvider.biases[i], 0, NULL, NULL);
		CL_CHECK(status);
	}
};


// the following interfaces make calls to OpenCL kernels

void MLPTrainerOCL::expandFloatVectorToMatrix(cl_mem  myVector, cl_mem myMatrix, int width, int height)
{
	cmn_expandFloatVectorToMatrix(this->CLCtx->m_queues[0],this->mykerns, myVector, myMatrix, width, height);
};

// use three different method to implement transposition depending on the size of the width and height
void MLPTrainerOCL::transpose_float_matrix(cl_mem src, cl_mem dst, cl_int width, cl_int height)
{
	if  ( (width % 32 == 0) && (height % 32 == 0) )
            cmn_transpose_matrix_32x32(this->CLCtx->m_queues[0],this->mykerns,src,dst,width,height);
	else
	    if (width % 4 == 0)
		    cmn_transpose_matrix_f4(this->CLCtx->m_queues[0],this->mykerns,src,dst,width,height);
		else
			cmn_transpose_matrix_simple(this->CLCtx->m_queues[0],this->mykerns,src,dst,width,height);
};


void MLPTrainerOCL::activate(int layer, cl_mem x, cl_mem y, int width, int height )
{
	switch (this->actFuncs[layer] ) {
	case AFUNC_SIGMOID:
		cmn_activate_sigmoid(this->CLCtx->m_queues[0],this->mykerns,x,y,width,height);
		return;
	case AFUNC_SOFTMAX:
	    cmn_activate_softmax(this->CLCtx->m_queues[0],this->mykerns,x,y,width,height);
		return;
	case AFUNC_TANH:
	    cmn_activate_tanh(this->CLCtx->m_queues[0],this->mykerns,x,y,width,height);
		return;
	case AFUNC_IDENTITY:
	    cmn_activate_identity(this->CLCtx->m_queues[0],this->mykerns,x,y,width,height);
        return;
	default:
		mlp_log("MLPTrainer", "The assigned activation function for this layer is not supported.");
		MLP_Exception("");
	};
};


void MLPTrainerOCL::calculateError(cl_mem output, cl_mem target, int width, int height, float &ret )
{
	switch (this->costFunc) {
	case CFUNC_SSE:
		cmn_calculateError_SSE(this->CLCtx->m_queues[0],this->mykerns,output,target,this->reduceMem,this->reduceBuff,width,height,ret);
		return;
	case CFUNC_CE:
		cmn_calculateError_CE(this->CLCtx->m_queues[0],this->mykerns,output,target,this->reduceMem,this->reduceBuff,width,height,ret);
		return;
	default:
		mlp_log("MLPTrainer", "The assigned cost function for this neural network is not supported.");
		MLP_Exception("");
	};
	return;
};


void MLPTrainerOCL::calculateDelta(cl_mem output, cl_mem target, cl_mem delta, int width, int height)
{
	if ( (this->costFunc == CFUNC_CE) && (this->actFuncs[this->nLayers-1] == AFUNC_SOFTMAX) ) {
		 cmn_calculateDelta_CE_Softmax(this->CLCtx->m_queues[0],this->mykerns,output,target,delta,width,height);
		 return;
	};
	if ( (this->costFunc == CFUNC_SSE) && (this->actFuncs[this->nLayers-1] == AFUNC_SIGMOID) ) {
		 cmn_calculateDelta_SSE_Sigmoid(this->CLCtx->m_queues[0],this->mykerns,output,target,delta,width,height);
		 return;
	};

	mlp_log("MLPTrainer", "The configuration for this neural network is not supported");
	MLP_Exception("");
	return;
};


void MLPTrainerOCL::derivative(int layer, cl_mem delta1, cl_mem y, cl_mem delta2, int width, int height )
{
	switch (this->actFuncs[layer] ) {
	case AFUNC_SIGMOID:
		cmn_derivative_sigmoid(this->CLCtx->m_queues[0],this->mykerns,delta1,y,delta2,width,height);
		return;
	case AFUNC_TANH:
		cmn_derivative_tanh(this->CLCtx->m_queues[0],this->mykerns,delta1,y,delta2,width,height);
		return;
	default:
		mlp_log("MLPTrainer", "The assigned activation function for this layer is not supported.");
		MLP_Exception("");
	};
};


int MLPTrainerOCL::batchTrainingWithCheckPointing(int maxBatches, int epoches, int startBatch, int startEpoch,  bool doChkPointing)
{
	if ( !this->initialized ) {
		 mlp_log("MLPTrainer", "This Trainer object should be setup with NetProvider and DataProvider first");
		 MLP_Exception("");
	};

	cl_int status;
	clAmdBlasStatus blasStatus;

	// the inputs for the MLP training
	float *l_features=NULL;
	float *l_labels=NULL;

	cl_mem OnesVector;     // in length of this->minibatch
	cl_mem *deltaT = new cl_mem[this->nLayers];
	cl_mem *biasesMatrix = new cl_mem[this->nLayers];
	cl_mem *varWeight1 = new cl_mem[this->nLayers];
	cl_mem *varWeight2 = new cl_mem[this->nLayers];
	cl_mem *lastVarWeight = varWeight1;
	cl_mem *curVarWeight = varWeight2;
	cl_mem *varBias1 = new cl_mem[this->nLayers];
	cl_mem *varBias2 = new cl_mem[this->nLayers];
	cl_mem *lastVarBias = varBias1;
	cl_mem *curVarBias = varBias2;


	// create deltaT buffer for each layer except for the input layer
	for (int i = 1; i < this->nLayers; i++) {
 	    deltaT[i] = clCreateBuffer(this->CLCtx->m_context,CL_MEM_READ_WRITE,sizeof(cl_float)*this->minibatch*this->dimensions[i],NULL,&status);
        CL_CHECK(status);
	};

	// create bias Matrix buffer for each layer except for the input layer
	for (int i = 1; i < this->nLayers; i++) {
 	    biasesMatrix[i] = clCreateBuffer(this->CLCtx->m_context,CL_MEM_READ_WRITE,sizeof(cl_float)*this->minibatch*this->dimensions[i],NULL,&status);
        CL_CHECK(status);

		this->expandFloatVectorToMatrix(this->biases[i],biasesMatrix[i],this->dimensions[i],this->minibatch);
	};

	// create buffer of a (1,1,...1) vector of length this->minibatch, it is used for updating the bias of each layer
	{
 		float *tmpHostBuff;

		tmpHostBuff = new float[this->minibatch];
		for (int k=0; k < this->minibatch; k++ )
			 tmpHostBuff[k] = 1.0f;

        OnesVector = clCreateBuffer(this->CLCtx->m_context,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,sizeof(cl_float)*this->minibatch,tmpHostBuff,&status);
        CL_CHECK(status);

	    delete [] tmpHostBuff;
	};


	// create last and current buffers for the variance of weights for each layer except for the input layer
	// initialize each <last buffer> for the variance of weights to all zeroes
	for (int i = 1; i < this->nLayers; i++) {
		float *tmpHostBuff;

		tmpHostBuff = new float[this->dimensions[i-1]*this->dimensions[i]];
		for (int k=0; k < this->dimensions[i-1]*this->dimensions[i]; k++ )
			 tmpHostBuff[k] = 0.0f;
 	    lastVarWeight[i] = clCreateBuffer(this->CLCtx->m_context,CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,sizeof(cl_float)*this->dimensions[i-1]*this->dimensions[i],
			                               tmpHostBuff,&status);
        CL_CHECK(status);

 	    curVarWeight[i] = clCreateBuffer(this->CLCtx->m_context,CL_MEM_READ_WRITE,sizeof(cl_float)*this->dimensions[i-1]*this->dimensions[i],NULL,&status);
        CL_CHECK(status);

		delete [] tmpHostBuff;

		tmpHostBuff = new float[this->dimensions[i]];
	    for (int k=0; k < this->dimensions[i]; k++ )
			 tmpHostBuff[k] = 0.0f;
		lastVarBias[i] = clCreateBuffer(this->CLCtx->m_context,CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,sizeof(cl_float)*this->dimensions[i], tmpHostBuff,&status);
        CL_CHECK(status);

	    curVarBias[i] = clCreateBuffer(this->CLCtx->m_context,CL_MEM_READ_WRITE,sizeof(cl_float)*this->dimensions[i],NULL,&status);
        CL_CHECK(status);

		delete [] tmpHostBuff;
	};

	// create reducing buffers on the host and device
	this->reduceMem = clCreateBuffer(this->CLCtx->m_context,CL_MEM_WRITE_ONLY,sizeof(cl_float)*this->minibatch,NULL,&status);
	CL_CHECK(status);
	this->reduceBuff = new float[this->minibatch];

    CL_CHECK(clFinish(this->CLCtx->m_queues[0]));

	//ofstream outfile;

	//outfile.open("output.txt", ios_base::out|ios_base::trunc);

	int myBatch;
	int myEpoch;

	this->currBatchNo = startBatch;
	this->currEpoch = startEpoch;

	myBatch = this->currBatchNo;
	myEpoch = this->currEpoch;

	while ( myEpoch < epoches ) {

	    while (  this->dataProviderp->batchAvailable() && (maxBatches == 0 || myBatch < maxBatches) ) {

			 MLP_CHECK(this->dataProviderp->getBatchData(this->minibatch,l_features,l_labels,true));  // blocking method

			 CL_CHECK(clEnqueueWriteBuffer(this->CLCtx->m_queues[0],this->inputs[1],CL_TRUE,0,sizeof(cl_float)*this->dimensions[0]*this->minibatch,l_features,0,NULL,NULL ));
			 CL_CHECK(clEnqueueWriteBuffer(this->CLCtx->m_queues[0],this->target,CL_TRUE,0,sizeof(cl_float)*this->dimensions[this->nLayers-1]*this->minibatch,l_labels,0,NULL,NULL ));

			 for (int i = 1; i < this->nLayers; i++) {

			 	 // Input[i] = Output[i-1] * Weight[i]     , here Weight[i] is in transposed form
				 blasStatus = clAmdBlasSgemm(clAmdBlasRowMajor,clAmdBlasNoTrans,clAmdBlasTrans,this->minibatch,this->dimensions[i],this->dimensions[i-1],1.0f,this->inputs[i],
					this->dimensions[i-1],this->weightT[i],this->dimensions[i-1],0.0f,this->inputs[(i+1)%this->nLayers],this->dimensions[i],1,&this->CLCtx->m_queues[0],0,NULL,NULL);
				 AMDBLAS_CHECK(blasStatus);

				 // Input[i] = Input[i] + 1.0 * Bias[i],   regarding the two Matrixes as  two vectors
				 blasStatus = clAmdBlasSaxpy(this->dimensions[i]*this->minibatch, 1.0f, biasesMatrix[i], 0, 1, this->inputs[(i+1)%this->nLayers], 0, 1, 1,
					                        &this->CLCtx->m_queues[0], 0, NULL, NULL);
				 AMDBLAS_CHECK(blasStatus);

				 // Output[i] = activate(Input[i])
				 this->activate(i, this->inputs[(i+1)%this->nLayers], this->inputs[(i+1)%this->nLayers], this->dimensions[i], this->minibatch);
			 }

			 float costval=0.0f;

			 //check_memory("Output", this->CLCtx->m_queues[0], this->output, this->dimensions[this->nLayers-1]*this->minibatch, check_zero);

 			 this->calculateError(this->output, this->target, this->dimensions[this->nLayers-1], this->minibatch, costval);

			 cout.precision(8);
			 cout << std::showpoint << std::fixed << endl;
			 cout << "Error Value for Batch  " << myBatch << " of Epoch " << myEpoch << ": " << costval << endl;

             CL_CHECK(clFinish(this->CLCtx->m_queues[0]));

		     this->calculateDelta(this->output, this->target, this->delta[this->nLayers-1], this->dimensions[this->nLayers-1], this->minibatch);

			 CL_CHECK(clFinish(this->CLCtx->m_queues[0]));

			 this->transpose_float_matrix(this->delta[this->nLayers-1], deltaT[this->nLayers-1], this->dimensions[this->nLayers-1], this->minibatch);

			 for ( int i = this->nLayers - 2; i > 0; i-- ) {
				 // Delta[i] = Delta[i+1] * WeightT[i+1],
				 blasStatus = clAmdBlasSgemm( clAmdBlasRowMajor, clAmdBlasNoTrans, clAmdBlasNoTrans, this->minibatch, this->dimensions[i], this->dimensions[i+1],1.0f, this->delta[i+1],
					this->dimensions[i+1], this->weightT[i+1], this->dimensions[i],  0.0f, this->delta[i], this->dimensions[i], 1, &this->CLCtx->m_queues[0],0,NULL,NULL );
				 AMDBLAS_CHECK(blasStatus);

				 // Delta[i] = derivative(Delta[i],Output[i])
				 this->derivative(i, this->delta[i],this->inputs[i+1], this->delta[i],this->dimensions[i], this->minibatch );

                 this->transpose_float_matrix(this->delta[i], deltaT[i], this->dimensions[i], this->minibatch);
			 }

	         CL_CHECK( clFinish(this->CLCtx->m_queues[0]) );

			 if ( doChkPointing)
			      DNN_LOCK(&this->chkPointingLock);

			 for ( int i = nLayers-1; i > 0; i-- ) {
				  float coef = this->etas[i];
				  float mm = this->momentum;

				  // curVarWeightT[i] = DeltaT[i] * Output[i-1] , here curVarWeight[i] is in transposed form
				  blasStatus = clAmdBlasSgemm( clAmdBlasRowMajor,clAmdBlasNoTrans,clAmdBlasNoTrans, this->dimensions[i], this->dimensions[i-1], this->minibatch, coef,
					  deltaT[i], this->minibatch, this->inputs[i],this->dimensions[i-1],0.0f,curVarWeight[i],this->dimensions[i-1],1,&this->CLCtx->m_queues[0],0,NULL,NULL);
				  AMDBLAS_CHECK(blasStatus);

				  // curVarWeightT[i] = curVarWeightT[i] + mm * lastVarWeightT[i], regarding the two Matrixes as two vectors
				  blasStatus = clAmdBlasSaxpy(this->dimensions[i]*this->dimensions[i-1],mm,lastVarWeight[i],0,1,curVarWeight[i],0,1,1,&this->CLCtx->m_queues[0],0,NULL,NULL);
                  AMDBLAS_CHECK(blasStatus);

				  // WeightT[i] = WeightT[i] + 1.0 * curVarWeightT[i],  regarding the two Matrixes as two vectors
				  blasStatus = clAmdBlasSaxpy(this->dimensions[i]*this->dimensions[i-1],1.0f,curVarWeight[i],0,1,this->weightT[i],0,1,1,&this->CLCtx->m_queues[0],0,NULL,NULL);
                  AMDBLAS_CHECK(blasStatus);

				  // curVarBias[i] = DeltaT[i] * (1,1, ... 1)T
                  blasStatus=clAmdBlasSgemv(clAmdBlasRowMajor, clAmdBlasNoTrans, this->dimensions[i], this->minibatch, coef, deltaT[i], this->minibatch, OnesVector,
					  0, 1, 0.0f, curVarBias[i], 0, 1, 1, &this->CLCtx->m_queues[0], 0, NULL, NULL);

                  // curVarBias[i] = curVarBias[i] + mm * lastVarBias[i]
				  blasStatus = clAmdBlasSaxpy(this->dimensions[i],mm,lastVarBias[i],0,1,curVarBias[i],0,1,1,&this->CLCtx->m_queues[0],0,NULL,NULL);
                  AMDBLAS_CHECK(blasStatus);


				  // Bias[i] = Bias[i] + 1.0 * curVarBias[i]
				  blasStatus = clAmdBlasSaxpy(this->dimensions[i],1.0f,curVarBias[i],0,1,this->biases[i],0,1,1,&this->CLCtx->m_queues[0],0,NULL,NULL);
                  AMDBLAS_CHECK(blasStatus);

				  this->expandFloatVectorToMatrix(this->biases[i], biasesMatrix[i], this->dimensions[i], this->minibatch);

			 };

			 if ( doChkPointing )
			     DNN_UNLOCK(&this->chkPointingLock);

			 // swap the curVarWeight and lastVarWeight pointers
			 cl_mem *tmpPointer;
			 tmpPointer = curVarWeight;
			 curVarWeight = lastVarWeight;
			 lastVarWeight = tmpPointer;

             // swap the curVarBias and lastVarBias pointers
		     tmpPointer = curVarBias;
			 curVarBias = lastVarBias;
			 lastVarBias = tmpPointer;


             // tell the data provider that I have done with current batch of data, want next batch of data
			 MLP_CHECK(this->dataProviderp->nextBatch());

			 myBatch++;

			 if ( doChkPointing ) {
                  DNN_LOCK(&this->chkPointingLock);
			      this->currBatchNo = myBatch;
				  this->currEpoch = myEpoch;
                  DNN_UNLOCK(&this->chkPointingLock);
			 };
	    } // end of all baches

		myEpoch++;
		myBatch = 0;
		this->dataProviderp->resetDataProvider();

		if ( doChkPointing ) {
             DNN_LOCK(&this->chkPointingLock);
			 this->currBatchNo = myBatch;
		     this->currEpoch = myEpoch;
             DNN_UNLOCK(&this->chkPointingLock);
		};
	};  // end of all epoches

	for (int i = 1; i < this->nLayers; i++) {
		CL_CHECK( clReleaseMemObject(deltaT[i]) );
	    CL_CHECK( clReleaseMemObject(biasesMatrix[i]) );
		CL_CHECK( clReleaseMemObject(varWeight1[i]) );
	    CL_CHECK( clReleaseMemObject(varWeight2[i]) );
		CL_CHECK( clReleaseMemObject(varBias1[i]) );
	    CL_CHECK( clReleaseMemObject(varBias2[i]) );
	};

	CL_CHECK( clReleaseMemObject(OnesVector) );
	CL_CHECK( clReleaseMemObject(this->reduceMem) );
	delete [] this->reduceBuff;

	delete [] deltaT;
	delete [] biasesMatrix;
	delete [] varWeight1;
	delete [] varWeight2;
	delete [] varBias1;
	delete [] varBias2;

	if ( maxBatches == 0 )
		 return(myEpoch * this->dataProviderp->getTotalBatches() + myBatch);
	else {
	     int realBatches;

	     realBatches = std::min<int>(this->dataProviderp->getTotalBatches(), maxBatches);
		 return(myEpoch * realBatches + myBatch);
	};
}



