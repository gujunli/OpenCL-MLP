/*
 *  COPYRIGHT:  Copyright (c) 2013 Advanced Micro Devices, Inc.  All rights reserved
 *
 *  Changed by  Qianfeng Zhang@amd.com ( March 2014 )
 *
 *  Written by  Junli Gu@amd.com ( Dec 2013 )
 */


#include <algorithm>
#include <clAmdBlas.h>

#include "MLPUtil.h"
#include "MLPOclCommon.h"
#include "MLPPredictorOCL.h"


//Class specific member shared by all instances
SingleDevClass *MLPPredictorOCL::CLCtx = NULL;
int MLPPredictorOCL::nInstances = 0;

MLP_Kerns MLPPredictorOCL::mykerns;

MLPPredictorOCL::MLPPredictorOCL()
{
	// class wide set up
	if ( this->nInstances++ == 0 )  {   // at the first instance
        this->CLCtx = new SingleDevClass(this->devType);

		clAmdBlasSetup();

		this->setup_ocl_kernels();
	}

	this->devType = DNN_OCL_DI_GPU;    // default OpenCL device

	this->inputs = NULL;
	this->weights = NULL;
	this->biases = NULL;
	this->biasMatrixes = NULL;

	this->initialized = false;
};


MLPPredictorOCL::MLPPredictorOCL(MLPNetProvider & netProvider, DNN_OCL_DEVTYPE dType, int _batchSize)
{
  	this->devType = dType;

	// class wide set up
	if ( this->nInstances++ == 0 )  {   // at the first instance
        this->CLCtx = new SingleDevClass(this->devType);

		clAmdBlasSetup();

		this->setup_ocl_kernels();
	}

    this->setupMLP(netProvider, _batchSize);
}

void MLPPredictorOCL::setupMLP(MLPNetProvider & netProvider, int _batchSize)
{

	this->_initialize(netProvider, _batchSize);

	this->create_ocl_buffers(netProvider);

	this->initialized = true;
}


MLPPredictorOCL::~MLPPredictorOCL()
{
	if ( --this->nInstances == 0 ) {    // at the last instance
        this->destroy_ocl_kernels();

		delete this->CLCtx;

		clAmdBlasTeardown();
	}

    this->release_ocl_buffers();
}

void MLPPredictorOCL::setup_ocl_kernels()
{
 	    cl_int status;
		char *kernel_src;

	    if ( read_srcfile("kernels.cl", kernel_src) < 0 ) {
		     mlp_log("MLPPredictor", "Failed to read kernel source file\n");
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

        this->mykerns.expandMatrix_kernel = clCreateKernel(this->CLCtx->m_program,"expandVectorToMatrix",&status);
	    CL_CHECK( status );

};

void MLPPredictorOCL::destroy_ocl_kernels()
{
		CL_CHECK( clReleaseKernel(this->mykerns.activate_sigmoid_kernel) );
	    CL_CHECK( clReleaseKernel(this->mykerns.activate_softmax_kernel1) );
        CL_CHECK( clReleaseKernel(this->mykerns.activate_softmax_kernel2) );
	    CL_CHECK( clReleaseKernel(this->mykerns.activate_tanh_kernel) );

		CL_CHECK( clReleaseKernel(this->mykerns.expandMatrix_kernel) );

		CL_CHECK( clReleaseProgram(this->CLCtx->m_program) );
};

void MLPPredictorOCL::create_ocl_buffers(MLPNetProvider &provider)
{
    cl_int status;

 	this->inputs = new cl_mem[this->nLayers];         // buffers for storing the input/output for each layers
	this->weights = new cl_mem[this->nLayers];        // weights for connecting the previous layer and current layer
	this->biases =  new cl_mem[this->nLayers];        // bias for each layer, added to the input of each layer

	// The Input/Output of layer i is stored in this->inputs[i+1], so this->inputs[1] is for the input layer, this->inputs[2] is for
	// the first hidden layer, this->inputs[0] is for the output layer
	for (int i = 1; i < this->nLayers; i++ )
	{
		this->inputs[i] = clCreateBuffer(this->CLCtx->m_context, CL_MEM_READ_WRITE, sizeof(cl_float)*this->dimensions[i-1]*this->batchSize,NULL,&status);
		CL_CHECK(status);

		this->weights[i] = clCreateBuffer(this->CLCtx->m_context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, sizeof(cl_float)*this->dimensions[i-1]*this->dimensions[i],
			                               provider.weights[i],&status);
		CL_CHECK(status);

		this->biases[i] = clCreateBuffer(this->CLCtx->m_context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, sizeof(cl_float)*this->dimensions[i],
			                               provider.biases[i],&status);
		CL_CHECK(status);
	}

	// for output layer
	this->output = clCreateBuffer(this->CLCtx->m_context, CL_MEM_READ_WRITE, sizeof(cl_float)*(this->dimensions[this->nLayers-1])*this->batchSize,NULL,&status);
	CL_CHECK(status);

	this->inputs[0] = this->output;

	this->biasMatrixes = new cl_mem[this->nLayers];

	// create bias Matrix buffer for each layer except for the input layer
	for (int i = 1; i < this->nLayers; i++) {
 	    this->biasMatrixes[i] = clCreateBuffer(this->CLCtx->m_context,CL_MEM_READ_WRITE,sizeof(cl_float)*this->batchSize*this->dimensions[i],NULL,&status);
        CL_CHECK(status);

		this->expandFloatVectorToMatrix(this->biases[i],this->biasMatrixes[i],this->dimensions[i],this->batchSize);
	};
};

void MLPPredictorOCL::release_ocl_buffers()
{
	for (int i = 1; i < this->nLayers; i++ ) {
		CL_CHECK( clReleaseMemObject(this->inputs[i]) );
		CL_CHECK( clReleaseMemObject(this->weights[i]) );
		CL_CHECK( clReleaseMemObject(this->biases[i]) );

		CL_CHECK( clReleaseMemObject(this->biasMatrixes[i]) );
	}

	CL_CHECK( clReleaseMemObject(this->output) );

	if ( this->inputs )
		delete [] this->inputs;
	if ( this->weights )
		delete [] this->weights;
	if ( this->biases )
		delete [] this->biases;
	if ( this->biasMatrixes )
		delete [] this->biasMatrixes;
};

// the following interfaces make calls to OpenCL kernels

void MLPPredictorOCL::expandFloatVectorToMatrix(cl_mem  myVector, cl_mem myMatrix, int width, int height)
{
	cmn_expandFloatVectorToMatrix(this->CLCtx->m_queues[0],this->mykerns, myVector, myMatrix, width, height);
};

void MLPPredictorOCL::activate(int layer, cl_mem x, cl_mem y, int width, int height )
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
		mlp_log("MLPPredictor", "The assigned activation function for this layer is not supported.");
		MLP_Exception("");
	};
};

void MLPPredictorOCL::batchPredicting(float *inVectors, float *outVectors)
{
	if ( !this->initialized) {
		 mlp_log("MLPPredictor", "This Predictor object should be setup with NetProvider and DataProvider first");
		 MLP_Exception("");
	};

	clAmdBlasStatus blasStatus;

	CL_CHECK(clEnqueueWriteBuffer(this->CLCtx->m_queues[0],this->inputs[1],CL_TRUE,0,sizeof(cl_float)*this->dimensions[0]*this->batchSize,inVectors,0,NULL,NULL));

	for (int i = 1; i < nLayers; i++) {
		 // Input[i] = Output[i-1] * Weight[i]
		 blasStatus = clAmdBlasSgemm(clAmdBlasRowMajor,clAmdBlasNoTrans,clAmdBlasNoTrans,this->batchSize,this->dimensions[i],this->dimensions[i-1],1.0f,this->inputs[i],
				this->dimensions[i-1],this->weights[i],this->dimensions[i],0.0f,this->inputs[(i+1)%this->nLayers],this->dimensions[i],1,&this->CLCtx->m_queues[0],0,NULL,NULL);
		 AMDBLAS_CHECK(blasStatus);

		 // Input[i] = Input[i] + 1.0 * Bias[i],   regarding the two Matrixes as  two vectors
		 blasStatus = clAmdBlasSaxpy(this->dimensions[i]*this->batchSize, 1.0f, this->biasMatrixes[i], 0, 1, this->inputs[(i+1)%this->nLayers], 0, 1, 1,
			                        &this->CLCtx->m_queues[0], 0, NULL, NULL);
		 AMDBLAS_CHECK(blasStatus);

		 // Output[i] = activate(Input[i])
		this->activate(i, this->inputs[(i+1)%this->nLayers], this->inputs[(i+1)%this->nLayers], this->dimensions[i], this->batchSize);
	}

	// read the output vectors from the device to the host layer so that they can be checked
	CL_CHECK(clEnqueueReadBuffer(this->CLCtx->m_queues[0],this->output,CL_TRUE,0,sizeof(cl_float)*this->dimensions[this->nLayers-1]*this->batchSize,outVectors,0,NULL,NULL));
};

void MLPPredictorOCL::singlePredicting(float *inVector, float *outVector)
{
	if ( !this->initialized) {
		 mlp_log("MLPPredictor", "This Predictor object should be setup with NetProvider and DataProvider first");
		 MLP_Exception("");
	};

	clAmdBlasStatus blasStatus;

	CL_CHECK(clEnqueueWriteBuffer(this->CLCtx->m_queues[0],this->inputs[1],CL_TRUE,0,sizeof(cl_float)*this->dimensions[0],inVector,0,NULL,NULL));

	for (int i = 1; i < nLayers; i++) {
		 // Input[i] = Output[i-1] * Weight[i], calculated using WeightT[i]*Output[i] to call the library interface
		 blasStatus=clAmdBlasSgemv(clAmdBlasRowMajor, clAmdBlasTrans, this->dimensions[i-1], this->dimensions[i], 1.0f, this->weights[i], this->dimensions[i], this->inputs[i],
					0, 1, 0.0f, this->inputs[(i+1)%this->nLayers], 0, 1, 1, &this->CLCtx->m_queues[0], 0, NULL, NULL);

		 // Input[i] = Input[i] + 1.0 * Bias[i]
		 blasStatus = clAmdBlasSaxpy(this->dimensions[i], 1.0f, this->biases[i], 0, 1, this->inputs[(i+1)%this->nLayers], 0, 1, 1,
			                        &this->CLCtx->m_queues[0], 0, NULL, NULL);


		 AMDBLAS_CHECK(blasStatus);

		 // Output[i] = activate(Input[i])
		this->activate(i, this->inputs[(i+1)%this->nLayers], this->inputs[(i+1)%this->nLayers], this->dimensions[i], 1);
	}

	// read the output vectors from the device to the host layer so that they can be checked
	CL_CHECK(clEnqueueReadBuffer(this->CLCtx->m_queues[0],this->output,CL_TRUE,0,sizeof(cl_float)*this->dimensions[this->nLayers-1],outVector,0,NULL,NULL));
};




