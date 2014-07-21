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
#include "MLPCommon.h"
#include "MLPTester.h"


//Class specific member shared by all instances
SingleDevClass *MLPTester::CLContext = NULL;
int MLPTester::nInstances = 0;

MLP_Kerns MLPTester::mykerns;

MLPTester::MLPTester()
{
	this->devType = MLP_OCL_DI_GPU;
	this->setDefault();
	this->dataProviderp = NULL;
	this->initialized = false;
}

MLPTester::MLPTester(MLPNetProvider & netProvider, MLPDataProvider & dataProvider, MLP_OCL_DEVTYPE dType, int _batchSize)
{
	if (  (_batchSize != dataProvider.m_batchSize) || dataProvider.dataMode != MLP_DATAMODE_TEST ) {
		   mlp_log("MLPTester", "The setting of the MLPDataProvider doesn't match the need of the MLPTester");
		   MLP_Exception("");
	};

	this->devType = dType;
	this->setDefault();
	this->_initialize(netProvider, _batchSize);
	this->dataProviderp = &dataProvider;
	this->initialized = true;
}

MLPTester::~MLPTester()
{
    this->_dispose();

	if ( --this->nInstances == 0 ) {
		CL_CHECK( clReleaseKernel(this->mykerns.activate_sigmoid_kernel) );
	    CL_CHECK( clReleaseKernel(this->mykerns.activate_softmax_kernel1) );
        CL_CHECK( clReleaseKernel(this->mykerns.activate_softmax_kernel2) );
	    CL_CHECK( clReleaseKernel(this->mykerns.activate_tanh_kernel) );

		CL_CHECK( clReleaseKernel(this->mykerns.expandMatrix_kernel) );

		CL_CHECK( clReleaseProgram(this->CLContext->m_program) );

		delete this->CLContext;

		clAmdBlasTeardown();
	}
}

// only called by the constructor
void MLPTester::setDefault()
{
	cl_int status;

	// class wide setting up
	if ( this->nInstances++ == 0 )  // first instance
	{
 	    this->CLContext = new SingleDevClass(this->devType);

		clAmdBlasSetup();

		char *kernel_src;
	    if ( read_srcfile("kernels.cl", kernel_src) < 0 ) {
		     mlp_log("MLPTester", "Failed to read kernel source file\n");
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

        this->mykerns.expandMatrix_kernel = clCreateKernel(this->CLContext->m_program,"expandVectorToMatrix",&status);
	    CL_CHECK( status );
	}

	// setting up needed by the instance only
	this->dimensions = NULL;
	this->nLayers = 0;
	this->batchSize = 0;

	this->inputs = NULL;
	this->weights = NULL;
	this->output = NULL;
	this->target = NULL;
	this->actFuncs = NULL;
}


// only called by the constructor and setupMLP()
void MLPTester::_initialize(MLPNetProvider & provider, int _batchSize)
{
	cl_int status;

    this->netType = provider.netType;
	this->nLayers = provider.nLayers;
	this->batchSize = _batchSize;
	this->dimensions = new int[this->nLayers];
	this->inputs = new cl_mem[this->nLayers];         // buffers for storing the input/output for each layers
	this->weights = new cl_mem[this->nLayers];        // weights for connecting the previous layer and current layer
	this->biases =  new cl_mem[this->nLayers];        // bias for each layer, added to the input of each layer
	this->actFuncs = new ACT_FUNC[this->nLayers];     // activating function for each layer

	for ( int i = 0; i < this->nLayers; i++ )
		this->dimensions[i] = provider.dimensions[i];

	for ( int i = 0; i < this->nLayers; i++ )
		this->actFuncs[i] = provider.actFuncs[i];

	// The Input/Output of layer i is stored in this->inputs[i+1], so this->inputs[1] is for the input layer, this->inputs[2] is for
	// the first hidden layer, this->inputs[0] is for the output layer
	for (int i = 1; i < this->nLayers; i++ )
	{
		this->inputs[i] = clCreateBuffer(this->CLContext->m_context, CL_MEM_READ_WRITE, sizeof(cl_float)*this->dimensions[i-1]*this->batchSize,NULL,&status);
		CL_CHECK(status);

		this->weights[i] = clCreateBuffer(this->CLContext->m_context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, sizeof(cl_float)*this->dimensions[i-1]*this->dimensions[i],
			                               provider.weights[i],&status);
		CL_CHECK(status);

		this->biases[i] = clCreateBuffer(this->CLContext->m_context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, sizeof(cl_float)*this->dimensions[i],
			                               provider.biases[i],&status);
		CL_CHECK(status);
	}

	// for output layer
	this->output = clCreateBuffer(this->CLContext->m_context, CL_MEM_READ_WRITE, sizeof(cl_float)*(this->dimensions[this->nLayers-1])*this->batchSize,NULL,&status);
	CL_CHECK(status);

	this->inputs[0] = this->output;

	this->biasMatrixes = new cl_mem[this->nLayers];

	// create bias Matrix buffer for each layer except for the input layer
	for (int i = 1; i < this->nLayers; i++) {
 	    this->biasMatrixes[i] = clCreateBuffer(this->CLContext->m_context,CL_MEM_READ_WRITE,sizeof(cl_float)*this->batchSize*this->dimensions[i],NULL,&status);
        CL_CHECK(status);

		this->expandFloatVectorToMatrix(this->biases[i],this->biasMatrixes[i],this->dimensions[i],this->batchSize);
	};


	this->target = clCreateBuffer(this->CLContext->m_context, CL_MEM_READ_WRITE, sizeof(cl_float)*this->dimensions[this->nLayers-1]*this->batchSize,NULL,&status);
	CL_CHECK(status);
}


// only called by the destructor
void MLPTester::_dispose()
{
	if ( this->nLayers == 0 )    // nothing to be released
		return;

	for (int i = 1; i < this->nLayers; i++ )
	{
		CL_CHECK( clReleaseMemObject(this->inputs[i]) );
		CL_CHECK( clReleaseMemObject(this->weights[i]) );
		CL_CHECK( clReleaseMemObject(this->biases[i]) );

		CL_CHECK( clReleaseMemObject(this->biasMatrixes[i]) );
	}

	CL_CHECK( clReleaseMemObject(this->output) );
	CL_CHECK( clReleaseMemObject(this->target) );

	if ( this->dimensions )
		delete [] this->dimensions;
	if ( this->actFuncs )
		delete [] this->actFuncs;
	if ( this->inputs )
		delete [] this->inputs;
	if ( this->weights )
		delete [] this->weights;
	if ( this->biases)
		delete [] this->biases;
	if ( this->biasMatrixes )
		delete [] this->biasMatrixes;
}

void MLPTester::setupMLP(MLPNetProvider & netProvider, MLPDataProvider & dataProvider, int batchSize)
{
	if (  (batchSize != dataProvider.m_batchSize) || dataProvider.dataMode != MLP_DATAMODE_TEST ) {
		   mlp_log("MLPTester", "The setting of the MLPDataProvider doesn't match the need of the MLPTester");
		   MLP_Exception("");
	};

	this->_initialize(netProvider, batchSize);
	this->dataProviderp = &dataProvider;
	this->initialized = true;
}

MLPDataProvider *MLPTester::getDataProvider()
{
	return(this->dataProviderp);
};


// the following interfaces make calls to OpenCL kernels

void MLPTester::expandFloatVectorToMatrix(cl_mem  myVector, cl_mem myMatrix, int width, int height)
{
	cmn_expandFloatVectorToMatrix(this->CLContext->m_cmd_queues[0],this->mykerns, myVector, myMatrix, width, height);
};


void MLPTester::activate(int layer, cl_mem x, cl_mem y, int width, int height )
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
		mlp_log("MLPTester", "The assigned activation function for this layer is not supported.");
		MLP_Exception("");
	};
};


void MLPTester::batchTesting(int maxBatches)
{
	if ( !this->initialized) {
		 mlp_log("MLPTester", "This Tester object should be setup with NetProvider and DataProvider first");
		 MLP_Exception("");
	};

	clAmdBlasStatus blasStatus;

	// the inputs for the MLP training
	float *features=NULL;        // buffer for minibatch number of input vectors
	float *labels=NULL;          // buffer for minibatch number of labels
	float *outputs=NULL;         // buffer for minibatch number of output vectors
	int veclen;                  // length of the output vector

	veclen = this->dimensions[this->nLayers-1];
	outputs = new float[veclen * this->batchSize];

	this->succTestFrames = 0;
	this->totalTestFrames = 0;

	int batches=0;
	while ( this->dataProviderp->batchAvailable() && ( maxBatches == 0 || batches < maxBatches ) ) {

			MLP_CHECK(this->dataProviderp->getBatchData(this->batchSize,features,labels,true));

			CL_CHECK(clEnqueueWriteBuffer(this->CLContext->m_cmd_queues[0],this->inputs[1],CL_TRUE,0,sizeof(cl_float)*this->dimensions[0]*this->batchSize,features,0,NULL,NULL));

			for (int i = 1; i < nLayers; i++) {
				// Input[i] = Output[i-1] * Weight[i]
				blasStatus = clAmdBlasSgemm(clAmdBlasRowMajor,clAmdBlasNoTrans,clAmdBlasNoTrans,this->batchSize,this->dimensions[i],this->dimensions[i-1],1.0f,this->inputs[i],
					this->dimensions[i-1],this->weights[i],this->dimensions[i],0.0f,this->inputs[(i+1)%this->nLayers],this->dimensions[i],1,&this->CLContext->m_cmd_queues[0],0,NULL,NULL);
				AMDBLAS_CHECK(blasStatus);

				// Input[i] = Input[i] + 1.0 * Bias[i],   regarding the two Matrixes as  two vectors
				blasStatus = clAmdBlasSaxpy(this->dimensions[i]*this->batchSize, 1.0f, this->biasMatrixes[i], 0, 1, this->inputs[(i+1)%this->nLayers], 0, 1, 1,
					                        &this->CLContext->m_cmd_queues[0], 0, NULL, NULL);
				AMDBLAS_CHECK(blasStatus);

				// Output[i] = activate(Input[i])
				this->activate(i, this->inputs[(i+1)%this->nLayers], this->inputs[(i+1)%this->nLayers], this->dimensions[i], this->batchSize);
			}

			// read the output vectors from the device to the host layer so that they can be checked
			CL_CHECK(clEnqueueReadBuffer(this->CLContext->m_cmd_queues[0],this->output,CL_TRUE,0,sizeof(cl_float)*this->dimensions[this->nLayers-1]*this->batchSize,outputs,0,NULL,NULL));

			this->totalTestFrames += this->batchSize;
			int succCount=0;
			for (int i=0; i< this->batchSize; i++) {
				if ( this->dataProviderp->frameMatching(&outputs[i*veclen], &labels[i*veclen], veclen) )  // output vector matches the label vector
					 succCount++;
			};

			this->succTestFrames += succCount;

			// tell the data provider that I have done with current batch of data, want next batch of data
			MLP_CHECK(this->dataProviderp->nextBatch());

			batches++;
	}


	delete [] outputs;
}


void MLPTester::getTestingStats(int &totalFrames, int &succFrames)
{
	totalFrames = this->totalTestFrames;
	succFrames = this->succTestFrames;
};

int MLPTester::getInputVectorSize()
{
	return(this->dimensions[0]);
};

int MLPTester::getOutputVectorSize()
{
	return(this->dimensions[this->nLayers-1]);
};

int MLPTester::getBatchSize()
{
	return(this->batchSize);
};

bool MLPTester::singleTesting(float *inVector, float *labelVector, VECTOR_MATCH matchFunc)
{
	if ( !this->initialized) {
		 mlp_log("MLPPredictor", "This Predictor object should be setup with NetProvider and DataProvider first");
		 MLP_Exception("");
	};

    cl_int status;
	clAmdBlasStatus blasStatus;
    int outputSize;
	float *outVector;

    outputSize = this->getOutputVectorSize();
    outVector = new float[outputSize];

	status = clEnqueueWriteBuffer(this->CLContext->m_cmd_queues[0],this->inputs[1],CL_TRUE,0,sizeof(cl_float)*this->dimensions[0],inVector,0,NULL,NULL);
    CL_CHECK(status);

	for (int i = 1; i < nLayers; i++) {
         // Input[i] = Output[i-1] * Weight[i], matrix * matrix is also OK, but less efficient
		 //blasStatus = clAmdBlasSgemm(clAmdBlasRowMajor,clAmdBlasNoTrans,clAmdBlasNoTrans,1,this->dimensions[i],this->dimensions[i-1],1.0f,this->inputs[i],
		 //  this->dimensions[i-1],this->weights[i],this->dimensions[i],0.0f,this->inputs[(i+1)%this->nLayers],this->dimensions[i],1,&this->CLContext->m_cmd_queues[0],0,NULL,NULL);

		 // Input[i] = Output[i-1] * Weight[i], calculated using WeightT[i]*Output[i] to call the library interface
		 blasStatus=clAmdBlasSgemv(clAmdBlasRowMajor, clAmdBlasTrans, this->dimensions[i-1], this->dimensions[i], 1.0f, this->weights[i], this->dimensions[i], this->inputs[i],
		 			0, 1, 0.0f, this->inputs[(i+1)%this->nLayers], 0, 1, 1, &this->CLContext->m_cmd_queues[0], 0, NULL, NULL);

		 AMDBLAS_CHECK(blasStatus);

		 // Input[i] = Input[i] + 1.0 * Bias[i]
		 blasStatus = clAmdBlasSaxpy(this->dimensions[i], 1.0f, this->biases[i], 0, 1, this->inputs[(i+1)%this->nLayers], 0, 1, 1,
			                        &this->CLContext->m_cmd_queues[0], 0, NULL, NULL);


		 AMDBLAS_CHECK(blasStatus);

		 // Output[i] = activate(Input[i])
		 this->activate(i, this->inputs[(i+1)%this->nLayers], this->inputs[(i+1)%this->nLayers], this->dimensions[i], 1);
	}

	// read the output vectors from the device to the host layer so that they can be checked
	status = clEnqueueReadBuffer(this->CLContext->m_cmd_queues[0],this->output,CL_TRUE,0,sizeof(cl_float)*this->dimensions[this->nLayers-1],outVector,0,NULL,NULL);
    CL_CHECK(status);

    bool result;

    result = (*matchFunc)(outVector, labelVector, outputSize);

    delete [] outVector;

    return(result);
}
