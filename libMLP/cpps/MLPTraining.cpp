/*
 *  COPYRIGHT:  Copyright (c) 2013 Advanced Micro Devices, Inc.  All rights reserved
 *
 *   Changed by Qianfeng Zhang@amd.com ( March 2014 )
 *
 *   Author:  Junli Gu@amd.com
 */

#include <algorithm>
#include <iostream>
#include <fstream>

#include <clAmdBlas.h>

#include "MLPUtil.h"
#include "oclUtil.h"
#include "MLPTrainer.h"

int MLPTrainer::batchTraining(int maxBatches, int epoches)
{
	return(this->batchTrainingWithCheckPointing(maxBatches, epoches, 0, 0, NULL));
};


int MLPTrainer::batchTrainingWithCheckPointing(int maxBatches, int epoches, int startBatch, int startEpoch,  bool doChkPointing)
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
 	    deltaT[i] = clCreateBuffer(this->CLContext->m_context,CL_MEM_READ_WRITE,sizeof(cl_float)*this->minibatch*this->dimensions[i],NULL,&status);
        CL_CHECK(status);
	};

	// create bias Matrix buffer for each layer except for the input layer
	for (int i = 1; i < this->nLayers; i++) {
 	    biasesMatrix[i] = clCreateBuffer(this->CLContext->m_context,CL_MEM_READ_WRITE,sizeof(cl_float)*this->minibatch*this->dimensions[i],NULL,&status);
        CL_CHECK(status);

		this->expandFloatVectorToMatrix(this->biases[i],biasesMatrix[i],this->dimensions[i],this->minibatch);
	};

	// create buffer of a (1,1,...1) vector of length this->minibatch, it is used for updating the bias of each layer
	{
 		float *tmpHostBuff;

		tmpHostBuff = new float[this->minibatch];
		for (int k=0; k < this->minibatch; k++ )
			 tmpHostBuff[k] = 1.0f;

        OnesVector = clCreateBuffer(this->CLContext->m_context,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,sizeof(cl_float)*this->minibatch,tmpHostBuff,&status);
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
 	    lastVarWeight[i] = clCreateBuffer(this->CLContext->m_context,CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,sizeof(cl_float)*this->dimensions[i-1]*this->dimensions[i],
			                               tmpHostBuff,&status);
        CL_CHECK(status);

 	    curVarWeight[i] = clCreateBuffer(this->CLContext->m_context,CL_MEM_READ_WRITE,sizeof(cl_float)*this->dimensions[i-1]*this->dimensions[i],NULL,&status);
        CL_CHECK(status);

		delete [] tmpHostBuff;

		tmpHostBuff = new float[this->dimensions[i]];
	    for (int k=0; k < this->dimensions[i]; k++ )
			 tmpHostBuff[k] = 0.0f;
		lastVarBias[i] = clCreateBuffer(this->CLContext->m_context,CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,sizeof(cl_float)*this->dimensions[i], tmpHostBuff,&status);
        CL_CHECK(status);

	    curVarBias[i] = clCreateBuffer(this->CLContext->m_context,CL_MEM_READ_WRITE,sizeof(cl_float)*this->dimensions[i],NULL,&status);
        CL_CHECK(status);

		delete [] tmpHostBuff;
	};

	// create reducing buffers on the host and device
	this->reduceMem = clCreateBuffer(this->CLContext->m_context,CL_MEM_WRITE_ONLY,sizeof(cl_float)*this->minibatch,NULL,&status);
	CL_CHECK(status);
	this->reduceBuff = new float[this->minibatch];

    CL_CHECK(clFinish(this->CLContext->m_cmd_queues[0]));

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

			 CL_CHECK(clEnqueueWriteBuffer(this->CLContext->m_cmd_queues[0],this->inputs[1],CL_TRUE,0,sizeof(cl_float)*this->dimensions[0]*this->minibatch,l_features,0,NULL,NULL ));
			 CL_CHECK(clEnqueueWriteBuffer(this->CLContext->m_cmd_queues[0],this->target,CL_TRUE,0,sizeof(cl_float)*this->dimensions[this->nLayers-1]*this->minibatch,l_labels,0,NULL,NULL ));

			 for (int i = 1; i < this->nLayers; i++) {

			 	 // Input[i] = Output[i-1] * Weight[i]     , here Weight[i] is in transposed form
				 blasStatus = clAmdBlasSgemm(clAmdBlasRowMajor,clAmdBlasNoTrans,clAmdBlasTrans,this->minibatch,this->dimensions[i],this->dimensions[i-1],1.0f,this->inputs[i],
					this->dimensions[i-1],this->weightT[i],this->dimensions[i-1],0.0f,this->inputs[(i+1)%this->nLayers],this->dimensions[i],1,&this->CLContext->m_cmd_queues[0],0,NULL,NULL);
				 AMDBLAS_CHECK(blasStatus);

				 // Input[i] = Input[i] + 1.0 * Bias[i],   regarding the two Matrixes as  two vectors
				 blasStatus = clAmdBlasSaxpy(this->dimensions[i]*this->minibatch, 1.0f, biasesMatrix[i], 0, 1, this->inputs[(i+1)%this->nLayers], 0, 1, 1,
					                        &this->CLContext->m_cmd_queues[0], 0, NULL, NULL);
				 AMDBLAS_CHECK(blasStatus);

				 // Output[i] = activate(Input[i])
				 this->activate(i, this->inputs[(i+1)%this->nLayers], this->inputs[(i+1)%this->nLayers], this->dimensions[i], this->minibatch);
			 }

			 float costval=0.0f;

			 //check_memory("Output", this->CLContext->m_cmd_queues[0], this->output, this->dimensions[this->nLayers-1]*this->minibatch, check_zero);

 			 this->calculateError(this->output, this->target, this->dimensions[this->nLayers-1], this->minibatch, costval);

			 cout.precision(8); 
			 cout << std::showpoint << std::fixed << endl;
			 cout << "Error Value for Batch  " << myBatch << " of Epoch " << myEpoch << ": " << costval << endl;

             CL_CHECK(clFinish(this->CLContext->m_cmd_queues[0]));

		     this->calculateDelta(this->output, this->target, this->delta[this->nLayers-1], this->dimensions[this->nLayers-1], this->minibatch);

			 CL_CHECK(clFinish(this->CLContext->m_cmd_queues[0]));

			 this->transpose_float_matrix(this->delta[this->nLayers-1], deltaT[this->nLayers-1], this->dimensions[this->nLayers-1], this->minibatch);

			 for ( int i = this->nLayers - 2; i > 0; i-- ) {
				 // Delta[i] = Delta[i+1] * WeightT[i+1],
				 blasStatus = clAmdBlasSgemm( clAmdBlasRowMajor, clAmdBlasNoTrans, clAmdBlasNoTrans, this->minibatch, this->dimensions[i], this->dimensions[i+1],1.0f, this->delta[i+1],
					this->dimensions[i+1], this->weightT[i+1], this->dimensions[i],  0.0f, this->delta[i], this->dimensions[i], 1, &this->CLContext->m_cmd_queues[0],0,NULL,NULL );
				 AMDBLAS_CHECK(blasStatus);

				 // Delta[i] = derivative(Delta[i],Output[i])
				 this->derivative(i, this->delta[i],this->inputs[i+1], this->delta[i],this->dimensions[i], this->minibatch );

                 this->transpose_float_matrix(this->delta[i], deltaT[i], this->dimensions[i], this->minibatch);
			 }

	         CL_CHECK( clFinish(this->CLContext->m_cmd_queues[0]) );

			 if ( doChkPointing)
			      MLP_LOCK(&this->chkPointingLock);

			 for ( int i = nLayers-1; i > 0; i-- ) {
				  float coef = this->etas[i];
				  float mm = this->momentum;

				  // curVarWeightT[i] = DeltaT[i] * Output[i-1] , here curVarWeight[i] is in transposed form
				  blasStatus = clAmdBlasSgemm( clAmdBlasRowMajor,clAmdBlasNoTrans,clAmdBlasNoTrans, this->dimensions[i], this->dimensions[i-1], this->minibatch, coef,
					  deltaT[i], this->minibatch, this->inputs[i],this->dimensions[i-1],0.0f,curVarWeight[i],this->dimensions[i-1],1,&this->CLContext->m_cmd_queues[0],0,NULL,NULL);
				  AMDBLAS_CHECK(blasStatus);

				  // curVarWeightT[i] = curVarWeightT[i] + mm * lastVarWeightT[i], regarding the two Matrixes as two vectors
				  blasStatus = clAmdBlasSaxpy(this->dimensions[i]*this->dimensions[i-1],mm,lastVarWeight[i],0,1,curVarWeight[i],0,1,1,&this->CLContext->m_cmd_queues[0],0,NULL,NULL);
                  AMDBLAS_CHECK(blasStatus);

				  // WeightT[i] = WeightT[i] + 1.0 * curVarWeightT[i],  regarding the two Matrixes as two vectors
				  blasStatus = clAmdBlasSaxpy(this->dimensions[i]*this->dimensions[i-1],1.0f,curVarWeight[i],0,1,this->weightT[i],0,1,1,&this->CLContext->m_cmd_queues[0],0,NULL,NULL);
                  AMDBLAS_CHECK(blasStatus);

				  // curVarBias[i] = DeltaT[i] * (1,1, ... 1)T
                  blasStatus=clAmdBlasSgemv(clAmdBlasRowMajor, clAmdBlasNoTrans, this->dimensions[i], this->minibatch, coef, deltaT[i], this->minibatch, OnesVector,
					  0, 1, 0.0f, curVarBias[i], 0, 1, 1, &this->CLContext->m_cmd_queues[0], 0, NULL, NULL);

                  // curVarBias[i] = curVarBias[i] + mm * lastVarBias[i]
				  blasStatus = clAmdBlasSaxpy(this->dimensions[i],mm,lastVarBias[i],0,1,curVarBias[i],0,1,1,&this->CLContext->m_cmd_queues[0],0,NULL,NULL);
                  AMDBLAS_CHECK(blasStatus);


				  // Bias[i] = Bias[i] + 1.0 * curVarBias[i]
				  blasStatus = clAmdBlasSaxpy(this->dimensions[i],1.0f,curVarBias[i],0,1,this->biases[i],0,1,1,&this->CLContext->m_cmd_queues[0],0,NULL,NULL);
                  AMDBLAS_CHECK(blasStatus);

				  this->expandFloatVectorToMatrix(this->biases[i], biasesMatrix[i], this->dimensions[i], this->minibatch);

			 };

			 if ( doChkPointing )
			     MLP_UNLOCK(&this->chkPointingLock);

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
                  MLP_LOCK(&this->chkPointingLock);
			      this->currBatchNo = myBatch;
				  this->currEpoch = myEpoch;
                  MLP_UNLOCK(&this->chkPointingLock);
			 };
	    } // end of all baches

		myEpoch++;
		myBatch = 0;
		this->dataProviderp->resetDataProvider();

		if ( doChkPointing ) {
             MLP_LOCK(&this->chkPointingLock);
			 this->currBatchNo = myBatch;
		     this->currEpoch = myEpoch;
             MLP_UNLOCK(&this->chkPointingLock);
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


