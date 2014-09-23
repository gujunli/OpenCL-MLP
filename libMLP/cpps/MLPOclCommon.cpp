/*
 *  COPYRIGHT:  Copyright (c) 2014 Advanced Micro Devices, Inc.  All rights reserved
 *
 *   Written by Qianfeng Zhang@amd.com ( March 2014 )
 *
 */

#include "DNNConstants.h"
#include "MLPUtil.h"
#include "MLPOclCommon.h"

#include <iostream>
#include <fstream>
#include <cmath>

// the simpleast method for transposition, the width can be any value
void cmn_transpose_matrix_simple(cl_command_queue &cmdQueue, MLP_Kerns &kerns, cl_mem &A_cl, cl_mem &At_cl, int width, int height)
{
	size_t locals[2];
	size_t globals[2];

	CL_CHECK( clSetKernelArg(kerns.transpose_sim_kernel, 0, sizeof(cl_mem), &A_cl) );
	CL_CHECK( clSetKernelArg(kerns.transpose_sim_kernel, 1, sizeof(cl_mem), &At_cl) );
	CL_CHECK( clSetKernelArg(kerns.transpose_sim_kernel, 2, sizeof(cl_int), &width) );
	CL_CHECK( clSetKernelArg(kerns.transpose_sim_kernel, 3, sizeof(cl_int), &height) );

	locals[0]=1;
	locals[1]=256;

	globals[0]= width;
	globals[1]= ROUNDK(height,256);

	CL_CHECK( clEnqueueNDRangeKernel(cmdQueue, kerns.transpose_sim_kernel, 2, NULL, globals,locals, 0, NULL, NULL) );
};

// do the matrix transposition by dividing the matrix into 32x32 size blocks, with each local group handles one block
void cmn_transpose_matrix_32x32(cl_command_queue &cmdQueue, MLP_Kerns &kerns, cl_mem &A_cl, cl_mem &At_cl, int width, int height)
{
	size_t locals[2];
	size_t globals[2];

	CL_CHECK( clSetKernelArg(kerns.transpose_kernel32, 0, sizeof(cl_mem), &A_cl) );
	CL_CHECK( clSetKernelArg(kerns.transpose_kernel32, 1, sizeof(cl_mem), &At_cl) );
	CL_CHECK( clSetKernelArg(kerns.transpose_kernel32, 2, sizeof(cl_int), &width) );
	CL_CHECK( clSetKernelArg(kerns.transpose_kernel32, 3, sizeof(cl_int), &height) );

	locals[0]=8;
	locals[1]=32;

	globals[0]= width/4;
	globals[1]= height;

	CL_CHECK( clEnqueueNDRangeKernel(cmdQueue, kerns.transpose_kernel32, 2, NULL, globals,locals, 0, NULL, NULL) );
};

// do the matrix transposition by dividing the matrix row into 4-unit segments, with each thread handles 4 units of one row
void cmn_transpose_matrix_f4(cl_command_queue &cmdQueue, MLP_Kerns &kerns, cl_mem &A_cl, cl_mem &At_cl, int width, int height)
{
	size_t locals[2];
	size_t globals[2];

	CL_CHECK( clSetKernelArg(kerns.transpose_kernel4, 0, sizeof(cl_mem), &A_cl) );
	CL_CHECK( clSetKernelArg(kerns.transpose_kernel4, 1, sizeof(cl_mem), &At_cl) );
	CL_CHECK( clSetKernelArg(kerns.transpose_kernel4, 2, sizeof(cl_int), &width) );
	CL_CHECK( clSetKernelArg(kerns.transpose_kernel4, 3, sizeof(cl_int), &height) );

	locals[0]=1;
	locals[1]=256;

	globals[0]= width/4;
	globals[1]= ROUNDK(height,256);

	CL_CHECK( clEnqueueNDRangeKernel(cmdQueue, kerns.transpose_kernel4, 2, NULL, globals,locals, 0, NULL, NULL) );
};



void cmn_expandFloatVectorToMatrix(cl_command_queue &cmdQueue, MLP_Kerns &kerns, cl_mem &myVector, cl_mem &myMatrix, int width, int height)
{
	size_t locals[2];
	size_t globals[2];

	CL_CHECK( clSetKernelArg(kerns.expandMatrix_kernel, 0, sizeof(cl_mem), &myVector) );
	CL_CHECK( clSetKernelArg(kerns.expandMatrix_kernel, 1, sizeof(cl_mem), &myMatrix) );
	CL_CHECK( clSetKernelArg(kerns.expandMatrix_kernel, 2, sizeof(cl_int), &width) );
	CL_CHECK( clSetKernelArg(kerns.expandMatrix_kernel, 3, sizeof(cl_int), &height) );

	locals[0]=16;
	locals[1]=16;

	globals[0]= ROUNDK(DIVUPK(width,4),16);
	globals[1]= ROUNDK(height,16);

	CL_CHECK( clEnqueueNDRangeKernel(cmdQueue, kerns.expandMatrix_kernel, 2, NULL, globals,locals, 0, NULL, NULL) );

	// CL_CHECK(clWaitForEvents(1, &event));
};

void cmn_activate_sigmoid(cl_command_queue &cmdQueue, MLP_Kerns &kerns, cl_mem &x, cl_mem &y, int width, int height )
{
	CL_CHECK( clSetKernelArg(kerns.activate_sigmoid_kernel, 0, sizeof(cl_mem), &x) );
	CL_CHECK( clSetKernelArg(kerns.activate_sigmoid_kernel, 1, sizeof(cl_mem), &y) );
	CL_CHECK( clSetKernelArg(kerns.activate_sigmoid_kernel, 2, sizeof(cl_uint), &width) );
	CL_CHECK( clSetKernelArg(kerns.activate_sigmoid_kernel, 3, sizeof(cl_uint), &height) );

	size_t locals[2];
	size_t globals[2];

	if ( DIVUPK(width,4) < 128 ) {   // one work group can cover whole row of units
		 // let pow be the upper value of DIVUPK(width,4)
		 int pow=1;
		 while ( pow < DIVUPK(width,4) )
			     pow *= 2;

	    locals[0] = pow;
	    locals[1] = 256/pow;
	    globals[0] = pow;
	    globals[1] = ROUNDK(height,256/pow);
	}
	else {                // need to split one row into multiple groups
	    locals[0] = 16;
	    locals[1] = 16;
	    globals[0] = ROUNDK(DIVUPK(width,4),16);
	    globals[1] = ROUNDK(height,16);
	};

	CL_CHECK( clEnqueueNDRangeKernel(cmdQueue,kerns.activate_sigmoid_kernel,2,NULL,globals,locals,0,NULL,NULL) );
}

void cmn_activate_tanh(cl_command_queue &cmdQueue, MLP_Kerns &kerns, cl_mem &x, cl_mem &y, int width, int height )
{
	CL_CHECK( clSetKernelArg(kerns.activate_tanh_kernel, 0, sizeof(cl_mem), &x) );
	CL_CHECK( clSetKernelArg(kerns.activate_tanh_kernel, 1, sizeof(cl_mem), &y) );
	CL_CHECK( clSetKernelArg(kerns.activate_tanh_kernel, 2, sizeof(cl_uint), &width) );
	CL_CHECK( clSetKernelArg(kerns.activate_tanh_kernel, 3, sizeof(cl_uint), &height) );

	size_t locals[2];
	size_t globals[2];

	if ( DIVUPK(width,4) < 128 ) {   // one work group can cover while row of units
		 // let pow be the upper value of DIVUPK(width,4)
		 int pow=1;
		 while ( pow < DIVUPK(width,4) )
			     pow *= 2;

	    locals[0] = pow;
	    locals[1] = 256/pow;
	    globals[0] = pow;
	    globals[1] = ROUNDK(height,256/pow);
	}
	else {                // need to split one row into multiple groups
	    locals[0] = 16;
	    locals[1] = 16;
	    globals[0] = ROUNDK(DIVUPK(width,4),16);
	    globals[1] = ROUNDK(height,16);
	};

	CL_CHECK( clEnqueueNDRangeKernel(cmdQueue,kerns.activate_tanh_kernel,2,NULL,globals,locals,0,NULL,NULL) );
};

void cmn_activate_softmax(cl_command_queue &cmdQueue, MLP_Kerns &kerns, cl_mem &x, cl_mem &y, int width, int height )
{
	if ( DIVUPK(width,4) < 256 ) {  // let each thread to handle 4 units, each row of units can be handled inside one work group
		 CL_CHECK( clSetKernelArg(kerns.activate_softmax_kernel1, 0, sizeof(cl_mem), &x) );
	     CL_CHECK( clSetKernelArg(kerns.activate_softmax_kernel1, 1, sizeof(cl_mem), &y) );
	     CL_CHECK( clSetKernelArg(kerns.activate_softmax_kernel1, 2, sizeof(cl_uint), &width) );
	     CL_CHECK( clSetKernelArg(kerns.activate_softmax_kernel1, 3, sizeof(cl_uint), &height) );

		 // let pow be the upper value of ROUNDK(width,4)
		 int pow=1;
		 while ( pow < DIVUPK(width,4) )
			     pow *= 2;

	     size_t locals[2];
	     size_t globals[2];

	     locals[0] = pow;
	     locals[1] = 256/pow;
	     globals[0] = pow;
	     globals[1] = ROUNDK(height,256/pow);

	     CL_CHECK( clEnqueueNDRangeKernel(cmdQueue,kerns.activate_softmax_kernel1,2,NULL,globals,locals,0,NULL,NULL) );
	}
	else {                // let each work group to handle one row of units, each thread handle "DIVUP(width,4)/256" number of units
     	 CL_CHECK( clSetKernelArg(kerns.activate_softmax_kernel2, 0, sizeof(cl_mem), &x) );
	     CL_CHECK( clSetKernelArg(kerns.activate_softmax_kernel2, 1, sizeof(cl_mem), &y) );
	     CL_CHECK( clSetKernelArg(kerns.activate_softmax_kernel2, 2, sizeof(cl_uint), &width) );
	     CL_CHECK( clSetKernelArg(kerns.activate_softmax_kernel2, 3, sizeof(cl_uint), &height) );

	     size_t locals[2];
	     size_t globals[2];

	     locals[0] = 256;
	     locals[1] = 1;
	     globals[0] = 256;
	     globals[1] = ROUNDK(height,1);

	     CL_CHECK( clEnqueueNDRangeKernel(cmdQueue,kerns.activate_softmax_kernel2,2,NULL,globals,locals,0,NULL,NULL) );
	};
};

void cmn_activate_identity(cl_command_queue &cmdQueue, MLP_Kerns &kerns, cl_mem &x, cl_mem &y, int width, int height )
{
};

void cmn_calculateError_SSE(cl_command_queue &cmdQueue, MLP_Kerns &kerns, cl_mem &output, cl_mem &target, cl_mem &reduceMem, float *reduceBuf, int width, int height, float &ret )
{
	if ( DIVUPK(width,4) < 256 ) {  // let each thread to handle 4 units, each row of units can be handled inside one work group
     	 CL_CHECK( clSetKernelArg(kerns.calculateError_SSE_kernel1, 0, sizeof(cl_mem), &output) );
	     CL_CHECK( clSetKernelArg(kerns.calculateError_SSE_kernel1, 1, sizeof(cl_mem), &target) );
		 CL_CHECK( clSetKernelArg(kerns.calculateError_SSE_kernel1, 2, sizeof(cl_mem), &reduceMem) );
	     CL_CHECK( clSetKernelArg(kerns.calculateError_SSE_kernel1, 3, sizeof(cl_uint), &width) );
	     CL_CHECK( clSetKernelArg(kerns.calculateError_SSE_kernel1, 4, sizeof(cl_uint), &height) );

		 // let pow be the upper value of ROUNDK(width,4)
		 int pow=1;
		 while ( pow < DIVUPK(width,4) )
			     pow *= 2;

	     size_t locals[2];
	     size_t globals[2];

	     locals[0] = pow;
	     locals[1] = 256/pow;
	     globals[0] = pow;
	     globals[1] = ROUNDK(height,256/pow);

	     CL_CHECK( clEnqueueNDRangeKernel(cmdQueue,kerns.calculateError_SSE_kernel1,2,NULL,globals,locals,0,NULL,NULL) );
	}
	else {                // let each work group to handle one row of units, each thread handle "DIVUP(width,4)/256" number of units
     	 CL_CHECK( clSetKernelArg(kerns.calculateError_SSE_kernel2, 0, sizeof(cl_mem), &output) );
	     CL_CHECK( clSetKernelArg(kerns.calculateError_SSE_kernel2, 1, sizeof(cl_mem), &target) );
		 CL_CHECK( clSetKernelArg(kerns.calculateError_SSE_kernel2, 2, sizeof(cl_mem), &reduceMem) );
	     CL_CHECK( clSetKernelArg(kerns.calculateError_SSE_kernel2, 3, sizeof(cl_uint), &width) );
	     CL_CHECK( clSetKernelArg(kerns.calculateError_SSE_kernel2, 4, sizeof(cl_uint), &height) );

	     size_t locals[2];
	     size_t globals[2];

	     locals[0] = 256;
	     locals[1] = 1;
	     globals[0] = 256;
	     globals[1] = ROUNDK(height,1);

	     CL_CHECK( clEnqueueNDRangeKernel(cmdQueue,kerns.calculateError_SSE_kernel2,2,NULL,globals,locals,0,NULL,NULL) );
	};

    CL_CHECK(clEnqueueReadBuffer(cmdQueue,reduceMem,CL_TRUE,0,sizeof(cl_float)*height,reduceBuf,0,NULL,NULL ));
	ret = 0.0f;
	for (int i=0; i< height; i++)
         ret += reduceBuf[i]/(float)height;  // calculate average error for frames
};

void cmn_calculateError_CE(cl_command_queue &cmdQueue, MLP_Kerns &kerns, cl_mem &output, cl_mem &target, cl_mem &reduceMem, float *reduceBuf, int width, int height, float &ret )
{
	if ( DIVUPK(width,4) < 256 ) {  // let each thread to handle 4 units, each row of units can be handled inside one work group
     	 CL_CHECK( clSetKernelArg(kerns.calculateError_CE_kernel1, 0, sizeof(cl_mem), &output) );
	     CL_CHECK( clSetKernelArg(kerns.calculateError_CE_kernel1, 1, sizeof(cl_mem), &target) );
		 CL_CHECK( clSetKernelArg(kerns.calculateError_CE_kernel1, 2, sizeof(cl_mem), &reduceMem) );
	     CL_CHECK( clSetKernelArg(kerns.calculateError_CE_kernel1, 3, sizeof(cl_uint), &width) );
	     CL_CHECK( clSetKernelArg(kerns.calculateError_CE_kernel1, 4, sizeof(cl_uint), &height) );

		 // let pow be the upper value of ROUNDK(width,4)
		 int pow=1;
		 while ( pow < DIVUPK(width,4) )
			     pow *= 2;

	     size_t locals[2];
	     size_t globals[2];

	     locals[0] = pow;
	     locals[1] = 256/pow;
	     globals[0] = pow;
	     globals[1] = ROUNDK(height,256/pow);

	     CL_CHECK( clEnqueueNDRangeKernel(cmdQueue,kerns.calculateError_CE_kernel1,2,NULL,globals,locals,0,NULL,NULL) );
	}
	else {                // let each work group to handle one row of units, each thread handle "DIVUP(width,4)/256" number of units
     	 CL_CHECK( clSetKernelArg(kerns.calculateError_CE_kernel2, 0, sizeof(cl_mem), &output) );
	     CL_CHECK( clSetKernelArg(kerns.calculateError_CE_kernel2, 1, sizeof(cl_mem), &target) );
		 CL_CHECK( clSetKernelArg(kerns.calculateError_CE_kernel2, 2, sizeof(cl_mem), &reduceMem) );
	     CL_CHECK( clSetKernelArg(kerns.calculateError_CE_kernel2, 3, sizeof(cl_uint), &width) );
	     CL_CHECK( clSetKernelArg(kerns.calculateError_CE_kernel2, 4, sizeof(cl_uint), &height) );

	     size_t locals[2];
	     size_t globals[2];

	     locals[0] = 256;
	     locals[1] = 1;
	     globals[0] = 256;
	     globals[1] = ROUNDK(height,1);

	     CL_CHECK( clEnqueueNDRangeKernel(cmdQueue,kerns.calculateError_CE_kernel2,2,NULL,globals,locals,0,NULL,NULL) );
	};

    CL_CHECK(clEnqueueReadBuffer(cmdQueue,reduceMem,CL_TRUE,0,sizeof(cl_float)*height,reduceBuf,0,NULL,NULL ));
	ret = 0.0f;
	for (int i=0; i< height; i++)
		 ret += reduceBuf[i]/(float)height;  // calculate average error for frames
};

void cmn_calculateDelta_SSE_Sigmoid(cl_command_queue &cmdQueue, MLP_Kerns &kerns, cl_mem &output, cl_mem &target, cl_mem &delta, int width, int height)
{
	CL_CHECK( clSetKernelArg(kerns.calculateDelta_SSE_Sigmoid_kernel, 0, sizeof(cl_mem), &output) );
	CL_CHECK( clSetKernelArg(kerns.calculateDelta_SSE_Sigmoid_kernel, 1, sizeof(cl_mem), &target) );
	CL_CHECK( clSetKernelArg(kerns.calculateDelta_SSE_Sigmoid_kernel, 2, sizeof(cl_mem), &delta) );
	CL_CHECK( clSetKernelArg(kerns.calculateDelta_SSE_Sigmoid_kernel, 3, sizeof(cl_int), &width) );
	CL_CHECK( clSetKernelArg(kerns.calculateDelta_SSE_Sigmoid_kernel, 4, sizeof(cl_int), &height) );

	size_t locals[2];
	size_t globals[2];

	if ( DIVUPK(width,4) < 128 ) {   // one work group can cover whole row of units
		// let pow be the upper value of DIVUPK(width,4)
		int pow=1;
		while ( pow < DIVUPK(width,4) )
		        pow *= 2;

	    locals[0] = pow;
	    locals[1] = 256/pow;
	    globals[0] = pow;
	    globals[1] = ROUNDK(height,256/pow);
	}
	else {                // need to split one row into multiple groups
	    locals[0] = 16;
	    locals[1] = 16;
	    globals[0] = ROUNDK(DIVUPK(width,4),16);
	    globals[1] = ROUNDK(height,16);
	};


	CL_CHECK( clEnqueueNDRangeKernel(cmdQueue,kerns.calculateDelta_SSE_Sigmoid_kernel,2,NULL,globals,locals,0,NULL,NULL) );
};

void cmn_calculateDelta_CE_Softmax(cl_command_queue &cmdQueue, MLP_Kerns &kerns, cl_mem &output, cl_mem &target, cl_mem &delta, int width, int height)
{
	CL_CHECK( clSetKernelArg(kerns.calculateDelta_CE_Softmax_kernel, 0, sizeof(cl_mem), &output) );
	CL_CHECK( clSetKernelArg(kerns.calculateDelta_CE_Softmax_kernel, 1, sizeof(cl_mem), &target) );
	CL_CHECK( clSetKernelArg(kerns.calculateDelta_CE_Softmax_kernel, 2, sizeof(cl_mem), &delta) );
	CL_CHECK( clSetKernelArg(kerns.calculateDelta_CE_Softmax_kernel, 3, sizeof(cl_int), &width) );
	CL_CHECK( clSetKernelArg(kerns.calculateDelta_CE_Softmax_kernel, 4, sizeof(cl_int), &height) );

	size_t locals[2];
	size_t globals[2];

	if ( DIVUPK(width,4) < 128 ) {   // one work group can cover whole row of units
		 // let pow be the upper value of DIVUPK(width,4)
		 int pow=1;
		 while ( pow < DIVUPK(width,4) )
			     pow *= 2;

	    locals[0] = pow;
	    locals[1] = 256/pow;
	    globals[0] = pow;
	    globals[1] = ROUNDK(height,256/pow);
	}
	else {                // need to split one row into multiple groups
	    locals[0] = 16;
	    locals[1] = 16;
	    globals[0] = ROUNDK(DIVUPK(width,4),16);
	    globals[1] = ROUNDK(height,16);
	};

	CL_CHECK( clEnqueueNDRangeKernel(cmdQueue,kerns.calculateDelta_CE_Softmax_kernel,2,NULL,globals,locals,0,NULL,NULL) );
};

void cmn_derivative_sigmoid(cl_command_queue &cmdQueue, MLP_Kerns &kerns, cl_mem &delta1, cl_mem &y, cl_mem &delta2, int width, int height)
{
	CL_CHECK( clSetKernelArg(kerns.derivative_sigmoid_kernel, 0, sizeof(cl_mem), &delta1) );
	CL_CHECK( clSetKernelArg(kerns.derivative_sigmoid_kernel, 1, sizeof(cl_mem), &y) );
	CL_CHECK( clSetKernelArg(kerns.derivative_sigmoid_kernel, 2, sizeof(cl_mem), &delta2) );
	CL_CHECK( clSetKernelArg(kerns.derivative_sigmoid_kernel, 3, sizeof(cl_uint), &width) );
	CL_CHECK( clSetKernelArg(kerns.derivative_sigmoid_kernel, 4, sizeof(cl_uint), &height) );

	size_t globals[2];
	size_t locals[2];

	if ( DIVUPK(width,4) < 128 ) {   // one work group can cover whole row of units
		 // let pow be the upper value of DIVUPK(width,4)
		 int pow=1;
		 while ( pow < DIVUPK(width,4) )
			     pow *= 2;

	    locals[0] = pow;
	    locals[1] = 256/pow;
	    globals[0] = pow;
	    globals[1] = ROUNDK(height,256/pow);
	}
	else {                // need to split one row into multiple groups
	    locals[0] = 16;
	    locals[1] = 16;
	    globals[0] = ROUNDK(DIVUPK(width,4),16);
	    globals[1] = ROUNDK(height,16);
	};

	CL_CHECK( clEnqueueNDRangeKernel(cmdQueue,kerns.derivative_sigmoid_kernel,2,NULL,globals,locals,0,NULL,NULL) );
};

void cmn_derivative_tanh(cl_command_queue &cmdQueue, MLP_Kerns &kerns, cl_mem &delta1, cl_mem &y, cl_mem &delta2, int width, int height)
{
	CL_CHECK( clSetKernelArg(kerns.derivative_tanh_kernel, 0, sizeof(cl_mem), &delta1) );
	CL_CHECK( clSetKernelArg(kerns.derivative_tanh_kernel, 1, sizeof(cl_mem), &y) );
	CL_CHECK( clSetKernelArg(kerns.derivative_tanh_kernel, 2, sizeof(cl_mem), &delta2) );
	CL_CHECK( clSetKernelArg(kerns.derivative_tanh_kernel, 3, sizeof(cl_uint), &width) );
	CL_CHECK( clSetKernelArg(kerns.derivative_tanh_kernel, 4, sizeof(cl_uint), &height) );

	size_t globals[2];
	size_t locals[2];

	if ( DIVUPK(width,4) < 128 ) {   // one work group can cover whole row of units
		 // let pow be the upper value of DIVUPK(width,4)
		 int pow=1;
		 while ( pow < DIVUPK(width,4) )
			     pow *= 2;

	    locals[0] = pow;
	    locals[1] = 256/pow;
	    globals[0] = pow;
	    globals[1] = ROUNDK(height,256/pow);
	}
	else {                // need to split one row into multiple groups
	    locals[0] = 16;
	    locals[1] = 16;
	    globals[0] = ROUNDK(DIVUPK(width,4),16);
	    globals[1] = ROUNDK(height,16);
	};

	CL_CHECK( clEnqueueNDRangeKernel(cmdQueue,kerns.derivative_tanh_kernel,2,NULL,globals,locals,0,NULL,NULL) );
};


// the following functions are only used for debugging

void print_dev_data(char *header, cl_command_queue &cmdQueue, cl_mem devBuf, int width, int height)
{
    fprint_dev_data(cout, header, cmdQueue, devBuf, width, height);
};

void fprint_dev_data(ostream &ofile, char *header, cl_command_queue &cmdQueue, cl_mem devBuf, int width, int height)
{
    float *hostBuf;

    hostBuf = new float[width * height];

    CL_CHECK(clEnqueueReadBuffer(cmdQueue,devBuf,CL_TRUE,0,sizeof(cl_float)*width*height,hostBuf,0,NULL,NULL ));

	ofile << endl << header << endl;

	for (int i=0; i< height; i++ ) {
		 ofile << "Row " << i << ":  " ;

		 for (int j=0; j< width; j++)
			  ofile << hostBuf[i*width+j] << " " ;

		 ofile << endl << endl;
	};

	delete [] hostBuf;
};

bool check_zero(float fval)
{
   return((fval == 0.0f)?true:false);
};

bool check_nan(float fval)
{
#ifdef _WIN32
       if ( _isnan((double)fval) )
           return(true);
	   else
		   return(false);
#else     // for Linux
       if ( isnan(fval) )
           return(true);
	   else
		   return(false);
#endif
};

bool check_inf(float fval)
{
#ifdef _WIN32
       if ( !_finite((double)fval) )
            return(true);
	   else
		    return(false);
#else     // for Linux
       if ( isinf(fval) )
            return(true);
	   else
		    return(false);
#endif
};


void check_memory(char *header, cl_command_queue &cmdQueue, cl_mem devBuf, int length, CHECK_FLOAT checkFunc)
{
    float *hostBuf;

    hostBuf = new float[length];

    CL_CHECK(clEnqueueReadBuffer(cmdQueue,devBuf,CL_TRUE,0,sizeof(cl_float)*length,hostBuf,0,NULL,NULL ));

    for (int i=0; i< length; i++) {
		 if  ( checkFunc(hostBuf[i]) ) {
               mlp_log(header,"Invalid data found");
		       if ( checkFunc == check_zero )
				    mlp_log("CHECK_MEMORY", "Zero value data found");
			   else
				   if ( checkFunc == check_nan )
					    mlp_log("CHECK_MEMORY", "NAN value data found");
				   else
					    mlp_log("CHECK_MEMORY", "INF value data found");

			   MLP_Exception("Invalid float data produced");
         };
    };

	delete [] hostBuf;
};


