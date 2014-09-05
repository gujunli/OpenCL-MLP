/*
 *  COPYRIGHT:  Copyright (c) 2014 Advanced Micro Devices, Inc.  All rights reserved
 *
 *   Written by Qianfeng Zhang@amd.com ( March 2014 )
 *
 */

#ifndef _MLP_COMMON_H_
#define _MLP_COMMON_H_

#include <CL/cl.h>
#include <iostream>
#include <fstream>

using namespace std;

typedef struct kernels {
	cl_kernel activate_sigmoid_kernel;
	cl_kernel activate_softmax_kernel1;
	cl_kernel activate_softmax_kernel2;
	cl_kernel activate_tanh_kernel;

	cl_kernel derivative_sigmoid_kernel;
	cl_kernel derivative_tanh_kernel;

	cl_kernel calculateError_SSE_kernel1;
	cl_kernel calculateError_SSE_kernel2;
	cl_kernel calculateError_CE_kernel1;
	cl_kernel calculateError_CE_kernel2;

	cl_kernel calculateDelta_SSE_Sigmoid_kernel;
	cl_kernel calculateDelta_CE_Softmax_kernel;

    cl_kernel transpose_kernel32;
    cl_kernel transpose_kernel4;
    cl_kernel transpose_sim_kernel;

    cl_kernel expandMatrix_kernel;
} MLP_Kerns;

extern void cmn_transpose_matrix_simple(cl_command_queue &cmdQueue, MLP_Kerns &kerns, cl_mem &A_cl, cl_mem &At_cl, int width, int height);
extern void cmn_transpose_matrix_32x32(cl_command_queue &cmdQueue, MLP_Kerns &kerns, cl_mem &A_cl, cl_mem &At_cl, int width, int height);
extern void cmn_transpose_matrix_f4(cl_command_queue &cmdQueue, MLP_Kerns &kerns, cl_mem &A_cl, cl_mem &At_cl, int width, int height);

extern void cmn_expandFloatVectorToMatrix(cl_command_queue &cmdQueue, MLP_Kerns &kerns, cl_mem  &myVector, cl_mem &myMatrix, int width, int height);
extern void cmn_activate_sigmoid(cl_command_queue &cmdQueue, MLP_Kerns &kerns, cl_mem &x, cl_mem &y, int width, int height );
extern void cmn_activate_tanh(cl_command_queue &cmdQueue, MLP_Kerns &kerns, cl_mem &x, cl_mem &y, int width, int height );
extern void cmn_activate_softmax(cl_command_queue &cmdQueue, MLP_Kerns &kerns, cl_mem &x, cl_mem &y, int width, int height );
extern void cmn_activate_identity(cl_command_queue &cmdQueue, MLP_Kerns &kerns, cl_mem &x, cl_mem &y, int width, int height );

void cmn_calculateError_SSE(cl_command_queue &cmdQueue, MLP_Kerns &kerns, cl_mem &output, cl_mem &target, cl_mem &reduceMem, float *reduceBuf, int width, int height, float &ret );
void cmn_calculateError_CE(cl_command_queue &cmdQueue, MLP_Kerns &kerns, cl_mem &output, cl_mem &target, cl_mem &reduceMem, float *reduceBuf, int width, int height, float &ret );

void cmn_calculateDelta_SSE_Sigmoid(cl_command_queue &cmdQueue, MLP_Kerns &kerns, cl_mem &output, cl_mem &target, cl_mem &delta, int width, int height);
void cmn_calculateDelta_CE_Softmax(cl_command_queue &cmdQueue, MLP_Kerns &kerns, cl_mem &output, cl_mem &target, cl_mem &delta, int width, int height);

void cmn_derivative_sigmoid(cl_command_queue &cmdQueue, MLP_Kerns &kerns, cl_mem &delta1, cl_mem &y, cl_mem &delta2, int width, int height);
void cmn_derivative_tanh(cl_command_queue &cmdQueue, MLP_Kerns &kerns, cl_mem &delta1, cl_mem &y, cl_mem &delta2, int width, int height);


void print_dev_data(char *header, cl_command_queue &cmdQueue, cl_mem devBuf, int width, int height);
void fprint_dev_data(ofstream &ofile, char *header, cl_command_queue &cmdQueue, cl_mem devBuf, int width, int height);


typedef bool (*CHECK_FLOAT)(float x);

bool check_zero(float x);
bool check_nan(float x);
bool check_inf(float x);
void check_memory(char *header, cl_command_queue &cmdQueue, cl_mem devBuf, int length, CHECK_FLOAT checkFunc);


#endif
