/*
 *  COPYRIGHT:  Copyright (c) 2014 Advanced Micro Devices, Inc.  All rights reserved
 *
 *   Written by Qianfeng Zhang@amd.com ( March 2014 )
 *
 */


#ifndef _SINGLE_DEV_CLASS_H_
#define _SINGLE_DEV_CLASS_H_

#include <CL/cl.h>

enum MLP_OCL_DEVTYPE {
	MLP_OCL_CPU, 
	MLP_OCL_IGPU, 
	MLP_OCL_DGPU,
	MLP_OCL_DI_GPU
}; 

class SingleDevClass
{			
public:
	cl_device_id m_device;
	cl_context m_context;
	cl_command_queue m_cmd_queues[2];
	int numQueues; 
	cl_program m_program;
    MLP_OCL_DEVTYPE devtype; 
public:
	SingleDevClass();
	SingleDevClass(MLP_OCL_DEVTYPE type); 
	~SingleDevClass();	
};

#endif 
