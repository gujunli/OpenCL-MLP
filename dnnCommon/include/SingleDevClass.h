/*
 *  COPYRIGHT:  Copyright (c) 2014 Advanced Micro Devices, Inc.  All rights reserved
 *
 *   Written by Qianfeng Zhang@amd.com ( March 2014 )
 *
 */


#ifndef _SINGLE_DEV_CLASS_H_
#define _SINGLE_DEV_CLASS_H_

#include <CL/cl.h>

enum DNN_OCL_DEVTYPE {
	DNN_OCL_CPU,
	DNN_OCL_IGPU,
	DNN_OCL_DGPU,
	DNN_OCL_DI_GPU
};

class SingleDevClass
{
public:
	cl_device_id m_device;
	cl_context m_context;
	cl_command_queue m_queues[2];
	int numQueues;
	cl_program m_program;
    DNN_OCL_DEVTYPE devtype;
public:
    LIBDNNAPI	SingleDevClass();
	LIBDNNAPI SingleDevClass(DNN_OCL_DEVTYPE type);
	LIBDNNAPI ~SingleDevClass();
};

#endif
