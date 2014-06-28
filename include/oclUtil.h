/*
 *  COPYRIGHT:  Copyright (c) 2014 Advanced Micro Devices, Inc.  All rights reserved
 *
 *   Written by Qianfeng Zhang@amd.com ( March 2014 )
 *
 */


#ifndef _OCL_UTIL_H
#define _OCL_UTIL_H

#include <CL/cl.h>

extern int choose_ocl_igpu_device(cl_device_id &theDevice);
extern int choose_ocl_dgpu_device(cl_device_id &theDevice);
extern int choose_ocl_cpu_device(cl_device_id &theDevice);

extern int choose_ocl_dgpu_devices(cl_device_id theDevices[], int num);

extern int setup_simple_ocl_context(cl_device_id &theDevice, cl_context &theContext, int numQueue, cl_command_queue *theQueues);

extern bool isAMDAPU(cl_device_id theDevice );
extern bool isIGPU(cl_device_id theDevice ); 
extern bool isAMDDevice(cl_device_id theDevice ); 

#endif