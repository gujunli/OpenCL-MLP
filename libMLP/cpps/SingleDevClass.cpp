/*
 *  COPYRIGHT:  Copyright (c) 2014 Advanced Micro Devices, Inc.  All rights reserved
 *
 *   Written by Qianfeng Zhang@amd.com ( March 2014 )
 *
 */

#include "DNNUtil.h"
#include "oclUtil.h"
#include "SingleDevClass.h"

SingleDevClass::SingleDevClass()
{
	int result;

	this->numQueues = 0;

    if (  (result=choose_ocl_dgpu_device(this->m_device)) < 0 ) {
          dnn_log("DNN", "Failed to choose one OpenCL discrete GPUd evice for the application, try to use integrated GPU device\n");
		  if ( (result=choose_ocl_igpu_device(this->m_device)) < 0 ) {
		        dnn_log("DNN", "Failed to choose one OpenCL platform and device for the application\n");
		        dnn_log_retval("DNN", result);
		        DNN_Exception("");
		  }
		  else
			    this->devtype = DNN_OCL_IGPU;
	}
	else
	      this->devtype = DNN_OCL_DGPU;

	if  ( (result=setup_simple_ocl_context(this->m_device, this->m_context, 1, &this->m_queues[0])) < 0 ) {
		  dnn_log("DNN", "Failed to setup OpenCL context and queue on the selected device\n");
		  dnn_log_retval("DNN", result);
		  DNN_Exception("");
	};

	this->numQueues = 1;
};

SingleDevClass::SingleDevClass(DNN_OCL_DEVTYPE type)
{
	int result=-1;
	enum DNN_OCL_DEVTYPE setType;

	this->numQueues = 0;
	setType = type;

	switch (type) {
        case DNN_OCL_DGPU:
            result = choose_ocl_dgpu_device(this->m_device);
			break;
        case DNN_OCL_IGPU:
            result = choose_ocl_igpu_device(this->m_device);
			break;
        case DNN_OCL_CPU:
            result = choose_ocl_cpu_device(this->m_device);
			break;
        case DNN_OCL_DI_GPU:
            if ( (result = choose_ocl_dgpu_device(this->m_device)) < 0 ) {
                  dnn_log("DNN", "Failed to choose one OpenCL discrete GPU device for the application, try to use integrated GPU device\n");
                  result = choose_ocl_igpu_device(this->m_device);
                  setType = DNN_OCL_IGPU;
            }
            else
                  setType = DNN_OCL_DGPU;
            break;
        default:
            dnn_log("DNN", "Incorrect OpenCL device type as parameter");
            DNN_Exception("");
	};

    if ( result < 0 ) {
		  dnn_log("DNN", "Failed to choose one OpenCL platform and device for the application\n");
		  dnn_log_retval("DNN", result);
		  DNN_Exception("");
	};

	if  ( (result=setup_simple_ocl_context(this->m_device, this->m_context, 1,  &this->m_queues[0])) < 0 ) {
		  dnn_log("DNN", "Failed to setup OpenCL context and queue on the selected device\n");
		  dnn_log_retval("DNN", result);
		  DNN_Exception("");
	};

	this->numQueues = 1;
	this->devtype = setType;
};


SingleDevClass::~SingleDevClass()
{
 	CL_CHECK( clReleaseContext(this->m_context) );

	CL_CHECK( clReleaseCommandQueue(this->m_queues[0]) );

	if ( this->numQueues == 2 )
	     CL_CHECK( clReleaseCommandQueue(this->m_queues[1]) );

	CL_CHECK( clReleaseDevice(this->m_device) );
};

