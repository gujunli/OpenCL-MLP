/*
 *  COPYRIGHT:  Copyright (c) 2014 Advanced Micro Devices, Inc.  All rights reserved
 *
 *   Written by Qianfeng Zhang@amd.com ( March 2014 )
 *
 */

#include <string>
#include <sstream>
#include <iostream>
#include <fstream>

#include <CL/cl.h>

#ifdef _WIN32
#include <intrin.h>
#endif

#include "oclUtil.h"

using namespace std;

static int choose_ocl_platform(cl_platform_id &thePlatform);
static float get_opencl_version(cl_device_id &theDevice);
static int measure_device(cl_device_id &theDevice, int &capability);

static const char *getProcessorVendor();

// setup a context with only one queue on the device, it is up to the application itself to set up more complex context
int setup_simple_ocl_context(cl_device_id &theDevice, cl_context &theContext, int numQueue, cl_command_queue *theQueues)
{
	cl_int status;
	cl_context context;
	cl_command_queue queue;
	cl_platform_id platform_id;
	// cl_context_properties cps[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)0, (cl_context_properties) 0};

	if ( numQueue < 1 || numQueue > 2 )
		 return(-4);

	if (  clGetDeviceInfo(theDevice, CL_DEVICE_PLATFORM, sizeof(cl_platform_id), &platform_id, NULL ) != CL_SUCCESS )
		  return(-1);

	//cps[1] = (cl_context_properties)platform_id;

	context = clCreateContext(NULL, 1, &theDevice, NULL, NULL, &status);
	if ( status != CL_SUCCESS )
		return(-2);

    queue = clCreateCommandQueue(context, theDevice, 0, &status);
    if ( status != CL_SUCCESS)
	    return(-3);

	theQueues[0] = queue;

	if ( numQueue == 2 ) {
         queue = clCreateCommandQueue(context, theDevice, 0, &status);
         if ( status != CL_SUCCESS)
	          return(-3);

         theQueues[1] = queue;
	};

	theContext = context;

	return(0);
};

// choose an integrated GPU device for OpenCL application
int choose_ocl_igpu_device(cl_device_id &theDevice)
{
    cl_platform_id platform_id;
	cl_device_id device_ids[5];
	cl_int status;
	cl_uint num_devices;

	if ( choose_ocl_platform(platform_id) < 0 ) {
		 return(-1);
	}

	// Try to find the integrated GPU
	status = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 5, device_ids, &num_devices);
	if ( status != CL_SUCCESS )
		 return(-2);
	else {
	     for (int i=0; i< (int)num_devices; i++) {
			 cl_bool unified_memory=false;

			 (void) clGetDeviceInfo(device_ids[i], CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(cl_bool), &unified_memory, NULL );

			 if (  unified_memory ) {   // this one is an integrated GPU
				  if ( get_opencl_version( device_ids[i]) >= 1.2f ) {
				       theDevice = device_ids[i];
				       return(0);
				  }
			 }
		 }
	}

    return(-3);
}

// choose an CPU device for OpenCL application
int choose_ocl_cpu_device(cl_device_id &theDevice)
{
    cl_platform_id platform_id;
	cl_device_id device_id;
	cl_int status;

	if ( choose_ocl_platform(platform_id) < 0 ) {
		 return(-1);
	}

	status = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
	if ( status == CL_SUCCESS ) {
		if ( get_opencl_version(device_id) >= 1.2f ) {
	         theDevice = device_id;
			 return(0);
		}
	}

    return(-2);
}

// choose an discrete GPU device for OpenCL application
int choose_ocl_dgpu_device(cl_device_id &theDevice)
{
    cl_platform_id platform_ids[4];
	cl_device_id device_ids[5], best_device;
	cl_int num_platforms=4;
	cl_uint num_devices;
	cl_int status;
    int  curr_capability, best_capability;

	status = clGetPlatformIDs(4, &platform_ids[0], (cl_uint*)&num_platforms);
	if ( status != CL_SUCCESS ) {
		return(-1);
	}

	best_capability = 0;

	for (int i=0; i< num_platforms; i++ ) {
	     status = clGetDeviceIDs(platform_ids[i], CL_DEVICE_TYPE_GPU, 5, device_ids, &num_devices);
 	     if ( status != CL_SUCCESS )
			  continue;

		 // choose the discrete GPU that has best capability
	     for (int j=0; j< (int)num_devices; j++) {
			  cl_bool unified_memory=false;

			  (void) clGetDeviceInfo(device_ids[j], CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(cl_bool), &unified_memory, NULL );
			  if ( unified_memory )  // skip the integrated GPU
				   continue;

		      if ( measure_device(device_ids[j], curr_capability) == 0 ) {
			        if ( curr_capability > best_capability ) {
						 best_capability = curr_capability;
						 best_device = device_ids[j];
				    }
			  }
			  else
				   continue;
		 }  // end of for
	}

	if ( best_capability > 0 ) {
		 theDevice = best_device;
		 return(0);
	};

    return(-2);
}


// choose the required number of GPU devices for Opencl application
int choose_ocl_dgpu_devices(cl_device_id theDevices[], int num)
{
	// to be implementd
	return(0);
};

bool isAMDAPU(cl_device_id theDevice )
{
	cl_uint  vendorID;
	cl_device_type deviceType;
	cl_bool unified_memory=false;
	size_t valuesize, sizeret;

     valuesize=sizeof(deviceType);
    (void) clGetDeviceInfo(theDevice, CL_DEVICE_TYPE, valuesize, &deviceType, &sizeret);

	 valuesize=sizeof(vendorID);
    (void) clGetDeviceInfo(theDevice, CL_DEVICE_VENDOR_ID, valuesize, &vendorID, &sizeret);

	(void) clGetDeviceInfo(theDevice, CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(cl_bool), &unified_memory, NULL );

	if (  (deviceType == CL_DEVICE_TYPE_GPU ) && unified_memory && ( vendorID == 0x1002 ) )
		   return(true);

	return(false);
};

bool isIGPU(cl_device_id theDevice )
{
	cl_device_type deviceType;
	cl_bool unified_memory=false;
	size_t valuesize, sizeret;

     valuesize=sizeof(deviceType);
    (void) clGetDeviceInfo(theDevice, CL_DEVICE_TYPE, valuesize, &deviceType, &sizeret);

	(void) clGetDeviceInfo(theDevice, CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(cl_bool), &unified_memory, NULL );

	if (  (deviceType == CL_DEVICE_TYPE_GPU ) && unified_memory  )
		   return(true);

	return(false);
};

bool isAMDDevice(cl_device_id theDevice )
{
	cl_uint  vendorID;
	size_t valuesize, sizeret;

	 valuesize=sizeof(vendorID);
    (void) clGetDeviceInfo(theDevice, CL_DEVICE_VENDOR_ID, valuesize, &vendorID, &sizeret);

	if ( vendorID == 0x1002 )
		  return(true);

	return(false);
};

static int measure_device(cl_device_id &theDevice, int &capability)
{
	cl_int  num_cu=0;
	cl_int  clock_freq=0;

    if  (clGetDeviceInfo(theDevice, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_int), &num_cu, NULL ) != CL_SUCCESS )
		 return(-1);

	if  (clGetDeviceInfo(theDevice, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(cl_int), &clock_freq, NULL ) != CL_SUCCESS )
		 return(-2);

	capability = num_cu * clock_freq;

	return(0);
};


// choose the OpenCL platform that is provided by the processor's vendor
static int choose_ocl_platform(cl_platform_id &thePlatform)
{
       cl_platform_id platform_ids[4];
	   cl_int num_platforms;

       char platformName[256];
       char platformVersion[256];
	   cl_float fversion;

	   size_t valuesize, sizeret;

	   cl_int status;
	   const char *cpuVendor;

	   cpuVendor = getProcessorVendor();

	   status = clGetPlatformIDs(4, &platform_ids[0], (cl_uint*)&num_platforms);
	   if ( status != CL_SUCCESS )
		    return(-1);
	   else {
	        for (int i=0; i< num_platforms; i++ ) {

                 valuesize=sizeof(platformName);
                 if ( clGetPlatformInfo(platform_ids[i], CL_PLATFORM_NAME, valuesize, platformName, &sizeret) != CL_SUCCESS )
					   continue;
				 valuesize=sizeof(platformVersion);
				 if ( clGetPlatformInfo(platform_ids[i], CL_PLATFORM_VERSION, valuesize, platformVersion, &sizeret ) != CL_SUCCESS )
					   continue;

				 istringstream mystream(platformVersion);
				 string f1;

				 mystream >> f1 >> fversion;   // "OpenCL %f"

				 string splatform(platformName);

				 // For AMD processor, AMD OpenCL platform should be used
				 // For Intel processor, the Intel OpenCL platform should be used
				 // The supported OpenCL version of the platform should be >= 1.2
				 if ( (splatform.find(cpuVendor) != string::npos) && ( fversion >= 1.2f) ) {
					  thePlatform = platform_ids[i];
					  return(0);
				 }
			};
	   };
	   return(-2);   // reach here if no OpenCL platform for CPU device is installed
};

// get the OpenCL C version supported by the device
static float get_opencl_version(cl_device_id &theDevice)
{
	char inputBuf[256];
	float version;

    (void) clGetDeviceInfo(theDevice, CL_DEVICE_OPENCL_C_VERSION, 256, inputBuf, NULL);

	istringstream mystream(inputBuf);

	string f1, f2;

	mystream >> f1 >> f2 >> version;    //  "OpenCL C %f"

    return(version);
}

#ifdef _WIN32
static const char *getProcessorVendor()
{
    char buff[32];
	int CPUInfo[4] = {-1};

	__cpuid(CPUInfo, 0);

	memset(buff, 0, sizeof(buff));
	*((int*)buff) =  CPUInfo[1];
	*((int*)(buff+4)) = CPUInfo[3];
	*((int*)(buff+8)) = CPUInfo[2];

	string vendor(buff);

	if  ( vendor.find("Intel") != string::npos )
		  return("Intel");
	else
		if ( vendor.find("AMD") != string::npos )
		     return("AMD");
		else
			 return("Other");
};
#else
static  const char *getProcessorVendor()
{
	string vendor("vendor_id");
    ifstream infile;

    infile.open("/proc/cpuinfo",ios_base::in);

    if ( ! infile.is_open() ) {
         return("Failed");
    }
    else {
         while ( !infile.eof() ) {
			 string  myline;

		     getline(infile,myline);
             if (  myline.compare(0, vendor.length(), vendor) == 0 ) {
                     if ( myline.find("Intel") != string::npos )
                          return("Intel");
                     else
						 if ( myline.find("AMD") != string::npos )
                             return("AMD");
                         else
                             return("Other");
                     break;
             }
		 };
         infile.close();
         return("Failed");
    }
}
#endif

