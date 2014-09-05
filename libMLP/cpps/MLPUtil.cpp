/*
 *  COPYRIGHT:  Copyright (c) 2014 Advanced Micro Devices, Inc.  All rights reserved
 *
 *   Written by Qianfeng Zhang@amd.com ( March 2014 )
 *
 */

#ifdef _WIN32
#include <time.h>
#include <Windows.h>
#else
#include <sys/time.h>
#include <unistd.h>
#endif

#include "MLPUtil.h"

#include <iostream>
#include <fstream>

#ifdef _WIN32
#define  DELTA_EPOCH_IN_MICROSECS  11644473600000000ULL
void getCurrentTime(struct mlp_tv *tv)
{
     FILETIME ft; 
	 unsigned __int64 tmpres=0; 

	 if ( tv != NULL ) {
		 GetSystemTimeAsFileTime(&ft); 
		 
		 tmpres |= ft.dwHighDateTime;
		 tmpres <<=32; 
		 tmpres |= ft.dwLowDateTime; 

		 tmpres -= DELTA_EPOCH_IN_MICROSECS; 
		 tmpres /= 10; 
		 tv->tv_sec = (long) (tmpres / 1000000UL);
		 tv->tv_usec = (long) (tmpres % 1000000UL); 
	 }

}
#else
void getCurrentTime(struct mlp_tv *tv)
{
	struct timeval tmptv; 
    if ( tv != NULL ) {
		 gettimeofday(&tmptv, NULL);
		 tv->tv_sec = tmptv.tv_sec;
		 tv->tv_usec = tmptv.tv_usec; 
	}
}
#endif

long diff_msec(struct mlp_tv *stv, struct mlp_tv *etv)
{
    if ( (stv != NULL) && (etv != NULL) ) {
		return( (etv->tv_sec - stv->tv_sec)*1000 + (etv->tv_usec - stv->tv_usec)/1000 ); 
	}
	else
		return(0); 
}

long diff_usec(struct mlp_tv *stv, struct mlp_tv *etv)
{
    if ( (stv != NULL) && (etv != NULL) ) {
		return( (etv->tv_sec - stv->tv_sec)*1000000 + (etv->tv_usec - stv->tv_usec) ); 
	}
	else
		return(0); 
}


int read_srcfile(const char *filename, char * &src_str)
{
	 ifstream fsrc; 
     int filelen;
     char *src_data;  // char string to hold kernel source

  	 // read the kernel file 
	 fsrc.open(filename, ios_base::in |ios_base::binary); 
	 if ( !fsrc.is_open() ) {
		  return(-1); 
	 };
	 fsrc.seekg(0,ios_base::end); 
	 filelen = (int)fsrc.tellg(); 
     fsrc.seekg(0); 

	 src_data = new char[filelen+1];
     fsrc.read(src_data, filelen);     
	 fsrc.close(); 
	
	 // ensure the string is NULL terminated
	 src_data[filelen]='\0';
	 src_str = src_data; 

	 return(0); 
}; 

#ifdef _WIN32             // Windows
int getLogicCoreNum()
{
      SYSTEM_INFO info; 

	  GetSystemInfo(&info); 

	  return(info.dwNumberOfProcessors); 
}
#else                   // Linux
int getLogicCoreNum()
{
    return(sysconf(_SC_NPROCESSORS_ONLN)); 
}
#endif

ofstream nBenchLog; 

void mlp_log(const char *header, const char *content)
{
	if ( ! nBenchLog.is_open() ) 
		   nBenchLog.open("mlp_log.txt",ios_base::out|ios_base::trunc);

    nBenchLog << header << ":" << content << endl; 
}

void mlp_log_retval(const char *header, int retVal)
{
	if ( ! nBenchLog.is_open() ) 
		   nBenchLog.open("mlp_log.txt",ios_base::out|ios_base::trunc);

    nBenchLog << header << ":" << retVal << endl; 
}; 

