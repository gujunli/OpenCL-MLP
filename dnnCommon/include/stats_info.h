#ifndef _STATS_INFO_H_
#define _STATS_INFO_H_

#include "DNNApiExport.h"

struct header_stats_file {
    char tag[4];                    // the tag of the states file, should be "STAT" 
    char dataset_name[20];          // the name the data set that produced the stats information (eg. MNIST, PTC English etc.) 
    unsigned int dimension;         // the dimension of the data vector, each dimension has its respective mean value and standard deviation
    unsigned int data_offset;       // the offset of the meanvalues and standard deviations in the file 
}; 

LIBDNNAPI extern void read_stats_info(const char *filePath, int sampleSize, float *meanvalues, float *stddevs); 
LIBDNNAPI extern void save_stats_info(const char *filePath, const char *datasetName, int sampleSize, float *meanvalues, float *stddevs);

#endif 

