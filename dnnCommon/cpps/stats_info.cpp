#include <fstream>
#include <cstring>

#include "DNNUtil.h"
#include "stats_info.h"
#include "conv_endian.h"

void read_stats_info(const char *filePath, int sampleSize, float *meanvalues, float *stddevs)
{
     ifstream statsFile;

     statsFile.open(filePath, ios_base::in|ios_base::binary);

     if ( ! statsFile.is_open() ) {
            dnn_log("STATS_INFO", "Failed to open the stats info file of the data set");
            DNN_Exception("");
     };

     struct header_stats_file header;
     int filelen;

     statsFile.seekg(16);      // 16 bytes reserved for checksum
     statsFile.read(reinterpret_cast<char*>(&header),sizeof(header));

     LEtoHostl(header.dimension);
     LEtoHostl(header.data_offset);

     if ( header.tag[0] != 'S' || header.tag[1] != 'T' || header.tag[2] != 'A' || header.tag[3] != 'T' ) {
          dnn_log("STATS_INFO", "Incorrect stats information file");
          DNN_Exception("");
     };

     statsFile.seekg(0,ios_base::end);
     filelen =  (int)statsFile.tellg();

     if ( filelen < (int) (header.data_offset + header.dimension * sizeof(float) * 2) ) {
          dnn_log("STATS_INFO", "The stats information file length is inconsistent with the size of data that should be recorded");
          DNN_Exception("");
     };

	 if ( (int) header.dimension != sampleSize ) {
          dnn_log("STATS_INFO", "The information from the stats information file length is inconsistent with that of the expected data sample size");
          DNN_Exception("");
	 };

     statsFile.seekg(header.data_offset);

     statsFile.read(reinterpret_cast<char *>(&meanvalues[0]), header.dimension*sizeof(float));
     statsFile.read(reinterpret_cast<char *>(&stddevs[0]), header.dimension*sizeof(float));

     for (int i=0; i< sampleSize; i++) {
          BytesToFloat(meanvalues[i]);
          BytesToFloat(stddevs[i]);
     };

     statsFile.close();
};

void save_stats_info(const char *filePath, const char *datasetName, int sampleSize, float *meanvalues, float *stddevs)
{
     struct header_stats_file header;
     ofstream statsFile;

     statsFile.open(filePath, ios_base::out|ios_base::binary|ios_base::trunc);

     if ( ! statsFile.is_open() ) {
            dnn_log("STATS_INFO", "Failed to create stats information file");
	        DNN_Exception("");
     };

     header.tag[0] = 'S';
     header.tag[1] = 'T';
     header.tag[2] = 'A';
     header.tag[3] = 'T';
     strncpy(&header.dataset_name[0], datasetName, 19);
     header.dimension = sampleSize;

     statsFile.seekp(16);            // first 16 bytes preserved for checksum-ing
     statsFile.write("FAIL", 4);     // 4-bytes used as the tag of the file
     statsFile.flush();

     header.data_offset = ((16 + sizeof(struct header_stats_file) + 1023)/1024 ) * 1024;    // round to 1024-byte boundary

     statsFile.seekp(header.data_offset);

     for (int i=0; i< sampleSize; i++)
          FloatToBytes(meanvalues[i]);
     for (int i=0; i< sampleSize; i++)
          FloatToBytes(stddevs[i]);

     statsFile.write(reinterpret_cast<char*>(&meanvalues[0]), sampleSize * sizeof(float));
     statsFile.write(reinterpret_cast<char*>(&stddevs[0]), sampleSize * sizeof(float));

     HostToLEl(header.dimension);
     HostToLEl(header.data_offset);

     statsFile.seekp(16);
     statsFile.write(reinterpret_cast<char*>(&header), sizeof(struct header_stats_file));
     statsFile.flush();

     statsFile.close();
};
