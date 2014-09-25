/*
 *  COPYRIGHT:  Copyright (c) 2014 Advanced Micro Devices, Inc.  All rights reserved
 *
 *   Written by Qianfeng Zhang@amd.com ( March 2014 )
 *
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <functional>
#include <vector>
#include <cstring>

#include "MLPUtil.h"
#include "MLPNetProvider.h"
#include "conv_endian.h"

using namespace std;

#define DEFAULT_LAYERS 8

static const int default_dimensions[DEFAULT_LAYERS] = {429,2048,2048,2048,2048,2048,2048,8991};

static const char *getNetTypeName(MLP_NETTYPE nettype);
static const char *getActFuncName(ACT_FUNC actFunc);
static const char *getCostFuncName(COST_FUNC costFunc);

static MLP_NETTYPE getNetTypeID(string &typeName);
static MLP_NETTYPE getNetTypeID(char *typeName);
static ACT_FUNC getActFuncID(string &funcName);
static ACT_FUNC getActFuncID(char *funcName);
static COST_FUNC getCostFuncID(string &funcName);

static void lowerCaselize(char *str);

MLPNetProvider::MLPNetProvider()
{
    this->netType = NETTYPE_MULTI_CLASSIFICATION;
    this->nLayers = DEFAULT_LAYERS;
    this->dimensions = new int[DEFAULT_LAYERS];
    this->biases = new float*[DEFAULT_LAYERS];
    this->weights = new float*[DEFAULT_LAYERS];
    this->etas = new float[DEFAULT_LAYERS];
    this->actFuncs = new ACT_FUNC[DEFAULT_LAYERS];

    for (int i=0; i< this->nLayers; i++)
        this->dimensions[i] = default_dimensions[i];

    this->biases[0] = NULL;
    for (int i=1; i< this->nLayers; i++)
        this->biases[i] = new float[this->dimensions[i]];

    this->weights[0] = NULL;
    for (int i=1; i< this->nLayers; i++)
        this->weights[i] = new float[this->dimensions[i-1]*this->dimensions[i]];

    this->weightsInitialize();
    this->biasesInitialize();
    this->etasInitialize();
    this->actFuncsInitialize();
    this->costFunc = CFUNC_SSE;
    this->momentum = 0.4f;
};


// this constructor should be used by the MLPTrainer, not the MLPTester and MLPPredictor
MLPNetProvider::MLPNetProvider(MLP_NETTYPE type, int layers, int dimensions_[], float etas_[], float momentum_, ACT_FUNC actFuncs_[], COST_FUNC costFunc_, bool DoInitialize)
{
    this->netType = type;
    this->nLayers = layers;
    this->dimensions = new int[this->nLayers];
    this->biases = new float*[this->nLayers];
    this->weights = new float*[this->nLayers];
    this->etas = new float[this->nLayers];
    this->actFuncs = new ACT_FUNC[this->nLayers];

    for (int i=0; i< this->nLayers; i++)
        this->dimensions[i] = dimensions_[i];

    this->etas[0] = 0.0f;
    for (int i=1; i< this->nLayers; i++)
        this->etas[i] = etas_[i];

    this->actFuncs[0] = ANOFUNC;
    for (int i=1; i< this->nLayers; i++)
        this->actFuncs[i] = actFuncs_[i];

    this->costFunc = costFunc_;
    this->momentum = momentum_;

    this->biases[0] = NULL;
    for (int i=1; i< this->nLayers; i++)
        this->biases[i] = new float[this->dimensions[i]];

    this->weights[0] = NULL;
    for (int i=1; i< this->nLayers; i++)
        this->weights[i] = new float[this->dimensions[i-1]*this->dimensions[i]];

    if ( DoInitialize ) {
        this->weightsInitialize();
        this->biasesInitialize();
    };
};


MLPNetProvider::MLPNetProvider(int layers, int dimensions_[], bool DoInitialize)
{
    this->nLayers = layers;
    this->dimensions = new int[this->nLayers];
    this->biases = new float*[this->nLayers];
    this->weights = new float*[this->nLayers];
    this->etas = new float[this->nLayers];
    this->actFuncs = new ACT_FUNC[this->nLayers];

    for (int i=0; i< this->nLayers; i++)
        this->dimensions[i] = dimensions_[i];

    this->biases[0] = NULL;
    for (int i=1; i< this->nLayers; i++)
        this->biases[i] = new float[this->dimensions[i]];

    this->weights[0] = NULL;
    for (int i=1; i< this->nLayers; i++)
        this->weights[i] = new float[this->dimensions[i-1]*this->dimensions[i]];

    if ( DoInitialize ) {
        this->netType = NETTYPE_MULTI_CLASSIFICATION;
        this->weightsInitialize();
        this->biasesInitialize();
        this->etasInitialize();
        this->actFuncsInitialize();
        this->costFunc = CFUNC_SSE;
        this->momentum = 0.4f;
    };
};


// remove the spaces from the front of the line ( provided by Zhitao Zhou@amd.com)
static void ltrim(string &str)
{
    str.erase(str.begin(), ::find_if(str.begin(),str.end(),std::not1(std::ptr_fun(::isspace))));
};


// this constructor should be used by the MLPTrainer, not the MLPTester and MLPPredictor
// the MLPTrainer will initialize its neural network parameters from the exiting config file and nnet data file
MLPNetProvider::MLPNetProvider(const char *dir, const char *trainingConfigFile, const char *nnetDataFile)
{
	string configFileName(dir);
	string nnetFileName(dir);

	configFileName.append(trainingConfigFile);
	nnetFileName.append(nnetDataFile);

	ifstream configFile;
	ifstream nnetFile;

	configFile.open(configFileName.c_str(),ios_base::in);
	nnetFile.open(nnetFileName.c_str(),ios_base::in|ios_base::binary);

	if ( ! configFile.is_open() || ! nnetFile.is_open() ) {
		   mlp_log("MLPNetProvider", "Failed to open MLP net config files for reading");
		   MLP_Exception("");
	};


	vector<string> lines;

    while ( ! configFile.eof() ) {
          string myline;

          getline(configFile,myline);
          ltrim(myline);

          if ( !myline.empty() && !(myline[0] == '#') )
			   lines.push_back(myline);

    };

    // read Network Type name
    string typeName1, typeName2;
    string key0("Network Type:");

	for (vector<string>::iterator it=lines.begin(); it != lines.end(); ++it) {
		  if ( (*it).compare(0,key0.length(),key0) == 0 ) {
                istringstream mystream((*it).substr(key0.length()));

                mystream >> typeName1 >> typeName2;
                transform(typeName1.begin(),typeName1.end(),typeName1.begin(),::tolower);      // porvided by ZhiTao Zhou
                transform(typeName2.begin(),typeName2.end(),typeName2.begin(),::tolower);      // porvided by ZhiTao Zhou
				typeName1 = typeName1 + " " + typeName2;
                break;
          };
    };
    MLP_NETTYPE typeID=getNetTypeID(typeName1);
	if ( typeID == NETNOTYPE ) {
		  mlp_log("MLPNetProvider", "The setting for <Network Type> from the config file is not correct");
		  MLP_Exception("");
	};
	this->netType = typeID;

    // read Layers information
    int layers=0;
    string key1("Layers:");
    for (vector<string>::iterator it=lines.begin(); it != lines.end(); ++it) {
          if ( (*it).compare(0,key1.length(),key1) == 0 ) {
                istringstream mystream((*it).substr(key1.length()));

                mystream >> layers;
                break;
          };
    };
	if ( (layers < 2) || (layers > 9 )) {
		  mlp_log("MLPNetProvider", "The setting for <Layers> from the config file is not correct");
		  MLP_Exception("");
	};
	this->nLayers = layers;
	this->dimensions = new int[this->nLayers];
	this->biases = new float*[this->nLayers];
	this->weights = new float*[this->nLayers];
	this->etas = new float[this->nLayers];
	this->actFuncs = new ACT_FUNC[this->nLayers];

    // read cost function name
    string funName;
    string key2("Cost Function:");
    for (vector<string>::iterator it=lines.begin(); it != lines.end(); ++it) {
		  if ( (*it).compare(0,key2.length(),key2) == 0 ) {
                istringstream mystream((*it).substr(key2.length()));

                mystream >> funName;
                transform(funName.begin(),funName.end(),funName.begin(),::toupper);      // porvided by ZhiTao Zhou
                break;
          };
    };
    COST_FUNC cfunID=getCostFuncID(funName);
	if ( cfunID == CNOFUNC ) {
		  mlp_log("MLPNetProvider", "The setting for <Cost Function> from the config file is not correct");
		  MLP_Exception("");
	};

	this->costFunc = cfunID;

	// read the parameters for the Input layer
	int layer0_dim=0;
    string key3("Layer 0:");
    for (vector<string>::iterator it=lines.begin(); it != lines.end(); ++it) {
          if ( (*it).compare(0,key3.length(),key3) == 0 ) {
                istringstream mystream((*it).substr(key3.length()));

                mystream >> layer0_dim;
                break;
          };
    };
    if ( layer0_dim < 2  ) {
		  mlp_log("MLPNetProvider", "The setting for <Layer 0> from the config file is not correct");
		  MLP_Exception("");
	};
	this->dimensions[0] = layer0_dim;

	// read parameters for the Hidden layers and the Output layer
	int *layers_dim = new int[layers-1];
    float *layers_eta = new float[layers-1];
    string *funNames = new string[layers-1];
    for (int k=0; k< layers-1; k++) {
          ostringstream keystream;
          keystream << "Layer " << k+1 << ":";

		  bool found=false;
          for (vector<string>::iterator it=lines.begin(); it != lines.end(); ++it) {
               if ( (*it).compare(0, keystream.str().length(),keystream.str()) == 0 ) {
                    istringstream mystream((*it).substr(keystream.str().length()));

                    mystream >> layers_dim[k] >> layers_eta[k] >> funNames[k];
                    transform(funNames[k].begin(),funNames[k].end(),funNames[k].begin(),::tolower);  // porvided by ZhiTao Zhou

					ACT_FUNC afunID=getActFuncID(funNames[k]);
					if ( (layers_dim[k] < 2) || (layers_eta[k] <= 0.0f) || (layers_eta[k] >= 1.0f) || (afunID == ANOFUNC) ) {
						  ostringstream errBuf;
						  errBuf << "The setting for layer " << k+1 << " is not correct";
		                  mlp_log("MLPNetProvider", errBuf.str().c_str());
		                  MLP_Exception("");
					};
                    found = true;
					this->dimensions[k+1] = layers_dim[k];
					this->etas[k+1] = layers_eta[k];
					this->actFuncs[k+1] = afunID;
                    break;
               };
          };
		  if ( !found ) {
			   ostringstream errBuf;

			   errBuf << "The setting for layer " << k+1 << " is not found" ;
		       mlp_log("MLPNetProvider", errBuf.str().c_str());
		       MLP_Exception("");
		  };
    };

	delete [] layers_dim;
    delete [] layers_eta;
    delete [] funNames;

	// read momentum value
	float mmValue;
	string key4("Momentum:");
    for (vector<string>::iterator it=lines.begin(); it != lines.end(); ++it) {
          if ( (*it).compare(0,key4.length(),key4) == 0 ) {
                istringstream mystream((*it).substr(key4.length()));

                mystream >> mmValue;
                break;
          };
    };
    if ( mmValue < 0.0f || mmValue >= 1.0f  ) {
		  mlp_log("MLPNetProvider", "The setting for <Momentum> from the config file may be not reasonable");
		  MLP_Exception("");
	};
	this->momentum = mmValue;

	// allocate memory for weights and biases data of each layer
	this->biases[0] = NULL;
	for (int i=1; i< this->nLayers; i++)
		this->biases[i] = new float[this->dimensions[i]];

	this->weights[0] = NULL;
	for (int i=1; i< this->nLayers; i++)
		this->weights[i] = new float[this->dimensions[i-1]*this->dimensions[i]];


    // read information from the neural network data file
	nnetFile.seekg(16);   // skip the checksum

	string corrMark("NNET");
	char Mark[5];

	nnetFile.read(&Mark[0], 4);
	Mark[4] = '\0';

	if ( corrMark != Mark ) {
		 mlp_log("MLPNetProvider", "checking the integrity of MLP neural network data file failed, discarded");
		 MLP_Exception("");
	};

	nnetFile.seekg(16+4);  // skip the checksum and the tag

	struct mlp_nnet_data_header header;

	nnetFile.read(reinterpret_cast<char*>(&header), sizeof(header));

	// convert the data in the header to host bytes sequence from Little Endian bytes sequence
	LEtoHostl(header.nLayers);
	for (int i=0; i < this->nLayers; i++)
	     LEtoHostl(header.layers[i].dimension);
	for (int i=1; i < this->nLayers; i++)
	     LEtoHostl(header.weight_offsets[i]);

    lowerCaselize(header.nnet_type);

    // check whether the information from the config file and the nnet data matches
    bool match=true;

    match = ( this->nLayers == (int)header.nLayers )? match:false;
    match = ( this->netType == getNetTypeID(header.nnet_type) )? match:false;
    if ( match ) {
         for (int i=0; i < this->nLayers; i++)
              match = ( this->dimensions[i] == (int)header.layers[i].dimension )? match:false;

         for (int i=1; i < this->nLayers; i++)
              match = ( this->actFuncs[i] == getActFuncID(header.layers[i].activation) )? match:false;
    };

    if ( ! match ) {
         mlp_log("MLPNetProvider", "The infomration from the config file and the neural network data file does not match");
         MLP_Exception("");
    };

	for (int i=1; i < this->nLayers; i++) {

	    nnetFile.seekg(header.weight_offsets[i]);

		for (int row=0; row < this->dimensions[i-1]; row++)
			 nnetFile.read(reinterpret_cast<char*>(&this->weights[i][row*this->dimensions[i]]), sizeof(float)*this->dimensions[i]);
		nnetFile.read(reinterpret_cast<char*>(&this->biases[i][0]),sizeof(float)*this->dimensions[i]);

		// convert to host float type from generice bytes
		for (int row=0; row < this->dimensions[i-1]; row++)
			 for (int col=0; col < this->dimensions[i]; col++)
                      BytesToFloat(this->weights[i][row*this->dimensions[i]+col]);
				      // LEtoHostl(*(unsigned int *)&this->weights[i][row*this->dimensions[i]+col]);

		for (int col=0; col < this->dimensions[i]; col++)
                 BytesToFloat(this->biases[i][col]);
			     //LEtoHostl(*(unsigned int *)&this->biases[i][col]);
	};

    configFile.close();
    nnetFile.close();
}

// this constuctor is used by the trainer to create a randomly initialized neural network
 MLPNetProvider::MLPNetProvider(const char *dir, const char *trainingConfigFile, bool DoInitialize)
{
	string configFileName(dir);
	string nnetFileName(dir);

	configFileName.append(trainingConfigFile);

	ifstream configFile;

	configFile.open(configFileName.c_str(),ios_base::in);

	if ( ! configFile.is_open() ) {
		   mlp_log("MLPNetProvider", "Failed to open MLP training configuration for reading");
		   MLP_Exception("");
	};

	vector<string> lines;

    while ( ! configFile.eof() ) {
          string myline;

          getline(configFile,myline);
          ltrim(myline);

          if ( !myline.empty() && !(myline[0] == '#') )
			   lines.push_back(myline);

    };

    // read Network Type name
    string typeName1, typeName2;
    string key0("Network Type:");

	for (vector<string>::iterator it=lines.begin(); it != lines.end(); ++it) {
		  if ( (*it).compare(0,key0.length(),key0) == 0 ) {
                istringstream mystream((*it).substr(key0.length()));

                mystream >> typeName1 >> typeName2;
                transform(typeName1.begin(),typeName1.end(),typeName1.begin(),::tolower);      // porvided by ZhiTao Zhou
                transform(typeName2.begin(),typeName2.end(),typeName2.begin(),::tolower);      // porvided by ZhiTao Zhou
				typeName1 = typeName1 + " " + typeName2;
                break;
          };
    };
    MLP_NETTYPE typeID=getNetTypeID(typeName1);
	if ( typeID == NETNOTYPE ) {
		  mlp_log("MLPNetProvider", "The setting for <Network Type> from the config file is not correct");
		  MLP_Exception("");
	};
	this->netType = typeID;

    // read Layers information
    int layers=0;
    string key1("Layers:");
    for (vector<string>::iterator it=lines.begin(); it != lines.end(); ++it) {
          if ( (*it).compare(0,key1.length(),key1) == 0 ) {
                istringstream mystream((*it).substr(key1.length()));

                mystream >> layers;
                break;
          };
    };
	if ( (layers < 2) || (layers > 9 )) {
		  mlp_log("MLPNetProvider", "The setting for <Layers> from the config file is not correct");
		  MLP_Exception("");
	};
	this->nLayers = layers;
	this->dimensions = new int[this->nLayers];
	this->biases = new float*[this->nLayers];
	this->weights = new float*[this->nLayers];
	this->etas = new float[this->nLayers];
	this->actFuncs = new ACT_FUNC[this->nLayers];

    // read cost function name
    string funName;
    string key2("Cost Function:");
    for (vector<string>::iterator it=lines.begin(); it != lines.end(); ++it) {
		  if ( (*it).compare(0,key2.length(),key2) == 0 ) {
                istringstream mystream((*it).substr(key2.length()));

                mystream >> funName;
                transform(funName.begin(),funName.end(),funName.begin(),::toupper);      // porvided by ZhiTao Zhou
                break;
          };
    };
    COST_FUNC cfunID=getCostFuncID(funName);
	if ( cfunID == CNOFUNC ) {
		  mlp_log("MLPNetProvider", "The setting for <Cost Function> from the config file is not correct");
		  MLP_Exception("");
	};

	this->costFunc = cfunID;

	// read the parameters for the Input layer
	int layer0_dim=0;
    string key3("Layer 0:");
    for (vector<string>::iterator it=lines.begin(); it != lines.end(); ++it) {
          if ( (*it).compare(0,key3.length(),key3) == 0 ) {
                istringstream mystream((*it).substr(key3.length()));

                mystream >> layer0_dim;
                break;
          };
    };
    if ( layer0_dim < 2  ) {
		  mlp_log("MLPNetProvider", "The setting for <Layer 0> from the config file is not correct");
		  MLP_Exception("");
	};
	this->dimensions[0] = layer0_dim;

	// read parameters for the Hidden layers and the Output layer
	int *layers_dim = new int[layers-1];
    float *layers_eta = new float[layers-1];
    string *funNames = new string[layers-1];
    for (int k=0; k< layers-1; k++) {
          ostringstream keystream;
          keystream << "Layer " << k+1 << ":";

		  bool found=false;
          for (vector<string>::iterator it=lines.begin(); it != lines.end(); ++it) {
               if ( (*it).compare(0, keystream.str().length(),keystream.str()) == 0 ) {
                    istringstream mystream((*it).substr(keystream.str().length()));

                    mystream >> layers_dim[k] >> layers_eta[k] >> funNames[k];
                    transform(funNames[k].begin(),funNames[k].end(),funNames[k].begin(),::tolower);  // porvided by ZhiTao Zhou

					ACT_FUNC afunID=getActFuncID(funNames[k]);
					if ( (layers_dim[k] < 2) || (layers_eta[k] <= 0.0f) || (layers_eta[k] >= 1.0f) || (afunID == ANOFUNC) ) {
						  ostringstream errBuf;
						  errBuf << "The setting for layer " << k+1 << " is not correct";
		                  mlp_log("MLPNetProvider", errBuf.str().c_str());
		                  MLP_Exception("");
					};
                    found = true;
					this->dimensions[k+1] = layers_dim[k];
					this->etas[k+1] = layers_eta[k];
					this->actFuncs[k+1] = afunID;
                    break;
               };
          };
		  if ( !found ) {
			   ostringstream errBuf;

			   errBuf << "The setting for layer " << k+1 << " is not found" ;
		       mlp_log("MLPNetProvider", errBuf.str().c_str());
		       MLP_Exception("");
		  };
    };

	delete [] layers_dim;
    delete [] layers_eta;
    delete [] funNames;

	// read momentum value
	float mmValue;
	string key4("Momentum:");
    for (vector<string>::iterator it=lines.begin(); it != lines.end(); ++it) {
          if ( (*it).compare(0,key4.length(),key4) == 0 ) {
                istringstream mystream((*it).substr(key4.length()));

                mystream >> mmValue;
                break;
          };
    };
    if ( mmValue < 0.0f || mmValue >= 1.0f  ) {
		  mlp_log("MLPNetProvider", "The setting for <Momentum> from the config file may be not reasonable");
		  MLP_Exception("");
	};
	this->momentum = mmValue;

	// allocate memory for weights and biases data of each layer
	this->biases[0] = NULL;
	for (int i=1; i< this->nLayers; i++)
		this->biases[i] = new float[this->dimensions[i]];

	this->weights[0] = NULL;
	for (int i=1; i< this->nLayers; i++)
		this->weights[i] = new float[this->dimensions[i-1]*this->dimensions[i]];

    configFile.close();

    if ( DoInitialize ) {
        this->weightsInitialize();
        this->biasesInitialize();
    };
};



// this constructor should be used by the MLPTester and MLPPredictor, not the MLPTrainer
MLPNetProvider::MLPNetProvider(const char *dir, const char *nnetDataFile)
{
    string nnetFileName(dir);

    nnetFileName.append(nnetDataFile);

    ifstream nnetFile;

    nnetFile.open(nnetFileName.c_str(),ios_base::in|ios_base::binary);

    if ( ! nnetFile.is_open() )
    {
        mlp_log("MLPNetProvider", "Failed to open MLP neural network data file for reading");
        MLP_Exception("");
    };

    nnetFile.seekg(16);   // skip the checksum

    string corrMark("NNET");
    char Mark[5];

    nnetFile.read(&Mark[0], 4);
    Mark[4] = '\0';

    if ( corrMark != Mark )
    {
        mlp_log("MLPNetProvider", "checking the integrity of MLP neural network data file failed, discarded");
        MLP_Exception("");
    };

    nnetFile.seekg(16+4);  // skip the checksum and the tag

    struct mlp_nnet_data_header header;

    nnetFile.read(reinterpret_cast<char*>(&header), sizeof(header));

    // convert the data in the header to host bytes sequence from Little Endian bytes sequence
    LEtoHostl(header.nLayers);
    for (int i=0; i < this->nLayers; i++)
        LEtoHostl(header.layers[i].dimension);
    for (int i=1; i < this->nLayers; i++)
        LEtoHostl(header.weight_offsets[i]);

    lowerCaselize(header.nnet_type);

    // allocate data structures for the MLPNetProvider
    this->nLayers = header.nLayers;
    this->netType = getNetTypeID(header.nnet_type);
    this->dimensions = new int[this->nLayers];
    this->biases = new float*[this->nLayers];
    this->weights = new float*[this->nLayers];
    this->etas = new float[this->nLayers];
    this->actFuncs = new ACT_FUNC[this->nLayers];

    // absorb the information directly from the nnet data file header
    for (int i=0; i< this->nLayers; i++)
        this->dimensions[i] = header.layers[i].dimension;
    for (int i=1; i< this->nLayers; i++)
        this->actFuncs[i] = getActFuncID(header.layers[i].activation);

    // allocate memory for weights and biases data of each layer
    this->biases[0] = NULL;
    for (int i=1; i< this->nLayers; i++)
        this->biases[i] = new float[this->dimensions[i]];

    this->weights[0] = NULL;
    for (int i=1; i< this->nLayers; i++)
        this->weights[i] = new float[this->dimensions[i-1]*this->dimensions[i]];

    // read the weights and biases information for all layers from the nnet data file
    for (int i=1; i < this->nLayers; i++)
    {

        nnetFile.seekg(header.weight_offsets[i]);

        for (int row=0; row < this->dimensions[i-1]; row++)
            nnetFile.read(reinterpret_cast<char*>(&this->weights[i][row*this->dimensions[i]]), sizeof(float)*this->dimensions[i]);
        nnetFile.read(reinterpret_cast<char*>(&this->biases[i][0]),sizeof(float)*this->dimensions[i]);

        // convert to host float type from generic bytes
        for (int row=0; row < this->dimensions[i-1]; row++)
            for (int col=0; col < this->dimensions[i]; col++)
                     BytesToFloat(this->weights[i][row*this->dimensions[i]+col]);
		             // LEtoHostl(*(unsigned int *)&this->weights[i][row*this->dimensions[i]+col]);

        for (int col=0; col < this->dimensions[i]; col++)
                     BytesToFloat(this->biases[i][col]);
					// LEtoHostl(*(unsigned int *)&this->biases[i][col]);
    };

    nnetFile.close();
};


MLPNetProvider::~MLPNetProvider()
{
    for (int i=0; i< this->nLayers; i++)
        if ( this->biases[i] )
            delete [] this->biases[i];

    for (int i=0; i < this->nLayers; i++)
        if ( this->weights[i] )
            delete [] this->weights[i];

    delete [] this->biases;
    delete [] this->weights;
    delete [] this->dimensions;
    delete [] this->etas;
    delete [] this->actFuncs;
};

void MLPNetProvider::biasesInitialize()
{
    float *datap;

    for (int i=1; i< this->nLayers; i++)
    {
        datap = this->biases[i];
        for (int k=0; k< this->dimensions[i]; k++)
        {
            *datap = 0.0f;
            datap++;
        };
    };
};

void MLPNetProvider::weightsInitialize()
{
    float *datap;

    struct dnn_tv tv;

    getCurrentTime(&tv);
    srand(tv.tv_usec); // use current time as random seed

    for (int i=1; i< this->nLayers; i++)
        for (int col=0; col< this->dimensions[i]; col++)              // go through all columns
        {
            datap = this->weights[i] + col;
            for (int row=0; row< this->dimensions[i-1]; row++)       // go through all lines
            {
                *datap = (float)rand()/((float)RAND_MAX+1.0f)-0.5f;
                datap += this->dimensions[i];
            };
        };
};

void MLPNetProvider::etasInitialize()
{
    this->etas[0] = 0;
    for (int i=1; i< this->nLayers; i++)
        this->etas[i] = 0.0002f;
};

void MLPNetProvider::actFuncsInitialize()
{
    this->actFuncs[0] = ANOFUNC;
    for (int i=1; i< this->nLayers; i++)
        this->actFuncs[i] = AFUNC_SIGMOID;
};

int MLPNetProvider::getInputLayerSize()
{
	return(this->dimensions[0]);
};

int MLPNetProvider::getOutputLayerSize()
{
	return(this->dimensions[this->nLayers-1]);
};


void MLPNetProvider::saveConfig(const char *dir, const char *trainingConfigFile, const char *nnetDataFile)
{
    string configFileName(dir);
    string nnetFileName(dir);

    configFileName.append(trainingConfigFile);
    nnetFileName.append(nnetDataFile);

    ofstream configFile;
    ofstream nnetFile;

    configFile.open(configFileName.c_str(),ios_base::out|ios_base::trunc);
    nnetFile.open(nnetFileName.c_str(),ios_base::out|ios_base::binary|ios_base::trunc);

    if ( ! configFile.is_open() || ! nnetFile.is_open() )
    {
        mlp_log("MLPNetProvider", "Failed to create MLP net config files");
        MLP_Exception("");
    };

    configFile.setf(ios_base::showpoint|ios_base::dec|ios_base::fixed);

    configFile << "### Configuration file for the neural network training, produced automatically "  << endl;
    configFile << "### You can do modification for your need "  << endl;

    configFile << endl << "Network Type: " << getNetTypeName(this->netType) << endl;
    configFile << endl << "Layers: " << this->nLayers << endl;
    configFile << endl << "Cost Function: " << getCostFuncName(this->costFunc) << endl;
    configFile.precision(2);
    configFile << endl << "Momentum: " << this->momentum << endl;
    configFile.precision(5);
    configFile << endl << "Layer 0: " << this->dimensions[0] << endl;
    for (int i=1; i < this->nLayers; i++ )
        configFile << "Layer " << i << ": " << this->dimensions[i] << " " << this->etas[i] << " " << getActFuncName(this->actFuncs[i]) << endl;

    configFile << endl;

    configFile.flush();

    // Save static network parameters into an binary file

    nnetFile.seekp(16);           // first 16 bytes preserved for checksum-ing
    nnetFile.write("FAIL", 4);    // 4-bytes used as the tag of the file
    nnetFile.flush();

    struct mlp_nnet_data_header header;

    // set up the header of the neural network data from the information in the MLPNetProvider
    header.nLayers = this->nLayers;
    strcpy(header.nnet_type, getNetTypeName(this->netType));
    for (int i=0; i < this->nLayers; i++)
        header.layers[i].dimension = this->dimensions[i];

    for (int i=1; i < this->nLayers; i++)
        strcpy(header.layers[i].activation, getActFuncName(this->actFuncs[i]));

    header.weight_offsets[1] = ( (16 + 4 + sizeof(header) + 1023 ) / 1024 ) * 1024;    // round to 1024n
    for (int i=2; i < this->nLayers; i++)
    {
        int bytes;

        bytes = this->dimensions[i-2]*this->dimensions[i-1]*sizeof(float) + this->dimensions[i-1]*sizeof(float);
        header.weight_offsets[i] = ( (header.weight_offsets[i-1] + bytes + 1023 ) / 1024 ) * 1024;     // round to 1024n
    };


    for (int i=1; i < this->nLayers; i++)
    {
        // go to the location for writing the weights and biases for this layer
        nnetFile.seekp(header.weight_offsets[i]);

        // convert to generice bytes from host float type
        for (int row=0; row < this->dimensions[i-1]; row++)
            for (int col=0; col < this->dimensions[i]; col++)
                 FloatToBytes(this->weights[i][row*this->dimensions[i]+col]);
                 // HostToLEl(*(unsigned int *)&this->weights[i][row*this->dimensions[i]+col]);

        for (int col=0; col < this->dimensions[i]; col++)
             FloatToBytes(this->biases[i][col]);
             // HostToLEl(*(unsigned int *)&this->biases[i][col]);

        // write the converted data to file
        for (int row=0; row < this->dimensions[i-1]; row++)
            nnetFile.write(reinterpret_cast<char*>(&this->weights[i][row*this->dimensions[i]]), sizeof(float)*this->dimensions[i]);

        nnetFile.write(reinterpret_cast<char*>(&this->biases[i][0]),sizeof(float)*this->dimensions[i]);

        // convert back to host float type from generic bytes 
        for (int row=0; row < this->dimensions[i-1]; row++)
            for (int col=0; col < this->dimensions[i]; col++)
                     BytesToFloat(this->weights[i][row*this->dimensions[i]+col]);
                     // LEtoHostl(*(unsigned int *)&this->weights[i][row*this->dimensions[i]+col]);

        for (int col=0; col < this->dimensions[i]; col++)
                 BytesToFloat(this->biases[i][col]);
                 // LEtoHostl(*(unsigned int *)&this->biases[i][col]);
    };

    // convert the data in the header to Little Endian bytes sequence from host bytes sequence
    HostToLEl(header.nLayers);
    for (int i=0; i < this->nLayers; i++)
        HostToLEl(header.layers[i].dimension);
    for (int i=1; i < this->nLayers; i++)
        HostToLEl(header.weight_offsets[i]);

    nnetFile.seekp(16+4);

    nnetFile.write(reinterpret_cast<char*>(&header), sizeof(header));

    nnetFile.seekp(16);            // skip the checksum (TODO)
    nnetFile.write("NNET", 4);     // write the tag of the file
    nnetFile.flush();

    configFile.close();
    nnetFile.close();
}



void MLPNetProvider::showConfig()
{
    cout << "Network Type: " << getNetTypeName(this->netType) << endl;
    cout << "Layers: " << this->nLayers << endl;
    cout << "Cost Function: " << getCostFuncName(this->costFunc) << endl;
    cout.precision(2);
    cout << "Momentum: " << this->momentum << endl;
    cout.precision(5);
    cout << "Layer 0: " << this->dimensions[0] << endl;
    for (int i=1; i < this->nLayers; i++ )
        cout << "Layer " << i << ": " << this->dimensions[i] << " " << this->etas[i] << " " << getActFuncName(this->actFuncs[i]) << endl;

    int myprec;

    myprec = (int)cout.precision();
    cout.precision(6);
    cout.setf(ios_base::showpoint|ios_base::dec|ios_base::fixed);

    cout << endl;
    for (int i=1; i < this->nLayers; i++)
    {
        cout << "Weights connecting layer " << i-1 << " to " << i << endl;
        for (int row=0; row < this->dimensions[i-1]; row++)
        {
            for (int col=0; col < this->dimensions[i]; col++)
                cout << this->weights[i][row*this->dimensions[i]+col] << " ";
            cout << endl;
        };
        cout << "Biases of layer " << i << endl;
        for (int col=0; col < this->dimensions[i]; col++)
            cout << this->biases[i][col]<< " ";
        cout << endl << endl;
    };

    cout << endl;

    cout.precision(myprec);
};


static const char *getNetTypeName(MLP_NETTYPE nettype)
{
    switch (nettype)
    {
    case NETTYPE_BIN_CLASSIFICATION:
        return("Binary Classification");
    case NETTYPE_MULTI_CLASSIFICATION:
        return("Multiple Classification");
    case NETTYPE_LINEAR_REGRESSION:
        return("Linear Regression");
    default:
        return("");
    };
};


static const char *getActFuncName(ACT_FUNC actFunc)
{
    switch (actFunc)
    {
    case AFUNC_SIGMOID:
        return("sigmoid");
    case AFUNC_SOFTMAX:
        return("softmax");
    case AFUNC_IDENTITY:
        return("identity");
    case AFUNC_RELU:
        return("relu");
    case AFUNC_TANH:
        return("tanh");
    default:
        return("");
    };
};

static const char *getCostFuncName(COST_FUNC costFunc)
{
    switch (costFunc)
    {
    case CFUNC_SSE:
        return("SSE");
    case CFUNC_CE:
        return("CE");
    default:
        return("");
    };
};

// the typeName should already be in lower case
static MLP_NETTYPE getNetTypeID(string &typeName)
{
    if ( typeName == "binary classification" )
        return(NETTYPE_BIN_CLASSIFICATION);
    if ( typeName == "multiple classification" )
        return(NETTYPE_MULTI_CLASSIFICATION);
    if ( typeName == "linear regression" )
        return(NETTYPE_LINEAR_REGRESSION);

    return(NETNOTYPE);
};

// the typeName should already be in lower case
static MLP_NETTYPE getNetTypeID(char *typeName)
{
    string sTypeName(typeName);

    if ( sTypeName == "binary classification" )
        return(NETTYPE_BIN_CLASSIFICATION);
    if ( sTypeName == "multiple classification" )
        return(NETTYPE_MULTI_CLASSIFICATION);
    if ( sTypeName == "linear regression" )
        return(NETTYPE_LINEAR_REGRESSION);

    return(NETNOTYPE);
};


// the funcName should already be in lower case
static ACT_FUNC getActFuncID(string &funcName)
{
    if ( funcName == "sigmoid" )
        return(AFUNC_SIGMOID);
    if ( funcName == "softmax" )
        return(AFUNC_SOFTMAX);
    if ( funcName == "linear" )
        return(AFUNC_IDENTITY);
    if ( funcName == "relu" )
        return(AFUNC_RELU);
    if ( funcName == "tanh" )
        return(AFUNC_TANH);

    return(ANOFUNC);
};

// the funcName should already be in lower case
static ACT_FUNC getActFuncID(char *funcName)
{
    string sFuncName(funcName);

    if ( sFuncName == "sigmoid" )
        return(AFUNC_SIGMOID);
    if ( sFuncName == "softmax" )
        return(AFUNC_SOFTMAX);
    if ( sFuncName == "linear" )
        return(AFUNC_IDENTITY);
    if ( sFuncName == "relu" )
        return(AFUNC_RELU);
    if ( sFuncName == "tanh" )
        return(AFUNC_TANH);

    return(ANOFUNC);
};


static COST_FUNC getCostFuncID(string &funcName)
{
    if ( funcName == "SSE" )
        return(CFUNC_SSE);
    if ( funcName == "CE" )
        return(CFUNC_CE);

    return(CNOFUNC);
};

static void lowerCaselize(char *str)
{
    if ( str ) {
         for (int i=0; i< (int)strlen(str); i++)
              if ( isupper(str[i]) )
                   str[i] = (char)tolower(str[i]);
    }
}
