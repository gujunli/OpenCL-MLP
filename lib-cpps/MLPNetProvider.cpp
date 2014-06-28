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

#include "MLPUtil.h"
#include "MLPNetProvider.h"

using namespace std;

#define DEFAULT_LAYERS 8

static const int default_dimensions[DEFAULT_LAYERS] = {429,2048,2048,2048,2048,2048,2048,8991};

static const char *getNetTypeName(MLP_NETTYPE nettype);
static const char *getActFuncName(ACT_FUNC actFunc);
static const char *getCostFuncName(COST_FUNC costFunc);
static MLP_NETTYPE getNetTypeID(string &typeName);
static ACT_FUNC getActFuncID(string &funcName);
static COST_FUNC getCostFuncID(string &funcName);

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

MLPNetProvider::MLPNetProvider(const char *configPath, const char *archFilename, const char *weightFilename)
{
	string archfname(configPath);
	string weightfname(configPath);

	archfname.append(archFilename);
	weightfname.append(weightFilename);

	ifstream archfile;
	ifstream weightfile;

	archfile.open(archfname.c_str(),ios_base::in);
	weightfile.open(weightfname.c_str(),ios_base::in|ios_base::binary);

	if ( ! archfile.is_open() || ! weightfile.is_open() ) {
		   mlp_log("MLPNET", "Failed to open MLP net config files for reading");
		   MLP_Exception("");
	};


	vector<string> lines;

    while ( ! archfile.eof() ) {
          string myline;

          getline(archfile,myline);
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
	this->netType = NETTYPE_MULTI_CLASSIFICATION;

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

	// read weights and biases data from the binary file
	weightfile.seekg(16);   // skip the checksum

	string corrMark("DONE");
	char Mark[5];

	weightfile.read(&Mark[0], 4);
	Mark[4] = '\0';

	if ( corrMark != Mark ) {
		 mlp_log("MLPNetProvider", "checking the integrity of the weight file failed, discarded");
		 MLP_Exception("");
	};

	for (int i=1; i < this->nLayers; i++) {
		for (int row=0; row < this->dimensions[i-1]; row++)
			 weightfile.read(reinterpret_cast<char*>(&this->weights[i][row*this->dimensions[i]]), sizeof(float)*this->dimensions[i]);
		weightfile.read(reinterpret_cast<char*>(&this->biases[i][0]),sizeof(float)*this->dimensions[i]);

		// convert to host bytes sequence from Big Endian bytes sequence
		/*
		for (int row=0; row < this->dimensions[i-1]; row++)
			 for (int col=0; col < this->dimensions[i]; col++)
				  BEtoHostl(*(unsigned int *)&this->weights[i][row*this->dimensions[i]+col]);
		for (int col=0; col < this->dimensions[i]; col++)
			 BEtoHostl(*(unsigned int *)&this->biases[i][col]);
		*/
	};

	archfile.close();
	weightfile.close();
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

	for (int i=1; i< this->nLayers; i++) {
		 datap = this->biases[i];
         for (int k=0; k< this->dimensions[i]; k++) {
			  *datap = 0.0f;
			  datap++;
		 };
	};
};

void MLPNetProvider::weightsInitialize()
{
	float *datap;

	struct mlp_tv tv;

	getCurrentTime(&tv);
	srand(tv.tv_usec); // use current time as random seed

	for (int i=1; i< this->nLayers; i++)
         for (int col=0; col< this->dimensions[i]; col++)       {      // go through all columns
			  datap = this->weights[i] + col;
		      for (int row=0; row< this->dimensions[i-1]; row++)  {    // go through all lines
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

void MLPNetProvider::saveConfig(const char *configPath, const char *archFilename, const char *weightFilename)
{
	string archfname(configPath);
	string weightfname(configPath);

	archfname.append(archFilename);
	weightfname.append(weightFilename);

	ofstream archfile;
	ofstream weightfile;

	archfile.open(archfname.c_str(),ios_base::out|ios_base::trunc);
	weightfile.open(weightfname.c_str(),ios_base::out|ios_base::binary|ios_base::trunc);

	if ( ! archfile.is_open() || ! weightfile.is_open() ) {
		   mlp_log("MLPNET", "Failed to create MLP net config files");
		   MLP_Exception("");
	};

	archfile << "### Configuration file for the neural network, produced automatically "  << endl;
	archfile << "### You can do modification for your need "  << endl;

	archfile << endl << "Network Type: " << getNetTypeName(this->netType) << endl;
	archfile << endl << "Layers: " << this->nLayers << endl;
	archfile << endl << "Cost Function: " << getCostFuncName(this->costFunc) << endl;
	archfile << endl << "Momentum: " << this->momentum << endl;
	archfile << endl << "Layer 0: " << this->dimensions[0] << endl;
	for (int i=1; i < this->nLayers; i++ )
		archfile << "Layer " << i << ": " << this->dimensions[i] << " " << this->etas[i] << " " << getActFuncName(this->actFuncs[i]) << endl;

	archfile.flush();

 	// Save the weights and Biases into binary file

	weightfile.seekp(16);   // first 16 bytes left as checksum
	weightfile.write("FAIL", 4);
	weightfile.flush();

	for (int i=1; i < this->nLayers; i++) {

        // convert to Big Endian bytes sequence from host bytes sequence
		/*
		for (int row=0; row < this->dimensions[i-1]; row++)
			 for (int col=0; col < this->dimensions[i]; col++)
				  HostToBEl(*(unsigned int *)&this->weights[i][row*this->dimensions[i]+col]);
		for (int col=0; col < this->dimensions[i]; col++)
			 HostToBEl(*(unsigned int *)&this->biases[i][col]);
		*/

		for (int row=0; row < this->dimensions[i-1]; row++)
			weightfile.write(reinterpret_cast<char*>(&this->weights[i][row*this->dimensions[i]]), sizeof(float)*this->dimensions[i]);

		weightfile.write(reinterpret_cast<char*>(&this->biases[i][0]),sizeof(float)*this->dimensions[i]);
	};

	weightfile.seekp(16);
	weightfile.write("DONE", 4);
	weightfile.flush();

	archfile.close();
	weightfile.close();
};

void MLPNetProvider::showConfig()
{
	cout << "Network Type: " << getNetTypeName(this->netType) << endl;
	cout << "Layers: " << this->nLayers << endl;
	cout << "Cost Function: " << getCostFuncName(this->costFunc) << endl;
    cout << "Momentum: " << this->momentum << endl;
	cout << "Layer 0: " << this->dimensions[0] << endl;
	for (int i=1; i < this->nLayers; i++ )
		cout << "Layer " << i << ": " << this->dimensions[i] << " " << this->etas[i] << " " << getActFuncName(this->actFuncs[i]) << endl;

	int myprec;

	myprec = (int)cout.precision();
	cout.precision(6);
	cout << endl;
	for (int i=1; i < this->nLayers; i++) {
		cout << "Weights connecting layer " << i-1 << " to " << i << endl;
		for (int row=0; row < this->dimensions[i-1]; row++) {
			for (int col=0; col < this->dimensions[i]; col++)
				 cout << this->weights[i][row*this->dimensions[i]+col] << " ";
			cout << endl;
		};
		cout << "Biases of layer " << i << endl;
		for (int col=0; col < this->dimensions[i]; col++)
			     cout << this->biases[i][col]<< " ";
		cout << endl << endl;
	};
	cout.precision(myprec);
};


static const char *getNetTypeName(MLP_NETTYPE nettype)
{
    switch (nettype) {
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
	switch (actFunc) {
	case AFUNC_SIGMOID:
		return("Sigmoid");
	case AFUNC_SOFTMAX:
		return("Softmax");
	case AFUNC_IDENTITY:
		return("Identity");
	case AFUNC_RELU:
		return("Relu");
	case AFUNC_TANH:
		return("Tanh");
	default:
		return("");
	};
};

static const char *getCostFuncName(COST_FUNC costFunc)
{
	switch (costFunc) {
	case CFUNC_SSE:
		return("SSE");
	case CFUNC_CE:
		return("CE");
	default:
		return("");
	};
};

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

static COST_FUNC getCostFuncID(string &funcName)
{
	if ( funcName == "SSE" )
		return(CFUNC_SSE);
	if ( funcName == "CE" )
		return(CFUNC_CE);

	return(CNOFUNC);
};
