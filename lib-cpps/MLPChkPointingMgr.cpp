/*
 *  COPYRIGHT:  Copyright (c) 2014 Advanced Micro Devices, Inc.  All rights reserved
 *
 *   Written by Qianfeng Zhang@amd.com ( March 2014 )
 *
 */

#include <iostream>
#include <fstream>
#include <sstream>

#include "MLPUtil.h"
#include "MLPChkPointingMgr.h"

using namespace std;

#define MLP_CHKPOINTING_PERIOD 1200                // checkpointing period in seconds, usually one hour

MLPCheckPointManager::MLPCheckPointManager()
{
	this->haveChkPoint = false;
	this->latestChkPoint = 0;
	this->latestValidChkPoint = 0;

	this->useChkPointing = false;
	this->running = 0;
};

MLPCheckPointManager::~MLPCheckPointManager()
{
};

void MLPCheckPointManager::cpFindAndLoad(const char *dirPath)
{
	ostringstream strFname;
	ifstream stateFile;
	fstream infoFile;
	string infoFname;

	this->chkPointPath = dirPath;
	this->chkPointPath += "/";
	infoFname = this->chkPointPath +  MLP_CHECKPOINTS_INFO;

	infoFile.open(infoFname.c_str(), ios_base::in);
	if ( infoFile.is_open() ) {   // foud checkpoints
		int tmpID;
		char Mark[5];
		string corrMark("DONE");

		infoFile >> tmpID;
		tmpID += 1;        // search one more checkpoint since the mlp_checkpoints.inf file might not have been written successfully

		this->latestChkPoint = tmpID;
		this->latestValidChkPoint = 0;

		for (int i=0; (i< MLP_MAX_CHECKPOINTS+1) && (tmpID > 0); i++, tmpID--) {
			 strFname.str("");
			 strFname << this->chkPointPath << MLP_CKPT_STATE_PREFIX << tmpID << MLP_CKPT_STATE_SUFFIX;   // Filename for checkpoint state file
		     stateFile.open(strFname.str().c_str(), ios_base::in | ios_base::binary );
		     if ( !stateFile.is_open() ) {
				   this->latestChkPoint--;
	               continue;
			 };

		     stateFile.seekg(16);
 	         stateFile.read(reinterpret_cast<char*>(&Mark[0]), 4);
	         Mark[4] = '\0';
	         if ( corrMark != Mark ) {
		          mlp_log("MLPChkPoint", "Checking the integrity of the checkpoint state file failed:");
				  mlp_log("MLPChkPoint", strFname.str().c_str());
				  mlp_log("MLPChkPoint", "The files by this checkpoint may need be cleaned manualy");
                  stateFile.close();
				  continue;
	         }
		     else {   // The checkpoint state is valid
				 stateFile.read(reinterpret_cast<char*>(&this->chkPointState), sizeof(struct MLPCheckPointState));
				 stateFile.close();

		         LEtoHostl(this->chkPointState.chkPointID);
		         LEtoHostl(this->chkPointState.cpBatchNo);
		         LEtoHostl(this->chkPointState.cpFrameNo);
				 LEtoHostl(this->chkPointState.cpEpoch);

				 this->latestValidChkPoint = tmpID;
				 this->haveChkPoint = true;

				 break;
 		     };
		};

		infoFile.close();

		// Update the info with latest valid checkpoint ID
		if ( this->haveChkPoint ) {
	         infoFile.open(infoFname.c_str(), ios_base::out | ios_base::trunc);
			 infoFile << this->latestValidChkPoint;
		};
	}
	else {
		this->haveChkPoint = false;
	};
};

void MLPCheckPointManager::cpUnload()
{
	this->haveChkPoint = false;
};

bool MLPCheckPointManager::cpAvailable()
{
	 return(this->haveChkPoint);
};

void MLPCheckPointManager::cpCleanUp(const char *dirPath)
{
	ostringstream strFname;
	ifstream stateFile;
	fstream infoFile;
	string infoFname;

	this->chkPointPath = dirPath;
	this->chkPointPath += "/";
	infoFname = this->chkPointPath +  MLP_CHECKPOINTS_INFO;

	infoFile.open(infoFname.c_str(), ios_base::in);
	if ( infoFile.is_open() ) {
		struct MLPCheckPointState tmpState;
		int tmpID;
		char Mark[5];
		string corrMark("DONE");

		infoFile >> tmpID;
		tmpID += 1;        // search one more checkpoint since the mlp_checkpoints.inf file might not have been written successfully

		for (int i=0; (i< MLP_MAX_CHECKPOINTS+1) && (tmpID > 0); i++, tmpID--) {
			 strFname.str("");
			 strFname << this->chkPointPath << MLP_CKPT_STATE_PREFIX << tmpID << MLP_CKPT_STATE_SUFFIX;   // Filename for checkpoint state file
		     stateFile.open(strFname.str().c_str(), ios_base::in | ios_base::binary );
		     if ( !stateFile.is_open() )
	               continue;                // Just go to check the next less numbered checkpoint

		     stateFile.seekg(16);
 	         stateFile.read(reinterpret_cast<char*>(&Mark[0]), 4);
	         Mark[4] = '\0';
	         if ( corrMark != Mark ) {
		          mlp_log("MLPChkPoint", "Checking the integrity of the checkpoint state file failed");
				  mlp_log("MLPChkPoint", strFname.str().c_str());
				  mlp_log("MLPChkPoint", "The files by this checkpoint may need be cleaned manualy");
                  stateFile.close();
				  remove(strFname.str().c_str());  // remove the checkpoint state file
	         }
		     else {
				 stateFile.read(reinterpret_cast<char*>(&tmpState), sizeof(struct MLPCheckPointState));

				 string tmpFname;

				 tmpFname =  tmpState.netConfPath;
				 tmpFname += tmpState.netConfArchFileName ;
				 remove(tmpFname.c_str());

				 tmpFname.clear();

				 tmpFname = tmpState.netConfPath;
				 tmpFname += tmpState.netConfDataFileName;
				 remove(tmpFname.c_str());

				 stateFile.close();
				 remove(strFname.str().c_str());
 		     };
		};

		infoFile.close();
	}

	remove(infoFname.c_str());
};

struct MLPCheckPointState * MLPCheckPointManager::getChkPointState()
{
	 return(&this->chkPointState);
};

void MLPCheckPointManager::enableCheckPointing(MLPTrainer & trainer, const char *dirPath)
{
	this->trainerp = &trainer;
	this->useChkPointing = true;

	string infoFname;
	ifstream infoFile;

	this->chkPointPath = dirPath;
	this->chkPointPath.append("/");

	infoFname = this->chkPointPath + MLP_CHECKPOINTS_INFO;

	infoFile.open(infoFname.c_str(), ios_base::in);
	if ( infoFile.is_open() ) {
		int tmpID;

		infoFile >> tmpID;
		this->chkPointState.chkPointID = (tmpID < 1)? 1UL: (unsigned int)(tmpID+1);
		infoFile.close();
	}
	else {
		this->chkPointState.chkPointID = 1UL;
	};
};

int MLPCheckPointManager::startCheckPointing()
{
	if ( ! this->useChkPointing )
		 return(-1);

	if ( this->running )
		 return(-2);

	this->running = true;
	MLP_CREATE_THREAD(&this->chkPointingTimer,MLPCheckPointManager::timer_fun,(void*)this);

	return(0);
};

int MLPCheckPointManager::endCheckPointing()
{
	if ( ! this->useChkPointing )
		 return(-1);

	if ( ! this->running )
		 return(-2);

    MLP_KILL_THREAD(this->chkPointingTimer);
	MLP_JOIN_THREAD(this->chkPointingTimer);
	this->running = false;

	return(0);
};


void *MLPCheckPointManager::timer_fun(void *argp)
{
	 MLPCheckPointManager *objp;

	 objp = ( MLPCheckPointManager *)argp;

	 mlp_log("MLPChkPoint", "MLPCheckPoining thread started");

	 while ( objp->running ) {
		   ostringstream strFname;
		   string infoFname;
		   fstream stateFile, infoFile;
		   int len;

		   MLP_SLEEP(MLP_CHKPOINTING_PERIOD);        // do checkpointing every one hour

		   // Produce the names of the network configuration files
           // Use the checkpoint directory itself to save the network configuration files
		   len = (int) objp->chkPointPath.copy(objp->chkPointState.netConfPath, 255);
		   objp->chkPointState.netConfPath[len] = '\0';

		   strFname << "mlp_cp_netarch_" << objp->chkPointState.chkPointID << ".conf";
		   len = (int) strFname.str().copy(objp->chkPointState.netConfArchFileName, 31);
		   objp->chkPointState.netConfArchFileName[len] = '\0';
		   strFname.str("");

		   strFname << "mlp_cp_weights_" << objp->chkPointState.chkPointID << ".dat";
		   len = (int) strFname.str().copy(objp->chkPointState.netConfDataFileName, 31);
		   objp->chkPointState.netConfDataFileName[len] = '\0';
		   strFname.str("");

           // Create the CheckPoint State file
		   strFname << objp->chkPointPath << MLP_CKPT_STATE_PREFIX << objp->chkPointState.chkPointID << MLP_CKPT_STATE_SUFFIX ;
		   stateFile.open(strFname.str().c_str(), ios_base::out | ios_base::trunc | ios_base::binary );
		   if ( !stateFile.is_open() ) {
	            mlp_log("MLPCHKPOINT", "Failed to create MLP Checkpoint State file");
		        MLP_Exception("");
		   };
		   stateFile.seekp(16);         // Leave some space for checksum
	       stateFile.write("FAIL", 4);  // Tag the file header before the real content being written to the files
	       stateFile.flush();

		   objp->trainerp->checkPointing(objp->chkPointState);

		   // Save the checkpoint state
		   HostToLEl(objp->chkPointState.chkPointID);
		   HostToLEl(objp->chkPointState.cpBatchNo);
		   HostToLEl(objp->chkPointState.cpFrameNo);
		   HostToLEl(objp->chkPointState.cpEpoch);

	       stateFile.write(reinterpret_cast<char*>(&objp->chkPointState), sizeof(struct MLPCheckPointState));

		   stateFile.seekp(16);
		   stateFile.write("DONE", 4);  // Tag the file header again after the real content is written
		   stateFile.flush();

		   LEtoHostl(objp->chkPointState.chkPointID);
		   LEtoHostl(objp->chkPointState.cpBatchNo);
		   LEtoHostl(objp->chkPointState.cpFrameNo);
		   LEtoHostl(objp->chkPointState.cpEpoch);

		   stateFile.close();

		   // Update the mlp_checkpoints.inf
		   infoFname = objp->chkPointPath + MLP_CHECKPOINTS_INFO;
		   infoFile.open(infoFname.c_str(), ios_base::out|ios_base::trunc);
		   if ( !infoFile.is_open() ) {
	            mlp_log("MLPChkPoint", "Failed to open MLP Checkpoint info file for writing");
		        MLP_Exception("");
		   };

		   infoFile << objp->chkPointState.chkPointID;

		   strFname.str("");

		   mlp_log("MLPChkPoint", "A new checkpoint state has just been saved");

		   // Clean up the obsolete checkpoint file, we only keep 5 checkpoints

		   int tmpID;
		   struct MLPCheckPointState tmpState;
	       string corrMark("DONE");
	       char Mark[5];

		   tmpID =  objp->chkPointState.chkPointID - MLP_MAX_CHECKPOINTS;
		   strFname << objp->chkPointPath << MLP_CKPT_STATE_PREFIX << tmpID << MLP_CKPT_STATE_SUFFIX ;   // Filename for checkpoint state file
		   stateFile.open(strFname.str().c_str(), ios_base::in | ios_base::binary );
		   if ( !stateFile.is_open() ) {
                goto next;
		   };
		   stateFile.seekg(16);

 	       stateFile.read(reinterpret_cast<char*>(&Mark[0]), 4);
	       Mark[4] = '\0';

	       if ( corrMark != Mark ) {
		        mlp_log("MLPChkPoint", "Checking the integrity of the checkpoint state file failed");
				mlp_log("MLPChkPoint", strFname.str().c_str());
				mlp_log("MLPChkPoint", "The other files by this checkpoint may need be cleaned manualy");
                stateFile.close();
				remove(strFname.str().c_str());  // remove the checkpoint state file
	       }
		   else {
			    stateFile.read(reinterpret_cast<char*>(&tmpState), sizeof(struct MLPCheckPointState));
				stateFile.close();
				remove(strFname.str().c_str());  // remove the checkpoint state file

				strFname.str("");
				strFname << tmpState.netConfPath << tmpState.netConfArchFileName ;       // remove the mlp_cp_netarch_xxx.conf file
				remove(strFname.str().c_str());

				strFname.str("");
				strFname << tmpState.netConfPath << tmpState.netConfDataFileName ;       // remove the mlp_cp_netweights_xxx.dat file
				remove(strFname.str().c_str());
		   };

	   next:
		   // next checkpoint
		   objp->chkPointState.chkPointID++;
	 };

	 return(0);
};
