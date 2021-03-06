/*
 *  COPYRIGHT:  Copyright (c) 2014 Advanced Micro Devices, Inc.  All rights reserved
 *
 *   Written by Qianfeng Zhang@amd.com ( March 2014 )
 *
 */

#ifndef _MLP_CHKPOINT_H_
#define _MLP_CHKPOINT_H_

#ifdef _WIN32
#include <Windows.h>
#else
#include <pthread.h>
#endif

#include <string>

#include "DNNApiExport.h"
#include "MLPTrainerBase.h"
#include "MLPChkPointState.h"

#define MLP_CHECKPOINTS_INFO  "mlp_checkpoints.inf"
#define MLP_CKPT_STATE_PREFIX "mlp_checkpoint_"          // Eg. the file name for checkpoint 4 will be  "mlp_checkpoint_4.dat"
#define MLP_CKPT_STATE_SUFFIX ".dat"

#define MLP_MAX_CHECKPOINTS  5                           // we want no more than 5 valid checkpoints for each checkpointed MLPTrainer

using namespace std;

class MLPCheckPointManager
{
private:
	string chkPointPath;

	bool haveChkPoint;
	struct MLPCheckPointState chkPointState;
	int  latestChkPoint;                        // Latest CheckPoint found under the assigned directory
	int  latestValidChkPoint;                   // Latest CheckPoint that has valid information, this one will be used for starting from

	bool useChkPointing;
	MLPTrainerBase *trainerp;                       // The MLPTrainer the CheckPoint Manager works for

#ifdef  _WIN32
    HANDLE chkPointingTimer;
#else
	pthread_t chkPointingTimer;
#endif
	bool running;                               // Indicate the checkpointing timer thread is running

	static void * timer_fun(void *argp);

public:
	LIBDNNAPI MLPCheckPointManager();
	LIBDNNAPI ~MLPCheckPointManager();
	LIBDNNAPI void cpFindAndLoad(const char *dirPath);    // Search the assigned directory to find a valid and latest checkpoint state and loat it into the manager
	LIBDNNAPI bool cpAvailable();                         // If the manager found an checkpoint state and have it loaded
	LIBDNNAPI void cpUnload();                            // Unload and release the checkpoint state from the manager
	LIBDNNAPI void cpCleanUp(const char *dirPath);        // Have all the files related to checkpoint states removed from the assigned directory

	LIBDNNAPI struct MLPCheckPointState *getChkPointState();

	LIBDNNAPI void enableCheckPointing(MLPTrainerBase & trainer, const char *dirPath);     // Ask the CheckPoint Manager to do checkpointing for the MLPTrainer
	LIBDNNAPI int startCheckPointing();
	LIBDNNAPI int endCheckPointing();
};

#endif
