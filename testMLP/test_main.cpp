/*
 *  COPYRIGHT:  Copyright (c) 2013 Advanced Micro Devices, Inc.  All rights reserved
 *
 *   Changed by:         Qianfeng Zhang@amd.com ( March 2014 )
 *
 *   Written By:               Junli Gu@amd.com ( Dec   2013 )
 */

#include <iostream>

using namespace std;

extern void simple_training();
extern void simple_batch_testing();
extern void simple_predicting();

extern void mnist_training();
extern void mnist_training2();
extern void mnist_training3();     // training with checkpointing support
extern void mnist_batch_testing();
extern void mnist_single_testing();
extern void mnist_predicting();

extern void ptc_uppercase_training();
extern void ptc_uppercase_training2();
extern void ptc_uppercase_training3();     // training with checkpointing support
extern void ptc_uppercase_batch_testing();
extern void ptc_uppercase_single_testing();
extern void ptc_uppercase_predicting();

extern void ptc_lowercase_training();
extern void ptc_lowercase_training2();
extern void ptc_lowercase_training3();     // training with checkpointing support
extern void ptc_lowercase_batch_testing();
extern void ptc_lowercase_single_testing();
extern void ptc_lowercase_predicting();

extern void ptc_digital_training();
extern void ptc_digital_training2();
extern void ptc_digital_training3();     // training with checkpointing support
extern void ptc_digital_batch_testing();
extern void ptc_digital_single_testing();
extern void ptc_digital_predicting();

extern void ptc_symbol_training();
extern void ptc_symbol_training2();
extern void ptc_symbol_training3();     // training with checkpointing support
extern void ptc_symbol_batch_testing();
extern void ptc_symbol_single_testing();
extern void ptc_symbol_predicting();


extern void vlp_ch_training();
extern void vlp_ch_training2();
extern void vlp_ch_training3();     // training with checkpointing support
extern void vlp_ch_batch_testing();
extern void vlp_ch_single_testing();
extern void vlp_ch_predicting();

extern void ptc_ch_training();
extern void ptc_ch_training2();
extern void ptc_ch_training3();     // training with checkpointing support
extern void ptc_ch_batch_testing();
extern void ptc_ch_single_testing();
extern void ptc_ch_predicting();

extern void iflytek_training();
extern void iflytek_training2();   // training with checkpointing support
extern void iflytek_batch_testing();
extern void iflytek_predicting();

extern void test_cp_cleanup();

int main()
{
	char anykey;

       iflytek_training2();
	//mnist_training();
	//mnist_training3();
	//ptc_ch_training3();
	//ptc_uppercase_training2();
	//ptc_lowercase_training();
	//ptc_digital_training();
	//ptc_symbol_training2();
	//vlp_ch_training2();
	//simple_training();

	//cout << "Press any key to continue ..." << endl;

	//cin >> anykey;

	//iflytek_batch_testing();
	 iflytek_predicting();
	//mnist_batch_testing();
	//mnist_single_testing();
	//ptc_ch_batch_testing();
	//ptc_uppercase_batch_testing();
	//ptc_lowercase_batch_testing();
	//ptc_digital_batch_testing();
	//ptc_symbol_batch_testing();
	//vlp_ch_batch_testing();
	//simple_batch_testing();
	//mnist_predicting();

	cout << "Press any key to end ..." << endl;

	cin >> anykey;

	return(0);
};

