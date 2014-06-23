/*
 *  COPYRIGHT:  Copyright (c) 2013 Advanced Micro Devices, Inc.  All rights reserved
 *
 *   Written by Qianfeng Zhang@amd.com
 *
 *   Kernels for various activation and derivation functions, various cost functions, transpose of matrix
 */

#define DIVUPK(val,K) (((val)+K-1)/(K))

__kernel void expandVectorToMatrix(global const float* myVector, global float *myMatrix, int width, int height)
{
    int gidx = get_global_id(0);
	int gidy = get_global_id(1);
	float4 val4;

	if ( (gidx < DIVUPK(width,4)) && (gidy < height) ) {
	     if ( gidx < width/4 ) {
	          val4 = vload4(0, (global float*) &myVector[gidx*4]);
	          vstore4(val4, 0, (global float*) &myMatrix[gidy*width+gidx*4]);
		 }
		 else {   // usually we need not go here since width is a multiple of 4
		      int left = width % 4;

			  for (int i=0; i< left; i++) {
			       float val;

				   val = myVector[gidx*4+i];
				   myMatrix[gidy*width+gidx*4+i] = val;
			  };
		 };
    }
};

__kernel void transpose_simple(global const float *src, global float *dst, int width, int height)
{
    int gidx = get_global_id(0);
	int gidy = get_global_id(1);

	if ( gidx < width && gidy < height )
	     dst[height*gidx+gidy]  = src[width*gidy+gidx];
};

__kernel void transpose_f4(global const float *src, global float *dst, int width, int height)
{
    int gidx = get_global_id(0);
	int gidy = get_global_id(1);
	float4 val4;

	if ( gidx < width/4 && gidy < height ) {
	     val4 = vload4(0, (global float*) &src[gidy*width+gidx*4]);
	     dst[(gidx*4)*height+gidy] = val4.s0;
	     dst[(gidx*4+1)*height+gidy] = val4.s1;
	     dst[(gidx*4+2)*height+gidy] = val4.s2;
	     dst[(gidx*4+3)*height+gidy] = val4.s3;
	};
};

// do the transposition using 32x32 size block
__kernel void transpose_32x32(global const float *src, global float *dst, int width, int height)
{
    int gidx = get_global_id(0);
	int gidy = get_global_id(1);
	int lidx = get_local_id(0);
	int lidy = get_local_id(1);
	int blkIdx = gidx / 8;
	int blkIdy = gidy / 32;
	float4 val4;

	local float lbuffer[32][32];     // 32*8 threads working one 32*32 block

	val4 = vload4(0, (global float*) &src[(blkIdy*32+lidy)*width+blkIdx*32+lidx*4]);   // 32*32 divided into 32*8 float4

    // store the float4 into transposed positions in the 32*32 block, so the whole 32*32 block is transposed by the local group
	lbuffer[lidx*4][lidy] =   val4.s0;
    lbuffer[lidx*4+1][lidy] = val4.s1;
    lbuffer[lidx*4+2][lidy] = val4.s2;
    lbuffer[lidx*4+3][lidy] = val4.s3;

    barrier(CLK_LOCAL_MEM_FENCE);

    val4 = vload4(0, (local float*) &lbuffer[lidy][lidx*4]);

    vstore4(val4, 0, (global float*) &dst[(blkIdx*32+lidy)*height+blkIdy*32+lidx*4]);
};

__kernel void activate_sigmoid(global const float *x, global float *y, int width, int height)
{
    int gidx = get_global_id(0);
	int gidy = get_global_id(1);

	if ( (gidx < DIVUPK(width,4)) && (gidy < height) ) {
	      if ( gidx < width/4  ) {
		       float4 xx4, yy4;

			   xx4 = vload4(0, (global float *)&x[gidy*width+gidx*4]);
			   yy4 = native_recip( 1.0f + native_exp(-xx4) );
			   vstore4(yy4, 0, (global float *)&y[gidy*width+gidx*4]);
		  }
		  else {   // usually we need not go here since width is a multiple of 4
		       int left = width % 4;

			   for (int i=0; i< left; i++) {
			        float xx;

					xx = x[gidy*width+gidx*4+i];
					y[gidy*width+gidx*4+i] = native_recip( 1.0f + native_exp(-xx) );
			   };
		  };
    };
}

__kernel void activate_tanh(global const float *x, global float *y, int width, int height)
{
    int gidx = get_global_id(0);
	int gidy = get_global_id(1);

	if ( (gidx < DIVUPK(width,4)) && (gidy < height) ) {
	      if ( gidx < width/4  ) {
		       float4 xx4, yy4;

			   xx4 = vload4(0, (global float *)&x[gidy*width+gidx*4]);
			   yy4 = tanh(xx4);
			   vstore4(yy4, 0, (global float *)&y[gidy*width+gidx*4]);
		  }
		  else {   // usually we need not go here since width is a multiple of 4
		       int left = width % 4;

			   for (int i=0; i< left; i++) {
			        float xx;

					xx = x[gidy*width+gidx*4+i];
					y[gidy*width+gidx*4+i] = tanh(xx);
			   };
		  };
    };
}

// let each thread to handle 4 units, each row of units can be handled inside one work group
__kernel void activate_softmax1(global const float *x, global float *y, int width, int height)
{
    int gidx = get_global_id(0);
	int gidy = get_global_id(1);
	int lidx = get_local_id(0);
	int lidy = get_local_id(1);
	int lsize0 = get_local_size(0);

	local float ltmpvals[256];

	if ( (lidx < DIVUPK(width,4)) && (gidy < height) ) {
	      float mysum = 0.0f;

	      if ( lidx < width/4 ) {
		       float4 xx4;

			   xx4 = vload4(0, (global float *)&x[gidy*width+lidx*4]);
			   mysum += native_exp(xx4.s0) + native_exp(xx4.s1) + native_exp(xx4.s2) + native_exp(xx4.s3);
		  }
		  else {    // usually we need not go here since width is a multiple of 4
		       int left = width % 4;

			   for (int i=0; i< left; i++) {
			        float xx;

					xx = x[gidy*width+lidx*4+i];
					mysum += native_exp(xx);
			   };
		  };
		  ltmpvals[lidy*lsize0+lidx] = mysum;
	}
	else
          ltmpvals[lidy*lsize0+lidx] = 0.0f;

    barrier(CLK_LOCAL_MEM_FENCE);

    // do the reducing to sum the values from all threads, the final sum is stored in ltmpvals[lidy*lsize0+0]
	int idx_size = lsize0/2;
	while ( idx_size ) {
	      if (  lidx < idx_size ) {
	 	        ltmpvals[lidy*lsize0+lidx] += ltmpvals[lidy*lsize0+lidx+idx_size];
		  };
		  idx_size = idx_size >> 1;
          barrier(CLK_LOCAL_MEM_FENCE);
	};

    // calculate the final softmax result for each unit
	if ( (lidx < DIVUPK(width,4)) && (gidy < height) ) {
	      if ( lidx < width/4 ) {
		       float4 xx4,yy4;

			   xx4 = vload4(0, (global float *)&x[gidy*width+lidx*4]);
			   yy4 = native_exp(xx4)/ltmpvals[lidy*lsize0];

			   vstore4(yy4, 0, (global float *)&y[gidy*width+lidx*4]);
		  }
		  else {    // usually we need not go here since width is a multiple of 4
		       int left = width % 4;

			   for (int i=0; i< left; i++) {
			        float xx;

					xx = x[gidy*width+lidx*4+i];
                    y[gidy*width+lidx*4+i] = native_exp(xx)/ltmpvals[lidy*lsize0];
			   };
		  };
	};
};


// let each work group to handle one row of units, each thread handle "DIVUP(width,4)/group_size" number of units,  group_size is 256
__kernel void activate_softmax2(global const float *x, global float *y, int width, int height)
{
    int gidx = get_global_id(0);
	int gidy = get_global_id(1);
	int lidx = get_local_id(0);

	local float ltmpvals[256];

	if ( gidy < height ) {
	     float mysum = 0.0f;
	     int myindex = lidx;

		 while ( myindex < DIVUPK(width,4) ) {
		       if ( myindex < width/4 ) {
		            float4 xx4;

			        xx4 = vload4(0, (global float *)&x[gidy*width+myindex*4]);
			        mysum += native_exp(xx4.s0) + native_exp(xx4.s1) + native_exp(xx4.s2) + native_exp(xx4.s3);
			   }
			   else {    // usually we need not go here since width is a multiple of 4
		            int left = width % 4;

			        for (int i=0; i< left; i++) {
			             float xx;

					     xx = x[gidy*width+myindex*4+i];
					     mysum += native_exp(xx);
					};
			   };
			   myindex += 256;
		 };

	     ltmpvals[lidx] = mysum;

	     barrier(CLK_LOCAL_MEM_FENCE);

	     // do the reducing to sum the values from all threads, the final sum is stored in ltmpvals[0]
	     int id_size = 256/2;
	     while ( id_size ) {
	          if (  lidx < id_size ) {
			        ltmpvals[lidx] += ltmpvals[lidx+id_size];
		      };
		      id_size = id_size >> 1;
              barrier(CLK_LOCAL_MEM_FENCE);
	     };

		 // calculate the final softmax result for each unit
	     myindex = lidx;
		 while ( myindex < DIVUPK(width,4) ) {
		       if ( myindex < width/4 ) {
		            float4 xx4, yy4;

			        xx4 = vload4(0, (global float *)&x[gidy*width+myindex*4]);
			        yy4 = native_exp(xx4)/ltmpvals[0];
					vstore4(yy4, 0, (global float *)&y[gidy*width+myindex*4]);
			   }
			   else {    // usually we need not go here since width is a multiple of 4
		            int left = width % 4;

			        for (int i=0; i< left; i++) {
			             float xx;

					     xx = x[gidy*width+myindex*4+i];
                         y[gidy*width+myindex*4+i] = native_exp(xx)/ltmpvals[0];
				    };
			   };
			   myindex += 256;
		 };
	};
};

__kernel void derivative_sigmoid(global float *delta1, global const float *y, global float *delta2, int width, int height)
{
	int gidx = get_global_id(0);
	int gidy = get_global_id(1);

	if ( (gidx < DIVUPK(width,4)) && (gidy < height) ) {
	      if ( gidx < width/4  ) {
		       float4 dd4, yy4;

			   dd4 = vload4(0, (global float *)&delta1[gidy*width+gidx*4]);
			   yy4 = vload4(0, (global float *)&y[gidy*width+gidx*4]);
			   yy4 = dd4*yy4*(1-yy4);

			   vstore4(yy4, 0, (global float *)&delta2[gidy*width+gidx*4]);
		  }
		  else {   // usually we need not go here since width is a multiple of 4
		       int left = width % 4;

			   for (int i=0; i< left; i++) {
			        float dd,yy;

					dd = delta1[gidy*width+gidx*4+i];
					yy = y[gidy*width+gidx*4+i];

					delta2[gidy*width+gidx*4+i] = dd*yy*(1-yy);
			   };
		  };
    };
}


__kernel void derivative_tanh(global float *delta1, global const float *y, global float *delta2, int width, int height)
{
	int gidx = get_global_id(0);
	int gidy = get_global_id(1);

	if ( (gidx < DIVUPK(width,4)) && (gidy < height) ) {
	      if ( gidx < width/4  ) {
		       float4 dd4, yy4;

			   dd4 = vload4(0, (global float *)&delta1[gidy*width+gidx*4]);
			   yy4 = vload4(0, (global float *)&y[gidy*width+gidx*4]);
			   yy4 = dd4*(1-yy4*yy4);

			   vstore4(yy4, 0, (global float *)&delta2[gidy*width+gidx*4]);
		  }
		  else {   // usually we need not go here since width is a multiple of 4
		       int left = width % 4;

			   for (int i=0; i< left; i++) {
			        float dd,yy;

					dd = delta1[gidy*width+gidx*4+i];
					yy = y[gidy*width+gidx*4+i];

					delta2[gidy*width+gidx*4+i] = dd*(1-yy*yy);
			   };
		  };
    };

}

// let each thread to handle 4 units, each row of units can be handled inside one work group
__kernel void calculateError_SSE1(global float* output,global float* target, global float *reduceOutput, int width,int height)
{
    int gidx = get_global_id(0);
	int gidy = get_global_id(1);
	int lidx = get_local_id(0);
	int lidy = get_local_id(1);
	int lsize0 = get_local_size(0);

	local float ltmpvals[256];

	if ( (lidx < DIVUPK(width,4)) && (gidy < height) ) {
	      float mysum=0.0f;

	      if ( lidx < width/4 ) {
		       float4 oo4,tt4;

			   oo4 = vload4(0, (global float *)&output[gidy*width+lidx*4]);
			   tt4 = vload4(0, (global float *)&target[gidy*width+lidx*4]);
			   oo4 = 0.5f * (oo4-tt4)*(oo4-tt4);
			   mysum = oo4.s0 + oo4.s1 + oo4.s2 + oo4.s3;
		  }
		  else {    // usually we need not go here since width is a multiple of 4
		       int left = width % 4;

			   for (int i=0; i< left; i++) {
			        float oo, tt;

					oo = output[gidy*width+lidx*4+i];
					tt = target[gidy*width+lidx*4+i];
					mysum += 0.5f * (oo-tt)*(oo-tt);
			   };
		  };
		  ltmpvals[lidy*lsize0+lidx] = mysum;
	}
	else
		  ltmpvals[lidy*lsize0+lidx] = 0.0f;

    barrier(CLK_LOCAL_MEM_FENCE);

    // do the reducing to sum the values from all threads, the final sum is stored in ltmpvals[lidy*lsize0+0]
	int idx_size = lsize0/2;
	while ( idx_size ) {
	      if (  lidx < idx_size ) {
	 	        ltmpvals[lidy*lsize0+lidx] += ltmpvals[lidy*lsize0+lidx+idx_size];
		  };
		  idx_size = idx_size >> 1;
          barrier(CLK_LOCAL_MEM_FENCE);
	};

	// write the reduced results to the host layer for further reducing
    if ( gidy < height && lidx == 0 )
	     reduceOutput[gidy] = ltmpvals[lidy*lsize0];
}

// let each work group to handle one row of units, each thread handle "DIVUP(width,4)/group_size" number of units,  group_size is 256
__kernel void calculateError_SSE2(global float* output,global float* target, global float *reduceOutput, int width,int height)
{
    int gidx = get_global_id(0);
	int gidy = get_global_id(1);
	int lidx = get_local_id(0);

	local float ltmpvals[256];

	if ( gidy < height ) {
	     float mysum = 0.0f;
	     int myindex = lidx;

		 while ( myindex < DIVUPK(width,4) ) {
		       if ( myindex < width/4 ) {
		            float4 oo4,tt4;

			        oo4 = vload4(0, (global float *)&output[gidy*width+myindex*4]);
					tt4 = vload4(0, (global float *)&target[gidy*width+myindex*4]);
			        oo4 = 0.5f * (oo4-tt4)*(oo4-tt4);
			        mysum += oo4.s0 + oo4.s1 + oo4.s2 + oo4.s3;
			   }
			   else {    // usually we need not go here since width is a multiple of 4
		            int left = width % 4;

			        for (int i=0; i< left; i++) {
			             float oo,tt;

					     oo = output[gidy*width+myindex*4+i];
						 tt = target[gidy*width+myindex*4+i];
                         mysum += 0.5f * (oo-tt)*(oo-tt);
					};
			   };
			   myindex += 256;
		 };

	     ltmpvals[lidx] = mysum;

	     barrier(CLK_LOCAL_MEM_FENCE);

	     // do the reducing to sum the values from all threads, the final sum is stored in ltmpvals[0]
	     int id_size = 256/2;
	     while ( id_size ) {
	          if (  lidx < id_size ) {
			        ltmpvals[lidx] += ltmpvals[lidx+id_size];
		      };
		      id_size = id_size >> 1;
              barrier(CLK_LOCAL_MEM_FENCE);
	     };

	     // write the reduced results to the host layer for further reducing
		 if ( lidx == 0 )
	          reduceOutput[gidy] = ltmpvals[0];
	};
};


// let each thread to handle 4 units, each row of units can be handled inside one work group
__kernel void calculateError_CE1(global float *output,global float *target,global float *reduceOutput, int width,int height)
{
    int gidx = get_global_id(0);
	int gidy = get_global_id(1);
	int lidx = get_local_id(0);
	int lidy = get_local_id(1);
	int lsize0 = get_local_size(0);

	local float ltmpvals[256];

	if ( (lidx < DIVUPK(width,4)) && (gidy < height) ) {
	      float mysum=0.0f;

	      if ( lidx < width/4 ) {
		       float4 oo4,tt4;

			   oo4 = vload4(0, (global float *)&output[gidy*width+lidx*4]);
			   tt4 = vload4(0, (global float *)&target[gidy*width+lidx*4]);
			   oo4 = (-1.0f) * tt4 * native_log(oo4);
			   mysum = oo4.s0 + oo4.s1 + oo4.s2 + oo4.s3;
		  }
		  else {    // usually we need not go here since width is a multiple of 4
		       int left = width % 4;

			   for (int i=0; i< left; i++) {
			        float oo, tt;

					oo = output[gidy*width+lidx*4+i];
					tt = target[gidy*width+lidx*4+i];
					mysum += (-1.0f) * tt * native_log(oo);
			   };
		  };
		  ltmpvals[lidy*lsize0+lidx] = mysum;
	}
	else
          ltmpvals[lidy*lsize0+lidx] = 0.0f;

    barrier(CLK_LOCAL_MEM_FENCE);

    // do the reducing to sum the values from all threads, the final sum is stored in ltmpvals[lidy*lsize0+0]
	int idx_size = lsize0/2;
	while ( idx_size ) {
	      if (  lidx < idx_size ) {
	 	        ltmpvals[lidy*lsize0+lidx] += ltmpvals[lidy*lsize0+lidx+idx_size];
		  };
		  idx_size = idx_size >> 1;
          barrier(CLK_LOCAL_MEM_FENCE);
	};

	// write the reduced results to the host layer for further reducing
    if ( gidy < height  && lidx == 0 )
	     reduceOutput[gidy] = ltmpvals[lidy*lsize0];
}

// let each work group to handle one row of units, each thread handle "DIVUP(width,4)/group_size" number of units,  group_size is 256
__kernel void calculateError_CE2(global float* output,global float* target,global float *reduceOutput, int width,int height)
{
    int gidx = get_global_id(0);
	int gidy = get_global_id(1);
	int lidx = get_local_id(0);

	local float ltmpvals[256];

	if ( gidy < height ) {
	     float mysum = 0.0f;
	     int myindex = lidx;

		 while ( myindex < DIVUPK(width,4) ) {
		       if ( myindex < width/4 ) {
		            float4 oo4,tt4;

			        oo4 = vload4(0, (global float *)&output[gidy*width+myindex*4]);
					tt4 = vload4(0, (global float *)&target[gidy*width+myindex*4]);
			        oo4 = (-1.0f) * tt4 * native_log(oo4);
			        mysum += oo4.s0 + oo4.s1 + oo4.s2 + oo4.s3;
			   }
			   else {    // usually we need not go here since width is a multiple of 4
		            int left = width % 4;

			        for (int i=0; i< left; i++) {
			             float oo,tt;

					     oo = output[gidy*width+myindex*4+i];
						 tt = target[gidy*width+myindex*4+i];
                         mysum += (-1.0f)* tt * native_log(oo);
				    };
			   };
			   myindex += 256;
		 };

	     ltmpvals[lidx] = mysum;

	     barrier(CLK_LOCAL_MEM_FENCE);

	     // do the reducing to sum the values from all threads, the final sum is stored in ltmpvals[0]
	     int id_size = 256/2;
	     while ( id_size ) {
	          if (  lidx < id_size ) {
			        ltmpvals[lidx] += ltmpvals[lidx+id_size];
		      };
		      id_size = id_size >> 1;
              barrier(CLK_LOCAL_MEM_FENCE);
	     };

	     // write the reduced results to the host layer for further reducing
		 if ( lidx == 0 )
	          reduceOutput[gidy] = ltmpvals[0];
	};
};


__kernel void calculateDelta_CE_Softmax(global float* output,global float* target,global float* delta,int width, int height)
{
	int gidx = get_global_id(0);
	int gidy = get_global_id(1);

	if ( (gidx < DIVUPK(width,4)) && (gidy < height) ) {
	      if ( gidx < width/4  ) {
		       float4 tt4,yy4;

	           yy4 = vload4(0, (global float *)&output[gidy*width+gidx*4]);
			   tt4 = vload4(0, (global float *)&target[gidy*width+gidx*4]);

			   yy4 = tt4-yy4;
			   vstore4(yy4, 0, (global float *)&delta[gidy*width+gidx*4]);
		  }
		  else {   // usually we need not go here since width is a multiple of 4
		       int left = width % 4;

			   for (int i=0; i< left; i++) {
			        float tt,yy;

					yy = output[gidy*width+gidx*4+i];
					tt = target[gidy*width+gidx*4+i];

					delta[gidy*width+gidx*4+i] = tt-yy;
			   };
		  };
    };

};


__kernel void calculateDelta_SSE_Sigmoid(global float* output,global float* target,global float* delta,int width, int height)
{
	int gidx = get_global_id(0);
	int gidy = get_global_id(1);

	if ( (gidx < DIVUPK(width,4)) && (gidy < height) ) {
	      if ( gidx < width/4  ) {
		       float4 tt4,yy4;

	           yy4 = vload4(0, (global float *)&output[gidy*width+gidx*4]);
			   tt4 = vload4(0, (global float *)&target[gidy*width+gidx*4]);

			   yy4 = (tt4-yy4)*yy4*(1-yy4);
			   vstore4(yy4, 0, (global float *)&delta[gidy*width+gidx*4]);
		  }
		  else {   // usually we need not go here since width is a multiple of 4
		       int left = width % 4;

			   for (int i=0; i< left; i++) {
			        float tt,yy;

					yy = output[gidy*width+gidx*4+i];
					tt = target[gidy*width+gidx*4+i];

					delta[gidy*width+gidx*4+i] = (tt-yy)*yy*(1-yy);
			   };
		  };
    };
};

