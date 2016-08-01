/***************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

#include "common.h"


void block2D_hybrid_coarsen_x(
        tiled_index<3>& tidx,
        float c0,float c1,
        const array_view<float>& A0, const array_view<float>& Anext,
        int nx, int ny, int nz) [[hc]]
{
    index<3> threadIdx(tidx.local);
    index<3> blockIdx(tidx.tile);
    index<3> blockDim(tidx.tile_dim);

	//thread coarsening along x direction
	const int i = blockIdx[0]*blockDim[0]*2+threadIdx[0];
	const int i2= blockIdx[0]*blockDim[0]*2+threadIdx[0]+blockDim[0];
    const int j = blockIdx[1]*blockDim[1]+threadIdx[1];
	const int sh_id=threadIdx[0] + threadIdx[1]*blockDim[0]*2;
	const int sh_id2=threadIdx[0] +blockDim[0]+ threadIdx[1]*blockDim[0]*2;
	
	//shared memeory
	tile_static float sh_A0[32*4*2];
	sh_A0[sh_id]=0.0f;
	sh_A0[sh_id2]=0.0f;
    tidx.barrier.wait();
	
	//get available region for load and store
	const bool w_region =  i>0 && j>0 &&(i<(nx-1)) &&(j<(ny-1)) ;
	const bool w_region2 =  j>0 &&(i2<nx-1) &&(j<ny-1) ;
	const bool x_l_bound = (threadIdx[0]==0);
	const bool x_h_bound = ((threadIdx[0]+blockDim[0])==(blockDim[0]*2-1));
	const bool y_l_bound = (threadIdx[1]==0);
	const bool y_h_bound = (threadIdx[1]==(blockDim[1]-1));

	//register for bottom and top planes
	//because of thread coarsening, we need to doulbe registers
	float bottom=0.0f,bottom2=0.0f,top=0.0f,top2=0.0f;
	
	//load data for bottom and current
	if((i<nx) &&(j<ny))
	{

		bottom=A0[Index3D (nx, ny, i, j, 0)];
		sh_A0[sh_id]=A0[Index3D (nx, ny, i, j, 1)];
	}
	if((i2<nx) &&(j<ny))
	{
		bottom2=A0[Index3D (nx, ny, i2, j, 0)];
		sh_A0[sh_id2]=A0[Index3D (nx, ny, i2, j, 1)];
	}

    tidx.barrier.wait();
	
	for(int k=1;k<nz-1;k++)
	{

		float a_left_right,a_up,a_down;		
		
		//load required data on xy planes
		//if it on shared memory, load from shared memory
		//if not, load from global memory
		if((i<nx) &&(j<ny))
			top=A0[Index3D (nx, ny, i, j, k+1)];
			
		if(w_region)
		{
			a_up        =y_h_bound?A0[Index3D (nx, ny, i, j+1, k )]:sh_A0[sh_id+2*blockDim[0]];
      a_down      =y_l_bound?A0[Index3D (nx, ny, i, j-1, k )]:sh_A0[sh_id-2*blockDim[0]];
			a_left_right=x_l_bound?A0[Index3D (nx, ny, i-1, j, k )]:sh_A0[sh_id-1];
	
			Anext[Index3D (nx, ny, i, j, k)] = (top + bottom + a_up + a_down + sh_A0[sh_id+1] +a_left_right)*c1
                                        -  sh_A0[sh_id]*c0;		
    }
		
		
		//load another block
		if((i2<nx) &&(j<ny))
			top2=A0[Index3D (nx, ny, i2, j, k+1)];
			
		if(w_region2)
		{
		  a_up        =y_h_bound?A0[Index3D (nx, ny, i2, j+1, k )]:sh_A0[sh_id2+2*blockDim[0]];
      a_down      =y_l_bound?A0[Index3D (nx, ny, i2, j-1, k )]:sh_A0[sh_id2-2*blockDim[0]];
			a_left_right=x_h_bound?A0[Index3D (nx, ny, i2+1, j, k )]:sh_A0[sh_id2+1];

			
			Anext[Index3D (nx, ny, i2, j, k)] = (top2 + bottom2 + a_up + a_down + a_left_right +sh_A0[sh_id2-1])*c1
                                        -  sh_A0[sh_id2]*c0;
		}

		//swap data
        tidx.barrier.wait();
        bottom=sh_A0[sh_id];
        sh_A0[sh_id]=top;
        bottom2=sh_A0[sh_id2];
        sh_A0[sh_id2]=top2;
        tidx.barrier.wait();
    }


}


