
/***************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/
#include <parboil.h>
#include <stdio.h>
#include <stdlib.h>
#include <hc.hpp>

using namespace hc;

#include "file.h"
#include "common.h"
#include "kernels.hpp"

static int read_data(float *A0, int nx,int ny,int nz,FILE *fp)
{	
    int s=0;
    for(int i=0;i<nz;i++)
    {
        for(int j=0;j<ny;j++)
        {
            for(int k=0;k<nx;k++)
            {
                fread(A0+s,sizeof(float),1,fp);
                s++;
            }
        }
    }
    return 0;
}

int main(int argc, char** argv) {
    struct pb_TimerSet timers;
    struct pb_Parameters *parameters;



    printf("HCC accelerated 7 points stencil codes****\n");
    printf("Original version by Li-Wen Chang <lchang20@illinois.edu> and I-Jui Sung<sung10@illinois.edu>\n");
    printf("This version maintained by Chris Rodrigues  ***********\n");
    parameters = pb_ReadParameters(&argc, argv);

    pb_InitializeTimerSet(&timers);
    pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

    //declaration
    int nx,ny,nz;
    int size;
    int iteration;
    float c0=1.0f/6.0f;
    float c1=1.0f/6.0f/6.0f;

    if (argc<5)
    {
        printf("Usage: probe nx ny nz tx ty t\n"
                "nx: the grid size x\n"
                "ny: the grid size y\n"
                "nz: the grid size z\n"
                "t: the iteration time\n");
        return -1;
    }

    nx = atoi(argv[1]);
    if (nx<1)
        return -1;
    ny = atoi(argv[2]);
    if (ny<1)
        return -1;
    nz = atoi(argv[3]);
    if (nz<1)
        return -1;
    iteration = atoi(argv[4]);
    if(iteration<1)
        return -1;


    //host data
    float *h_A0;
    float *h_Anext;




    size=nx*ny*nz;

    h_A0=(float*)malloc(sizeof(float)*size);
    h_Anext=(float*)malloc(sizeof(float)*size);
    pb_SwitchToTimer(&timers, pb_TimerID_IO);
    FILE *fp = fopen(parameters->inpFiles[0], "rb");
    read_data(h_A0, nx,ny,nz,fp);
    fclose(fp);

    pb_SwitchToTimer(&timers, pb_TimerID_COPY);
    //memory allocation
    array_view<float> d_A0(size, h_A0);
    array_view<float> d_Anext(size, h_Anext);

    //memory copy
    copy(d_A0, d_Anext);

    pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

    //only use tx-by-ty threads
    int tx=32;
    int ty=4;

    int block[3] = {tx, ty, 1};
    //also change threads size maping from tx by ty to 2tx x ty
    int grid[3] = {(nx+tx*2-1)/(tx*2)*tx,(ny+ty-1)/ty*ty,1};
    // int sh_size = tx*2*ty*sizeof(float);	

    //main execution
    pb_SwitchToTimer(&timers, pb_TimerID_KERNEL);
    for(int t=0;t<iteration;t++)
    {
        parallel_for_each(extent<3>(grid).tile(block[0], block[1], block[2]),
                [=] (tiled_index<3> tidx) [[hc]]
                {
                block2D_hybrid_coarsen_x(tidx, c0, c1, d_A0, d_Anext, nx, ny, nz);
                });
        array_view<float> d_temp = std::move(d_A0);
        d_A0 = std::move(d_Anext);
        d_Anext = std::move(d_temp);

    }

    array_view<float> d_temp = std::move(d_A0);
    d_A0 = std::move(d_Anext);
    d_Anext = std::move(d_temp);



    pb_SwitchToTimer(&timers, pb_TimerID_COPY);
    d_Anext.synchronize();

    if (parameters->outFile) {
        pb_SwitchToTimer(&timers, pb_TimerID_IO);
        outputData(parameters->outFile,h_Anext,nx,ny,nz);

    }
    pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

    free (h_A0);
    free (h_Anext);
    pb_SwitchToTimer(&timers, pb_TimerID_NONE);

    pb_PrintTimerSet(&timers);
    pb_FreeParameters(parameters);

    return 0;

}
