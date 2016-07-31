/***************************************************************************
 *
 *            (C) Copyright 2010 The Board of Trustees of the
 *                        University of Illinois
 *                         All Rights Reserved
 *
 ***************************************************************************/


#include <parboil.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <hc.hpp>
#include <hc_math.hpp>

#include "util.h"

using namespace hc;

#include "histo_final.hpp"
#include "histo_intermediates.hpp"
#include "histo_main.hpp"
#include "histo_prescan.hpp"


/******************************************************************************
* Implementation: GPU
* Details:
* in the GPU implementation of histogram, we begin by computing the span of the
* input values into the histogram. Then the histogramming computation is carried
* out by a (BLOCK_X, BLOCK_Y) sized grid, where every group of Y (same X)
* computes its own partial histogram for a part of the input, and every Y in the
* group exclusively writes to a portion of the span computed in the beginning.
* Finally, a reduction is performed to combine all the partial histograms into
* the final result.
******************************************************************************/

int main(int argc, char* argv[]) {
  struct pb_TimerSet *timersPtr;
  struct pb_Parameters *parameters;

  parameters = pb_ReadParameters(&argc, argv);
  if (!parameters)
    return -1;

  if(!parameters->inpFiles[0]){
    fputs("Input file expected\n", stderr);
    return -1;
  }

  timersPtr = (struct pb_TimerSet *) malloc (sizeof(struct pb_TimerSet));


  //appendDefaultTimerSet(NULL);


  if (timersPtr == NULL) {
    fprintf(stderr, "Could not append default timer set!\n");
    exit(1);
  }

  struct pb_TimerSet timers = *timersPtr;

//  pb_CreateTimer(&timers, "myTimer!", 0);


  pb_InitializeTimerSet(&timers);

  pb_AddSubTimer(&timers, "Input", pb_TimerID_IO);
  pb_AddSubTimer(&timers, "Output", pb_TimerID_IO);

  char *prescans = "PreScanKernel";
  char *postpremems = "PostPreMems";
  char *intermediates = "IntermediatesKernel";
  char *mains = "MainKernel";
  char *finals = "FinalKernel";

  pb_AddSubTimer(&timers, prescans, pb_TimerID_KERNEL);
  pb_AddSubTimer(&timers, postpremems, pb_TimerID_KERNEL);
  pb_AddSubTimer(&timers, intermediates, pb_TimerID_KERNEL);
  pb_AddSubTimer(&timers, mains, pb_TimerID_KERNEL);
  pb_AddSubTimer(&timers, finals, pb_TimerID_KERNEL);

//  pb_SwitchToTimer(&timers, pb_TimerID_IO);
  pb_SwitchToSubTimer(&timers, "Input", pb_TimerID_IO);

  int numIterations;
  if (argc >= 2){
    numIterations = atoi(argv[1]);
  } else {
    fputs("Expected at least one command line argument\n", stderr);
    return -1;
  }

  unsigned int img_width, img_height;
  unsigned int histo_width, histo_height;

  FILE* f = fopen(parameters->inpFiles[0],"rb");
  int result = 0;

  result += fread(&img_width,    sizeof(unsigned int), 1, f);
  result += fread(&img_height,   sizeof(unsigned int), 1, f);
  result += fread(&histo_width,  sizeof(unsigned int), 1, f);
  result += fread(&histo_height, sizeof(unsigned int), 1, f);

  if (result != 4){
    fputs("Error reading input and output dimensions from file\n", stderr);
    return -1;
  }

  unsigned int* img = (unsigned int*) malloc (img_width*img_height*sizeof(unsigned int));
  unsigned char* histo = (unsigned char*) calloc (histo_width*histo_height, sizeof(unsigned char));

  result = fread(img, sizeof(unsigned int), img_width*img_height, f);

  fclose(f);

  if (result != img_width*img_height){
    fputs("Error reading input array from file\n", stderr);
    return -1;
  }

  int even_width = ((img_width+1)/2)*2;

  array_view<unsigned int> input(even_width*(((img_height+UNROLL-1)/UNROLL)*UNROLL));
  array_view<unsigned int> ranges(2);
  array_view<uchar4> sm_mappings(img_width*img_height);
  array_view<unsigned int> global_subhisto(img_width*histo_height);
  array_view<unsigned short> global_histo(img_width*histo_height);
  array_view<unsigned int> global_overflow(img_width*histo_height);
  array_view<unsigned char> final_histo(img_width*histo_height);


  for (int y=0; y < img_height; y++){
    array_view<unsigned int> src(img_width, img + y*img_width);
    copy(src, input.section(y*even_width, img_width));
  }

  //pb_SwitchToTimer(&timers, pb_TimerID_KERNEL);
  pb_SwitchToSubTimer(&timers, NULL, pb_TimerID_KERNEL);


  unsigned int *zeroData = (unsigned int *) calloc(img_width*histo_height, sizeof(unsigned int));


  for (int iter = 0; iter < numIterations; iter++) {
    unsigned int ranges_h[2] = {UINT32_MAX, 0};

    ranges[0] = ranges_h[0];
    ranges[1] = ranges_h[1];

    parallel_for_each(extent<1>(PRESCAN_BLOCKS_X * PRESCAN_THREADS).tile(PRESCAN_THREADS),
            [=] (tiled_index<1> tidx) [[hc]]
            {
            histo_prescan_kernel(tidx, input, PRESCAN_BLOCKS_X * PRESCAN_THREADS,
                img_height*img_width, ranges);
            });

    pb_SwitchToSubTimer(&timers, postpremems , pb_TimerID_KERNEL);

    ranges_h[0] = ranges[0];
    ranges_h[1] = ranges[1];

    copy(zeroData, zeroData + img_width*histo_height, global_subhisto);
    //    cudaMemset(global_subhisto,0,img_width*histo_height*sizeof(unsigned int));

    pb_SwitchToSubTimer(&timers, intermediates, pb_TimerID_KERNEL);

    int t = (img_width+1)/2;
    int e = ((img_height + UNROLL-1)/UNROLL) * t;
    parallel_for_each(extent<1>(e).tile(t), [=] (tiled_index<1> tidx) [[hc]]
            {
            histo_intermediates_kernel(tidx, input.reinterpret_as<uint2>(),
                img_height, img_width, (img_width+1)/2, sm_mappings);
            });

    pb_SwitchToSubTimer(&timers, mains, pb_TimerID_KERNEL);

    parallel_for_each(extent<2>(BLOCK_X * THREADS, ranges_h[1] - ranges_h[0] + 1).tile(THREADS, 1),
            [=] (tiled_index<2> tidx) [[hc]]
            {
            histo_main_kernel(tidx, sm_mappings, BLOCK_X * THREADS,
                img_height*img_width, ranges[0], ranges[1],
                histo_height, histo_width, global_subhisto,
                global_histo.reinterpret_as<unsigned int>(), global_overflow);
            });

    pb_SwitchToSubTimer(&timers, finals, pb_TimerID_KERNEL);

    parallel_for_each(extent<1>(BLOCK_X*3*512).tile(512),
            [=] (tiled_index<1> tidx) [[hc]]
            {
            histo_final_kernel(tidx, BLOCK_X*3*512,
                ranges[0], ranges[1],
                histo_height, histo_width,
                global_subhisto, global_histo.reinterpret_as<unsigned int>(),
                global_overflow, final_histo.reinterpret_as<unsigned int>());
            });
  }

  pb_SwitchToSubTimer(&timers, "Output", pb_TimerID_IO);
  //  pb_SwitchToTimer(&timers, pb_TimerID_IO);


  copy(final_histo.section(0, histo_height*histo_width), histo);

  if (parameters->outFile) {
    dump_histo_img(histo, histo_height, histo_width, parameters->outFile);
  }

  //pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
  pb_SwitchToSubTimer(&timers, NULL, pb_TimerID_COMPUTE);

  free(img);
  free(histo);

  pb_SwitchToSubTimer(&timers, NULL, pb_TimerID_NONE);

  printf("\n");
  pb_PrintTimerSet(&timers);
  pb_FreeParameters(parameters);

  return 0;
}
