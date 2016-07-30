/***********************************************************************************
  Implementing Breadth first search on CUDA using algorithm given in DAC'10
  paper "An Effective GPU Implementation of Breadth-First Search"

  Copyright (c) 2010 University of Illinois at Urbana-Champaign.
  All rights reserved.

  Permission to use, copy, modify and distribute this software and its documentation for
  educational purpose is hereby granted without fee, provided that the above copyright
  notice and this permission notice appear in all copies of this software and that you do
  not sell the software.

  THE SOFTWARE IS PROVIDED "AS IS" AND WITHOUT WARRANTY OF ANY KIND,EXPRESS, IMPLIED OR
  OTHERWISE.

  Author: Lijiuan Luo (lluo3@uiuc.edu)
  Revised for Parboil 2 Benchmark Suite by: Geng Daniel Liu (gengliu2@illinois.edu)
 ************************************************************************************/
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <parboil.h>
#include <deque>
#include <iostream>
#include <hc.hpp>

#include "config.h"
FILE *fp;
struct Node {
  int x;
  int y;
};
struct Edge {
  int x;
  int y;
};

using namespace hc;

#include "kernel.hpp"
//Somehow "cudaMemset" does not work. So I use cudaMemcpy of constant variables for initialization
const int h_top = 1;
const int zero = 0;

////////////////////////////////////////////////////////////////////////////////
// Main Program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
  //the number of nodes in the graph
  int num_of_nodes = 0;
  //the number of edges in the graph
  int num_of_edges = 0;
  struct pb_Parameters *params;
  struct pb_TimerSet timers;

  pb_InitializeTimerSet(&timers);
  params = pb_ReadParameters(&argc, argv);
  if ((params->inpFiles[0] == NULL) || (params->inpFiles[1] != NULL))
  {
    fprintf(stderr, "Expecting one input filename\n");
    exit(-1);
  }

  pb_SwitchToTimer(&timers, pb_TimerID_IO);
  //printf("Reading File\n");
  //Read in Graph from a file
  fp = fopen(params->inpFiles[0],"r");
  if(!fp)
  {
    printf("Error Reading graph file\n");
    return 0;
  }
  int source;

  fscanf(fp,"%d",&num_of_nodes);
  // allocate host memory
  Node* h_graph_nodes = (Node*) malloc(sizeof(Node)*num_of_nodes);
  int *color = (int*) malloc(sizeof(int)*num_of_nodes);
  int start, edgeno;
  // initalize the memory
  for( unsigned int i = 0; i < num_of_nodes; i++)
  {
    fscanf(fp,"%d %d",&start,&edgeno);
    h_graph_nodes[i].x = start;
    h_graph_nodes[i].y = edgeno;
    color[i]=WHITE;
  }
  //read the source node from the file
  fscanf(fp,"%d",&source);
  fscanf(fp,"%d",&num_of_edges);
  int id,cost;
  Edge* h_graph_edges = (Edge*) malloc(sizeof(Edge)*num_of_edges);
  for(int i=0; i < num_of_edges ; i++)
  {
    fscanf(fp,"%d",&id);
    fscanf(fp,"%d",&cost);
    h_graph_edges[i].x = id;
    h_graph_edges[i].y = cost;
  }
  if(fp)
    fclose(fp);

  // allocate mem for the result on host side
  int* h_cost = (int*) malloc( sizeof(int)*num_of_nodes);
  for(int i = 0; i < num_of_nodes; i++){
    h_cost[i] = INF;
  }
  h_cost[source] = 0;

  pb_SwitchToTimer(&timers, pb_TimerID_COPY);

  //Copy the Node list to device memory
  array_view<const Node> d_graph_nodes(num_of_nodes, h_graph_nodes);
  array_view<const Edge> d_graph_edges(num_of_edges, h_graph_edges);

  array_view<int> d_color(num_of_nodes, color);
  array_view<int> d_cost(num_of_nodes, h_cost);
  array_view<int> d_q1(num_of_nodes);
  array_view<int> d_q2(num_of_nodes);
  array_view<int> tail(1);
  array_view<int> front_cost_d(1);

  printf("Starting GPU kernel\n");
  pb_SwitchToTimer(&timers, pb_TimerID_KERNEL);

  int num_of_blocks;
  int num_of_threads_per_block;

  tail[0] = h_top;
  d_cost[source] = zero;
  d_q1[0] = source;

  int num_t;//number of threads
  int k=0;//BFS level index

  do
  {
    num_t = tail[0];
    tail[0] = zero;

    if(num_t == 0){//frontier is empty
      break;
    }

    num_of_blocks = 1;
    num_of_threads_per_block = num_t;
    if(num_of_threads_per_block <NUM_BIN)
      num_of_threads_per_block = NUM_BIN;
    if(num_t>MAX_THREADS_PER_BLOCK)
    {
      num_of_blocks = (int)ceil(num_t/(double)MAX_THREADS_PER_BLOCK);
      num_of_threads_per_block = MAX_THREADS_PER_BLOCK;
    }
    if(num_of_blocks == 1)//will call "BFS_in_GPU_kernel"
      num_of_threads_per_block = MAX_THREADS_PER_BLOCK;
    if(num_of_blocks >1 && num_of_blocks <= NUM_SM)// will call "BFS_kernel_multi_blk_inGPU"
      num_of_blocks = NUM_SM;

    extent<1>  grid(num_of_blocks * num_of_threads_per_block);
    tiled_extent<1>  tile = grid.tile(num_of_threads_per_block);

    if(k%2 == 0){
        parallel_for_each(tile, [=] (tiled_index<1> tidx) [[hc]]
                {
                BFS_kernel(tidx, d_q1,d_q2, d_graph_nodes,
                    d_graph_edges, d_color, d_cost, num_t, tail,GRAY0,k);
                });
    }
    else{
        parallel_for_each(tile, [=] (tiled_index<1> tidx) [[hc]]
                {
                BFS_kernel(tidx, d_q2,d_q1, d_graph_nodes,
                    d_graph_edges, d_color, d_cost, num_t, tail,GRAY1,k);
                });
    }
    k++;
  }
  while(1);
  pb_SwitchToTimer(&timers, pb_TimerID_COPY);
  printf("GPU kernel done\n");

  // copy result from device to host
  d_cost.synchronize();
  d_color.synchronize();

  //Store the result into a file
  pb_SwitchToTimer(&timers, pb_TimerID_IO);
  FILE *fp = fopen(params->outFile,"w");
  fprintf(fp, "%d\n", num_of_nodes);
  for(int i=0;i<num_of_nodes;i++)
    fprintf(fp,"%d %d\n",i,h_cost[i]);
  fclose(fp);

  // cleanup memory
  free( h_graph_nodes);
  free( h_graph_edges);
  free( color);
  free( h_cost);
  pb_SwitchToTimer(&timers, pb_TimerID_NONE);
  pb_PrintTimerSet(&timers);
  pb_FreeParameters(params);
  return 0;
}
