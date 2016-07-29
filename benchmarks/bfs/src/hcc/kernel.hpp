/***************************************************************************
 *cr
 *cr            (C) Copyright 2007 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/
/*
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
*/
#ifndef _KERNEL_H_
#define _KERNEL_H_
/*
Define colors for BFS
1) the definition of White, gray and black comes from the text book "Introduction to Algorithms"
2) For path search problems, people may choose to use different colors to record the found paths.
Therefore we reserve numbers (0-16677216) for this purpose. Only nodes with colors bigger than
UP_LIMIT are free to visit
3) We define two gray shades to differentiate between the new frontier nodes and the old frontier nodes that
 have not been marked BLACK
*/

#include "config.h"

// A group of local queues of node IDs, used by an entire thread block.
// Multiple queues are used to reduce memory contention.
// Thread i uses queue number (i % NUM_BIN).
struct LocalQueues {
  // tail[n] is the index of the first empty array in elems[n]
  int tail[NUM_BIN];

  // Queue elements.
  // The contents of queue n are elems[n][0 .. tail[n] - 1].
  int elems[NUM_BIN][W_QUEUE_SIZE];

  // The number of threads sharing queue n.  We use this number to
  // compute a reduction over the queue.
  int sharers[NUM_BIN];

  // Initialize or reset the queue at index 'index'.
  // Normally run in parallel for all indices.
  void reset(int index, int block_dim) [[hc]] {
    tail[index] = 0;		// Queue contains nothing

    // Number of sharers is (threads per block / number of queues)
    // If division is not exact, assign the leftover threads to the first
    // few queues.
    sharers[index] =
      (block_dim >> EXP) +
      (index < (block_dim & MOD_OP));
  }

  // Append 'value' to queue number 'index'.  If queue is full, the
  // append operation fails and *overflow is set to 1.
  void append(int index, array_view<int>& overflow, int value) [[hc]] {
    // Queue may be accessed concurrently, so
    // use an atomic operation to reserve a queue index.
    int tail_index = atomic_fetch_add(&tail[index], 1);
    if (tail_index >= W_QUEUE_SIZE)
      overflow[0] = 1;
    else
      elems[index][tail_index] = value;
  }

  // Perform a scan on the number of elements in queues in a a LocalQueue.
  // This function should be executed by one thread in a thread block.
  //
  // The result of the scan is used to concatenate all queues; see
  // 'concatenate'.
  //
  // The array prefix_q will hold the scan result on output:
  // [0, tail[0], tail[0] + tail[1], ...]
  //
  // The total number of elements is returned.
  int size_prefix_sum(int (&prefix_q)[NUM_BIN]) [[hc]] {
    prefix_q[0] = 0;
    for(int i = 1; i < NUM_BIN; i++){
      prefix_q[i] = prefix_q[i-1] + tail[i-1];
    }
    return prefix_q[NUM_BIN-1] + tail[NUM_BIN-1];
  }

  // Concatenate and copy all queues to the destination.
  // This function should be executed by all threads in a thread block.
  //
  // prefix_q should contain the result of 'size_prefix_sum'.
  void concatenate(int *dst, int (&prefix_q)[NUM_BIN], int threadId) [[hc]] {
    // Thread n processes elems[n % NUM_BIN][n / NUM_BIN, ...]
    int q_i = threadId & MOD_OP; // w-queue index
    int local_shift = threadId >> EXP; // shift within a w-queue

    while(local_shift < tail[q_i]){
      dst[prefix_q[q_i] + local_shift] = elems[q_i][local_shift];

      //multiple threads are copying elements at the same time,
      //so we shift by multiple elements for next iteration
      local_shift += sharers[q_i];
    }
  }

  void concatenate(array_view<int>& dst, int (&prefix_q)[NUM_BIN], int threadId) [[hc]] {
    // Thread n processes elems[n % NUM_BIN][n / NUM_BIN, ...]
    int q_i = threadId & MOD_OP; // w-queue index
    int local_shift = threadId >> EXP; // shift within a w-queue

    while(local_shift < tail[q_i]){
      dst[prefix_q[q_i] + local_shift] = elems[q_i][local_shift];

      //multiple threads are copying elements at the same time,
      //so we shift by multiple elements for next iteration
      local_shift += sharers[q_i];
    }
  }
};

//Inter-block sychronization
//This only works when there is only one block per SM
void start_global_barrier(int fold, int* count, tiled_index<1>& tidx) [[hc]] {
  tidx.barrier.wait();

  if(tidx.tile[0] == 0){
    atomic_fetch_add(count, 1);
    int count_val = atomic_fetch_or(count, 0);
    while( count_val < NUM_SM*fold){
        count_val = atomic_fetch_or(count, 0);
    }
  }
  tidx.barrier.wait();

}

// Process a single graph node from the active frontier.  Mark nodes and
// put new frontier nodes on the queue.
//
// 'pid' is the ID of the node to process.
// 'index' is the local queue to use, chosen based on the thread ID.
// The output goes in 'local_q' and 'overflow'.
// Other parameters are inputs.
void
visit_node(int pid,
	   int index,
	   LocalQueues &local_q,
       array_view<Node>& g_graph_node,
       array_view<Edge>& g_graph_edge,
	   array_view<int>& overflow,
	   array_view<int>& g_color,
	   array_view<int>& g_cost,
	   int gray_shade) [[hc]]
{
  g_color[pid] = BLACK;		// Mark this node as visited
  int cur_cost = g_cost[pid];	// Look up shortest-path distance to this node
  Node cur_node = g_graph_node[pid];

  // For each outgoing edge
  for(int i = cur_node.x; i < cur_node.y + cur_node.x; i++) {
    Edge cur_edge = g_graph_edge[i];
    int id = cur_edge.x;
    int cost = cur_edge.y;
    cost += cur_cost;
    int orig_cost = atomic_fetch_min(&g_cost[id],cost);

    // If this outgoing edge makes a shorter path than any previously
    // discovered path
    if(orig_cost > cost){
      int old_color = atomic_exchange(&g_color[id],gray_shade);
      if(old_color != gray_shade) {
	//push to the queue
	local_q.append(index, overflow, id);
      }
    }
  }
}

//-------------------------------------------------
//This is the version for one-block situation. The propagation idea is basically the same as
//BFS_kernel.
//The major differences are:
// 1) This kernel can propagate though multiple BFS levels (while loop) using __synchThreads() between levels
// 2) the intermediate queues are stored in shared memory (next_wf)
//\param q1: the current frontier queue when the kernel is launched
//\param q2: the new frontier queue when the  kernel returns
//--------------------------------------------------
void
BFS_in_GPU_kernel(tiled_index<1>& tidx,
                  array_view<int>& q1,
                  array_view<int>& q2,
                  array_view<Node>& g_graph_nodes,
                  array_view<Edge>& g_graph_edges,
                  array_view<int>& g_color,
                  array_view<int>& g_cost,
                  int no_of_nodes,
                  array_view<int>& tail,
                  int gray_shade,
                  int k,
                  array_view<int>& overflow) [[hc]]
{
  tile_static LocalQueues local_q;
  tile_static int prefix_q[NUM_BIN];
  tile_static int next_wf[MAX_THREADS_PER_BLOCK];
  tile_static int  tot_sum;
  int threadId = tidx.local[0];
  int blockId = tidx.tile[0];
  if(threadId == 0)	
    tot_sum = 0;//total number of new frontier nodes
  while(1){//propage through multiple BFS levels until the wavfront overgrows one-block limit
    if(threadId < NUM_BIN){
      local_q.reset(threadId, tidx.tile_dim[0]);
    }
    tidx.barrier.wait();
    int tid = blockId*MAX_THREADS_PER_BLOCK + threadId;
    if( tid<no_of_nodes)
    {
      int pid;
      if(tot_sum == 0)//this is the first BFS level of current kernel call
        pid = q1[tid];
      else
        pid = next_wf[tid];//read the current frontier info from last level's propagation

      // Visit a node from the current frontier; update costs, colors, and
      // output queue
      visit_node(pid, threadId & MOD_OP, local_q, g_graph_nodes, g_graph_edges,
              overflow, g_color, g_cost, gray_shade);
    }
    tidx.barrier.wait();
    if(threadId == 0){
      tail[0] = tot_sum = local_q.size_prefix_sum(prefix_q);
    }
    tidx.barrier.wait();

    if(tot_sum == 0)//the new frontier becomes empty; BFS is over
      return;
    if (tot_sum > MAX_THREADS_PER_BLOCK) {
        local_q.concatenate(q2, prefix_q, threadId);
        return;
    }
    //the new frontier is still within one-block limit;
    //stay in current kernel
    local_q.concatenate(next_wf, prefix_q, threadId);

    no_of_nodes = tot_sum;
    tidx.barrier.wait();
    if(threadId == 0){
        if(gray_shade == GRAY0)
            gray_shade = GRAY1;
        else
            gray_shade = GRAY0;
    }
  }//while

}	
//----------------------------------------------------------------
//This BFS kernel propagates through multiple levels using global synchronization
//The basic propagation idea is the same as "BFS_kernel"
//The major differences are:
// 1) propagate through multiple levels by using GPU global sync ("start_global_barrier")
// 2) use q1 and q2 alternately for the intermediate queues
//\param q1: the current frontier when the kernel is called
//\param q2: possibly the new frontier when the kernel returns depending on how many levels of propagation
//           has been done in current kernel; the new frontier could also be stored in q1
//\param switch_k: whether or not to adjust the "k" value on the host side
//                Normally on the host side, when "k" is even, q1 is the current frontier; when "k" is
//                odd, q2 is the current frontier; since this kernel can propagate through multiple levels,
//                the k value may need to be adjusted when this kernel returns.
//\param global_kt: the total number of global synchronizations,
//                   or the number of times to call "start_global_barrier"
//--------------------------------------------------------------
void
BFS_kernel_multi_blk_inGPU(tiled_index<1>& tidx,
                           array_view<int>& q1,
                           array_view<int>& q2,
                           array_view<Node>& g_graph_nodes,
                           array_view<Edge>& g_graph_edges,
                           array_view<int>& g_color,
                           array_view<int>& g_cost,
                           array_view<int>& no_of_nodes,
                           array_view<int>& tail,
                           int gray_shade,
                           int k,
                           array_view<int>& switch_k,
                           array_view<int>& max_nodes_per_block,
                           array_view<int>& global_kt,
                           array_view<int>& overflow,
                           array_view<int>& count,
                           array_view<int>& no_of_nodes_vol,
                           array_view<int>& stay_vol) [[hc]]
{
   tile_static LocalQueues local_q;
   tile_static int prefix_q[NUM_BIN];
   tile_static int shift;
   tile_static int no_of_nodes_sm;
   tile_static int odd_time;// the odd level of propagation within current kernel
   int threadId = tidx.local[0];
   int blockId = tidx.tile[0];
   if(threadId == 0){
     odd_time = 1;//true;
     if(blockId == 0)
       no_of_nodes_vol[0] = no_of_nodes[0];
   }
   int kt = atomic_fetch_or(&global_kt[0],0);// the total count of GPU global synchronization
   while (1){//propagate through multiple levels
     if(threadId < NUM_BIN){
       local_q.reset(threadId, tidx.tile_dim[0]);
     }
     if(threadId == 0)
       no_of_nodes_sm = no_of_nodes_vol[0];
     tidx.barrier.wait();

     int tid = blockId*MAX_THREADS_PER_BLOCK + threadId;
     if( tid<no_of_nodes_sm)
     {
       // Read a node ID from the current input queue
         int pid = 0;
       if (odd_time)
           pid = atomic_fetch_or(&q1[tid], 0);
       else
           pid = atomic_fetch_or(&q2[tid], 0);

       // Visit a node from the current frontier; update costs, colors, and
       // output queue
       visit_node(pid, threadId & MOD_OP, local_q, g_graph_nodes, g_graph_edges,
               overflow, g_color, g_cost, gray_shade);
     }
     tidx.barrier.wait();

     // Compute size of the output and allocate space in the global queue
     if(threadId == 0){
       int tot_sum = local_q.size_prefix_sum(prefix_q);
       shift = atomic_fetch_add(&tail[0], tot_sum);
     }
     tidx.barrier.wait();

     // Copy to the current output queue in global memory
     int q_i = threadId & MOD_OP;
     int local_shift = threadId >> EXP;
     while (local_shift < local_q.tail[q_i]) {
         if (odd_time)
             q2[shift+prefix_q[q_i]+local_shift] = local_q.elems[q_i][local_shift];
         else
             q1[shift+prefix_q[q_i]+local_shift] = local_q.elems[q_i][local_shift];
         local_shift += local_q.sharers[q_i];
     }

     if(threadId == 0){
       odd_time = (odd_time+1)%2;
       if(gray_shade == GRAY0)
         gray_shade = GRAY1;
       else
         gray_shade = GRAY0;
     }

     //synchronize among all the blks
     start_global_barrier(kt+1, &count[0], tidx);
     if(blockId == 0 && threadId == 0){
       stay_vol[0] = 0;
       if(tail[0]< NUM_SM*MAX_THREADS_PER_BLOCK && tail[0] > MAX_THREADS_PER_BLOCK){
         stay_vol[0] = 1;
         no_of_nodes_vol[0] = tail[0];
         tail[0] = 0;
       }
     }
     start_global_barrier(kt+2, &count[0], tidx);
     kt+= 2;
     if(stay_vol[0] == 0)
     {
       if(blockId == 0 && threadId == 0)
       {
         global_kt[0] = kt;
         switch_k[0] = (odd_time+1)%2;
         no_of_nodes[0] = no_of_nodes_vol[0];
       }
       return;
     }
   }
}

/*****************************************************************************
  This is the  most general version of BFS kernel, i.e. no assumption about #block in the grid
  \param q1: the array to hold the current frontier
  \param q2: the array to hold the new frontier
  \param g_graph_nodes: the nodes in the input graph
  \param g_graph_edges: the edges i nthe input graph
  \param g_color: the colors of nodes
  \param g_cost: the costs of nodes
  \param no_of_nodes: the number of nodes in the current frontier
  \param tail: pointer to the location of the tail of the new frontier. *tail is the size of the new frontier
  \param gray_shade: the shade of the gray in current BFS propagation. See GRAY0, GRAY1 macro definitions for more details
  \param k: the level of current propagation in the BFS tree. k= 0 for the first propagation.
 ***********************************************************************/
void
BFS_kernel(tiled_index<1>& tidx,
           array_view<int>& q1,
           array_view<int>& q2,
           array_view<Node>& g_graph_nodes,
           array_view<Edge>& g_graph_edges,
           array_view<int>& g_color,
           array_view<int>& g_cost,
           int no_of_nodes,
           array_view<int>& tail,
           int gray_shade,
           int k,
           array_view<int>& overflow) [[hc]]
{
  tile_static LocalQueues local_q;
  tile_static int prefix_q[NUM_BIN];//the number of elementss in the w-queues ahead of
  //current w-queue, a.k.a prefix sum
  tile_static int shift;

  int threadId = tidx.local[0];
  int blockId = tidx.tile[0];
  if(threadId < NUM_BIN){
    local_q.reset(threadId, tidx.tile_dim[0]);
  }
  tidx.barrier.wait();

  //first, propagate and add the new frontier elements into w-queues
  int tid = blockId*MAX_THREADS_PER_BLOCK + threadId;
  if( tid < no_of_nodes)
  {
    // Visit a node from the current frontier; update costs, colors, and
    // output queue
    visit_node(q1[tid], threadId & MOD_OP, local_q, g_graph_nodes, g_graph_edges,
            overflow, g_color, g_cost, gray_shade);
  }
  tidx.barrier.wait();

  // Compute size of the output and allocate space in the global queue
  if(threadId == 0){
    //now calculate the prefix sum
    int tot_sum = local_q.size_prefix_sum(prefix_q);
    //the offset or "shift" of the block-level queue within the
    //grid-level queue is determined by atomic operation
    shift = atomic_fetch_add(&tail[0],tot_sum);
  }
  tidx.barrier.wait();

  //now copy the elements from w-queues into grid-level queues.
  //Note that we have bypassed the copy to/from block-level queues for efficiency reason
  local_q.concatenate(&q2[shift], prefix_q, threadId);
}
#endif
