/***************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

/*############################################################################*/

#ifndef _LBM_H_
#define _LBM_H_

/*############################################################################*/


/*############################################################################*/

typedef enum {C = 0,
              N, S, E, W, T, B,
              NE, NW, SE, SW,
              NT, NB, ST, SB,
              ET, EB, WT, WB,
              FLAGS, N_CELL_ENTRIES} CELL_ENTRIES;
#define N_DISTR_FUNCS FLAGS

typedef enum {OBSTACLE    = 1 << 0,
              ACCEL       = 1 << 1,
              IN_OUT_FLOW = 1 << 2} CELL_FLAGS;


#include <hc.hpp>
#include "layout_config.h"
#include "lbm_macros.h"

using namespace hc;

/*############################################################################*/
#ifdef __cplusplus
extern "C" {
#endif
void LBM_allocateGrid( float** ptr );
void LBM_freeGrid( float** ptr );
void LBM_initializeGrid( LBM_Grid grid );
void LBM_initializeSpecialCellsForLDC( LBM_Grid grid );
void LBM_loadObstacleFile( LBM_Grid grid, const char* filename );
void LBM_swapGrids( array_view<float> *grid1, array_view<float> *grid2 );
void LBM_showGridStatistics( LBM_Grid Grid );
void LBM_storeVelocityField( LBM_Grid grid, const char* filename,
                           const BOOL binary );

/* HCC ***********************************************************************/

void HCC_LBM_allocateGrid( array_view<float>* ptr );
void HCC_LBM_freeGrid( array_view<float>* ptr );
void HCC_LBM_initializeGrid( array_view<float>* d_grid, float** h_grid );
void HCC_LBM_getDeviceGrid( array_view<float>* d_grid, float** h_grid );
void HCC_LBM_performStreamCollide( array_view<float>& srcGrid, array_view<float>& dstGrid );
#ifdef __cplusplus
}
#endif

/*############################################################################*/

#endif /* _LBM_H_ */
