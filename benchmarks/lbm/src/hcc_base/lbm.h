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

void LBM_allocateGrid( float** ptr );
void LBM_freeGrid( float** ptr );
void LBM_initializeGrid( float* grid );
void LBM_initializeSpecialCellsForLDC( float* grid );
void LBM_loadObstacleFile( float* grid, const char* filename );
void LBM_swapGrids( array_view<float>** grid1, array_view<float>** grid2 );
void LBM_showGridStatistics( float* Grid );
void LBM_storeVelocityField( float* grid, const char* filename,
                           const BOOL binary );

/* HCC *********************************************************************/

void HCC_LBM_allocateGrid(array_view<float>** ptr );
void HCC_LBM_freeGrid( array_view<float>* ptr );
void HCC_LBM_initializeGrid( array_view<float>* d_grid, float* h_grid );
void HCC_LBM_getDeviceGrid( array_view<float>* d_grid, float* h_grid );
void HCC_LBM_performStreamCollide( array_view<float>& srcGrid, array_view<float>& dstGrid );

/*############################################################################*/

#endif /* _LBM_H_ */
