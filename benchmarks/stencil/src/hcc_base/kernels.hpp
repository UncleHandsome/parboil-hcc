/***************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

#include "common.h"


void naive_kernel(
        tiled_index<3>& tidx,
        float c0, float c1,
        const array_view<float>& A0, const array_view<float>& Anext,
        int nx, int ny, int nz) [[hc]]
{
    int i = tidx.local[0];
    int j = tidx.tile[0] + 1;
    int k = tidx.tile[1] + 1;
	if(i>0)
	{
    Anext[Index3D (nx, ny, i, j, k)] =
	(A0[Index3D (nx, ny, i, j, k + 1)] +
	A0[Index3D (nx, ny, i, j, k - 1)] +
	A0[Index3D (nx, ny, i, j + 1, k)] +
	A0[Index3D (nx, ny, i, j - 1, k)] +
	A0[Index3D (nx, ny, i + 1, j, k)] +
	A0[Index3D (nx, ny, i - 1, j, k)])*c1
	- A0[Index3D (nx, ny, i, j, k)]*c0;
	}
}



