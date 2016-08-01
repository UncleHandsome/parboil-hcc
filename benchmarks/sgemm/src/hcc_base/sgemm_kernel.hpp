/***************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

/*
 * Kernel of dense matrix-matrix multiplication kernel.
 */


// Parameters of tile sizes
#define TILE_SZ 16

void mysgemmNT(
        tiled_index<2>& tidx,
        const array_view<const float>& A, int lda,
        const array_view<const float>& B, int ldb,
        const array_view<float>& C, int ldc,
        int k, float alpha, float beta ) [[hc]]
{
    float c = 0.0f;
    int m = tidx.global[0];
    int n = tidx.global[1];
    for (int i = 0; i < k; ++i) {
	float a = A[m + i * lda];
	float b = B[n + i * ldb];
	c += a * b;
    }
    C[m+n*ldc] = C[m+n*ldc] * beta + alpha * c;
}

void basicSgemm( char transa, char transb, int m, int n, int k, float alpha,
        array_view<const float>& A, int lda,
        array_view<const float>& B, int ldb, float beta,
        array_view<float>& C, int ldc )
{
  if ((transa != 'N') && (transa != 'n')) {
    std::cerr << "unsupported value of 'transa' in regtileSgemm()" << std::endl;
    return;
  }

  if ((transb != 'T') && (transb != 't')) {
    std::cerr << "unsupported value of 'transb' in regtileSgemm()" << std::endl;
    return;
  }

  // In this code we assume the matrix sizes are multiple of tile size
  if ((m%TILE_SZ) || (n%TILE_SZ)) {
    std::cerr << "unsupported size of matrix. m should be multiple of " << TILE_SZ
      << "; n should be multiple of " << TILE_SZ << std::endl;
  }


  int threads[2] = {TILE_SZ, TILE_SZ};
  int grid[2] = {m/TILE_SZ*threads[0], n/TILE_SZ*threads[1]};
  parallel_for_each(extent<2>(grid).tile(threads[0], threads[1]),
          [=] (tiled_index<2> tidx) [[hc]]
          {
          mysgemmNT(tidx, A, lda, B, ldb, C, ldc, k, alpha, beta);
          });
}

