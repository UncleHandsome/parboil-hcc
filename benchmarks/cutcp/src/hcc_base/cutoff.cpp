/***************************************************************************
 *cr
 *cr            (C) Copyright 2008-2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <hc.hpp>
#include <hc_math.hpp>
#include "atom.h"
#include "cutoff.h"
#include "parboil.h"

using namespace hc;

#ifdef __DEVICE_EMULATION__
#define DEBUG
/* define which grid block and which thread to examine */
#define BX  0
#define BY  0
#define TX  0
#define TY  0
#define TZ  0
#define EMU(code) do { \
  if (bx==BX && by==BY && \
      tx==TX && ty==TY && tz==TZ) { \
    code; \
  } \
} while (0)
#define INT(n)    printf("%s = %d\n", #n, n)
#define FLOAT(f)  printf("%s = %g\n", #f, (double)(f))
#define INT3(n)   printf("%s = %d %d %d\n", #n, (n).x, (n).y, (n).z)
#define FLOAT4(f) printf("%s = %g %g %g %g\n", #f, (double)(f).x, \
    (double)(f).y, (double)(f).z, (double)(f).w)
#else
#define EMU(code)
#define INT(n)
#define FLOAT(f)
#define INT3(n)
#define FLOAT4(f)
#endif

typedef struct _int3 { int x, y, z; } int3;
typedef struct _float4 { float x, y, z, w; } float4;


/*
 * neighbor list:
 * stored in constant memory as table of offsets
 * flat index addressing is computed by kernel
 *
 * reserve enough memory for 11^3 stencil of grid cells
 * this fits within 16K of memory
 */
#define NBRLIST_DIM  11
#define NBRLIST_MAXLEN (NBRLIST_DIM * NBRLIST_DIM * NBRLIST_DIM)

/*
 * atom bins cached into shared memory for processing
 *
 * this reserves 4K of shared memory for 32 atom bins each containing 8 atoms,
 * should permit scheduling of up to 3 thread blocks per SM
 */
#define BIN_DEPTH         8  /* max number of atoms per bin */
#define BIN_SIZE         32  /* size of bin in floats */
#define BIN_CACHE_MAXLEN 32  /* max number of atom bins to cache */

#define BIN_LENGTH      4.f  /* spatial length in Angstroms */
#define BIN_INVLEN  (1.f / BIN_LENGTH)
/* assuming density of 1 atom / 10 A^3, expectation is 6.4 atoms per bin
 * so that bin fill should be 80% (for non-empty regions of space) */

#define REGION_SIZE     512  /* number of floats in lattice region */
#define SUB_REGION_SIZE 128  /* number of floats in lattice sub-region */

/*
 * potential lattice is decomposed into size 8^3 lattice point "regions"
 *
 * THIS IMPLEMENTATION:  one thread per lattice point
 * thread block size 128 gives 4 thread blocks per region
 * kernel is invoked for each x-y plane of regions,
 * where gx is 4*(x region dimension) so that bx
 * can absorb the z sub-region index in its 2 lowest order bits
 *
 * Regions are stored contiguously in memory in row-major order
 *
 * The bins have to not only cover the region, but they need to surround
 * the outer edges so that region sides and corners can still use
 * neighbor list stencil.  The binZeroAddr is actually a shifted pointer into
 * the bin array (binZeroAddr = binBaseAddr + (c*binDim_y + c)*binDim_x + c)
 * where c = ceil(cutoff / binsize).  This allows for negative offsets to
 * be added to myBinIndex.
 *
 * The (0,0,0) spatial origin corresponds to lower left corner of both
 * regionZeroAddr and binZeroAddr.  The atom coordinates are translated
 * during binning to enforce this assumption.
 */
static void hcc_cutoff_potential_lattice(
    tiled_index<3> tidx,
    int binDim_x,
    int binDim_y,
    const array_view<float4>& binZeroAddr,    /* address of atom bins starting at origin */
    float h,                /* lattice spacing */
    float cutoff2,          /* square of cutoff distance */
    float inv_cutoff2,
    const array_view<float>& regionZeroAddr,/* address of lattice regions starting at origin */
    int zRegionIndex,
    const array_view<const int3>& NbrList
    ) [[hc]]
{
  tile_static float AtomBinCache[BIN_CACHE_MAXLEN * BIN_DEPTH * 4];
  tile_static int3 myBinIndex;

  int tz = tidx.local[2], ty = tidx.local[1], tx = tidx.local[0];
  int bz = tidx.tile[2], by = tidx.tile[1], bx = tidx.tile[0];
  int gz = tidx.tile_dim[2], gy = tidx.tile_dim[1], gx = tidx.tile_dim[0];

  //const int xRegionIndex = (bx >> 2);
  //const int yRegionIndex = by;

  /* thread id */
  const int tid = (tz*8 + ty)*8 + tx;

  /* neighbor index */
  int nbrid;

  /* spatial coordinate of this lattice point */
  float x = (8 * (bx >> 2) + tx) * h;
  float y = (8 * by + ty) * h;
  float z = (8 * zRegionIndex + 2*(bx&3) + tz) * h;

  int totalbins = 0;
  int numbins;

  /* bin number determined by center of region */
  myBinIndex.x = (int) floorf((8 * (bx >> 2) + 4) * h * BIN_INVLEN);
  myBinIndex.y = (int) floorf((8 * by + 4) * h * BIN_INVLEN);
  myBinIndex.z = (int) floorf((8 * zRegionIndex + 4) * h * BIN_INVLEN);

  /* first neighbor in list for me to cache */
  nbrid = (tid >> 4);

  numbins = BIN_CACHE_MAXLEN;

  float energy = 0.f;
  int NbrListLen = NbrList.get_extent()[0];
  for (totalbins = 0;  totalbins < NbrListLen;  totalbins += numbins) {
    int bincnt;

    /* start of where to write in shared memory */
    int startoff = BIN_SIZE * (tid >> 4);

    /* each half-warp to cache up to 4 atom bins */
    for (bincnt = 0;  bincnt < 4 && nbrid < NbrListLen;  bincnt++, nbrid += 8) {
      int i = myBinIndex.x + NbrList[nbrid].x;
      int j = myBinIndex.y + NbrList[nbrid].y;
      int k = myBinIndex.z + NbrList[nbrid].z;

      /* determine global memory location of atom bin */
      array_view<float> p_global =
          binZeroAddr.reinterpret_as<float>().section(index<1>((((k*binDim_y) + j)*binDim_x + i) * BIN_SIZE));

      /* coalesced read from global memory -
       * retain same ordering in shared memory for now */
      int tidmask = tid & 15;
      int binIndex = startoff + bincnt*8*BIN_SIZE;

      AtomBinCache[binIndex + tidmask   ] = p_global[tidmask   ];
      AtomBinCache[binIndex + tidmask+16] = p_global[tidmask+16];
    }
    tidx.barrier.wait();

    /* no warp divergence */
    if (totalbins + BIN_CACHE_MAXLEN > NbrListLen) {
      numbins = NbrListLen - totalbins;
    }

    for (bincnt = 0;  bincnt < numbins;  bincnt++) {
      int i;
      float r2;

      for (i = 0;  i < BIN_DEPTH;  i++) {
        float ax = AtomBinCache[bincnt * BIN_SIZE + i*4];
        float ay = AtomBinCache[bincnt * BIN_SIZE + i*4 + 1];
        float az = AtomBinCache[bincnt * BIN_SIZE + i*4 + 2];
        float aq = AtomBinCache[bincnt * BIN_SIZE + i*4 + 3];
        if (0.f == aq) break;  /* no more atoms in bin */
        r2 = (ax - x) * (ax - x) + (ay - y) * (ay - y) + (az - z) * (az - z);
        if (r2 < cutoff2) {
          float s = (1.f - r2 * inv_cutoff2);
          energy += aq * rsqrtf(r2) * s * s;
        }
      } /* end loop over atoms in bin */
    } /* end loop over cached atom bins */
    tidx.barrier.wait();

  } /* end loop over neighbor list */

  /* store into global memory */
  /* this is the start of the sub-region indexed by tid */
  int mySubRegionAddr = ((zRegionIndex*gy
        + by)*(gx>>2) + (bx >> 2))*REGION_SIZE
        + (bx&3)*SUB_REGION_SIZE;
  regionZeroAddr[mySubRegionAddr + tid] = energy;
}




extern "C" int gpu_compute_cutoff_potential_lattice(
    struct pb_TimerSet *timers,
    Lattice *lattice,                  /* the lattice */
    float cutoff,                      /* cutoff distance */
    Atoms *atoms,                      /* array of atoms */
    int verbose                        /* print info/debug messages */
    )
{
  int nx = lattice->dim.nx;
  int ny = lattice->dim.ny;
  int nz = lattice->dim.nz;
  float xlo = lattice->dim.lo.x;
  float ylo = lattice->dim.lo.y;
  float zlo = lattice->dim.lo.z;
  float h = lattice->dim.h;
  int natoms = atoms->size;
  Atom *atom = atoms->atoms;

  int3 nbrlist[NBRLIST_MAXLEN];
  int nbrlistlen = 0;

  int binHistoFull[BIN_DEPTH+1] = { 0 };   /* clear every array element */
  int binHistoCover[BIN_DEPTH+1] = { 0 };  /* clear every array element */
  int num_excluded = 0;

  int xRegionDim, yRegionDim, zRegionDim;
  int xRegionIndex, yRegionIndex, zRegionIndex;
  int xOffset, yOffset, zOffset;
  int lnx, lny, lnz, lnall;
  float *regionZeroAddr, *thisRegion;
  int index, indexRegion;

  int c;
  int3 binDim;
  int nbins;
  float4 *binBaseAddr, *binZeroAddr;
  int *bincntBaseAddr, *bincntZeroAddr;
  Atoms *extra = NULL;

  int i, j, k, n;
  int sum, total;

  float avgFillFull, avgFillCover;
  const float cutoff2 = cutoff * cutoff;
  const float inv_cutoff2 = 1.f / cutoff2;

  int3 g, b;

  // Caller has made the 'compute' timer active

  /* pad lattice to be factor of 8 in each dimension */
  xRegionDim = (int) ceilf(nx/8.f);
  yRegionDim = (int) ceilf(ny/8.f);
  zRegionDim = (int) ceilf(nz/8.f);

  lnx = 8 * xRegionDim;
  lny = 8 * yRegionDim;
  lnz = 8 * zRegionDim;
  lnall = lnx * lny * lnz;

  /* will receive energies from HCC */
  regionZeroAddr = (float *) malloc(lnall * sizeof(float));

  /* create bins */
  c = (int) ceil(cutoff * BIN_INVLEN);  /* count extra bins around lattice */
  binDim.x = (int) ceil(lnx * h * BIN_INVLEN) + 2*c;
  binDim.y = (int) ceil(lny * h * BIN_INVLEN) + 2*c;
  binDim.z = (int) ceil(lnz * h * BIN_INVLEN) + 2*c;
  nbins = binDim.x * binDim.y * binDim.z;
  binBaseAddr = (float4 *) calloc(nbins * BIN_DEPTH, sizeof(float4));
  binZeroAddr = binBaseAddr + ((c * binDim.y + c) * binDim.x + c) * BIN_DEPTH;

  bincntBaseAddr = (int *) calloc(nbins, sizeof(int));
  bincntZeroAddr = bincntBaseAddr + (c * binDim.y + c) * binDim.x + c;

  /* create neighbor list */
  if (ceilf(BIN_LENGTH / (8*h)) == floorf(BIN_LENGTH / (8*h))) {
    float s = sqrtf(3);
    float r2 = (cutoff + s*BIN_LENGTH) * (cutoff + s*BIN_LENGTH);
    int cnt = 0;
    /* develop neighbor list around 1 cell */
    if (2*c + 1 > NBRLIST_DIM) {
      fprintf(stderr, "must have cutoff <= %f\n",
          (NBRLIST_DIM-1)/2 * BIN_LENGTH);
      return -1;
    }
    for (k = -c;  k <= c;  k++) {
      for (j = -c;  j <= c;  j++) {
        for (i = -c;  i <= c;  i++) {
          if ((i*i + j*j + k*k)*BIN_LENGTH*BIN_LENGTH >= r2) continue;
          nbrlist[cnt].x = i;
          nbrlist[cnt].y = j;
          nbrlist[cnt].z = k;
          cnt++;
        }
      }
    }
    nbrlistlen = cnt;
  }
  else if (8*h <= 2*BIN_LENGTH) {
    float s = 2.f*sqrtf(3);
    float r2 = (cutoff + s*BIN_LENGTH) * (cutoff + s*BIN_LENGTH);
    int cnt = 0;
    /* develop neighbor list around 3-cube of cells */
    if (2*c + 3 > NBRLIST_DIM) {
      fprintf(stderr, "must have cutoff <= %f\n",
          (NBRLIST_DIM-3)/2 * BIN_LENGTH);
      return -1;
    }
    for (k = -c;  k <= c;  k++) {
      for (j = -c;  j <= c;  j++) {
        for (i = -c;  i <= c;  i++) {
          if ((i*i + j*j + k*k)*BIN_LENGTH*BIN_LENGTH >= r2) continue;
          nbrlist[cnt].x = i;
          nbrlist[cnt].y = j;
          nbrlist[cnt].z = k;
          cnt++;
        }
      }
    }
    nbrlistlen = cnt;
  }
  else {
    fprintf(stderr, "must have h <= %f\n", 0.25 * BIN_LENGTH);
    return -1;
  }

  /* perform geometric hashing of atoms into bins */
  {
    /* array of extra atoms, permit average of one extra per bin */
    Atom *extra_atoms = (Atom *) calloc(nbins, sizeof(Atom));
    int extra_len = 0;

    for (n = 0;  n < natoms;  n++) {
      float4 p;
      p.x = atom[n].x - xlo;
      p.y = atom[n].y - ylo;
      p.z = atom[n].z - zlo;
      p.w = atom[n].q;
      i = (int) floorf(p.x * BIN_INVLEN);
      j = (int) floorf(p.y * BIN_INVLEN);
      k = (int) floorf(p.z * BIN_INVLEN);
      if (i >= -c && i < binDim.x - c &&
	  j >= -c && j < binDim.y - c &&
	  k >= -c && k < binDim.z - c &&
	  atom[n].q != 0) {
	int index = (k * binDim.y + j) * binDim.x + i;
	float4 *bin = binZeroAddr + index * BIN_DEPTH;
	int bindex = bincntZeroAddr[index];
	if (bindex < BIN_DEPTH) {
	  /* copy atom into bin and increase counter for this bin */
	  bin[bindex] = p;
	  bincntZeroAddr[index]++;
	}
	else {
	  /* add index to array of extra atoms to be computed with CPU */
	  if (extra_len >= nbins) {
	    fprintf(stderr, "exceeded space for storing extra atoms\n");
	    return -1;
	  }
	  extra_atoms[extra_len] = atom[n];
	  extra_len++;
	}
      }
      else {
	/* excluded atoms are either outside bins or neutrally charged */
	num_excluded++;
      }
    }

    /* Save result */
    extra = (Atoms *)malloc(sizeof(Atoms));
    extra->atoms = extra_atoms;
    extra->size = extra_len;
  }

  /* bin stats */
  sum = total = 0;
  for (n = 0;  n < nbins;  n++) {
    binHistoFull[ bincntBaseAddr[n] ]++;
    sum += bincntBaseAddr[n];
    total += BIN_DEPTH;
  }
  avgFillFull = sum / (float) total;
  sum = total = 0;
  for (k = 0;  k < binDim.z - 2*c;  k++) {
    for (j = 0;  j < binDim.y - 2*c;  j++) {
      for (i = 0;  i < binDim.x - 2*c;  i++) {
        int index = (k * binDim.y + j) * binDim.x + i;
        binHistoCover[ bincntZeroAddr[index] ]++;
        sum += bincntZeroAddr[index];
        total += BIN_DEPTH;
      }
    }
  }
  avgFillCover = sum / (float) total;

  if (verbose) {
    /* report */
    printf("number of atoms = %d\n", natoms);
    printf("lattice spacing = %g\n", h);
    printf("cutoff distance = %g\n", cutoff);
    printf("\n");
    printf("requested lattice dimensions = %d %d %d\n", nx, ny, nz);
    printf("requested space dimensions = %g %g %g\n", nx*h, ny*h, nz*h);
    printf("expanded lattice dimensions = %d %d %d\n", lnx, lny, lnz);
    printf("expanded space dimensions = %g %g %g\n", lnx*h, lny*h, lnz*h);
    printf("number of bytes for lattice data = %u\n", lnall*sizeof(float));
    printf("\n");
    printf("bin padding thickness = %d\n", c);
    printf("bin cover dimensions = %d %d %d\n",
        binDim.x - 2*c, binDim.y - 2*c, binDim.z - 2*c);
    printf("bin full dimensions = %d %d %d\n", binDim.x, binDim.y, binDim.z);
    printf("number of bins = %d\n", nbins);
    printf("total number of atom slots = %d\n", nbins * BIN_DEPTH);
    printf("%% overhead space = %g\n",
        (natoms / (double) (nbins * BIN_DEPTH)) * 100);
    printf("number of bytes for bin data = %u\n",
        nbins * BIN_DEPTH * sizeof(float4));
    printf("\n");
    printf("bin histogram with padding:\n");
    sum = 0;
    for (n = 0;  n <= BIN_DEPTH;  n++) {
      printf("     number of bins with %d atoms:  %d\n", n, binHistoFull[n]);
      sum += binHistoFull[n];
    }
    printf("     total number of bins:  %d\n", sum);
    printf("     %% average fill:  %g\n", avgFillFull * 100);
    printf("\n");
    printf("bin histogram excluding padding:\n");
    sum = 0;
    for (n = 0;  n <= BIN_DEPTH;  n++) {
      printf("     number of bins with %d atoms:  %d\n", n, binHistoCover[n]);
      sum += binHistoCover[n];
    }
    printf("     total number of bins:  %d\n", sum);
    printf("     %% average fill:  %g\n", avgFillCover * 100);
    printf("\n");
    printf("number of extra atoms = %d\n", extra->size);
    printf("%% atoms that are extra = %g\n", (extra->size / (double) natoms) * 100);
    printf("\n");

    /* sanity check on bins */
    sum = 0;
    for (n = 0;  n <= BIN_DEPTH;  n++) {
      sum += n * binHistoFull[n];
    }
    sum += extra->size + num_excluded;
    printf("sanity check on bin histogram with edges:  "
        "sum + others = %d\n", sum);
    sum = 0;
    for (n = 0;  n <= BIN_DEPTH;  n++) {
      sum += n * binHistoCover[n];
    }
    sum += extra->size + num_excluded;
    printf("sanity check on bin histogram excluding edges:  "
        "sum + others = %d\n", sum);
    printf("\n");

    /* neighbor list */
    printf("neighbor list length = %d\n", nbrlistlen);
    printf("\n");
  }

  b.x = 8;
  b.y = 8;
  b.z = 2;
  g.x = 4 * xRegionDim * b.x;
  g.y = yRegionDim * b.y;
  g.z = 1 * b.z;
  tiled_extent<3> text = extent<3>(g.x, g.y, g.z).tile(b.x, b.y, b.z);

  /* allocate and initialize memory on HCC device */
  pb_SwitchToTimer(timers, pb_TimerID_COPY);
  if (verbose) {
    printf("Allocating %.2fMB on HCC device for potentials\n",
           lnall * sizeof(float) / (double) (1024*1024));
  }
  std::vector<float> l(lnall);
  array_view<float> regionZeroHcc(lnall, l);
  if (verbose) {
    printf("Allocating %.2fMB on HCC device for atom bins\n",
           nbins * BIN_DEPTH * sizeof(float4) / (double) (1024*1024));
  }
  int len = ((c * binDim.y + c) * binDim.x + c) * BIN_DEPTH;
  array_view<float4> binZeroHcc(nbins * BIN_DEPTH - len, binBaseAddr+len);
  array_view<const int3> NbrList(nbrlistlen, nbrlist);

  if (verbose)
    printf("\n");

  /* loop over z-dimension, invoke HCC kernel for each x-y plane */
  pb_SwitchToTimer(timers, pb_TimerID_KERNEL);
  printf("Invoking HCC kernel on %d region planes...\n", zRegionDim);
  for (zRegionIndex = 0;  zRegionIndex < zRegionDim;  zRegionIndex++) {
    printf("  computing plane %d\r", zRegionIndex);
    fflush(stdout);
    parallel_for_each(text, [=] (tiled_index<3> tidx) [[hc]] {
            hcc_cutoff_potential_lattice(tidx, binDim.x, binDim.y, binZeroHcc, h,
                cutoff2, inv_cutoff2, regionZeroHcc, zRegionIndex, NbrList);
            }
            ).wait();
  }
  printf("Finished HCC kernel calls                        \n");

  /* copy result regions from HCC device */
  pb_SwitchToTimer(timers, pb_TimerID_COPY);
  regionZeroHcc.synchronize();

  /* transpose regions back into lattice */
  pb_SwitchToTimer(timers, pb_TimerID_COMPUTE);
  for (k = 0;  k < nz;  k++) {
    zRegionIndex = (k >> 3);
    zOffset = (k & 7);

    for (j = 0;  j < ny;  j++) {
      yRegionIndex = (j >> 3);
      yOffset = (j & 7);

      for (i = 0;  i < nx;  i++) {
        xRegionIndex = (i >> 3);
        xOffset = (i & 7);

        thisRegion = regionZeroAddr
          + ((zRegionIndex * yRegionDim + yRegionIndex) * xRegionDim
              + xRegionIndex) * REGION_SIZE;

        indexRegion = (zOffset * 8 + yOffset) * 8 + xOffset;
        index = (k * ny + j) * nx + i;

        lattice->lattice[index] = thisRegion[indexRegion];
      }
    }
  }

  /* handle extra atoms */
  if (extra->size > 0) {
    printf("computing extra atoms on CPU\n");
    if (cpu_compute_cutoff_potential_lattice(lattice, cutoff, extra)) {
      fprintf(stderr, "cpu_compute_cutoff_potential_lattice() failed "
          "for extra atoms\n");
      return -1;
    }
    printf("\n");
  }

  /* cleanup memory allocations */
  free(regionZeroAddr);
  free(binBaseAddr);
  free(bincntBaseAddr);
  free_atom(extra);

  return 0;
}
