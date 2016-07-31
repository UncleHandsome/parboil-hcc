#include <stdio.h>
#include <stdint.h>

void histo_prescan_kernel (
        tiled_index<1>& tidx,
        const array_view<unsigned int>& input,
        int N, int size,
        const array_view<unsigned int>& minmax) [[hc]]
{
    tile_static float Avg[PRESCAN_THREADS];
    tile_static float StdDev[PRESCAN_THREADS];

    int tx = tidx.tile[0];
    int bx = tidx.local[0];
    int dx = tidx.tile_dim[0];

    int stride = size * dx/N;
    int addr = bx*stride+tx;
    int end = bx*stride + stride/8; // Only sample 1/8th of the input data

    // Compute the average per thread
    float avg = 0.0;
    unsigned int count = 0;
    while (addr < end){
        avg += input[addr];
        count++;
        addr += dx;
    }
    avg /= count;
    Avg[tx] = avg;

    // Compute the standard deviation per thread
    int addr2 = bx*stride+tx;
    float stddev = 0;
    while (addr2 < end){
        stddev += (input[addr2]-avg)*(input[addr2]-avg);
        addr2 += dx;
    }
    stddev /= count;
    StdDev[tx] = sqrtf(stddev);

#define SUM(stride__)\
if(tx < stride__){\
    Avg[tx] += Avg[tx+stride__];\
    StdDev[tx] += StdDev[tx+stride__];\
}

    // Add all the averages and standard deviations from all the threads
    // and take their arithmetic average (as a simplified approximation of the
    // real average and standard deviation.
#if (PRESCAN_THREADS >= 32)
    for (int stride = PRESCAN_THREADS/2; stride >= 32; stride = stride >> 1){
        tidx.barrier.wait();
	SUM(stride);
    }
#endif
#if (PRESCAN_THREADS >= 16)
    SUM(16);
#endif
#if (PRESCAN_THREADS >= 8)
    SUM(8);
#endif
#if (PRESCAN_THREADS >= 4)
    SUM(4);
#endif
#if (PRESCAN_THREADS >= 2)
    SUM(2);
#endif

    if (tx == 0){
        float avg = Avg[0]+Avg[1];
	avg /= PRESCAN_THREADS;
	float stddev = StdDev[0]+StdDev[1];
	stddev /= PRESCAN_THREADS;

        // Take the maximum and minimum range from all the blocks. This will
        // be the final answer. The standard deviation is taken out to 10 sigma
        // away from the average. The value 10 was obtained empirically.
	    atomic_fetch_min(&minmax[0],((unsigned int)(avg-10*stddev))/(KB*1024));
        atomic_fetch_max(&minmax[1],((unsigned int)(avg+10*stddev))/(KB*1024));
    }
}
