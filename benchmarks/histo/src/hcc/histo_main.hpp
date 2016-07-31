#include <stdio.h>

void testIncrementGlobal (
        const array_view<unsigned int>& global_histo,
        unsigned int sm_range_min,
        unsigned int sm_range_max,
        const uchar4 sm) [[hc]]
{
        const unsigned int range = sm.x;
        const unsigned int indexhi = sm.y;
        const unsigned int indexlo = sm.z;
        const unsigned int offset  = sm.w;

        /* Scan for inputs that are outside the central region of histogram */
        if (range < sm_range_min || range > sm_range_max)
        {
                const unsigned int bin = range * BINS_PER_BLOCK + offset / 8 + (indexlo << 2) + (indexhi << 10);
                const unsigned int bin_div2 = bin / 2;
                const unsigned int bin_offset = (bin % 2 == 1) ? 16 : 0;

                unsigned int old_val = global_histo[bin_div2];
                unsigned short old_bin = (old_val >> bin_offset) & 0xFFFF;

                if (old_bin < 255)
                {
                        atomic_fetch_add (&global_histo[bin_div2], 1 << bin_offset);
                }
        }
}

void testIncrementLocal (
        const array_view<unsigned int>& global_overflow,
        unsigned int smem[KB][256],
        const unsigned int myRange,
        const uchar4 sm) [[hc]]
{
        const unsigned int range = sm.x;
        const unsigned int indexhi = sm.y;
        const unsigned int indexlo = sm.z;
        const unsigned int offset  = sm.w;

        /* Scan for inputs that are inside the central region of histogram */
        if (range == myRange)
        {
                /* Atomically increment shared memory */
                unsigned int add = (unsigned int)(1 << offset);
                unsigned int prev = atomic_fetch_add (&smem[indexhi][indexlo], add);

                /* Check if current bin overflowed */
                unsigned int prev_bin_val = (prev >> offset) & 0x000000FF;

                /* If there was an overflow, record it and record if it cascaded into other bins */
                if (prev_bin_val == 0x000000FF)
                {
                        const unsigned int bin =
                                range * BINS_PER_BLOCK +
                                offset / 8 + (indexlo << 2) + (indexhi << 10);

                        bool can_overflow_to_bin_plus_1 = (offset < 24) ? true : false;
                        bool can_overflow_to_bin_plus_2 = (offset < 16) ? true : false;
                        bool can_overflow_to_bin_plus_3 = (offset <  8) ? true : false;

                        bool overflow_into_bin_plus_1 = false;
                        bool overflow_into_bin_plus_2 = false;
                        bool overflow_into_bin_plus_3 = false;

                        unsigned int prev_bin_plus_1_val = (prev >> (offset +  8)) & 0x000000FF;
                        unsigned int prev_bin_plus_2_val = (prev >> (offset + 16)) & 0x000000FF;
                        unsigned int prev_bin_plus_3_val = (prev >> (offset + 24)) & 0x000000FF;

                        if (can_overflow_to_bin_plus_1 &&        prev_bin_val == 0x000000FF) overflow_into_bin_plus_1 = true;
                        if (can_overflow_to_bin_plus_2 && prev_bin_plus_1_val == 0x000000FF) overflow_into_bin_plus_2 = true;
                        if (can_overflow_to_bin_plus_3 && prev_bin_plus_2_val == 0x000000FF) overflow_into_bin_plus_3 = true;

                        unsigned int bin_plus_1_add;
                        unsigned int bin_plus_2_add;
                        unsigned int bin_plus_3_add;

                        if (overflow_into_bin_plus_1) bin_plus_1_add = (prev_bin_plus_1_val < 0x000000FF) ? 0xFFFFFFFF : 0x000000FF;
                        if (overflow_into_bin_plus_2) bin_plus_2_add = (prev_bin_plus_2_val < 0x000000FF) ? 0xFFFFFFFF : 0x000000FF;
                        if (overflow_into_bin_plus_3) bin_plus_3_add = (prev_bin_plus_3_val < 0x000000FF) ? 0xFFFFFFFF : 0x000000FF;

                                                      atomic_fetch_add (&global_overflow[bin],   256);
                        if (overflow_into_bin_plus_1) atomic_fetch_add (&global_overflow[bin+1], bin_plus_1_add);
                        if (overflow_into_bin_plus_2) atomic_fetch_add (&global_overflow[bin+2], bin_plus_2_add);
                        if (overflow_into_bin_plus_3) atomic_fetch_add (&global_overflow[bin+3], bin_plus_3_add);
                }
        }
}

void clearMemory (unsigned int smem[KB][256], int tx, int bx) [[hc]]
{
        for (int i = tx; i < BINS_PER_BLOCK / 4; i += bx)
        {
                ((unsigned int*)smem)[i] = 0;
        }
}

void copyMemory (const array_view<unsigned int>& dst, unsigned int src[KB][256], int tx, int bx) [[hc]]
{
        for (int i = tx; i < BINS_PER_BLOCK/4; i += bx)
        {
            dst[i] = ((unsigned int*)src)[i];
        }
}

void histo_main_kernel (
        tiled_index<2>& tidx,
        const array_view<uchar4>& sm_mappings,
        int N,
        unsigned int num_elements,
        unsigned int sm_range_min,
        unsigned int sm_range_max,
        unsigned int histo_height,
        unsigned int histo_width,
        const array_view<unsigned int>& global_subhisto,
        const array_view<unsigned int>& global_histo,
        const array_view<unsigned int>& global_overflow) [[hc]]
{
        /* Most optimal solution uses 24 * 1024 bins per threadblock */
        tile_static unsigned int sub_histo[KB][256];
        int ty = tidx.tile[1];
        int bx = tidx.tile_dim[0];
        int tx = tidx.tile[0];

        /* Each threadblock contributes to a specific 24KB range of histogram,
         * and also scans every N-th line for interesting data.  N = gridDim.x
         */
        unsigned int local_scan_range = sm_range_min + ty;
        unsigned int local_scan_load = tidx.global[0];

        clearMemory (sub_histo, tx, bx);
        tidx.barrier.wait();

        if (ty == 0)
        {
                /* Loop through and scan the input */
                while (local_scan_load < num_elements)
                {
                        /* Read buffer */
                        uchar4 sm = sm_mappings[local_scan_load];
                        local_scan_load += N;

                        /* Check input */
                        testIncrementLocal (
                                global_overflow,
                                sub_histo,
                                local_scan_range,
                                sm
                        );
                        testIncrementGlobal (
                                global_histo,
                                sm_range_min,
                                sm_range_max,
                                sm
                        );
                }
        }
        else
        {
                /* Loop through and scan the input */
                while (local_scan_load < num_elements)
                {
                        /* Read buffer */
                        uchar4 sm = sm_mappings[local_scan_load];
                        local_scan_load += N;

                        /* Check input */
                        testIncrementLocal (
                                global_overflow,
                                sub_histo,
                                local_scan_range,
                                sm
                        );
                }
        }

        /* Store sub histogram to global memory */
        unsigned int store_index = tx * (histo_height * histo_width / 4) + (local_scan_range * BINS_PER_BLOCK / 4);

        tidx.barrier.wait();
        copyMemory (global_subhisto, sub_histo, store_index + tx, bx);
}
