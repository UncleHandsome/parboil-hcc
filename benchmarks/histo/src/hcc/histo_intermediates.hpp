#include <stdio.h>

void calculateBin (
        const unsigned int bin,
        uchar4& sm_mapping) [[hc]]
{
        unsigned char offset  =  bin        %   4;
        unsigned char indexlo = (bin >>  2) % 256;
        unsigned char indexhi = (bin >> 10) %  KB;
        unsigned char block   =  bin / BINS_PER_BLOCK;

        offset *= 8;

        uchar4 sm;
        sm.x = block;
        sm.y = indexhi;
        sm.z = indexlo;
        sm.w = offset;

        sm_mapping = sm;
}

void histo_intermediates_kernel (
        tiled_index<1>& tidx,
        const array_view<uint2>& input,
        unsigned int height,
        unsigned int width,
        unsigned int input_pitch,
        const array_view<uchar4>& sm_mappings) [[hc]]
{
        int blockId = tidx.tile[0];
        int threadId = tidx.local[0];
        int dimId = tidx.tile_dim[0];
        unsigned int line = UNROLL * blockId;// 16 is the unroll factor;

        uint2 *load_bin = &input[line * input_pitch + threadId];

        unsigned int store = line * width + threadId;
        bool skip = (width % 2) && (threadId == (dimId - 1));

        #pragma unroll
        for (int i = 0; i < UNROLL; i++)
        {
                uint2 bin_value = *load_bin;

                calculateBin (
                        bin_value.x,
                        sm_mappings[store]
                );

                if (!skip) calculateBin (
                        bin_value.y,
                        sm_mappings[store + dimId]
                );

                load_bin += input_pitch;
                store += width;
        }
}
