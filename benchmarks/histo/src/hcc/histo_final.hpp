/***************************************************************************
 *
 *            (C) Copyright 2010 The Board of Trustees of the
 *                        University of Illinois
 *                         All Rights Reserved
 *
 ***************************************************************************/

#define MIN(a, b) a < b ? a : b
/* Combine all the sub-histogram results into one final histogram */
void histo_final_kernel (
    tiled_index<1>& tidx,
    unsigned int N,
    unsigned int sm_range_min,
    unsigned int sm_range_max,
    unsigned int histo_height,
    unsigned int histo_width,
    const array_view<unsigned int>& global_subhisto,
    const array_view<unsigned int>& global_histo,
    const array_view<unsigned int>& global_overflow,
    const array_view<unsigned int>& final_histo) [[hc]] //final output
{
    unsigned int start_offset = tidx.global[0];
    const ushort4 zero_short  = {0, 0, 0, 0};
    const uint4 zero_int      = {0, 0, 0, 0};

    unsigned int size_low_histo = sm_range_min * BINS_PER_BLOCK;
    unsigned int size_mid_histo = (sm_range_max - sm_range_min +1) * BINS_PER_BLOCK;

    /* Clear lower region of global histogram */
    for (unsigned int i = start_offset; i < size_low_histo/4; i += N)
    {
        ushort4 global_histo_data = global_histo.reinterpret_as<ushort4>()[i];
        global_histo.reinterpret_as<ushort4>()[i] = zero_short;

        global_histo_data.x = MIN (global_histo_data.x, 255);
        global_histo_data.y = MIN (global_histo_data.y, 255);
        global_histo_data.z = MIN (global_histo_data.z, 255);
        global_histo_data.w = MIN (global_histo_data.w, 255);

        uchar4 final_histo_data = {
            (unsigned char)global_histo_data.x,
            (unsigned char)global_histo_data.y,
            (unsigned char)global_histo_data.z,
            (unsigned char)global_histo_data.w
        };

        final_histo.reinterpret_as<uchar4>()[i] = final_histo_data;
    }

    /* Clear the middle region of the overflow buffer */
    for (unsigned int i = (size_low_histo/4) + start_offset; i < (size_low_histo+size_mid_histo)/4; i += N)
    {
        uint4 global_histo_data = global_overflow.reinterpret_as<uint4>()[i];
        global_overflow.reinterpret_as<uint4>()[i] = zero_int;

        uint4 internal_histo_data = {
            global_histo_data.x,
            global_histo_data.y,
            global_histo_data.z,
            global_histo_data.w
        };

        #pragma unroll
        for (int j = 0; j < BLOCK_X; j++)
        {
            unsigned int bin4in = global_subhisto[i + j * histo_height * histo_width / 4];
            internal_histo_data.x += (bin4in >>  0) & 0xFF;
            internal_histo_data.y += (bin4in >>  8) & 0xFF;
            internal_histo_data.z += (bin4in >> 16) & 0xFF;
            internal_histo_data.w += (bin4in >> 24) & 0xFF;
        }

        internal_histo_data.x = MIN (internal_histo_data.x, 255);
        internal_histo_data.y = MIN (internal_histo_data.y, 255);
        internal_histo_data.z = MIN (internal_histo_data.z, 255);
        internal_histo_data.w = MIN (internal_histo_data.w, 255);

        uchar4 final_histo_data = {
            (unsigned char)internal_histo_data.x,
            (unsigned char)internal_histo_data.y,
            (unsigned char)internal_histo_data.z,
            (unsigned char)internal_histo_data.w
        };

        final_histo.reinterpret_as<uchar4>()[i] = final_histo_data;
    }

    /* Clear the upper region of global histogram */
    for (unsigned int i = ((size_low_histo+size_mid_histo)/4) + start_offset; i < (histo_height*histo_width)/4; i += N)
    {
        ushort4 global_histo_data = global_histo.reinterpret_as<ushort4>()[i];
        global_histo.reinterpret_as<ushort4>()[i] = zero_short;

        global_histo_data.x = MIN (global_histo_data.x, 255);
        global_histo_data.y = MIN (global_histo_data.y, 255);
        global_histo_data.z = MIN (global_histo_data.z, 255);
        global_histo_data.w = MIN (global_histo_data.w, 255);

        uchar4 final_histo_data = {
            (unsigned char)global_histo_data.x,
            (unsigned char)global_histo_data.y,
            (unsigned char)global_histo_data.z,
            (unsigned char)global_histo_data.w
        };

        final_histo.reinterpret_as<uchar4>()[i] = final_histo_data;
    }
}
