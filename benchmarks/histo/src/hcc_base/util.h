#define KB                      8
#define BINS_PER_BLOCK          (KB * 1024)
#define BLOCK_X         14

#define PRESCAN_THREADS     512
#define PRESCAN_BLOCKS_X    64

#if KB == 24
        #define THREADS         768
#elif KB == 48
        #define THREADS         1024
#else //KB == 16 or other
        #define THREADS         512
#endif

#define UNROLL 16

typedef struct uchar4 { unsigned char x, y, z, w; ~uchar4() [[hc, cpu]] {} } uchar4;
typedef struct ushort4 { unsigned short x, y, z, w; } ushort4;
typedef struct uint4 { unsigned int x, y, z, w; } uint4;
typedef struct uint2 { unsigned int x, y; } uint2;

void dump_histo_img(unsigned char* histo, unsigned int height, unsigned int width, const char *filename);
