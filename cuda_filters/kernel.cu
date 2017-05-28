
#include <Windows.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "cuda_filters.h"
#include "kernel.h"

using namespace cudafilter;

static int nblocks(int n, int width) {
    return (n + width - 1) / width;
}

__global__ void kl_convert_yc_to_yca(PIXEL_YCA* yca, PIXEL_YC* yc, int pitch, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        PIXEL_YC s = yc[y * pitch + x];
        PIXEL_YCA d = { s.y, s.cb, s.cr };
        yca[y * width + x] = d;
    }
}

void convert_yc_to_yca(PIXEL_YCA* yca, PIXEL_YC* yc, int pitch, int width, int height)
{
    dim3 threads(32, 16);
    dim3 blocks(nblocks(width, threads.x), nblocks(height, threads.y));
    kl_convert_yc_to_yca<<<blocks, threads>>>(yca, yc, pitch, width, height);
}

__global__ void kl_convert_yca_to_yc(PIXEL_YC* yc, PIXEL_YCA* yca, int pitch, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        PIXEL_YCA s = yca[y * width + x];
        PIXEL_YC d = { s.y, s.cb, s.cr };
        yc[y * pitch + x] = d;
    }
}

void convert_yca_to_yc(PIXEL_YC* yc, PIXEL_YCA* yca, int pitch, int width, int height)
{
    dim3 threads(32, 16);
    dim3 blocks(nblocks(width, threads.x), nblocks(height, threads.y));
    kl_convert_yca_to_yc<<<blocks, threads>>>(yc, yca, pitch, width, height);
}

template <typename T>
static __device__ T get_min(T a, T b) {
    return a < b ? a : b;
}

template <typename T>
static __device__ T get_max(T a, T b) {
    return a > b ? a : b;
}

template <>
static __device__ short4 get_max(short4 a, short4 b) {
    short4 max_value;
    max_value.x = get_max(a.x, b.x);
    max_value.y = get_max(a.y, b.y);
    max_value.z = get_max(a.z, b.z);
    return max_value;
}

template <typename T>
static __device__ T get_min(T a, T b, T c, T d) {
    return get_min(get_min(a,b), get_min(c,d));
}

template <typename T>
static __device__ T get_max(T a, T b, T c, T d) {
    return get_max(get_max(a, b), get_max(c, d));
}

// ランダムな128bit列をランダムな -range ～ range にして返す
// range は0～127以下
static __device__ char random_range(uint8_t random, char range) {
    return ((((range << 1) + 1) * (int)random) >> 8) - range;
}

static __device__ short4 get_abs_diff(short4 a, short4 b) {
    short4 diff;
    diff.x = abs(a.x - b.x);
    diff.y = abs(a.y - b.y);
    diff.z = abs(a.z - b.z);
    return diff;
}

static __device__ short4 get_avg(short4 a, short4 b) {
    short4 avg;
    avg.x = (a.x + b.x + 1) >> 1;
    avg.y = (a.y + b.y + 1) >> 1;
    avg.z = (a.z + b.z + 1) >> 1;
    return avg;
}

static __device__ short4 get_avg(short4 a, short4 b, short4 c, short4 d) {
    short4 avg;
    avg.x = (a.x + b.x + c.x + d.x + 2) >> 2;
    avg.y = (a.y + b.y + c.y + d.y + 2) >> 2;
    avg.z = (a.z + b.z + c.z + d.z + 2) >> 2;
    return avg;
}

__constant__ BandingParam band_prm;

__device__ void dev_reduce_banding(
    bool check, int sample_mode, bool blur_first, bool yc_and, bool dither_enable,
    short4* __restrict__ dst, const short4* __restrict__ src, const uint8_t* __restrict__ rand)
{
    const short4 YC_YELLOW = { 3514, -626,    73 };
    const short4 YC_BLACK = { 1013,    0,     0 };

    const int ditherY = band_prm.ditherY;
    const int ditherC = band_prm.ditherC;
    const int width = band_prm.width;
    const int height = band_prm.height;
    const int range = band_prm.range;
    const int threshold_y = band_prm.threshold_y;
    const int threshold_cb = band_prm.threshold_cb;
    const int threshold_cr = band_prm.threshold_cr;
    const int field_mask = band_prm.interlaced ? 0xfe : 0xff;

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    const int rand_step = width * height;
    const int offset = y * width + x;

    if (x < width && y < height) {

        const int range_limited = get_min(range,
            get_min(y, height - y - 1, x, width - x - 1));
        const char refA = random_range(rand[offset + rand_step * 0], range_limited);
        const char refB = random_range(rand[offset + rand_step * 1], range_limited);

        short4 src_val = src[offset];
        short4 avg, diff;

        if (sample_mode == 0) {
            const int ref = (char)(refA & field_mask) * width + refB;

            avg = src[offset + ref];
            diff = get_abs_diff(src_val, avg);

        }
        else if (sample_mode == 1) {
            const int ref = (char)(refA & field_mask) * width + refB;

            short4 ref_p = src[offset + ref];
            short4 ref_m = src[offset - ref];

            avg = get_avg(ref_p, ref_m);
            diff = blur_first
                ? get_abs_diff(src_val, avg)
                : get_max(get_abs_diff(src_val, ref_p),
                    get_abs_diff(src_val, ref_m));
        }
        else {
            const int ref_0 = (char)(refA & field_mask) * width + refB;
            const int ref_1 = refA - (char)(refB & field_mask) * width;

            short4 ref_0p = src[offset + ref_0];
            short4 ref_0m = src[offset - ref_0];
            short4 ref_1p = src[offset + ref_1];
            short4 ref_1m = src[offset - ref_1];

            avg = get_avg(ref_0p, ref_0m, ref_1p, ref_1m);
            diff = blur_first
                ? get_abs_diff(src_val, avg)
                : get_max(get_abs_diff(src_val, ref_0p),
                    get_abs_diff(src_val, ref_0m),
                    get_abs_diff(src_val, ref_1p),
                    get_abs_diff(src_val, ref_1m));
        }

        short4 dst_val;
        if (check) {
            if (yc_and) {
                dst_val = ((diff.x < threshold_y) && (diff.y < threshold_cb) && (diff.z < threshold_cr)) ? YC_YELLOW : YC_BLACK;
            }
            else {
                dst_val = ((diff.x < threshold_y) || (diff.y < threshold_cb) || (diff.z < threshold_cr)) ? YC_YELLOW : YC_BLACK;
            }
        }
        else {
            if (yc_and) {
                dst_val = ((diff.x < threshold_y) && (diff.y < threshold_cb) && (diff.z < threshold_cr)) ? avg : src_val;
            }
            else {
                dst_val.x = (diff.x < threshold_y) ? avg.x : src_val.x;
                dst_val.y = (diff.y < threshold_cb) ? avg.y : src_val.y;
                dst_val.z = (diff.z < threshold_cr) ? avg.z : src_val.z;
            }

            if (dither_enable) {
                dst_val.x += random_range(rand[offset + rand_step * 2], ditherY);
                dst_val.y += random_range(rand[offset + rand_step * 3], ditherC);
                dst_val.z += random_range(rand[offset + rand_step * 4], ditherC);
            }
        }

        dst[offset] = dst_val;
    }
}

template <int sample_mode, bool blur_first, bool yc_and, bool dither_enable>
__global__ void kl_reduce_banding(short4* __restrict__ dst, const short4* __restrict__ src, const uint8_t* __restrict__ rand)
{
    dev_reduce_banding(false, sample_mode, blur_first, yc_and, dither_enable, dst, src, rand);
}

__global__ void kl_reduce_banding_check(
    int sample_mode, bool blur_first, bool yc_and, bool dither_enable,
    short4* __restrict__ dst, const short4* __restrict__ src, const uint8_t* __restrict__ rand)
{
    dev_reduce_banding(true, sample_mode, blur_first, yc_and, dither_enable, dst, src, rand);
}

template <int sample_mode, bool blur_first, bool yc_and, bool dither_enable>
void run_reduce_banding(BandingParam * prm, short4* dev_dst, const short4* dev_src, const uint8_t* dev_rand, cudaStream_t stream)
{
    dim3 threads(32, 16);
    dim3 blocks(nblocks(prm->width, threads.x), nblocks(prm->height, threads.y));
    kl_reduce_banding<sample_mode, blur_first, yc_and, dither_enable>
        <<<blocks, threads, 0, stream >>>(dev_dst, dev_src, dev_rand);
}

void reduce_banding(BandingParam * prm, PIXEL_YCA* dev_dst, const PIXEL_YCA* dev_src, const uint8_t* dev_rand, cudaStream_t stream)
{
    void(*kernel_table[3][2][2][2])(BandingParam * prm, short4* dev_dst, const short4* dev_src, const uint8_t* dev_rand, cudaStream_t stream) = {
        {
            {
                { run_reduce_banding<0, false, false, false>, NULL },
                { run_reduce_banding<0, false, true, false>, NULL },
            },{
                { NULL, NULL },
                { NULL, NULL },
            }
        },{
            {
                { run_reduce_banding<1, false, false, true>, run_reduce_banding<1, false, false, true> },
                { run_reduce_banding<1, false, true, false>, run_reduce_banding<1, false, true, true> },
            },{
                { run_reduce_banding<1, true, false, true>, run_reduce_banding<1, true, false, true> },
                { run_reduce_banding<1, true, true, false>, run_reduce_banding<1, true, true, true> },
            }
        },{
            {
                { run_reduce_banding<2, false, false, true>, run_reduce_banding<2, false, false, true> },
                { run_reduce_banding<2, false, true, false>, run_reduce_banding<2, false, true, true> },
            },{
                { run_reduce_banding<2, true, false, true>, run_reduce_banding<2, true, false, true> },
                { run_reduce_banding<2, true, true, false>, run_reduce_banding<2, true, true, true> },
            }
        }
    };

    cudaStream_t s;

    CUDA_CHECK(cudaMemcpyToSymbolAsync(band_prm, prm, sizeof(*prm), 0, cudaMemcpyHostToDevice, stream));

    bool blur_first = (prm->sample_mode != 0) && (prm->blur_first != 0);
    bool yc_and = (prm->yc_and != 0);
    bool dither_enable = (prm->sample_mode != 0) && (prm->ditherY > 0 || prm->ditherC > 0);

    if (prm->check) {
        dim3 threads(32, 16);
        dim3 blocks(nblocks(prm->width, threads.x), nblocks(prm->height, threads.y));
        kl_reduce_banding_check<<<blocks, threads, 0, stream >>>(
            prm->sample_mode, blur_first, yc_and, dither_enable, (short4*)dev_dst, (const short4*)dev_src, dev_rand);
    }
    else {
        kernel_table[prm->sample_mode][blur_first][yc_and][dither_enable](prm, (short4*)dev_dst, (const short4*)dev_src, dev_rand, stream);
    }
}

__constant__ EdgeLevelParam edge_prm;

template <bool check, bool interlaced, bool bw_enable>
__global__ void kl_edgelevel(short4* __restrict__ dst, const short4* __restrict__ src)
{
    const short4 YC_ORANGE = { 2255, -836,  1176 }; //調整 - 明 - 白補正対象
    const short4 YC_YELLOW = { 3514, -626,    73 }; //調整 - 明
    const short4 YC_SKY = { 3702,  169,  -610 }; //調整 - 暗
    const short4 YC_BLUE = { 1900, 1240,  -230 }; //調整 - 暗 - 黒補正対象
    const short4 YC_BLACK = { 1013,    0,     0 }; //エッジでない

    const int width = edge_prm.width;
    const int height = edge_prm.height;
    const int str = edge_prm.str;
    const int thrs = edge_prm.thrs;
    const int bc = edge_prm.bc;
    const int wc = edge_prm.wc;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    src += y * width + x;
    dst += y * width + x;

    if (y <= 1 || y >= height - 2 || x <= 1 || x >= width - 2) {
        if (x < width && y < height) {
            *dst = check ? YC_BLACK : *src;
        }
    }
    else {
        short4 srcv = *src;
        short4 dstv;

        int hmax, hmin, vmax, vmin, avg;
        hmax = hmin = src[-2].x;
        vmax = vmin = src[-2 * width].x;

        if (interlaced) {
            for (int i = -1; i < 3; ++i) {
                hmax = max(hmax, src[i].x);
                hmin = min(hmin, src[i].x);
            }
            for (int i = 0; i < 3; i += 2) {
                vmax = max(vmax, src[i*width].x);
                vmin = min(vmin, src[i*width].x);
            }
        }
        else {
            for (int i = -1; i < 3; ++i) {
                hmax = max(hmax, src[i].x);
                hmin = min(hmin, src[i].x);
                vmax = max(vmax, src[i*width].x);
                vmin = min(vmin, src[i*width].x);
            }
        }

        if (hmax - hmin < vmax - vmin)
            hmax = vmax, hmin = vmin;

        if (check) {
            if (hmax - hmin > thrs) {
                avg = (hmax + hmin) >> 1;
                if (bw_enable && srcv.x == hmin)
                    dstv = YC_BLUE;
                else if (bw_enable && srcv.x == hmax)
                    dstv = YC_ORANGE;
                else
                    dstv = (srcv.x > avg) ? YC_YELLOW : YC_SKY;
            }
            else {
                dstv = YC_BLACK;
            }
        }
        else {
            if (hmax - hmin > thrs) {
                avg = (hmin + hmax) >> 1;

                if (bw_enable) {
                    if (src->x == hmin)
                        hmin -= bc;
                    hmin -= bc;
                    if (src->x == hmax)
                        hmax += wc;
                    hmax += wc;
                }

                dstv.x = min(max(srcv.x + ((srcv.x - avg) * str >> 4), hmin), hmax);
            }
            else {
                dstv.x = srcv.x;
            }

            dstv.y = srcv.y;
            dstv.z = srcv.z;
            dstv.w = 0;
        }

        *dst = dstv;
    }
}

template <bool check, bool interlaced, bool bw_enable>
void run_edgelevel(EdgeLevelParam * prm, short4* dev_dst, const short4* dev_src, cudaStream_t stream)
{
    dim3 threads(32, 16);
    dim3 blocks(nblocks(prm->width, threads.x), nblocks(prm->height, threads.y));
    kl_edgelevel<check, interlaced, bw_enable> << <blocks, threads, 0, stream>> >(dev_dst, dev_src);
}

void edgelevel(EdgeLevelParam * prm, PIXEL_YCA* dev_dst, const PIXEL_YCA* dev_src, cudaStream_t stream)
{
    void(*kernel_table[2][2][2])(EdgeLevelParam * prm, short4* dev_dst, const short4* dev_src, cudaStream_t stream) = {
        {
            { run_edgelevel<false, false, false>, run_edgelevel<false, false, true> },
            { run_edgelevel<false, true, false>, run_edgelevel<false, true, true> }
        },
        {
            { run_edgelevel<true, false, false>, run_edgelevel<true, false, true> },
            { run_edgelevel<true, true, false>, run_edgelevel<true, true, true> }
        }
    };

    CUDA_CHECK(cudaMemcpyToSymbolAsync(edge_prm, prm, sizeof(*prm), 0, cudaMemcpyHostToDevice, stream));

    bool check = (prm->check != 0);
    bool interlaced = (prm->interlaced != 0);
    bool bw_enable = (prm->bc > 0 || prm->wc > 0);

    kernel_table[check][interlaced][bw_enable](prm, (short4*)dev_dst, (const short4*)dev_src, stream);
}
