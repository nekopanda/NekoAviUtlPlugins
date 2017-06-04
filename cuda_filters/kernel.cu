
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

#pragma region YUV YC48 Converter

int depth_scale_factor(int depth) {
    switch (depth) {
    case 8:
        return 0;
    case 10:
    case 12:
        return 2;
    }
    throw_exception_("Unsupported color depth");
    return 0;
}

// ランダムな128bit列をランダムな -range ～ range にして返す
// range は0～127以下
static __device__ char random_range(uint8_t random, char range) {
    return ((((range << 1) + 1) * (int)random) >> 8) - range;
}

static __device__ int clamp(int v, int minimum, int maximum) {
	return (v < minimum)
		? minimum
		: (v > maximum)
		? maximum
		: v;
}

// 8,10bitのときはAviUtlの変換式
template <int depth> __device__ int yuv_to_yc48_luma(int Y) {
	return ((Y * 1197) >> 6) - 299;
}
template <int depth> __device__ int yuv_to_yc48_chroma(int UV) {
	return ((UV - 128) * 4681 + 164) >> 8;
}

// 12bitのときはAviUtlの変換式だとオーバーフローするので独自の変換式
// 値の差に影響がないように一次式の傾き成分はAviUtlと同じにする
template <> __device__ int yuv_to_yc48_luma<12>(int Y) {
	return (Y * 1197) >> 8;
}
template <> __device__ int yuv_to_yc48_chroma<12>(int UV) {
	return ((UV * 4681) >> 10);
}

// clamp BT.601, BT.709，BT.2020 はこれでいい。他は知らない
//template <int depth> __device__ int clamp_luma(int v) {
//	return clamp(v, 16 << (depth - 8), 235 << (depth - 8));
//}
//template <int depth> __device__ int clamp_chroma(int v) {
//	return clamp(v, 16 << (depth - 8), 240 << (depth - 8));
//}

// 範囲外にあってもそのまま
// オーバーフローとアンダーフローだけ制限する
template <int depth> __device__ int clamp_luma(int v) {
	return clamp(v, 0, (1 << depth) - 1);
}
template <int depth> __device__ int clamp_chroma(int v) {
	return clamp(v, 0, (1 << depth) - 1);
}

// 8,10bitのときはAviUtlの変換式
// 変換前後で値の差が出ないようにYの変換式だけ若干修正
// dithは-127～127の乱数
template <int src_depth, int dst_depth> struct yc48_to_yuv {
    enum {
        SHIFT_LUMA = 12 + src_depth - dst_depth,
        SHIFT_CHROMA = 7 + src_depth - dst_depth
    };
    static __device__ int luma(int Y, int dith) {
        return clamp_luma<dst_depth>((Y * 219 + 2109 + (16 << 12) + (dith << (SHIFT_LUMA - 8))) >> SHIFT_LUMA);
    }
    static __device__ int chroma(int UV, int dith) {
        return clamp_chroma<dst_depth>(((UV + 2048) * 7 + 66 + (16 << 7) + ((dith << 3) >> (11 - SHIFT_CHROMA))) >> SHIFT_CHROMA);
    }
};

// 12bitのときはAviUtlの変換式だとオーバーフローするので独自の変換式
template <int dst_depth> struct yc48_to_yuv<12, dst_depth> {
    enum {
        SHIFT_LUMA = 10 + 12 - dst_depth,
        SHIFT_CHROMA = 5 + 12 - dst_depth
    };
    static __device__ int luma(int Y, int dith) {
        return clamp_luma<dst_depth>((Y * 219 + 629 + (dith << (SHIFT_LUMA - 8))) >> SHIFT_LUMA);
    }
    static __device__ int chroma(int UV, int dith) {
        return clamp_chroma<dst_depth>((UV * 7 + 21 + ((dith << 1) >> (9 - SHIFT_CHROMA))) >> SHIFT_CHROMA);
    }
};

// AviUtlのYUY2からYC48に変換するときの色差補間専用
static __device__ int get_avg_aviutl(int a, int b) {
	return (a + b) >> 1;
}

// sharedメモリのサイズからCUDAブロックは(256,1,1)に限定
template <typename T, int depth>
__global__ void kl_convert_yuv_to_yca(
    bool interlaced,
	const T* __restrict__ Y, const T* __restrict__ U, const T* __restrict__ V,
	int lsY, int lsU, int lsV,
	short4* yca, int width, int height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ int s_pixU[256];
    __shared__ int s_pixV[256];

    // 縦方向クロマは補間しないで同じ値をそのまま入れる
    // どうせ捨てられるので勘弁
    int cy = interlaced ? (((y >> 1) & ~1) | (y & 1)) : (y >> 1);

    // まず奇数画素の色差は右の画素と同じにする（右端だけは左と同じにする）
    int cx = (x >> 1) + ((x < width - 1) ? (x & 1) : 0);
    int offY = x + lsY * y;
    int offU = cx + lsU * cy;
    int offV = cx + lsV * cy;
    int pixY;

    if (x < width) {
        pixY = yuv_to_yc48_luma<depth>(Y[offY]);
        s_pixU[threadIdx.x] = yuv_to_yc48_chroma<depth>(U[offU]);
        s_pixV[threadIdx.x] = yuv_to_yc48_chroma<depth>(V[offV]);
    }

    __syncthreads();

    if (x < width) {
        int pixU = s_pixU[threadIdx.x];
        int pixV = s_pixV[threadIdx.x];
        if (x & 1) {
		    // 奇数画素の色差は左の画素との平均にする
            pixU = get_avg_aviutl(s_pixU[threadIdx.x - 1], pixU);
            pixV = get_avg_aviutl(s_pixV[threadIdx.x - 1], pixV);
		}
        short4 d = { (short)pixY, (short)pixU, (short)pixV };
		yca[y * width + x] = d;
#if 0
        if (y == 183 && x == 937) {
            printf("to_yca: %d -> %d, %d -> %d, %d -> %d\n", Y[offY], pixY, U[offU], pixU, V[offV], pixV);
        }
#endif
	}
}

template <typename T, int depth>
void run_convert_yuv_to_yca(YUVtoYCAParam* prm, FrameYV12 yuv, PIXEL_YCA* yca, cudaStream_t stream)
{
    dim3 threads(256);
    dim3 blocks(nblocks(prm->width, threads.x), prm->height);
    kl_convert_yuv_to_yca<T, depth> << <blocks, threads, 0, stream>> >(
        prm->interlaced != 0,
        (T*)yuv.y, (T*)yuv.u, (T*)yuv.v,
        prm->frame_info.linesizeY, prm->frame_info.linesizeU, prm->frame_info.linesizeV,
        (short4*)yca, prm->width, prm->height);
}

static int depth_index(int depth) {
    switch (depth) {
    case 8: return 0;
    case 10: return 1;
    case 12: return 2;
    }
    throw_exception_("Unsupported color depth");
    return -1;
}

void convert_yuv_to_yca(YUVtoYCAParam* prm, FrameYV12 yuv, PIXEL_YCA* yca, cudaStream_t stream)
{
    void(*table[3])(YUVtoYCAParam* prm, FrameYV12 yuv, PIXEL_YCA* yca, cudaStream_t stream) = {
        run_convert_yuv_to_yca<uint8_t, 8>,
        run_convert_yuv_to_yca<uint16_t, 10>,
        run_convert_yuv_to_yca<uint16_t, 12>
    };
    table[depth_index(prm->frame_info.depth)](prm, yuv, yca, stream);
}

// CUDAブロックは(256,1,1)に限定
template <typename T, int src_depth, int dst_depth, bool dither>
__global__ void kl_convert_yca_to_yuv(
    bool interlaced,
    T* Y, T* U, T* V,
    int lsY, int lsU, int lsV,
    const short4* __restrict__ yca, int width, int height,
    const uint8_t* __restrict__ rand)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    const int rand_step = width * height;
    const int offset = y * width + x;

    if (x < width) {
        short4 d = yca[y * width + x];

        int offY = x + lsY * y;
        int dithY = dither ? random_range(rand[offset + rand_step * 0], 127) : 0;
        Y[offY] = yc48_to_yuv<src_depth, dst_depth>::luma(d.x, dithY);

        // 色差は偶数画素のみ
        if ((x & 1) == 0 && (interlaced ? (((y + 1) & 2) == 0) : ((y & 1) == 0))) {
            int cy = y >> 1;
            int cx = x >> 1;
            int offU = cx + lsU * cy;
            int offV = cx + lsV * cy;
            int dithU = dither ? random_range(rand[offset + rand_step * 1], 127) : 0;
            int dithV = dither ? random_range(rand[offset + rand_step * 2], 127) : 0;
            U[offU] = yc48_to_yuv<src_depth, dst_depth>::chroma(d.y, dithU);
            V[offV] = yc48_to_yuv<src_depth, dst_depth>::chroma(d.z, dithV);
        }
#if 0
        if (y == 183 && x == 937) {
            //printf("to_yuv: %d -> %d, %d -> %d, %d -> %d\n", d.x, Y[offY], d.y, U[offU], d.z, V[offV]);
            printf("to_yuv: %d -> %d\n", d.x, Y[offY]);
        }
#endif
    }
}

template <typename T, int src_depth, int dst_depth, bool dither>
void run_convert_yca_to_yuv(YCAtoYUVParam* prm, FrameYV12 yuv, PIXEL_YCA* yca, const uint8_t* rand, cudaStream_t stream)
{
    dim3 threads(256);
    dim3 blocks(nblocks(prm->width, threads.x), prm->height);
    kl_convert_yca_to_yuv<T, src_depth, dst_depth, dither> << <blocks, threads, 0, stream >> >(
        prm->interlaced != 0,
        (T*)yuv.y, (T*)yuv.u, (T*)yuv.v,
        prm->frame_info.linesizeY, prm->frame_info.linesizeU, prm->frame_info.linesizeV,
        (short4*)yca, prm->width, prm->height, rand);
}

void convert_yca_to_yuv(YCAtoYUVParam* prm, FrameYV12 yuv, PIXEL_YCA* yca, const uint8_t* rand, cudaStream_t stream)
{
    void(*table[3][3][2])(YCAtoYUVParam* prm, FrameYV12 yuv, PIXEL_YCA* yca, const uint8_t* rand, cudaStream_t stream) = {
        {
            { run_convert_yca_to_yuv<uint8_t, 8, 8, false>, run_convert_yca_to_yuv<uint8_t, 8, 8, true> },
            { run_convert_yca_to_yuv<uint16_t, 8, 10, false>, run_convert_yca_to_yuv<uint16_t, 8, 10, true> },
            { run_convert_yca_to_yuv<uint16_t, 8, 12, false>, run_convert_yca_to_yuv<uint16_t, 8, 12, true> }
        }, {
            { run_convert_yca_to_yuv<uint8_t, 10, 8, false>, run_convert_yca_to_yuv<uint8_t, 10, 8, true> },
            { run_convert_yca_to_yuv<uint16_t, 10, 10, false>, run_convert_yca_to_yuv<uint16_t, 10, 10, true> },
            { run_convert_yca_to_yuv<uint16_t, 10, 12, false>, run_convert_yca_to_yuv<uint16_t, 10, 12, true> }
        }, {
            { run_convert_yca_to_yuv<uint8_t, 12, 8, false>, run_convert_yca_to_yuv<uint8_t, 12, 8, true> },
            { run_convert_yca_to_yuv<uint16_t, 12, 10, false>, run_convert_yca_to_yuv<uint16_t, 12, 10, true> },
            { run_convert_yca_to_yuv<uint16_t, 12, 12, false>, run_convert_yca_to_yuv<uint16_t, 12, 12, true> }
        }
    };
    table[depth_index(prm->src_depth)][depth_index(prm->frame_info.depth)][prm->dither](prm, yuv, yca, rand, stream);
}

#pragma endregion

#pragma region TemporalNR

static __device__ short4 get_abs_diff(short4 a, short4 b) {
	short4 diff;
	diff.x = abs(a.x - b.x);
	diff.y = abs(a.y - b.y);
	diff.z = abs(a.z - b.z);
	return diff;
}

__constant__ TemporalNRKernelParam tnr_param;
__constant__ FrameYV12 src_yv12_frames[TEMPNR_MAX_BATCH + TEMPNR_MAX_DIST * 2];
__constant__ PIXEL_YC*  src_yc_frames[TEMPNR_MAX_BATCH + TEMPNR_MAX_DIST * 2];
__constant__ PIXEL_YCA* dst_yca_frames[TEMPNR_MAX_BATCH];
__constant__ PIXEL_YC* dst_yc_frames[TEMPNR_MAX_BATCH];

void temporal_nr_scale_param(TemporalNRParam* prm, int depth)
{
    int shift = depth_scale_factor(depth);
    if (shift == 0) {
        return;
    }

    prm->threshY <<= shift;
    prm->threshCb <<= shift;
    prm->threshCr <<= shift;
    prm->checkThresh <<= shift;
}

template <typename T, int depth, bool aviutl, bool check>
__global__ void kl_temporal_nr()
{
	const PIXEL_YC YC_YELLOW = { 3514, -626, 73 };
	const PIXEL_YC YC_BLACK = { 1013, 0, 0 };

	const int b = threadIdx.y;
	const int lx = threadIdx.x;
	const int x = threadIdx.x + blockDim.x * blockIdx.x;
	const int y = blockIdx.y;

	const int nframes = tnr_param.nframes;
	const int mid = tnr_param.temporalDistance;
	const int pitch = tnr_param.pitch;
	const int width = tnr_param.width;
	const int temporalWidth = tnr_param.temporalWidth;
	const int threshY = tnr_param.threshY;
	const int threshCb = tnr_param.threshCb;
	const int threshCr = tnr_param.threshCr;
	const bool interlaced = tnr_param.interlaced;
	const int lsY = tnr_param.frame_info.linesizeY;
	const int lsU = tnr_param.frame_info.linesizeU;
	const int lsV = tnr_param.frame_info.linesizeV;
    const int checkThresh = tnr_param.checkThresh;

	// [nframes][32]
	// CUDAブロックの幅はここの長さの他に色差補間の関係で偶数制約があることに注意
	extern __shared__ void* s__[];
	short4 (*pixel_cache)[32] = (short4(*)[32])s__;

	int offY;

	// pixel_cacheにデータを入れる
	if (aviutl) {
		if (x < width) {
			offY = x + pitch * y;
			for (int i = b; i < nframes; i += blockDim.y) {
				PIXEL_YC src = src_yc_frames[i][offY];
				short4 yuv = { src.y, src.cb, src.cr };
				pixel_cache[i][lx] = yuv;
			}
		}
	}
	else {
		// 入力YUVからYC48に変換 //

		// まず奇数画素の色差は右の画素と同じにする（右端だけは左と同じにする）
		int cy = interlaced ? (((y >> 1) & ~1) | (y & 1)) : (y >> 1);
		int cx = (x >> 1) + ((x < width - 1) ? (x & 1) : 0);
		offY = x + lsY * y;
		int offU = cx + lsU * cy;
		int offV = cx + lsV * cy;
		if (x < width) {
			for (int i = b; i < nframes; i += blockDim.y) {
				const T* __restrict__ Y = (T*)src_yv12_frames[i].y;
				const T* __restrict__ U = (T*)src_yv12_frames[i].u;
				const T* __restrict__ V = (T*)src_yv12_frames[i].v;
                short4 d = {
                    (short)yuv_to_yc48_luma<depth>(Y[offY]),
                    (short)yuv_to_yc48_chroma<depth>(U[offU]),
                    (short)yuv_to_yc48_chroma<depth>(V[offV])
                };
				pixel_cache[i][lx] = d;
			}
		}

		__syncthreads();

		// 奇数画素の色差は左の画素との平均にする
		if (x < width && (x & 1)) {
			for (int i = b; i < nframes; i += blockDim.y) {
                short4 r = pixel_cache[i][lx];
                short4 l = pixel_cache[i][lx - 1];
                short4 d = {
                    (short)get_avg_aviutl(r.x, l.x),
                    (short)get_avg_aviutl(r.y, l.y),
                    (short)get_avg_aviutl(r.z, l.z)
                };
				pixel_cache[i][lx] = d;
			}
		}
	}

	__syncthreads();

	if (x < width) {
		short4 center = pixel_cache[b + mid][lx];

		// 重み合計を計算
		int pixel_count = 0;
		for (int i = 0; i < temporalWidth; ++i) {
			short4 ref = pixel_cache[b + i][lx];
			short4 diff = get_abs_diff(center, ref);
			if (diff.x <= threshY && diff.y <= threshCb && diff.z <= threshCr) {
				++pixel_count;
			}
		}

		float factor = 1.f / pixel_count;

		// ピクセル値を算出
		float dY = 0;
		float dU = 0;
		float dV = 0;
		for (int i = 0; i < temporalWidth; ++i) {
			short4 ref = pixel_cache[b + i][lx];
			short4 diff = get_abs_diff(center, ref);
			if (diff.x <= threshY && diff.y <= threshCb && diff.z <= threshCr) {
				dY += factor * ref.x;
				dU += factor * ref.y;
				dV += factor * ref.z;
			}
		}

		if (check && aviutl) {
			// 画素値が変わったかどうか
			short4 yca = { (short)rintf(dY), (short)rintf(dU), (short)rintf(dV) };
			short4 diff = get_abs_diff(yca, center);
			bool is_changed = (diff.x > checkThresh || diff.y > checkThresh || diff.z > checkThresh);
			dst_yc_frames[b][offY] = is_changed ? YC_YELLOW : YC_BLACK;
		}
		else if (aviutl) {
			// AviUtl互換形式
			PIXEL_YC yc = { (short)rintf(dY), (short)rintf(dU), (short)rintf(dV) };
			dst_yc_frames[b][offY] = yc;
		}
		else {
			PIXEL_YCA yca = { (short)rintf(dY), (short)rintf(dU), (short)rintf(dV) };
			dst_yca_frames[b][offY] = yca;
		}
	}
}

template <typename T, int depth, bool aviutl, bool check>
void run_temporal_nr(const TemporalNRKernelParam& param, cudaStream_t stream)
{
	dim3 threads(32, param.batchSize);
	dim3 blocks(nblocks(param.width, threads.x), param.height);
	int shared_size = threads.x*param.nframes*sizeof(short4);
	kl_temporal_nr<T, depth, aviutl, check> << <blocks, threads, shared_size, stream >> >();
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());
}

void temporal_nr(const TemporalNRKernelParam& param, const FrameYV12* src_frames, PIXEL_YCA* const * dst_frames, cudaStream_t stream)
{
	CUDA_CHECK(cudaMemcpyToSymbolAsync(tnr_param, &param,
        sizeof(param), 0, cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyToSymbolAsync(src_yv12_frames, src_frames,
        sizeof(FrameYV12) * param.nframes, 0, cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyToSymbolAsync(dst_yca_frames, dst_frames,
        sizeof(PIXEL_YCA*) * param.batchSize, 0, cudaMemcpyHostToDevice, stream));

    if (param.check) {
        throw_exception_("Check mode is only available on AviUtl mode");
    }
    else {
        switch (param.frame_info.depth) {
        case 8:
            run_temporal_nr<uint8_t, 8, false, false>(param, stream);
            break;
        case 10:
            run_temporal_nr<uint16_t, 10, false, false>(param, stream);
            break;
        case 12:
            run_temporal_nr<uint16_t, 12, false, false>(param, stream);
            break;
        default:
            throw_exception_("Unsupported color depth");
        }
    }

}

void temporal_nr(const TemporalNRKernelParam& param, PIXEL_YC* const * src_frames, PIXEL_YC* const * dst_frames)
{
	CUDA_CHECK(cudaMemcpyToSymbol(tnr_param, &param, sizeof(param), 0));
	CUDA_CHECK(cudaMemcpyToSymbol(src_yc_frames, src_frames, sizeof(PIXEL_YC*) * param.nframes, 0));
	CUDA_CHECK(cudaMemcpyToSymbol(dst_yc_frames, dst_frames, sizeof(PIXEL_YC*) * param.batchSize, 0));

    if (param.check) {
        run_temporal_nr<uint8_t, 8, true, true>(param, NULL);
    }
    else {
        run_temporal_nr<uint8_t, 8, true, false>(param, NULL);
    }
}

#pragma endregion

#pragma region Banding

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

void reduce_banding_scale_param(BandingParam* prm, int depth)
{
    int shift = depth_scale_factor(depth);
    if (shift == 0) {
        return;
    }

    prm->ditherY <<= shift;
    prm->ditherC <<= shift;
    prm->threshold_y <<= shift;
    prm->threshold_cb <<= shift;
    prm->threshold_cr <<= shift;
}

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

#pragma endregion

#pragma region EdgeLevel

__constant__ EdgeLevelParam edge_prm;

void edgelevel_scale_param(EdgeLevelParam* prm, int depth)
{
    int shift = depth_scale_factor(depth);
    if (shift == 0) {
        return;
    }

    prm->str <<= shift;
    prm->thrs <<= shift;
    prm->bc <<= shift;
    prm->wc <<= shift;
}

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

#pragma endregion
