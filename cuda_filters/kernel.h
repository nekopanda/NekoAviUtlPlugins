#pragma once

#include <stdint.h>
#include <stdarg.h>
#include <stdio.h>

#include "cuda_filters.h"

#define ENABLE_PERF

#define THROW(message) \
  throw_exception_("Exception thrown at %s:%d\r\nMessage: " message, __FILE__, __LINE__)

#define THROWF(fmt, ...) \
  throw_exception_("Exception thrown at %s:%d\r\nMessage: " fmt, __FILE__, __LINE__, __VA_ARGS__)

static void throw_exception_(const char* fmt, ...)
{
    char buf[300];
    va_list arg;
    va_start(arg, fmt);
    vsnprintf_s(buf, sizeof(buf), fmt, arg);
    va_end(arg);
    printf(buf);
    throw buf;
}

#define CUDA_CHECK(call) \
		do { \
			cudaError_t err__ = call; \
			if (err__ != cudaSuccess) { \
				THROWF("[CUDA Error] %d: %s", err__, cudaGetErrorString(err__)); \
			} \
		} while (0)


struct TemporalNRKernelParam : cudafilter::TemporalNRParam {
    TemporalNRKernelParam() { }

	TemporalNRKernelParam(const cudafilter::TemporalNRParam& base)
		: cudafilter::TemporalNRParam(base) { }

	int nframes;
	int temporalWidth;
	cudafilter::FrameInfo frame_info;
};

// PIXEL_YC <-> PIXEL_YCA
void convert_yc_to_yca(cudafilter::PIXEL_YCA* yca, cudafilter::PIXEL_YC* yc, int pitch, int width, int height);
void convert_yca_to_yc(cudafilter::PIXEL_YC* yc, cudafilter::PIXEL_YCA* yca, int pitch, int width, int height);

// FrameYV12 <-> PIXEL_YCA
void convert_yuv_to_yca(cudafilter::YUVtoYCAParam* prm, cudafilter::FrameYV12 yuv, cudafilter::PIXEL_YCA* yca, cudaStream_t stream);
void convert_yca_to_yuv(cudafilter::YCAtoYUVParam* prm, cudafilter::FrameYV12 yuv, cudafilter::PIXEL_YCA* yca, const uint8_t* rand, cudaStream_t stream);

// 時間軸ノイズリダクション
void temporal_nr_scale_param(cudafilter::TemporalNRParam* prm, int depth);
void temporal_nr(const TemporalNRKernelParam& param, const cudafilter::FrameYV12* src_frames, cudafilter::PIXEL_YCA* const * dst_frames, cudaStream_t stream);
void temporal_nr(const TemporalNRKernelParam& param, cudafilter::PIXEL_YC* const * src_frames, cudafilter::PIXEL_YC* const * dst_frames);

// バンディング低減
void reduce_banding_scale_param(cudafilter::BandingParam* prm, int depth);
void reduce_banding(cudafilter::BandingParam * prm, cudafilter::PIXEL_YCA* dev_dst, const cudafilter::PIXEL_YCA* dev_src, const uint8_t* dev_rand, cudaStream_t stream);

// エッジレベル調整
void edgelevel_scale_param(cudafilter::EdgeLevelParam* prm, int depth);
void edgelevel(cudafilter::EdgeLevelParam * prm, cudafilter::PIXEL_YCA * dev_dst, const cudafilter::PIXEL_YCA * dev_src, cudaStream_t stream);

