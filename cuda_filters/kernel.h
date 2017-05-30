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
	TemporalNRKernelParam(const cudafilter::TemporalNRParam& base)
		: cudafilter::TemporalNRParam(base) { }

	int nframes;
	int temporalWidth;
	cudafilter::FRAME_INFO frame_info;
};

void convert_yc_to_yca(cudafilter::PIXEL_YCA* yca, cudafilter::PIXEL_YC* yc, int pitch, int width, int height);
void convert_yca_to_yc(cudafilter::PIXEL_YC* yc, cudafilter::PIXEL_YCA* yca, int pitch, int width, int height);
void reduce_banding(cudafilter::BandingParam * prm, cudafilter::PIXEL_YCA* dev_dst, const cudafilter::PIXEL_YCA* dev_src, const uint8_t* dev_rand, cudaStream_t stream);
void edgelevel(cudafilter::EdgeLevelParam * prm, cudafilter::PIXEL_YCA * dev_dst, const cudafilter::PIXEL_YCA * dev_src, cudaStream_t stream);

