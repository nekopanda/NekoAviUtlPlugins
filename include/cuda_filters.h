#pragma once

#include <stdint.h>

#ifdef __CUDA_FILTER_EXPORT__
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __declspec(dllimport)
#endif

struct CUstream_st;

namespace cudafilter {

    struct PIXEL_YC {
        short    y;
        short    cb;
        short    cr;
    };

    struct PIXEL_YCA {
        short    y;
        short    cb;
        short    cr;
        short    a; // 使用しない
    };

	struct FRAME_YV12 {
		unsigned char* y;
		unsigned char* u;
		unsigned char* v;
	};

	struct FRAME_INFO {
		int depth;
		// 以下の3つはバイト数ではなく要素数
		int linesizeY;
		int linesizeU;
		int linesizeV;
	};

    struct Image {
        int pitch;
        int width;
        int height;
	};

	enum TEMPORAL_NR_CONST {
		TEMPNR_MAX_DIST = 31,
		TEMPNR_MAX_BATCH = 16
	};

	struct TemporalNRParam : Image {
		int temporalDistance;
		int threshY;
		int threshCb;
		int threshCr;
		int batchSize;
		int thresh;
		bool interlaced;
		bool check;
	};

	class TemporalNRInternal;

	class EXPORT TemporalNRFilter {
	public:
		TemporalNRFilter();
		~TemporalNRFilter();
		// src,dstはCUDAメモリ
		bool proc(TemporalNRParam* param,
			const FRAME_INFO* frame_info, const FRAME_YV12* src_frames,
			PIXEL_YCA* const * dst_frames, CUstream_st* stream);
		// src,dstはCPUメモリ
		bool proc(TemporalNRParam* param,
			PIXEL_YC* const * src_frames, PIXEL_YC* const * dst_frames);
	private:
		TemporalNRInternal* data;
	};

    struct BandingParam : Image {
        int check;
        int seed;
        int ditherY;
        int ditherC;
        int rand_each_frame;
        int sample_mode;
        int blur_first;
        int yc_and;
        int range;
        int threshold_y;
        int threshold_cb;
        int threshold_cr;
        int interlaced;
        int frame_number;
    };

    class ReduceBandingInternal;

    class EXPORT ReduceBandingFilter {
    public:
        ReduceBandingFilter();
        ~ReduceBandingFilter();
        // src,dstはCUDAメモリ
        bool proc(BandingParam* param,
			PIXEL_YCA* src, PIXEL_YCA* dst, CUstream_st* stream);
        // src,dstはCPUメモリ
        bool proc(BandingParam* param,
			PIXEL_YC* src, PIXEL_YC* dst);
    private:
        ReduceBandingInternal* data;
    };

    struct EdgeLevelParam : Image {
        int check;
        int str;
        int thrs;
        int bc;
        int wc;
        int interlaced;
    };

    class EdgeLevelInternal;

    class EXPORT EdgeLevelFilter {
    public:
        EdgeLevelFilter();
        ~EdgeLevelFilter();
        // src,dstはCUDAメモリ
        bool proc(EdgeLevelParam* param,
			PIXEL_YCA* src, PIXEL_YCA* dst, CUstream_st* stream);
        // src,dstはCPUメモリ
        bool proc(EdgeLevelParam* param,
			PIXEL_YC* src, PIXEL_YC* dst);
    private:
        EdgeLevelInternal* data;
    };

} // namespace cudafilter

#undef EXPORT
