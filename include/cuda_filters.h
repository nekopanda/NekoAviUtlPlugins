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

	struct FrameYV12 {
		void* y;
		void* u;
		void* v;
	};

	struct FrameInfo {
		int depth;
		// 以下の3つはバイト数ではなく要素数
		int linesizeY;
		int linesizeU;
		int linesizeV;
	};

    struct Image {
        int pitch; // PIXEL_YCでのみ有効（PIXEL_YCAはパディングなし）
        int width;
        int height;
	};

    struct YUVtoYCAParam : Image {
        FrameInfo frame_info;
        int interlaced;
    };

    struct YCAtoYUVParam : YUVtoYCAParam {
        int src_depth;
        int dither;
        int seed;
        int rand_each_frame;
        int frame_number;
    };

    class YCAConverterInternal;

    class EXPORT YCAConverter {
    public:
        YCAConverter();
        ~YCAConverter();
        // src,dstはCUDAメモリ
        bool toYCA(YUVtoYCAParam* prm, FrameYV12 src, PIXEL_YCA* dst, CUstream_st* stream);
        bool toYUV(YCAtoYUVParam* prm, FrameYV12 dst, PIXEL_YCA* src, CUstream_st* stream);
        // src,dstはCPUメモリ
        bool toYUV(YCAtoYUVParam* prm, PIXEL_YC* src, PIXEL_YC* dst);
    private:
        YCAConverterInternal* data;
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
		int checkThresh;
		bool interlaced;
		bool check;
	};

	class TemporalNRInternal;

	class EXPORT TemporalNRFilter {
	public:
		TemporalNRFilter();
		~TemporalNRFilter();
		// src,dstはCUDAメモリ
		bool proc(TemporalNRParam* prm, 
            const FrameInfo* frame_info, const FrameYV12* src_frames,
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

    struct CudaFilterParams {
        bool enable_temporal_nr;
        bool enable_banding;
        bool enable_edgelevel;

        YCAtoYUVParam convert_param;
        TemporalNRParam temporal_nr_param;
        BandingParam banding_param;
        EdgeLevelParam edgelevel_param;
    };

    class CudaFiltersInternal;

    class EXPORT CudaFilters {
    public:
        CudaFilters(const CudaFilterParams* params);
        ~CudaFilters();

        bool sendFrame(const FrameInfo* frame_info, const FrameYV12* frame);
        int recvFrame();
        bool flush();
    private:
        CudaFiltersInternal* data;
    };

} // namespace cudafilter

#undef EXPORT
