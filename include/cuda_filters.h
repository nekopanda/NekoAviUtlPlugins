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
        short    a; // �g�p���Ȃ�
    };

    struct Image {
        int pitch;
        int width;
        int height;
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
        // src,dst��CUDA������
        bool proc(BandingParam* param, PIXEL_YCA* src, PIXEL_YCA* dst, CUstream_st* stream);
        // src,dst��CPU������
        bool proc(BandingParam* param, PIXEL_YC* src, PIXEL_YC* dst);
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
        // src,dst��CUDA������
        bool proc(EdgeLevelParam* param, PIXEL_YCA* src, PIXEL_YCA* dst, CUstream_st* stream);
        // src,dst��CPU������
        bool proc(EdgeLevelParam* param, PIXEL_YC* src, PIXEL_YC* dst);
    private:
        EdgeLevelInternal* data;
    };

} // namespace cudafilter

#undef EXPORT
