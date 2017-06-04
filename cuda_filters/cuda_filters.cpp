#include "cuda_filters.h"

#include <Windows.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <vector>

#include "kernel.h"
#include "xor_rand.h"
#include "plugin_utils.h"

namespace cudafilter {

    class PerformanceTimer
    {
        enum {
            PRINT_CYCLE = 10,
            MAX_SECTIONS = 5
        };
    public:
        PerformanceTimer()
            : times()
            , cycle()
        { }
        void start() {
            section = 0;
            QueryPerformanceCounter((LARGE_INTEGER*)&prev);
        }
        void next() {
            int64_t now;
            QueryPerformanceCounter((LARGE_INTEGER*)&now);
            times[cycle][section++] = now - prev;
            prev = now;
        }
        void end() {
            next();
            if (++cycle == PRINT_CYCLE) {
                print();
                cycle = 0;
            }
        }
    private:
        int64_t times[PRINT_CYCLE][MAX_SECTIONS];
        int cycle;
        int section;
        int64_t prev;

        void print() {
            int64_t freq;
            QueryPerformanceFrequency((LARGE_INTEGER*)&freq);

            double total = 0.0;
            for (int i = 0; i < MAX_SECTIONS; ++i) {
                int64_t sum = 0;
                for (int c = 0; c < PRINT_CYCLE; ++c) {
                    sum += times[c][i];
                }
                double avg = (double)sum / freq * 1000.0;
                total += avg;
                printf("%2d: %f ms\n", i, avg);
            }
            printf("total: %f ms\n", total);
        }
    };

#ifdef ENABLE_PERF
    PerformanceTimer* g_timer = NULL;
#define TIMER_START g_timer->start()
#define TIMER_NEXT CUDA_CHECK(cudaDeviceSynchronize()); g_timer->next()
#define TIMER_END CUDA_CHECK(cudaDeviceSynchronize()); g_timer->end()
#else
#define TIMER_START
#define TIMER_NEXT
#define TIMER_END
#endif

    class RandomSource
    {
    public:
        RandomSource(
            int width, int height, int seed, bool rand_each_frame,
            int max_per_pixel, int max_frames, int frame_skip_len)
            : width(width)
            , height(height)
            , seed(seed)
            , rand_each_frame(rand_each_frame)
            , max_per_pixel(max_per_pixel)
            , max_frames(max_frames)
            , frame_skip_len(frame_skip_len)
        {
            int length = width * (height * max_per_pixel + max_frames * frame_skip_len);
            CUDA_CHECK(cudaMalloc((void**)&dev_rand, length));

            uint8_t* rand_buf = (uint8_t*)malloc(length);

            xor128_t xor;
            xor128_init(&xor, seed);

            int i = 0;
            for (; i <= length - 4; i += 4) {
                xor128(&xor);
                *(uint32_t*)(rand_buf + i) = xor.w;
            }
            if (i < length) {
                xor128(&xor);
                memcpy(&rand_buf[i], &xor.w, length - i);
            }

            CUDA_CHECK(cudaMemcpy(dev_rand, rand_buf, length, cudaMemcpyHostToDevice));

            free(rand_buf);
        }

        ~RandomSource()
        {
            CUDA_CHECK(cudaFree(dev_rand));
        }

        const uint8_t* getRand(int frame) const
        {
            if (rand_each_frame) {
                return dev_rand + width * frame_skip_len * (frame % max_frames);
            }
            return dev_rand;
        }

        bool isSame(int owidth, int oheight, int oseed, bool orand_each_frame)
        {
            if (owidth != width ||
                oheight != height ||
                oseed != seed ||
                orand_each_frame != rand_each_frame) {
                return false;
            }
            return true;
        }

    private:
        uint8_t *dev_rand;
        int width, height, seed;
        int max_per_pixel;
        int max_frames;
        int frame_skip_len;
        bool rand_each_frame;
    };

    class PixelYCA {
    public:
        PixelYCA(Image info, PIXEL_YC* src, PIXEL_YC* dst)
            : info(info), src(src), dst(dst)
        {
            int yc_size = info.pitch * info.height;
            int yca_size = info.width * info.height;

            CUDA_CHECK(cudaMalloc(&dev_src, yc_size * sizeof(PIXEL_YC)));
            CUDA_CHECK(cudaMalloc(&dev_dsrc, yca_size * sizeof(PIXEL_YCA)));
            CUDA_CHECK(cudaMalloc(&dev_ddst, yca_size * sizeof(PIXEL_YCA)));
            CUDA_CHECK(cudaMalloc(&dev_dst, yc_size * sizeof(PIXEL_YC)));

            CUDA_CHECK(cudaMemcpyAsync(
                dev_src, src, yc_size * sizeof(PIXEL_YC), cudaMemcpyHostToDevice));

            convert_yc_to_yca(dev_dsrc, dev_src, info.pitch, info.width, info.height);
        }
        ~PixelYCA() {
            int yc_size = info.pitch * info.height;

            convert_yca_to_yc(dev_dst, dev_ddst, info.pitch, info.width, info.height);

            CUDA_CHECK(cudaMemcpy(
                dst, dev_dst, yc_size * sizeof(PIXEL_YC), cudaMemcpyDeviceToHost));

            CUDA_CHECK(cudaFree(dev_src));
            CUDA_CHECK(cudaFree(dev_dsrc));
            CUDA_CHECK(cudaFree(dev_ddst));
            CUDA_CHECK(cudaFree(dev_dst));
        }
        PIXEL_YCA* getsrc() { return (PIXEL_YCA*)dev_dsrc; }
        PIXEL_YCA* getdst() { return (PIXEL_YCA*)dev_ddst; }
    private:
        Image info;
        PIXEL_YC *src, *dst, *dev_src, *dev_dst;
        PIXEL_YCA *dev_dsrc, *dev_ddst;
    };

    class YCAConverterInternal
    {
    public:
        YCAConverterInternal(YCAtoYUVParam* prm)
            : rand(prm->width, prm->height, prm->seed, prm->rand_each_frame != 0, 3, 16, 200)
        { }
        const uint8_t* getRand(int frame) const {
            return rand.getRand(frame);
        }
        bool isSame(YCAtoYUVParam* prm) {
            return rand.isSame(prm->width, prm->height, prm->seed, prm->rand_each_frame != 0);
        }
    private:
        RandomSource rand;
    };

#pragma region YCAConverter

    YCAConverter::YCAConverter()
        : data(NULL)
    {
#ifdef ENABLE_PERF
        if (g_timer == NULL) {
            init_console();
            g_timer = new PerformanceTimer();
        }
#endif
    }

    YCAConverter::~YCAConverter()
    {
        if (data != NULL) {
            delete data;
            data = NULL;
        }
    }

    bool YCAConverter::toYCA(YUVtoYCAParam* prm, FrameYV12 src, PIXEL_YCA* dst, CUstream_st* stream)
    {
        try {
            TIMER_START;
            convert_yuv_to_yca(prm, src, dst, stream);
            TIMER_END;
            return true;
        }
        catch (const char*) {}
        return false;
    }

    bool YCAConverter::toYUV(YCAtoYUVParam* prm, FrameYV12 dst, PIXEL_YCA* src, CUstream_st* stream)
    {
        try {
            TIMER_START;
            if (data == NULL) {
                data = new YCAConverterInternal(prm);
            }
            else if (data->isSame(prm) == false) {
                delete data;
                data = new YCAConverterInternal(prm);
            }
            TIMER_NEXT;
            convert_yca_to_yuv(prm, dst, src, data->getRand(prm->frame_number), stream);
            TIMER_END;
            return true;
        }
        catch (const char*) {}
        return false;
    }

    bool YCAConverter::toYUV(YCAtoYUVParam* prm_, PIXEL_YC* src, PIXEL_YC* dst)
    {
        try {
            TIMER_START;
            YCAtoYUVParam prm = *prm_;
            prm.frame_info.depth = 8;
            prm.frame_info.linesizeY = prm.width;
            prm.frame_info.linesizeU = prm.width / 2;
            prm.frame_info.linesizeV = prm.width / 2;
            PixelYCA pixelYCA(prm, src, dst);
            TIMER_NEXT;
            if (data == NULL) {
                data = new YCAConverterInternal(&prm);
            }
            else if (data->isSame(prm_) == false) {
                delete data;
                data = new YCAConverterInternal(&prm);
            }
            TIMER_NEXT;
            FrameYV12 yuv;
            CUDA_CHECK(cudaMalloc((void**)&yuv.y, prm.width * (prm.height * 3 / 2)));
            yuv.u = (uint8_t*)yuv.y + prm.width * prm.height;
            yuv.v = (uint8_t*)yuv.u + (prm.width/2) * (prm.height/2);
            TIMER_NEXT;
            //cudaMemcpy(pixelYCA.getdst(), pixelYCA.getsrc(), prm.width*prm.height*sizeof(PIXEL_YCA), cudaMemcpyDeviceToDevice);
            convert_yca_to_yuv(&prm, yuv, pixelYCA.getsrc(), data->getRand(prm.frame_number), NULL);
            convert_yuv_to_yca(&prm, yuv, pixelYCA.getdst(), NULL);
            TIMER_NEXT;
            CUDA_CHECK(cudaFree(yuv.y));
            TIMER_END;
            return true;
        }
        catch (const char*) {}
        return false;
    }

#pragma endregion

	class TemporalNRInternal {
	public:
		TemporalNRInternal(const TemporalNRParam& param)
			: param(param)
		{
			frame_size = param.pitch * param.height;
			allocFrames(dev_src_frames,
				param.batchSize + param.temporalDistance * 2, frame_size);
			allocFrames(dev_dst_frames,
				param.batchSize, frame_size);
		}
		~TemporalNRInternal() {
			freeFrames(dev_src_frames);
			freeFrames(dev_dst_frames);
		}
		int getFrameSize() const {
			return frame_size;
		}
		const std::vector<PIXEL_YC*>& getSrcFrames() const {
			return dev_src_frames;
		}
		const std::vector<PIXEL_YC*>& getDstFrames() const {
			return dev_dst_frames;
		}
		bool isSame(TemporalNRParam *prm)
		{
			if (prm->pitch != param.pitch ||
				prm->height != param.height ||
				prm->batchSize != param.batchSize ||
				prm->temporalDistance != param.temporalDistance) {
				return false;
			}
			return true;
		}

	private:
		const TemporalNRParam param;
		int frame_size;
		std::vector<PIXEL_YC*> dev_src_frames;
		std::vector<PIXEL_YC*> dev_dst_frames;

		static void allocFrames(
			std::vector<PIXEL_YC*>& frames, int nframes, int frame_size)
		{
			for (int i = 0; i < nframes; ++i) {
				void* ptr;
				CUDA_CHECK(cudaMalloc(&ptr, frame_size * sizeof(PIXEL_YC)));
				frames.push_back((PIXEL_YC*)ptr);
			}
		}

		static void freeFrames(std::vector<PIXEL_YC*>& frames) {
			for (int i = 0; i < (int)frames.size(); ++i) {
				CUDA_CHECK(cudaFree(frames[i]));
			}
			frames.clear();
		}
	};

#pragma region TemporalNRFilter

	TemporalNRKernelParam makeTemporalNRKernelParam(const TemporalNRParam& param, FrameInfo* frame_info)
	{
		TemporalNRKernelParam ret = param;
		ret.nframes = param.batchSize + param.temporalDistance * 2;
		ret.temporalWidth = param.temporalDistance * 2 + 1;
		if (frame_info) {
			ret.frame_info = *frame_info;
		}
		return ret;
	}

	TemporalNRFilter::TemporalNRFilter()
		: data(NULL)
	{
#ifdef ENABLE_PERF
		if (g_timer == NULL) {
			init_console();
			g_timer = new PerformanceTimer();
		}
#endif
	}
	TemporalNRFilter::~TemporalNRFilter()
	{
		if (data != NULL) {
			delete data;
			data = NULL;
		}
	}
	bool TemporalNRFilter::proc(
		TemporalNRParam* prm, const FrameYV12* src_frames,
		PIXEL_YCA* const * dst_frames, CUstream_st* stream)
	{
		// TODO:
        return false;
	}
	bool TemporalNRFilter::proc(
		TemporalNRParam* prm, PIXEL_YC* const * src_frames,
		PIXEL_YC* const * dst_frames)
	{
		try {
			TIMER_START;
			if (data == NULL) {
				data = new TemporalNRInternal(*prm);
			}
			else if (data->isSame(prm) == false) {
				delete data;
				data = new TemporalNRInternal(*prm);
			}
			TIMER_NEXT;
			int frame_bytes = data->getFrameSize() * sizeof(PIXEL_YC);
			auto& dev_src_frames = data->getSrcFrames();
			auto& dev_dst_frames = data->getDstFrames();
			for (int i = 0; i < (int)dev_src_frames.size(); ++i) {
				CUDA_CHECK(cudaMemcpyAsync(
					dev_src_frames[i], src_frames[i], frame_bytes, cudaMemcpyHostToDevice));
			}
			TIMER_NEXT;
			auto paramex = makeTemporalNRKernelParam(*prm, NULL);
            
            temporal_nr(paramex, dev_src_frames.data(), dev_dst_frames.data());
			
            TIMER_NEXT;
			for (int i = 0; i < (int)dev_dst_frames.size(); ++i) {
				CUDA_CHECK(cudaMemcpyAsync(
					dst_frames[i], dev_dst_frames[i], frame_bytes, cudaMemcpyDeviceToHost));
			}
			CUDA_CHECK(cudaDeviceSynchronize());
			TIMER_END;
			return true;
		}
		catch (const char*) {}
		return false;
	}

#pragma endregion

    class ReduceBandingInternal
    {
    public:
        ReduceBandingInternal(BandingParam* prm)
            : rand(prm->width, prm->height, prm->seed, prm->rand_each_frame != 0, 5, 16, 200)
        { }
        const uint8_t* getRand(int frame) const {
            return rand.getRand(frame);
        }
        bool isSame(BandingParam* prm) {
            return rand.isSame(prm->width, prm->height, prm->seed, prm->rand_each_frame != 0);
        }
    private:
        RandomSource rand;
    };

#pragma region ReduceBandingFilter

    ReduceBandingFilter::ReduceBandingFilter()
        : data(NULL)
    {
#ifdef ENABLE_PERF
        if (g_timer == NULL) {
            init_console();
            g_timer = new PerformanceTimer();
        }
#endif
    }

    ReduceBandingFilter::~ReduceBandingFilter()
    {
        if (data != NULL) {
            delete data;
            data = NULL;
        }
    }

    bool ReduceBandingFilter::proc(
		BandingParam* prm, PIXEL_YCA* src, PIXEL_YCA* dst, CUstream_st* stream)
    {
        try {
            TIMER_START;
            if (data == NULL) {
                data = new ReduceBandingInternal(prm);
            }
            else if (data->isSame(prm) == false) {
                delete data;
                data = new ReduceBandingInternal(prm);
            }
            TIMER_NEXT;
            reduce_banding(prm, dst, src, data->getRand(prm->frame_number), stream);
            TIMER_END;
            return true;
        }
        catch (const char*) {}
        return false;
    }

    bool ReduceBandingFilter::proc(BandingParam* prm, PIXEL_YC* src, PIXEL_YC* dst)
    {
        try {
            TIMER_START;
            {
                PixelYCA pixelYCA(*prm, src, dst);
                TIMER_NEXT;
                //proc(prm, pixelYCA.getsrc(), pixelYCA.getdst());
                if (data == NULL) {
                    data = new ReduceBandingInternal(prm);
                }
                else if (data->isSame(prm) == false) {
                    delete data;
                    data = new ReduceBandingInternal(prm);
                }
                TIMER_NEXT;
                reduce_banding(prm, pixelYCA.getdst(),
					pixelYCA.getsrc(), data->getRand(prm->frame_number), NULL);
                TIMER_NEXT;
            }
            TIMER_END;
            return true;
        }
        catch (const char*) {}
        return false;
    }

#pragma endregion

    class EdgeLevelInternal
    {
    };

#pragma region EdgeLevelFilter

    EdgeLevelFilter::EdgeLevelFilter()
        : data(NULL)
    {
#ifdef ENABLE_PERF
        if (g_timer == NULL) {
            init_console();
            g_timer = new PerformanceTimer();
        }
#endif
    }

    EdgeLevelFilter::~EdgeLevelFilter()
    {
        // do nothing
    }

    bool EdgeLevelFilter::proc(
		EdgeLevelParam* prm, PIXEL_YCA* src, PIXEL_YCA* dst, CUstream_st* stream)
    {
        try {
            TIMER_START;
            edgelevel(prm, dst, src, stream);
            TIMER_END;
            return true;
        }
        catch (const char*) {}
        return false;
    }

    bool EdgeLevelFilter::proc(EdgeLevelParam* prm, PIXEL_YC* src, PIXEL_YC* dst)
    {
        try {
            TIMER_START;
            {
                PixelYCA pixelYCA(*prm, src, dst);
                TIMER_NEXT;
                edgelevel(prm, pixelYCA.getdst(), pixelYCA.getsrc(), NULL);
                TIMER_NEXT;
            }
            TIMER_END;
            return true;
        }
        catch (const char*) {}
        return false;
    }

#pragma endregion

} // namespace cudafilter
