#include "cuda_filters.h"

#include <Windows.h>

#undef max
#undef min

#include <stdio.h>
#include <cuda_runtime.h>
#include <vector>
#include <deque>
#include <algorithm>

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

	TemporalNRKernelParam makeTemporalNRKernelParam(const TemporalNRParam& param, const FrameInfo* frame_info)
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
        TemporalNRParam* prm,
        const FrameInfo* frame_info, const FrameYV12* src_frames,
        PIXEL_YCA* const * dst_frames, CUstream_st* stream)
    {
        try {
            TIMER_START;
            auto paramex = makeTemporalNRKernelParam(*prm, frame_info);
            temporal_nr(paramex, src_frames, dst_frames, stream);
            CUDA_CHECK(cudaDeviceSynchronize());
            TIMER_END;
        }
        catch (const char*) {}
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

    class FrameBatcher
    {
    public:

        // temporalWidth: 1枚の出力に必要な枚数
        // batchSize: 1度のbatchで処理する出力枚数（最大）
        void init(int temporalWidth, int batchSize)
        {
            temporalWidth_ = temporalWidth;
            batchSize_ = batchSize;
            dup_first_ = temporalWidth_ / 2;
            frames_.clear();
        }

        void put(const FrameYV12* frame)
        {
		    frames_.push_back(*frame);

            int nframebuf = temporalWidth_ + batchSize_ - 1;

            if (dup_first_ + frames_.size() == nframebuf) {
                // バッファが全て埋まったので処理する
                launchBatch(dup_first_, 0, batchSize_);
                dup_first_ = std::max(0, dup_first_ - batchSize_);
            }
        }

        void finish()
        {
            int nframebuf = temporalWidth_ + batchSize_ - 1;
            int half = temporalWidth_ / 2;
            int remain = nframebuf - half;

            int dup_last = nframebuf - (dup_first_ + (int)frames_.size());
            for (; dup_last < remain; dup_last += batchSize_) {
                int validSize = std::min(batchSize_, remain - dup_last);
                launchBatch(dup_first_, dup_last, validSize);
                dup_first_ = std::max(0, dup_first_ - batchSize_);
            }
        }

        virtual void batch(std::vector<FrameYV12>& frames, int unique_offset, int num_batch) = 0;

    private:
        std::deque<FrameYV12> frames_;
        int temporalWidth_;
        int batchSize_;
        int dup_first_;

        void launchBatch(int dup_first, int dup_last, int num_out)
        {
            int nframebuf = temporalWidth_ + batchSize_ - 1;
            int half = temporalWidth_ / 2;
            std::vector<FrameYV12> frames(nframebuf);
            // 残りは全部lastで埋める
            dup_last = nframebuf - (dup_first + frames_.size());
            int fidx = 0;
            for (int i = 0; i < dup_first; ++i) {
                frames[fidx++] = frames_.front();
            }
            for (int i = 0; i < (int)frames_.size(); ++i) {
                frames[fidx++] = frames_[i];
            }
            for (int i = 0; i < dup_last; ++i) {
                frames[fidx++] = frames_.back();
            }

            batch(frames, half, num_out);

            for (int i = dup_first; i < num_out; ++i) {
                frames_.pop_front();
            }
        }
    };

    class CudaFiltersInternal : private FrameBatcher
    {
    public:
        CudaFiltersInternal(const CudaFilterParams* params)
            : prm(*params)
            , frameIntialized(false)
            , completeCount(0)
        {
            //
        }
        
        ~CudaFiltersInternal()
        {
            deinitFrame();
        }

        bool sendFrame(const FrameInfo* finfo, const FrameYV12* frame)
        {
            initFrame(finfo);

            if (finfo->depth != frame_info.depth ||
                finfo->linesizeY != frame_info.linesizeY ||
                finfo->linesizeU != frame_info.linesizeU ||
                finfo->linesizeV != frame_info.linesizeV)
            {
                THROW("フレームメモリが一貫していないためフィルタ処理できません");
            }

            // フレームメモリ割り当て
            if (yv_buffer.size() == 0) {
                // バッファがないので1枚完了を待つ
                waitOneFrame();
                completeOneFrame();
            }
            FrameData fdata;
            fdata.h = *frame;
            fdata.d = yv_buffer.back();
            yv_buffer.pop_back();
            batchedFrames.push_back(fdata);

            // GPUへ転送
            int bytes_per_pixel = (frame_info.depth > 8) ? 2 : 1;
            int fullH = prm.convert_param.height;
            int halfH = fullH >> 1;
            int sizeY = fullH * frame_info.linesizeY * bytes_per_pixel;
            int sizeU = halfH * frame_info.linesizeU * bytes_per_pixel;
            int sizeV = halfH * frame_info.linesizeV * bytes_per_pixel;
            CUDA_CHECK(cudaMemcpyAsync(fdata.d.y, fdata.h.y, sizeY, cudaMemcpyHostToDevice, stream[STM_SEND]));
            CUDA_CHECK(cudaMemcpyAsync(fdata.d.u, fdata.h.u, sizeU, cudaMemcpyHostToDevice, stream[STM_SEND]));
            CUDA_CHECK(cudaMemcpyAsync(fdata.d.v, fdata.h.v, sizeV, cudaMemcpyHostToDevice, stream[STM_SEND]));

            // フレーム追加
            FrameBatcher::put(&fdata.d);

            return true;
        }

        int recvFrame()
        {
            while (completeEvents.size()) {
                auto ret = cudaEventQuery(completeEvents[0]);
                if (ret == cudaErrorNotReady) {
                    // まだ完了していない
                    break;
                }
                CUDA_CHECK(ret);
                completeOneFrame();
            }

            int ret = completeCount;
            completeCount = 0;

            return ret;
        }

        bool flush()
        {
            FrameBatcher::finish();
            
            // 全てのフレームの完了を待つ
            while (completeEvents.size()) {
                waitOneFrame();
                completeOneFrame();
            }

            return true;
        }

    private:
        struct FrameData {
            // h: ホスト側, d: デバイス側
            FrameYV12 h, d;
        };

        CudaFilterParams prm;
        YCAConverter converter;
        TemporalNRFilter temporal_nr;
        ReduceBandingFilter banding;
        EdgeLevelFilter edgelevel;

        bool frameIntialized;

        // 入力フレームの情報（すべての入力フレームで一貫していないとダメ）
        FrameInfo frame_info;

        // 処理待ちフレーム
        std::deque<FrameData> batchedFrames;

        // 処理完了待ちフレーム
        std::deque<FrameData> waitingFrames;

        // デバイスメモリのプール
        std::vector<FrameYV12> yv_buffer;
        std::vector<PIXEL_YCA*> yca_buffer;

        // 完了イベント
        std::deque<cudaEvent_t> completeEvents;

        // まだ報告していない完了数
        int completeCount;

        enum {
            STM_SEND, STM_KERNEL, STM_RECV, STM_MAX
        };
        cudaStream_t stream[3];

        void initFrame(const FrameInfo* finfo)
        {
            if (frameIntialized) {
                return;
            }

            frame_info = *finfo;

            for (int i = 0; i < STM_MAX; ++i) {
                CUDA_CHECK(cudaStreamCreate(&stream[i]));
            }

            // フレームプールを作成
            int temporalWidth = prm.enable_temporal_nr
                ? (prm.temporal_nr_param.temporalDistance * 2 + 1)
                : 1;
            int batchSzie = prm.enable_temporal_nr
                ? prm.temporal_nr_param.batchSize
                : 1;
            int max_frames = batchSzie * 2 + temporalWidth + batchSzie + 4;
            for (int i = 0; i < max_frames; ++i) {
                yv_buffer.push_back(allocYVFrame());
            }
            for (int i = 0; i < batchSzie + 1; ++i) {
                yca_buffer.push_back(allocYCFrame());
            }

            FrameBatcher::init(temporalWidth, batchSzie);

            completeCount = 0;

            frameIntialized = true;
        }

        void deinitFrame()
        {
            if (!frameIntialized) {
                return;
            }

            for (int i = 0; i < STM_MAX; ++i) {
                CUDA_CHECK(cudaStreamDestroy(stream[i]));
                stream[i] = NULL;
            }
            
            // 本当はないはずだが、フレームが残っていたら消す
            for (int i = 0; i < (int)batchedFrames.size(); ++i) {
                yv_buffer.push_back(batchedFrames[i].d);
            }
            batchedFrames.clear();
            for (int i = 0; i < (int)waitingFrames.size(); ++i) {
                yv_buffer.push_back(waitingFrames[i].d);
            }
            waitingFrames.clear();

            // フレームプールを削除
            for (int i = 0; i < (int)yv_buffer.size(); ++i) {
                deallocYVFrame(yv_buffer[i]);
            }
            yv_buffer.clear();
            for (int i = 0; i < (int)yca_buffer.size(); ++i) {
                deallocYCFrame(yca_buffer[i]);
            }
            yca_buffer.clear();

            frameIntialized = false;
        }

        FrameYV12 allocYVFrame()
        {
            FrameYV12 frame = {};
            int bytes_per_pixel = (frame_info.depth > 8) ? 2 : 1;
            int fullH = prm.convert_param.height;
            int halfH = fullH >> 1;
            int sizeY = fullH * frame_info.linesizeY * bytes_per_pixel;
            int sizeU = halfH * frame_info.linesizeU * bytes_per_pixel;
            int sizeV = halfH * frame_info.linesizeV * bytes_per_pixel;
            CUDA_CHECK(cudaMalloc(&frame.y, sizeY + sizeU + sizeV));
            frame.u = (char*)frame.y + sizeY;
            frame.v = (char*)frame.u + sizeU;
        }

        void deallocYVFrame(FrameYV12 frame)
        {
            CUDA_CHECK(cudaFree(frame.y));
        }

        PIXEL_YCA* allocYCFrame()
        {
            int width = prm.convert_param.width;
            int height = prm.convert_param.height;
            PIXEL_YCA* ptr = NULL;
            CUDA_CHECK(cudaMalloc((void**)&ptr, width * height * sizeof(PIXEL_YCA)));
            return ptr;
        }

        void deallocYCFrame(PIXEL_YCA* ptr)
        {
            CUDA_CHECK(cudaFree(ptr));
        }

        void waitOneFrame()
        {
            if (completeEvents.size() == 0) {
                THROW("処理が開始していないのでフレームを待つことができません");
            }
            CUDA_CHECK(cudaEventSynchronize(completeEvents[0]));
        }

        void completeOneFrame()
        {
            // イベントは用済みなので削除
            CUDA_CHECK(cudaEventDestroy(completeEvents[0]));
            completeEvents.pop_front();

            // デバイスメモリを回収
            yv_buffer.push_back(waitingFrames.front().d);
            
            // フレーム削除
            waitingFrames.pop_front();

            ++completeCount;
        }

        // 時間軸ノイズ除去以外のフィルタを適用
        void applyFrameFilter(PIXEL_YCA* yca_data, PIXEL_YCA* yca_tmp, FrameYV12 dst, cudaStream_t stream)
        {
            if (prm.enable_edgelevel) {
                edgelevel.proc(&prm.edgelevel_param, yca_data, yca_tmp, stream);
                std::swap(yca_data, yca_tmp);
            }
            if (prm.enable_banding) {
                banding.proc(&prm.banding_param, yca_data, yca_tmp, stream);
                std::swap(yca_data, yca_tmp);
            }
            converter.toYUV(&prm.convert_param, dst, yca_data, stream);
        }

        virtual void batch(std::vector<FrameYV12>& frames, int unique_offset, int num_batch)
        {
            PIXEL_YCA* yca_data = yca_buffer.front();
            PIXEL_YCA* yca_tmp = yca_buffer.back();

            cudaStream_t stSend = stream[STM_SEND];
            cudaStream_t stKernel = stream[STM_KERNEL];
            cudaStream_t stRecv = stream[STM_KERNEL];

            cudaEvent_t ev;
            CUDA_CHECK(cudaEventCreate(&ev));

            // stSend -> stKernel の順番付け
            CUDA_CHECK(cudaEventRecord(ev, stSend));
            CUDA_CHECK(cudaStreamWaitEvent(stKernel, ev, 0));

            // フィルタ処理実行
            if (prm.enable_temporal_nr) {
                if (frames.size() < unique_offset * 2 + num_batch) {
                    THROW("バッチフレーム数が合いません");
                }
                temporal_nr.proc(&prm.temporal_nr_param, &frame_info, frames.data(), yca_buffer.data(), stKernel);
                for (int i = 0; i < num_batch; ++i) {
                    applyFrameFilter(yca_buffer[i], yca_tmp, frames[unique_offset + i], stKernel);
                }
            }
            else {
                if (unique_offset > 0) {
                    THROW("時間軸ノイズ除去が無効なのに余分なフレームがあります");
                }
                for (int i = 0; i < num_batch; ++i) {
                    converter.toYCA(&prm.convert_param, frames[unique_offset + i], yca_data, stKernel);
                    applyFrameFilter(yca_data, yca_tmp, frames[unique_offset + i], stKernel);
                }
            }

            // stKernel -> stRecv の順番付け
            CUDA_CHECK(cudaEventRecord(ev, stKernel));
            CUDA_CHECK(cudaStreamWaitEvent(stRecv, ev, 0));

            // CPUへ転送
            for (int i = 0; i < num_batch; ++i) {
                // チェック
                if (frames[unique_offset + i].y != batchedFrames.front().d.y) {
                    THROW("バッチフレームと合いません");
                }

                auto fdata = batchedFrames.front();

                int bytes_per_pixel = (frame_info.depth > 8) ? 2 : 1;
                int fullH = prm.convert_param.height;
                int halfH = fullH >> 1;
                int sizeY = fullH * frame_info.linesizeY * bytes_per_pixel;
                int sizeU = halfH * frame_info.linesizeU * bytes_per_pixel;
                int sizeV = halfH * frame_info.linesizeV * bytes_per_pixel;
                CUDA_CHECK(cudaMemcpyAsync(fdata.h.y, fdata.d.y, sizeY, cudaMemcpyDeviceToHost, stRecv));
                CUDA_CHECK(cudaMemcpyAsync(fdata.h.u, fdata.d.u, sizeU, cudaMemcpyDeviceToHost, stRecv));
                CUDA_CHECK(cudaMemcpyAsync(fdata.h.v, fdata.d.v, sizeV, cudaMemcpyDeviceToHost, stRecv));

                waitingFrames.push_back(fdata);
                batchedFrames.pop_front();

                // 完了イベントを追加
                cudaEvent_t frameCompleteEvent;
                CUDA_CHECK(cudaEventCreate(&frameCompleteEvent));
                CUDA_CHECK(cudaEventRecord(frameCompleteEvent, stRecv));
                completeEvents.push_back(frameCompleteEvent);
            }

            CUDA_CHECK(cudaEventDestroy(ev));
        }
    };

#pragma region CudaFilters

    CudaFilters::CudaFilters(const CudaFilterParams* params)
    {
        data = new CudaFiltersInternal(params);
    }

    CudaFilters::~CudaFilters()
    {
        delete data;
        data = NULL;
    }

    bool CudaFilters::sendFrame(const FrameInfo* frame_info, const FrameYV12* frame)
    {
        try {
            return data->sendFrame(frame_info, frame);
        }
        catch (const char*) {}
        return false;
    }

    int CudaFilters::recvFrame()
    {
        try {
            return data->recvFrame();
        }
        catch (const char*) {}
        return -1;
    }

    bool CudaFilters::flush()
    {
        try {
            return data->flush();
        }
        catch (const char*) {}
        return false;
    }

#pragma endregion

} // namespace cudafilter
