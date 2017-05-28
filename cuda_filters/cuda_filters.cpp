#include "cuda_filters.h"

#include <Windows.h>
#include <stdio.h>
#include <cuda_runtime.h>

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

    class PixelYCA {
    public:
        PixelYCA(Image info, PIXEL_YC* src, PIXEL_YC* dst) : info(info), src(src), dst(dst) {
            int yc_size = info.pitch * info.height;
            int yca_size = info.width * info.height;

            CUDA_CHECK(cudaMalloc(&dev_src, yc_size * sizeof(PIXEL_YC)));
            CUDA_CHECK(cudaMalloc(&dev_dsrc, yca_size * sizeof(PIXEL_YCA)));
            CUDA_CHECK(cudaMalloc(&dev_ddst, yca_size * sizeof(PIXEL_YCA)));
            CUDA_CHECK(cudaMalloc(&dev_dst, yc_size * sizeof(PIXEL_YC)));

            CUDA_CHECK(cudaMemcpyAsync(dev_src, src, yc_size * sizeof(PIXEL_YC), cudaMemcpyHostToDevice));

            convert_yc_to_yca(dev_dsrc, dev_src, info.pitch, info.width, info.height);
        }
        ~PixelYCA() {
            int yc_size = info.pitch * info.height;

            convert_yca_to_yc(dev_dst, dev_ddst, info.pitch, info.width, info.height);

            CUDA_CHECK(cudaMemcpy(dst, dev_dst, yc_size * sizeof(PIXEL_YC), cudaMemcpyDeviceToHost));

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

    enum {
        MAX_RAND_N = 5,
        MAX_RAND_FRAMES = 16,
        FRAME_SKIP_LEN = 200,
    };

    class ReduceBandingInternal
    {
    public:
        ReduceBandingInternal(BandingParam *prm)
        {
            width = prm->width;
            height = prm->height;
            seed = prm->seed;

            int length = width * (height * MAX_RAND_N + MAX_RAND_FRAMES * FRAME_SKIP_LEN);
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

        ~ReduceBandingInternal()
        {
            CUDA_CHECK(cudaFree(dev_rand));
        }

        const uint8_t* getRand(int frame) const
        {
            if (rand_each_frame) {
                return dev_rand + width * FRAME_SKIP_LEN * (frame % MAX_RAND_FRAMES);
            }
            return dev_rand;
        }

        bool isSame(BandingParam *prm)
        {
            if (prm->width != width ||
                prm->height != height ||
                prm->seed != seed ||
                prm->rand_each_frame != rand_each_frame) {
                return false;
            }
            return true;
        }

    private:
        uint8_t *dev_rand;
        int width, height;
        int seed, rand_each_frame;
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

    bool ReduceBandingFilter::proc(BandingParam* prm, PIXEL_YCA* src, PIXEL_YCA* dst, CUstream_st* stream)
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
                reduce_banding(prm, pixelYCA.getdst(), pixelYCA.getsrc(), data->getRand(prm->frame_number), NULL);
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

    bool EdgeLevelFilter::proc(EdgeLevelParam* prm, PIXEL_YCA* src, PIXEL_YCA* dst, CUstream_st* stream)
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
