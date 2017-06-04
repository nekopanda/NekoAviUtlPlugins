// YUV変換出力確認用フィルタ

#include <windows.h>

#undef max
#undef min

#include <vector>
#include <algorithm>

#define DEFINE_GLOBAL
#include "filter.h"
#include "plugin_utils.h"
#include "cuda_filters.h"

using cudafilter::YCAtoYUVParam;
using cudafilter::YCAConverter;

//---------------------------------------------------------------------
//        フィルタ構造体定義
//---------------------------------------------------------------------
#define    TRACK_N    1                                                                                    //  トラックバーの数
TCHAR    *track_name[] = { "seed" };    //  トラックバーの名前
int        track_default[] = { 0 };    //  トラックバーの初期値
int        track_s[] = { 0 };    //  トラックバーの下限値
int        track_e[] = { 63 };    //  トラックバーの上限値

#define    CHECK_N    4                                                                   //  チェックボックスの数
TCHAR    *check_name[] = { "フィールド処理", "ディザ", "毎フレーム乱数を生成", "CUDA" }; //  チェックボックスの名前
int         check_default[] = { 1, 1, 0, 1 };    //  チェックボックスの初期値 (値は0か1)

FILTER_DLL filter = {
    FILTER_FLAG_EX_INFORMATION,    //    フィルタのフラグ
                                   //    FILTER_FLAG_ALWAYS_ACTIVE        : フィルタを常にアクティブにします
                                   //    FILTER_FLAG_CONFIG_POPUP        : 設定をポップアップメニューにします
                                   //    FILTER_FLAG_CONFIG_CHECK        : 設定をチェックボックスメニューにします
                                   //    FILTER_FLAG_CONFIG_RADIO        : 設定をラジオボタンメニューにします
                                   //    FILTER_FLAG_EX_DATA                : 拡張データを保存出来るようにします。
                                   //    FILTER_FLAG_PRIORITY_HIGHEST    : フィルタのプライオリティを常に最上位にします
                                   //    FILTER_FLAG_PRIORITY_LOWEST        : フィルタのプライオリティを常に最下位にします
                                   //    FILTER_FLAG_WINDOW_THICKFRAME    : サイズ変更可能なウィンドウを作ります
                                   //    FILTER_FLAG_WINDOW_SIZE            : 設定ウィンドウのサイズを指定出来るようにします
                                   //    FILTER_FLAG_DISP_FILTER            : 表示フィルタにします
                                   //    FILTER_FLAG_EX_INFORMATION        : フィルタの拡張情報を設定できるようにします
                                   //    FILTER_FLAG_NO_CONFIG            : 設定ウィンドウを表示しないようにします
                                   //    FILTER_FLAG_AUDIO_FILTER        : オーディオフィルタにします
                                   //    FILTER_FLAG_RADIO_BUTTON        : チェックボックスをラジオボタンにします
                                   //    FILTER_FLAG_WINDOW_HSCROLL        : 水平スクロールバーを持つウィンドウを作ります
                                   //    FILTER_FLAG_WINDOW_VSCROLL        : 垂直スクロールバーを持つウィンドウを作ります
                                   //    FILTER_FLAG_IMPORT                : インポートメニューを作ります
                                   //    FILTER_FLAG_EXPORT                : エクスポートメニューを作ります
    0, 0,                        //    設定ウインドウのサイズ (FILTER_FLAG_WINDOW_SIZEが立っている時に有効)
    "YUV変換出力確認用フィルタ",    //    フィルタの名前
    TRACK_N,                    //    トラックバーの数 (0なら名前初期値等もNULLでよい)
    track_name,                    //    トラックバーの名前郡へのポインタ
    track_default,                //    トラックバーの初期値郡へのポインタ
    track_s, track_e,            //    トラックバーの数値の下限上限 (NULLなら全て0～256)
    CHECK_N,                    //    チェックボックスの数 (0なら名前初期値等もNULLでよい)
    check_name,                    //    チェックボックスの名前郡へのポインタ
    check_default,                //    チェックボックスの初期値郡へのポインタ
    func_proc,                    //    フィルタ処理関数へのポインタ (NULLなら呼ばれません)
    func_init,                    //    開始時に呼ばれる関数へのポインタ (NULLなら呼ばれません)
    func_exit,                    //    終了時に呼ばれる関数へのポインタ (NULLなら呼ばれません)
    NULL,                        //    設定が変更されたときに呼ばれる関数へのポインタ (NULLなら呼ばれません)
    NULL,                        //    設定ウィンドウにウィンドウメッセージが来た時に呼ばれる関数へのポインタ (NULLなら呼ばれません)
    NULL, NULL,                    //    システムで使いますので使用しないでください
    NULL,                        //  拡張データ領域へのポインタ (FILTER_FLAG_EX_DATAが立っている時に有効)
    NULL,                        //  拡張データサイズ (FILTER_FLAG_EX_DATAが立っている時に有効)
    "0.0.1",
    //  フィルタ情報へのポインタ (FILTER_FLAG_EX_INFORMATIONが立っている時に有効)
    NULL,                        //    セーブが開始される直前に呼ばれる関数へのポインタ (NULLなら呼ばれません)
    NULL,                        //    セーブが終了した直前に呼ばれる関数へのポインタ (NULLなら呼ばれません)
};

static YCAtoYUVParam readSetting(FILTER* fp, FILTER_PROC_INFO *fpip) {
    YCAtoYUVParam s;
    s.pitch = fpip->max_w;
    s.width = fpip->w;
    s.height = fpip->h;
    s.interlaced = (fp->check[0] != 0);
    s.src_depth = 8;
    s.dither = (fp->check[1] != 0);
    s.seed = fp->track[0];
    s.rand_each_frame = (fp->check[2] != 0);
    s.frame_number = fpip->frame;
    return s;
}

static YCAConverter* cuda_filter = NULL;

//---------------------------------------------------------------------
//        フィルタ構造体のポインタを渡す関数
//---------------------------------------------------------------------
EXTERN_C FILTER_DLL __declspec(dllexport) * __stdcall GetFilterTable(void)
{
    return &filter;
}

BOOL func_init(FILTER *fp) {
    init_console();
    return TRUE;
}

BOOL func_exit(FILTER *fp) {
    if (cuda_filter != NULL) {
        delete cuda_filter;
        cuda_filter = NULL;
    }
    return TRUE;
}

//---------------------------------------------------------------------
//        フィルタ処理関数
//---------------------------------------------------------------------

static int clamp(int v, int minimum, int maximum) {
    return (v < minimum)
        ? minimum
        : (v > maximum)
        ? maximum
        : v;
}

int yuv_to_yc48_luma(int Y) {
    return ((Y * 1197) >> 6) - 299;
}
int yuv_to_yc48_chroma(int UV) {
    return ((UV - 128) * 4681 + 164) >> 8;
}

int clamp_luma(int v) {
    return clamp(v, 16, 235);
}
int clamp_chroma(int v) {
    return clamp(v, 16, 240);
}

template <int src_depth, int dst_depth> struct yc48_to_yuv {
    enum {
        SHIFT_LUMA = 12 + src_depth - dst_depth,
        SHIFT_CHROMA = 7 + src_depth - dst_depth
    };
    static int luma(int Y, int dith) {
        return clamp_luma((Y * 219 + 2109 + (16 << 12) + (dith << (SHIFT_LUMA - 8))) >> SHIFT_LUMA);
    }
    static int chroma(int UV, int dith) {
        return clamp_chroma(((UV + 2048) * 7 + 66 + (16 << 7) + ((dith << 3) >> (11 - SHIFT_CHROMA))) >> SHIFT_CHROMA);
    }
};

void check(PIXEL_YC* src, PIXEL_YC* dst, int dither, int interlaced, int pitch, int width, int height)
{
    const int thresh = dither ? (18 * 2) : 18;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            auto s = src[y * pitch + x];
            auto d = dst[y * pitch + x];
            if (std::abs(s.y - d.y) > thresh) {
                if (s.y >= -299 && s.y <= 4470) {
                    printf("[yuvemu] フィルタエラー\n");
                }
            }
            // クロマは420で必要なところだけ
            if ((x & 1) == 0 && (interlaced ? (((y + 1) & 2) == 0) : ((y & 1) == 0))) {
                if (std::abs(s.cb - d.cb) > thresh || std::abs(s.cr - d.cr) > thresh) {
                    if (s.cb >= -2340 && s.cb <= 2322 && s.cr >= -2340 && s.cr <= 2322) {
                        printf("[yuvemu] フィルタエラー\n");
                    }
                }
            }
        }
    }
}

BOOL func_proc(FILTER *fp, FILTER_PROC_INFO *fpip)
{
    auto param = readSetting(fp, fpip);
    if (cuda_filter == NULL) {
        cuda_filter = new YCAConverter();
    }

    PIXEL_YC ps = fpip->ycp_edit[1928 * 183 + 937];

    int y = yc48_to_yuv<8, 8>::luma(ps.y, 0);
    int u = yc48_to_yuv<8, 8>::chroma(ps.cb, 0);
    int v = yc48_to_yuv<8, 8>::chroma(ps.cr, 0);

    PIXEL_YC pd;

    pd.y = yuv_to_yc48_luma(y);
    pd.cb = yuv_to_yc48_chroma(u);
    pd.cr = yuv_to_yc48_chroma(v);

    fpip->ycp_temp[1928 * 183 + 937] = pd;

    int y2 = yc48_to_yuv<8, 8>::luma(pd.y, 0);
    int u2 = yc48_to_yuv<8, 8>::chroma(pd.cb, 0);
    int v2 = yc48_to_yuv<8, 8>::chroma(pd.cr, 0);

    if (y != y2 || u != u2 || v != v2) {
        printf("!!\n");
    }

    cuda_filter->toYUV(&param, (cudafilter::PIXEL_YC*)fpip->ycp_edit, (cudafilter::PIXEL_YC*)fpip->ycp_temp);

    check(fpip->ycp_edit, fpip->ycp_temp, param.dither, param.interlaced, fpip->max_w, fpip->w, fpip->h);

    std::swap(fpip->ycp_edit, fpip->ycp_temp);
    return TRUE;
}
