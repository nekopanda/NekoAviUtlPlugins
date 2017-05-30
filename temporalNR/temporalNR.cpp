// 時間軸ノイズ低減フィルタ

#include <windows.h>

#undef max
#undef min

#include <vector>
#include <algorithm>

#define DEFINE_GLOBAL
#include "filter.h"
#include "plugin_utils.h"
#include "cuda_filters.h"

using cudafilter::TemporalNRParam;
using cudafilter::TemporalNRFilter;

enum {
	TEMPNR_MAX_DIST = cudafilter::TEMPNR_MAX_DIST,
	TEMPNR_MAX_BATCH = cudafilter::TEMPNR_MAX_BATCH
};

//---------------------------------------------------------------------
//        フィルタ構造体定義
//---------------------------------------------------------------------
#define    TRACK_N    6                                                                                    //  トラックバーの数
TCHAR    *track_name[] = { "フレーム距離", "Y", "Cb", "Cr", "バッチサイズ", "閾値(test)" };    //  トラックバーの名前
int        track_default[] = { 15, 8, 8, 8, 8, 8 };    //  トラックバーの初期値
int        track_s[] = { 1, 0, 0, 0, 1, 0 };    //  トラックバーの下限値
int        track_e[] = { TEMPNR_MAX_DIST, 31, 31, 31, TEMPNR_MAX_BATCH, 31 };    //  トラックバーの上限値

#define    CHECK_N    3                                                                   //  チェックボックスの数
TCHAR    *check_name[] = { "フィールド処理", "判定表示", "CUDA" }; //  チェックボックスの名前
int         check_default[] = { 0, 0, 1 };    //  チェックボックスの初期値 (値は0か1)

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
	"時間軸ノイズ低減",    //    フィルタの名前
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

static TemporalNRParam readSetting(FILTER* fp, FILTER_PROC_INFO *fpip) {
	TemporalNRParam s;
	s.pitch = fpip->max_w;
	s.width = fpip->w;
	s.height = fpip->h;
	s.temporalDistance = fp->track[0];
	s.threshY = fp->track[1];
	s.threshCb = fp->track[2];
	s.threshCr = fp->track[3];
	s.batchSize = fp->track[4];
	s.thresh = fp->track[5];
	s.interlaced = (fp->check[0] != 0);
	s.check = (fp->check[1] != 0);
	return s;
}

class TemporalNRFrameCache
{
public:
	TemporalNRFrameCache(const TemporalNRParam& param)
		: param(param)
	{
		for (int i = 0; i < param.batchSize; ++i) {
			frames.push_back((cudafilter::PIXEL_YC*)malloc(param.pitch * param.height * sizeof(PIXEL_YC)));
		}
	}

	~TemporalNRFrameCache()
	{
		for (int i = 0; i < (int)frames.size(); ++i) {
			free(frames[i]);
		}
		frames.clear();
	}

	const std::vector<cudafilter::PIXEL_YC*> getFrames() const {
		return frames;
	}

	bool isSame(const TemporalNRParam& o) const {
		return o.batchSize == param.batchSize &&
			o.pitch == param.pitch &&
			o.height == param.height;
	}

private:
	const TemporalNRParam param;

	std::vector<cudafilter::PIXEL_YC*> frames;
};

static int cache_frame_block_idx_;
static TemporalNRFrameCache* cache_ = NULL;
static TemporalNRFilter* filter_ = NULL;

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
	if (cache_ != NULL) {
		delete cache_;
		cache_ = NULL;
	}
	if (filter_ != NULL) {
		delete filter_;
		filter_ = NULL;
	}
	return TRUE;
}

//---------------------------------------------------------------------
//        フィルタ処理関数
//---------------------------------------------------------------------

BOOL func_proc(FILTER *fp, FILTER_PROC_INFO *fpip)
{
	auto param = readSetting(fp, fpip);
	if (cache_ == NULL) {
		cache_ = new TemporalNRFrameCache(param);
		cache_frame_block_idx_ = -1;
	}
	else if (cache_->isSame(param) == false) {
		delete cache_;
		cache_ = new TemporalNRFrameCache(param);
		cache_frame_block_idx_ = -1;
	}

	int frame_block_idx = fpip->frame / param.batchSize;
	if (cache_frame_block_idx_ != frame_block_idx) {
		// キャッシュにないなら作る
		
		// キャッシュ必要枚数を設定
		int n_src_cache = param.batchSize + param.temporalDistance * 2 + 8;
		fp->exfunc->set_ycp_filtering_cache_size(fp, fpip->max_w, fpip->h, n_src_cache, NULL);

		// ソースフレームを取得
		std::vector<cudafilter::PIXEL_YC*> src_frames;
		int base_frame_idx = fpip->frame / param.batchSize * param.batchSize;
		for (int i = -param.temporalDistance; i < param.batchSize + param.temporalDistance; ++i) {
			int fidx = std::max(0, std::min(fpip->frame_n, base_frame_idx + i));
			int w, h;
			PIXEL_YC* frame_ptr = fp->exfunc->get_ycp_filtering_cache_ex(fp, fpip->editp, fidx, &w, &h);
			src_frames.push_back((cudafilter::PIXEL_YC*)frame_ptr);
		}

		// 実行
		auto& dst_frames = cache_->getFrames();

		if (filter_ == NULL) {
			filter_ = new TemporalNRFilter();
		}
		filter_->proc(&param, src_frames.data(), dst_frames.data());

		cache_frame_block_idx_ = frame_block_idx;
	}

	int frame_idx = fpip->frame % param.batchSize;
	cudafilter::PIXEL_YC* ptr = cache_->getFrames()[frame_idx];
	memcpy(fpip->ycp_temp, ptr, fpip->max_w * fpip->h);
	std::swap(fpip->ycp_edit, fpip->ycp_temp);
	
	return TRUE;
}
