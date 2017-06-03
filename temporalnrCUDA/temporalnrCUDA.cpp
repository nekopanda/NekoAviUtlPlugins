// ���Ԏ��m�C�Y�ጸ�t�B���^

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
//        �t�B���^�\���̒�`
//---------------------------------------------------------------------
#define    TRACK_N    6                                                                                    //  �g���b�N�o�[�̐�
TCHAR    *track_name[] = { "�t���[������", "Y", "Cb", "Cr", "�o�b�`�T�C�Y", "臒l(test)" };    //  �g���b�N�o�[�̖��O
int        track_default[] = { 15, 8, 8, 8, 8, 8 };    //  �g���b�N�o�[�̏����l
int        track_s[] = { 1, 0, 0, 0, 1, 0 };    //  �g���b�N�o�[�̉����l
int        track_e[] = { TEMPNR_MAX_DIST, 31, 31, 31, TEMPNR_MAX_BATCH, 31 };    //  �g���b�N�o�[�̏���l

#define    CHECK_N    3                                                                   //  �`�F�b�N�{�b�N�X�̐�
TCHAR    *check_name[] = { "�t�B�[���h����", "����\��", "CUDA" }; //  �`�F�b�N�{�b�N�X�̖��O
int         check_default[] = { 0, 0, 1 };    //  �`�F�b�N�{�b�N�X�̏����l (�l��0��1)

FILTER_DLL filter = {
	FILTER_FLAG_EX_INFORMATION,    //    �t�B���^�̃t���O
	//    FILTER_FLAG_ALWAYS_ACTIVE        : �t�B���^����ɃA�N�e�B�u�ɂ��܂�
	//    FILTER_FLAG_CONFIG_POPUP        : �ݒ���|�b�v�A�b�v���j���[�ɂ��܂�
	//    FILTER_FLAG_CONFIG_CHECK        : �ݒ���`�F�b�N�{�b�N�X���j���[�ɂ��܂�
	//    FILTER_FLAG_CONFIG_RADIO        : �ݒ�����W�I�{�^�����j���[�ɂ��܂�
	//    FILTER_FLAG_EX_DATA                : �g���f�[�^��ۑ��o����悤�ɂ��܂��B
	//    FILTER_FLAG_PRIORITY_HIGHEST    : �t�B���^�̃v���C�I���e�B����ɍŏ�ʂɂ��܂�
	//    FILTER_FLAG_PRIORITY_LOWEST        : �t�B���^�̃v���C�I���e�B����ɍŉ��ʂɂ��܂�
	//    FILTER_FLAG_WINDOW_THICKFRAME    : �T�C�Y�ύX�\�ȃE�B���h�E�����܂�
	//    FILTER_FLAG_WINDOW_SIZE            : �ݒ�E�B���h�E�̃T�C�Y���w��o����悤�ɂ��܂�
	//    FILTER_FLAG_DISP_FILTER            : �\���t�B���^�ɂ��܂�
	//    FILTER_FLAG_EX_INFORMATION        : �t�B���^�̊g������ݒ�ł���悤�ɂ��܂�
	//    FILTER_FLAG_NO_CONFIG            : �ݒ�E�B���h�E��\�����Ȃ��悤�ɂ��܂�
	//    FILTER_FLAG_AUDIO_FILTER        : �I�[�f�B�I�t�B���^�ɂ��܂�
	//    FILTER_FLAG_RADIO_BUTTON        : �`�F�b�N�{�b�N�X�����W�I�{�^���ɂ��܂�
	//    FILTER_FLAG_WINDOW_HSCROLL        : �����X�N���[���o�[�����E�B���h�E�����܂�
	//    FILTER_FLAG_WINDOW_VSCROLL        : �����X�N���[���o�[�����E�B���h�E�����܂�
	//    FILTER_FLAG_IMPORT                : �C���|�[�g���j���[�����܂�
	//    FILTER_FLAG_EXPORT                : �G�N�X�|�[�g���j���[�����܂�
	0, 0,                        //    �ݒ�E�C���h�E�̃T�C�Y (FILTER_FLAG_WINDOW_SIZE�������Ă��鎞�ɗL��)
	"���Ԏ��m�C�Y�ጸ",    //    �t�B���^�̖��O
	TRACK_N,                    //    �g���b�N�o�[�̐� (0�Ȃ疼�O�����l����NULL�ł悢)
	track_name,                    //    �g���b�N�o�[�̖��O�S�ւ̃|�C���^
	track_default,                //    �g���b�N�o�[�̏����l�S�ւ̃|�C���^
	track_s, track_e,            //    �g���b�N�o�[�̐��l�̉������ (NULL�Ȃ�S��0�`256)
	CHECK_N,                    //    �`�F�b�N�{�b�N�X�̐� (0�Ȃ疼�O�����l����NULL�ł悢)
	check_name,                    //    �`�F�b�N�{�b�N�X�̖��O�S�ւ̃|�C���^
	check_default,                //    �`�F�b�N�{�b�N�X�̏����l�S�ւ̃|�C���^
	func_proc,                    //    �t�B���^�����֐��ւ̃|�C���^ (NULL�Ȃ�Ă΂�܂���)
	func_init,                    //    �J�n���ɌĂ΂��֐��ւ̃|�C���^ (NULL�Ȃ�Ă΂�܂���)
	func_exit,                    //    �I�����ɌĂ΂��֐��ւ̃|�C���^ (NULL�Ȃ�Ă΂�܂���)
	NULL,                        //    �ݒ肪�ύX���ꂽ�Ƃ��ɌĂ΂��֐��ւ̃|�C���^ (NULL�Ȃ�Ă΂�܂���)
	NULL,                        //    �ݒ�E�B���h�E�ɃE�B���h�E���b�Z�[�W���������ɌĂ΂��֐��ւ̃|�C���^ (NULL�Ȃ�Ă΂�܂���)
	NULL, NULL,                    //    �V�X�e���Ŏg���܂��̂Ŏg�p���Ȃ��ł�������
	NULL,                        //  �g���f�[�^�̈�ւ̃|�C���^ (FILTER_FLAG_EX_DATA�������Ă��鎞�ɗL��)
	NULL,                        //  �g���f�[�^�T�C�Y (FILTER_FLAG_EX_DATA�������Ă��鎞�ɗL��)
	"0.0.1",
	//  �t�B���^���ւ̃|�C���^ (FILTER_FLAG_EX_INFORMATION�������Ă��鎞�ɗL��)
	NULL,                        //    �Z�[�u���J�n����钼�O�ɌĂ΂��֐��ւ̃|�C���^ (NULL�Ȃ�Ă΂�܂���)
	NULL,                        //    �Z�[�u���I���������O�ɌĂ΂��֐��ւ̃|�C���^ (NULL�Ȃ�Ă΂�܂���)
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
//        �t�B���^�\���̂̃|�C���^��n���֐�
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
//        �t�B���^�����֐�
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
		// �L���b�V���ɂȂ��Ȃ���
		
		// �L���b�V���K�v������ݒ�
		int n_src_cache = param.batchSize + param.temporalDistance * 2 + 8;
		fp->exfunc->set_ycp_filtering_cache_size(fp, fpip->max_w, fpip->h, n_src_cache, NULL);

		// �\�[�X�t���[�����擾
		std::vector<cudafilter::PIXEL_YC*> src_frames;
		int base_frame_idx = fpip->frame / param.batchSize * param.batchSize;
		for (int i = -param.temporalDistance; i < param.batchSize + param.temporalDistance; ++i) {
			int fidx = std::max(0, std::min(fpip->frame_n, base_frame_idx + i));
			int w, h;
			PIXEL_YC* frame_ptr = fp->exfunc->get_ycp_filtering_cache_ex(fp, fpip->editp, fidx, &w, &h);
			src_frames.push_back((cudafilter::PIXEL_YC*)frame_ptr);
		}

		// ���s
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
