// YUV�ϊ��o�͊m�F�p�t�B���^

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
//        �t�B���^�\���̒�`
//---------------------------------------------------------------------
#define    TRACK_N    1                                                                                    //  �g���b�N�o�[�̐�
TCHAR    *track_name[] = { "seed" };    //  �g���b�N�o�[�̖��O
int        track_default[] = { 0 };    //  �g���b�N�o�[�̏����l
int        track_s[] = { 0 };    //  �g���b�N�o�[�̉����l
int        track_e[] = { 63 };    //  �g���b�N�o�[�̏���l

#define    CHECK_N    4                                                                   //  �`�F�b�N�{�b�N�X�̐�
TCHAR    *check_name[] = { "�t�B�[���h����", "�f�B�U", "���t���[�������𐶐�", "CUDA" }; //  �`�F�b�N�{�b�N�X�̖��O
int         check_default[] = { 1, 1, 0, 1 };    //  �`�F�b�N�{�b�N�X�̏����l (�l��0��1)

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
    "YUV�ϊ��o�͊m�F�p�t�B���^",    //    �t�B���^�̖��O
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
    if (cuda_filter != NULL) {
        delete cuda_filter;
        cuda_filter = NULL;
    }
    return TRUE;
}

//---------------------------------------------------------------------
//        �t�B���^�����֐�
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
                    printf("[yuvemu] �t�B���^�G���[\n");
                }
            }
            // �N���}��420�ŕK�v�ȂƂ��낾��
            if ((x & 1) == 0 && (interlaced ? (((y + 1) & 2) == 0) : ((y & 1) == 0))) {
                if (std::abs(s.cb - d.cb) > thresh || std::abs(s.cr - d.cr) > thresh) {
                    if (s.cb >= -2340 && s.cb <= 2322 && s.cr >= -2340 && s.cr <= 2322) {
                        printf("[yuvemu] �t�B���^�G���[\n");
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
