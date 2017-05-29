// ���Ԏ��m�C�Y�ጸ�t�B���^

#include <windows.h>

#define DEFINE_GLOBAL
#include "filter.h"
#include "plugin_utils.h"

//---------------------------------------------------------------------
//        �t�B���^�\���̒�`
//---------------------------------------------------------------------
#define    TRACK_N    6                                                                                    //  �g���b�N�o�[�̐�
TCHAR    *track_name[] = { "�t���[������", "Y", "Cb", "Cr", "�o�b�`�T�C�Y", "臒l(test)" };    //  �g���b�N�o�[�̖��O
int        track_default[] = { 15, 8, 8, 8, 8, 8 };    //  �g���b�N�o�[�̏����l
int        track_s[] = { 1, 0, 0, 0, 1, 0 };    //  �g���b�N�o�[�̉����l
int        track_e[] = { 63, 31, 31, 31, 32, 31 };    //  �g���b�N�o�[�̏���l

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


//---------------------------------------------------------------------
//        �t�B���^�\���̂̃|�C���^��n���֐�
//---------------------------------------------------------------------
EXTERN_C FILTER_DLL __declspec(dllexport) * __stdcall GetFilterTable(void)
{
	return &filter;
}

BOOL func_init(FILTER *fp) {
	init_console();
	// TODO:
	return TRUE;
}

BOOL func_exit(FILTER *fp) {
	// TODO:
	return TRUE;
}

//---------------------------------------------------------------------
//        �t�B���^�����֐�
//---------------------------------------------------------------------
BOOL func_proc(FILTER *fp, FILTER_PROC_INFO *fpip)
{
	fpip->frame
	// TODO:
	return TRUE;
}
