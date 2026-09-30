#ifndef QWEN_ASR_ROCM_H
#define QWEN_ASR_ROCM_H
#ifdef __cplusplus
extern "C" {
#endif
#include "qwen_asr.h"
int qwen_rocm_init(qwen_ctx_t *ctx);
void qwen_rocm_free(qwen_ctx_t *ctx);
int qwen_rocm_selftest(void);
#ifdef __cplusplus
}
#endif
#endif
