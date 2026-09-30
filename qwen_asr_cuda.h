#ifndef QWEN_ASR_CUDA_H
#define QWEN_ASR_CUDA_H
#ifdef __cplusplus
extern "C" {
#endif
#include "qwen_asr.h"
int qwen_cuda_init(qwen_ctx_t *ctx);
void qwen_cuda_free(qwen_ctx_t *ctx);
int qwen_cuda_selftest(void);
#ifdef __cplusplus
}
#endif
#endif
