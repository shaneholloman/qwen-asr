# qwen_asr — Qwen3-ASR Pure C Inference Engine
# Makefile

CC = gcc
CFLAGS_BASE = -Wall -Wextra -O3 -march=native -ffast-math
LDFLAGS = -lm -lpthread

# Platform detection
UNAME_S := $(shell uname -s)

# Source files
SRCS = qwen_asr.c qwen_asr_kernels.c qwen_asr_kernels_generic.c qwen_asr_kernels_neon.c qwen_asr_kernels_avx.c qwen_asr_audio.c qwen_asr_encoder.c qwen_asr_decoder.c qwen_asr_tokenizer.c qwen_asr_safetensors.c
OBJS = $(SRCS:.c=.o)
MAIN = main.c
TARGET = qwen_asr

# Optional Linux AMD GPU backend. Separate objects/binary let CPU and ROCm
# builds coexist. Changing GPU_ARCH selects a fresh set of GPU objects.
ROCM_PATH ?= /opt/rocm
HIPCC ?= $(ROCM_PATH)/bin/hipcc
GPU_ARCH ?= gfx1151
ROCM_CFLAGS ?= -Wall -Wextra -O3 -march=native
HIPFLAGS ?= -O3 -std=c++17
ROCM_LDFLAGS = -L$(ROCM_PATH)/lib -Wl,-rpath,$(ROCM_PATH)/lib -lrocblas -lm -lpthread
ROCM_BUILD = build/rocm/$(GPU_ARCH)
ROCM_OBJS = $(SRCS:%.c=$(ROCM_BUILD)/%.o)
ROCM_TARGET = qwen_asr_rocm
ROCM_TEST = $(ROCM_BUILD)/kernel-test
ROCM_MODEL_DIR ?= qwen3-asr-0.6b

# Native CUDA backend. sm_121 is the GB10 GPU in DGX Spark.
CUDA_PATH ?= /usr/local/cuda
NVCC ?= $(CUDA_PATH)/bin/nvcc
CUDA_ARCH ?= sm_121
CUDA_CFLAGS ?= -Wall -Wextra -O3 -march=native
NVCCFLAGS ?= -O3 -std=c++17 -lineinfo
CUDA_LDFLAGS = -L$(CUDA_PATH)/lib64 -Xlinker=-rpath,$(CUDA_PATH)/lib64 -lcublas -lm -lpthread
CUDA_BUILD = build/cuda/$(CUDA_ARCH)
CUDA_OBJS = $(SRCS:%.c=$(CUDA_BUILD)/%.o)
CUDA_TARGET = qwen_asr_cuda
CUDA_TEST = $(CUDA_BUILD)/kernel-test
CUDA_MODEL_DIR ?= qwen3-asr-0.6b

# Debug build flags
DEBUG_CFLAGS = -Wall -Wextra -g -O0 -DDEBUG -fsanitize=address

.PHONY: all clean clean-cpu clean-rocm debug info help blas rocm test test-stream-cache test-rocm test-rocm-asr FORCE
.PHONY: cuda clean-cuda test-cuda test-cuda-asr

# Default: show available targets
all: help

help:
	@echo "qwen_asr — Qwen3-ASR Pure C Inference - Build Targets"
	@echo ""
	@echo "Choose a backend:"
	@echo "  make blas     - With BLAS acceleration (Accelerate/OpenBLAS)"
	@echo "  make rocm     - AMD GPU acceleration (HIP/rocBLAS; builds qwen_asr_rocm)"
	@echo "  make cuda     - NVIDIA GPU acceleration (CUDA/cuBLAS; builds qwen_asr_cuda)"
	@echo ""
	@echo "Other targets:"
	@echo "  make debug    - Debug build with AddressSanitizer"
	@echo "  make test     - Run regression suite (requires ./qwen_asr and model files)"
	@echo "  make test-stream-cache - Run stream cache on/off equivalence check"
	@echo "  make test-rocm - Run ROCm kernel tests (no model needed)"
	@echo "  make test-rocm-asr - Run ROCm transcription checks (needs 0.6B model)"
	@echo "  make test-cuda - Run CUDA kernel tests (no model needed)"
	@echo "  make test-cuda-asr - Run CUDA transcription checks (needs 0.6B model)"
	@echo "  make clean    - Remove build artifacts"
	@echo "  make info     - Show build configuration"
	@echo ""
	@echo "Example: make blas && ./qwen_asr -d model_dir -i audio.wav"

# =============================================================================
# Backend: blas (Accelerate on macOS, OpenBLAS on Linux)
# =============================================================================
ifeq ($(UNAME_S),Darwin)
blas: CFLAGS = $(CFLAGS_BASE) -DUSE_BLAS -DACCELERATE_NEW_LAPACK
blas: LDFLAGS += -framework Accelerate
else
blas: CFLAGS = $(CFLAGS_BASE) -DUSE_BLAS -DUSE_OPENBLAS -I/usr/include/openblas
blas: LDFLAGS += -lopenblas
endif
blas:
	@$(MAKE) clean-cpu
	@$(MAKE) $(TARGET) CFLAGS="$(CFLAGS)" LDFLAGS="$(LDFLAGS)"
	@echo ""
	@echo "Built with BLAS backend"

# =============================================================================
# Backend: rocm (HIP kernels + rocBLAS, tested on gfx1151 / Strix Halo)
# =============================================================================
rocm: $(ROCM_TARGET)

# Always select the requested architecture, even when switching back to an
# older cached build whose objects predate the top-level executable.
$(ROCM_TARGET): $(ROCM_BUILD)/qwen_asr_rocm FORCE
	cp $< $@

FORCE:

$(ROCM_BUILD)/qwen_asr_rocm: $(ROCM_OBJS) $(ROCM_BUILD)/main.o $(ROCM_BUILD)/qwen_asr_rocm.o
	$(HIPCC) $(HIPFLAGS) --offload-arch=$(GPU_ARCH) -o $@ $^ $(ROCM_LDFLAGS)

$(ROCM_BUILD)/%.o: %.c
	@mkdir -p $(@D)
	$(CC) $(ROCM_CFLAGS) -DUSE_ROCM -MMD -MP -c $< -o $@

$(ROCM_BUILD)/qwen_asr_rocm.o: qwen_asr_rocm.hip qwen_asr_rocm.h qwen_asr.h rocm/kernels.h
	@mkdir -p $(@D)
	$(HIPCC) $(HIPFLAGS) --offload-arch=$(GPU_ARCH) -I$(ROCM_PATH)/include -DUSE_ROCM -c -x hip $< -o $@

$(ROCM_BUILD)/qwen_asr_rocm_test.o: qwen_asr_rocm.hip qwen_asr_rocm.h qwen_asr.h rocm/kernels.h tests/rocm/selftest.h
	@mkdir -p $(@D)
	$(HIPCC) $(HIPFLAGS) --offload-arch=$(GPU_ARCH) -I$(ROCM_PATH)/include -DUSE_ROCM -DQWEN_ROCM_TEST -c -x hip $< -o $@

$(ROCM_BUILD)/kernel_test.o: tests/rocm/kernel_test.c qwen_asr_rocm.h qwen_asr.h
	@mkdir -p $(@D)
	$(CC) $(ROCM_CFLAGS) -DUSE_ROCM -I. -c $< -o $@

$(ROCM_TEST): $(ROCM_OBJS) $(ROCM_BUILD)/kernel_test.o $(ROCM_BUILD)/qwen_asr_rocm_test.o
	$(HIPCC) $(HIPFLAGS) --offload-arch=$(GPU_ARCH) -o $@ $^ $(ROCM_LDFLAGS)

test-rocm: $(ROCM_TEST)
	./$(ROCM_TEST)

test-rocm-asr: $(ROCM_TARGET)
	python3 tests/rocm/smoke.py --binary ./$(ROCM_TARGET) --model "$(ROCM_MODEL_DIR)"

-include $(ROCM_OBJS:.o=.d) $(ROCM_BUILD)/main.d

# =============================================================================
# Backend: cuda (native CUDA kernels + cuBLAS, tested on DGX Spark / sm_121)
# =============================================================================
cuda: $(CUDA_TARGET)

$(CUDA_TARGET): $(CUDA_BUILD)/qwen_asr_cuda FORCE
	cp $< $@

$(CUDA_BUILD)/qwen_asr_cuda: $(CUDA_OBJS) $(CUDA_BUILD)/main.o $(CUDA_BUILD)/qwen_asr_cuda.o
	$(NVCC) $(NVCCFLAGS) -arch=$(CUDA_ARCH) -o $@ $^ $(CUDA_LDFLAGS)

$(CUDA_BUILD)/%.o: %.c
	@mkdir -p $(@D)
	$(CC) $(CUDA_CFLAGS) -DUSE_CUDA -MMD -MP -c $< -o $@

$(CUDA_BUILD)/qwen_asr_cuda.o: qwen_asr_cuda.cu qwen_asr_cuda.h qwen_asr.h cuda/kernels.cuh
	@mkdir -p $(@D)
	$(NVCC) $(NVCCFLAGS) -arch=$(CUDA_ARCH) -DUSE_CUDA -c $< -o $@

$(CUDA_BUILD)/qwen_asr_cuda_test.o: qwen_asr_cuda.cu qwen_asr_cuda.h qwen_asr.h cuda/kernels.cuh tests/cuda/selftest.h
	@mkdir -p $(@D)
	$(NVCC) $(NVCCFLAGS) -arch=$(CUDA_ARCH) -DUSE_CUDA -DQWEN_CUDA_TEST -c $< -o $@

$(CUDA_BUILD)/kernel_test.o: tests/cuda/kernel_test.c qwen_asr_cuda.h qwen_asr.h
	@mkdir -p $(@D)
	$(CC) $(CUDA_CFLAGS) -DUSE_CUDA -I. -c $< -o $@

$(CUDA_TEST): $(CUDA_OBJS) $(CUDA_BUILD)/kernel_test.o $(CUDA_BUILD)/qwen_asr_cuda_test.o
	$(NVCC) $(NVCCFLAGS) -arch=$(CUDA_ARCH) -o $@ $^ $(CUDA_LDFLAGS)

test-cuda: $(CUDA_TEST)
	./$(CUDA_TEST)

test-cuda-asr: $(CUDA_TARGET)
	python3 tests/cuda/smoke.py --binary ./$(CUDA_TARGET) --model "$(CUDA_MODEL_DIR)"

-include $(CUDA_OBJS:.o=.d) $(CUDA_BUILD)/main.d

# =============================================================================
# Build rules
# =============================================================================
$(TARGET): $(OBJS) main.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c qwen_asr.h qwen_asr_kernels.h
	$(CC) $(CFLAGS) -c -o $@ $<

# Debug build
debug: CFLAGS = $(DEBUG_CFLAGS)
debug: LDFLAGS += -fsanitize=address
debug:
	@$(MAKE) clean-cpu
	@$(MAKE) $(TARGET) CFLAGS="$(CFLAGS)" LDFLAGS="$(LDFLAGS)"

# =============================================================================
# Utilities
# =============================================================================
clean: clean-cpu clean-rocm clean-cuda

clean-cpu:
	rm -f $(OBJS) main.o $(TARGET)

clean-rocm:
	rm -rf build/rocm
	rm -f $(ROCM_TARGET)

clean-cuda:
	rm -rf build/cuda
	rm -f $(CUDA_TARGET)

info:
	@echo "Platform: $(UNAME_S)"
	@echo "Compiler: $(CC)"
	@echo "ROCm compiler: $(HIPCC)"
	@echo "ROCm architecture: $(GPU_ARCH)"
	@echo "CUDA compiler: $(NVCC)"
	@echo "CUDA architecture: $(CUDA_ARCH)"
	@echo ""
ifeq ($(UNAME_S),Darwin)
	@echo "Backend: blas (Apple Accelerate)"
else
	@echo "Backend: blas (OpenBLAS)"
endif

test:
	./asr_regression.py --binary ./qwen_asr --model-dir qwen3-asr-1.7b

test-stream-cache:
	./asr_regression.py --stream-cache-check-only --binary ./qwen_asr --stream-cache-model-dir qwen3-asr-0.6b

# =============================================================================
# Dependencies
# =============================================================================
qwen_asr.o: qwen_asr.c qwen_asr.h qwen_asr_kernels.h qwen_asr_safetensors.h qwen_asr_audio.h qwen_asr_tokenizer.h
qwen_asr_kernels.o: qwen_asr_kernels.c qwen_asr_kernels.h qwen_asr_kernels_impl.h
qwen_asr_kernels_generic.o: qwen_asr_kernels_generic.c qwen_asr_kernels_impl.h
qwen_asr_kernels_neon.o: qwen_asr_kernels_neon.c qwen_asr_kernels_impl.h
qwen_asr_kernels_avx.o: qwen_asr_kernels_avx.c qwen_asr_kernels_impl.h
qwen_asr_audio.o: qwen_asr_audio.c qwen_asr_audio.h
qwen_asr_encoder.o: qwen_asr_encoder.c qwen_asr.h qwen_asr_kernels.h qwen_asr_safetensors.h
qwen_asr_decoder.o: qwen_asr_decoder.c qwen_asr.h qwen_asr_kernels.h qwen_asr_safetensors.h
qwen_asr_tokenizer.o: qwen_asr_tokenizer.c qwen_asr_tokenizer.h
qwen_asr_safetensors.o: qwen_asr_safetensors.c qwen_asr_safetensors.h
main.o: main.c qwen_asr.h qwen_asr_kernels.h
