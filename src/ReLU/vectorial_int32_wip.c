#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <x86intrin.h>
#include <omp.h>
#include <math.h>


void loop_relu_pipeline(int32_t* data, size_t n) {
    if (n < 8) return;  // nada a fazer

    size_t iterations = n / 8;

    __asm__ __volatile__ (
    ".intel_syntax noprefix;\n"

    "mov r10, %[ptr];\n"
    "mov r11, %[iters];\n"
    "add r10, 32;\n"

    // warm-up: carregar primeiro bloco → ymm0
    "vmovaps ymm0, [r10-32];\n"
    "dec r11;\n"

    "pipeline_loop:\n"
    "vmovaps ymm1, [r10];\n"       // pré-carrega próximo

    // processar ymm0
    //"vpxor ymm3, ymm3, ymm3;\n"
    "vpcmpgtd ymm2, ymm0, ymm3;\n"
    "vpand ymm2, ymm0, ymm2;\n"
    "vmovaps [r10 - 32], ymm2;\n"

    // preparar próxima rodada
    "vmovaps ymm0, ymm1;\n"

    "add r10, 32;\n"
    "dec r11;\n"
    "prefetcht0 [r10];\n"
    "prefetcht0 [r10-32];\n"
    "jnz pipeline_loop;\n"

    // flush final
    //"vpxor ymm3, ymm3, ymm3;\n"
    "vpcmpgtd ymm2, ymm0, ymm3;\n"
    "vpand ymm2, ymm0, ymm2;\n"
    "vmovaps [r10 - 32], ymm2;\n"

    ".att_syntax prefix;\n"
    :
    : [ptr] "r"(data), [iters] "r"(iterations)
    : "r10", "r11",
      "ymm0", "ymm1", "ymm2", "ymm3", "memory"
);
}


void relu_avx2(int32_t* restrict data, size_t n, size_t block_size) {

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; i += block_size) {
        size_t end = (i + block_size < n) ? (i + block_size) : n;
        size_t chunk = end - i;
        // apenas múltiplos de 8 elementos
        size_t padded = chunk - (chunk % 8);
        if (padded > 0) {
		//_mm_prefetch((char*)&data[i+8], _MM_HINT_T0);
		loop_relu_pipeline(&data[i], padded);
    	}
    }
}


long long checksum(const int32_t* data, size_t n) {
    long long sum = 0;
    for (size_t i = 0; i < n; ++i)
        sum += data[i];
    return sum;
}

#define MEASURE_SINGLE(fn_call, label, data_ptr, data_size) do { \
    repopula_dados(data_ptr, data_size); \
    unsigned long long t0 = __rdtsc(); \
    fn_call; \
    unsigned long long t1 = __rdtsc(); \
    unsigned long long cycles = (t1 - t0); \
    printf("%s: %llu cycles\n", label, cycles); \
} while(0)

#define MEASURE(fn_call, label, num_iter, data_ptr, data_size) do { \
    unsigned long long* cycles_arr = (unsigned long long*)malloc(num_iter * sizeof(unsigned long long)); \
    unsigned long long total_cycles = 0; \
    for (int i = 0; i < num_iter; i++) { \
        repopula_dados(data_ptr, data_size); \
        unsigned long long t0 = __rdtsc(); \
        fn_call; \
        unsigned long long t1 = __rdtsc(); \
        cycles_arr[i] = (t1 - t0); \
        total_cycles += cycles_arr[i]; \
    } \
    unsigned long long mean = total_cycles / num_iter; \
    unsigned long long squared_diff_sum = 0; \
    for (int i = 0; i < num_iter; i++) { \
        unsigned long long diff = cycles_arr[i] - mean; \
        squared_diff_sum += diff * diff; \
    } \
    double variance = squared_diff_sum / (double) num_iter; \
    double stddev = sqrt(variance); \
    printf("%s: %llu cycles (mean)\n", label, mean); \
    printf("%s: %f cycles (stddev)\n", label, stddev); \
    free(cycles_arr); \
} while(0)

void repopula_dados(int32_t* data, size_t n) {
    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i)
        data[i] = (i*i % 2000) - 1000;
}

#define SIZE 1024*16*32


int main() {
    int32_t* b = aligned_alloc(64, SIZE * sizeof(int32_t));
    int32_t num_iter = 8192;
    int32_t block = 256;
    omp_set_num_threads(6);
    MEASURE(relu_avx2(b, SIZE, block), "AVX2 (int32), 524.288 elements", num_iter, b, SIZE);

    printf("Checksum: %lld\n", checksum(b, SIZE));
    printf("Correctsum: %s\n", (checksum(b, SIZE) == 111090131) ? "TRUE" : "FALSE");
    free(b);
    return 0;
}
