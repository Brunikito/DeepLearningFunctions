#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <omp.h>
#include <x86intrin.h>
#include <CL/cl.h>
#include <time.h>

#define SIZE 1024*1024*16

#define RELU_BLOCK(offset) \
        "vmovaps ymm1, [rdi+" #offset "];\n" \
        "vpmaxsd ymm1, ymm1, ymm0;\n" \
        "vmovaps [rdi+" #offset "], ymm1;\n"

#define PREFETCH_BLOCK(offset) \
        "prefetcht0 [rdi +" #offset "];\n"


void ReLU_int32(__m256i* data) {
    __asm__ __volatile__ (
        ".intel_syntax noprefix;\n"
        PREFETCH_BLOCK(0)
        PREFETCH_BLOCK(64)
        PREFETCH_BLOCK(128)
        PREFETCH_BLOCK(192)
        PREFETCH_BLOCK(256)
        PREFETCH_BLOCK(320)
        PREFETCH_BLOCK(384)
        PREFETCH_BLOCK(448)

        "vpxor ymm0, ymm0, ymm0;\n"

        RELU_BLOCK(0)
        RELU_BLOCK(32)
        RELU_BLOCK(64)
        RELU_BLOCK(96)
        RELU_BLOCK(128)
        RELU_BLOCK(160)
        RELU_BLOCK(192)
        RELU_BLOCK(224)
        RELU_BLOCK(256)
        RELU_BLOCK(288)
        RELU_BLOCK(320)
        RELU_BLOCK(352)
        RELU_BLOCK(384)
        RELU_BLOCK(416)
        RELU_BLOCK(448)
        RELU_BLOCK(480)

        ".att_syntax prefix;\n"
        :
        :
        : "rdi", "ymm0", "ymm1"
    );
}

void do_iter_cpu(int32_t* arr, int32_t size) {
    #pragma omp parallel for schedule(static)
    for (int32_t i = 0; i < size; i += 128) {
        ReLU_int32((__m256i*)&arr[i]);
    }
}

char* load_kernel_source(const char* filename) {
    FILE* fp = fopen(filename, "r");
    if (!fp) {
        perror("fopen");
        exit(1);
    }
    fseek(fp, 0, SEEK_END);
    size_t size = ftell(fp);
    rewind(fp);
    char* source = malloc(size + 1);
    fread(source, 1, size, fp);
    source[size] = '\0';
    fclose(fp);
    return source;
}

void check_cl_error(cl_int err, const char* msg) {
    if (err != CL_SUCCESS) {
        fprintf(stderr, "OpenCL error (%d): %s\n", err, msg);
        exit(1);
    }
}

void relu_opencl(int32_t* data, int n) {
    cl_platform_id platforms[8];
    cl_device_id device = NULL;
    cl_context context = NULL;
    cl_command_queue queue = NULL;
    cl_program program = NULL;
    cl_kernel kernel = NULL;
    cl_mem buffer = NULL;
    cl_int err;

    cl_uint num_platforms = 0;
    clGetPlatformIDs(8, platforms, &num_platforms);

    for (cl_uint i = 0; i < num_platforms; ++i) {
        // Primeiro tenta GPU
        if (clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 1, &device, NULL) == CL_SUCCESS) {
            printf("Usando dispositivo GPU da plataforma %u\n", i);
            break;
        }
        // Depois tenta CPU
        if (clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_CPU, 1, &device, NULL) == CL_SUCCESS) {
            printf("Usando dispositivo CPU da plataforma %u\n", i);
            break;
        }
    }

    if (!device) {
        fprintf(stderr, "Erro: Nenhum dispositivo OpenCL disponível.\n");
        return;
    }

    char device_name[128];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
    printf("Dispositivo selecionado: %s\n", device_name);

    context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    queue = clCreateCommandQueueWithProperties(context, device, 0, NULL);

    char* source = load_kernel_source("relu.cl");
    program = clCreateProgramWithSource(context, 1, (const char**)&source, NULL, NULL);

    if (clBuildProgram(program, 1, &device, NULL, NULL, NULL) != CL_SUCCESS) {
        // Erro ao compilar, printa o log
        size_t log_size = 0;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* log = malloc(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        fprintf(stderr, "Erro ao compilar o kernel:\n%s\n", log);
        free(log);
        goto cleanup;
    }

    kernel = clCreateKernel(program, "relu_kernel", NULL);

    buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int32_t) * n, data, &err);
    check_cl_error(err, "clCreateBuffer");
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer);
    clSetKernelArg(kernel, 1, sizeof(int), &n);

    size_t global_work_size = (size_t)n;
    size_t local_work_size = 128;
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
    clFinish(queue);
    clEnqueueReadBuffer(queue, buffer, CL_TRUE, 0, sizeof(int32_t) * n, data, 0, NULL, NULL);

cleanup:
    if (buffer) clReleaseMemObject(buffer);
    if (kernel) clReleaseKernel(kernel);
    if (program) clReleaseProgram(program);
    if (queue) clReleaseCommandQueue(queue);
    if (context) clReleaseContext(context);
    if (source) free(source);
}

#define MEASURE(fn_call, cycles) do { \
    struct timespec ts_start, ts_end; \
    unsigned long long t0, t1; \
    clock_gettime(CLOCK_MONOTONIC, &ts_start); \
    t0 = __rdtsc(); \
    fn_call; \
    t1 = __rdtsc(); \
    clock_gettime(CLOCK_MONOTONIC, &ts_end); \
    *cycles += t1 - t0; \
    double elapsed_us = (ts_end.tv_sec - ts_start.tv_sec) * 1e6 + (ts_end.tv_nsec - ts_start.tv_nsec) / 1e3; \
    printf("Tempo: %.3f us, Ciclos: %llu\n", elapsed_us, (unsigned long long)(t1 - t0)); \
} while(0)

int main() {
    int32_t* vec_cpu = aligned_alloc(32, SIZE * sizeof(int32_t));
    int32_t* vec_gpu = aligned_alloc(32, SIZE * sizeof(int32_t));

    for (int32_t i = 0; i < SIZE; ++i) {
        vec_cpu[i] = i - 1000000;
        vec_gpu[i] = vec_cpu[i];
    }

    int64_t cycles_cpu = 0, cycles_gpu = 0;

    MEASURE(do_iter_cpu(vec_cpu, SIZE), &cycles_cpu);
    MEASURE(relu_opencl(vec_gpu, SIZE), &cycles_gpu);

    printf("CPU AVX2+OpenMP: %llu cycles\n", cycles_cpu);
    printf("GPU OpenCL:      %llu cycles\n", cycles_gpu);

    for (int i = 0; i < SIZE; i++) {
        if (vec_cpu[i] != vec_gpu[i]) {
            printf("Mismatch at %d: CPU=%d GPU=%d\n", i, vec_cpu[i], vec_gpu[i]);
            break;
        }
    }

    free(vec_cpu);
    free(vec_gpu);
    return 0;
}
