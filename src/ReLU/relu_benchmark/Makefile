CC = gcc
CFLAGS = -O3 -mavx2 -fopenmp -std=c11 -DCL_TARGET_OPENCL_VERSION=300
LDFLAGS = -lOpenCL

all: relu_benchmark

relu_benchmark: relu_benchmark.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f relu_benchmark