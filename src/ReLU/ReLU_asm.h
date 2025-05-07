/* 
# RELU OPTIMIZATION

## IDEA OF OPTIMIZATION:
### NON CONDITIONAL 2-CPU-CYCLE-OPERATION:
### 1 cicle SAR -> gets an mask of the signal bit (1 if negative, 0 if positive);
### 1 cicle ANDN -> AND with 0 if negative, AND with 1 if positive.

## COMPATIBILITY:
### There is a more compatible version, with 3-CPU-CYCLE-OPERATION:
### 1 cicle SAR -> gets an mask of the signal bit (1 if negative, 0 if positive);
### 1 cicle NOT -> inverts the maske (0 if negative, 1 if positive);
### 1 cicle AND -> AND the input with the mask, to get the output.

## RETURNS
### x if x >= 0, 0 oterwise.
*/

#ifndef RELU_ASM_H
#define RELU_ASM_H

#include <stdint.h>

typedef float float32_t;
typedef double float64_t;

#define ReLU_2C(x) _Generic((x),\
	int8_t: ReLU_2C_int8,\
	int16_t: ReLU_2C_int16,\
	int32_t: ReLU_2C_int32,\
	int64_t: ReLU_2C_int64,\
	float32_t: ReLU_2C_float32,\
	float64_t: ReLU_2C_float64\
	) (x)

#define ReLU_3C(x) _Generic((x),\
	int8_t: ReLU_3C_int8,\
	int16_t: ReLU_3C_int16,\
	int32_t: ReLU_3C_int32,\
	int64_t: ReLU_3C_int64,\
	float32_t: ReLU_3C_float32,\
	float64_t: ReLU_3C_float64) (x)

static inline int32_t ReLU_2C_int32(int32_t x) {
	__asm__ (
		"sar $31, %0\n\t"
		"andn %0, %0, %1\n\t"
		: "=r"(x)
		: "r"(x)
		);
	return x;
}

static inline int8_t ReLU_2C_int8(int8_t x) {
	return (int8_t)ReLU_2C_int32((int32_t)(x));
}

static inline int16_t ReLU_2C_int16(int16_t x) {
	return (int16_t)ReLU_2C_int32((int32_t)(x));
}

static inline int64_t ReLU_2C_int64(int64_t x) {
	__asm__ (
		"sar $63, %0\n\t"
		"andn %0, %0, %1\n\t"
		: "=r"(x)
		: "r"(x)
		);
	return x;
}

static inline float32_t ReLU_2C_float32(float32_t x) {
	__asm__ (
		"sar $31, %0\n\t"
		"andn %0, %0, %1\n\t"
		: "=r"(*(uint32_t*)&x)
		: "r"(*(uint32_t*)&x)
		);
	return *(float32_t*)&x;
}

static inline float64_t ReLU_2C_float64(float64_t x) {
	__asm__ (
		"sar $63, %0\n\t"
		"andn %0, %0, %1\n\t"
		: "=r"(*(uint64_t*)&x)
		: "r"(*(uint64_t*)&x)
		);
	return *(float64_t*)&x;
}

static inline int8_t ReLU_3C_int8(int8_t x) {
	__asm__ (
		"sar $7, %0\n\t"
		"not %0\n\t"
		"and %0, %1\n\t"
		: "=r"(x)
		: "r"(x)
		);
	return x;
}

static inline int16_t ReLU_3C_int16(int16_t x) {
	__asm__ (
		"sar $15, %0\n\t"
		"not %0\n\t"
		"and %0, %1\n\t"
		: "=r"(x)
		: "r"(x)
		);
	return x;
}

static inline int32_t ReLU_3C_int32(int32_t x) {
	__asm__ (
		"sar $31, %0\n\t"
		"not %0\n\t"
		"and %0, %1\n\t"
		: "=r"(x)
		: "r"(x)
		);
	return x;
}

static inline int64_t ReLU_3C_int64(int64_t x) {
	__asm__ (
		"sar $63, %0\n\t"
		"not %0\n\t"
		"and %0, %1\n\t"
		: "=r"(x)
		: "r"(x)
		);
	return x;
}

static inline float32_t ReLU_3C_float32(float32_t x) {
	__asm__ (
		"sar $31, %0\n\t"
		"not %0\n\t"
		"and %0, %1\n\t"
		: "=r"(*(uint32_t*)&x)
		: "r"(*(uint32_t*)&x)
		);
	return *(float32_t*)&x;
}

static inline float64_t ReLU_3C_float64(float64_t x) {
	__asm__ (
		"sar $63, %0\n\t"
		"not %0\n\t"
		"and %0, %1\n\t"
		: "=r"(*(uint64_t*)&x)
		: "r"(*(uint64_t*)&x)
		);
	return *(float64_t*)&x;
}

#endif
