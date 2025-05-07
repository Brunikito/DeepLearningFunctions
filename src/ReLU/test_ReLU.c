#include <stdio.h>
#include "ReLU_asm.h"

// Função auxiliar para comparar o valor esperado com o resultado
#define EXPECT_EQUAL_INT(x, expected) \
    if ((x) == (expected)) { \
        printf("PASS: Expected %d, got %d\n", expected, x); \
    } else { \
        printf("FAIL: Expected %d, got %d\n", expected, x); \
    }

#define EXPECT_EQUAL_FLOAT(x, expected) \
    if ((x) == (expected)) { \
        printf("PASS: Expected %.6f, got %.6f\n", expected, x); \
    } else { \
        printf("FAIL: Expected %.6f, got %.6f\n", expected, x); \
    }

int main() {
    // Testando para inteiros
    int8_t int8_input = -5;
    int16_t int16_input = -12345;
    int32_t int32_input = -123456789;
    int64_t int64_input = -9876543210;

    printf("Testing ReLU_2C with int types:\n");
    EXPECT_EQUAL_INT(ReLU_2C(int8_input), (int8_input >= 0 ? int8_input : 0));
    EXPECT_EQUAL_INT(ReLU_2C(int16_input), (int16_input >= 0 ? int16_input : 0));
    EXPECT_EQUAL_INT(ReLU_2C(int32_input), (int32_input >= 0 ? int32_input : 0));
    EXPECT_EQUAL_INT(ReLU_2C(int64_input), (int64_input >= 0 ? int64_input : 0));

    printf("\nTesting ReLU_3C with int types:\n");
    EXPECT_EQUAL_INT(ReLU_3C(int8_input), (int8_input >= 0 ? int8_input : 0));
    EXPECT_EQUAL_INT(ReLU_3C(int16_input), (int16_input >= 0 ? int16_input : 0));
    EXPECT_EQUAL_INT(ReLU_3C(int32_input), (int32_input >= 0 ? int32_input : 0));
    EXPECT_EQUAL_INT(ReLU_3C(int64_input), (int64_input >= 0 ? int64_input : 0));

    // Testando para floats
    float32_t float32_input = -3.14f;
    float64_t float64_input = -2.718;

    printf("\nTesting ReLU_2C with float types:\n");
    EXPECT_EQUAL_FLOAT(ReLU_2C(float32_input), (float32_input >= 0 ? float32_input : 0));
    EXPECT_EQUAL_FLOAT(ReLU_2C(float64_input), (float64_input >= 0 ? float64_input : 0));

    printf("\nTesting ReLU_3C with float types:\n");
    EXPECT_EQUAL_FLOAT(ReLU_3C(float32_input), (float32_input >= 0 ? float32_input : 0));
    EXPECT_EQUAL_FLOAT(ReLU_3C(float64_input), (float64_input >= 0 ? float64_input : 0));

    return 0;
}
