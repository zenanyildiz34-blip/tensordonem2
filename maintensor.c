#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>

/* ================================
   DATA TYPE ENUM
================================ */
typedef enum {
    TYPE_FLOAT32,
    TYPE_FLOAT16,
    TYPE_INT8
} TensorType;

/* ================================
   UNION (bit-level access)
================================ */
typedef union {
    float f32;
    uint32_t raw;
} BitAccess;

/* ================================
   TENSOR STRUCTURE
================================ */
typedef struct {
    int rows;
    int cols;
    TensorType type;

    union {
        float *f32;
        uint16_t *f16; //tinyml tensörünün c deki çekirdeği
        int8_t *i8;
    } data;

    float scale;
    int zero_point;

} Tensor;


/* ================================
   FLOAT32 <-> FLOAT16 (IEEE Basic)
================================ */

uint16_t float32_to_float16(float f)
{
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));

    uint16_t sign = (bits >> 31) & 1;
    int16_t exponent = ((bits >> 23) & 0xFF) - 127 + 15;
    uint16_t mantissa = (bits >> 13) & 0x3FF;

    if (exponent <= 0) exponent = 0;
    if (exponent >= 31) exponent = 31;

    return (sign << 15) | (exponent << 10) | mantissa;
}

float float16_to_float32(uint16_t h)
{
    uint16_t sign = (h >> 15) & 1;
    uint16_t exponent = (h >> 10) & 0x1F;
    uint16_t mantissa = h & 0x3FF;

    int32_t exp32 = exponent - 15 + 127;
    uint32_t bits = (sign << 31) | (exp32 << 23) | (mantissa << 13);

    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}


/* ================================
   CREATE & DESTROY
================================ */

Tensor create_tensor(int rows, int cols, TensorType type, float scale, int zp)
{
    Tensor t;
    t.rows = rows;
    t.cols = cols;
    t.type = type;
    t.scale = scale;
    t.zero_point = zp;

    int total = rows * cols;

    if (type == TYPE_FLOAT32)
        t.data.f32 = malloc(sizeof(float) * total);
    else if (type == TYPE_FLOAT16)
        t.data.f16 = malloc(sizeof(uint16_t) * total);
    else
        t.data.i8 = malloc(sizeof(int8_t) * total);

    return t;
}

void destroy_tensor(Tensor *t)
{
    if (t->type == TYPE_FLOAT32)
        free(t->data.f32);
    else if (t->type == TYPE_FLOAT16)
        free(t->data.f16);
    else
        free(t->data.i8);
}


/* ================================
   MEMORY USAGE
================================ */

size_t memory_usage(Tensor *t)
{
    int total = t->rows * t->cols;

    if (t->type == TYPE_FLOAT32) return total * 4;
    if (t->type == TYPE_FLOAT16) return total * 2;
    return total * 1;
}


/* ================================
   QUANTIZATION
================================ */

int8_t quantize(float v, float scale, int zp)
{
    float q = v / scale + zp;

    if (q > 127) q = 127;
    if (q < -128) q = -128;

    return (int8_t)round(q);
}

float dequantize(int8_t q, float scale, int zp)
{
    return (q - zp) * scale;
}


/* ================================
   PRINT TENSOR
================================ */

void print_tensor(Tensor *t)
{
    for (int i = 0; i < t->rows; i++)
    {
        for (int j = 0; j < t->cols; j++)
        {
            int idx = i * t->cols + j;
            float value;

            if (t->type == TYPE_FLOAT32)
                value = t->data.f32[idx];

            else if (t->type == TYPE_FLOAT16)
                value = float16_to_float32(t->data.f16[idx]);

            else
                value = dequantize(t->data.i8[idx], t->scale, t->zero_point);

            printf("%.3f ", value);
        }
        printf("\n");
    }
}


/* ================================
   MATRIX MULTIPLICATION
   Y = A x B
================================ */

void matmul(Tensor *A, Tensor *B, Tensor *Y)
{
    for (int i = 0; i < A->rows; i++)
    {
        for (int j = 0; j < B->cols; j++)
        {
            float sum = 0;

            for (int k = 0; k < A->cols; k++)
            {
                float a = (A->type == TYPE_FLOAT32)
                    ? A->data.f32[i*A->cols + k]
                    : dequantize(A->data.i8[i*A->cols + k], A->scale, A->zero_point);

                float b = (B->type == TYPE_FLOAT32)
                    ? B->data.f32[k*B->cols + j]
                    : dequantize(B->data.i8[k*B->cols + j], B->scale, B->zero_point);

                sum += a * b;
            }

            Y->data.f32[i*Y->cols + j] = sum;
        }
    }
}


/* ================================
   MAIN
================================ */

int main()
{
    printf("=== TinyML Tensor System ===\n\n");

    /* FP32 Tensor */
    Tensor A = create_tensor(2,2,TYPE_FLOAT32,0.1f,0);
    A.data.f32[0]=1.0f; A.data.f32[1]=2.0f;
    A.data.f32[2]=3.0f; A.data.f32[3]=4.0f;

    printf("Original FP32 Tensor:\n");
    print_tensor(&A);

    /* Quantized Tensor */
    Tensor Q = create_tensor(2,2,TYPE_INT8,0.1f,0);

    for(int i=0;i<4;i++)
        Q.data.i8[i] = quantize(A.data.f32[i],Q.scale,Q.zero_point);

    printf("\nQuantized Tensor (Dequantized View):\n");
    print_tensor(&Q);

    /* Matrix Multiplication */
    Tensor Y = create_tensor(2,2,TYPE_FLOAT32,0,0);
    matmul(&A,&A,&Y);

    printf("\nMatrix Multiplication (A x A):\n");
    print_tensor(&Y);

    /* Union Demo */
    BitAccess u;
    u.f32 = 0.75f;

    printf("\nUnion Demo:\n");
    printf("Float: %f\n", u.f32);
    printf("Raw Hex: 0x%X\n", u.raw);

    printf("\nMemory Usage:\n");
    printf("FP32: %zu bytes\n", memory_usage(&A));
    printf("INT8: %zu bytes\n", memory_usage(&Q));

    destroy_tensor(&A);
    destroy_tensor(&Q);
    destroy_tensor(&Y);

    return 0;
}