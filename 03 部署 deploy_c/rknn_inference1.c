#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "rknn_api.h"

#define MODEL_PATH  "/root/modulation_resnet_4d_fixed.rknn"
#define DATA_PATH   "/root/test_data.bin"
#define LABEL_PATH  "/root/test_labels.bin"
#define RESULT_PATH "/root/pred_int8.txt"

#define NUM_SAMPLES 44000
#define INPUT_ELEMS 512
#define NUM_CLASSES 11

static inline int8_t quantize_i8(float x, float scale, int32_t zp) {
    if (scale <= 0.0f) return 0;
    float q = x / scale + (float)zp;
    if (q < -128.0f) q = -128.0f;
    else if (q > 127.0f) q = 127.0f;
    return (int8_t)nearbyintf(q);
}

int main(void) {
    int ret;
    FILE *fp = fopen(MODEL_PATH, "rb");
    fseek(fp, 0, SEEK_END); long model_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    void *model_data = malloc(model_size);
    fread(model_data, 1, model_size, fp); fclose(fp);

    rknn_context ctx = 0;
    ret = rknn_init(&ctx, model_data, model_size, 0, NULL);
    if (ret < 0) { printf("rknn_init failed\n"); return -1; }

    rknn_tensor_attr in_native = {0}; in_native.index = 0;
    rknn_query(ctx, RKNN_QUERY_NATIVE_INPUT_ATTR, &in_native, sizeof(in_native));

    rknn_tensor_attr out_native = {0}; out_native.index = 0;
    rknn_query(ctx, RKNN_QUERY_NATIVE_OUTPUT_ATTR, &out_native, sizeof(out_native));

    printf("INPUT  native: dims=%d,%d,%d,%d scale=%f zp=%d\n",
           in_native.dims[0], in_native.dims[1], in_native.dims[2], in_native.dims[3],
           in_native.scale, in_native.zp);
    printf("OUTPUT native: dims=%d,%d,%d,%d,%d size=%d\n",
           out_native.dims[0], out_native.dims[1], out_native.dims[2],
           out_native.dims[3], out_native.dims[4], out_native.size_with_stride);

    // 输入配置
    in_native.type = RKNN_TENSOR_INT8;
    in_native.pass_through = 1;
    rknn_tensor_mem *input_mem = rknn_create_mem(ctx, in_native.size_with_stride);
    rknn_set_io_mem(ctx, input_mem, &in_native);

    // 输出配置
    out_native.type = RKNN_TENSOR_INT8;
    out_native.pass_through = 1;
    rknn_tensor_mem *output_mem = rknn_create_mem(ctx, out_native.size_with_stride);
    rknn_set_io_mem(ctx, output_mem, &out_native);

    FILE *dfp = fopen(DATA_PATH, "rb");
    FILE *lfp = fopen(LABEL_PATH, "rb");
    FILE *rfp = fopen(RESULT_PATH, "w");
    int8_t *label_buf = malloc(NUM_SAMPLES);
    fread(label_buf, 1, NUM_SAMPLES, lfp);

    float  *src = malloc(INPUT_ELEMS * sizeof(float));
    int8_t *dst_i8 = (int8_t *)input_mem->virt_addr;
    int8_t *out_i8 = (int8_t *)output_mem->virt_addr;

    int correct = 0;
    for (int i = 0; i < NUM_SAMPLES; ++i) {
        fread(src, sizeof(float), INPUT_ELEMS, dfp);

        // 按 NCHW 平铺量化
        memset(dst_i8, (uint8_t)(int8_t)in_native.zp, in_native.size_with_stride);
        for (int k = 0; k < INPUT_ELEMS; ++k) {
            dst_i8[k] = quantize_i8(src[k], in_native.scale, in_native.zp);
        }

        // 🌟 关键：把 CPU 算好的数据同步给 NPU
        rknn_mem_sync(ctx, input_mem, RKNN_MEMORY_SYNC_TO_DEVICE);

        rknn_run(ctx, NULL);

        // 🌟 关键：把 NPU 算好的结果同步给 CPU
        rknn_mem_sync(ctx, output_mem, RKNN_MEMORY_SYNC_FROM_DEVICE);

        int pred = 0;
        int8_t best = out_i8[0];
        for (int j = 1; j < NUM_CLASSES; ++j) {
            if (out_i8[j] > best) { best = out_i8[j]; pred = j; }
        }

        fprintf(rfp, "%d\n", pred);
        if (pred == label_buf[i]) correct++;

        if ((i + 1) % 5000 == 0) {
            printf("  [%5d/%d]  running acc = %.4f\n", i + 1, NUM_SAMPLES, (float)correct / (i + 1));
        }
    }

    printf("\n========== 板端最终结果 ==========\n");
    printf("总样本数 : %d\n", NUM_SAMPLES);
    printf("板端准确率: %.4f\n", (float)correct / NUM_SAMPLES);

    free(src); free(label_buf);
    fclose(dfp); fclose(lfp); fclose(rfp);
    rknn_destroy_mem(ctx, input_mem);
    rknn_destroy_mem(ctx, output_mem);
    rknn_destroy(ctx); free(model_data);
    return 0;
}
