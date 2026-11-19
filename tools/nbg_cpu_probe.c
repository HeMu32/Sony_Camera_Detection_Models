/* Execute one unmodified NBG using the VeriSilicon OpenVX CPU simulator. */
#include <VX/vx.h>
#include <VX/vx_khr_import_kernel.h>
#include <VX/vx_khr_nn.h>
#include <VX/vx_khr_nn_internal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_DIMS 6

static void check(vx_status status, const char *step)
{
    if (status != VX_SUCCESS) {
        fprintf(stderr, "%s failed: %d\n", step, status);
        exit(1);
    }
}

static size_t type_bytes(vx_enum type)
{
    switch (type) {
    case VX_TYPE_INT8: case VX_TYPE_UINT8: return 1;
    case VX_TYPE_INT16: case VX_TYPE_UINT16: case VX_TYPE_FLOAT16: return 2;
    case VX_TYPE_INT32: case VX_TYPE_UINT32: case VX_TYPE_FLOAT32: return 4;
    default:
        fprintf(stderr, "unsupported tensor data type: %#x\n", type);
        exit(1);
    }
}

static unsigned char *read_file(const char *path, size_t exact)
{
    FILE *file = fopen(path, "rb");
    if (!file || fseek(file, 0, SEEK_END) != 0) {
        perror(path);
        exit(1);
    }
    long length = ftell(file);
    if (length <= 0 || (exact && (size_t)length != exact) ||
        fseek(file, 0, SEEK_SET) != 0) {
        fprintf(stderr, "%s: invalid length %ld (expected %zu)\n", path, length, exact);
        exit(1);
    }
    unsigned char *bytes = malloc(length);
    if (!bytes || fread(bytes, 1, length, file) != (size_t)length) {
        fprintf(stderr, "%s: read failed\n", path);
        exit(1);
    }
    fclose(file);
    return bytes;
}

struct tensor_io {
    vx_tensor tensor;
    vx_enum direction;
    vx_size rank;
    vx_size shape[MAX_DIMS];
    vx_size stride[MAX_DIMS];
    size_t bytes;
};

static struct tensor_io bind_tensor(vx_context context, vx_kernel kernel, vx_node node,
                                   vx_uint32 index)
{
    struct tensor_io io = {0};
    vx_parameter param = vxGetKernelParameterByIndex(kernel, index);
    vx_meta_format meta;
    vx_uint32 dims[MAX_DIMS];
    vx_tensor_create_params_t spec = {0};
    check(vxGetStatus((vx_reference)param), "vxGetKernelParameterByIndex");
    check(vxQueryParameter(param, VX_PARAMETER_DIRECTION, &io.direction,
                           sizeof(io.direction)), "query direction");
    check(vxQueryParameter(param, VX_PARAMETER_META_FORMAT, &meta, sizeof(meta)), "query meta");
    check(vxQueryMetaFormatAttribute(meta, VX_TENSOR_NUMBER_OF_DIMS, &io.rank,
                                     sizeof(io.rank)), "query rank");
    if (!io.rank || io.rank > MAX_DIMS ||
        (io.direction != VX_INPUT && io.direction != VX_OUTPUT)) {
        fprintf(stderr, "invalid tensor rank or direction for parameter %u\n", index);
        exit(1);
    }
    check(vxQueryMetaFormatAttribute(meta, VX_TENSOR_DIMS, io.shape,
                                     sizeof(io.shape)), "query dimensions");
    check(vxQueryMetaFormatAttribute(meta, VX_TENSOR_DATA_TYPE, &spec.data_format,
                                     sizeof(spec.data_format)), "query type");
    check(vxQueryMetaFormatAttribute(meta, VX_TENSOR_QUANT_FORMAT, &spec.quant_format,
                                     sizeof(spec.quant_format)), "query quantization");
    if (spec.quant_format == VX_QUANT_AFFINE_SCALE) {
        check(vxQueryMetaFormatAttribute(meta, VX_TENSOR_ZERO_POINT,
                                         &spec.quant_data.affine.zeroPoint,
                                         sizeof(spec.quant_data.affine.zeroPoint)), "query zero point");
        check(vxQueryMetaFormatAttribute(meta, VX_TENSOR_SCALE, &spec.quant_data.affine.scale,
                                         sizeof(spec.quant_data.affine.scale)), "query scale");
    } else if (spec.quant_format == VX_QUANT_DYNAMIC_FIXED_POINT) {
        check(vxQueryMetaFormatAttribute(meta, VX_TENSOR_FIXED_POINT_POSITION,
                                         &spec.quant_data.dfp.fixed_point_pos,
                                         sizeof(spec.quant_data.dfp.fixed_point_pos)), "query fixed point");
    } else if (spec.quant_format != VX_QUANT_NONE) {
        fprintf(stderr, "unsupported tensor quantization: %#x\n", spec.quant_format);
        exit(1);
    }
    io.bytes = type_bytes(spec.data_format);
    printf("param %u %s dtype=%#x quant=%#x shape=", index,
           io.direction == VX_INPUT ? "input" : "output", spec.data_format, spec.quant_format);
    for (size_t d = 0; d < io.rank; ++d) {
        if (!io.shape[d] || io.shape[d] > UINT32_MAX || SIZE_MAX / io.bytes < io.shape[d]) {
            fprintf(stderr, "invalid tensor size\n");
            exit(1);
        }
        dims[d] = (vx_uint32)io.shape[d];
        io.stride[d] = io.bytes;
        io.bytes *= io.shape[d];
        printf("%s%u", d ? "x" : "", dims[d]);
    }
    printf(" bytes=%zu", io.bytes);
    if (spec.quant_format == VX_QUANT_AFFINE_SCALE) {
        printf(" scale=%.9g zp=%d", spec.quant_data.affine.scale,
               spec.quant_data.affine.zeroPoint);
    }
    puts("");
    spec.num_of_dims = (vx_uint32)io.rank;
    spec.sizes = dims;
    io.tensor = vxCreateTensor2(context, &spec, sizeof(spec));
    check(vxGetStatus((vx_reference)io.tensor), "vxCreateTensor2");
    check(vxSetParameterByIndex(node, index, (vx_reference)io.tensor), "vxSetParameterByIndex");
    vxReleaseParameter(&param);
    return io;
}

int main(int argc, char **argv)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc < 2 || (argc == 2 && strcmp(argv[1], "--help") == 0)) {
        fprintf(stderr, "usage: %s model.nb [output-prefix raw-input-0 ...]\n"
                        "With no output prefix, import and verify without execution.\n", argv[0]);
        return 2;
    }
    unsigned char *model = read_file(argv[1], 0);
    vx_context context = vxCreateContext();
    check(vxGetStatus((vx_reference)context), "vxCreateContext");
    vx_graph graph = vxCreateGraph(context);
    check(vxGetStatus((vx_reference)graph), "vxCreateGraph");
    vx_kernel kernel = vxImportKernelFromURL(context, VX_VIVANTE_IMPORT_KERNEL_FROM_POINTER,
                                             (const vx_char *)model);
    check(vxGetStatus((vx_reference)kernel), "vxImportKernelFromURL");
    vx_uint32 count = 0;
    check(vxQueryKernel(kernel, VX_KERNEL_PARAMETERS, &count, sizeof(count)), "query parameters");
    if (!count || count > 256) {
        fprintf(stderr, "invalid parameter count: %u\n", count);
        return 1;
    }
    vx_node node = vxCreateGenericNode(graph, kernel);
    check(vxGetStatus((vx_reference)node), "vxCreateGenericNode");
    struct tensor_io io[256];
    vx_size start[MAX_DIMS] = {0};
    int next_arg = 3;
    for (vx_uint32 i = 0; i < count; ++i) {
        io[i] = bind_tensor(context, kernel, node, i);
        if (argc > 2 && io[i].direction == VX_INPUT) {
            if (next_arg >= argc) {
                fprintf(stderr, "missing raw input for parameter %u\n", i);
                return 2;
            }
            unsigned char *input = read_file(argv[next_arg++], io[i].bytes);
            check(vxCopyTensorPatch(io[i].tensor, io[i].rank, start, io[i].shape, io[i].stride,
                                    input, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST), "write input");
            free(input);
        }
    }
    if (argc > 2 && next_arg != argc) {
        fprintf(stderr, "too many raw input arguments\n");
        return 2;
    }
    check(vxVerifyGraph(graph), "vxVerifyGraph");
    puts("graph verified");
    if (argc > 2) {
        if (getenv("NBG_PROBE_SLOT08_SENTINELS")) {
            static const vx_uint32 slots[] = {1, 8, 15};
            if (count != 22 || io[0].direction != VX_INPUT) {
                fputs("sentinel probe expects the slot-08 interface\n", stderr);
                return 2;
            }
            for (size_t j = 0; j < sizeof(slots) / sizeof(slots[0]); ++j) {
                const struct tensor_io *out = &io[slots[j]];
                unsigned char pattern[6];
                if (out->direction != VX_OUTPUT || out->bytes != sizeof(pattern) ||
                    out->rank != 4 || out->shape[0] != 1 || out->shape[1] != 1 ||
                    out->shape[2] != 6 || out->shape[3] != 1) {
                    fputs("unexpected sentinel tensor geometry\n", stderr);
                    return 2;
                }
                for (size_t k = 0; k < sizeof(pattern); ++k)
                    pattern[k] = (unsigned char)(0xa0 + 8 * j + k);
                check(vxCopyTensorPatch(out->tensor, out->rank, start, out->shape, out->stride,
                                        pattern, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST),
                      "seed slot-08 sentinel output");
            }
        }
        check(vxProcessGraph(graph), "vxProcessGraph");
        for (vx_uint32 i = 0, out = 0; i < count; ++i) {
            if (io[i].direction != VX_OUTPUT) {
                continue;
            }
            unsigned char *data = malloc(io[i].bytes);
            char path[4096];
            int n = snprintf(path, sizeof(path), "%s.%u.raw", argv[2], out++);
            if (!data || n < 0 || (size_t)n >= sizeof(path)) {
                fprintf(stderr, "output allocation or path failure\n");
                return 1;
            }
            check(vxCopyTensorPatch(io[i].tensor, io[i].rank, start, io[i].shape, io[i].stride,
                                    data, VX_READ_ONLY, VX_MEMORY_TYPE_HOST), "read output");
            FILE *file = fopen(path, "wb");
            if (!file || fwrite(data, 1, io[i].bytes, file) != io[i].bytes || fclose(file)) {
                perror(path);
                return 1;
            }
            printf("wrote %s\n", path);
            free(data);
        }
    }
    for (vx_uint32 i = 0; i < count; ++i) {
        vxReleaseTensor(&io[i].tensor);
    }
    vxReleaseNode(&node);
    vxReleaseKernel(&kernel);
    vxReleaseGraph(&graph);
    vxReleaseContext(&context);
    free(model);
    return 0;
}
