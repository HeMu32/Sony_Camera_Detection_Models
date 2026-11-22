/* Optional read-only coefficient trace for the pinned x86-64 VIP V8 SDK. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static FILE *stream;
static FILE *instructions;
static unsigned operations, limit, tile_limit, post_limit;
static bool parameters_only, depthwise_taps;
static _Thread_local unsigned active;
static _Thread_local unsigned remaining;
static _Thread_local unsigned tiles;
static _Thread_local unsigned post_channel;
static _Thread_local uint16_t post_count[512];
static _Thread_local uint16_t output_count[512];

static void emit(uint32_t type, uint32_t a, uint32_t b, uint32_t c,
                 uint32_t d, uint32_t e, uint32_t f, uint32_t g)
{
    const uint32_t row[] = {type, active, a, b, c, d, e, f, g};
    if (fwrite(row, sizeof(row), 1, stream) != 1) {
        perror("coefficient trace");
        exit(1);
    }
}

static void *require_symbol(const char *name)
{
    void *result = dlsym(RTLD_NEXT, name);
    if (!result) {
        fprintf(stderr, "Unsupported simulator coefficient ABI: %s\n", name);
        exit(1);
    }
    return result;
}

__attribute__((constructor))
static void open_trace(void)
{
    const char *path = getenv("NBG_COEFF_TRACE");
    if (!path || !*path)
        return;
    const char *value = getenv("NBG_COEFF_TRACE_OPS");
    unsigned long n = value ? strtoul(value, NULL, 10) : 8;
    if (!n || n > 1000) {
        fputs("NBG_COEFF_TRACE_OPS must be 1..1000\n", stderr);
        exit(1);
    }
    limit = (unsigned)n;
    value = getenv("NBG_CONV_TRACE_TILES");
    n = value ? strtoul(value, NULL, 10) : 0;
    if (n > 2048) {
        fputs("NBG_CONV_TRACE_TILES must be 0..2048\n", stderr);
        exit(1);
    }
    tile_limit = (unsigned)n;
    value = getenv("NBG_CONV_TRACE_POST_SAMPLES");
    n = value ? strtoul(value, NULL, 10) : 512;
    if (!n || n > 512) {
        fputs("NBG_CONV_TRACE_POST_SAMPLES must be 1..512\n", stderr);
        exit(1);
    }
    post_limit = (unsigned)n;
    const char *params = getenv("NBG_COEFF_PARAMS_ONLY");
    parameters_only = params && !strcmp(params, "1");
    const char *depthwise = getenv("NBG_POSE_DEPTHWISE_TAPS");
    depthwise_taps = depthwise && !strcmp(depthwise, "1");
    stream = fopen(path, "wbx");
    if (!stream || fwrite("NBCOEF1\0", 8, 1, stream) != 1) {
        perror("open coefficient trace");
        exit(1);
    }
    const char *commands = getenv("NBG_NN_COMMANDS");
    if (commands && *commands) {
        instructions = fopen(commands, "wbx");
        if (!instructions || fwrite("NBNNC01\0", 8, 1, instructions) != 1) {
            perror("open NN command trace");
            exit(1);
        }
    }
}

bool _ZN12cmoNeuralNet11DoNeuralNetEj(void *self, unsigned address)
{
    static bool (*original)(void *, unsigned);
    if (!original)
        original = require_symbol(__func__);
    if (stream && ++operations <= limit) {
        active = operations;
        remaining = 3000000;
        tiles = 0;
        memset(post_count, 0, sizeof(post_count));
        memset(output_count, 0, sizeof(output_count));
        post_channel = UINT32_MAX;
        emit(0, address, 0, 0, 0, 0, 0, 0);
    }
    bool result = original(self, address);
    if (active) {
        emit(3, remaining, 0, 0, 0, 0, 0, 0);
        fflush(stream);
        active = 0;
    }
    return result;
}

void _ZN12cmoNeuralNet9GetNnInstEj(unsigned char *self, unsigned index)
{
    static void (*original)(unsigned char *, unsigned);
    if (!original)
        original = require_symbol(__func__);
    original(self, index);
    if (active && instructions) {
        const uint32_t header[] = {active, index};
        /* GetNnInst loads 32 command words at object offset 0x2718. */
        if (fwrite(header, sizeof(header), 1, instructions) != 1 ||
            fwrite(self + 0x2718, 0x80, 1, instructions) != 1) {
            perror("NN command trace");
            exit(1);
        }
    }
}

void _ZN12cmoNeuralNet11SetCoefDataEjjjjs(void *self, unsigned x, unsigned y,
                                       unsigned z, unsigned k, short value)
{
    static void (*original)(void *, unsigned, unsigned, unsigned, unsigned, short);
    if (!original)
        original = require_symbol(__func__);
    if (active && remaining) {
        --remaining;
        emit(1, x, y, z, k, (uint32_t)(int32_t)value, 0, 0);
    }
    original(self, x, y, z, k, value);
}

void _ZN12cmoNeuralNet11SetCoefDataEjjjjshh(void *self, unsigned x, unsigned y,
                                         unsigned z, unsigned k, short value,
                                         unsigned char flag, unsigned char lane)
{
    static void (*original)(void *, unsigned, unsigned, unsigned, unsigned,
                            short, unsigned char, unsigned char);
    if (!original)
        original = require_symbol(__func__);
    if (active && remaining) {
        --remaining;
        emit(2, x, y, z, k, (uint32_t)(int32_t)value, flag, lane);
    }
    original(self, x, y, z, k, value, flag, lane);
}

void _ZN12cmoNeuralNet21SetDepthwiseVZReorderEtt(void *self, unsigned short logical,
                                                unsigned short kernel)
{
    static void (*original)(void *, unsigned short, unsigned short);
    if (!original)
        original = require_symbol(__func__);
    original(self, logical, kernel);
    if (active && depthwise_taps)
        emit(10, logical, kernel, 0, 0, 0, 0, 0);
}

static uint32_t get_word(const unsigned char *self, size_t offset)
{
    uint32_t word;
    memcpy(&word, self + offset, sizeof(word));
    return word;
}

void _ZN12cmoNeuralNet10SetVipSramEv(unsigned char *self)
{
    static void (*original)(unsigned char *);
    if (!original)
        original = require_symbol(__func__);
    original(self);
    const char *path = active == 1 ? getenv("NBG_POSE_STEM_SRAM") : NULL;
    if (!path || !*path)
        return;

    /* Snapshot the known model's input region after the four TP stripes. */
    const unsigned start = 0x800, output_base = get_word(self, 0x8b4);
    if (get_word(self, 0x8ac) != 0x400800 ||
        (output_base != 0x459b00 && output_base != 0x4fc700 && output_base != 0x492500)) {
        fputs("Unexpected pose stem SRAM address\n", stderr);
        exit(1);
    }
    const unsigned end = output_base - 0x400000;
    unsigned char *owner, *sram;
    memcpy(&owner, self + 0x778, sizeof(owner));
    if (!owner) {
        fputs("Missing pose stem SRAM owner\n", stderr);
        exit(1);
    }
    memcpy(&sram, owner + 8, sizeof(sram));
    if (!sram || get_word(sram, 0x21c) < end) {
        fputs("Unexpected pose stem SRAM size\n", stderr);
        exit(1);
    }
    void (*read_data)(void *, unsigned, unsigned, unsigned char *) =
        require_symbol("_ZN10cmoVipSram14DirectReadDataEjjPh");
    unsigned char *data = malloc(end - start);
    FILE *file = fopen(path, "wbx");
    if (!data || !file) {
        perror("pose stem SRAM trace");
        exit(1);
    }
    read_data(sram, start, end - start, data);
    if (fwrite(data, end - start, 1, file) != 1 || fclose(file) != 0) {
        perror("pose stem SRAM trace");
        exit(1);
    }
    free(data);
}

void _ZN12cmoNeuralNet11PostAddBiasEv(unsigned char *self)
{
    static void (*original)(unsigned char *);
    if (!original)
        original = require_symbol(__func__);
    if (active && (tile_limit || parameters_only) &&
        (self[0x270e] == 0 || self[0x270e] == 2)) {
        const int64_t *bias;
        memcpy(&bias, self + 0x27d8, sizeof(bias));
        if (bias) {
            unsigned count = get_word(self, 0x2728) & 0x3fff;
            for (unsigned k = 0; k < count; ++k) {
                uint64_t value = (uint64_t)bias[k];
                emit(6, k, (uint32_t)value, (uint32_t)(value >> 32), 0, 0, 0, 0);
            }
        }
    }
    original(self);
}

void _ZN12cmoNeuralNet22UpdatePostMultAndShiftEt(unsigned char *self, unsigned short channel)
{
    static void (*original)(unsigned char *, unsigned short);
    if (!original)
        original = require_symbol(__func__);
    original(self, channel);
    if (active && (tile_limit || parameters_only)) {
        post_channel = channel;
        if (channel < 512 && !post_count[channel]) {
            emit(7, channel, get_word(self, 0x2850), get_word(self, 0x2854),
                 get_word(self, 0x5dac), self[0x5daa], self[0x2771], self[0x276b]);
            if (parameters_only)
                post_count[channel] = 1;
        }
    }
}

int _ZN12cmoNeuralNet16PostProcessAccumExx(unsigned char *self, int64_t a, int64_t b)
{
    static int (*original)(unsigned char *, int64_t, int64_t);
    if (!original)
        original = require_symbol(__func__);
    int value = original(self, a, b);
    if (active && tile_limit && !parameters_only &&
        post_channel < 512 && post_count[post_channel] < post_limit) {
        uint64_t input = (uint64_t)a, skip = (uint64_t)b;
        emit(8, post_channel, (uint32_t)input, (uint32_t)(input >> 32),
             (uint32_t)skip, (uint32_t)(skip >> 32), (uint32_t)value, 0);
        ++post_count[post_channel];
    }
    return value;
}

void _ZN12cmoNeuralNet10OutputDataEjjji(void *self, unsigned x, unsigned y,
                                      unsigned channel, int value)
{
    static void (*original)(void *, unsigned, unsigned, unsigned, int);
    if (!original)
        original = require_symbol(__func__);
    if (active && tile_limit && !parameters_only && channel < 512 && output_count[channel] < post_limit) {
        emit(9, channel, x, y, (uint32_t)value, 0, 0, 0);
        ++output_count[channel];
    }
    original(self, x, y, channel, value);
}

/* Record input patches and independent CPU accumulator snapshots for V8 int8. */
void _ZN12cmoNeuralNet9conv_int8EttthhttPA64_cPs15_Int16ConvStagej(
    unsigned char *self, unsigned short start, unsigned short x0, unsigned short y0,
    unsigned char width, unsigned char height, unsigned short taps, unsigned short channel,
    signed char (*input)[64], short *weight, unsigned stage, unsigned flags)
{
    typedef void (*conv)(unsigned char *, unsigned short, unsigned short, unsigned short,
                         unsigned char, unsigned char, unsigned short, unsigned short,
                         signed char (*)[64], short *, unsigned, unsigned);
    static conv original;
    static int64_t (*get_accum)(void *, unsigned, unsigned, unsigned);
    if (!original)
        original = require_symbol(__func__);
    if (!active || parameters_only || tiles >= tile_limit || stage || !taps || taps > 9 || !width ||
        (self[0x2795] & 0x80) || ((self[0x2772] & 2) && !depthwise_taps) ||
        (self[0x270e] != 0 && self[0x270e] != 2)) {
        original(self, start, x0, y0, width, height, taps, channel,
                 input, weight, stage, flags);
        return;
    }
    if (!get_accum)
        get_accum = require_symbol("_ZN12cmoNeuralNet17GetAccumulateDataEjjj");
    unsigned tile = ++tiles;
    unsigned out_w = (get_word(self, 0x2724) >> 6) & 0x1fff;
    unsigned out_h = (get_word(self, 0x2726) & 0xffff) >> 3;
    unsigned out_c = get_word(self, 0x2728) & 0x3fff;
    int64_t before[64] = {0};
    bool valid[64] = {false};
    for (unsigned i = 0; i < 64; ++i) {
        unsigned u = (start + i) % width + x0;
        unsigned v = (start + i) / width + y0;
        if ((start + i) / width < height && u < out_w && v < out_h && channel < out_c) {
            valid[i] = true;
            before[i] = get_accum(self, u, v, channel);
            for (unsigned k = 0; k < taps; ++k)
                emit(4, tile, i, k, (uint32_t)(int32_t)input[k][i],
                     (uint32_t)(int32_t)weight[k], 0, 0);
        }
    }
    original(self, start, x0, y0, width, height, taps, channel,
             input, weight, stage, flags);
    for (unsigned i = 0; i < 64; ++i) {
        if (!valid[i])
            continue;
        unsigned u = (start + i) % width + x0;
        unsigned v = (start + i) / width + y0;
        uint64_t delta = (uint64_t)get_accum(self, u, v, channel) - (uint64_t)before[i];
        emit(5, tile, i, u, v, channel, (uint32_t)delta, (uint32_t)(delta >> 32));
    }
}
