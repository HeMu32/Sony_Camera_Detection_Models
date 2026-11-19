/* Read-only TP trace for the pinned x86-64 V8 SDK. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static FILE *stream, *writes;
static unsigned nn_count, tp_count;
static _Thread_local unsigned active, samples;
static _Thread_local unsigned writing, writes_count;

static void require_write(FILE *file, const void *p, size_t bytes)
{
    if (fwrite(p, bytes, 1, file) != 1) {
        perror("NBG trace write");
        exit(1);
    }
}

static void event(uint32_t kind, uint32_t ordinal, uint32_t a, uint32_t b)
{
    const uint32_t row[] = {kind, ordinal, a, b};
    require_write(stream, row, sizeof(row));
}

static void nn_write(uint32_t coordinate, uint32_t value)
{
    const uint32_t row[] = {coordinate, value};
    require_write(writes, row, sizeof(row));
}

static void *resolve(const char *name)
{
    void *fn = dlsym(RTLD_NEXT, name);
    if (!fn) {
        fprintf(stderr, "Unsupported TP trace ABI: %s\n", name);
        exit(1);
    }
    return fn;
}

static FILE *open_file(const char *variable, const char *magic)
{
    const char *path = getenv(variable);
    if (!path || !*path)
        return NULL;
    FILE *file = fopen(path, "wbx");
    if (!file) {
        perror("NBG trace open");
        exit(1);
    }
    require_write(file, magic, 8);
    return file;
}

__attribute__((constructor))
static void open_trace(void)
{
    stream = open_file("NBG_TP_TRACE", "NBTP001");
    writes = open_file("NBG_NN_WRITES", "NBNW001");
}

bool _ZN12cmoNeuralNet11DoNeuralNetEj(void *self, unsigned index)
{
    static bool (*original)(void *, unsigned);
    if (!original)
        original = resolve(__func__);
    unsigned ordinal = ++nn_count;
    if (stream)
        event(1, ordinal, index, tp_count);
    if (writes && ordinal <= 25) {
        writing = ordinal;
        writes_count = 0;
        nn_write(UINT32_MAX, ordinal);
    }
    bool result = original(self, index);
    if (writing) {
        nn_write(UINT32_MAX - 1, writes_count);
        fflush(writes);
        writing = 0;
    }
    return result;
}

void _ZN12cmoNeuralNet10OutputDataEjjji(void *self, unsigned x, unsigned y,
                                      unsigned channel, int value)
{
    static void (*original)(void *, unsigned, unsigned, unsigned, int);
    if (!original)
        original = resolve(__func__);
    if (writing) {
        if (x >= 1024 || y >= 1024 || channel >= 1024) {
            fputs("NN coordinates outside trace format\n", stderr);
            exit(1);
        }
        nn_write(x | (y << 10) | (channel << 20), (uint32_t)value);
        ++writes_count;
    }
    original(self, x, y, channel, value);
}

bool _ZN11cmoTpEngine18DoTensorProcessingEj(void *self, unsigned index)
{
    static bool (*original)(void *, unsigned);
    if (!original)
        original = resolve(__func__);
    if (stream) {
        active = ++tp_count;
        samples = 0;
        event(2, active, index, nn_count);
    }
    bool result = original(self, index);
    if (active) {
        event(5, active, samples, result);
        fflush(stream);
        active = 0;
    }
    return result;
}

void _ZN11cmoTpEngine9GetTpInstEj(unsigned char *self, unsigned index)
{
    static void (*original)(unsigned char *, unsigned);
    if (!original)
        original = resolve(__func__);
    original(self, index);
    if (!active)
        return;
    uint32_t base;
    memcpy(&base, self + 0x8a4, sizeof(base));
    event(3, active, base, index);
    require_write(stream, self + 0x990, 0x80);
}

unsigned _ZN11cmoTpEngine13AluProcessingEj(void *self, unsigned input)
{
    static unsigned (*original)(void *, unsigned);
    if (!original)
        original = resolve(__func__);
    unsigned output = original(self, input);
    if (active && samples < 65536) {
        event(4, active, input, output);
        ++samples;
    }
    return output;
}
