/*
 * Compatibility for standalone Sony v1.13/v1.15/v1.22 NBGs on TIM-VX's
 * x86 OpenVX SDK 6.4.15_CL690884A. Use only with the Sony 0xd4 graphs.
 * The legacy VIP SRAM patch stores the address in bytes, even when the
 * final command encodes a 64-byte unit. Convert the runtime copy only.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int vxoBinaryGraph_Patch32bitVa(void *graph, void *command, const uint32_t *patch,
                               uint64_t base, uint32_t index, void *reference)
{
    typedef int (*patch_fn)(void *, void *, const uint32_t *, uint64_t, uint32_t, void *);
    patch_fn apply = dlsym(RTLD_NEXT, "vxoBinaryGraph_Patch32bitVa");
    if (!apply) {
        fputs("Cannot resolve SDK relocation function\n", stderr);
        exit(1);
    }
    uint32_t version;
    memcpy(&version, (char *)graph + 0x1c, sizeof(version));
    if ((version == 0x1000d || version == 0x1000f || version == 0x10016) &&
        patch[2] == 8 && (patch[5] == 1 || patch[5] == 11)) {
        uint32_t value;
        unsigned char *ptr = (unsigned char *)command + patch[1];
        memcpy(&value, ptr, sizeof(value));
        uint32_t mask = patch[5] == 1 ? 0x03ffffff : UINT32_MAX;
        uint32_t address = value & mask;
        if (address % 64 || address > 0x200000 || patch[4] != 0) {
            fprintf(stderr, "Unsupported legacy VIP SRAM address %#" PRIx32
                            " (origin %#" PRIx32 ")\n", address, patch[4]);
            exit(1);
        }
        value = (value & ~mask) | (address >> 6);
        memcpy(ptr, &value, sizeof(value));
    }
    return apply(graph, command, patch, base, index, reference);
}
