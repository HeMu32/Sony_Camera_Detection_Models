/* Optional relocation trace for the pinned TIM-VX x86 OpenVX SDK. */
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
    static unsigned count;
    uint32_t offset = patch[1];
    unsigned char *ptr = (unsigned char *)command + offset;
    uint32_t before, after;
    memcpy(&before, ptr, sizeof(before));
    int status = apply(graph, command, patch, base, index, reference);
    memcpy(&after, ptr, sizeof(after));
    if (count++ < 120) {
        fprintf(stderr,
                "reloc %u op=%u type=%u offset=%#" PRIx32 " source=%u tensor=%d "
                "orig=%#" PRIx32 " masktype=%u base=%#" PRIx64 " data=%#" PRIx32
                " -> %#" PRIx32 " status=%d\n",
                count, index, patch[0], offset, patch[2], (int32_t)patch[3],
                patch[4], patch[5], base, before, after, status);
    }
    return status;
}
