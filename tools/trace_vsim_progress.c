/* Report entry and return of simulator NN and TP operations. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

typedef void (*operation_fn)(void *, unsigned);

static void execute(const char *symbol, char unit, void *parser, unsigned address)
{
    operation_fn original = dlsym(RTLD_NEXT, symbol);
    if (!original) {
        fprintf(stderr, "Unresolved simulator operation: %s\n", symbol);
        exit(1);
    }
    static unsigned operation;
    unsigned id = ++operation;
    fprintf(stderr, "op %u %c addr=%#x begin\n", id, unit, address);
    original(parser, address);
    fprintf(stderr, "op %u %c end\n", id, unit);
}

void _ZN11StateParser9executeNNEj(void *parser, unsigned address)
{
    execute("_ZN11StateParser9executeNNEj", 'N', parser, address);
}

void _ZN11StateParser9executeTPEj(void *parser, unsigned address)
{
    execute("_ZN11StateParser9executeTPEj", 'T', parser, address);
}
