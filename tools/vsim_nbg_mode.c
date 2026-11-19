/*
 * Compatibility preload for TIM-VX x86 SDK 6.4.15_CL690884A.
 * Imported standalone NBGs do not have the simulator debugger's shader object.
 * Select libEmulator's existing NBG path before its worker executes NN/TP commands.
 * The model file, command stream and coefficients are left unchanged.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

void _ZN10VSimulator10newVXStateEv(void *simulator)
{
    void (*initialize)(void *) = dlsym(RTLD_NEXT, "_ZN10VSimulator10newVXStateEv");
    bool *nbg_mode = dlsym(RTLD_NEXT, "_ZN8_VXState5m_nbgE");
    if (!initialize || !nbg_mode) {
        fputs("Unsupported libEmulator build for the NBG mode shim\n", stderr);
        exit(1);
    }
    initialize(simulator);
    *nbg_mode = true;
}
