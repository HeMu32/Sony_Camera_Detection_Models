/* Optional Linux x86-64 diagnostic preload, not required for inference. */
#define _GNU_SOURCE
#include <execinfo.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <ucontext.h>
#include <unistd.h>

static void trace_fault(int signo, siginfo_t *info, void *context)
{
    const ucontext_t *regs = context;
    dprintf(STDERR_FILENO, "\nsignal %d, code %d, address %p, ip %#llx\n",
            signo, info->si_code, info->si_addr,
            (unsigned long long)regs->uc_mcontext.gregs[REG_RIP]);
    void *stack[30];
    int frames = backtrace(stack, sizeof(stack) / sizeof(*stack));
    backtrace_symbols_fd(stack, frames, STDERR_FILENO);
    int fd = open("/proc/self/maps", O_RDONLY);
    if (fd != -1) {
        char data[4096];
        ssize_t length;
        while ((length = read(fd, data, sizeof(data))) > 0) {
            if (write(STDERR_FILENO, data, length) != length) {
                break;
            }
        }
        close(fd);
    }
    _exit(128 + signo);
}

__attribute__((constructor))
static void install_fault_handler(void)
{
    void *warmup[2];
    backtrace(warmup, sizeof(warmup) / sizeof(*warmup));
    struct sigaction action = {0};
    action.sa_sigaction = trace_fault;
    action.sa_flags = SA_SIGINFO | SA_RESETHAND;
    sigemptyset(&action.sa_mask);
    sigaction(SIGSEGV, &action, NULL);
    sigaction(SIGBUS, &action, NULL);
}
