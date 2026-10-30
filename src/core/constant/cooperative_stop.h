#ifndef CUDADEM_COOPERATIVE_STOP_H
#define CUDADEM_COOPERATIVE_STOP_H

#include <atomic>
#include <csignal>
#include <stdexcept>
#include <signal.h>

namespace cudadem {
// Lock-free atomics are signal-safe and publish across OpenMP host threads.
// The handler never touches CUDA, I/O, allocation, or solver state. Poll only
// at a closed integration boundary, collectively in distributed runners.
static_assert(std::atomic<std::sig_atomic_t>::is_always_lock_free,
              "Cooperative stop requires a signal-safe lock-free atomic flag");
inline std::atomic<std::sig_atomic_t> cooperative_stop_signal{0};
inline void cooperative_stop_handler(int signal_number) noexcept {
    cooperative_stop_signal.store(signal_number, std::memory_order_relaxed);
}
inline bool cooperative_stop_requested() noexcept {
    return cooperative_stop_signal.load(std::memory_order_relaxed) != 0;
}
class CooperativeStopSignals {
    struct sigaction old_usr1_{}, old_term_{}, old_int_{};
public:
    CooperativeStopSignals() {
        cooperative_stop_signal.store(0, std::memory_order_relaxed);
        struct sigaction action{};
        action.sa_handler = cooperative_stop_handler;
        sigemptyset(&action.sa_mask);
        action.sa_flags = SA_RESTART;
        if (sigaction(SIGUSR1, &action, &old_usr1_) != 0)
            throw std::runtime_error("Cannot install SIGUSR1 cooperative-stop handler");
        if (sigaction(SIGTERM, &action, &old_term_) != 0) {
            sigaction(SIGUSR1, &old_usr1_, nullptr);
            throw std::runtime_error("Cannot install SIGTERM cooperative-stop handler");
        }
        if (sigaction(SIGINT, &action, &old_int_) != 0) {
            sigaction(SIGTERM, &old_term_, nullptr);
            sigaction(SIGUSR1, &old_usr1_, nullptr);
            throw std::runtime_error("Cannot install SIGINT cooperative-stop handler");
        }
    }
    ~CooperativeStopSignals() {
        sigaction(SIGUSR1, &old_usr1_, nullptr);
        sigaction(SIGTERM, &old_term_, nullptr);
        sigaction(SIGINT, &old_int_, nullptr);
    }
    CooperativeStopSignals(const CooperativeStopSignals&) = delete;
    CooperativeStopSignals& operator=(const CooperativeStopSignals&) = delete;
};
} // namespace cudadem
#endif
