#ifndef GRAPH_BENCHMARK_TIMER_HPP
#define GRAPH_BENCHMARK_TIMER_HPP

#include <chrono>

namespace vpd {

class Timer {
public:
    Timer()
        : start_time_(
              clock_type::now()
          ) {}

    void restart() {
        start_time_ =
            clock_type::now();
    }

    double elapsed_seconds() const {
        const clock_type::time_point stop_time =
            clock_type::now();

        return std::chrono::duration<double>(
            stop_time -
            start_time_
        ).count();
    }

private:
    using clock_type =
        std::chrono::steady_clock;

    clock_type::time_point start_time_;
};

}  // namespace vpd

#endif