#include <benchmark/benchmark.h>

#include "kernels.hpp"

static void axpy_bench_2(benchmark::State &bs) { axpy_bench<__float128>(bs); }

BENCHMARK(axpy_bench_2)
    ->UseManualTime()
    ->Complexity(benchmark::oN)
    ->Repetitions(3)
    ->RangeMultiplier(2)
    ->Range(1L << 8, 1L << 24)
    ->DisplayAggregatesOnly();
