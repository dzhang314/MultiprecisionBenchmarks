#include <benchmark/benchmark.h>
#include <qd/dd_real.h>
#include <qd/qd_real.h>

#include "kernels.hpp"

static void axpy_bench_2(benchmark::State &bs) { axpy_bench<dd_real>(bs); }
static void axpy_bench_4(benchmark::State &bs) { axpy_bench<qd_real>(bs); }

static void dot_bench_2(benchmark::State &bs) { dot_bench<dd_real>(bs); }
static void dot_bench_4(benchmark::State &bs) { dot_bench<qd_real>(bs); }

static void gemv_bench_2(benchmark::State &bs) { gemv_bench<dd_real>(bs); }
static void gemv_bench_4(benchmark::State &bs) { gemv_bench<qd_real>(bs); }

static void gemm_bench_2(benchmark::State &bs) { gemm_bench<dd_real>(bs); }
static void gemm_bench_4(benchmark::State &bs) { gemm_bench<qd_real>(bs); }

BENCHMARK(axpy_bench_2)
    ->UseManualTime()
    ->Complexity(benchmark::oN)
    ->Repetitions(3)
    ->RangeMultiplier(2)
    ->Range(1L << 8, 1L << 24)
    ->DisplayAggregatesOnly();

BENCHMARK(axpy_bench_4)
    ->UseManualTime()
    ->Complexity(benchmark::oN)
    ->Repetitions(3)
    ->RangeMultiplier(2)
    ->Range(1L << 8, 1L << 24)
    ->DisplayAggregatesOnly();

BENCHMARK(dot_bench_2)
    ->UseManualTime()
    ->Complexity(benchmark::oN)
    ->Repetitions(3)
    ->RangeMultiplier(2)
    ->Range(1L << 8, 1L << 24)
    ->DisplayAggregatesOnly();

BENCHMARK(dot_bench_4)
    ->UseManualTime()
    ->Complexity(benchmark::oN)
    ->Repetitions(3)
    ->RangeMultiplier(2)
    ->Range(1L << 8, 1L << 24)
    ->DisplayAggregatesOnly();

BENCHMARK(gemv_bench_2)
    ->UseManualTime()
    ->Complexity(benchmark::oNSquared)
    ->Repetitions(3)
    ->RangeMultiplier(2)
    ->Range(1L << 4, 1L << 12)
    ->DisplayAggregatesOnly();

BENCHMARK(gemv_bench_4)
    ->UseManualTime()
    ->Complexity(benchmark::oNSquared)
    ->Repetitions(3)
    ->RangeMultiplier(2)
    ->Range(1L << 4, 1L << 12)
    ->DisplayAggregatesOnly();

BENCHMARK(gemm_bench_2)
    ->UseManualTime()
    ->Complexity(benchmark::oNCubed)
    ->Repetitions(3)
    ->RangeMultiplier(2)
    ->Range(1L << 3, 1L << 9)
    ->DisplayAggregatesOnly();

BENCHMARK(gemm_bench_4)
    ->UseManualTime()
    ->Complexity(benchmark::oNCubed)
    ->Repetitions(3)
    ->RangeMultiplier(2)
    ->Range(1L << 3, 1L << 9)
    ->DisplayAggregatesOnly();
