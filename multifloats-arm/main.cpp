#include <benchmark/benchmark.h>

#include "MultiFloats.hpp"
#include "kernels.hpp"

static void axpy_bench_1(benchmark::State &bs) { axpy_bench<f64x1>(bs); }
static void axpy_bench_2(benchmark::State &bs) { axpy_bench<f64x2>(bs); }
static void axpy_bench_3(benchmark::State &bs) { axpy_bench<f64x3>(bs); }
static void axpy_bench_4(benchmark::State &bs) { axpy_bench<f64x4>(bs); }

static void dot_bench_1(benchmark::State &bs) { dot_bench<f64x1>(bs); }
static void dot_bench_2(benchmark::State &bs) { dot_bench<f64x2>(bs); }
static void dot_bench_3(benchmark::State &bs) { dot_bench<f64x3>(bs); }
static void dot_bench_4(benchmark::State &bs) { dot_bench<f64x4>(bs); }

static void gemv_bench_1(benchmark::State &bs) { gemv_bench<f64x1>(bs); }
static void gemv_bench_2(benchmark::State &bs) { gemv_bench<f64x2>(bs); }
static void gemv_bench_3(benchmark::State &bs) { gemv_bench<f64x3>(bs); }
static void gemv_bench_4(benchmark::State &bs) { gemv_bench<f64x4>(bs); }

static void gemm_bench_1(benchmark::State &bs) { gemm_bench<f64x1>(bs); }
static void gemm_bench_2(benchmark::State &bs) { gemm_bench<f64x2>(bs); }
static void gemm_bench_3(benchmark::State &bs) { gemm_bench<f64x3>(bs); }
static void gemm_bench_4(benchmark::State &bs) { gemm_bench<f64x4>(bs); }

BENCHMARK(axpy_bench_1)
    ->UseManualTime()
    ->Complexity(benchmark::oN)
    ->Repetitions(3)
    ->RangeMultiplier(2)
    ->Range(1L << 8, 1L << 24)
    ->DisplayAggregatesOnly();

BENCHMARK(axpy_bench_2)
    ->UseManualTime()
    ->Complexity(benchmark::oN)
    ->Repetitions(3)
    ->RangeMultiplier(2)
    ->Range(1L << 8, 1L << 24)
    ->DisplayAggregatesOnly();

BENCHMARK(axpy_bench_3)
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

BENCHMARK(dot_bench_1)
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

BENCHMARK(dot_bench_3)
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

BENCHMARK(gemv_bench_1)
    ->UseManualTime()
    ->Complexity(benchmark::oNSquared)
    ->Repetitions(3)
    ->RangeMultiplier(2)
    ->Range(1L << 4, 1L << 12)
    ->DisplayAggregatesOnly();

BENCHMARK(gemv_bench_2)
    ->UseManualTime()
    ->Complexity(benchmark::oNSquared)
    ->Repetitions(3)
    ->RangeMultiplier(2)
    ->Range(1L << 4, 1L << 12)
    ->DisplayAggregatesOnly();

BENCHMARK(gemv_bench_3)
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

BENCHMARK(gemm_bench_1)
    ->UseManualTime()
    ->Complexity(benchmark::oNCubed)
    ->Repetitions(3)
    ->RangeMultiplier(2)
    ->Range(1L << 3, 1L << 9)
    ->DisplayAggregatesOnly();

BENCHMARK(gemm_bench_2)
    ->UseManualTime()
    ->Complexity(benchmark::oNCubed)
    ->Repetitions(3)
    ->RangeMultiplier(2)
    ->Range(1L << 3, 1L << 9)
    ->DisplayAggregatesOnly();

BENCHMARK(gemm_bench_3)
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
