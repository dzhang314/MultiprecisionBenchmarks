#include <benchmark/benchmark.h>
#include <campary/multi_prec_certif.h>

#include "kernels.hpp"

static void axpy_bench_1(benchmark::State &bs) {
    axpy_bench<multi_prec<1>>(bs);
}
static void axpy_bench_2(benchmark::State &bs) {
    axpy_bench<multi_prec<2>>(bs);
}
static void axpy_bench_3(benchmark::State &bs) {
    axpy_bench<multi_prec<3>>(bs);
}
static void axpy_bench_4(benchmark::State &bs) {
    axpy_bench<multi_prec<4>>(bs);
}

static void dot_bench_1(benchmark::State &bs) { dot_bench<multi_prec<1>>(bs); }
static void dot_bench_2(benchmark::State &bs) { dot_bench<multi_prec<2>>(bs); }
static void dot_bench_3(benchmark::State &bs) { dot_bench<multi_prec<3>>(bs); }
static void dot_bench_4(benchmark::State &bs) { dot_bench<multi_prec<4>>(bs); }

static void gemv_bench_1(benchmark::State &bs) {
    gemv_bench<multi_prec<1>>(bs);
}
static void gemv_bench_2(benchmark::State &bs) {
    gemv_bench<multi_prec<2>>(bs);
}
static void gemv_bench_3(benchmark::State &bs) {
    gemv_bench<multi_prec<3>>(bs);
}
static void gemv_bench_4(benchmark::State &bs) {
    gemv_bench<multi_prec<4>>(bs);
}

static void gemm_bench_1(benchmark::State &bs) {
    gemm_bench<multi_prec<1>>(bs);
}
static void gemm_bench_2(benchmark::State &bs) {
    gemm_bench<multi_prec<2>>(bs);
}
static void gemm_bench_3(benchmark::State &bs) {
    gemm_bench<multi_prec<3>>(bs);
}
static void gemm_bench_4(benchmark::State &bs) {
    gemm_bench<multi_prec<4>>(bs);
}

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
