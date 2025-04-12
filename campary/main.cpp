#include <benchmark/benchmark.h>
#include <campary/multi_prec.h>

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
