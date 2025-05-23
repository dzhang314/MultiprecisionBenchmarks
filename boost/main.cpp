#include <benchmark/benchmark.h>
#include <boost/multiprecision/cpp_bin_float.hpp>

#include "kernels.hpp"

using boost::multiprecision::backends::cpp_bin_float;
using boost::multiprecision::backends::digit_base_2;

using mp1_t = boost::multiprecision::number<
    cpp_bin_float<53, digit_base_2, void, std::int16_t, -1022, 1023>,
    boost::multiprecision::et_off>;

using mp2_t = boost::multiprecision::number<
    cpp_bin_float<103, digit_base_2, void, std::int16_t, -1022, 1023>,
    boost::multiprecision::et_off>;

using mp3_t = boost::multiprecision::number<
    cpp_bin_float<156, digit_base_2, void, std::int16_t, -1022, 1023>,
    boost::multiprecision::et_off>;

using mp4_t = boost::multiprecision::number<
    cpp_bin_float<209, digit_base_2, void, std::int16_t, -1022, 1023>,
    boost::multiprecision::et_off>;

static void axpy_bench_1(benchmark::State &bs) { axpy_bench<mp1_t>(bs); }
static void axpy_bench_2(benchmark::State &bs) { axpy_bench<mp2_t>(bs); }
static void axpy_bench_3(benchmark::State &bs) { axpy_bench<mp3_t>(bs); }
static void axpy_bench_4(benchmark::State &bs) { axpy_bench<mp4_t>(bs); }

static void dot_bench_1(benchmark::State &bs) { dot_bench<mp1_t>(bs); }
static void dot_bench_2(benchmark::State &bs) { dot_bench<mp2_t>(bs); }
static void dot_bench_3(benchmark::State &bs) { dot_bench<mp3_t>(bs); }
static void dot_bench_4(benchmark::State &bs) { dot_bench<mp4_t>(bs); }

static void gemv_bench_1(benchmark::State &bs) { gemv_bench<mp1_t>(bs); }
static void gemv_bench_2(benchmark::State &bs) { gemv_bench<mp2_t>(bs); }
static void gemv_bench_3(benchmark::State &bs) { gemv_bench<mp3_t>(bs); }
static void gemv_bench_4(benchmark::State &bs) { gemv_bench<mp4_t>(bs); }

static void gemm_bench_1(benchmark::State &bs) { gemm_bench<mp1_t>(bs); }
static void gemm_bench_2(benchmark::State &bs) { gemm_bench<mp2_t>(bs); }
static void gemm_bench_3(benchmark::State &bs) { gemm_bench<mp3_t>(bs); }
static void gemm_bench_4(benchmark::State &bs) { gemm_bench<mp4_t>(bs); }

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
