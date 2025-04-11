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
