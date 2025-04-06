#include <cassert>
#include <cstdint>
#include <cstdlib>

#include <benchmark/benchmark.h>
#include <boost/multiprecision/cpp_bin_float.hpp>

using boost::multiprecision::backends::cpp_bin_float;
using boost::multiprecision::backends::digit_base_2;

using boost_float_t = boost::multiprecision::number<
    cpp_bin_float<102, digit_base_2, void, std::int16_t, -1022, 1023>,
    boost::multiprecision::et_off>;

static void axpy(boost_float_t *y, boost_float_t a, const boost_float_t *x,
                 std::size_t n) {
#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) { y[i] += a * x[i]; }
}

static void axpy_bench(benchmark::State &bs) {

    const std::size_t n = static_cast<std::size_t>(bs.range(0));

    boost_float_t *const y =
        static_cast<boost_float_t *>(std::malloc(n * sizeof(boost_float_t)));

    const boost_float_t a = static_cast<boost_float_t>(0.5);

    boost_float_t *const x =
        static_cast<boost_float_t *>(std::malloc(n * sizeof(boost_float_t)));

#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) {
        x[i] = static_cast<boost_float_t>(i);
    }

    for (auto _ : bs) {
        bs.PauseTiming();
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
            y[i] =
                static_cast<boost_float_t>(2.0) * static_cast<boost_float_t>(i);
        }
        bs.ResumeTiming();
        axpy(y, a, x, n);
        bs.PauseTiming();
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
            assert(y[i] == static_cast<boost_float_t>(2.5) *
                               static_cast<boost_float_t>(i));
        }
        bs.ResumeTiming();
    }

    bs.SetComplexityN(static_cast<benchmark::ComplexityN>(n));
    bs.SetItemsProcessed(static_cast<std::int64_t>(n));
}

BENCHMARK(axpy_bench)
    ->Repetitions(8)
    ->RangeMultiplier(2)
    ->Range(1L, 1L << 28)
    ->UseRealTime()
    ->DisplayAggregatesOnly();
