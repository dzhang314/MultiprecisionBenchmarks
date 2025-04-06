#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdlib>

#include <benchmark/benchmark.h>
#include <boost/multiprecision/cpp_bin_float.hpp>

using boost::multiprecision::backends::cpp_bin_float;
using boost::multiprecision::backends::digit_base_2;

using mp_t = boost::multiprecision::number<
    cpp_bin_float<102, digit_base_2, void, std::int16_t, -1022, 1023>,
    boost::multiprecision::et_off>;

static void axpy(mp_t *y, mp_t a, const mp_t *x, std::size_t n) {
#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) { y[i] += a * x[i]; }
}

static void axpy_bench(benchmark::State &bs) {

    const std::size_t n = static_cast<std::size_t>(bs.range(0));

    mp_t *const y = static_cast<mp_t *>(std::malloc(n * sizeof(mp_t)));

    const mp_t a = static_cast<mp_t>(0.5);

    mp_t *const x = static_cast<mp_t *>(std::malloc(n * sizeof(mp_t)));

#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) { x[i] = static_cast<mp_t>(i); }

    for (auto _ : bs) {
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
            y[i] = static_cast<mp_t>(2.0) * static_cast<mp_t>(i);
        }
        const auto start = std::chrono::high_resolution_clock::now();
        axpy(y, a, x, n);
        const auto stop = std::chrono::high_resolution_clock::now();
        bs.SetIterationTime(
            std::chrono::duration<double>(stop - start).count());
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
            assert(y[i] == static_cast<mp_t>(2.5) * static_cast<mp_t>(i));
        }
    }

    bs.SetComplexityN(static_cast<benchmark::ComplexityN>(n));
    bs.SetItemsProcessed(static_cast<std::int64_t>(n) * bs.iterations());
}

BENCHMARK(axpy_bench)
    ->UseManualTime()
    ->Complexity(benchmark::oN)
    ->Repetitions(3)
    ->RangeMultiplier(2)
    ->Range(1L << 8, 1L << 24)
    ->DisplayAggregatesOnly();
