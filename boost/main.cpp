#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdlib>

#include <benchmark/benchmark.h>
#include <boost/multiprecision/cpp_bin_float.hpp>

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

static void axpy(mp1_t *y, mp1_t a, const mp1_t *x, std::size_t n) {
#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) { y[i] += a * x[i]; }
}

static void axpy(mp2_t *y, mp2_t a, const mp2_t *x, std::size_t n) {
#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) { y[i] += a * x[i]; }
}

static void axpy(mp3_t *y, mp3_t a, const mp3_t *x, std::size_t n) {
#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) { y[i] += a * x[i]; }
}

static void axpy(mp4_t *y, mp4_t a, const mp4_t *x, std::size_t n) {
#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) { y[i] += a * x[i]; }
}

static void axpy_bench_1(benchmark::State &bs) {

    const std::size_t n = static_cast<std::size_t>(bs.range(0));

    mp1_t *const y = static_cast<mp1_t *>(std::malloc(n * sizeof(mp1_t)));

    const mp1_t a = static_cast<mp1_t>(0.5);

    mp1_t *const x = static_cast<mp1_t *>(std::malloc(n * sizeof(mp1_t)));

#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) { x[i] = static_cast<mp1_t>(i); }

    for (auto _ : bs) {
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
            y[i] = static_cast<mp1_t>(2.0) * static_cast<mp1_t>(i);
        }
        const auto start = std::chrono::high_resolution_clock::now();
        axpy(y, a, x, n);
        const auto stop = std::chrono::high_resolution_clock::now();
        bs.SetIterationTime(
            std::chrono::duration<double>(stop - start).count());
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
            assert(y[i] == static_cast<mp1_t>(2.5) * static_cast<mp1_t>(i));
        }
    }

    bs.SetComplexityN(static_cast<benchmark::ComplexityN>(n));
    bs.SetItemsProcessed(static_cast<std::int64_t>(n) * bs.iterations());

    std::free(x);
    std::free(y);
}

static void axpy_bench_2(benchmark::State &bs) {

    const std::size_t n = static_cast<std::size_t>(bs.range(0));

    mp2_t *const y = static_cast<mp2_t *>(std::malloc(n * sizeof(mp2_t)));

    const mp2_t a = static_cast<mp2_t>(0.5);

    mp2_t *const x = static_cast<mp2_t *>(std::malloc(n * sizeof(mp2_t)));

#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) { x[i] = static_cast<mp2_t>(i); }

    for (auto _ : bs) {
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
            y[i] = static_cast<mp2_t>(2.0) * static_cast<mp2_t>(i);
        }
        const auto start = std::chrono::high_resolution_clock::now();
        axpy(y, a, x, n);
        const auto stop = std::chrono::high_resolution_clock::now();
        bs.SetIterationTime(
            std::chrono::duration<double>(stop - start).count());
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
            assert(y[i] == static_cast<mp2_t>(2.5) * static_cast<mp2_t>(i));
        }
    }

    bs.SetComplexityN(static_cast<benchmark::ComplexityN>(n));
    bs.SetItemsProcessed(static_cast<std::int64_t>(n) * bs.iterations());

    std::free(x);
    std::free(y);
}

static void axpy_bench_3(benchmark::State &bs) {

    const std::size_t n = static_cast<std::size_t>(bs.range(0));

    mp3_t *const y = static_cast<mp3_t *>(std::malloc(n * sizeof(mp3_t)));

    const mp3_t a = static_cast<mp3_t>(0.5);

    mp3_t *const x = static_cast<mp3_t *>(std::malloc(n * sizeof(mp3_t)));

#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) { x[i] = static_cast<mp3_t>(i); }

    for (auto _ : bs) {
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
            y[i] = static_cast<mp3_t>(2.0) * static_cast<mp3_t>(i);
        }
        const auto start = std::chrono::high_resolution_clock::now();
        axpy(y, a, x, n);
        const auto stop = std::chrono::high_resolution_clock::now();
        bs.SetIterationTime(
            std::chrono::duration<double>(stop - start).count());
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
            assert(y[i] == static_cast<mp3_t>(2.5) * static_cast<mp3_t>(i));
        }
    }

    bs.SetComplexityN(static_cast<benchmark::ComplexityN>(n));
    bs.SetItemsProcessed(static_cast<std::int64_t>(n) * bs.iterations());

    std::free(x);
    std::free(y);
}

static void axpy_bench_4(benchmark::State &bs) {

    const std::size_t n = static_cast<std::size_t>(bs.range(0));

    mp4_t *const y = static_cast<mp4_t *>(std::malloc(n * sizeof(mp4_t)));

    const mp4_t a = static_cast<mp4_t>(0.5);

    mp4_t *const x = static_cast<mp4_t *>(std::malloc(n * sizeof(mp4_t)));

#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) { x[i] = static_cast<mp4_t>(i); }

    for (auto _ : bs) {
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
            y[i] = static_cast<mp4_t>(2.0) * static_cast<mp4_t>(i);
        }
        const auto start = std::chrono::high_resolution_clock::now();
        axpy(y, a, x, n);
        const auto stop = std::chrono::high_resolution_clock::now();
        bs.SetIterationTime(
            std::chrono::duration<double>(stop - start).count());
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
            assert(y[i] == static_cast<mp4_t>(2.5) * static_cast<mp4_t>(i));
        }
    }

    bs.SetComplexityN(static_cast<benchmark::ComplexityN>(n));
    bs.SetItemsProcessed(static_cast<std::int64_t>(n) * bs.iterations());

    std::free(x);
    std::free(y);
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
