#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include <benchmark/benchmark.h>
#include <cln/float.h>

static void axpy(cln::cl_F *y, cln::cl_F a, const cln::cl_F *x, std::size_t n) {
#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) { y[i] += a * x[i]; }
}

static void axpy_bench_1(benchmark::State &bs) {

    const std::size_t n = static_cast<std::size_t>(bs.range(0));
    const cln::float_format_t format = cln::float_format(15);
    cln::default_float_format = format;

    std::vector<cln::cl_F> y(n);
    const cln::cl_F a = cln::cl_float(0.5, format);
    std::vector<cln::cl_F> x(n);

#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) {
        x[i] = cln::cl_float(static_cast<double>(i), format);
    }

    for (auto _ : bs) {
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
            y[i] = cln::cl_float(2.0 * static_cast<double>(i), format);
        }
        const auto start = std::chrono::high_resolution_clock::now();
        axpy(y.data(), a, x.data(), n);
        const auto stop = std::chrono::high_resolution_clock::now();
        bs.SetIterationTime(
            std::chrono::duration<double>(stop - start).count());
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
            assert(y[i] == 2.5 * static_cast<double>(i));
        }
    }

    bs.SetComplexityN(static_cast<benchmark::ComplexityN>(n));
    bs.SetItemsProcessed(static_cast<std::int64_t>(n) * bs.iterations());
}

static void axpy_bench_2(benchmark::State &bs) {

    const std::size_t n = static_cast<std::size_t>(bs.range(0));
    const cln::float_format_t format = cln::float_format(31);
    cln::default_float_format = format;

    std::vector<cln::cl_F> y(n);
    const cln::cl_F a = cln::cl_float(0.5, format);
    std::vector<cln::cl_F> x(n);

#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) {
        x[i] = cln::cl_float(static_cast<double>(i), format);
    }

    for (auto _ : bs) {
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
            y[i] = cln::cl_float(2.0 * static_cast<double>(i), format);
        }
        const auto start = std::chrono::high_resolution_clock::now();
        axpy(y.data(), a, x.data(), n);
        const auto stop = std::chrono::high_resolution_clock::now();
        bs.SetIterationTime(
            std::chrono::duration<double>(stop - start).count());
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
            assert(y[i] == 2.5 * static_cast<double>(i));
        }
    }

    bs.SetComplexityN(static_cast<benchmark::ComplexityN>(n));
    bs.SetItemsProcessed(static_cast<std::int64_t>(n) * bs.iterations());
}

static void axpy_bench_3(benchmark::State &bs) {

    const std::size_t n = static_cast<std::size_t>(bs.range(0));
    const cln::float_format_t format = cln::float_format(46);
    cln::default_float_format = format;

    std::vector<cln::cl_F> y(n);
    const cln::cl_F a = cln::cl_float(0.5, format);
    std::vector<cln::cl_F> x(n);

#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) {
        x[i] = cln::cl_float(static_cast<double>(i), format);
    }

    for (auto _ : bs) {
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
            y[i] = cln::cl_float(2.0 * static_cast<double>(i), format);
        }
        const auto start = std::chrono::high_resolution_clock::now();
        axpy(y.data(), a, x.data(), n);
        const auto stop = std::chrono::high_resolution_clock::now();
        bs.SetIterationTime(
            std::chrono::duration<double>(stop - start).count());
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
            assert(y[i] == 2.5 * static_cast<double>(i));
        }
    }

    bs.SetComplexityN(static_cast<benchmark::ComplexityN>(n));
    bs.SetItemsProcessed(static_cast<std::int64_t>(n) * bs.iterations());
}

static void axpy_bench_4(benchmark::State &bs) {

    const std::size_t n = static_cast<std::size_t>(bs.range(0));
    const cln::float_format_t format = cln::float_format(62);
    cln::default_float_format = format;

    std::vector<cln::cl_F> y(n);
    const cln::cl_F a = cln::cl_float(0.5, format);
    std::vector<cln::cl_F> x(n);

#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) {
        x[i] = cln::cl_float(static_cast<double>(i), format);
    }

    for (auto _ : bs) {
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
            y[i] = cln::cl_float(2.0 * static_cast<double>(i), format);
        }
        const auto start = std::chrono::high_resolution_clock::now();
        axpy(y.data(), a, x.data(), n);
        const auto stop = std::chrono::high_resolution_clock::now();
        bs.SetIterationTime(
            std::chrono::duration<double>(stop - start).count());
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
            assert(y[i] == 2.5 * static_cast<double>(i));
        }
    }

    bs.SetComplexityN(static_cast<benchmark::ComplexityN>(n));
    bs.SetItemsProcessed(static_cast<std::int64_t>(n) * bs.iterations());
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
