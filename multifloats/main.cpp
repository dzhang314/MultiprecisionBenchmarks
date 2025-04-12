#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <utility>

#include <benchmark/benchmark.h>
#include <omp.h>

#include "MultiFloats.hpp"

constexpr std::pair<std::size_t, std::size_t>
split_range(std::size_t n, int num_threads, int thread_id) {
    const std::size_t m = static_cast<std::size_t>(num_threads);
    const std::size_t i = static_cast<std::size_t>(thread_id);
    const std::size_t q = n / m;
    const std::size_t r = n % m;
    const std::size_t chunk_size = (i < r) ? (q + 1) : q;
    const std::size_t first = (i < r) ? (q * i + i) : (q * i + r);
    const std::size_t last = first + chunk_size;
    return {first, last};
}

static void axpy(double *y0, f64x1 a, const double *x0, std::size_t n) {
#pragma omp parallel
    {
        const std::pair<std::size_t, std::size_t> range =
            split_range(n, omp_get_num_threads(), omp_get_thread_num());
#pragma omp simd simdlen(8)
        for (std::size_t i = range.first; i < range.second; ++i) {
            const f64x1 y(y0[i]);
            const f64x1 x(x0[i]);
            const f64x1 z = y + a * x;
            y0[i] = z._limbs[0];
        }
    }
}

static void axpy(double *y0, double *y1, f64x2 a, const double *x0,
                 const double *x1, std::size_t n) {
#pragma omp parallel
    {
        const std::pair<std::size_t, std::size_t> range =
            split_range(n, omp_get_num_threads(), omp_get_thread_num());
#pragma omp simd simdlen(8)
        for (std::size_t i = range.first; i < range.second; ++i) {
            const f64x2 y(y0[i], y1[i]);
            const f64x2 x(x0[i], x1[i]);
            const f64x2 z = y + a * x;
            y0[i] = z._limbs[0];
            y1[i] = z._limbs[1];
        }
    }
}

static void axpy(double *y0, double *y1, double *y2, f64x3 a, const double *x0,
                 const double *x1, const double *x2, std::size_t n) {
#pragma omp parallel
    {
        const std::pair<std::size_t, std::size_t> range =
            split_range(n, omp_get_num_threads(), omp_get_thread_num());
#pragma omp simd simdlen(8)
        for (std::size_t i = range.first; i < range.second; ++i) {
            const f64x3 y(y0[i], y1[i], y2[i]);
            const f64x3 x(x0[i], x1[i], x2[i]);
            const f64x3 z = y + a * x;
            y0[i] = z._limbs[0];
            y1[i] = z._limbs[1];
            y2[i] = z._limbs[2];
        }
    }
}

static void axpy(double *y0, double *y1, double *y2, double *y3, f64x4 a,
                 const double *x0, const double *x1, const double *x2,
                 const double *x3, std::size_t n) {
#pragma omp parallel
    {
        const std::pair<std::size_t, std::size_t> range =
            split_range(n, omp_get_num_threads(), omp_get_thread_num());
#pragma omp simd simdlen(8)
        for (std::size_t i = range.first; i < range.second; ++i) {
            const f64x4 y(y0[i], y1[i], y2[i], y3[i]);
            const f64x4 x(x0[i], x1[i], x2[i], x3[i]);
            const f64x4 z = y + a * x;
            y0[i] = z._limbs[0];
            y1[i] = z._limbs[1];
            y2[i] = z._limbs[2];
            y3[i] = z._limbs[3];
        }
    }
}

static f64x1 dot(const double *x0, const double *y0, std::size_t n) {
    f64x1 result = 0.0;
#pragma omp parallel
    {
        f64x1 local_sum = 0.0;
        const std::pair<std::size_t, std::size_t> range =
            split_range(n, omp_get_num_threads(), omp_get_thread_num());
#pragma omp simd simdlen(8) reduction(+ : local_sum)
        for (std::size_t i = range.first; i < range.second; ++i) {
            const f64x1 x(x0[i]);
            const f64x1 y(y0[i]);
            local_sum += x * y;
        }
#pragma omp critical
        { result += local_sum; }
    }
    return result;
}

static f64x2 dot(const double *x0, const double *x1, const double *y0,
                 const double *y1, std::size_t n) {
    f64x2 result = 0.0;
#pragma omp parallel
    {
        f64x2 local_sum = 0.0;
        const std::pair<std::size_t, std::size_t> range =
            split_range(n, omp_get_num_threads(), omp_get_thread_num());
#pragma omp simd simdlen(8) reduction(+ : local_sum)
        for (std::size_t i = range.first; i < range.second; ++i) {
            const f64x2 x(x0[i], x1[i]);
            const f64x2 y(y0[i], y1[i]);
            local_sum += x * y;
        }
#pragma omp critical
        { result += local_sum; }
    }
    return result;
}

static f64x3 dot(const double *x0, const double *x1, const double *x2,
                 const double *y0, const double *y1, const double *y2,
                 std::size_t n) {
    f64x3 result = 0.0;
#pragma omp parallel
    {
        f64x3 local_sum = 0.0;
        const std::pair<std::size_t, std::size_t> range =
            split_range(n, omp_get_num_threads(), omp_get_thread_num());
#pragma omp simd simdlen(8) reduction(+ : local_sum)
        for (std::size_t i = range.first; i < range.second; ++i) {
            const f64x3 x(x0[i], x1[i], x2[i]);
            const f64x3 y(y0[i], y1[i], y2[i]);
            local_sum += x * y;
        }
#pragma omp critical
        { result += local_sum; }
    }
    return result;
}

static f64x4 dot(const double *x0, const double *x1, const double *x2,
                 const double *x3, const double *y0, const double *y1,
                 const double *y2, const double *y3, std::size_t n) {
    f64x4 result = 0.0;
#pragma omp parallel
    {
        f64x4 local_sum = 0.0;
        const std::pair<std::size_t, std::size_t> range =
            split_range(n, omp_get_num_threads(), omp_get_thread_num());
#pragma omp simd simdlen(8) reduction(+ : local_sum)
        for (std::size_t i = range.first; i < range.second; ++i) {
            const f64x4 x(x0[i], x1[i], x2[i], x3[i]);
            const f64x4 y(y0[i], y1[i], y2[i], y3[i]);
            local_sum += x * y;
        }
#pragma omp critical
        { result += local_sum; }
    }
    return result;
}

static void axpy_bench_1(benchmark::State &bs) {

    const std::size_t n = static_cast<std::size_t>(bs.range(0));

    double *const y0 = static_cast<double *>(std::malloc(n * sizeof(double)));

    const f64x1 a = static_cast<f64x1>(0.5);

    double *const x0 = static_cast<double *>(std::malloc(n * sizeof(double)));

#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) { x0[i] = static_cast<double>(i); }

    for (auto _ : bs) {
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
            y0[i] = 2.0 * static_cast<double>(i);
        }
        const auto start = std::chrono::high_resolution_clock::now();
        axpy(y0, a, x0, n);
        const auto stop = std::chrono::high_resolution_clock::now();
        bs.SetIterationTime(
            std::chrono::duration<double>(stop - start).count());
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
            assert(y0[i] == 2.5 * static_cast<double>(i));
        }
    }

    bs.SetComplexityN(static_cast<benchmark::ComplexityN>(n));
    bs.SetItemsProcessed(static_cast<std::int64_t>(n) * bs.iterations());

    std::free(x0);
    std::free(y0);
}

static void axpy_bench_2(benchmark::State &bs) {

    const std::size_t n = static_cast<std::size_t>(bs.range(0));

    double *const y0 = static_cast<double *>(std::malloc(n * sizeof(double)));
    double *const y1 = static_cast<double *>(std::malloc(n * sizeof(double)));

    const f64x2 a = static_cast<f64x2>(0.5);

    double *const x0 = static_cast<double *>(std::malloc(n * sizeof(double)));
    double *const x1 = static_cast<double *>(std::malloc(n * sizeof(double)));

#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) {
        x0[i] = static_cast<double>(i);
        x1[i] = 0.0;
    }

    for (auto _ : bs) {
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
            y0[i] = 2.0 * static_cast<double>(i);
            y1[i] = 0.0;
        }
        const auto start = std::chrono::high_resolution_clock::now();
        axpy(y0, y1, a, x0, x1, n);
        const auto stop = std::chrono::high_resolution_clock::now();
        bs.SetIterationTime(
            std::chrono::duration<double>(stop - start).count());
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
            assert(y0[i] == 2.5 * static_cast<double>(i));
            assert(y1[i] == 0.0);
        }
    }

    bs.SetComplexityN(static_cast<benchmark::ComplexityN>(n));
    bs.SetItemsProcessed(static_cast<std::int64_t>(n) * bs.iterations());

    std::free(x1);
    std::free(x0);
    std::free(y1);
    std::free(y0);
}

static void axpy_bench_3(benchmark::State &bs) {

    const std::size_t n = static_cast<std::size_t>(bs.range(0));

    double *const y0 = static_cast<double *>(std::malloc(n * sizeof(double)));
    double *const y1 = static_cast<double *>(std::malloc(n * sizeof(double)));
    double *const y2 = static_cast<double *>(std::malloc(n * sizeof(double)));

    const f64x3 a = static_cast<f64x3>(0.5);

    double *const x0 = static_cast<double *>(std::malloc(n * sizeof(double)));
    double *const x1 = static_cast<double *>(std::malloc(n * sizeof(double)));
    double *const x2 = static_cast<double *>(std::malloc(n * sizeof(double)));

#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) {
        x0[i] = static_cast<double>(i);
        x1[i] = 0.0;
        x2[i] = 0.0;
    }

    for (auto _ : bs) {
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
            y0[i] = 2.0 * static_cast<double>(i);
            y1[i] = 0.0;
            y2[i] = 0.0;
        }
        const auto start = std::chrono::high_resolution_clock::now();
        axpy(y0, y1, y2, a, x0, x1, x2, n);
        const auto stop = std::chrono::high_resolution_clock::now();
        bs.SetIterationTime(
            std::chrono::duration<double>(stop - start).count());
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
            assert(y0[i] == 2.5 * static_cast<double>(i));
            assert(y1[i] == 0.0);
            assert(y2[i] == 0.0);
        }
    }

    bs.SetComplexityN(static_cast<benchmark::ComplexityN>(n));
    bs.SetItemsProcessed(static_cast<std::int64_t>(n) * bs.iterations());

    std::free(x2);
    std::free(x1);
    std::free(x0);
    std::free(y2);
    std::free(y1);
    std::free(y0);
}

static void axpy_bench_4(benchmark::State &bs) {

    const std::size_t n = static_cast<std::size_t>(bs.range(0));

    double *const y0 = static_cast<double *>(std::malloc(n * sizeof(double)));
    double *const y1 = static_cast<double *>(std::malloc(n * sizeof(double)));
    double *const y2 = static_cast<double *>(std::malloc(n * sizeof(double)));
    double *const y3 = static_cast<double *>(std::malloc(n * sizeof(double)));

    const f64x4 a = static_cast<f64x4>(0.5);

    double *const x0 = static_cast<double *>(std::malloc(n * sizeof(double)));
    double *const x1 = static_cast<double *>(std::malloc(n * sizeof(double)));
    double *const x2 = static_cast<double *>(std::malloc(n * sizeof(double)));
    double *const x3 = static_cast<double *>(std::malloc(n * sizeof(double)));

#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) {
        x0[i] = static_cast<double>(i);
        x1[i] = 0.0;
        x2[i] = 0.0;
        x3[i] = 0.0;
    }

    for (auto _ : bs) {
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
            y0[i] = 2.0 * static_cast<double>(i);
            y1[i] = 0.0;
            y2[i] = 0.0;
            y3[i] = 0.0;
        }
        const auto start = std::chrono::high_resolution_clock::now();
        axpy(y0, y1, y2, y3, a, x0, x1, x2, x3, n);
        const auto stop = std::chrono::high_resolution_clock::now();
        bs.SetIterationTime(
            std::chrono::duration<double>(stop - start).count());
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
            assert(y0[i] == 2.5 * static_cast<double>(i));
            assert(y1[i] == 0.0);
            assert(y2[i] == 0.0);
            assert(y3[i] == 0.0);
        }
    }

    bs.SetComplexityN(static_cast<benchmark::ComplexityN>(n));
    bs.SetItemsProcessed(static_cast<std::int64_t>(n) * bs.iterations());

    std::free(x3);
    std::free(x2);
    std::free(x1);
    std::free(x0);
    std::free(y3);
    std::free(y2);
    std::free(y1);
    std::free(y0);
}

static void dot_bench_1(benchmark::State &bs) {

    const std::size_t n = static_cast<std::size_t>(bs.range(0));

    double *const x0 = static_cast<double *>(std::malloc(n * sizeof(double)));

#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) { x0[i] = 1.5; }

    double *const y0 = static_cast<double *>(std::malloc(n * sizeof(double)));

#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) { y0[i] = 2.5; }

    for (auto _ : bs) {
        const auto start = std::chrono::high_resolution_clock::now();
        const f64x1 result = dot(x0, y0, n);
        const auto stop = std::chrono::high_resolution_clock::now();
        bs.SetIterationTime(
            std::chrono::duration<double>(stop - start).count());
        assert(result == 3.75 * static_cast<double>(n));
    }

    bs.SetComplexityN(static_cast<benchmark::ComplexityN>(n));
    bs.SetItemsProcessed(static_cast<std::int64_t>(n) * bs.iterations());

    std::free(y0);
    std::free(x0);
}

static void dot_bench_2(benchmark::State &bs) {

    const std::size_t n = static_cast<std::size_t>(bs.range(0));

    double *const x0 = static_cast<double *>(std::malloc(n * sizeof(double)));
    double *const x1 = static_cast<double *>(std::malloc(n * sizeof(double)));

#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) {
        x0[i] = 1.5;
        x1[i] = 0.0;
    }

    double *const y0 = static_cast<double *>(std::malloc(n * sizeof(double)));
    double *const y1 = static_cast<double *>(std::malloc(n * sizeof(double)));

#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) {
        y0[i] = 2.5;
        y1[i] = 0.0;
    }

    for (auto _ : bs) {
        const auto start = std::chrono::high_resolution_clock::now();
        const f64x2 result = dot(x0, x1, y0, y1, n);
        const auto stop = std::chrono::high_resolution_clock::now();
        bs.SetIterationTime(
            std::chrono::duration<double>(stop - start).count());
        assert(result == 3.75 * static_cast<double>(n));
    }

    bs.SetComplexityN(static_cast<benchmark::ComplexityN>(n));
    bs.SetItemsProcessed(static_cast<std::int64_t>(n) * bs.iterations());

    std::free(y1);
    std::free(y0);
    std::free(x1);
    std::free(x0);
}

static void dot_bench_3(benchmark::State &bs) {

    const std::size_t n = static_cast<std::size_t>(bs.range(0));

    double *const x0 = static_cast<double *>(std::malloc(n * sizeof(double)));
    double *const x1 = static_cast<double *>(std::malloc(n * sizeof(double)));
    double *const x2 = static_cast<double *>(std::malloc(n * sizeof(double)));

#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) {
        x0[i] = 1.5;
        x1[i] = 0.0;
        x2[i] = 0.0;
    }

    double *const y0 = static_cast<double *>(std::malloc(n * sizeof(double)));
    double *const y1 = static_cast<double *>(std::malloc(n * sizeof(double)));
    double *const y2 = static_cast<double *>(std::malloc(n * sizeof(double)));

#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) {
        y0[i] = 2.5;
        y1[i] = 0.0;
        y2[i] = 0.0;
    }

    for (auto _ : bs) {
        const auto start = std::chrono::high_resolution_clock::now();
        const f64x3 result = dot(x0, x1, x2, y0, y1, y2, n);
        const auto stop = std::chrono::high_resolution_clock::now();
        bs.SetIterationTime(
            std::chrono::duration<double>(stop - start).count());
        assert(result == 3.75 * static_cast<double>(n));
    }

    bs.SetComplexityN(static_cast<benchmark::ComplexityN>(n));
    bs.SetItemsProcessed(static_cast<std::int64_t>(n) * bs.iterations());

    std::free(y2);
    std::free(y1);
    std::free(y0);
    std::free(x2);
    std::free(x1);
    std::free(x0);
}

static void dot_bench_4(benchmark::State &bs) {

    const std::size_t n = static_cast<std::size_t>(bs.range(0));

    double *const x0 = static_cast<double *>(std::malloc(n * sizeof(double)));
    double *const x1 = static_cast<double *>(std::malloc(n * sizeof(double)));
    double *const x2 = static_cast<double *>(std::malloc(n * sizeof(double)));
    double *const x3 = static_cast<double *>(std::malloc(n * sizeof(double)));

#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) {
        x0[i] = 1.5;
        x1[i] = 0.0;
        x2[i] = 0.0;
        x3[i] = 0.0;
    }

    double *const y0 = static_cast<double *>(std::malloc(n * sizeof(double)));
    double *const y1 = static_cast<double *>(std::malloc(n * sizeof(double)));
    double *const y2 = static_cast<double *>(std::malloc(n * sizeof(double)));
    double *const y3 = static_cast<double *>(std::malloc(n * sizeof(double)));

#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) {
        y0[i] = 2.5;
        y1[i] = 0.0;
        y2[i] = 0.0;
        y3[i] = 0.0;
    }

    for (auto _ : bs) {
        const auto start = std::chrono::high_resolution_clock::now();
        const f64x4 result = dot(x0, x1, x2, x3, y0, y1, y2, y3, n);
        const auto stop = std::chrono::high_resolution_clock::now();
        bs.SetIterationTime(
            std::chrono::duration<double>(stop - start).count());
        assert(result == 3.75 * static_cast<double>(n));
    }

    bs.SetComplexityN(static_cast<benchmark::ComplexityN>(n));
    bs.SetItemsProcessed(static_cast<std::int64_t>(n) * bs.iterations());

    std::free(y3);
    std::free(y2);
    std::free(y1);
    std::free(y0);
    std::free(x3);
    std::free(x2);
    std::free(x1);
    std::free(x0);
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
