#include <cassert>
#include <chrono>
#include <cln/floatformat.h>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include <benchmark/benchmark.h>
#include <cln/float.h>

static void axpy(cln::cl_F *y, cln::cl_F a, const cln::cl_F *x, std::size_t n) {
#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) { y[i] += a * x[i]; }
}

static cln::cl_F dot(const cln::cl_F *x, const cln::cl_F *y, std::size_t n,
                     cln::float_format_t format) {
    cln::cl_F result = cln::cl_float(0.0, format);
#pragma omp parallel
    {
        cln::cl_F local_sum = cln::cl_float(0.0, format);
#pragma omp for schedule(static)
        for (std::size_t i = 0; i < n; ++i) { local_sum += x[i] * y[i]; }
#pragma omp critical
        { result += local_sum; }
    }
    return result;
}

static void gemv(cln::cl_F *y, const cln::cl_F *A, const cln::cl_F *x,
                 std::size_t n) {
#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) { y[i] += A[i * n + j] * x[j]; }
    }
}

static void gemm(cln::cl_F *C, const cln::cl_F *A, const cln::cl_F *B,
                 std::size_t n) {
#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t k = 0; k < n; ++k) {
            for (std::size_t j = 0; j < n; ++j) {
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
}

static void axpy_bench(benchmark::State &bs, cln::float_format_t format) {

    const std::size_t n = static_cast<std::size_t>(bs.range(0));
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

static void dot_bench(benchmark::State &bs, cln::float_format_t format) {

    const std::size_t n = static_cast<std::size_t>(bs.range(0));
    cln::default_float_format = format;

    std::vector<cln::cl_F> x(n);

#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) { x[i] = cln::cl_float(1.5, format); }

    std::vector<cln::cl_F> y(n);

#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) { y[i] = cln::cl_float(2.5, format); }

    for (auto _ : bs) {
        const auto start = std::chrono::high_resolution_clock::now();
        const cln::cl_F result = dot(x.data(), y.data(), n, format);
        const auto stop = std::chrono::high_resolution_clock::now();
        bs.SetIterationTime(
            std::chrono::duration<double>(stop - start).count());
        assert(result == 3.75 * static_cast<double>(n));
    }

    bs.SetComplexityN(static_cast<benchmark::ComplexityN>(n));
    bs.SetItemsProcessed(static_cast<std::int64_t>(n) * bs.iterations());
}

static void gemv_bench(benchmark::State &bs, cln::float_format_t format) {

    const std::size_t n = static_cast<std::size_t>(bs.range(0));
    cln::default_float_format = format;

    std::vector<cln::cl_F> y(n);

    std::vector<cln::cl_F> A(n * n);

#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n * n; ++i) {
        A[i] = cln::cl_float(1.5, format);
    }

    std::vector<cln::cl_F> x(n);

#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) { x[i] = cln::cl_float(2.5, format); }

    for (auto _ : bs) {
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
            y[i] = cln::cl_float(0.0, format);
        }
        const auto start = std::chrono::high_resolution_clock::now();
        gemv(y.data(), A.data(), x.data(), n);
        const auto stop = std::chrono::high_resolution_clock::now();
        bs.SetIterationTime(
            std::chrono::duration<double>(stop - start).count());
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
            assert(y[i] == 3.75 * static_cast<double>(n));
        }
    }

    bs.SetComplexityN(static_cast<benchmark::ComplexityN>(n));
    bs.SetItemsProcessed(static_cast<std::int64_t>(n * n) * bs.iterations());
}

static void gemm_bench(benchmark::State &bs, cln::float_format_t format) {

    const std::size_t n = static_cast<std::size_t>(bs.range(0));
    cln::default_float_format = format;

    std::vector<cln::cl_F> C(n * n);

    std::vector<cln::cl_F> A(n * n);

#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n * n; ++i) {
        A[i] = cln::cl_float(1.5, format);
    }

    std::vector<cln::cl_F> B(n * n);

#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n * n; ++i) {
        B[i] = cln::cl_float(2.5, format);
    }

    for (auto _ : bs) {
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n * n; ++i) {
            C[i] = cln::cl_float(0.0, format);
        }
        const auto start = std::chrono::high_resolution_clock::now();
        gemm(C.data(), A.data(), B.data(), n);
        const auto stop = std::chrono::high_resolution_clock::now();
        bs.SetIterationTime(
            std::chrono::duration<double>(stop - start).count());
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n * n; ++i) {
            assert(C[i] == 3.75 * static_cast<double>(n));
        }
    }

    bs.SetComplexityN(static_cast<benchmark::ComplexityN>(n));
    bs.SetItemsProcessed(static_cast<std::int64_t>(n * n * n) *
                         bs.iterations());
}

static void axpy_bench_1(benchmark::State &bs) {
    axpy_bench(bs, cln::float_format(14));
}
static void axpy_bench_2(benchmark::State &bs) {
    axpy_bench(bs, cln::float_format(30));
}
static void axpy_bench_3(benchmark::State &bs) {
    axpy_bench(bs, cln::float_format(45));
}
static void axpy_bench_4(benchmark::State &bs) {
    axpy_bench(bs, cln::float_format(61));
}

static void dot_bench_1(benchmark::State &bs) {
    dot_bench(bs, cln::float_format(14));
}
static void dot_bench_2(benchmark::State &bs) {
    dot_bench(bs, cln::float_format(30));
}
static void dot_bench_3(benchmark::State &bs) {
    dot_bench(bs, cln::float_format(45));
}
static void dot_bench_4(benchmark::State &bs) {
    dot_bench(bs, cln::float_format(61));
}

static void gemv_bench_1(benchmark::State &bs) {
    gemv_bench(bs, cln::float_format(14));
}
static void gemv_bench_2(benchmark::State &bs) {
    gemv_bench(bs, cln::float_format(30));
}
static void gemv_bench_3(benchmark::State &bs) {
    gemv_bench(bs, cln::float_format(45));
}
static void gemv_bench_4(benchmark::State &bs) {
    gemv_bench(bs, cln::float_format(61));
}

static void gemm_bench_1(benchmark::State &bs) {
    gemm_bench(bs, cln::float_format(14));
}
static void gemm_bench_2(benchmark::State &bs) {
    gemm_bench(bs, cln::float_format(30));
}
static void gemm_bench_3(benchmark::State &bs) {
    gemm_bench(bs, cln::float_format(45));
}
static void gemm_bench_4(benchmark::State &bs) {
    gemm_bench(bs, cln::float_format(61));
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

// BENCHMARK(dot_bench_1)
//     ->UseManualTime()
//     ->Complexity(benchmark::oN)
//     ->Repetitions(3)
//     ->RangeMultiplier(2)
//     ->Range(1L << 8, 1L << 24)
//     ->DisplayAggregatesOnly();

// BENCHMARK(dot_bench_2)
//     ->UseManualTime()
//     ->Complexity(benchmark::oN)
//     ->Repetitions(3)
//     ->RangeMultiplier(2)
//     ->Range(1L << 8, 1L << 24)
//     ->DisplayAggregatesOnly();

// BENCHMARK(dot_bench_3)
//     ->UseManualTime()
//     ->Complexity(benchmark::oN)
//     ->Repetitions(3)
//     ->RangeMultiplier(2)
//     ->Range(1L << 8, 1L << 24)
//     ->DisplayAggregatesOnly();

// BENCHMARK(dot_bench_4)
//     ->UseManualTime()
//     ->Complexity(benchmark::oN)
//     ->Repetitions(3)
//     ->RangeMultiplier(2)
//     ->Range(1L << 8, 1L << 24)
//     ->DisplayAggregatesOnly();

// BENCHMARK(gemv_bench_1)
//     ->UseManualTime()
//     ->Complexity(benchmark::oNSquared)
//     ->Repetitions(3)
//     ->RangeMultiplier(2)
//     ->Range(1L << 4, 1L << 12)
//     ->DisplayAggregatesOnly();

// BENCHMARK(gemv_bench_2)
//     ->UseManualTime()
//     ->Complexity(benchmark::oNSquared)
//     ->Repetitions(3)
//     ->RangeMultiplier(2)
//     ->Range(1L << 4, 1L << 12)
//     ->DisplayAggregatesOnly();

// BENCHMARK(gemv_bench_3)
//     ->UseManualTime()
//     ->Complexity(benchmark::oNSquared)
//     ->Repetitions(3)
//     ->RangeMultiplier(2)
//     ->Range(1L << 4, 1L << 12)
//     ->DisplayAggregatesOnly();

// BENCHMARK(gemv_bench_4)
//     ->UseManualTime()
//     ->Complexity(benchmark::oNSquared)
//     ->Repetitions(3)
//     ->RangeMultiplier(2)
//     ->Range(1L << 4, 1L << 12)
//     ->DisplayAggregatesOnly();

// BENCHMARK(gemm_bench_1)
//     ->UseManualTime()
//     ->Complexity(benchmark::oNCubed)
//     ->Repetitions(3)
//     ->RangeMultiplier(2)
//     ->Range(1L << 3, 1L << 9)
//     ->DisplayAggregatesOnly();

// BENCHMARK(gemm_bench_2)
//     ->UseManualTime()
//     ->Complexity(benchmark::oNCubed)
//     ->Repetitions(3)
//     ->RangeMultiplier(2)
//     ->Range(1L << 3, 1L << 9)
//     ->DisplayAggregatesOnly();

// BENCHMARK(gemm_bench_3)
//     ->UseManualTime()
//     ->Complexity(benchmark::oNCubed)
//     ->Repetitions(3)
//     ->RangeMultiplier(2)
//     ->Range(1L << 3, 1L << 9)
//     ->DisplayAggregatesOnly();

// BENCHMARK(gemm_bench_4)
//     ->UseManualTime()
//     ->Complexity(benchmark::oNCubed)
//     ->Repetitions(3)
//     ->RangeMultiplier(2)
//     ->Range(1L << 3, 1L << 9)
//     ->DisplayAggregatesOnly();
