#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdlib>

#include <benchmark/benchmark.h>
#include <flint/arf.h>

template <slong PRECISION>
static void axpy(arf_t *y, const arf_t a, const arf_t *x, std::size_t n) {
#pragma omp parallel
    {
        arf_t temp;
        arf_init(temp);
#pragma omp for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
            arf_mul(temp, a, x[i], PRECISION, ARF_RND_NEAR);
            arf_add(y[i], y[i], temp, PRECISION, ARF_RND_NEAR);
        }
        arf_clear(temp);
    }
}

template <slong PRECISION>
static void dot(arf_t result, const arf_t *x, const arf_t *y, std::size_t n) {
    arf_zero(result);
#pragma omp parallel
    {
        arf_t local_sum;
        arf_init(local_sum);
        arf_zero(local_sum);
        arf_t temp;
        arf_init(temp);
#pragma omp for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
            arf_mul(temp, x[i], y[i], PRECISION, ARF_RND_NEAR);
            arf_add(local_sum, local_sum, temp, PRECISION, ARF_RND_NEAR);
        }
#pragma omp critical
        { arf_add(result, result, local_sum, PRECISION, ARF_RND_NEAR); }
        arf_clear(temp);
        arf_clear(local_sum);
    }
}

template <slong PRECISION>
static void gemv(arf_t *y, const arf_t *A, const arf_t *x, std::size_t n) {
#pragma omp parallel
    {
        arf_t temp;
        arf_init(temp);
#pragma omp for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                arf_mul(temp, A[i * n + j], x[j], PRECISION, ARF_RND_NEAR);
                arf_add(y[i], y[i], temp, PRECISION, ARF_RND_NEAR);
            }
        }
        arf_clear(temp);
    }
}

template <slong PRECISION>
static void gemm(arf_t *C, const arf_t *A, const arf_t *B, std::size_t n) {
#pragma omp parallel
    {
        arf_t temp;
        arf_init(temp);
#pragma omp for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t k = 0; k < n; ++k) {
                for (std::size_t j = 0; j < n; ++j) {
                    arf_mul(temp, A[i * n + k], B[k * n + j], PRECISION,
                            ARF_RND_NEAR);
                    arf_add(C[i * n + j], C[i * n + j], temp, PRECISION,
                            ARF_RND_NEAR);
                }
            }
        }
        arf_clear(temp);
    }
}

template <slong PRECISION>
static void axpy_bench(benchmark::State &bs) {

    const std::size_t n = static_cast<std::size_t>(bs.range(0));

    arf_t *const y = static_cast<arf_t *>(std::malloc(n * sizeof(arf_t)));
    for (std::size_t i = 0; i < n; ++i) { arf_init(y[i]); }

    arf_t a;
    arf_init(a);
    arf_set_d(a, 0.5);

    arf_t *const x = static_cast<arf_t *>(std::malloc(n * sizeof(arf_t)));
    for (std::size_t i = 0; i < n; ++i) { arf_init(x[i]); }

#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) {
        arf_set_d(x[i], static_cast<double>(i));
    }

    for (auto _ : bs) {
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
            arf_set_d(y[i], 2.0 * static_cast<double>(i));
        }
        const auto start = std::chrono::high_resolution_clock::now();
        axpy<PRECISION>(y, a, x, n);
        const auto stop = std::chrono::high_resolution_clock::now();
        bs.SetIterationTime(
            std::chrono::duration<double>(stop - start).count());
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
            assert(arf_cmp_d(y[i], 2.5 * static_cast<double>(i)) == 0);
        }
    }

    bs.SetComplexityN(static_cast<benchmark::ComplexityN>(n));
    bs.SetItemsProcessed(static_cast<std::int64_t>(n) * bs.iterations());

    for (std::size_t i = 0; i < n; ++i) { arf_clear(x[i]); }
    std::free(x);

    arf_clear(a);

    for (std::size_t i = 0; i < n; ++i) { arf_clear(y[i]); }
    std::free(y);
}

template <slong PRECISION>
static void dot_bench(benchmark::State &bs) {

    const std::size_t n = static_cast<std::size_t>(bs.range(0));

    arf_t *const x = static_cast<arf_t *>(std::malloc(n * sizeof(arf_t)));
    for (std::size_t i = 0; i < n; ++i) { arf_init(x[i]); }

#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) { arf_set_d(x[i], 1.5); }

    arf_t *const y = static_cast<arf_t *>(std::malloc(n * sizeof(arf_t)));
    for (std::size_t i = 0; i < n; ++i) { arf_init(y[i]); }

#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) { arf_set_d(y[i], 2.5); }

    arf_t result;
    arf_init(result);

    for (auto _ : bs) {
        arf_zero(result);
        const auto start = std::chrono::high_resolution_clock::now();
        dot<PRECISION>(result, x, y, n);
        const auto stop = std::chrono::high_resolution_clock::now();
        bs.SetIterationTime(
            std::chrono::duration<double>(stop - start).count());
        assert(arf_cmp_d(result, 3.75 * static_cast<double>(n)) == 0);
    }

    bs.SetComplexityN(static_cast<benchmark::ComplexityN>(n));
    bs.SetItemsProcessed(static_cast<std::int64_t>(n) * bs.iterations());

    arf_clear(result);

    for (std::size_t i = 0; i < n; ++i) { arf_clear(y[i]); }
    std::free(y);

    for (std::size_t i = 0; i < n; ++i) { arf_clear(x[i]); }
    std::free(x);
}

template <slong PRECISION>
static void gemv_bench(benchmark::State &bs) {

    const std::size_t n = static_cast<std::size_t>(bs.range(0));

    arf_t *const y = static_cast<arf_t *>(std::malloc(n * sizeof(arf_t)));
    for (std::size_t i = 0; i < n; ++i) { arf_init(y[i]); }

    arf_t *const A = static_cast<arf_t *>(std::malloc(n * n * sizeof(arf_t)));
    for (std::size_t i = 0; i < n * n; ++i) { arf_init(A[i]); }

#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n * n; ++i) { arf_set_d(A[i], 1.5); }

    arf_t *const x = static_cast<arf_t *>(std::malloc(n * sizeof(arf_t)));
    for (std::size_t i = 0; i < n; ++i) { arf_init(x[i]); }

#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) { arf_set_d(x[i], 2.5); }

    for (auto _ : bs) {
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n; ++i) { arf_zero(y[i]); }
        const auto start = std::chrono::high_resolution_clock::now();
        gemv<PRECISION>(y, A, x, n);
        const auto stop = std::chrono::high_resolution_clock::now();
        bs.SetIterationTime(
            std::chrono::duration<double>(stop - start).count());
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
            assert(arf_cmp_d(y[i], 3.75 * static_cast<double>(n)) == 0);
        }
    }

    bs.SetComplexityN(static_cast<benchmark::ComplexityN>(n));
    bs.SetItemsProcessed(static_cast<std::int64_t>(n * n) * bs.iterations());

    for (std::size_t i = 0; i < n; ++i) { arf_clear(x[i]); }
    std::free(x);

    for (std::size_t i = 0; i < n * n; ++i) { arf_clear(A[i]); }
    std::free(A);

    for (std::size_t i = 0; i < n; ++i) { arf_clear(y[i]); }
    std::free(y);
}

template <slong PRECISION>
static void gemm_bench(benchmark::State &bs) {

    const std::size_t n = static_cast<std::size_t>(bs.range(0));

    arf_t *const C = static_cast<arf_t *>(std::malloc(n * n * sizeof(arf_t)));
    for (std::size_t i = 0; i < n * n; ++i) { arf_init(C[i]); }

    arf_t *const A = static_cast<arf_t *>(std::malloc(n * n * sizeof(arf_t)));
    for (std::size_t i = 0; i < n * n; ++i) { arf_init(A[i]); }

#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n * n; ++i) { arf_set_d(A[i], 1.5); }

    arf_t *const B = static_cast<arf_t *>(std::malloc(n * n * sizeof(arf_t)));
    for (std::size_t i = 0; i < n * n; ++i) { arf_init(B[i]); }

#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n * n; ++i) { arf_set_d(B[i], 2.5); }

    for (auto _ : bs) {
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n * n; ++i) { arf_zero(C[i]); }
        const auto start = std::chrono::high_resolution_clock::now();
        gemm<PRECISION>(C, A, B, n);
        const auto stop = std::chrono::high_resolution_clock::now();
        bs.SetIterationTime(
            std::chrono::duration<double>(stop - start).count());
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n * n; ++i) {
            assert(arf_cmp_d(C[i], 3.75 * static_cast<double>(n)) == 0);
        }
    }

    bs.SetComplexityN(static_cast<benchmark::ComplexityN>(n));
    bs.SetItemsProcessed(static_cast<std::int64_t>(n * n * n) *
                         bs.iterations());

    for (std::size_t i = 0; i < n * n; ++i) { arf_clear(B[i]); }
    std::free(B);

    for (std::size_t i = 0; i < n * n; ++i) { arf_clear(A[i]); }
    std::free(A);

    for (std::size_t i = 0; i < n * n; ++i) { arf_clear(C[i]); }
    std::free(C);
}

static void axpy_bench_1(benchmark::State &bs) { axpy_bench<53>(bs); }
static void axpy_bench_2(benchmark::State &bs) { axpy_bench<103>(bs); }
static void axpy_bench_3(benchmark::State &bs) { axpy_bench<156>(bs); }
static void axpy_bench_4(benchmark::State &bs) { axpy_bench<208>(bs); }

static void dot_bench_1(benchmark::State &bs) { dot_bench<53>(bs); }
static void dot_bench_2(benchmark::State &bs) { dot_bench<103>(bs); }
static void dot_bench_3(benchmark::State &bs) { dot_bench<156>(bs); }
static void dot_bench_4(benchmark::State &bs) { dot_bench<208>(bs); }

static void gemv_bench_1(benchmark::State &bs) { gemv_bench<53>(bs); }
static void gemv_bench_2(benchmark::State &bs) { gemv_bench<103>(bs); }
static void gemv_bench_3(benchmark::State &bs) { gemv_bench<156>(bs); }
static void gemv_bench_4(benchmark::State &bs) { gemv_bench<208>(bs); }

static void gemm_bench_1(benchmark::State &bs) { gemm_bench<53>(bs); }
static void gemm_bench_2(benchmark::State &bs) { gemm_bench<103>(bs); }
static void gemm_bench_3(benchmark::State &bs) { gemm_bench<156>(bs); }
static void gemm_bench_4(benchmark::State &bs) { gemm_bench<208>(bs); }

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
