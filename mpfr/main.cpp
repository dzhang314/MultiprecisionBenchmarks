#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdlib>

#include <benchmark/benchmark.h>
#include <mpfr.h>

template <mpfr_prec_t PRECISION>
static void axpy(mpfr_t *y, mpfr_t a, const mpfr_t *x, std::size_t n) {
#pragma omp parallel
    {
        mpfr_t temp;
        mpfr_init2(temp, PRECISION);
#pragma omp for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
            mpfr_mul(temp, a, x[i], MPFR_RNDF);
            mpfr_add(y[i], y[i], temp, MPFR_RNDF);
        }
        mpfr_clear(temp);
    }
}

template <mpfr_prec_t PRECISION>
static void dot(mpfr_t result, const mpfr_t *x, const mpfr_t *y,
                std::size_t n) {
    mpfr_set_zero(result, +1);
#pragma omp parallel
    {
        mpfr_t local_sum;
        mpfr_init2(local_sum, PRECISION);
        mpfr_set_zero(local_sum, +1);
        mpfr_t temp;
        mpfr_init2(temp, PRECISION);
#pragma omp for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
            mpfr_mul(temp, x[i], y[i], MPFR_RNDF);
            mpfr_add(local_sum, local_sum, temp, MPFR_RNDF);
        }
#pragma omp critical
        { mpfr_add(result, result, local_sum, MPFR_RNDF); }
        mpfr_clear(temp);
        mpfr_clear(local_sum);
    }
}

template <mpfr_prec_t PRECISION>
static void gemv(mpfr_t *y, const mpfr_t *A, const mpfr_t *x, std::size_t n) {
#pragma omp parallel
    {
        mpfr_t temp;
        mpfr_init2(temp, PRECISION);
#pragma omp for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                mpfr_mul(temp, A[i * n + j], x[j], MPFR_RNDF);
                mpfr_add(y[i], y[i], temp, MPFR_RNDF);
            }
        }
        mpfr_clear(temp);
    }
}

template <mpfr_prec_t PRECISION>
static void gemm(mpfr_t *C, const mpfr_t *A, const mpfr_t *B, std::size_t n) {
#pragma omp parallel
    {
        mpfr_t temp;
        mpfr_init2(temp, PRECISION);
#pragma omp for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t k = 0; k < n; ++k) {
                for (std::size_t j = 0; j < n; ++j) {
                    mpfr_mul(temp, A[i * n + k], B[k * n + j], MPFR_RNDF);
                    mpfr_add(C[i * n + j], C[i * n + j], temp, MPFR_RNDF);
                }
            }
        }
        mpfr_clear(temp);
    }
}

template <mpfr_prec_t PRECISION>
static void axpy_bench(benchmark::State &bs) {

    const std::size_t n = static_cast<std::size_t>(bs.range(0));

    mpfr_t *const y = static_cast<mpfr_t *>(std::malloc(n * sizeof(mpfr_t)));
    for (std::size_t i = 0; i < n; ++i) { mpfr_init2(y[i], PRECISION); }

    mpfr_t a;
    mpfr_init2(a, PRECISION);
    mpfr_set_d(a, 0.5, MPFR_RNDF);

    mpfr_t *const x = static_cast<mpfr_t *>(std::malloc(n * sizeof(mpfr_t)));
    for (std::size_t i = 0; i < n; ++i) { mpfr_init2(x[i], PRECISION); }

#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) {
        mpfr_set_d(x[i], static_cast<double>(i), MPFR_RNDF);
    }

    for (auto _ : bs) {
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
            mpfr_set_d(y[i], 2.0 * static_cast<double>(i), MPFR_RNDF);
        }
        const auto start = std::chrono::high_resolution_clock::now();
        axpy<PRECISION>(y, a, x, n);
        const auto stop = std::chrono::high_resolution_clock::now();
        bs.SetIterationTime(
            std::chrono::duration<double>(stop - start).count());
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
            assert(mpfr_cmp_d(y[i], 2.5 * static_cast<double>(i)) == 0);
        }
    }

    bs.SetComplexityN(static_cast<benchmark::ComplexityN>(n));
    bs.SetItemsProcessed(static_cast<std::int64_t>(n) * bs.iterations());

    for (std::size_t i = 0; i < n; ++i) { mpfr_clear(x[i]); }
    std::free(x);

    mpfr_clear(a);

    for (std::size_t i = 0; i < n; ++i) { mpfr_clear(y[i]); }
    std::free(y);
}

template <mpfr_prec_t PRECISION>
static void dot_bench(benchmark::State &bs) {

    const std::size_t n = static_cast<std::size_t>(bs.range(0));

    mpfr_t *const x = static_cast<mpfr_t *>(std::malloc(n * sizeof(mpfr_t)));
    for (std::size_t i = 0; i < n; ++i) { mpfr_init2(x[i], PRECISION); }

#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) { mpfr_set_d(x[i], 1.5, MPFR_RNDF); }

    mpfr_t *const y = static_cast<mpfr_t *>(std::malloc(n * sizeof(mpfr_t)));
    for (std::size_t i = 0; i < n; ++i) { mpfr_init2(y[i], PRECISION); }

#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) { mpfr_set_d(y[i], 2.5, MPFR_RNDF); }

    mpfr_t result;
    mpfr_init2(result, PRECISION);

    for (auto _ : bs) {
        mpfr_set_zero(result, +1);
        const auto start = std::chrono::high_resolution_clock::now();
        dot<PRECISION>(result, x, y, n);
        const auto stop = std::chrono::high_resolution_clock::now();
        bs.SetIterationTime(
            std::chrono::duration<double>(stop - start).count());
        assert(mpfr_cmp_d(result, 3.75 * static_cast<double>(n)) == 0);
    }

    bs.SetComplexityN(static_cast<benchmark::ComplexityN>(n));
    bs.SetItemsProcessed(static_cast<std::int64_t>(n) * bs.iterations());

    mpfr_clear(result);

    for (std::size_t i = 0; i < n; ++i) { mpfr_clear(y[i]); }
    std::free(y);

    for (std::size_t i = 0; i < n; ++i) { mpfr_clear(x[i]); }
    std::free(x);
}

template <mpfr_prec_t PRECISION>
static void gemv_bench(benchmark::State &bs) {

    const std::size_t n = static_cast<std::size_t>(bs.range(0));

    mpfr_t *const y = static_cast<mpfr_t *>(std::malloc(n * sizeof(mpfr_t)));
    for (std::size_t i = 0; i < n; ++i) { mpfr_init2(y[i], PRECISION); }

    mpfr_t *const A =
        static_cast<mpfr_t *>(std::malloc(n * n * sizeof(mpfr_t)));
    for (std::size_t i = 0; i < n * n; ++i) { mpfr_init2(A[i], PRECISION); }

#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n * n; ++i) {
        mpfr_set_d(A[i], 1.5, MPFR_RNDF);
    }

    mpfr_t *const x = static_cast<mpfr_t *>(std::malloc(n * sizeof(mpfr_t)));
    for (std::size_t i = 0; i < n; ++i) { mpfr_init2(x[i], PRECISION); }

#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) { mpfr_set_d(x[i], 2.5, MPFR_RNDF); }

    for (auto _ : bs) {
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n; ++i) { mpfr_set_zero(y[i], +1); }
        const auto start = std::chrono::high_resolution_clock::now();
        gemv<PRECISION>(y, A, x, n);
        const auto stop = std::chrono::high_resolution_clock::now();
        bs.SetIterationTime(
            std::chrono::duration<double>(stop - start).count());
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
            assert(mpfr_cmp_d(y[i], 3.75 * static_cast<double>(n)) == 0);
        }
    }

    bs.SetComplexityN(static_cast<benchmark::ComplexityN>(n));
    bs.SetItemsProcessed(static_cast<std::int64_t>(n * n) * bs.iterations());

    for (std::size_t i = 0; i < n; ++i) { mpfr_clear(x[i]); }
    std::free(x);

    for (std::size_t i = 0; i < n * n; ++i) { mpfr_clear(A[i]); }
    std::free(A);

    for (std::size_t i = 0; i < n; ++i) { mpfr_clear(y[i]); }
    std::free(y);
}

template <mpfr_prec_t PRECISION>
static void gemm_bench(benchmark::State &bs) {

    const std::size_t n = static_cast<std::size_t>(bs.range(0));

    mpfr_t *const C =
        static_cast<mpfr_t *>(std::malloc(n * n * sizeof(mpfr_t)));
    for (std::size_t i = 0; i < n * n; ++i) { mpfr_init2(C[i], PRECISION); }

    mpfr_t *const A =
        static_cast<mpfr_t *>(std::malloc(n * n * sizeof(mpfr_t)));
    for (std::size_t i = 0; i < n * n; ++i) { mpfr_init2(A[i], PRECISION); }

#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n * n; ++i) {
        mpfr_set_d(A[i], 1.5, MPFR_RNDF);
    }

    mpfr_t *const B =
        static_cast<mpfr_t *>(std::malloc(n * n * sizeof(mpfr_t)));
    for (std::size_t i = 0; i < n * n; ++i) { mpfr_init2(B[i], PRECISION); }

#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n * n; ++i) {
        mpfr_set_d(B[i], 2.5, MPFR_RNDF);
    }

    for (auto _ : bs) {
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n * n; ++i) { mpfr_set_zero(C[i], +1); }
        const auto start = std::chrono::high_resolution_clock::now();
        gemm<PRECISION>(C, A, B, n);
        const auto stop = std::chrono::high_resolution_clock::now();
        bs.SetIterationTime(
            std::chrono::duration<double>(stop - start).count());
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n * n; ++i) {
            assert(mpfr_cmp_d(C[i], 3.75 * static_cast<double>(n)) == 0);
        }
    }

    bs.SetComplexityN(static_cast<benchmark::ComplexityN>(n));
    bs.SetItemsProcessed(static_cast<std::int64_t>(n * n * n) *
                         bs.iterations());

    for (std::size_t i = 0; i < n * n; ++i) { mpfr_clear(B[i]); }
    std::free(B);

    for (std::size_t i = 0; i < n * n; ++i) { mpfr_clear(A[i]); }
    std::free(A);

    for (std::size_t i = 0; i < n * n; ++i) { mpfr_clear(C[i]); }
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
