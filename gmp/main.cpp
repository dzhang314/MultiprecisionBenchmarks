#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdlib>

#include <benchmark/benchmark.h>
#include <gmp.h>

template <mp_bitcnt_t PRECISION>
static void axpy(mpf_t *y, mpf_t a, const mpf_t *x, std::size_t n) {
#pragma omp parallel
    {
        mpf_t temp;
        mpf_init2(temp, PRECISION);
#pragma omp for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
            mpf_mul(temp, a, x[i]);
            mpf_add(y[i], y[i], temp);
        }
        mpf_clear(temp);
    }
}

template <mp_bitcnt_t PRECISION>
static void dot(mpf_t result, const mpf_t *x, const mpf_t *y, std::size_t n) {
    mpf_set_d(result, 0.0);
#pragma omp parallel
    {
        mpf_t local_sum;
        mpf_t temp;
        mpf_init2(local_sum, PRECISION);
        mpf_init2(temp, PRECISION);
        mpf_set_d(local_sum, 0.0);
#pragma omp for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
            mpf_mul(temp, x[i], y[i]);
            mpf_add(local_sum, local_sum, temp);
        }
#pragma omp critical
        { mpf_add(result, result, local_sum); }
        mpf_clear(temp);
        mpf_clear(local_sum);
    }
}

template <mp_bitcnt_t PRECISION>
static void axpy_bench(benchmark::State &bs) {

    const std::size_t n = static_cast<std::size_t>(bs.range(0));

    mpf_t *const y = static_cast<mpf_t *>(std::malloc(n * sizeof(mpf_t)));
    for (std::size_t i = 0; i < n; ++i) { mpf_init2(y[i], PRECISION); }

    mpf_t a;
    mpf_init2(a, PRECISION);
    mpf_set_d(a, 0.5);

    mpf_t *const x = static_cast<mpf_t *>(std::malloc(n * sizeof(mpf_t)));
    for (std::size_t i = 0; i < n; ++i) { mpf_init2(x[i], PRECISION); }

#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) {
        mpf_set_d(x[i], static_cast<double>(i));
    }

    for (auto _ : bs) {
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
            mpf_set_d(y[i], 2.0 * static_cast<double>(i));
        }
        const auto start = std::chrono::high_resolution_clock::now();
        axpy<PRECISION>(y, a, x, n);
        const auto stop = std::chrono::high_resolution_clock::now();
        bs.SetIterationTime(
            std::chrono::duration<double>(stop - start).count());
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
            assert(mpf_cmp_d(y[i], 2.5 * static_cast<double>(i)) == 0);
        }
    }

    bs.SetComplexityN(static_cast<benchmark::ComplexityN>(n));
    bs.SetItemsProcessed(static_cast<std::int64_t>(n) * bs.iterations());

    for (std::size_t i = 0; i < n; ++i) { mpf_clear(x[i]); }
    std::free(x);

    mpf_clear(a);

    for (std::size_t i = 0; i < n; ++i) { mpf_clear(y[i]); }
    std::free(y);
}

template <mp_bitcnt_t PRECISION>
static void dot_bench(benchmark::State &bs) {

    const std::size_t n = static_cast<std::size_t>(bs.range(0));

    mpf_t *const x = static_cast<mpf_t *>(std::malloc(n * sizeof(mpf_t)));
    for (std::size_t i = 0; i < n; ++i) { mpf_init2(x[i], PRECISION); }

#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) { mpf_set_d(x[i], 1.5); }

    mpf_t *const y = static_cast<mpf_t *>(std::malloc(n * sizeof(mpf_t)));
    for (std::size_t i = 0; i < n; ++i) { mpf_init2(y[i], PRECISION); }

#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) { mpf_set_d(y[i], 2.5); }

    mpf_t result;
    mpf_init2(result, PRECISION);

    for (auto _ : bs) {
        mpf_set_d(result, 0.0);
        const auto start = std::chrono::high_resolution_clock::now();
        dot<PRECISION>(result, x, y, n);
        const auto stop = std::chrono::high_resolution_clock::now();
        bs.SetIterationTime(
            std::chrono::duration<double>(stop - start).count());
        assert(mpf_cmp_d(result, 3.75 * static_cast<double>(n)) == 0);
    }

    bs.SetComplexityN(static_cast<benchmark::ComplexityN>(n));
    bs.SetItemsProcessed(static_cast<std::int64_t>(n) * bs.iterations());

    mpf_clear(result);

    for (std::size_t i = 0; i < n; ++i) { mpf_clear(y[i]); }
    std::free(y);

    for (std::size_t i = 0; i < n; ++i) { mpf_clear(x[i]); }
    std::free(x);
}

static void axpy_bench_1(benchmark::State &bs) { axpy_bench<53>(bs); }
static void axpy_bench_2(benchmark::State &bs) { axpy_bench<103>(bs); }
static void axpy_bench_3(benchmark::State &bs) { axpy_bench<156>(bs); }
static void axpy_bench_4(benchmark::State &bs) { axpy_bench<208>(bs); }

static void dot_bench_1(benchmark::State &bs) { dot_bench<53>(bs); }
static void dot_bench_2(benchmark::State &bs) { dot_bench<103>(bs); }
static void dot_bench_3(benchmark::State &bs) { dot_bench<156>(bs); }
static void dot_bench_4(benchmark::State &bs) { dot_bench<208>(bs); }

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
