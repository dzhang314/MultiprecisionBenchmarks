#include <cassert>
#include <cstdint>
#include <cstdlib>

#include <benchmark/benchmark.h>
#include <gmp.h>

static void axpy(mpf_t *y, mpf_t a, const mpf_t *x, std::size_t n,
                 mp_bitcnt_t prec) {
#pragma omp parallel
    {
        mpf_t temp;
        mpf_init2(temp, prec);
#pragma omp for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
            mpf_mul(temp, a, x[i]);
            mpf_add(y[i], y[i], temp);
        }
        mpf_clear(temp);
    }
}

static void axpy_bench(benchmark::State &bs) {

    const std::size_t n = static_cast<std::size_t>(bs.range(0));
    constexpr mp_bitcnt_t prec = 104;

    mpf_t *const y = static_cast<mpf_t *>(std::malloc(n * sizeof(mpf_t)));
    for (std::size_t i = 0; i < n; ++i) { mpf_init2(y[i], prec); }

    mpf_t a;
    mpf_init2(a, prec);
    mpf_set_d(a, 0.5);

    mpf_t *const x = static_cast<mpf_t *>(std::malloc(n * sizeof(mpf_t)));
    for (std::size_t i = 0; i < n; ++i) { mpf_init2(x[i], prec); }

#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) {
        mpf_set_d(x[i], static_cast<double>(i));
    }

    for (auto _ : bs) {
        bs.PauseTiming();
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
            mpf_set_d(y[i], 2.0 * static_cast<double>(i));
        }
        bs.ResumeTiming();
        axpy(y, a, x, n, prec);
        bs.PauseTiming();
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
            assert(mpf_cmp_d(y[i], 2.5 * static_cast<double>(i)) == 0);
        }
        bs.ResumeTiming();
    }

    bs.SetComplexityN(static_cast<benchmark::ComplexityN>(n));
    bs.SetItemsProcessed(static_cast<std::int64_t>(n));

    for (std::size_t i = 0; i < n; ++i) { mpf_clear(x[i]); }
    std::free(x);

    mpf_clear(a);

    for (std::size_t i = 0; i < n; ++i) { mpf_clear(y[i]); }
    std::free(y);
}

BENCHMARK(axpy_bench)
    ->Repetitions(8)
    ->RangeMultiplier(2)
    ->Range(1L, 1L << 28)
    ->UseRealTime()
    ->DisplayAggregatesOnly();
