#include <cassert>
#include <cstdint>
#include <cstdlib>

#include <benchmark/benchmark.h>
#include <mpfr.h>

static void axpy(mpfr_t *y, mpfr_t a, const mpfr_t *x, std::size_t n,
                 mpfr_prec_t prec) {
#pragma omp parallel
    {
        void *temp_buffer = alloca(mpfr_custom_get_size(prec));
        mpfr_custom_init(temp_buffer, prec);
        mpfr_t temp;
        mpfr_custom_init_set(temp, MPFR_REGULAR_KIND, 0, prec, temp_buffer);
#pragma omp for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
            mpfr_mul(temp, a, x[i], MPFR_RNDF);
            mpfr_add(y[i], y[i], temp, MPFR_RNDF);
        }
    }
}

static void axpy_bench(benchmark::State &bs) {

    const std::size_t n = static_cast<std::size_t>(bs.range(0));
    constexpr mpfr_prec_t prec = 104;

    mpfr_t *const y = static_cast<mpfr_t *>(std::malloc(n * sizeof(mpfr_t)));
    for (std::size_t i = 0; i < n; ++i) { mpfr_init2(y[i], prec); }

    mpfr_t a;
    mpfr_init2(a, prec);
    mpfr_set_d(a, 0.5, MPFR_RNDF);

    mpfr_t *const x = static_cast<mpfr_t *>(std::malloc(n * sizeof(mpfr_t)));
    for (std::size_t i = 0; i < n; ++i) { mpfr_init2(x[i], prec); }

#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) {
        mpfr_set_d(x[i], static_cast<double>(i), MPFR_RNDF);
    }

    for (auto _ : bs) {
        bs.PauseTiming();
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
            mpfr_set_d(y[i], 2.0 * static_cast<double>(i), MPFR_RNDF);
        }
        bs.ResumeTiming();
        axpy(y, a, x, n, prec);
        bs.PauseTiming();
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
            assert(mpfr_cmp_d(y[i], 2.5 * static_cast<double>(i)) == 0);
        }
        bs.ResumeTiming();
    }

    bs.SetComplexityN(static_cast<benchmark::ComplexityN>(n));
    bs.SetItemsProcessed(static_cast<std::int64_t>(n));

    for (std::size_t i = 0; i < n; ++i) { mpfr_clear(x[i]); }
    std::free(x);

    mpfr_clear(a);

    for (std::size_t i = 0; i < n; ++i) { mpfr_clear(y[i]); }
    std::free(y);
}

BENCHMARK(axpy_bench)
    ->Repetitions(8)
    ->RangeMultiplier(2)
    ->Range(1L, 1L << 28)
    ->UseRealTime()
    ->DisplayAggregatesOnly();
