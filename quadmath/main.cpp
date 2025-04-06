#include <cassert>
#include <cstdint>
#include <cstdlib>

#include <benchmark/benchmark.h>

static void axpy(__float128 *y, __float128 a, const __float128 *x,
                 std::size_t n) {
#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) { y[i] += a * x[i]; }
}

static void axpy_bench(benchmark::State &bs) {

    const std::size_t n = static_cast<std::size_t>(bs.range(0));

    __float128 *const y =
        static_cast<__float128 *>(std::malloc(n * sizeof(__float128)));

    const __float128 a = static_cast<__float128>(0.5);

    __float128 *const x =
        static_cast<__float128 *>(std::malloc(n * sizeof(__float128)));

#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) { x[i] = static_cast<__float128>(i); }

    for (auto _ : bs) {
        bs.PauseTiming();
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
            y[i] = static_cast<__float128>(2.0) * static_cast<__float128>(i);
        }
        bs.ResumeTiming();
        axpy(y, a, x, n);
        bs.PauseTiming();
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
            assert(y[i] ==
                   static_cast<__float128>(2.5) * static_cast<__float128>(i));
        }
        bs.ResumeTiming();
    }

    bs.SetComplexityN(static_cast<benchmark::ComplexityN>(n));
    bs.SetItemsProcessed(static_cast<std::int64_t>(n));
}

BENCHMARK(axpy_bench)
    ->Repetitions(8)
    ->RangeMultiplier(2)
    ->Range(1L, 1L << 28)
    ->UseRealTime()
    ->DisplayAggregatesOnly();
