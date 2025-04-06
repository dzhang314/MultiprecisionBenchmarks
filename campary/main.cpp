#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdlib>

#include <benchmark/benchmark.h>
#include <campary/multi_prec.h>

static void axpy(multi_prec<2> *y, multi_prec<2> a, const multi_prec<2> *x,
                 std::size_t n) {
#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) { y[i] += a * x[i]; }
}

static void axpy_bench(benchmark::State &bs) {

    const std::size_t n = static_cast<std::size_t>(bs.range(0));

    multi_prec<2> *const y =
        static_cast<multi_prec<2> *>(std::malloc(n * sizeof(multi_prec<2>)));

    const multi_prec<2> a = static_cast<multi_prec<2>>(0.5);

    multi_prec<2> *const x =
        static_cast<multi_prec<2> *>(std::malloc(n * sizeof(multi_prec<2>)));

#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) { x[i] = static_cast<double>(i); }

    for (auto _ : bs) {
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
            y[i] = 2.0 * static_cast<double>(i);
        }
        const auto start = std::chrono::high_resolution_clock::now();
        axpy(y, a, x, n);
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

BENCHMARK(axpy_bench)
    ->UseManualTime()
    ->Complexity(benchmark::oN)
    ->Repetitions(3)
    ->RangeMultiplier(2)
    ->Range(1L << 8, 1L << 24)
    ->DisplayAggregatesOnly();
