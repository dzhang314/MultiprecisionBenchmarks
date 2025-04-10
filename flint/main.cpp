#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdlib>

#include <benchmark/benchmark.h>
#include <flint/arf.h>

static void axpy(arf_t *y, const arf_t a, const arf_t *x, std::size_t n,
                 slong prec) {
#pragma omp parallel
    {
        arf_t temp;
        arf_init(temp);
#pragma omp for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
            arf_mul(temp, a, x[i], prec, ARF_RND_NEAR);
            arf_add(y[i], y[i], temp, prec, ARF_RND_NEAR);
        }
        arf_clear(temp);
    }
}

static void axpy_bench_1(benchmark::State &bs) {
    const std::size_t n = static_cast<std::size_t>(bs.range(0));
    constexpr slong prec = 53;

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
        axpy(y, a, x, n, prec);
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

static void axpy_bench_2(benchmark::State &bs) {
    const std::size_t n = static_cast<std::size_t>(bs.range(0));
    constexpr slong prec = 103;

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
        axpy(y, a, x, n, prec);
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

static void axpy_bench_3(benchmark::State &bs) {
    const std::size_t n = static_cast<std::size_t>(bs.range(0));
    constexpr slong prec = 156;

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
        axpy(y, a, x, n, prec);
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

static void axpy_bench_4(benchmark::State &bs) {
    const std::size_t n = static_cast<std::size_t>(bs.range(0));
    constexpr slong prec = 209;

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
        axpy(y, a, x, n, prec);
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
