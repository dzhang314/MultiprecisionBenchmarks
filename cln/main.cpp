#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include <benchmark/benchmark.h>
#include <cln/float.h>

static void axpy(cln::cl_F *y, cln::cl_F a, const cln::cl_F *x, std::size_t n) {
#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) { y[i] += a * x[i]; }
}

static void axpy_bench(benchmark::State &bs) {

    const std::size_t n = static_cast<std::size_t>(bs.range(0));
    const cln::float_format_t format = cln::float_format(30);
    cln::default_float_format = format;

    std::vector<cln::cl_F> y(n);
    const cln::cl_F a = cln::cl_float(0.5, format);
    std::vector<cln::cl_F> x(n);

#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) {
        x[i] = cln::cl_float(static_cast<double>(i), format);
    }

    for (auto _ : bs) {
        bs.PauseTiming();
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
            y[i] = cln::cl_float(2.0 * static_cast<double>(i), format);
        }
        bs.ResumeTiming();
        axpy(y.data(), a, x.data(), n);
        bs.PauseTiming();
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
            assert(y[i] == 2.5 * static_cast<double>(i));
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
