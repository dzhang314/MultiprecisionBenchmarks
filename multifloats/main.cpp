#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <tuple>

#include <benchmark/benchmark.h>

template <typename T> constexpr std::tuple<T, T> two_sum(T a, T b) {
    const T s = a + b;
    const T a_prime = s - b;
    const T b_prime = s - a_prime;
    const T a_err = a - a_prime;
    const T b_err = b - b_prime;
    const T e = a_err + b_err;
    return std::make_tuple(s, e);
}

template <typename T> constexpr std::tuple<T, T> fast_two_sum(T a, T b) {
    const T s = a + b;
    const T b_prime = s - a;
    const T e = b - b_prime;
    return std::make_tuple(s, e);
}

template <typename T> constexpr std::tuple<T, T> two_prod(T a, T b) {
    const T p = a * b;
    const T e = __builtin_fma(a, b, -p);
    return std::make_tuple(p, e);
}

struct f64x2 {
    double _limbs[2];
    constexpr f64x2(double x) : _limbs{x, 0.0} {}
    constexpr f64x2(double x, double y) : _limbs{x, y} {}
    constexpr bool operator==(double rhs) const {
        return (_limbs[0] == rhs) & (_limbs[1] == 0.0);
    }
};

constexpr f64x2 operator+(const f64x2 x, const f64x2 y) {
    const double a = x._limbs[0];
    const double b = y._limbs[0];
    const double c = x._limbs[1];
    const double d = y._limbs[1];
    const auto [a1, b1] = two_sum(a, b);
    const auto [c1, d1] = two_sum(c, d);
    const auto [a2, c2] = fast_two_sum(a1, c1);
    const double b2 = b1 + d1;
    const double b3 = b2 + c2;
    const auto [a4, b4] = fast_two_sum(a2, b3);
    return f64x2{a4, b4};
}

constexpr f64x2 operator*(const f64x2 x, const f64x2 y) {
    const auto [a, b] = two_prod(x._limbs[0], y._limbs[0]);
    const double c = x._limbs[0] * y._limbs[1];
    const double d = x._limbs[1] * y._limbs[0];
    const double c1 = c + d;
    const double b2 = b + c1;
    const auto [a3, b3] = fast_two_sum(a, b2);
    return f64x2{a3, b3};
}

static void axpy(f64x2 *y, f64x2 a, const f64x2 *x, std::size_t n) {
#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) { y[i] = y[i] + a * x[i]; }
}

static void axpy_bench(benchmark::State &bs) {

    const std::size_t n = static_cast<std::size_t>(bs.range(0));

    f64x2 *const y = static_cast<f64x2 *>(std::malloc(n * sizeof(f64x2)));

    const f64x2 a = static_cast<f64x2>(0.5);

    f64x2 *const x = static_cast<f64x2 *>(std::malloc(n * sizeof(f64x2)));

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

    std::free(x);
    std::free(y);
}

BENCHMARK(axpy_bench)
    ->UseManualTime()
    ->Complexity(benchmark::oN)
    ->Repetitions(3)
    ->RangeMultiplier(2)
    ->Range(1L << 8, 1L << 24)
    ->DisplayAggregatesOnly();
