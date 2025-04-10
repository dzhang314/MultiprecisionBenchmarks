#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <utility>

#include <benchmark/benchmark.h>
#include <omp.h>

template <typename T> constexpr std::pair<T, T> two_sum(T a, T b) {
    const T s = a + b;
    const T a_prime = s - b;
    const T b_prime = s - a_prime;
    const T a_err = a - a_prime;
    const T b_err = b - b_prime;
    const T e = a_err + b_err;
    return {s, e};
}

template <typename T> constexpr std::pair<T, T> fast_two_sum(T a, T b) {
    const T s = a + b;
    const T b_prime = s - a;
    const T e = b - b_prime;
    return {s, e};
}

template <typename T> constexpr std::pair<T, T> two_prod(T a, T b) {
    const T p = a * b;
    const T e = __builtin_fma(a, b, -p);
    return {p, e};
}

struct f64x1 {
    double _limbs[1];
    constexpr f64x1(double x) : _limbs{x} {}
    constexpr bool operator==(double rhs) const { return _limbs[0] == rhs; }
};

constexpr f64x1 operator+(const f64x1 x, const f64x1 y) {
    const double a = x._limbs[0];
    const double b = y._limbs[0];
    return f64x1{a + b};
}

constexpr f64x1 operator*(const f64x1 x, const f64x1 y) {
    const double a = x._limbs[0];
    const double b = y._limbs[0];
    return f64x1{a * b};
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

struct f64x3 {
    double _limbs[3];
    constexpr f64x3(double x) : _limbs{x, 0.0, 0.0} {}
    constexpr f64x3(double x, double y, double z) : _limbs{x, y, z} {}
    constexpr bool operator==(double rhs) const {
        return (_limbs[0] == rhs) & (_limbs[1] == 0.0) & (_limbs[2] == 0.0);
    }
};

constexpr f64x3 operator+(const f64x3 x, const f64x3 y) {
    const double a = x._limbs[0];
    const double b = y._limbs[0];
    const double c = x._limbs[1];
    const double d = y._limbs[1];
    const double e = x._limbs[2];
    const double f = y._limbs[2];
    const auto [a1, b1] = two_sum(a, b);
    const auto [c1, d1] = two_sum(c, d);
    const auto [e1, f1] = two_sum(e, f);
    const auto [a2, c2] = fast_two_sum(a1, c1);
    const double b2 = b1 + f1;
    const auto [d2, e2] = two_sum(d1, e1);
    const auto [a3, d3] = fast_two_sum(a2, d2);
    const auto [b3, c3] = two_sum(b2, c2);
    const double c4 = c3 + e2;
    const auto [c5, d5] = two_sum(c4, d3);
    const auto [b6, c6] = two_sum(b3, c5);
    const auto [a7, b7] = fast_two_sum(a3, b6);
    const double c7 = c6 + d5;
    const auto [b8, c8] = fast_two_sum(b7, c7);
    return f64x3{a7, b8, c8};
}

constexpr f64x3 operator*(const f64x3 x, const f64x3 y) {
    const auto [a, b] = two_prod(x._limbs[0], y._limbs[0]);
    const auto [c, e] = two_prod(x._limbs[0], y._limbs[1]);
    const auto [d, f] = two_prod(x._limbs[1], y._limbs[0]);
    const double g = x._limbs[0] * y._limbs[2];
    const double h = x._limbs[1] * y._limbs[1];
    const double i = x._limbs[2] * y._limbs[0];
    const auto [c1, d1] = two_sum(c, d);
    const double e1 = e + f;
    const double g1 = g + i;
    const auto [b2, c2] = two_sum(b, c1);
    const double g2 = g1 + h;
    const auto [a3, b3] = fast_two_sum(a, b2);
    const double c3 = c2 + d1;
    const double e3 = e1 + g2;
    const double c4 = c3 + e3;
    const auto [b5, c5] = fast_two_sum(b3, c4);
    const auto [a6, b6] = fast_two_sum(a3, b5);
    const auto [b7, c7] = fast_two_sum(b6, c5);
    return f64x3{a6, b7, c7};
}

struct f64x4 {
    double _limbs[4];
    constexpr f64x4(double x) : _limbs{x, 0.0, 0.0, 0.0} {}
    constexpr f64x4(double x, double y, double z, double w)
        : _limbs{x, y, z, w} {}
    constexpr bool operator==(double rhs) const {
        return (_limbs[0] == rhs) & (_limbs[1] == 0.0) & (_limbs[2] == 0.0) &
               (_limbs[3] == 0.0);
    }
};

constexpr f64x4 operator+(const f64x4 x, const f64x4 y) {
    const double a = x._limbs[0];
    const double b = y._limbs[0];
    const double c = x._limbs[1];
    const double d = y._limbs[1];
    const double e = x._limbs[2];
    const double f = y._limbs[2];
    const double g = x._limbs[3];
    const double h = y._limbs[3];
    const auto [a1, b1] = two_sum(a, b);
    const auto [c1, d1] = two_sum(c, d);
    const auto [e1, f1] = two_sum(e, f);
    const auto [g1, h1] = two_sum(g, h);
    const auto [a2, c2] = fast_two_sum(a1, c1);
    const double b2 = b1 + h1;
    const auto [d2, e2] = two_sum(d1, e1);
    const auto [f2, g2] = two_sum(f1, g1);
    const auto [b3, g3] = two_sum(b2, g2);
    const auto [c3, d3] = fast_two_sum(c2, d2);
    const auto [e3, f3] = two_sum(e2, f2);
    const auto [a4, c4] = fast_two_sum(a2, c3);
    const auto [d4, e4] = fast_two_sum(d3, e3);
    const auto [b5, d5] = two_sum(b3, d4);
    const double e5 = e4 + f3;
    const auto [b6, c6] = two_sum(b5, c4);
    const auto [d6, e6] = two_sum(d5, e5);
    const auto [a7, b7] = fast_two_sum(a4, b6);
    const auto [c7, d7] = fast_two_sum(c6, d6);
    const double e8 = e6 + g3;
    const auto [b8, c8] = fast_two_sum(b7, c7);
    const double d9 = d7 + e8;
    const auto [a10, b10] = fast_two_sum(a7, b8);
    const auto [c10, d10] = fast_two_sum(c8, d9);
    const auto [b11, c11] = fast_two_sum(b10, c10);
    const auto [c12, d12] = fast_two_sum(c11, d10);
    return f64x4{a10, b11, c12, d12};
}

constexpr f64x4 operator*(const f64x4 x, const f64x4 y) {
    const auto [a, b] = two_prod(x._limbs[0], y._limbs[0]);
    const auto [c, e] = two_prod(x._limbs[0], y._limbs[1]);
    const auto [d, f] = two_prod(x._limbs[1], y._limbs[0]);
    const auto [g, j] = two_prod(x._limbs[0], y._limbs[2]);
    const auto [h, k] = two_prod(x._limbs[1], y._limbs[1]);
    const auto [i, l] = two_prod(x._limbs[2], y._limbs[0]);
    const double m = x._limbs[0] * y._limbs[3];
    const double n = x._limbs[1] * y._limbs[2];
    const double o = x._limbs[2] * y._limbs[1];
    const double p = x._limbs[3] * y._limbs[0];
    const auto [c1, d1] = two_sum(c, d);
    const auto [e1, f1] = two_sum(e, f);
    const auto [g1, i1] = two_sum(g, i);
    const double j1 = j + l;
    const double m1 = m + p;
    const double n1 = n + o;
    const auto [b2, c2] = two_sum(b, c1);
    const auto [e2, h2] = two_sum(e1, h);
    const double f2 = f1 + j1;
    const double i2 = i1 + k;
    const double m2 = m1 + n1;
    const auto [a3, b3] = fast_two_sum(a, b2);
    const auto [c3, d3] = fast_two_sum(c2, d1);
    const auto [e3, g3] = two_sum(e2, g1);
    const double f3 = f2 + m2;
    const double h3 = h2 + i2;
    const auto [c4, e4] = two_sum(c3, e3);
    const double d4 = d3 + h3;
    const double f4 = f3 + g3;
    const double d5 = d4 + e4;
    const auto [c6, d6] = two_sum(c4, d5);
    const auto [b7, c7] = two_sum(b3, c6);
    const double d7 = d6 + f4;
    const auto [a8, b8] = fast_two_sum(a3, b7);
    const auto [c8, d8] = two_sum(c7, d7);
    const auto [b9, c9] = two_sum(b8, c8);
    const auto [c10, d10] = fast_two_sum(c9, d8);
    return f64x4{a8, b9, c10, d10};
}

constexpr std::pair<std::size_t, std::size_t>
split_range(std::size_t n, int num_threads, int thread_id) {
    const std::size_t m = static_cast<std::size_t>(num_threads);
    const std::size_t i = static_cast<std::size_t>(thread_id);
    const std::size_t q = n / m;
    const std::size_t r = n % m;
    const std::size_t chunk_size = (i < r) ? (q + 1) : q;
    const std::size_t first = (i < r) ? (q * i + i) : (q * i + r);
    const std::size_t last = first + chunk_size;
    return {first, last};
}

static void axpy(double *y0, f64x1 a, const double *x0, std::size_t n) {
#pragma omp parallel
    {
        const std::pair<std::size_t, std::size_t> range =
            split_range(n, omp_get_num_threads(), omp_get_thread_num());
#pragma omp simd simdlen(8)
        for (std::size_t i = range.first; i < range.second; ++i) {
            const f64x1 y(y0[i]);
            const f64x1 x(x0[i]);
            const f64x1 z = y + a * x;
            y0[i] = z._limbs[0];
        }
    }
}

static void axpy(double *y0, double *y1, f64x2 a, const double *x0,
                 const double *x1, std::size_t n) {
#pragma omp parallel
    {
        const std::pair<std::size_t, std::size_t> range =
            split_range(n, omp_get_num_threads(), omp_get_thread_num());
#pragma omp simd simdlen(8)
        for (std::size_t i = range.first; i < range.second; ++i) {
            const f64x2 y(y0[i], y1[i]);
            const f64x2 x(x0[i], x1[i]);
            const f64x2 z = y + a * x;
            y0[i] = z._limbs[0];
            y1[i] = z._limbs[1];
        }
    }
}

static void axpy(double *y0, double *y1, double *y2, f64x3 a, const double *x0,
                 const double *x1, const double *x2, std::size_t n) {
#pragma omp parallel
    {
        const std::pair<std::size_t, std::size_t> range =
            split_range(n, omp_get_num_threads(), omp_get_thread_num());
#pragma omp simd simdlen(8)
        for (std::size_t i = range.first; i < range.second; ++i) {
            const f64x3 y(y0[i], y1[i], y2[i]);
            const f64x3 x(x0[i], x1[i], x2[i]);
            const f64x3 z = y + a * x;
            y0[i] = z._limbs[0];
            y1[i] = z._limbs[1];
            y2[i] = z._limbs[2];
        }
    }
}

static void axpy(double *y0, double *y1, double *y2, double *y3, f64x4 a,
                 const double *x0, const double *x1, const double *x2,
                 const double *x3, std::size_t n) {
#pragma omp parallel
    {
        const std::pair<std::size_t, std::size_t> range =
            split_range(n, omp_get_num_threads(), omp_get_thread_num());
#pragma omp simd simdlen(8)
        for (std::size_t i = range.first; i < range.second; ++i) {
            const f64x4 y(y0[i], y1[i], y2[i], y3[i]);
            const f64x4 x(x0[i], x1[i], x2[i], x3[i]);
            const f64x4 z = y + a * x;
            y0[i] = z._limbs[0];
            y1[i] = z._limbs[1];
            y2[i] = z._limbs[2];
            y3[i] = z._limbs[3];
        }
    }
}

static void axpy_bench_1(benchmark::State &bs) {

    const std::size_t n = static_cast<std::size_t>(bs.range(0));

    double *const y0 = static_cast<double *>(std::malloc(n * sizeof(double)));

    const f64x1 a = static_cast<f64x1>(0.5);

    double *const x0 = static_cast<double *>(std::malloc(n * sizeof(double)));

#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) { x0[i] = static_cast<double>(i); }

    for (auto _ : bs) {
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
            y0[i] = 2.0 * static_cast<double>(i);
        }
        const auto start = std::chrono::high_resolution_clock::now();
        axpy(y0, a, x0, n);
        const auto stop = std::chrono::high_resolution_clock::now();
        bs.SetIterationTime(
            std::chrono::duration<double>(stop - start).count());
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
            assert(y0[i] == 2.5 * static_cast<double>(i));
        }
    }

    bs.SetComplexityN(static_cast<benchmark::ComplexityN>(n));
    bs.SetItemsProcessed(static_cast<std::int64_t>(n) * bs.iterations());

    std::free(x0);
    std::free(y0);
}

static void axpy_bench_2(benchmark::State &bs) {

    const std::size_t n = static_cast<std::size_t>(bs.range(0));

    double *const y0 = static_cast<double *>(std::malloc(n * sizeof(double)));
    double *const y1 = static_cast<double *>(std::malloc(n * sizeof(double)));

    const f64x2 a = static_cast<f64x2>(0.5);

    double *const x0 = static_cast<double *>(std::malloc(n * sizeof(double)));
    double *const x1 = static_cast<double *>(std::malloc(n * sizeof(double)));

#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) {
        x0[i] = static_cast<double>(i);
        x1[i] = 0.0;
    }

    for (auto _ : bs) {
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
            y0[i] = 2.0 * static_cast<double>(i);
            y1[i] = 0.0;
        }
        const auto start = std::chrono::high_resolution_clock::now();
        axpy(y0, y1, a, x0, x1, n);
        const auto stop = std::chrono::high_resolution_clock::now();
        bs.SetIterationTime(
            std::chrono::duration<double>(stop - start).count());
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
            assert(y0[i] == 2.5 * static_cast<double>(i));
            assert(y1[i] == 0.0);
        }
    }

    bs.SetComplexityN(static_cast<benchmark::ComplexityN>(n));
    bs.SetItemsProcessed(static_cast<std::int64_t>(n) * bs.iterations());

    std::free(x1);
    std::free(x0);
    std::free(y1);
    std::free(y0);
}

static void axpy_bench_3(benchmark::State &bs) {

    const std::size_t n = static_cast<std::size_t>(bs.range(0));

    double *const y0 = static_cast<double *>(std::malloc(n * sizeof(double)));
    double *const y1 = static_cast<double *>(std::malloc(n * sizeof(double)));
    double *const y2 = static_cast<double *>(std::malloc(n * sizeof(double)));

    const f64x3 a = static_cast<f64x3>(0.5);

    double *const x0 = static_cast<double *>(std::malloc(n * sizeof(double)));
    double *const x1 = static_cast<double *>(std::malloc(n * sizeof(double)));
    double *const x2 = static_cast<double *>(std::malloc(n * sizeof(double)));

#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) {
        x0[i] = static_cast<double>(i);
        x1[i] = 0.0;
        x2[i] = 0.0;
    }

    for (auto _ : bs) {
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
            y0[i] = 2.0 * static_cast<double>(i);
            y1[i] = 0.0;
            y2[i] = 0.0;
        }
        const auto start = std::chrono::high_resolution_clock::now();
        axpy(y0, y1, y2, a, x0, x1, x2, n);
        const auto stop = std::chrono::high_resolution_clock::now();
        bs.SetIterationTime(
            std::chrono::duration<double>(stop - start).count());
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
            assert(y0[i] == 2.5 * static_cast<double>(i));
            assert(y1[i] == 0.0);
            assert(y2[i] == 0.0);
        }
    }

    bs.SetComplexityN(static_cast<benchmark::ComplexityN>(n));
    bs.SetItemsProcessed(static_cast<std::int64_t>(n) * bs.iterations());

    std::free(x2);
    std::free(x1);
    std::free(x0);
    std::free(y2);
    std::free(y1);
    std::free(y0);
}

static void axpy_bench_4(benchmark::State &bs) {

    const std::size_t n = static_cast<std::size_t>(bs.range(0));

    double *const y0 = static_cast<double *>(std::malloc(n * sizeof(double)));
    double *const y1 = static_cast<double *>(std::malloc(n * sizeof(double)));
    double *const y2 = static_cast<double *>(std::malloc(n * sizeof(double)));
    double *const y3 = static_cast<double *>(std::malloc(n * sizeof(double)));

    const f64x4 a = static_cast<f64x4>(0.5);

    double *const x0 = static_cast<double *>(std::malloc(n * sizeof(double)));
    double *const x1 = static_cast<double *>(std::malloc(n * sizeof(double)));
    double *const x2 = static_cast<double *>(std::malloc(n * sizeof(double)));
    double *const x3 = static_cast<double *>(std::malloc(n * sizeof(double)));

#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) {
        x0[i] = static_cast<double>(i);
        x1[i] = 0.0;
        x2[i] = 0.0;
        x3[i] = 0.0;
    }

    for (auto _ : bs) {
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
            y0[i] = 2.0 * static_cast<double>(i);
            y1[i] = 0.0;
            y2[i] = 0.0;
            y3[i] = 0.0;
        }
        const auto start = std::chrono::high_resolution_clock::now();
        axpy(y0, y1, y2, y3, a, x0, x1, x2, x3, n);
        const auto stop = std::chrono::high_resolution_clock::now();
        bs.SetIterationTime(
            std::chrono::duration<double>(stop - start).count());
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
            assert(y0[i] == 2.5 * static_cast<double>(i));
            assert(y1[i] == 0.0);
            assert(y2[i] == 0.0);
            assert(y3[i] == 0.0);
        }
    }

    bs.SetComplexityN(static_cast<benchmark::ComplexityN>(n));
    bs.SetItemsProcessed(static_cast<std::int64_t>(n) * bs.iterations());

    std::free(x3);
    std::free(x2);
    std::free(x1);
    std::free(x0);
    std::free(y3);
    std::free(y2);
    std::free(y1);
    std::free(y0);
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
