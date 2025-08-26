#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <utility>

#include <benchmark/benchmark.h>
#include <omp.h>

#include "MultiFloats.hpp"

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

static f64x1 dot(const double *x0, const double *y0, std::size_t n) {
    f64x1 result = 0.0;
#pragma omp parallel
    {
        f64x1 scalar_sum = 0.0;
        v2f64x1 vector_sum = 0.0;
        const std::pair<std::size_t, std::size_t> range =
            split_range(n, omp_get_num_threads(), omp_get_thread_num());
        std::size_t i = range.first;
        for (; i + 2 <= range.second; i += 2) {
            const v2f64x1 x(vld1q_f64(x0 + i));
            const v2f64x1 y(vld1q_f64(y0 + i));
            vector_sum += x * y;
        }
        for (; i < range.second; ++i) {
            const f64x1 x(x0[i]);
            const f64x1 y(y0[i]);
            scalar_sum += x * y;
        }
        scalar_sum += vsum(vector_sum);
#pragma omp critical
        { result += scalar_sum; }
    }
    return result;
}

static f64x2 dot(const double *x0, const double *x1, const double *y0,
                 const double *y1, std::size_t n) {
    f64x2 result = 0.0;
#pragma omp parallel
    {
        f64x2 scalar_sum = 0.0;
        v2f64x2 vector_sum = 0.0;
        const std::pair<std::size_t, std::size_t> range =
            split_range(n, omp_get_num_threads(), omp_get_thread_num());
        std::size_t i = range.first;
        for (; i + 2 <= range.second; i += 2) {
            const v2f64x2 x(vld1q_f64(x0 + i), vld1q_f64(x1 + i));
            const v2f64x2 y(vld1q_f64(y0 + i), vld1q_f64(y1 + i));
            vector_sum += x * y;
        }
        for (; i < range.second; ++i) {
            const f64x2 x(x0[i], x1[i]);
            const f64x2 y(y0[i], y1[i]);
            scalar_sum += x * y;
        }
        scalar_sum += vsum(vector_sum);
#pragma omp critical
        { result += scalar_sum; }
    }
    return result;
}

static f64x3 dot(const double *x0, const double *x1, const double *x2,
                 const double *y0, const double *y1, const double *y2,
                 std::size_t n) {
    f64x3 result = 0.0;
#pragma omp parallel
    {
        f64x3 scalar_sum = 0.0;
        v2f64x3 vector_sum = 0.0;
        const std::pair<std::size_t, std::size_t> range =
            split_range(n, omp_get_num_threads(), omp_get_thread_num());
        std::size_t i = range.first;
        for (; i + 2 <= range.second; i += 2) {
            const v2f64x3 x(vld1q_f64(x0 + i), vld1q_f64(x1 + i),
                            vld1q_f64(x2 + i));
            const v2f64x3 y(vld1q_f64(y0 + i), vld1q_f64(y1 + i),
                            vld1q_f64(y2 + i));
            vector_sum += x * y;
        }
        for (; i < range.second; ++i) {
            const f64x3 x(x0[i], x1[i], x2[i]);
            const f64x3 y(y0[i], y1[i], y2[i]);
            scalar_sum += x * y;
        }
        scalar_sum += vsum(vector_sum);
#pragma omp critical
        { result += scalar_sum; }
    }
    return result;
}

static f64x4 dot(const double *x0, const double *x1, const double *x2,
                 const double *x3, const double *y0, const double *y1,
                 const double *y2, const double *y3, std::size_t n) {
    f64x4 result = 0.0;
#pragma omp parallel
    {
        f64x4 scalar_sum = 0.0;
        v2f64x4 vector_sum = 0.0;
        const std::pair<std::size_t, std::size_t> range =
            split_range(n, omp_get_num_threads(), omp_get_thread_num());
        std::size_t i = range.first;
        for (; i + 2 <= range.second; i += 2) {
            const v2f64x4 x(vld1q_f64(x0 + i), vld1q_f64(x1 + i),
                            vld1q_f64(x2 + i), vld1q_f64(x3 + i));
            const v2f64x4 y(vld1q_f64(y0 + i), vld1q_f64(y1 + i),
                            vld1q_f64(y2 + i), vld1q_f64(y3 + i));
            vector_sum += x * y;
        }
        for (; i < range.second; ++i) {
            const f64x4 x(x0[i], x1[i], x2[i], x3[i]);
            const f64x4 y(y0[i], y1[i], y2[i], y3[i]);
            scalar_sum += x * y;
        }
        scalar_sum += vsum(vector_sum);
#pragma omp critical
        { result += scalar_sum; }
    }
    return result;
}

static void gemv(double *y0, const double *A0, const double *x0,
                 std::size_t n) {
#pragma omp parallel for
    for (std::size_t i = 0; i < n; ++i) {
        f64x1 scalar_sum = 0.0;
        v2f64x1 vector_sum = 0.0;
        std::size_t j = 0;
        for (; j + 2 <= n; j += 2) {
            const v2f64x1 x(vld1q_f64(x0 + j));
            const v2f64x1 A(vld1q_f64(A0 + i * n + j));
            vector_sum += A * x;
        }
        for (; j < n; ++j) {
            const f64x1 x(x0[j]);
            const f64x1 A(A0[i * n + j]);
            scalar_sum += A * x;
        }
        scalar_sum += vsum(vector_sum);
        y0[i] = scalar_sum._limbs[0];
    }
}

static void gemv(double *y0, double *y1, const double *A0, const double *A1,
                 const double *x0, const double *x1, std::size_t n) {
#pragma omp parallel for
    for (std::size_t i = 0; i < n; ++i) {
        f64x2 scalar_sum = 0.0;
        v2f64x2 vector_sum = 0.0;
        std::size_t j = 0;
        for (; j + 2 <= n; j += 2) {
            const v2f64x2 x(vld1q_f64(x0 + j), vld1q_f64(x1 + j));
            const v2f64x2 A(vld1q_f64(A0 + i * n + j),
                            vld1q_f64(A1 + i * n + j));
            vector_sum += A * x;
        }
        for (; j < n; ++j) {
            const f64x2 x(x0[j], x1[j]);
            const f64x2 A(A0[i * n + j], A1[i * n + j]);
            scalar_sum += A * x;
        }
        scalar_sum += vsum(vector_sum);
        y0[i] = scalar_sum._limbs[0];
        y1[i] = scalar_sum._limbs[1];
    }
}

static void gemv(double *y0, double *y1, double *y2, const double *A0,
                 const double *A1, const double *A2, const double *x0,
                 const double *x1, const double *x2, std::size_t n) {
#pragma omp parallel for
    for (std::size_t i = 0; i < n; ++i) {
        f64x3 scalar_sum = 0.0;
        v2f64x3 vector_sum = 0.0;
        std::size_t j = 0;
        for (; j + 2 <= n; j += 2) {
            const v2f64x3 x(vld1q_f64(x0 + j), vld1q_f64(x1 + j),
                            vld1q_f64(x2 + j));
            const v2f64x3 A(vld1q_f64(A0 + i * n + j),
                            vld1q_f64(A1 + i * n + j),
                            vld1q_f64(A2 + i * n + j));
            vector_sum += A * x;
        }
        for (; j < n; ++j) {
            const f64x3 x(x0[j], x1[j], x2[j]);
            const f64x3 A(A0[i * n + j], A1[i * n + j], A2[i * n + j]);
            scalar_sum += A * x;
        }
        scalar_sum += vsum(vector_sum);
        y0[i] = scalar_sum._limbs[0];
        y1[i] = scalar_sum._limbs[1];
        y2[i] = scalar_sum._limbs[2];
    }
}

static void gemv(double *y0, double *y1, double *y2, double *y3,
                 const double *A0, const double *A1, const double *A2,
                 const double *A3, const double *x0, const double *x1,
                 const double *x2, const double *x3, std::size_t n) {
#pragma omp parallel for
    for (std::size_t i = 0; i < n; ++i) {
        f64x4 scalar_sum = 0.0;
        v2f64x4 vector_sum = 0.0;
        std::size_t j = 0;
        for (; j + 2 <= n; j += 2) {
            const v2f64x4 x(vld1q_f64(x0 + j), vld1q_f64(x1 + j),
                            vld1q_f64(x2 + j), vld1q_f64(x3 + j));
            const v2f64x4 A(
                vld1q_f64(A0 + i * n + j), vld1q_f64(A1 + i * n + j),
                vld1q_f64(A2 + i * n + j), vld1q_f64(A3 + i * n + j));
            vector_sum += A * x;
        }
        for (; j < n; ++j) {
            const f64x4 x(x0[j], x1[j], x2[j], x3[j]);
            const f64x4 A(A0[i * n + j], A1[i * n + j], A2[i * n + j],
                          A3[i * n + j]);
            scalar_sum += A * x;
        }
        scalar_sum += vsum(vector_sum);
        y0[i] = scalar_sum._limbs[0];
        y1[i] = scalar_sum._limbs[1];
        y2[i] = scalar_sum._limbs[2];
        y3[i] = scalar_sum._limbs[3];
    }
}

static void gemm(double *C0, const double *A0, const double *B0,
                 std::size_t n) {
#pragma omp parallel
    {
        const std::pair<std::size_t, std::size_t> range =
            split_range(n, omp_get_num_threads(), omp_get_thread_num());
        for (std::size_t i = range.first; i < range.second; ++i) {
            for (std::size_t k = 0; k < n; ++k) {
                const f64x1 A(A0[i * n + k]);
#pragma omp simd simdlen(8)
                for (std::size_t j = 0; j < n; ++j) {
                    const f64x1 B(B0[k * n + j]);
                    const f64x1 C(C0[i * n + j]);
                    const f64x1 z = C + A * B;
                    C0[i * n + j] = z._limbs[0];
                }
            }
        }
    }
}

static void gemm(double *C0, double *C1, const double *A0, const double *A1,
                 const double *B0, const double *B1, std::size_t n) {
#pragma omp parallel
    {
        const std::pair<std::size_t, std::size_t> range =
            split_range(n, omp_get_num_threads(), omp_get_thread_num());
        for (std::size_t i = range.first; i < range.second; ++i) {
            for (std::size_t k = 0; k < n; ++k) {
                const f64x2 A(A0[i * n + k], A1[i * n + k]);
#pragma omp simd simdlen(8)
                for (std::size_t j = 0; j < n; ++j) {
                    const f64x2 B(B0[k * n + j], B1[k * n + j]);
                    const f64x2 C(C0[i * n + j], C1[i * n + j]);
                    const f64x2 z = C + A * B;
                    C0[i * n + j] = z._limbs[0];
                    C1[i * n + j] = z._limbs[1];
                }
            }
        }
    }
}

static void gemm(double *C0, double *C1, double *C2, const double *A0,
                 const double *A1, const double *A2, const double *B0,
                 const double *B1, const double *B2, std::size_t n) {
#pragma omp parallel
    {
        const std::pair<std::size_t, std::size_t> range =
            split_range(n, omp_get_num_threads(), omp_get_thread_num());
        for (std::size_t i = range.first; i < range.second; ++i) {
            for (std::size_t k = 0; k < n; ++k) {
                const f64x3 A(A0[i * n + k], A1[i * n + k], A2[i * n + k]);
#pragma omp simd simdlen(8)
                for (std::size_t j = 0; j < n; ++j) {
                    const f64x3 B(B0[k * n + j], B1[k * n + j], B2[k * n + j]);
                    const f64x3 C(C0[i * n + j], C1[i * n + j], C2[i * n + j]);
                    const f64x3 z = C + A * B;
                    C0[i * n + j] = z._limbs[0];
                    C1[i * n + j] = z._limbs[1];
                    C2[i * n + j] = z._limbs[2];
                }
            }
        }
    }
}

static void gemm(double *C0, double *C1, double *C2, double *C3,
                 const double *A0, const double *A1, const double *A2,
                 const double *A3, const double *B0, const double *B1,
                 const double *B2, const double *B3, std::size_t n) {
#pragma omp parallel
    {
        const std::pair<std::size_t, std::size_t> range =
            split_range(n, omp_get_num_threads(), omp_get_thread_num());
        for (std::size_t i = range.first; i < range.second; ++i) {
            for (std::size_t k = 0; k < n; ++k) {
                const f64x4 A(A0[i * n + k], A1[i * n + k], A2[i * n + k],
                              A3[i * n + k]);
#pragma omp simd simdlen(8)
                for (std::size_t j = 0; j < n; ++j) {
                    const f64x4 B(B0[k * n + j], B1[k * n + j], B2[k * n + j],
                                  B3[k * n + j]);
                    const f64x4 C(C0[i * n + j], C1[i * n + j], C2[i * n + j],
                                  C3[i * n + j]);
                    const f64x4 z = C + A * B;
                    C0[i * n + j] = z._limbs[0];
                    C1[i * n + j] = z._limbs[1];
                    C2[i * n + j] = z._limbs[2];
                    C3[i * n + j] = z._limbs[3];
                }
            }
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

static void dot_bench_1(benchmark::State &bs) {

    const std::size_t n = static_cast<std::size_t>(bs.range(0));

    double *const x0 = static_cast<double *>(std::malloc(n * sizeof(double)));

#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) { x0[i] = 1.5; }

    double *const y0 = static_cast<double *>(std::malloc(n * sizeof(double)));

#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) { y0[i] = 2.5; }

    for (auto _ : bs) {
        const auto start = std::chrono::high_resolution_clock::now();
        const f64x1 result = dot(x0, y0, n);
        const auto stop = std::chrono::high_resolution_clock::now();
        bs.SetIterationTime(
            std::chrono::duration<double>(stop - start).count());
        assert(result == 3.75 * static_cast<double>(n));
    }

    bs.SetComplexityN(static_cast<benchmark::ComplexityN>(n));
    bs.SetItemsProcessed(static_cast<std::int64_t>(n) * bs.iterations());

    std::free(y0);
    std::free(x0);
}

static void dot_bench_2(benchmark::State &bs) {

    const std::size_t n = static_cast<std::size_t>(bs.range(0));

    double *const x0 = static_cast<double *>(std::malloc(n * sizeof(double)));
    double *const x1 = static_cast<double *>(std::malloc(n * sizeof(double)));

#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) {
        x0[i] = 1.5;
        x1[i] = 0.0;
    }

    double *const y0 = static_cast<double *>(std::malloc(n * sizeof(double)));
    double *const y1 = static_cast<double *>(std::malloc(n * sizeof(double)));

#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) {
        y0[i] = 2.5;
        y1[i] = 0.0;
    }

    for (auto _ : bs) {
        const auto start = std::chrono::high_resolution_clock::now();
        const f64x2 result = dot(x0, x1, y0, y1, n);
        const auto stop = std::chrono::high_resolution_clock::now();
        bs.SetIterationTime(
            std::chrono::duration<double>(stop - start).count());
        assert(result == 3.75 * static_cast<double>(n));
    }

    bs.SetComplexityN(static_cast<benchmark::ComplexityN>(n));
    bs.SetItemsProcessed(static_cast<std::int64_t>(n) * bs.iterations());

    std::free(y1);
    std::free(y0);
    std::free(x1);
    std::free(x0);
}

static void dot_bench_3(benchmark::State &bs) {

    const std::size_t n = static_cast<std::size_t>(bs.range(0));

    double *const x0 = static_cast<double *>(std::malloc(n * sizeof(double)));
    double *const x1 = static_cast<double *>(std::malloc(n * sizeof(double)));
    double *const x2 = static_cast<double *>(std::malloc(n * sizeof(double)));

#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) {
        x0[i] = 1.5;
        x1[i] = 0.0;
        x2[i] = 0.0;
    }

    double *const y0 = static_cast<double *>(std::malloc(n * sizeof(double)));
    double *const y1 = static_cast<double *>(std::malloc(n * sizeof(double)));
    double *const y2 = static_cast<double *>(std::malloc(n * sizeof(double)));

#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) {
        y0[i] = 2.5;
        y1[i] = 0.0;
        y2[i] = 0.0;
    }

    for (auto _ : bs) {
        const auto start = std::chrono::high_resolution_clock::now();
        const f64x3 result = dot(x0, x1, x2, y0, y1, y2, n);
        const auto stop = std::chrono::high_resolution_clock::now();
        bs.SetIterationTime(
            std::chrono::duration<double>(stop - start).count());
        assert(result == 3.75 * static_cast<double>(n));
    }

    bs.SetComplexityN(static_cast<benchmark::ComplexityN>(n));
    bs.SetItemsProcessed(static_cast<std::int64_t>(n) * bs.iterations());

    std::free(y2);
    std::free(y1);
    std::free(y0);
    std::free(x2);
    std::free(x1);
    std::free(x0);
}

static void dot_bench_4(benchmark::State &bs) {

    const std::size_t n = static_cast<std::size_t>(bs.range(0));

    double *const x0 = static_cast<double *>(std::malloc(n * sizeof(double)));
    double *const x1 = static_cast<double *>(std::malloc(n * sizeof(double)));
    double *const x2 = static_cast<double *>(std::malloc(n * sizeof(double)));
    double *const x3 = static_cast<double *>(std::malloc(n * sizeof(double)));

#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) {
        x0[i] = 1.5;
        x1[i] = 0.0;
        x2[i] = 0.0;
        x3[i] = 0.0;
    }

    double *const y0 = static_cast<double *>(std::malloc(n * sizeof(double)));
    double *const y1 = static_cast<double *>(std::malloc(n * sizeof(double)));
    double *const y2 = static_cast<double *>(std::malloc(n * sizeof(double)));
    double *const y3 = static_cast<double *>(std::malloc(n * sizeof(double)));

#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) {
        y0[i] = 2.5;
        y1[i] = 0.0;
        y2[i] = 0.0;
        y3[i] = 0.0;
    }

    for (auto _ : bs) {
        const auto start = std::chrono::high_resolution_clock::now();
        const f64x4 result = dot(x0, x1, x2, x3, y0, y1, y2, y3, n);
        const auto stop = std::chrono::high_resolution_clock::now();
        bs.SetIterationTime(
            std::chrono::duration<double>(stop - start).count());
        assert(result == 3.75 * static_cast<double>(n));
    }

    bs.SetComplexityN(static_cast<benchmark::ComplexityN>(n));
    bs.SetItemsProcessed(static_cast<std::int64_t>(n) * bs.iterations());

    std::free(y3);
    std::free(y2);
    std::free(y1);
    std::free(y0);
    std::free(x3);
    std::free(x2);
    std::free(x1);
    std::free(x0);
}

#include <iostream>

static void gemv_bench_1(benchmark::State &bs) {

    const std::size_t n = static_cast<std::size_t>(bs.range(0));

    double *const y0 = static_cast<double *>(std::malloc(n * sizeof(double)));

    double *const A0 =
        static_cast<double *>(std::malloc(n * n * sizeof(double)));

#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n * n; ++i) { A0[i] = 1.5; }

    double *const x0 = static_cast<double *>(std::malloc(n * sizeof(double)));

#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) { x0[i] = 2.5; }

    for (auto _ : bs) {
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n; ++i) { y0[i] = 0.0; }
        const auto start = std::chrono::high_resolution_clock::now();
        gemv(y0, A0, x0, n);
        const auto stop = std::chrono::high_resolution_clock::now();
        bs.SetIterationTime(
            std::chrono::duration<double>(stop - start).count());
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
#pragma omp critical
            {
                if (y0[i] != 3.75 * static_cast<double>(n)) {
                    std::cout << "y0[" << i << "] = " << y0[i] << std::endl;
                    std::cout << "Expected: " << 3.75 * static_cast<double>(n)
                              << std::endl;
                    std::cout << "n = " << n << std::endl;
                    std::cout << "i = " << i << std::endl;
                    assert(false);
                }
            }
        }
    }

    bs.SetComplexityN(static_cast<benchmark::ComplexityN>(n));
    bs.SetItemsProcessed(static_cast<std::int64_t>(n * n) * bs.iterations());

    std::free(x0);
    std::free(A0);
    std::free(y0);
}

static void gemv_bench_2(benchmark::State &bs) {

    const std::size_t n = static_cast<std::size_t>(bs.range(0));

    double *const y0 = static_cast<double *>(std::malloc(n * sizeof(double)));
    double *const y1 = static_cast<double *>(std::malloc(n * sizeof(double)));

    double *const A0 =
        static_cast<double *>(std::malloc(n * n * sizeof(double)));
    double *const A1 =
        static_cast<double *>(std::malloc(n * n * sizeof(double)));

#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n * n; ++i) {
        A0[i] = 1.5;
        A1[i] = 0.0;
    }

    double *const x0 = static_cast<double *>(std::malloc(n * sizeof(double)));
    double *const x1 = static_cast<double *>(std::malloc(n * sizeof(double)));
#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) {
        x0[i] = 2.5;
        x1[i] = 0.0;
    }

    for (auto _ : bs) {
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
            y0[i] = 0.0;
            y1[i] = 0.0;
        }
        const auto start = std::chrono::high_resolution_clock::now();
        gemv(y0, y1, A0, A1, x0, x1, n);
        const auto stop = std::chrono::high_resolution_clock::now();
        bs.SetIterationTime(
            std::chrono::duration<double>(stop - start).count());
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
            assert(y0[i] == 3.75 * static_cast<double>(n));
            assert(y1[i] == 0.0);
        }
    }

    bs.SetComplexityN(static_cast<benchmark::ComplexityN>(n));
    bs.SetItemsProcessed(static_cast<std::int64_t>(n * n) * bs.iterations());

    std::free(x1);
    std::free(x0);
    std::free(A1);
    std::free(A0);
    std::free(y1);
    std::free(y0);
}

static void gemv_bench_3(benchmark::State &bs) {

    const std::size_t n = static_cast<std::size_t>(bs.range(0));

    double *const y0 = static_cast<double *>(std::malloc(n * sizeof(double)));
    double *const y1 = static_cast<double *>(std::malloc(n * sizeof(double)));
    double *const y2 = static_cast<double *>(std::malloc(n * sizeof(double)));

    double *const A0 =
        static_cast<double *>(std::malloc(n * n * sizeof(double)));
    double *const A1 =
        static_cast<double *>(std::malloc(n * n * sizeof(double)));
    double *const A2 =
        static_cast<double *>(std::malloc(n * n * sizeof(double)));

#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n * n; ++i) {
        A0[i] = 1.5;
        A1[i] = 0.0;
        A2[i] = 0.0;
    }

    double *const x0 = static_cast<double *>(std::malloc(n * sizeof(double)));
    double *const x1 = static_cast<double *>(std::malloc(n * sizeof(double)));
    double *const x2 = static_cast<double *>(std::malloc(n * sizeof(double)));

#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) {
        x0[i] = 2.5;
        x1[i] = 0.0;
        x2[i] = 0.0;
    }

    for (auto _ : bs) {
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
            y0[i] = 0.0;
            y1[i] = 0.0;
            y2[i] = 0.0;
        }
        const auto start = std::chrono::high_resolution_clock::now();
        gemv(y0, y1, y2, A0, A1, A2, x0, x1, x2, n);
        const auto stop = std::chrono::high_resolution_clock::now();
        bs.SetIterationTime(
            std::chrono::duration<double>(stop - start).count());
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
            assert(y0[i] == 3.75 * static_cast<double>(n));
            assert(y1[i] == 0.0);
            assert(y2[i] == 0.0);
        }
    }

    bs.SetComplexityN(static_cast<benchmark::ComplexityN>(n));
    bs.SetItemsProcessed(static_cast<std::int64_t>(n * n) * bs.iterations());

    std::free(x2);
    std::free(x1);
    std::free(x0);
    std::free(A2);
    std::free(A1);
    std::free(A0);
    std::free(y2);
    std::free(y1);
    std::free(y0);
}

static void gemv_bench_4(benchmark::State &bs) {

    const std::size_t n = static_cast<std::size_t>(bs.range(0));

    double *const y0 = static_cast<double *>(std::malloc(n * sizeof(double)));
    double *const y1 = static_cast<double *>(std::malloc(n * sizeof(double)));
    double *const y2 = static_cast<double *>(std::malloc(n * sizeof(double)));
    double *const y3 = static_cast<double *>(std::malloc(n * sizeof(double)));

    double *const A0 =
        static_cast<double *>(std::malloc(n * n * sizeof(double)));
    double *const A1 =
        static_cast<double *>(std::malloc(n * n * sizeof(double)));
    double *const A2 =
        static_cast<double *>(std::malloc(n * n * sizeof(double)));
    double *const A3 =
        static_cast<double *>(std::malloc(n * n * sizeof(double)));

#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n * n; ++i) {
        A0[i] = 1.5;
        A1[i] = 0.0;
        A2[i] = 0.0;
        A3[i] = 0.0;
    }

    double *const x0 = static_cast<double *>(std::malloc(n * sizeof(double)));
    double *const x1 = static_cast<double *>(std::malloc(n * sizeof(double)));
    double *const x2 = static_cast<double *>(std::malloc(n * sizeof(double)));
    double *const x3 = static_cast<double *>(std::malloc(n * sizeof(double)));

#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) {
        x0[i] = 2.5;
        x1[i] = 0.0;
        x2[i] = 0.0;
        x3[i] = 0.0;
    }

    for (auto _ : bs) {
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
            y0[i] = 0.0;
            y1[i] = 0.0;
            y2[i] = 0.0;
            y3[i] = 0.0;
        }
        const auto start = std::chrono::high_resolution_clock::now();
        gemv(y0, y1, y2, y3, A0, A1, A2, A3, x0, x1, x2, x3, n);
        const auto stop = std::chrono::high_resolution_clock::now();
        bs.SetIterationTime(
            std::chrono::duration<double>(stop - start).count());
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
            assert(y0[i] == 3.75 * static_cast<double>(n));
            assert(y1[i] == 0.0);
            assert(y2[i] == 0.0);
            assert(y3[i] == 0.0);
        }
    }

    bs.SetComplexityN(static_cast<benchmark::ComplexityN>(n));
    bs.SetItemsProcessed(static_cast<std::int64_t>(n * n) * bs.iterations());

    std::free(x3);
    std::free(x2);
    std::free(x1);
    std::free(x0);
    std::free(A3);
    std::free(A2);
    std::free(A1);
    std::free(A0);
    std::free(y3);
    std::free(y2);
    std::free(y1);
    std::free(y0);
}

static void gemm_bench_1(benchmark::State &bs) {

    const std::size_t n = static_cast<std::size_t>(bs.range(0));

    double *const C0 =
        static_cast<double *>(std::malloc(n * n * sizeof(double)));

    double *const A0 =
        static_cast<double *>(std::malloc(n * n * sizeof(double)));

#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n * n; ++i) { A0[i] = 1.5; }

    double *const B0 =
        static_cast<double *>(std::malloc(n * n * sizeof(double)));

#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n * n; ++i) { B0[i] = 2.5; }

    for (auto _ : bs) {
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n * n; ++i) { C0[i] = 0.0; }
        const auto start = std::chrono::high_resolution_clock::now();
        gemm(C0, A0, B0, n);
        const auto stop = std::chrono::high_resolution_clock::now();
        bs.SetIterationTime(
            std::chrono::duration<double>(stop - start).count());
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n * n; ++i) {
            assert(C0[i] == 3.75 * static_cast<double>(n));
        }
    }

    bs.SetComplexityN(static_cast<benchmark::ComplexityN>(n));
    bs.SetItemsProcessed(static_cast<std::int64_t>(n * n * n) *
                         bs.iterations());

    std::free(B0);
    std::free(A0);
    std::free(C0);
}

static void gemm_bench_2(benchmark::State &bs) {

    const std::size_t n = static_cast<std::size_t>(bs.range(0));

    double *const C0 =
        static_cast<double *>(std::malloc(n * n * sizeof(double)));
    double *const C1 =
        static_cast<double *>(std::malloc(n * n * sizeof(double)));

    double *const A0 =
        static_cast<double *>(std::malloc(n * n * sizeof(double)));
    double *const A1 =
        static_cast<double *>(std::malloc(n * n * sizeof(double)));

#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n * n; ++i) {
        A0[i] = 1.5;
        A1[i] = 0.0;
    }

    double *const B0 =
        static_cast<double *>(std::malloc(n * n * sizeof(double)));
    double *const B1 =
        static_cast<double *>(std::malloc(n * n * sizeof(double)));

#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n * n; ++i) {
        B0[i] = 2.5;
        B1[i] = 0.0;
    }

    for (auto _ : bs) {
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n * n; ++i) {
            C0[i] = 0.0;
            C1[i] = 0.0;
        }
        const auto start = std::chrono::high_resolution_clock::now();
        gemm(C0, C1, A0, A1, B0, B1, n);
        const auto stop = std::chrono::high_resolution_clock::now();
        bs.SetIterationTime(
            std::chrono::duration<double>(stop - start).count());
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n * n; ++i) {
            assert(C0[i] == 3.75 * static_cast<double>(n));
            assert(C1[i] == 0.0);
        }
    }

    bs.SetComplexityN(static_cast<benchmark::ComplexityN>(n));
    bs.SetItemsProcessed(static_cast<std::int64_t>(n * n * n) *
                         bs.iterations());

    std::free(B1);
    std::free(B0);
    std::free(A1);
    std::free(A0);
    std::free(C1);
    std::free(C0);
}

static void gemm_bench_3(benchmark::State &bs) {

    const std::size_t n = static_cast<std::size_t>(bs.range(0));

    double *const C0 =
        static_cast<double *>(std::malloc(n * n * sizeof(double)));
    double *const C1 =
        static_cast<double *>(std::malloc(n * n * sizeof(double)));
    double *const C2 =
        static_cast<double *>(std::malloc(n * n * sizeof(double)));

    double *const A0 =
        static_cast<double *>(std::malloc(n * n * sizeof(double)));
    double *const A1 =
        static_cast<double *>(std::malloc(n * n * sizeof(double)));
    double *const A2 =
        static_cast<double *>(std::malloc(n * n * sizeof(double)));

#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n * n; ++i) {
        A0[i] = 1.5;
        A1[i] = 0.0;
        A2[i] = 0.0;
    }

    double *const B0 =
        static_cast<double *>(std::malloc(n * n * sizeof(double)));
    double *const B1 =
        static_cast<double *>(std::malloc(n * n * sizeof(double)));
    double *const B2 =
        static_cast<double *>(std::malloc(n * n * sizeof(double)));

#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n * n; ++i) {
        B0[i] = 2.5;
        B1[i] = 0.0;
        B2[i] = 0.0;
    }

    for (auto _ : bs) {
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n * n; ++i) {
            C0[i] = 0.0;
            C1[i] = 0.0;
            C2[i] = 0.0;
        }
        const auto start = std::chrono::high_resolution_clock::now();
        gemm(C0, C1, C2, A0, A1, A2, B0, B1, B2, n);
        const auto stop = std::chrono::high_resolution_clock::now();
        bs.SetIterationTime(
            std::chrono::duration<double>(stop - start).count());
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n * n; ++i) {
            assert(C0[i] == 3.75 * static_cast<double>(n));
            assert(C1[i] == 0.0);
            assert(C2[i] == 0.0);
        }
    }

    bs.SetComplexityN(static_cast<benchmark::ComplexityN>(n));
    bs.SetItemsProcessed(static_cast<std::int64_t>(n * n * n) *
                         bs.iterations());

    std::free(B2);
    std::free(B1);
    std::free(B0);
    std::free(A2);
    std::free(A1);
    std::free(A0);
    std::free(C2);
    std::free(C1);
    std::free(C0);
}

static void gemm_bench_4(benchmark::State &bs) {

    const std::size_t n = static_cast<std::size_t>(bs.range(0));

    double *const C0 =
        static_cast<double *>(std::malloc(n * n * sizeof(double)));
    double *const C1 =
        static_cast<double *>(std::malloc(n * n * sizeof(double)));
    double *const C2 =
        static_cast<double *>(std::malloc(n * n * sizeof(double)));
    double *const C3 =
        static_cast<double *>(std::malloc(n * n * sizeof(double)));

    double *const A0 =
        static_cast<double *>(std::malloc(n * n * sizeof(double)));
    double *const A1 =
        static_cast<double *>(std::malloc(n * n * sizeof(double)));
    double *const A2 =
        static_cast<double *>(std::malloc(n * n * sizeof(double)));
    double *const A3 =
        static_cast<double *>(std::malloc(n * n * sizeof(double)));

#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n * n; ++i) {
        A0[i] = 1.5;
        A1[i] = 0.0;
        A2[i] = 0.0;
        A3[i] = 0.0;
    }

    double *const B0 =
        static_cast<double *>(std::malloc(n * n * sizeof(double)));
    double *const B1 =
        static_cast<double *>(std::malloc(n * n * sizeof(double)));
    double *const B2 =
        static_cast<double *>(std::malloc(n * n * sizeof(double)));
    double *const B3 =
        static_cast<double *>(std::malloc(n * n * sizeof(double)));

#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n * n; ++i) {
        B0[i] = 2.5;
        B1[i] = 0.0;
        B2[i] = 0.0;
        B3[i] = 0.0;
    }

    for (auto _ : bs) {
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n * n; ++i) {
            C0[i] = 0.0;
            C1[i] = 0.0;
            C2[i] = 0.0;
            C3[i] = 0.0;
        }
        const auto start = std::chrono::high_resolution_clock::now();
        gemm(C0, C1, C2, C3, A0, A1, A2, A3, B0, B1, B2, B3, n);
        const auto stop = std::chrono::high_resolution_clock::now();
        bs.SetIterationTime(
            std::chrono::duration<double>(stop - start).count());
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n * n; ++i) {
            assert(C0[i] == 3.75 * static_cast<double>(n));
            assert(C1[i] == 0.0);
            assert(C2[i] == 0.0);
            assert(C3[i] == 0.0);
        }
    }

    bs.SetComplexityN(static_cast<benchmark::ComplexityN>(n));
    bs.SetItemsProcessed(static_cast<std::int64_t>(n * n * n) *
                         bs.iterations());

    std::free(B3);
    std::free(B2);
    std::free(B1);
    std::free(B0);
    std::free(A3);
    std::free(A2);
    std::free(A1);
    std::free(A0);
    std::free(C3);
    std::free(C2);
    std::free(C1);
    std::free(C0);
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
