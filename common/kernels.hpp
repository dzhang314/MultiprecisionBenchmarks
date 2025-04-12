#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdlib>

#include <benchmark/benchmark.h>

template <typename T>
static inline void axpy(T *y, T a, const T *x, std::size_t n) {
#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) { y[i] += a * x[i]; }
}

template <typename T>
static inline T dot(const T *x, const T *y, std::size_t n) {
    T result = static_cast<T>(0.0);
#pragma omp parallel
    {
        T local_sum = static_cast<T>(0.0);
#pragma omp for schedule(static)
        for (std::size_t i = 0; i < n; ++i) { local_sum += x[i] * y[i]; }
#pragma omp critical
        { result += local_sum; }
    }
    return result;
}

template <typename T>
static inline void gemv(T *y, const T *A, const T *x, std::size_t n) {
#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) { y[i] += A[i * n + j] * x[j]; }
    }
}

template <typename T>
static inline void gemm(T *C, const T *A, const T *B, std::size_t n) {
#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t k = 0; k < n; ++k) {
            for (std::size_t j = 0; j < n; ++j) {
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
}

template <typename T>
static inline void axpy_bench(benchmark::State &bs) {

    const std::size_t n = static_cast<std::size_t>(bs.range(0));

    T *const y = static_cast<T *>(std::malloc(n * sizeof(T)));
    const T a = static_cast<T>(0.5);
    T *const x = static_cast<T *>(std::malloc(n * sizeof(T)));

#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) {
        x[i] = static_cast<T>(static_cast<double>(i));
    }

    for (auto _ : bs) {
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
            y[i] = static_cast<T>(2.0) * static_cast<T>(static_cast<double>(i));
        }
        const auto start = std::chrono::high_resolution_clock::now();
        axpy(y, a, x, n);
        const auto stop = std::chrono::high_resolution_clock::now();
        bs.SetIterationTime(
            std::chrono::duration<double>(stop - start).count());
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
            assert(y[i] == (static_cast<T>(2.5) *
                            static_cast<T>(static_cast<double>(i))));
        }
    }

    bs.SetComplexityN(static_cast<benchmark::ComplexityN>(n));
    bs.SetItemsProcessed(static_cast<std::int64_t>(n) * bs.iterations());

    std::free(x);
    std::free(y);
}

template <typename T>
static inline void dot_bench(benchmark::State &bs) {

    const std::size_t n = static_cast<std::size_t>(bs.range(0));

    T *const x = static_cast<T *>(std::malloc(n * sizeof(T)));

#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) { x[i] = static_cast<T>(1.5); }

    T *const y = static_cast<T *>(std::malloc(n * sizeof(T)));

#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) { y[i] = static_cast<T>(2.5); }

    for (auto _ : bs) {
        const auto start = std::chrono::high_resolution_clock::now();
        const T result = dot(x, y, n);
        const auto stop = std::chrono::high_resolution_clock::now();
        bs.SetIterationTime(
            std::chrono::duration<double>(stop - start).count());
        assert(result ==
               static_cast<T>(3.75) * static_cast<T>(static_cast<double>(n)));
    }

    bs.SetComplexityN(static_cast<benchmark::ComplexityN>(n));
    bs.SetItemsProcessed(static_cast<std::int64_t>(n) * bs.iterations());

    std::free(y);
    std::free(x);
}

template <typename T>
static inline void gemv_bench(benchmark::State &bs) {

    const std::size_t n = static_cast<std::size_t>(bs.range(0));

    T *const y = static_cast<T *>(std::malloc(n * sizeof(T)));

    T *const A = static_cast<T *>(std::malloc(n * n * sizeof(T)));

#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n * n; ++i) { A[i] = static_cast<T>(1.5); }

    T *const x = static_cast<T *>(std::malloc(n * sizeof(T)));

#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) { x[i] = static_cast<T>(2.5); }

    for (auto _ : bs) {
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n; ++i) { y[i] = static_cast<T>(0.0); }
        const auto start = std::chrono::high_resolution_clock::now();
        gemv(y, A, x, n);
        const auto stop = std::chrono::high_resolution_clock::now();
        bs.SetIterationTime(
            std::chrono::duration<double>(stop - start).count());
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
            assert(y[i] == (static_cast<T>(3.75) *
                            static_cast<T>(static_cast<double>(n))));
        }
    }

    bs.SetComplexityN(static_cast<benchmark::ComplexityN>(n));
    bs.SetItemsProcessed(static_cast<std::int64_t>(n * n) * bs.iterations());

    std::free(x);
    std::free(A);
    std::free(y);
}

template <typename T>
static inline void gemm_bench(benchmark::State &bs) {

    const std::size_t n = static_cast<std::size_t>(bs.range(0));

    T *const C = static_cast<T *>(std::malloc(n * n * sizeof(T)));

    T *const A = static_cast<T *>(std::malloc(n * n * sizeof(T)));

#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n * n; ++i) { A[i] = static_cast<T>(1.5); }

    T *const B = static_cast<T *>(std::malloc(n * n * sizeof(T)));

#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n * n; ++i) { B[i] = static_cast<T>(2.5); }

    for (auto _ : bs) {
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n * n; ++i) { C[i] = static_cast<T>(0.0); }
        const auto start = std::chrono::high_resolution_clock::now();
        gemm(C, A, B, n);
        const auto stop = std::chrono::high_resolution_clock::now();
        bs.SetIterationTime(
            std::chrono::duration<double>(stop - start).count());
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n * n; ++i) {
            assert(C[i] == (static_cast<T>(3.75) *
                            static_cast<T>(static_cast<double>(n))));
        }
    }

    bs.SetComplexityN(static_cast<benchmark::ComplexityN>(n));
    bs.SetItemsProcessed(static_cast<std::int64_t>(n * n * n) *
                         bs.iterations());

    std::free(B);
    std::free(A);
    std::free(C);
}
