#ifndef MULTIFLOATS_HPP_INCLUDED
#define MULTIFLOATS_HPP_INCLUDED

#include <tuple>
#include <type_traits>

template <typename T>
static constexpr std::tuple<T, T> two_sum(const T a, const T b) {
    const T s = a + b;
    const T a_prime = s - b;
    const T b_prime = s - a_prime;
    const T a_err = a - a_prime;
    const T b_err = b - b_prime;
    const T e = a_err + b_err;
    return {s, e};
}

template <typename T>
static constexpr std::tuple<T, T> fast_two_sum(const T a, const T b) {
    const T s = a + b;
    const T b_prime = s - a;
    const T e = b - b_prime;
    return {s, e};
}

static inline std::tuple<double, double> two_prod(const double a,
                                                  const double b) {
    const double p = a * b;
    const double e = __builtin_fma(a, b, -p);
    return {p, e};
}

template <typename T>
static constexpr T zero();

template <>
constexpr double zero<double>() {
    return 0.0;
}

template <typename T>
static constexpr T from_scalar(const double x);

template <>
constexpr double from_scalar(const double x) {
    return x;
}

template <typename T, int N>
struct MultiFloat {

    T _limbs[N];

    constexpr MultiFloat() : _limbs{} {
        for (int i = 0; i < N; ++i) { _limbs[i] = zero<T>(); }
    }

    constexpr MultiFloat(const double x) : _limbs{} {
        _limbs[0] = from_scalar<T>(x);
        for (int i = 1; i < N; ++i) { _limbs[i] = zero<T>(); }
    }

    template <typename... Args, typename = std::enable_if_t<
                                    (sizeof...(Args) == N) &&
                                    (std::is_convertible_v<Args, T> && ...)>>
    constexpr MultiFloat(Args &&...args)
        : _limbs{static_cast<T>(std::forward<Args>(args))...} {}

    constexpr bool operator==(const MultiFloat rhs) const {
        for (int i = 0; i < N; ++i) {
            if (!(_limbs[i] == rhs._limbs[i])) {
                return false;
            }
        }
        return true;
    }

    constexpr bool operator==(const T rhs) const {
        bool result = true;
        result &= (_limbs[0] == rhs);
        for (int i = 1; i < N; ++i) { result &= (_limbs[i] == zero<T>()); }
        return result;
    }

    constexpr MultiFloat &operator+=(const MultiFloat rhs) {
        *this = *this + rhs;
        return *this;
    }

    constexpr MultiFloat &operator*=(const MultiFloat rhs) {
        *this = *this * rhs;
        return *this;
    }

}; // struct MultiFloat<T, N>

template <typename T>
static constexpr MultiFloat<T, 1> operator+(const MultiFloat<T, 1> x,
                                            const MultiFloat<T, 1> y) {
    const T a = x._limbs[0];
    const T b = y._limbs[0];
    return MultiFloat<T, 1>{a + b};
}

template <typename T>
static constexpr MultiFloat<T, 1> operator*(const MultiFloat<T, 1> x,
                                            const MultiFloat<T, 1> y) {
    const T a = x._limbs[0];
    const T b = y._limbs[0];
    return MultiFloat<T, 1>{a * b};
}

template <typename T>
static constexpr MultiFloat<T, 2> operator+(const MultiFloat<T, 2> x,
                                            const MultiFloat<T, 2> y) {
    const T a = x._limbs[0];
    const T b = y._limbs[0];
    const T c = x._limbs[1];
    const T d = y._limbs[1];
    const auto [a1, b1] = two_sum(a, b);
    const auto [c1, d1] = two_sum(c, d);
    const auto [a2, c2] = fast_two_sum(a1, c1);
    const T b2 = b1 + d1;
    const T b3 = b2 + c2;
    const auto [a4, b4] = fast_two_sum(a2, b3);
    return MultiFloat<T, 2>{a4, b4};
}

template <typename T>
static constexpr MultiFloat<T, 2> operator*(const MultiFloat<T, 2> x,
                                            const MultiFloat<T, 2> y) {
    const auto [a, b] = two_prod(x._limbs[0], y._limbs[0]);
    const T c = x._limbs[0] * y._limbs[1];
    const T d = x._limbs[1] * y._limbs[0];
    const T c1 = c + d;
    const T b2 = b + c1;
    const auto [a3, b3] = fast_two_sum(a, b2);
    return MultiFloat<T, 2>{a3, b3};
}

template <typename T>
static constexpr MultiFloat<T, 3> operator+(const MultiFloat<T, 3> x,
                                            const MultiFloat<T, 3> y) {
    const T a = x._limbs[0];
    const T b = y._limbs[0];
    const T c = x._limbs[1];
    const T d = y._limbs[1];
    const T e = x._limbs[2];
    const T f = y._limbs[2];
    const auto [a1, b1] = two_sum(a, b);
    const auto [c1, d1] = two_sum(c, d);
    const auto [e1, f1] = two_sum(e, f);
    const auto [a2, c2] = fast_two_sum(a1, c1);
    const T b2 = b1 + f1;
    const auto [d2, e2] = two_sum(d1, e1);
    const auto [a3, d3] = fast_two_sum(a2, d2);
    const auto [b3, c3] = two_sum(b2, c2);
    const T c4 = c3 + e2;
    const auto [c5, d5] = two_sum(c4, d3);
    const auto [b6, c6] = two_sum(b3, c5);
    const auto [a7, b7] = fast_two_sum(a3, b6);
    const T c7 = c6 + d5;
    const auto [b8, c8] = fast_two_sum(b7, c7);
    return MultiFloat<T, 3>{a7, b8, c8};
}

template <typename T>
static constexpr MultiFloat<T, 3> operator*(const MultiFloat<T, 3> x,
                                            const MultiFloat<T, 3> y) {
    const auto [a, b] = two_prod(x._limbs[0], y._limbs[0]);
    const auto [c, e] = two_prod(x._limbs[0], y._limbs[1]);
    const auto [d, f] = two_prod(x._limbs[1], y._limbs[0]);
    const T g = x._limbs[0] * y._limbs[2];
    const T h = x._limbs[1] * y._limbs[1];
    const T i = x._limbs[2] * y._limbs[0];
    const auto [c1, d1] = two_sum(c, d);
    const T e1 = e + f;
    const T g1 = g + i;
    const auto [b2, c2] = two_sum(b, c1);
    const T g2 = g1 + h;
    const auto [a3, b3] = fast_two_sum(a, b2);
    const T c3 = c2 + d1;
    const T e3 = e1 + g2;
    const T c4 = c3 + e3;
    const auto [b5, c5] = fast_two_sum(b3, c4);
    const auto [a6, b6] = fast_two_sum(a3, b5);
    const auto [b7, c7] = fast_two_sum(b6, c5);
    return MultiFloat<T, 3>{a6, b7, c7};
}

template <typename T>
static constexpr MultiFloat<T, 4> operator+(const MultiFloat<T, 4> x,
                                            const MultiFloat<T, 4> y) {
    const T a = x._limbs[0];
    const T b = y._limbs[0];
    const T c = x._limbs[1];
    const T d = y._limbs[1];
    const T e = x._limbs[2];
    const T f = y._limbs[2];
    const T g = x._limbs[3];
    const T h = y._limbs[3];
    const auto [a1, b1] = two_sum(a, b);
    const auto [c1, d1] = two_sum(c, d);
    const auto [e1, f1] = two_sum(e, f);
    const auto [g1, h1] = two_sum(g, h);
    const auto [a2, c2] = fast_two_sum(a1, c1);
    const T b2 = b1 + h1;
    const auto [d2, e2] = two_sum(d1, e1);
    const auto [f2, g2] = two_sum(f1, g1);
    const auto [b3, g3] = two_sum(b2, g2);
    const auto [c3, d3] = fast_two_sum(c2, d2);
    const auto [e3, f3] = two_sum(e2, f2);
    const auto [a4, c4] = fast_two_sum(a2, c3);
    const auto [d4, e4] = fast_two_sum(d3, e3);
    const auto [b5, d5] = two_sum(b3, d4);
    const T e5 = e4 + f3;
    const auto [b6, c6] = two_sum(b5, c4);
    const auto [d6, e6] = two_sum(d5, e5);
    const auto [a7, b7] = fast_two_sum(a4, b6);
    const auto [c7, d7] = fast_two_sum(c6, d6);
    const T e8 = e6 + g3;
    const auto [b8, c8] = fast_two_sum(b7, c7);
    const T d9 = d7 + e8;
    const auto [a10, b10] = fast_two_sum(a7, b8);
    const auto [c10, d10] = fast_two_sum(c8, d9);
    const auto [b11, c11] = fast_two_sum(b10, c10);
    const auto [c12, d12] = fast_two_sum(c11, d10);
    return MultiFloat<T, 4>{a10, b11, c12, d12};
}

template <typename T>
static constexpr MultiFloat<T, 4> operator*(const MultiFloat<T, 4> x,
                                            const MultiFloat<T, 4> y) {
    const auto [a, b] = two_prod(x._limbs[0], y._limbs[0]);
    const auto [c, e] = two_prod(x._limbs[0], y._limbs[1]);
    const auto [d, f] = two_prod(x._limbs[1], y._limbs[0]);
    const auto [g, j] = two_prod(x._limbs[0], y._limbs[2]);
    const auto [h, k] = two_prod(x._limbs[1], y._limbs[1]);
    const auto [i, l] = two_prod(x._limbs[2], y._limbs[0]);
    const T m = x._limbs[0] * y._limbs[3];
    const T n = x._limbs[1] * y._limbs[2];
    const T o = x._limbs[2] * y._limbs[1];
    const T p = x._limbs[3] * y._limbs[0];
    const auto [c1, d1] = two_sum(c, d);
    const auto [e1, f1] = two_sum(e, f);
    const auto [g1, i1] = two_sum(g, i);
    const T j1 = j + l;
    const T m1 = m + p;
    const T n1 = n + o;
    const auto [b2, c2] = two_sum(b, c1);
    const auto [e2, h2] = two_sum(e1, h);
    const T f2 = f1 + j1;
    const T i2 = i1 + k;
    const T m2 = m1 + n1;
    const auto [a3, b3] = fast_two_sum(a, b2);
    const auto [c3, d3] = fast_two_sum(c2, d1);
    const auto [e3, g3] = two_sum(e2, g1);
    const T f3 = f2 + m2;
    const T h3 = h2 + i2;
    const auto [c4, e4] = two_sum(c3, e3);
    const T d4 = d3 + h3;
    const T f4 = f3 + g3;
    const T d5 = d4 + e4;
    const auto [c6, d6] = two_sum(c4, d5);
    const auto [b7, c7] = two_sum(b3, c6);
    const T d7 = d6 + f4;
    const auto [a8, b8] = fast_two_sum(a3, b7);
    const auto [c8, d8] = two_sum(c7, d7);
    const auto [b9, c9] = two_sum(b8, c8);
    const auto [c10, d10] = fast_two_sum(c9, d8);
    return MultiFloat<T, 4>{a8, b9, c10, d10};
}

using f64x1 = MultiFloat<double, 1>;
using f64x2 = MultiFloat<double, 2>;
using f64x3 = MultiFloat<double, 3>;
using f64x4 = MultiFloat<double, 4>;

#endif // MULTIFLOATS_HPP_INCLUDED
