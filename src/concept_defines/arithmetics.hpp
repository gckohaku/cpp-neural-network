#ifndef ARITHMETICS_HPP
#define ARITHMETICS_HPP

#include <concepts>

template <typename T, typename U, typename V>
concept Addable = requires(T t, U u) {
    { t + u } -> std::same_as<V>;
};

template <typename T>
concept ClosedAddable = Addable<T, T, T>;

template <typename T, typename U, typename V>
concept Subtractable = requires(T t, U u) {
    { t - u } -> std::same_as<V>;
};

template <typename T>
concept ClosedSubtractable = Subtractable<T, T, T>;

template <typename T, typename U, typename V>
concept Multipliable = requires(T t, U u) {
    { t * u } -> std::same_as<V>;
};

template <typename T>
concept ClosedMultipliable = Multipliable<T, T, T>;

template <typename T, typename U, typename V>
concept Divisible = requires(T t, U u) {
    {t / u} -> std::same_as<V>;
};

template <typename T>
concept ClosedDivisible = Divisible<T, T, T>;

#endif