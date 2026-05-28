#include <complex>
#include <concepts>

template <typename T>
concept MatrixElement = (std::floating_point<T> || std::integral<T>);

// template <typename M, typename K>
// concept MatrixConcept = MatrixElement<M> && requires(M a, M b) {
//     // 行列同士の演算
//     { a + b } -> std::same_as<M>;
//     { a - b } -> std::same_as<M>;
//     { a * b } -> std::same_as<M>;
//     { a.dot(b) } -> std::same_as<M>;
// } && requires(M m, K k) {
//     // スカラーと行列の演算
//     { k * m } -> std::same_as<M>;
//     { m * k } -> std::same_as<M>;
// };

template <typename T, typename U, typename V>
concept Addable = requires(T t, U u) {
    { t + u } -> std::same_as<V>;
};

template <typename T, typename U, typename V>
concept Subtractable = requires(T t, U u) {
    { t - u } -> std::same_as<V>;
};

template <typename T, typename U, typename V>
concept Multipliable = requires(T t, U u) {
    { t * u } -> std::same_as<V>;
};

template <typename T, typename U, typename V>
concept HasDotProduct = requires(T t, U u) {
    { t.dot(u) } -> std::same_as<V>;
};

template <typename M, typename K>
concept MatrixArithmetic = MatrixElement<K> && Addable<M, M, M> &&
    Subtractable<M, M, M> && Multipliable<M, M, M> && HasDotProduct<M, M, M> &&
    Multipliable<K, M, M> && Multipliable<M, K, M>;