#include <concepts>

#include "src/concept_defines/alithmetics.hpp"

template <typename T>
concept MatrixElement = (std::floating_point<T> || std::integral<T>);


template <typename T, typename U, typename V>
concept HasDotProduct = requires (T t, U u) {
    {t.dot(u)} -> std::same_as<V>;
};

template <typename M, typename N, typename L, typename K>
concept MatrixArithmetic = MatrixElement<K> &&ClosedAddable<M> &&
    ClosedSubtractable<M> && ClosedMultipliable<M> && HasDotProduct<M, N, L> &&
    Multipliable<K, M, M> && Multipliable<M, K, M>;