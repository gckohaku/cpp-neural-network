#include <concepts>

#include "src/concept_defines/alithmetics.hpp"

template <typename K>
concept MatrixElement = (std::floating_point<K> || std::integral<K>);

template <typename M1, typename M2, typename M3>
concept HasDotProduct = requires(M1 t, M2 u) {
    { t.dot(u) } -> std::same_as<M3>;
};

template <typename M, typename K>
concept MatrixClosedArithmetic =
    MatrixElement<K> && ClosedAddable<M> && ClosedSubtractable<M> &&
    ClosedMultipliable<M>;