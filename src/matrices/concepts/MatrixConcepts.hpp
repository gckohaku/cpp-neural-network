#include <concepts>

template <typename T>
concept MatrixElement = (std::floating_point<T> || std::integral<T>);

template <typename M, typename K>
concept MatrixConcept = MatrixElement<K> &&
                        // 行列同士の演算
                        requires(M a, M b) {
                          { a + b } -> std::same_as<M>;
                          { a - b } -> std::same_as<M>;
                          { a * b } -> std::same_as<M>;
                          { a.dot(b) } -> std::same_as<M>;
                        } &&
                        // スカラーと行列の演算
                        requires(M m, K k) {
                          { k * m } -> std::same_as<M>;
                          { m * k } -> std::same_as<M>;
                        };