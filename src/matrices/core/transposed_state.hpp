#ifndef MKNNLIB_MATRICES_CORE_TRANSPOSED_STATE_HPP
#define MKNNLIB_MATRICES_CORE_TRANSPOSED_STATE_HPP

#include <cstdint>
#include <type_traits>
namespace mknnlib::matrix::core {
    /**
     *  @enum TransposedState
     *  @brief 行列の転置状態を表す列挙体。1ビット目は転置されているか、2ビット目は複素共役かを表す。
     *  - None  
     *  行列の状態を保つ
     *  - Transpose  
     *  転置されている状態
     *  - ConjugateNone  
     *  複素共役行列の状態
     *  - ConjugateTranspose  
     *  エルミート転置 (複素共役転置) されている状態
     */
    enum class TransposedState : std::uint8_t {
        None = 0b00,
        Transpose = 0b01,
        ConjugateNone = 0b10,
        ConjugateTranspose = 0b11
    };

    constexpr TransposedState operator^(TransposedState rhs, TransposedState lhs) {
        using T = std::underlying_type_t<TransposedState>;
        return static_cast<TransposedState>(static_cast<T>(lhs) ^ static_cast<T>(rhs));
    }
}

#endif