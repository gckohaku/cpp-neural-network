#ifndef MKNNLIB_MATRICES_MATRIX_STATIC_OPERATOR_OVERRIDES_INL
#define MKNNLIB_MATRICES_MATRIX_STATIC_OPERATOR_OVERRIDES_INL

#include <cstddef>
#include "src/concept_defines/types/type_concepts.hpp"
#include "src/matrices/matrix_static.hpp"

namespace mknnlib::matrix {
// arithmetics compound operators
template <typename K, size_t Row, size_t Col, typename Backend>
    requires mk_concepts::BLASSupported<Backend, K>
inline auto Matrix<K, Row, Col, Backend>::operator+=(const Matrix& x) -> Matrix& {
    if constexpr (mk_concepts::SingleFloatingPoint<K>) {
        cblas_saxpy(Row * Col, 1.0, x._elements.data(), 1, this->_elements.data(), 1);
    } else if constexpr (mk_concepts::DoubleFloatingPoint<K>) {
        cblas_daxpy(Row * Col, 1.0, x._elements.data(), 1, this->_elements.data(), 1);
    }
    return *this;
}

template <typename K, size_t Row, size_t Col, typename Backend>
    requires mk_concepts::BLASSupported<Backend, K>
inline auto Matrix<K, Row, Col, Backend>::operator+=(const Matrix<K, Row, std::dynamic_extent, Backend>& x) -> Matrix& {
#if !defined(NDEBUG)
    if (Col != x.ColumnSize()) {
        std::string errorString = "Mismatch matrix size for matrix product.\n";
        errorString += "this size    : " + this->GetSizeString() + "\n";
        errorString += "opponent size: " + x.GetSizeString() + ".\n";

        throw std::domain_error(errorString);
    }
#endif

    if constexpr (mk_concepts::SingleFloatingPoint<K>) {
        cblas_saxpy(Row * Col, 1.0, x._elements.data(), 1, this->_elements.data(), 1);
    } else if constexpr (mk_concepts::DoubleFloatingPoint<K>) {
        cblas_daxpy(Row * Col, 1.0, x._elements.data(), 1, this->_elements.data(), 1);
    }
    return *this;
}
}  // namespace mknnlib
#endif