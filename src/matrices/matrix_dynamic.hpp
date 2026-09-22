#ifndef MKNNLIB_MATRICES_MATRIX_DYNAMIC_HPP
#define MKNNLIB_MATRICES_MATRIX_DYNAMIC_HPP

#include <cblas.h>
#include <openblas_config.h>

#include <algorithm>
#include <boost/operators.hpp>
#include <cassert>
#include <cstddef>
#include <mdspan>
#include <ostream>
#include <span>
#include <vector>

#include "src/concept_defines/types/type_concepts.hpp"
#include "src/matrices/core/blas_dispatchers/blas_primary_template.hpp"
#include "src/matrices/core/blas_storages/blas_storage.hpp"  // IWYU pragma: keep
#include "src/matrices/matrix_template_base.hpp"
#include "src/matrices/forward_declarations/dynamic_operations.hpp"

namespace mknnlib::matrix {
template <typename K, typename Backend, typename Layout>
    requires mk_concepts::BLASSupported<Backend, K>
class Matrix<K, std::dynamic_extent, std::dynamic_extent, Backend, Layout>;

template <typename K, typename Backend, typename Layout>
    requires mk_concepts::BLASSupported<Backend, K>
std::ostream& operator<<(std::ostream& os, const Matrix<K, std::dynamic_extent, std::dynamic_extent, Backend, Layout>& mat);

template <typename K, size_t Row, size_t Col, typename Backend, typename Layout>
    requires mk_concepts::BLASSupported<Backend, K>
class Matrix;

template <typename K, typename Backend, typename Layout>
    requires mk_concepts::BLASSupported<Backend, K>
class Matrix<K, std::dynamic_extent, std::dynamic_extent, Backend, Layout> :
    private boost::subtractable<Matrix<K, std::dynamic_extent, std::dynamic_extent, Backend, Layout>>,
    private boost::multipliable<Matrix<K, std::dynamic_extent, std::dynamic_extent, Backend, Layout>> {
    template <typename T, size_t FRow, size_t FCol, typename FBackend, typename FLayout>
        requires mk_concepts::BLASSupported<FBackend, T>
    friend class Matrix;
    // type alias
    using MatrixExtent = std::extents<size_t, std::dynamic_extent, std::dynamic_extent>;
    using MdView = std::mdspan<K, MatrixExtent, std::layout_left>;

private:
    size_t _rowSize = 0;
    size_t _columnSize = 0;
    core::Storage<Backend, K, std::dynamic_extent> _elements;
    MdView _span;

public:
    // TODO: ドキュメントをちゃんと書く
    /* begin constructors declaration */
    Matrix<K, std::dynamic_extent, std::dynamic_extent, Backend, Layout>();
    Matrix<K, std::dynamic_extent, std::dynamic_extent, Backend, Layout>(const size_t RowSize, const size_t columnSize);
    Matrix<K, std::dynamic_extent, std::dynamic_extent, Backend, Layout>(
        const size_t RowSize, const size_t columnSize, const std::vector<K> elements);
    // copy constructor
    Matrix<K, std::dynamic_extent, std::dynamic_extent, Backend, Layout>(
        const Matrix<K, std::dynamic_extent, std::dynamic_extent, Backend, Layout>& mat);
    // move constructor
    Matrix(Matrix&&) = default;
    /* end constructors declaration */

    /* begin operator overloads declaration */
    // copy assignment operator
    Matrix<K, std::dynamic_extent, std::dynamic_extent, Backend, Layout>& operator=(
        const Matrix<K, std::dynamic_extent, std::dynamic_extent, Backend, Layout>& x);
    // move assignment operator
    Matrix& operator=(Matrix&&) = default;
    // arithmetics compound operators
    Matrix& operator+=(const Matrix& x);
    Matrix& operator-=(const Matrix& x);
    Matrix& operator*=(const Matrix& x);

    // arithmetics binary operators
    // friend Matrix operator+(const Matrix& lhs, const Matrix& rhs);

    // 2 dimensions index
    K& operator[](const size_t a, const size_t b);

    // ostream
    friend std::ostream& operator<< <>(
        std::ostream& os, const Matrix<K, std::dynamic_extent, std::dynamic_extent, Backend, Layout>& mat);
    /* end operator overloads declaration */

    /* begin matrix unique functions declaration */
    size_t RowSize();
    size_t RowSize() const;
    size_t ColumnSize();
    size_t ColumnSize() const;
    std::string GetSizeString() const;
    core::Storage<Backend, K, std::dynamic_extent>& Elements();
    constexpr std::vector<K>& ElementsRange() noexcept;
    constexpr const std::vector<K>& ElementsRange() const noexcept;
    // K* _elements.data();
    // const K* _elements.data() const;
    /* end matrix unique functions declaration */

    /* begin matrix unique arithmetics declaration */
    Matrix Dot(const Matrix mat);

    template <size_t OppRow, size_t OppCol>
    Matrix<K, std::dynamic_extent, OppCol, Backend, Layout> Dot(const Matrix<K, OppRow, OppCol, Backend, Layout>);

    template <size_t OppRow>
    Matrix<K, std::dynamic_extent, std::dynamic_extent, Backend, Layout> Dot(
        const Matrix<K, OppRow, std::dynamic_extent, Backend, Layout>);

    template <size_t OppCol>
    Matrix<K, std::dynamic_extent, OppCol, Backend, Layout> Dot(const Matrix<K, std::dynamic_extent, OppCol, Backend, Layout>);
    /* end matrix unique arithmetics declaration */
};

/* begin constructors definition */
template <typename K, typename Backend, typename Layout>
    requires mk_concepts::BLASSupported<Backend, K>
Matrix<K, std::dynamic_extent, std::dynamic_extent, Backend, Layout>::Matrix() :
    _rowSize(0), _columnSize(0), _elements{}, _span(_elements.data(), MatrixExtent{}) {}

template <typename K, typename Backend, typename Layout>
    requires mk_concepts::BLASSupported<Backend, K>
Matrix<K, std::dynamic_extent, std::dynamic_extent, Backend, Layout>::Matrix(const size_t rowSize, const size_t columnSize) :
    _rowSize(rowSize),
    _columnSize(columnSize),
    _elements(rowSize * columnSize),
    _span(_elements.data(), MatrixExtent{}) {}

template <typename K, typename Backend, typename Layout>
    requires mk_concepts::BLASSupported<Backend, K>
Matrix<K, std::dynamic_extent, std::dynamic_extent, Backend, Layout>::Matrix(
    const size_t rowSize, const size_t columnSize, const std::vector<K> elements) :
    _rowSize(rowSize), _columnSize(columnSize), _elements(elements), _span(_elements.data(), MatrixExtent{}) {}

// copy constructor
template <typename K, typename Backend, typename Layout>
    requires mk_concepts::BLASSupported<Backend, K>
Matrix<K, std::dynamic_extent, std::dynamic_extent, Backend, Layout>::Matrix(
    const Matrix<K, std::dynamic_extent, std::dynamic_extent, Backend, Layout>& mat) :
    _rowSize(mat.RowSize()),
    _columnSize(mat.ColumnSize()),
    _elements(mat._elements),
    _span(_elements.data(), MatrixExtent{}) {}
/* end constructors definition */

// arithmetics
template <typename K, typename Backend, typename Layout>
    requires mk_concepts::BLASSupported<Backend, K>
inline Matrix<K, std::dynamic_extent, std::dynamic_extent, Backend, Layout>&
Matrix<K, std::dynamic_extent, std::dynamic_extent, Backend, Layout>::operator+=(
    const Matrix<K, std::dynamic_extent, std::dynamic_extent, Backend, Layout>& x) {
#if !defined(NDEBUG)
    if (this->RowSize() != x.RowSize() || this->ColumnSize() != x.ColumnSize()) {
        std::string errorString = "Mismatch matrix size for matrix product.\n";
        errorString += "this size    : " + this->GetSizeString() + "\n";
        errorString += "opponent size: " + x.GetSizeString() + ".\n";

        throw std::domain_error(errorString);
    }
#endif
    core::BLAS<Backend, K>::axpy(static_cast<blasint>(this->_rowSize * this->_columnSize), 1.0, x._elements.data(), 1,
        this->_elements.data(), 1);
    return *this;
}

template <typename K, typename Backend, typename Layout>
    requires mk_concepts::BLASSupported<Backend, K>
inline Matrix<K, std::dynamic_extent, std::dynamic_extent, Backend, Layout>&
Matrix<K, std::dynamic_extent, std::dynamic_extent, Backend, Layout>::operator-=(
    const Matrix<K, std::dynamic_extent, std::dynamic_extent, Backend, Layout>& x) {
#if !defined(NDEBUG)
    if (this->RowSize() != x.RowSize() || this->ColumnSize() != x.ColumnSize()) {
        std::string errorString = "Mismatch matrix size for matrix product.\n";
        errorString += "this size    : " + this->GetSizeString() + "\n";
        errorString += "opponent size: " + x.GetSizeString() + ".\n";

        throw std::domain_error(errorString);
    }
#endif
    core::BLAS<Backend, K>::axpy(static_cast<blasint>(this->_rowSize * this->_columnSize), -1.0, x._elements.data(), 1,
        this->_elements.data(), 1);
    return *this;
}

template <typename K, typename Backend, typename Layout>
    requires mk_concepts::BLASSupported<Backend, K>
Matrix<K, std::dynamic_extent, std::dynamic_extent, Backend, Layout>&
Matrix<K, std::dynamic_extent, std::dynamic_extent, Backend, Layout>::operator*=(
    const Matrix<K, std::dynamic_extent, std::dynamic_extent, Backend, Layout>& x) {
#if !defined(NDEBUG)
    if (this->RowSize() != x.RowSize() || this->ColumnSize() != x.ColumnSize()) {
        std::string errorString = "Mismatch matrix size for matrix product.\n";
        errorString += "this size    : " + this->GetSizeString() + "\n";
        errorString += "opponent size: " + x.GetSizeString() + ".\n";

        throw std::domain_error(errorString);
    }
#endif
    // hadamard product is not into BLAS
    std::ranges::transform(
        this->_elements.elements(), x._elements.elements(), this->_elements.begin(), std::multiplies<>());
    return *this;
}

// arithmetics binary operators
// Dynamic + Dynamic
template <typename K, typename Backend, typename Layout>
    requires mk_concepts::BLASSupported<Backend, K>
inline Matrix<K, std::dynamic_extent, std::dynamic_extent, Backend, Layout> operator+(
    const Matrix<K, std::dynamic_extent, std::dynamic_extent, Backend, Layout>& lhs,
    const Matrix<K, std::dynamic_extent, std::dynamic_extent, Backend, Layout>& rhs) {
#if !defined(NDEBUG)
    if (lhs.RowSize() != rhs.RowSize() || lhs.ColumnSize() != rhs.ColumnSize()) {
        std::string errorString = "Mismatch matrix size for matrix product.\n";
        errorString += "lhs size: " + lhs.GetSizeString() + "\n";
        errorString += "rhs size: " + rhs.GetSizeString() + ".\n";

        throw std::domain_error(errorString);
    }
#endif
    auto result = lhs;
    result += rhs;
    return result;
}

// ostream
template <typename K, typename Backend, typename Layout>
    requires mk_concepts::BLASSupported<Backend, K>
std::ostream& operator<<(std::ostream& os, const Matrix<K, std::dynamic_extent, std::dynamic_extent, Backend, Layout>& mat) {
    for (size_t i = 0; i < mat.RowSize(); i++) {
        for (size_t j = 0; j < mat.ColumnSize(); j++) {
            os << mat._span[i, j] << " ";
        }
        os << std::endl;
    }
    return os;
}
/* end operator overloads definition */

/* begin matrix unique functions definition */
template <typename K, typename Backend, typename Layout>
    requires mk_concepts::BLASSupported<Backend, K>
size_t Matrix<K, std::dynamic_extent, std::dynamic_extent, Backend, Layout>::RowSize() {
    return this->_rowSize;
}

template <typename K, typename Backend, typename Layout>
    requires mk_concepts::BLASSupported<Backend, K>
size_t Matrix<K, std::dynamic_extent, std::dynamic_extent, Backend, Layout>::RowSize() const {
    return this->_rowSize;
}

template <typename K, typename Backend, typename Layout>
    requires mk_concepts::BLASSupported<Backend, K>
size_t Matrix<K, std::dynamic_extent, std::dynamic_extent, Backend, Layout>::ColumnSize() {
    return this->_columnSize;
}

template <typename K, typename Backend, typename Layout>
    requires mk_concepts::BLASSupported<Backend, K>
size_t Matrix<K, std::dynamic_extent, std::dynamic_extent, Backend, Layout>::ColumnSize() const {
    return this->_columnSize;
}

template <typename K, typename Backend, typename Layout>
    requires mk_concepts::BLASSupported<Backend, K>
std::string Matrix<K, std::dynamic_extent, std::dynamic_extent, Backend, Layout>::GetSizeString() const {
    return "(" + std::to_string(this->RowSize()) + ", " + std::to_string(this->ColumnSize()) + ")";
}

template <typename K, typename Backend, typename Layout>
    requires mk_concepts::BLASSupported<Backend, K>
inline core::Storage<Backend, K, std::dynamic_extent>&
Matrix<K, std::dynamic_extent, std::dynamic_extent, Backend, Layout>::Elements() {
    return this->_elements;
}

template <typename K, typename Backend, typename Layout>
    requires mk_concepts::BLASSupported<Backend, K>
inline constexpr std::vector<K>&
Matrix<K, std::dynamic_extent, std::dynamic_extent, Backend, Layout>::ElementsRange() noexcept {
    return this->_elements.elements();
}

template <typename K, typename Backend, typename Layout>
    requires mk_concepts::BLASSupported<Backend, K>
inline constexpr const std::vector<K>& Matrix<K, std::dynamic_extent, std::dynamic_extent, Backend, Layout>::ElementsRange()
    const noexcept {
    return this->_elements.elements();
}
/* end matrix unique functions definition */

/* begin matrix unique arithmetics definition */
template <typename K, typename Backend, typename Layout>
    requires mk_concepts::BLASSupported<Backend, K>
Matrix<K, std::dynamic_extent, std::dynamic_extent, Backend, Layout>
Matrix<K, std::dynamic_extent, std::dynamic_extent, Backend, Layout>::Dot(
    Matrix<K, std::dynamic_extent, std::dynamic_extent, Backend, Layout> mat) {
#if !defined(NDEBUG)
    if (this->ColumnSize() != mat.RowSize()) {
        std::string errorString = "Mismatch matrix size for matrix product.\n";
        errorString += "this size    : " + this->GetSizeString() + "\n";
        errorString += "opponent size: " + mat.GetSizeString() + ".\n";

        throw std::domain_error(errorString);
    }
#endif
    auto res = Matrix<K, std::dynamic_extent, std::dynamic_extent, Backend, Layout>(this->RowSize(), mat.ColumnSize());
    if constexpr (mk_concepts::SingleFloatingPoint<K>) {
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, static_cast<blasint>(this->RowSize()),
            static_cast<blasint>(mat.ColumnSize()), static_cast<blasint>(this->ColumnSize()), 1.0,
            this->_elements.data(), static_cast<blasint>(this->RowSize()), mat._elements.data(),
            static_cast<blasint>(mat.RowSize()), 0.0, res._elements.data(), static_cast<blasint>(this->RowSize()));
    } else if (mk_concepts::DoubleFloatingPoint<K>) {
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, static_cast<blasint>(this->RowSize()),
            static_cast<blasint>(mat.ColumnSize()), static_cast<blasint>(this->ColumnSize()), 1.0,
            this->_elements.data(), static_cast<blasint>(this->RowSize()), mat._elements.data(),
            static_cast<blasint>(mat.RowSize()), 0.0, res._elements.data(), static_cast<blasint>(this->RowSize()));
    }

    return res;
}

template <typename K, typename Backend, typename Layout>
    requires mk_concepts::BLASSupported<Backend, K>
template <size_t OppRow, size_t OppCol>
Matrix<K, std::dynamic_extent, OppCol, Backend, Layout> Matrix<K, std::dynamic_extent, std::dynamic_extent, Backend, Layout>::Dot(
    Matrix<K, OppRow, OppCol, Backend, Layout> mat) {
#if !defined(NDEBUG)
    if (this->ColumnSize() != OppRow) {
        std::string errorString = "Mismatch matrix size for matrix product.\n";
        errorString += "this size    : " + this->GetSizeString() + "\n";
        errorString += "opponent size: " + mat.GetSizeString() + ".\n";

        throw std::domain_error(errorString);
    }
#endif
    auto res = MatrixColumnStaticOpen<K, OppCol>(this->RowSize());
    if constexpr (mk_concepts::SingleFloatingPoint<K>) {
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, static_cast<blasint>(this->RowSize()), OppCol,
            static_cast<blasint>(this->ColumnSize()), 1.0, this->_elements.data(),
            static_cast<blasint>(this->RowSize()), mat._elements.data(), OppRow, 0.0, res._elements.data(),
            static_cast<blasint>(this->RowSize()));
    } else if (mk_concepts::DoubleFloatingPoint<K>) {
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, static_cast<blasint>(this->RowSize()), OppCol,
            static_cast<blasint>(this->ColumnSize()), 1.0, this->_elements.data(),
            static_cast<blasint>(this->RowSize()), mat._elements.data(), OppRow, 0.0, res._elements.data(),
            static_cast<blasint>(this->RowSize()));
    }

    return res;
}

template <typename K, typename Backend, typename Layout>
    requires mk_concepts::BLASSupported<Backend, K>
template <size_t OppRow>
Matrix<K, std::dynamic_extent, std::dynamic_extent, Backend, Layout>
Matrix<K, std::dynamic_extent, std::dynamic_extent, Backend, Layout>::Dot(Matrix<K, OppRow, std::dynamic_extent, Backend, Layout> mat) {
#if !defined(NDEBUG)
    if (this->ColumnSize() != OppRow) {
        std::string errorString = "Mismatch matrix size for matrix product.\n";
        errorString += "this size    : " + this->GetSizeString() + "\n";
        errorString += "opponent size: " + mat.GetSizeString() + ".\n";

        throw std::domain_error(errorString);
    }
#endif
    auto res = Matrix<K, std::dynamic_extent, std::dynamic_extent, Backend, Layout>(this->RowSize(), mat.ColumnSize());
    if constexpr (mk_concepts::SingleFloatingPoint<K>) {
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, static_cast<blasint>(this->RowSize()),
            static_cast<blasint>(mat.ColumnSize()), static_cast<blasint>(this->ColumnSize()), 1.0,
            this->_elements.data(), static_cast<blasint>(this->RowSize()), mat._elements.data(), OppRow, 0.0,
            res._elements.data(), static_cast<blasint>(this->RowSize()));
    } else if (mk_concepts::DoubleFloatingPoint<K>) {
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, static_cast<blasint>(this->RowSize()),
            static_cast<blasint>(mat.ColumnSize()), static_cast<blasint>(this->ColumnSize()), 1.0,
            this->_elements.data(), static_cast<blasint>(this->RowSize()), mat._elements.data(), OppRow, 0.0,
            res._elements.data(), static_cast<blasint>(this->RowSize()));
    }

    return res;
}

template <typename K, typename Backend, typename Layout>
    requires mk_concepts::BLASSupported<Backend, K>
template <size_t OppCol>
Matrix<K, std::dynamic_extent, OppCol, Backend, Layout> Matrix<K, std::dynamic_extent, std::dynamic_extent, Backend, Layout>::Dot(
    Matrix<K, std::dynamic_extent, OppCol, Backend, Layout> mat) {
#if !defined(NDEBUG)
    if (this->ColumnSize() != mat.RowSize()) {
        std::string errorString = "Mismatch matrix size for matrix product.\n";
        errorString += "this size    : " + this->GetSizeString() + "\n";
        errorString += "opponent size: " + mat.GetSizeString() + ".\n";

        throw std::domain_error(errorString);
    }
#endif
    auto res = MatrixColumnStaticOpen<K, OppCol>(this->RowSize());
    if constexpr (mk_concepts::SingleFloatingPoint<K>) {
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, static_cast<blasint>(this->RowSize()), OppCol,
            static_cast<blasint>(this->ColumnSize()), 1.0, this->_elements.data(),
            static_cast<blasint>(this->RowSize()), mat._elements.data(), static_cast<blasint>(mat.RowSize()), 0.0,
            res._elements.data(), static_cast<blasint>(this->RowSize()));
    } else if (mk_concepts::DoubleFloatingPoint<K>) {
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, static_cast<blasint>(this->RowSize()), OppCol,
            static_cast<blasint>(this->ColumnSize()), 1.0, this->_elements.data(),
            static_cast<blasint>(this->RowSize()), mat._elements.data(), static_cast<blasint>(mat.RowSize()), 0.0,
            res._elements.data(), static_cast<blasint>(this->RowSize()));
    }

    return res;
}
/* end matrix unique arithmetics definition */
}  // namespace mknnlib::matrix

#endif