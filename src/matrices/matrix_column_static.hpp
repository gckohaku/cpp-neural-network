#ifndef MKNNLIB_MATRICES_MATRIX_COLUMN_STATIC_HPP
#define MKNNLIB_MATRICES_MATRIX_COLUMN_STATIC_HPP

#include <cblas.h>

#include <algorithm>
#include <boost/operators.hpp>
#include <cstddef>
#include <functional>
#include <mdspan>
#include <ostream>
#include <span>

#include "src/concept_defines/types/type_concepts.hpp"
#include "src/matrices/matrix_static.hpp"
#include "src/matrices/matrix_template_base.hpp"
#include "src/matrices/core/blas_storages/blas_storage.hpp" // IWYU pragma: keep

namespace mknnlib::matrix {
template <typename K, size_t Col, typename Backend>
    requires mk_concepts::BLASSupported<Backend, K>
class Matrix<K, std::dynamic_extent, Col, Backend>;
template <typename K, size_t Col, typename Backend>
    requires mk_concepts::BLASSupported<Backend, K>
std::ostream& operator<<(std::ostream& os, const Matrix<K, std::dynamic_extent, Col, Backend>& mat);

template <typename K, size_t Row, size_t Col, typename Backend>
    requires mk_concepts::BLASSupported<Backend, K>
class Matrix;

template <typename K, size_t Col, typename Backend>
    requires mk_concepts::BLASSupported<Backend, K>
class Matrix<K, std::dynamic_extent, Col, Backend> :
    private boost::addable<Matrix<K, std::dynamic_extent, Col, Backend>>,
    private boost::subtractable<Matrix<K, std::dynamic_extent, Col, Backend>>,
    private boost::multipliable<Matrix<K, std::dynamic_extent, Col, Backend>> {
    static_assert(Col != std::dynamic_extent);

    template <typename T, size_t FRow, size_t FCol, typename FBackend>
        requires mk_concepts::BLASSupported<FBackend, T>
    friend class Matrix;
    // type alias
    using MatrixExtent = std::extents<size_t, std::dynamic_extent, Col>;
    using MdView = std::mdspan<K, MatrixExtent, std::layout_left>;

private:
    size_t _rowSize = 0;
    size_t _columnSize = Col;
    core::Storage<Backend, K, std::dynamic_extent> _elements;
    MdView _span;

public:
    // TODO: ドキュメントをちゃんと書く
    /* begin constructors declaration */
    Matrix<K, std::dynamic_extent, Col, Backend>();
    Matrix<K, std::dynamic_extent, Col, Backend>(const size_t rowSize);
    Matrix<K, std::dynamic_extent, Col, Backend>(const size_t rowSize, const std::vector<K> elements);
    // copy constructor
    Matrix<K, std::dynamic_extent, Col, Backend>(const Matrix<K, std::dynamic_extent, Col, Backend>& mat);
    /* end constructors declaration */

    /* begin operator overloads declaration */
    // copy assignment operator
    Matrix<K, std::dynamic_extent, Col, Backend>& operator=(const Matrix<K, std::dynamic_extent, Col, Backend>& x);
    // arithmetics
    Matrix<K, std::dynamic_extent, Col, Backend>& operator+=(const Matrix<K, std::dynamic_extent, Col, Backend>& x)
        requires mk_concepts::SingleFloatingPoint<K>;
    Matrix<K, std::dynamic_extent, Col, Backend>& operator+=(const Matrix<K, std::dynamic_extent, Col, Backend>& x)
        requires mk_concepts::DoubleFloatingPoint<K>;
    Matrix<K, std::dynamic_extent, Col, Backend>& operator-=(const Matrix<K, std::dynamic_extent, Col, Backend>& x)
        requires mk_concepts::SingleFloatingPoint<K>;
    Matrix<K, std::dynamic_extent, Col, Backend>& operator-=(const Matrix<K, std::dynamic_extent, Col, Backend>& x)
        requires mk_concepts::DoubleFloatingPoint<K>;
    Matrix<K, std::dynamic_extent, Col, Backend>& operator*=(const Matrix<K, std::dynamic_extent, Col, Backend>& x);

    // 2 dimensions index
    K& operator[](const size_t a, const size_t b);

    // ostream
    friend std::ostream& operator<< <>(std::ostream& os, const Matrix<K, std::dynamic_extent, Col, Backend>& mat);
    /* end operator overloads declaration */

    /* begin matrix unique functions declaration */
    size_t RowSize();
    size_t RowSize() const;
    constexpr size_t ColumnSize();
    constexpr size_t ColumnSize() const;
    std::string GetSizeString() const;
    core::Storage<Backend, K, std::dynamic_extent>& Elements();
    // K* _elements.data();
    // const K* _elements.data() const;
    /* end matrix unique functions declaration */

    /* begin matrix unique arithmetics declaration */
    template <size_t OppCol>
    Matrix<K, std::dynamic_extent, OppCol, Backend> Dot(const Matrix<K, std::dynamic_extent, OppCol, Backend> mat);

    template <size_t OppRow, size_t OppCol>
    Matrix<K, std::dynamic_extent, OppCol, Backend> Dot(const Matrix<K, OppRow, OppCol, Backend> mat);

    template <size_t OppRow>
    Matrix<K, std::dynamic_extent, std::dynamic_extent, Backend> Dot(
        const Matrix<K, OppRow, std::dynamic_extent, Backend> mat);

    Matrix<K, std::dynamic_extent, std::dynamic_extent, Backend> Dot(
        const Matrix<K, std::dynamic_extent, std::dynamic_extent, Backend> mat);
    /* end matrix unique arithmetics declaration */
};

/* begin constructors definition */
template <typename K, size_t Col, typename Backend>
    requires mk_concepts::BLASSupported<Backend, K>
Matrix<K, std::dynamic_extent, Col, Backend>::Matrix() :
    _rowSize(0), _columnSize(Col), _elements{}, _span(_elements.data(), MatrixExtent{}) {}

template <typename K, size_t Col, typename Backend>
    requires mk_concepts::BLASSupported<Backend, K>
Matrix<K, std::dynamic_extent, Col, Backend>::Matrix(const size_t rowSize) :
    _rowSize(rowSize), _columnSize(Col), _elements(rowSize * Col, K{}), _span(_elements.data(), MatrixExtent{}) {}

template <typename K, size_t Col, typename Backend>
    requires mk_concepts::BLASSupported<Backend, K>
Matrix<K, std::dynamic_extent, Col, Backend>::Matrix(const size_t rowSize, const std::vector<K> elements) :
    _rowSize(rowSize), _columnSize(Col), _elements{elements}, _span(_elements.data(), MatrixExtent{}) {}

// copy constructor
template <typename K, size_t Col, typename Backend>
    requires mk_concepts::BLASSupported<Backend, K>
Matrix<K, std::dynamic_extent, Col, Backend>::Matrix(const Matrix<K, std::dynamic_extent, Col, Backend>& mat) :
    _rowSize(mat.RowSize()), _columnSize(Col), _elements(mat._elements), _span(_elements.data(), MatrixExtent{}) {}
/* end constructors definition */

/* begin operator overloads definition */
// copy assignment operator
template <typename K, size_t Col, typename Backend>
    requires mk_concepts::BLASSupported<Backend, K>
Matrix<K, std::dynamic_extent, Col, Backend>& Matrix<K, std::dynamic_extent, Col, Backend>::operator=(
    const Matrix<K, std::dynamic_extent, Col, Backend>& x) {
    if (this != &x) {
        _elements = x._elements;
        _span = MdView(_elements, MatrixExtent{_rowSize});
    }
    return *this;
}

// arithmetics
template <typename K, size_t Col, typename Backend>
    requires mk_concepts::BLASSupported<Backend, K>
inline Matrix<K, std::dynamic_extent, Col, Backend>& Matrix<K, std::dynamic_extent, Col, Backend>::operator+=(
    const Matrix<K, std::dynamic_extent, Col, Backend>& x)
    requires mk_concepts::SingleFloatingPoint<K>
{
#if !defined(NDEBUG)
    if (this->RowSize() != x.RowSize()) {
        std::string errorString = "Mismatch matrix size for matrix product.\n";
        errorString += "this size    : " + this->GetSizeString() + "\n";
        errorString += "opponent size: " + x.GetSizeString() + ".\n";

        throw std::domain_error(errorString);
    }
#endif
    cblas_saxpy(static_cast<blasint>(this->_rowSize * Col), 1.0, x._elements.data(), 1, this->_elements.data(), 1);
    return *this;
}

template <typename K, size_t Col, typename Backend>
    requires mk_concepts::BLASSupported<Backend, K>
inline Matrix<K, std::dynamic_extent, Col, Backend>& Matrix<K, std::dynamic_extent, Col, Backend>::operator+=(
    const Matrix<K, std::dynamic_extent, Col, Backend>& x)
    requires mk_concepts::DoubleFloatingPoint<K>
{
#if !defined(NDEBUG)
    if (this->RowSize() != x.RowSize()) {
        std::string errorString = "Mismatch matrix size for matrix product.\n";
        errorString += "this size    : " + this->GetSizeString() + "\n";
        errorString += "opponent size: " + x.GetSizeString() + ".\n";

        throw std::domain_error(errorString);
    }
#endif
    cblas_daxpy(static_cast<blasint>(this->_rowSize * Col), 1.0, x._elements.data(), 1, this->_elements.data(), 1);
    return *this;
}

template <typename K, size_t Col, typename Backend>
    requires mk_concepts::BLASSupported<Backend, K>
inline Matrix<K, std::dynamic_extent, Col, Backend>& Matrix<K, std::dynamic_extent, Col, Backend>::operator-=(
    const Matrix<K, std::dynamic_extent, Col, Backend>& x)
    requires mk_concepts::SingleFloatingPoint<K>
{
#if !defined(NDEBUG)
    if (this->RowSize() != x.RowSize()) {
        std::string errorString = "Mismatch matrix size for matrix product.\n";
        errorString += "this size    : " + this->GetSizeString() + "\n";
        errorString += "opponent size: " + x.GetSizeString() + ".\n";

        throw std::domain_error(errorString);
    }
#endif
    cblas_saxpy(static_cast<blasint>(this->_rowSize * Col), -1.0, x._elements.data(), 1, this->_elements.data(), 1);
    return *this;
}

template <typename K, size_t Col, typename Backend>
    requires mk_concepts::BLASSupported<Backend, K>
inline Matrix<K, std::dynamic_extent, Col, Backend>& Matrix<K, std::dynamic_extent, Col, Backend>::operator-=(
    const Matrix<K, std::dynamic_extent, Col, Backend>& x)
    requires mk_concepts::DoubleFloatingPoint<K>
{
#if !defined(NDEBUG)
    if (this->RowSize() != x.RowSize()) {
        std::string errorString = "Mismatch matrix size for matrix product.\n";
        errorString += "this size    : " + this->GetSizeString() + "\n";
        errorString += "opponent size: " + x.GetSizeString() + ".\n";

        throw std::domain_error(errorString);
    }
#endif
    cblas_daxpy(static_cast<blasint>(this->_rowSize * Col), -1.0, x._elements.data(), 1, this->_elements.data(), 1);
    return *this;
}

template <typename K, size_t Col, typename Backend>
    requires mk_concepts::BLASSupported<Backend, K>
Matrix<K, std::dynamic_extent, Col, Backend>& Matrix<K, std::dynamic_extent, Col, Backend>::operator*=(
    const Matrix<K, std::dynamic_extent, Col, Backend>& x) {
#if !defined(NDEBUG)
    if (this->RowSize() != x.RowSize()) {
        std::string errorString = "Mismatch matrix size for matrix product.\n";
        errorString += "this size    : " + this->GetSizeString() + "\n";
        errorString += "opponent size: " + x.GetSizeString() + ".\n";

        throw std::domain_error(errorString);
    }
#endif
    // hadamard product is not into BLAS
    std::ranges::transform(this->_elements, x._elements, this->_elements.begin(), std::multiplies<>());
    return *this;
}

// ostream
template <typename K, size_t Col, typename Backend>
    requires mk_concepts::BLASSupported<Backend, K>
std::ostream& operator<<(std::ostream& os, const Matrix<K, std::dynamic_extent, Col, Backend>& mat) {
    for (size_t i = 0; i < mat._rowSize; i++) {
        for (size_t j = 0; j < Col; j++) {
            os << mat._span[i, j] << " ";
        }
        os << std::endl;
    }
    return os;
}

/* end operator overloads definition */

/* begin matrix unique functions definition */
template <typename K, size_t Col, typename Backend>
    requires mk_concepts::BLASSupported<Backend, K>
size_t Matrix<K, std::dynamic_extent, Col, Backend>::RowSize() {
    return this->_rowSize;
}

template <typename K, size_t Col, typename Backend>
    requires mk_concepts::BLASSupported<Backend, K>
size_t Matrix<K, std::dynamic_extent, Col, Backend>::RowSize() const {
    return this->_rowSize;
}

template <typename K, size_t Col, typename Backend>
    requires mk_concepts::BLASSupported<Backend, K>
constexpr size_t Matrix<K, std::dynamic_extent, Col, Backend>::ColumnSize() {
    return Col;
}

template <typename K, size_t Col, typename Backend>
    requires mk_concepts::BLASSupported<Backend, K>
constexpr size_t Matrix<K, std::dynamic_extent, Col, Backend>::ColumnSize() const {
    return Col;
}

template <typename K, size_t Col, typename Backend>
    requires mk_concepts::BLASSupported<Backend, K>
std::string Matrix<K, std::dynamic_extent, Col, Backend>::GetSizeString() const {
    return "(" + std::to_string(this->RowSize()) + ", " + std::to_string(Col) + ")";
}

template <typename K, size_t Col, typename Backend>
    requires mk_concepts::BLASSupported<Backend, K>
inline core::Storage<Backend, K, std::dynamic_extent>& Matrix<K, std::dynamic_extent, Col, Backend>::Elements() {
    return this->_elements;
}

// template <typename K, size_t Col, typename Backend>
// requires mk_concepts::BLASSupported<Backend, K>
// inline K* Matrix<K, std::dynamic_extent, Col, Backend>::_elements.data() {
//     return this->_elements.data();
// }

// template <typename K, size_t Col, typename Backend>
// requires mk_concepts::BLASSupported<Backend, K>
// inline const K* Matrix<K, std::dynamic_extent, Col, Backend>::_elements.data() const {
//     return this->_elements.data();
// }
/* end matrix unique functions definition */

/* begin matrix unique arithmetics definition */
template <typename K, size_t Col, typename Backend>
    requires mk_concepts::BLASSupported<Backend, K>
template <size_t OppCol>
Matrix<K, std::dynamic_extent, OppCol, Backend> Matrix<K, std::dynamic_extent, Col, Backend>::Dot(
    Matrix<K, std::dynamic_extent, OppCol, Backend> mat) {
#if !defined(NDEBUG)
    if (Col != mat.RowSize()) {
        std::string errorString = "Mismatch matrix size for matrix product.\n";
        errorString += "this size    : " + this->GetSizeString() + "\n";
        errorString += "opponent size: " + mat.GetSizeString() + ".\n";

        throw std::domain_error(errorString);
    }
#endif
    auto res = Matrix<K, std::dynamic_extent, OppCol>(this->RowSize());
    if constexpr (mk_concepts::SingleFloatingPoint<K>) {
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, static_cast<blasint>(this->RowSize()), OppCol, Col, 1.0,
            this->_elements.data(), static_cast<blasint>(this->RowSize()), mat._elements.data(),
            static_cast<blasint>(mat.RowSize()), 0.0, res._elements.data(), static_cast<blasint>(this->RowSize()));
    } else if (mk_concepts::DoubleFloatingPoint<K>) {
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, static_cast<blasint>(this->RowSize()), OppCol, Col, 1.0,
            this->_elements.data(), static_cast<blasint>(this->RowSize()), mat._elements.data(),
            static_cast<blasint>(mat.RowSize()), 0.0, res._elements.data(), static_cast<blasint>(this->RowSize()));
    }

    return res;
}

template <typename K, size_t Col, typename Backend>
    requires mk_concepts::BLASSupported<Backend, K>
template <size_t OppRow, size_t OppCol>
Matrix<K, std::dynamic_extent, OppCol, Backend> Matrix<K, std::dynamic_extent, Col, Backend>::Dot(
    Matrix<K, OppRow, OppCol, Backend> mat) {
    auto res = Matrix<K, std::dynamic_extent, OppCol>(this->RowSize());
    if constexpr (mk_concepts::SingleFloatingPoint<K>) {
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, static_cast<blasint>(this->RowSize()), OppCol, Col, 1.0,
            this->_elements.data(), static_cast<blasint>(this->RowSize()), mat._elements.data(), OppRow, 0.0,
            res._elements.data(), static_cast<blasint>(this->RowSize()));
    } else if (mk_concepts::DoubleFloatingPoint<K>) {
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, static_cast<blasint>(this->RowSize()), OppCol, Col, 1.0,
            this->_elements.data(), static_cast<blasint>(this->RowSize()), mat._elements.data(), OppRow, 0.0,
            res._elements.data(), static_cast<blasint>(this->RowSize()));
    }

    return res;
}

template <typename K, size_t Col, typename Backend>
    requires mk_concepts::BLASSupported<Backend, K>
template <size_t OppRow>
Matrix<K, std::dynamic_extent, std::dynamic_extent, Backend> Matrix<K, std::dynamic_extent, Col, Backend>::Dot(
    Matrix<K, OppRow, std::dynamic_extent, Backend> mat) {
    auto res = MatrixDynamicOpen<K>(this->RowSize(), mat.ColumnSize());
    if constexpr (mk_concepts::SingleFloatingPoint<K>) {
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, static_cast<blasint>(this->RowSize()),
            static_cast<blasint>(mat.ColumnSize()), Col, 1.0, this->_elements.data(),
            static_cast<blasint>(this->RowSize()), mat._elements.data(), OppRow, 0.0, res._elements.data(),
            static_cast<blasint>(this->RowSize()));
    } else if (mk_concepts::DoubleFloatingPoint<K>) {
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, static_cast<blasint>(this->RowSize()),
            static_cast<blasint>(mat.ColumnSize()), Col, 1.0, this->_elements.data(),
            static_cast<blasint>(this->RowSize()), mat._elements.data(), OppRow, 0.0, res._elements.data(),
            static_cast<blasint>(this->RowSize()));
    }

    return res;
}

template <typename K, size_t Col, typename Backend>
    requires mk_concepts::BLASSupported<Backend, K>
Matrix<K, std::dynamic_extent, std::dynamic_extent, Backend> Matrix<K, std::dynamic_extent, Col, Backend>::Dot(
    Matrix<K, std::dynamic_extent, std::dynamic_extent, Backend> mat) {
#if !defined(NDEBUG)
    if (Col != mat.RowSize()) {
        std::string errorString = "Mismatch matrix size for matrix product.\n";
        errorString += "this size    : " + this->GetSizeString() + "\n";
        errorString += "opponent size: " + mat.GetSizeString() + ".\n";

        throw std::domain_error(errorString);
    }
#endif
    auto res = MatrixDynamicOpen<K>(this->RowSize(), mat.ColumnSize());
    if constexpr (mk_concepts::SingleFloatingPoint<K>) {
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, static_cast<blasint>(this->RowSize()),
            static_cast<blasint>(mat.ColumnSize()), Col, 1.0, this->_elements.data(),
            static_cast<blasint>(this->RowSize()), mat._elements.data(), static_cast<blasint>(mat.RowSize()), 0.0,
            res._elements.data(), static_cast<blasint>(this->RowSize()));
    } else if (mk_concepts::DoubleFloatingPoint<K>) {
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, static_cast<blasint>(this->RowSize()),
            static_cast<blasint>(mat.ColumnSize()), Col, 1.0, this->_elements.data(),
            static_cast<blasint>(this->RowSize()), mat._elements.data(), static_cast<blasint>(mat.RowSize()), 0.0,
            res._elements.data(), static_cast<blasint>(this->RowSize()));
    }

    return res;
}
/* end matrix unique arithmetics definition */
}  // namespace mknnlib::matrix

#endif