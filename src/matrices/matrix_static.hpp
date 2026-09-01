#ifndef MKNNLIB_MATRICES_MATRIX_STATIC_HPP
#define MKNNLIB_MATRICES_MATRIX_STATIC_HPP

#include <cblas.h>

#include <algorithm>
#include <array>
#include <boost/operators.hpp>
#include <cassert>
#include <cstddef>
#include <functional>
#include <mdspan>
#include <ostream>
#include <span>

#include "src/concept_defines/types/type_concepts.hpp"
#include "src/matrices/core/blas_backends.hpp"
#include "src/matrices/matrix_template_base.hpp"
#include "src/matrices/forward_declarations/static_operations.hpp"
#include "src/matrices/core/blas_storages/blas_storage.hpp" // IWYU pragma: keep

namespace mknnlib::matrix {
template <typename K, size_t Row, size_t Col, typename Backend = core::OpenBLASBackend>
    requires mk_concepts::BLASSupported<Backend, K>
std::ostream& operator<<(std::ostream& os, const Matrix<K, Row, Col, Backend>& mat);

template <typename K, size_t Row, size_t Col, typename Backend>
    requires mk_concepts::BLASSupported<Backend, K>
class Matrix;

template <typename K, size_t Row, size_t Col, typename Backend>
    requires mk_concepts::BLASSupported<Backend, K>
class Matrix :
    private boost::multipliable<Matrix<K, Row, Col, Backend>>,
    private boost::multipliable<Matrix<K, Row, Col, Backend>, Matrix<K, Row>> {
    static_assert(Row != std::dynamic_extent);
    static_assert(Col != std::dynamic_extent);

    template <typename T, size_t FRow, size_t FCol, typename FBackend>
        requires mk_concepts::BLASSupported<FBackend, T>
    friend class Matrix;

    // type alias
    using MatrixExtent = std::extents<size_t, Row, Col>;
    using MdView = std::mdspan<K, MatrixExtent, std::layout_left>;

private:
    size_t _rowSize = Row;
    size_t _columnSize = Col;
    // std::array<K, Row * Col> _elements;
    core::Storage<Backend, K, Row * Col> _elements;
    MdView _span;

public:
    // TODO: ドキュメントをちゃんと書く
    /* begin constructors declaration */
    Matrix();
    Matrix(const std::array<K, Row * Col> elements);
    // copy constructor
    Matrix(const Matrix& mat);
    /* end constructors declaration */

    /* begin operator overloads declaration */
    // copy assignment operator
    Matrix& operator=(const Matrix& x);
    // arithmetics compound operators
    Matrix& operator+=(const Matrix& x);
    Matrix& operator+=(const Matrix<K, Row, std::dynamic_extent, Backend>& x);
    Matrix& operator-=(const Matrix& x);
    Matrix& operator-=(const Matrix<K, Row, std::dynamic_extent, Backend>& x);
    Matrix& operator*=(const Matrix& x);
    Matrix& operator*=(const Matrix<K, Row, std::dynamic_extent, Backend>& x);

    // arithmetics binary operators
    friend Matrix operator+ <K, Row, Col, Backend>(Matrix lhs, Matrix rhs);
    friend Matrix operator+ <K, Row, Col, Backend>(Matrix lhs, Matrix<K, Row, std::dynamic_extent, Backend> rhs);
    friend Matrix operator- <K, Row, Col, Backend>(Matrix lhs, Matrix rhs);
    friend Matrix operator- <K, Row, Col, Backend>(Matrix lhs, Matrix<K, Row, std::dynamic_extent, Backend> rhs);

    // 2 dimensions index
    K& operator[](const size_t a, const size_t b);

    // ostream
    friend std::ostream& operator<< <>(std::ostream& os, const Matrix& mat);
    /* end operator overloads declaration */

    /* begin matrix unique functions declaration */
    constexpr size_t RowSize();
    constexpr size_t ColumnSize();
    std::string GetSizeString() const;
    core::Storage<Backend, K, Row * Col>& Elements();
    // K* _elements.data();
    // const K* _elements.data() const;

    /* end matrix unique functions declaration */

    /* begin matrix unique arithmetics declaration */
    template <size_t OppCol>
    Matrix<K, Row, OppCol, Backend> Dot(const Matrix<K, Col, OppCol, Backend> mat)
        requires mk_concepts::SingleFloatingPoint<K>;
    template <size_t OppCol>
    Matrix<K, Row, OppCol, Backend> Dot(const Matrix<K, Col, OppCol, Backend> mat)
        requires mk_concepts::DoubleFloatingPoint<K>;

    Matrix<K, Row, std::dynamic_extent, Backend> Dot(const Matrix<K, std::dynamic_extent, Col, Backend>& mat)
        requires mk_concepts::SingleFloatingPoint<K>;
    Matrix<K, Row, std::dynamic_extent, Backend> Dot(const Matrix<K, std::dynamic_extent, Col, Backend>& mat)
        requires mk_concepts::DoubleFloatingPoint<K>;

    template <size_t OppCol>
    Matrix<K, Row, OppCol, Backend> Dot(const Matrix<K, std::dynamic_extent, OppCol, Backend>& mat)
        requires mk_concepts::SingleFloatingPoint<K>;
    template <size_t OppCol>
    Matrix<K, Row, OppCol, Backend> Dot(const Matrix<K, std::dynamic_extent, OppCol, Backend>& mat)
        requires mk_concepts::DoubleFloatingPoint<K>;

    Matrix<K, Row, std::dynamic_extent, Backend> Dot(const Matrix<K, std::dynamic_extent, std::dynamic_extent, Backend>& mat)
        requires mk_concepts::SingleFloatingPoint<K>;
    Matrix<K, Row, std::dynamic_extent, Backend> Dot(const Matrix<K, std::dynamic_extent, std::dynamic_extent, Backend>& mat)
        requires mk_concepts::DoubleFloatingPoint<K>;
    /* end matrix unique arithmetics declaration */
};

/* begin constructors definition */
template <typename K, size_t Row, size_t Col, typename Backend>
    requires mk_concepts::BLASSupported<Backend, K>
Matrix<K, Row, Col, Backend>::Matrix() :
    _rowSize(Row), _columnSize(Col), _elements{}, _span(_elements.data(), MatrixExtent{}) {}

template <typename K, size_t Row, size_t Col, typename Backend>
    requires mk_concepts::BLASSupported<Backend, K>
Matrix<K, Row, Col, Backend>::Matrix(const std::array<K, Row * Col> elements) :
    _rowSize(Row), _columnSize(Col), _elements(elements), _span(_elements.data(), MatrixExtent{}) {}

// copy constructor
template <typename K, size_t Row, size_t Col, typename Backend>
    requires mk_concepts::BLASSupported<Backend, K>
Matrix<K, Row, Col, Backend>::Matrix(const Matrix<K, Row, Col, Backend>& mat) :
    _rowSize(Row), _columnSize(Col), _elements(mat._elements), _span(_elements.data(), MatrixExtent{}) {}
/* end constructors definition */

/* begin operator overloads definition */
// copy assignment operator
template <typename K, size_t Row, size_t Col, typename Backend>
    requires mk_concepts::BLASSupported<Backend, K>
Matrix<K, Row, Col, Backend>& Matrix<K, Row, Col, Backend>::operator=(const Matrix<K, Row, Col, Backend>& x) {
    if (this != &x) {
        _elements = x._elements;
        _span = MdView(_elements, MatrixExtent{});
    }
    return *this;
}

// arithmetics compound operators
template <typename K, size_t Row, size_t Col, typename Backend>
    requires mk_concepts::BLASSupported<Backend, K>
inline auto Matrix<K, Row, Col, Backend>::operator+=(const Matrix& x) -> Matrix& {
    if constexpr (mk_concepts::SingleFloatingPoint<K>) {
        cblas_saxpy(Row * Col, 1.0, x._elements.data(), 1, this->_elements.data(), 1);
        return *this;
    } else if constexpr (mk_concepts::DoubleFloatingPoint<K>) {
        cblas_daxpy(Row * Col, 1.0, x._elements.data(), 1, this->_elements.data(), 1);
        return *this;
    }
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
        return *this;
    } else if constexpr (mk_concepts::DoubleFloatingPoint<K>) {
        cblas_daxpy(Row * Col, 1.0, x._elements.data(), 1, this->_elements.data(), 1);
        return *this;
    }
}

template <typename K, size_t Row, size_t Col, typename Backend>
    requires mk_concepts::BLASSupported<Backend, K>
inline auto Matrix<K, Row, Col, Backend>::operator-=(const Matrix& x) -> Matrix& {
    if constexpr (mk_concepts::SingleFloatingPoint<K>) {
        cblas_saxpy(Row * Col, -1.0, x._elements.data(), 1, this->_elements.data(), 1);
        return *this;
    } else if constexpr (mk_concepts::DoubleFloatingPoint<K>) {
        cblas_daxpy(Row * Col, -1.0, x._elements.data(), 1, this->_elements.data(), 1);
        return *this;
    }
}

template <typename K, size_t Row, size_t Col, typename Backend>
    requires mk_concepts::BLASSupported<Backend, K>
inline auto Matrix<K, Row, Col, Backend>::operator-=(const Matrix<K, Row, std::dynamic_extent, Backend>& x) -> Matrix& {
#if !defined(NDEBUG)
    if (Col != x.ColumnSize()) {
        std::string errorString = "Mismatch matrix size for matrix product.\n";
        errorString += "this size    : " + this->GetSizeString() + "\n";
        errorString += "opponent size: " + x.GetSizeString() + ".\n";

        throw std::domain_error(errorString);
    }
#endif
    if constexpr (mk_concepts::SingleFloatingPoint<K>) {
        cblas_saxpy(Row * Col, -1.0, x._elements.data(), 1, this->_elements.data(), 1);
        return *this;
    } else if constexpr (mk_concepts::DoubleFloatingPoint<K>) {
        cblas_daxpy(Row * Col, -1.0, x._elements.data(), 1, this->_elements.data(), 1);
        return *this;
    }
}

template <typename K, size_t Row, size_t Col, typename Backend>
    requires mk_concepts::BLASSupported<Backend, K>
auto Matrix<K, Row, Col, Backend>::operator*=(const Matrix<K, Row, Col, Backend>& x) -> Matrix& {
    // hadamard product is not into BLAS
    std::ranges::transform(this->_elements, x._elements, this->_elements.begin(), std::multiplies<>());
    return *this;
}

template <typename K, size_t Row, size_t Col, typename Backend>
    requires mk_concepts::BLASSupported<Backend, K>
auto Matrix<K, Row, Col, Backend>::operator*=(const Matrix<K, Row, std::dynamic_extent, Backend>& x) -> Matrix& {
#if !defined(NDEBUG)
    if (Col != x.ColumnSize()) {
        std::string errorString = "Mismatch matrix size for matrix product.\n";
        errorString += "this size    : " + this->GetSizeString() + "\n";
        errorString += "opponent size: " + x.GetSizeString() + ".\n";

        throw std::domain_error(errorString);
    }
#endif
    // hadamard product is not into BLAS
    std::ranges::transform(this->_elements, x.Elements(), this->_elements.begin(), std::multiplies<>());
    return *this;
}

// arithmetics binary operators
template <typename K, size_t Row, size_t Col, typename Backend>
    requires mk_concepts::BLASSupported<Backend, K>
inline Matrix<K, Row, Col, Backend> operator+(Matrix<K, Row, Col, Backend> lhs, Matrix<K, Row, Col, Backend> rhs) {
    auto result = lhs;
    if constexpr (mk_concepts::SingleFloatingPoint<K>) {
        cblas_saxpy(Row * Col, 1.0, rhs._elements.data(), 1, result._elements.data(), 1);
        return result;
    } else if constexpr (mk_concepts::DoubleFloatingPoint<K>) {
        cblas_daxpy(Row * Col, 1.0, rhs._elements.data(), 1, result._elements.data(), 1);
        return result;
    }
}

template <typename K, size_t Row, size_t Col, typename Backend>
    requires mk_concepts::BLASSupported<Backend, K>
inline Matrix<K, Row, Col, Backend> operator+(
    Matrix<K, Row, Col, Backend> lhs, Matrix<K, Row, std::dynamic_extent, Backend> rhs) {
#if !defined(NDEBUG)
    if (Col != rhs.ColumnSize()) {
        std::string errorString = "Mismatch matrix size for matrix product.\n";
        errorString += "this size    : " + lhs.GetSizeString() + "\n";
        errorString += "opponent size: " + rhs.GetSizeString() + ".\n";

        throw std::domain_error(errorString);
    }
#endif
    auto result = Matrix<K, Row, Col, Backend>();
    if constexpr (mk_concepts::SingleFloatingPoint<K>) {
        cblas_saxpy(Row * Col, 1.0, rhs._elements.data(), 1, result._elements.data(), 1);
        return result;
    } else if constexpr (mk_concepts::DoubleFloatingPoint<K>) {
        cblas_daxpy(Row * Col, 1.0, rhs._elements.data(), 1, result._elements.data(), 1);
        return result;
    }
}

template <typename K, size_t Row, size_t Col, typename Backend>
    requires mk_concepts::SingleFloatingPoint<K>
inline Matrix<K, Row, Col, Backend> operator-(Matrix<K, Row, Col, Backend> lhs, Matrix<K, Row, Col, Backend> rhs) {
    auto result = lhs;
    cblas_saxpy(Row * Col, -1.0, rhs._elements.data(), 1, result._elements.data(), 1);
    return result;
}

template <typename K, size_t Row, size_t Col, typename Backend>
    requires mk_concepts::DoubleFloatingPoint<K>
inline Matrix<K, Row, Col, Backend> operator-(Matrix<K, Row, Col, Backend> lhs, Matrix<K, Row, Col, Backend> rhs) {
    auto result = lhs;
    cblas_daxpy(Row * Col, -1.0, rhs._elements.data(), 1, result._elements.data(), 1);
    return result;
}

template <typename K, size_t Row, size_t Col, typename Backend>
    requires mk_concepts::SingleFloatingPoint<K>
Matrix<K, Row, Col, Backend> operator-(Matrix<K, Row, Col, Backend> lhs, Matrix<K, Row> rhs) {
#if !defined(NDEBUG)
    if (Col != rhs.ColumnSize()) {
        std::string errorString = "Mismatch matrix size for matrix product.\n";
        errorString += "this size    : " + lhs.GetSizeString() + "\n";
        errorString += "opponent size: " + rhs.GetSizeString() + ".\n";

        throw std::domain_error(errorString);
    }
#endif
    auto result = lhs;
    cblas_saxpy(Row * Col, -1.0, rhs._elements.data(), 1, result._elements.data(), 1);
    return result;
}

template <typename K, size_t Row, size_t Col, typename Backend>
    requires mk_concepts::DoubleFloatingPoint<K>
Matrix<K, Row, Col, Backend> operator-(Matrix<K, Row, Col, Backend> lhs, Matrix<K, Row> rhs) {
#if !defined(NDEBUG)
    if (Col != rhs.ColumnSize()) {
        std::string errorString = "Mismatch matrix size for matrix product.\n";
        errorString += "this size    : " + lhs.GetSizeString() + "\n";
        errorString += "opponent size: " + rhs.GetSizeString() + ".\n";

        throw std::domain_error(errorString);
    }
#endif
    auto result = lhs;
    cblas_daxpy(Row * Col, -1.0, rhs._elements.data(), 1, result._elements.data(), 1);
    return result;
}

// ostream
template <typename K, size_t Row, size_t Col, typename Backend>
std::ostream& operator<<(std::ostream& os, const Matrix<K, Row, Col, Backend>& mat) {
    for (size_t i = 0; i < Row; i++) {
        for (size_t j = 0; j < Col; j++) {
            os << mat._span[i, j] << " ";
        }
        os << std::endl;
    }
    return os;
}
/* end operator overloads definition */

/* begin matrix unique functions definition */
template <typename K, size_t Row, size_t Col, typename Backend>
    requires mk_concepts::BLASSupported<Backend, K>
constexpr size_t Matrix<K, Row, Col, Backend>::RowSize() {
    return Row;
}

template <typename K, size_t Row, size_t Col, typename Backend>
    requires mk_concepts::BLASSupported<Backend, K>
constexpr size_t Matrix<K, Row, Col, Backend>::ColumnSize() {
    return Col;
}

template <typename K, size_t Row, size_t Col, typename Backend>
    requires mk_concepts::BLASSupported<Backend, K>
std::string Matrix<K, Row, Col, Backend>::GetSizeString() const {
    return "(" + std::to_string(Row) + ", " + std::to_string(Col) + ")";
}

template <typename K, size_t Row, size_t Col, typename Backend>
    requires mk_concepts::BLASSupported<Backend, K>
inline core::Storage<Backend, K, Row * Col>& Matrix<K, Row, Col, Backend>::Elements() {
    return this->_elements;
}

// template <typename K, size_t Row, size_t Col, typename Backend>
// inline K* MatrixStaticOpen<K, Row, Col>::_elements.data() {
//     return this->_elements.data();
// }

// template <typename K, size_t Row, size_t Col, typename Backend>
// inline const K* MatrixStaticOpen<K, Row, Col>::_elements.data() const {
//     return this->_elements.data();
// }
/* end matrix unique functions definition */

/* begin matrix unique arithmetics definition */
template <typename K, size_t Row, size_t Col, typename Backend>
    requires mk_concepts::BLASSupported<Backend, K>
template <size_t OppCol>
inline Matrix<K, Row, OppCol, Backend> Matrix<K, Row, Col, Backend>::Dot(const Matrix<K, Col, OppCol, Backend> mat)
    requires mk_concepts::SingleFloatingPoint<K>
{
    auto res = Matrix<K, Row, OppCol, Backend>();
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, Row, static_cast<blasint>(OppCol), Col, 1.0, this->_elements.data(), Row,
        mat._elements.data(), Col, 0.0, res._elements.data(), Row);
    return res;
}

template <typename K, size_t Row, size_t Col, typename Backend>
    requires mk_concepts::BLASSupported<Backend, K>
template <size_t OppCol>
inline Matrix<K, Row, OppCol, Backend> Matrix<K, Row, Col, Backend>::Dot(const Matrix<K, Col, OppCol, Backend> mat)
    requires mk_concepts::DoubleFloatingPoint<K>
{
    auto res = Matrix<K, Row, OppCol>();
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, Row, static_cast<blasint>(OppCol), Col, 1.0, this->_elements.data(), Row,
        mat._elements.data(), Col, 0.0, res._elements.data(), Row);
    return res;
}

template <typename K, size_t Row, size_t Col, typename Backend>
    requires mk_concepts::BLASSupported<Backend, K>
inline Matrix<K, Row, std::dynamic_extent, Backend> Matrix<K, Row, Col, Backend>::Dot(const Matrix<K, std::dynamic_extent, Col, Backend>& mat)
    requires mk_concepts::SingleFloatingPoint<K>
{
    // const size_t matColumnSize = mat.ColumnSize();
    // assert(matColumnSize <= INT_MAX);
    auto res = Matrix<K, Row, std::dynamic_extent, Backend>(mat.ColumnSize());
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, Row, static_cast<blasint>(mat.ColumnSize()), Col, 1.0,
        this->_elements.data(), Row, mat._elements.data(), Col, 0.0, res._elements.data(), Row);
    return res;
}

template <typename K, size_t Row, size_t Col, typename Backend>
    requires mk_concepts::BLASSupported<Backend, K>
inline Matrix<K, Row, std::dynamic_extent, Backend> Matrix<K, Row, Col, Backend>::Dot(const Matrix<K, std::dynamic_extent, Col, Backend>& mat)
    requires mk_concepts::DoubleFloatingPoint<K>
{
    // const size_t matColumnSize = mat.ColumnSize();
    // assert(matColumnSize <= INT_MAX);
    auto res = Matrix<K, Row>(mat.ColumnSize());
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, Row, static_cast<blasint>(mat.ColumnSize()), Col, 1.0,
        this->_elements.data(), Row, mat._elements.data(), Col, 0.0, res._elements.data(), Row);
    return res;
}

template <typename K, size_t Row, size_t Col, typename Backend>
    requires mk_concepts::BLASSupported<Backend, K>
template <size_t OppCol>
inline Matrix<K, Row, OppCol, Backend> Matrix<K, Row, Col, Backend>::Dot(const Matrix<K, std::dynamic_extent, OppCol, Backend>& mat)
    requires mk_concepts::SingleFloatingPoint<K>
{
#if !defined(NDEBUG)
    if (Col != mat.RowSize()) {
        std::string errorString = "Mismatch matrix size for matrix product.\n";
        errorString += "this size    : " + this->GetSizeString() + "\n";
        errorString += "opponent size: " + mat.GetSizeString() + ".\n";

        throw std::domain_error(errorString);
    }
#endif

    auto res = Matrix<K, Row, OppCol>();
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, Row, OppCol, Col, 1.0, this->_elements.data(), Row,
        mat._elements.data(), Col, 0.0, res._elements.data(), Row);
    return res;
}

template <typename K, size_t Row, size_t Col, typename Backend>
    requires mk_concepts::BLASSupported<Backend, K>
template <size_t OppCol>
inline Matrix<K, Row, OppCol, Backend> Matrix<K, Row, Col, Backend>::Dot(const Matrix<K, std::dynamic_extent, OppCol, Backend>& mat)
    requires mk_concepts::DoubleFloatingPoint<K>
{
#if !defined(NDEBUG)
    if (Col != mat.RowSize()) {
        std::string errorString = "Mismatch matrix size for matrix product.\n";
        errorString += "this size    : " + this->GetSizeString() + "\n";
        errorString += "opponent size: " + mat.GetSizeString() + ".\n";

        throw std::domain_error(errorString);
    }
#endif

    auto res = Matrix<K, Row, OppCol>();
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, Row, OppCol, Col, 1.0, this->_elements.data(), Row,
        mat._elements.data(), Col, 0.0, res._elements.data(), Row);
    return res;
}

template <typename K, size_t Row, size_t Col, typename Backend>
    requires mk_concepts::BLASSupported<Backend, K>
inline Matrix<K, Row, std::dynamic_extent, Backend> Matrix<K, Row, Col, Backend>::Dot(const Matrix<K, std::dynamic_extent, std::dynamic_extent, Backend>& mat)
    requires mk_concepts::SingleFloatingPoint<K>
{
#if !defined(NDEBUG)
    if (Col != mat.RowSize()) {
        std::string errorString = "Mismatch matrix size for matrix product.\n";
        errorString += "this size    : " + this->GetSizeString() + "\n";
        errorString += "opponent size: " + mat.GetSizeString() + ".\n";

        throw std::domain_error(errorString);
    }
#endif

    auto res = Matrix<K, Row>(mat.ColumnSize());
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, Row, static_cast<blasint>(mat.ColumnSize()), Col, 1.0,
        this->_elements.data(), Row, mat._elements.data(), Col, 0.0, res._elements.data(), Row);
    return res;
}

template <typename K, size_t Row, size_t Col, typename Backend>
    requires mk_concepts::BLASSupported<Backend, K>
inline Matrix<K, Row, std::dynamic_extent, Backend> Matrix<K, Row, Col, Backend>::Dot(const Matrix<K, std::dynamic_extent, std::dynamic_extent, Backend>& mat)
    requires mk_concepts::DoubleFloatingPoint<K>
{
#if !defined(NDEBUG)
    if (Col != mat.RowSize()) {
        std::string errorString = "Mismatch matrix size for matrix product.\n";
        errorString += "this size    : " + this->GetSizeString() + "\n";
        errorString += "opponent size: " + mat.GetSizeString() + ".\n";

        throw std::domain_error(errorString);
    }
#endif

    auto res = Matrix<K, Row>(mat.ColumnSize());
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, Row, static_cast<blasint>(mat.ColumnSize()), Col, 1.0,
        this->_elements.data(), Row, mat._elements.data(), Col, 0.0, res._elements.data(), Row);
    return res;
}
/* end matrix unique arithmetics definition */
}  // namespace mknnlib::matrix

#endif