#ifndef TESTS_TEST_TYPE_DEFINES_HPP
#define TESTS_TEST_TYPE_DEFINES_HPP

#include <boost/mp11/algorithm.hpp>
#include <boost/mpl/list.hpp>
#include <boost/mp11.hpp>

#include "src/matrices/core/blas_backends.hpp"

using CheckMatrixElementType = std::tuple<float, double>;
using CheckBLASBackend = std::tuple<mknnlib::matrix::core::OpenBLASBackend>;

template <typename T, typename U>
struct CheckTypes {
    using first = T;
    using second = U;
};

using CheckTypeCases = boost::mp11::mp_product<CheckTypes, CheckMatrixElementType, CheckBLASBackend>;

#endif