#include <array>
#include <cblas.h>
#include <iostream>
#include <ostream>
#include <span>
#include <vector>

#include "src/matrices/core/blas_backends.hpp"
#include "src/matrices/matrix.hpp"
// #include "src/matrices/matrix_template_base.hpp"

using mknnlib::matrix::Matrix;
using mknnlib::matrix::MatrixRowStaticOpen;
using mknnlib::matrix::core::OpenBLASBackend;

int main() {
    // a row, c row value
    constexpr size_t m = 2;
    // a column, b row value
    constexpr size_t k = 3;
    // b column, c column value
    constexpr size_t n = 2;

    std::array<float, m * k> a = {1.0, 4.0, 2.0, 5.0, 3.0, 6.0};
    // std::array<float, k * n> b = {7.0, 9.0, 2.0, 8.0, 1.0, 3.0};

    std::vector<float> rowStatic = {2.0, 4.0, 6.0, 8.0, 10.0, 12.0};

    auto z = Matrix<float, m, k, OpenBLASBackend>(rowStatic);

    std::vector<float> c(m * n, 0.0);

    auto x = Matrix<float, m, std::dynamic_extent, OpenBLASBackend>(a, k);


    // x += z;

    // std::cout << x << std::endl;

    // std::cout << "\n==========\n" << "z:\n" << z << "\n\n" << std::endl;

    // z += x;

    // std::cout << z << std::endl;

    std::cout << "x:\n" << x << std::endl;
    std::cout << "z:\n" << z << std::endl;

    std::cout << "==========" << std::endl;

    auto res = x + z;

    std::cout << "res:\n" << res << std::endl;

    return 0;
}