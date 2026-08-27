#include <array>
#include <cblas.h>
#include <iostream>
#include <ostream>
#include <vector>

#include "matrices/matrix_static.hpp"
#include "src/matrices/matrix_row_static.hpp"

using mknnlib::matrix::Matrix;
using mknnlib::matrix::MatrixRowStaticOpen;

int main() {
    // a row, c row value
    constexpr size_t m = 2;
    // a column, b row value
    constexpr size_t k = 3;
    // b column, c column value
    constexpr size_t n = 2;

    // float alpha = 1.0;
    // float beta = 0.0;

    std::array<float, m * k> a = {1.0, 4.0, 2.0, 5.0, 3.0, 6.0};
    // std::array<float, k * n> b = {7.0, 9.0, 2.0, 8.0, 1.0, 3.0};

    std::vector<float> rowStatic = {2.0, 4.0, 6.0, 8.0, 10.0, 12.0};
    auto z = MatrixRowStaticOpen<float, k>(n, rowStatic);

    std::vector<float> c(m * n, 0.0);

    auto x = Matrix<float, m, k>(a);
    // auto y = MatrixStaticOpen<float, k, n>(b);
    
    // auto res = x.Dot(y);
    auto resRowStatic = x.Dot(z);

    std::cout << resRowStatic << std::endl;

    return 0;
}