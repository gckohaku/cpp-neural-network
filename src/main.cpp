#include <array>
#include <cblas.h>
#include <iostream>
#include <ostream>
#include <vector>

#include "matrices/matrix_static.hpp"
#include "src/matrices/matrix_row_static.hpp"

using mknnlib::matrix::MatrixStatic;
using mknnlib::matrix::MatrixRowStatic;

int main() {
    // a row, c row value
    int m = 2;
    // a column, b row value
    int k = 3;
    // b column, c column value
    int n = 2;

    float alpha = 1.0;
    float beta = 0.0;

    std::array<float, 6> a = {1.0, 4.0, 2.0, 5.0, 3.0, 6.0};
    std::array<float, 6> b = {7.0, 9.0, 2.0, 8.0, 1.0, 3.0};

    std::vector<float> rowStatic = {2.0, 4.0, 6.0, 8.0, 10.0, 12.0};
    auto z = MatrixRowStatic<float, 3>(2, rowStatic);

    std::vector<float> c(m * n, 0.0);

    auto x = MatrixStatic<float, 2, 3>(a);
    auto y = MatrixStatic<float, 3, 2>(b);
    
    auto res = x.Dot(y);
    auto resRowStatic = x.Dot(z);

    std::cout << resRowStatic << std::endl;

    return 0;
}