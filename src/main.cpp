#include <array>
#include <cblas.h>
#include <iostream>
#include <ostream>
#include <vector>

#include "matrices/matrix_static.hpp"

int main() {
    // a row, c row value
    int m = 2;
    // a column, b row value
    int k = 3;
    // b column, c column value
    int n = 2;

    float alpha = 1.0;
    float beta = 0.0;

    std::array<float, 6> a = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    std::array<float, 6> b = {2.0, 3.0, 5.0, 6.0, 8.0, 11.0};

    std::vector<float> c(m * n, 0.0);

    auto x = MatrixStatic<float, 2, 3>(a);
    auto y = MatrixStatic<float, 2, 3>(b);
    auto res = x + y;

    std::cout << x << "\n+\n" << y << std::endl << "↓" << std::endl;

    std::cout << res << std::endl;

    return 0;
}