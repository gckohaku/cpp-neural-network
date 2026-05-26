#include <array>
#include <cblas.h>
#include <iostream>
#include <vector>

int main() {
    // a row, c row value
    int m = 2;
    // a column, b row value
    int k = 3;
    // b column, c column value
    int n = 2;

    float alpha = 1.0;
    float beta = 0.0;

    std::vector<float> a = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    std::vector<float> b = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

    std::vector<float> c(m * n, 0.0);

    // call cblas_sgemm
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, a.data(), k, b.data(), n, beta, c.data(), n);

    // output result
    std::cout << "Result matrix c:" << std::endl;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << c[i * n + j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}