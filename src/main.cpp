#include <iostream>
#include <eigen3/Eigen/Dense>

int main() {
    Eigen::Matrix3d m = Eigen::Matrix3d::Random();

    std::cout << m << std::endl;
}