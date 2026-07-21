#include <boost/mpl/list.hpp>
#include <vector>

typedef boost::mpl::list<float, double> MatrixProductCheckType;

template <typename T>
std::vector<T> MakeArrayA() {
    return std::vector<T>{static_cast<T>(5.3), static_cast<T>(6.7), static_cast<T>(3.7), static_cast<T>(4.1),
        static_cast<T>(7.4), static_cast<T>(8), static_cast<T>(3.4), static_cast<T>(8.2), static_cast<T>(4.5)};
}

template <typename T>
std::vector<T> MakeArrayB() {
    return std::vector<T>{static_cast<T>(5.7), static_cast<T>(5.5), static_cast<T>(0.3), static_cast<T>(8.3),
        static_cast<T>(2), static_cast<T>(1.4), static_cast<T>(7.2), static_cast<T>(3.1), static_cast<T>(7.9)};
}

template <typename T>
std::vector<T> MakeExpectedAB() {
    return std::vector<T>{static_cast<T>(53.78), static_cast<T>(81.35), static_cast<T>(66.44), static_cast<T>(56.95),
        static_cast<T>(81.89), static_cast<T>(53.01), static_cast<T>(77.73), static_cast<T>(135.96),
        static_cast<T>(86.99)};
}

template <typename T>
std::vector<T> MakeExpectedBA() {
    return std::vector<T>{static_cast<T>(112.46), static_cast<T>(54.02), static_cast<T>(40.2), static_cast<T>(142.39),
        static_cast<T>(62.15), static_cast<T>(74.79), static_cast<T>(119.84), static_cast<T>(49.05),
        static_cast<T>(48.05)};
}

template <typename T>
std::vector<T> MakeAcceptableErrorAB() {
    return std::vector<T>{static_cast<T>(83.85), static_cast<T>(105.65), static_cast<T>(104.525), static_cast<T>(98),
        static_cast<T>(129), static_cast<T>(88.3), static_cast<T>(100), static_cast<T>(178.8), static_cast<T>(119.6)};
}

template <typename T>
std::vector<T> MakeAcceptableErrorBA() {
    return std::vector<T>{static_cast<T>(160), static_cast<T>(78.2), static_cast<T>(45.425), static_cast<T>(221.2),
        static_cast<T>(102), static_cast<T>(110.425), static_cast<T>(203.8), static_cast<T>(78.4),
        static_cast<T>(70.45)};
}