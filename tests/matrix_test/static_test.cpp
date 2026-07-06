#include <boost/test/included/unit_test.hpp>
#include <boost/test/tools/interface.hpp>
#include <boost/test/tools/old/interface.hpp>
#include <boost/test/unit_test_suite.hpp>
#include <type_traits>

#include "src/matrices/matrix_static.hpp"
#include "tests/test_utilities.hpp"

using mknnlib::matrix::MatrixStatic;
namespace tt = boost::test_tools;

BOOST_AUTO_TEST_CASE(matrix_static_test) {
    auto arrayA = std::array<float, 3 * 3>{5.3f, 6.7f, 3.7f, 4.1f, 7.4f, 8.0f, 3.4f, 8.2f, 4.5f};
    auto arrayB = std::array<float, 3 * 3>{5.7f, 5.5f, 0.3f, 8.3f, 2.0f, 1.4f, 7.2f, 3.1f, 7.9f};

    auto expectedAB = std::array<float, 3 * 3>{53.78f, 81.35f, 66.44f, 56.95f, 81.89f, 53.01f, 77.73f, 135.96f, 86.99f};
    auto expectedBA =
        std::array<float, 3 * 3>{112.46f, 54.02f, 40.2f, 142.39f, 62.15f, 74.79f, 119.84f, 49.05f, 48.05f};

    auto A = MatrixStatic<float, 3, 3>(arrayA);
    auto B = MatrixStatic<float, 3, 3>(arrayB);

    auto acceptableErrorAB = std::array<float, 3 * 3>{83.85f, 105.65f, 104.525f, 98.0f, 129.0f, 88.3f, 100.0f, 178.8f, 119.6f};

    auto AB = A * B;
    auto BA = B * A;

    auto typeCheckA = std::is_same<decltype(AB), MatrixStatic<float, 3, 3>>::value;
    auto typeCheckB = std::is_same<decltype(BA), MatrixStatic<float, 3, 3>>::value;
    BOOST_CHECK(typeCheckA);
    BOOST_CHECK(typeCheckB);
    CheckCloseEachArrayElement<float, 3 * 3>(AB.Elements(), expectedAB, acceptableErrorAB);
    CheckCloseEachArrayElement<float, 3 * 3>(BA.Elements(), expectedBA, acceptableErrorAB);
}