#include <boost/mpl/list.hpp>
#include <boost/test/included/unit_test.hpp>
#include <boost/test/tools/interface.hpp>
#include <boost/test/tools/old/interface.hpp>
#include <boost/test/unit_test_suite.hpp>
#include <cstddef>
#include <limits>
#include <type_traits>

#include "src/matrices/matrix_static.hpp"
#include "tests/test_matrix_add_array_defines.hpp"

using mknnlib::matrix::MatrixStatic;

namespace matrix_test_add {
BOOST_AUTO_TEST_CASE_TEMPLATE(matrix_static_test, T, boost::mpl::list<float>) {
    // 3x3
    constexpr size_t rowAAndB = 3;
    constexpr size_t columnAAndB = 3;
    constexpr size_t elementSizeAAndB = rowAAndB * columnAAndB;

    auto arrayA = MakeArrayA<T>();
    auto arrayB = MakeArrayB<T>();

    auto A = MatrixStatic<T, rowAAndB, columnAAndB>(arrayA);
    auto B = MatrixStatic<T, rowAAndB, columnAAndB>(arrayB);

    auto expectedAPlusB = MakeExpectedArrayAPlusB<T>();

    auto acceptableErrorAPlusB = MakeAcceptableErrorArrayAPlusB<T>();

    for (size_t i = 0; i < elementSizeAAndB; i++) {
        acceptableErrorAPlusB[i] *= std::numeric_limits<T>::epsilon();
    }

    auto APlusB = A + B;

    auto typeCheckAPlusB = std::is_same<decltype(APlusB), MatrixStatic<T, rowAAndB, columnAAndB>>::value;
    BOOST_CHECK(typeCheckAPlusB);
    CheckCloseEachArrayElement<T, elementSizeAAndB>(APlusB.Elements(), expectedAPlusB, acceptableErrorAPlusB);
}
}  // namespace matrix_test_add