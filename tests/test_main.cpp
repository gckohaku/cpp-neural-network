#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/included/unit_test.hpp>

#include "tests/matrix_test/matrix_product/static_test.cpp"
#include "tests/matrix_test/matrix_product/row_static_test.cpp"
#include "tests/matrix_test/matrix_product/column_static_test.cpp"
#include "tests/matrix_test/matrix_product/dynamic_test.cpp"
#include "tests/matrix_test/matrix_product/static_and_row_static_test.cpp"
#include "tests/matrix_test/matrix_product/static_and_column_static_test.cpp"