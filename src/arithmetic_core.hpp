#ifndef MKNNLIB_ARITHMETIC_CORE_HPP
#define MKNNLIB_ARITHMETIC_CORE_HPP

#include <concepts>
namespace mknnlib::arithmetics {
struct AddTag;
struct SubstructTag;
struct MultiplyTag;

template <typename Op, typename L, typename R>
struct CompoundOperation;

template <typename Op, typename L, typename R>
concept CompoundOperationDefined = requires(L& lhs, const R& rhs) {
    { CompoundOperation<Op, L, R>::apply(lhs, rhs) } -> std::same_as<L>;
};

template <typename T>
class ArithmeticMixin {
public:
    template <typename U>
        requires CompoundOperationDefined<AddTag, T, U>
    constexpr T& operator+=(const U& rhs) noexcept(
        noexcept(CompoundOperation<AddTag, T, U>::apply(static_cast<T&>(*this), rhs))
    ) {
        return CompoundOperation<AddTag, T, U>::apply(static_cast<T&>(*this), rhs);
    }
};
}  // namespace mknnlib::arithmetics
#endif