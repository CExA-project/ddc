#pragma once

#include "view.hpp"

namespace ddc {

namespace detail {

template <class... BSplines, class Layout, class MemorySpace>
KOKKOS_FUNCTION void ConstantExtrapolationRuleNestedLooper(
        std::tuple<std::array<double, BSplines::degree() + 1>...> vals,
        ddc::ChunkSpan<double const, ddc::DiscreteDomain<BSplines...>, Layout, MemorySpace> const
                spline_coef,
        std::tuple<ddc::DiscreteElement<BSplines>...> idx,
        std::tuple<std::conditional_t<true, std::size_t, BSplines>...> sizes,
        std::tuple<std::conditional_t<true, std::size_t, BSplines>...> iterator,
        std::size_t index,
        int& y)
{
    if (index == sizeof...(BSplines) - 1) {
        double tmp = 1;
        for (std::size_t i = 0; i < sizeof...(BSplines); i++) {
            tmp *= vals[i][iterator[i]];
        }
        std::size_t j = 0;
        y += tmp
             * spline_coef(ddc::DiscreteElement<BSplines...>(
                     (idx[j] + iterator[j++] + 0 * BSplines::degree())...));
        return;
    }

    for (iterator[index] = 0; iterator[index] < sizes[index]; iterator[index]++) {
        ConstantExtrapolationRuleNestedLooper(iterator, index + 1, y);
    }
}

} // namespace detail

template <class... DDim>
struct ConstantExtrapolationRule
{
private:
    ddc::DiscreteElement<DDim...> m_eval_pos;

public:
    explicit ConstantExtrapolationRule(ddc::DiscreteElement<DDim...> eval_pos)
        : m_eval_pos(eval_pos)
    {
    }

    template <class CoordType, class... BSplines, class Layout, class MemorySpace>
    KOKKOS_FUNCTION double operator()(
            CoordType,
            ddc::ChunkSpan<
                    double const,
                    ddc::DiscreteDomain<BSplines...>,
                    Layout,
                    MemorySpace> const spline_coef) const
    {
        std::tuple<std::array<double, BSplines::degree() + 1>...> vals;
        std::tuple<ddc::DiscreteElement<BSplines>...> idx;
        std::tuple<std::conditional_t<true, std::size_t, BSplines>...> iterator(
                (0 * BSplines::degree())...);
        std::tuple<std::conditional_t<true, std::size_t, BSplines>...> sizes(
                (BSplines::degree() + 1)...);

        std::size_t i = 0;
        (idx[i] = ddc::discrete_space<BSplines>().eval_basis(vals[i++], m_eval_pos), ...);

        double y = 0.0;
        ddc::detail::
                ConstantExtrapolationRuleHelper(vals, spline_coef, idx, sizes, iterator, index, y);

        return y;
    }
};
} // namespace ddc
