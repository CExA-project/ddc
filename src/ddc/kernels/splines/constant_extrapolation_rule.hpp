// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstddef>

#include <ddc/ddc.hpp>

#include <Kokkos_Core.hpp>

namespace ddc {

/**
 * @brief A functor for describing a spline boundary value by a constant extrapolation for 2D evaluator.
 *
 * To define the value of a function on B-splines out of the domain, we here use a constant
 * extrapolation on the edge.
 */
template <class DimI, class... DimNI>
struct ConstantExtrapolationRule
{
private:
    ddc::Coordinate<DimI> m_eval_pos;

public:
#if DDC_BUILD_DEPRECATED_CODE()
    /**
     * @brief Instantiate a ConstantExtrapolationRule.
     *
     * The boundary value will be the same as at the coordinate given in a dimension given.
     * The dimension of the input defines the dimension of the boundary condition.
     * The second and the third parameters are needed in case of non-periodic splines on the
     * dimension off-interest (the complementary dimension of the boundary condition),
     * because the evaluator can receive coordinates outside the domain in both dimension.
     *
     * @param[in] eval_pos Coordinate in the dimension given inside the domain where we will evaluate each points outside the domain.
     * @param[in] eval_pos_not_interest_min The minimum coordinate inside the domain on the complementary dimension of the boundary condition.
     * @param[in] eval_pos_not_interest_max The maximum coordinate inside the domain on the complementary dimension of the boundary condition.
     *
     * @deprecated Use the single parameter constructor instead, the boundaries are now retrieved from the BSplines boundaries
     */
    [[deprecated("Use the single parameter constructor instead, the boundaries are now retrieved from the BSplines boundaries")]] explicit ConstantExtrapolationRule(
            ddc::Coordinate<DimI> eval_pos,
            [[maybe_unused]] ddc::Coordinate<DimNI...> eval_pos_not_interest_min,
            [[maybe_unused]] ddc::Coordinate<DimNI...> eval_pos_not_interest_max)
        requires(sizeof...(DimNI) == 1)
        : m_eval_pos(eval_pos)
    {
    }
#endif

    /**
     * @brief Instantiate a ConstantExtrapolationRule.
     *
     * The boundary value will be the same as at the coordinate given in a dimension given.
     * The dimension of the input defines the dimension of the boundary condition.
     * No second and third parameters are needed in case of periodic splines on the
     * dimension off-interest (the complementary dimension of the boundary condition).
     *
     * @param[in] eval_pos Coordinate in the dimension given inside the domain where we will evaluate each points outside the domain.
     */
    explicit ConstantExtrapolationRule(ddc::Coordinate<DimI> eval_pos) : m_eval_pos(eval_pos) {}

    /**
     * @brief Get the value of the function on B-splines at a coordinate outside the domain.
     *
     * In the dimension defined in the constructor Dim1 (or Dim2), it sets the coordinate pos_1 (or pos_2)
     * given at the m_eval_pos coordinate if it is outside the domain.
     * If the coordinate on the complementary dimension of the boundary condition dimension ddc::Coordinate<DimNI>(coord_extrap) is
     * outside the domain, then it also sets the coordinate at eval_pos_not_interest_min
     * (if ddc::Coordinate<DimNI>(coord_extrap) @f$ < @f$ eval_pos_not_interest_min) or
     * at eval_pos_not_interest_max (if ddc::Coordinate<DimNI>(coord_extrap) @f$ > @f$ eval_pos_not_interest_max).
     *
     * @param[in] coord_extrap The coordinates where we want to evaluate the function on B-splines
     * @param[in] spline_coef The coefficients of the function on B-splines.
     *
     *@return A double with the value of the function on B-splines evaluated at the coordinate.
     */
    template <class CoordType, class... BSplines, class Layout, class MemorySpace>
    KOKKOS_FUNCTION double operator()(
            CoordType coord_extrap,
            ddc::ChunkSpan<
                    double const,
                    ddc::DiscreteDomain<BSplines...>,
                    Layout,
                    MemorySpace> const spline_coef) const
    {
        static_assert(in_tags_v<DimI, to_type_seq_t<CoordType>>);
        static_assert((in_tags_v<DimNI, to_type_seq_t<CoordType>> && ...));
        using TypeSeqBSplines = ddc::detail::TypeSeq<BSplines...>;

        ddc::Coordinate<DimI, DimNI...> const
        coord_eval(m_eval_pos, get_eval_pos<TypeSeqBSplines>(ddc::select<DimNI>(coord_extrap))...);

        auto vals_ptr = cexa::make_tuple(std::array<double, BSplines::degree() + 1> {}...);
        auto const vals = cexa::make_tuple(
                Kokkos::mdspan<double, Kokkos::extents<std::size_t, BSplines::degree() + 1>>(
                        cexa::get<ddc::type_seq_rank_v<BSplines, TypeSeqBSplines>>(vals_ptr)
                                .data())...);

        auto const jmin = cexa::make_tuple(
                ddc::discrete_space<BSplines>().eval_basis(
                        cexa::get<ddc::type_seq_rank_v<BSplines, TypeSeqBSplines>>(vals),
                        ddc::Coordinate<typename BSplines::continuous_dimension_type>(
                                coord_eval))...);

        static constexpr std::size_t dimension = sizeof...(BSplines);

        double y = 0.0;
        for_each(
                std::array<std::size_t, dimension> {(BSplines::degree() + 1)...},
                [&](std::array<std::size_t, dimension> idx) {
                    y += spline_coef(
                                 ddc::DiscreteElement<BSplines...>((
                                         cexa::get<ddc::type_seq_rank_v<BSplines, TypeSeqBSplines>>(
                                                 jmin)
                                         + idx[ddc::type_seq_rank_v<
                                                 BSplines,
                                                 TypeSeqBSplines>])...))
                         * (cexa::get<ddc::type_seq_rank_v<BSplines, TypeSeqBSplines>>(
                                    vals)[idx[ddc::type_seq_rank_v<BSplines, TypeSeqBSplines>]]
                            * ...);
                });

        return y;
    }

private:
    template <class TypeSeqBSplines, class QDim>
    KOKKOS_INLINE_FUNCTION ddc::Coordinate<QDim> get_eval_pos(
            ddc::Coordinate<QDim> coord_extrap) const
    {
        static_assert(ddc::in_tags_v<QDim, ddc::detail::TypeSeq<DimNI...>>);
        using BSplinesNI = ddc::type_seq_find_cdim_t<QDim, TypeSeqBSplines>;
        if constexpr (BSplinesNI::is_periodic()) {
            return ddc::Coordinate<QDim>(coord_extrap);
        } else {
            return Kokkos::
                    clamp(ddc::Coordinate<QDim>(coord_extrap),
                          ddc::discrete_space<BSplinesNI>().rmin(),
                          ddc::discrete_space<BSplinesNI>().rmax());
        }
    }

    template <std::size_t N, class Functor, class... Is>
    KOKKOS_INLINE_FUNCTION static void for_each(
            std::array<std::size_t, N> const& bounds,
            Functor const& f,
            Is... is)
    {
        static constexpr std::size_t I = sizeof...(Is);
        if constexpr (I == N) {
            f(std::array<std::size_t, N> {is...});
        } else {
            for (std::size_t i = 0; i < bounds[I]; ++i) {
                for_each(bounds, f, is..., i);
            }
        }
    }
};

} // namespace ddc
