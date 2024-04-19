// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cassert>
#include <memory>
#include <vector>

#include <ddc/ddc.hpp>

#include "view.hpp"

namespace ddc {

namespace detail {

template <class T>
struct NonUniformBsplinesKnots : NonUniformPointSampling<typename T::tag_type>
{
};

struct NonUniformBSplinesBase
{
};

} // namespace detail

/**
 * The type of a non-uniform BSplines 1D basis.
 *
 * @tparam Tag The tag identifying the continuous dimension which supports the building of the BSplines.
 * @tparam D The degree of the BSplines.
 */
template <class Tag, std::size_t D>
class NonUniformBSplines : detail::NonUniformBSplinesBase
{
    static_assert(D > 0, "Parameter `D` must be positive");

public:
    /// @brief The tag identifying the continuous dimension which supports the building of the BSplines.
    using tag_type = Tag;

    /// @brief The discrete dimension identifying BSplines.
    using discrete_dimension_type = NonUniformBSplines;

    /// @brief The degree of BSplines.
    static constexpr std::size_t degree() noexcept
    {
        return D;
    }

    /// @brief Indicates if the BSplines are periodic or not.
    static constexpr bool is_periodic() noexcept
    {
        return Tag::PERIODIC;
    }

    /// @brief Indicates if the BSplines are radial or not (should be deprecated soon because this concept is not in the scope of DDC).
    static constexpr bool is_radial() noexcept
    {
        return false;
    }

    /// @brief Indicates if the BSplines are uniform or not (this is not the case here).
    static constexpr bool is_uniform() noexcept
    {
        return false;
    }

    /// @brief Impl
    template <class DDim, class MemorySpace>
    class Impl
    {
        template <class ODDim, class OMemorySpace>
        friend class Impl;

    private:
        using mesh_type = detail::NonUniformBsplinesKnots<DDim>;

        ddc::DiscreteDomain<mesh_type> m_domain;

        int m_nknots;

    public:
        /// @brief The type of discrete dimension identifying BSplines.
        using discrete_dimension_type = NonUniformBSplines;

        /// @brief The type of discrete domain indexing the BSplines.
        using discrete_domain_type = DiscreteDomain<DDim>;

        /// @brief The type of discrete element indexing a BSpline.
        using discrete_element_type = DiscreteElement<DDim>;

        /// @brief The type of discrete vector representing an "indexes displacement" between two BSplines.
        using discrete_vector_type = DiscreteVector<DDim>;

        Impl() = default;

        /// @brief Construct a `Impl` using a brace-list, i.e. `Impl bsplines({0., 1.})`
        explicit Impl(std::initializer_list<ddc::Coordinate<Tag>> breaks)
            : Impl(breaks.begin(), breaks.end())
        {
        }

        /// @brief Construct a `Impl` using a C++20 "common range".
        explicit Impl(std::vector<ddc::Coordinate<Tag>> const& breaks)
            : Impl(breaks.begin(), breaks.end())
        {
        }

        /// @brief Construct a `Impl` using a pair of iterators.
        template <class RandomIt>
        Impl(RandomIt breaks_begin, RandomIt breaks_end);

        /// @brief Copy-constructs from another Impl with different Kokkos memory space
        template <class OriginMemorySpace>
        explicit Impl(Impl<DDim, OriginMemorySpace> const& impl)
            : m_domain(impl.m_domain)
            , m_nknots(impl.m_nknots)
        {
        }

        /// @brief Copy-constructs
        Impl(Impl const& x) = default;

        /// @brief Move-constructs
        Impl(Impl&& x) = default;

        /// @brief Destructs
        ~Impl() = default;

        /// @brief Copy-assigns
        Impl& operator=(Impl const& x) = default;

        /// @brief Move-assigns
        Impl& operator=(Impl&& x) = default;

        /** @brief Evaluates BSplines at a given coordinate
         *
         * The values are computed for every BSplines at the given coordinate x. A spline approximation at coordinate x is a linear combination of those BSplines evaluations weigthed with splines coefficients of the spline-transformed initial discrete function.
         * @param[out] values The values of the BSplines evaluated at coordinate x. It has to be a (1+degree) 1D mdspan.
         * @param[in] x The coordinate where BSplines are evaluated.
         * @return TODO
         */
        KOKKOS_INLINE_FUNCTION discrete_element_type
        eval_basis(DSpan1D values, ddc::Coordinate<Tag> const& x) const;

        /** @brief Evaluates BSplines derivatives at a given coordinate
         *
         * The derivatives are computed for every BSplines at the given coordinate x. A spline approximation of a derivative at coordinate x is a linear combination of those BSplines derivatives weigthed with splines coefficients of the spline-transformed initial discrete function.
         * @param[out] derivs The derivatives of the BSplines evaluated at coordinate x. It has to be a (1+degree) 1D mdspan.
         * @param[in] x The coordinate where BSplines derivatives are evaluated.
         * @return TODO
         */
        KOKKOS_INLINE_FUNCTION discrete_element_type
        eval_deriv(DSpan1D derivs, ddc::Coordinate<Tag> const& x) const;

        /** @brief Evaluates BSplines values and n derivatives at a given coordinate
         *
         * The values and derivatives are computed for every BSplines at the given coordinate x.
         * @param[out] derivs The values and n derivatives of the BSplines evaluated at coordinate x. It has to be a (1+degree)*(1+n) 2D mdspan.
         * @param[in] x The coordinates where BSplines derivatives are evaluated.
         * @param[in] n The number of derivatives to evaluate (in addition to the BSplines values themselves).
         * @return TODO
         */
        KOKKOS_INLINE_FUNCTION discrete_element_type eval_basis_and_n_derivs(
                ddc::DSpan2D derivs,
                ddc::Coordinate<Tag> const& x,
                std::size_t n) const;

        /** @brief Compute the integrals of the BSplines.
         *
         * @param[out] int_vals The values of the integrals. It has to be a (1+nbasis) 1D mdspan.
         * @return The values of the integrals.
         */
        template <class Layout, class MemorySpace2>
        KOKKOS_INLINE_FUNCTION ddc::ChunkSpan<double, discrete_domain_type, Layout, MemorySpace2>
        integrals(
                ddc::ChunkSpan<double, discrete_domain_type, Layout, MemorySpace2> int_vals) const;

        /** @brief Returns the coordinate of the knot corresponding to the given index for a BSpline.
         *
         * @param[in] knot_idx Integer identifying index of the knot.
         * @return Coordinate of the knot.
         */
        KOKKOS_INLINE_FUNCTION ddc::Coordinate<Tag> get_knot(int knot_idx) const noexcept
        {
            return ddc::coordinate(ddc::DiscreteElement<mesh_type>(knot_idx + degree()));
        }

        /** @brief Returns the coordinate of the first support knot associated to a discrete_element identifying a BSpline.
         *
         * @param[in] ix DiscreteElement identifying the BSpline.
         * @return Coordinate of the knot.
         */
        KOKKOS_INLINE_FUNCTION ddc::Coordinate<Tag> get_first_support_knot(
                discrete_element_type const& ix) const
        {
            return ddc::coordinate(ddc::DiscreteElement<mesh_type>(ix.uid()));
        }

        /** @brief Returns the coordinate of the last support knot associated to a discrete_element identifying a BSpline.
         *
         * @param[in] ix DiscreteElement identifying the BSpline.
         * @return Coordinate of the knot.
         */
        KOKKOS_INLINE_FUNCTION ddc::Coordinate<Tag> get_last_support_knot(
                discrete_element_type const& ix) const
        {
            return ddc::coordinate(ddc::DiscreteElement<mesh_type>(ix.uid() + degree() + 1));
        }

        /** @brief Returns the coordinate of the \f$n\f$-th knot associated to a discrete_element identifying a BSpline.
         *
         * @param[in] ix DiscreteElement identifying the BSpline.
         * @param[in] n Integer indexing a knot in the support of the BSpline.
         * @return Coordinate of the knot.
         */
        KOKKOS_INLINE_FUNCTION ddc::Coordinate<Tag> get_support_knot_n(
                discrete_element_type const& ix,
                int n) const
        {
            return ddc::coordinate(ddc::DiscreteElement<mesh_type>(ix.uid() + n));
        }

        /** @brief Returns the coordinate of the lower boundary knot.
         *
         * @return Coordinate of the lower boundary knot.
         */
        KOKKOS_INLINE_FUNCTION ddc::Coordinate<Tag> rmin() const noexcept
        {
            return get_knot(0);
        }

        /** @brief Returns the coordinate of the upper boundary knot.
         *
         * @return Coordinate of the upper boundary knot.
         */
        KOKKOS_INLINE_FUNCTION ddc::Coordinate<Tag> rmax() const noexcept
        {
            return get_knot(ncells());
        }

        /** @brief Returns the length of the domain supporting knots.
         *
         * @return The length of the domain supporting knots.
         */
        KOKKOS_INLINE_FUNCTION double length() const noexcept
        {
            return rmax() - rmin();
        }

        /** @brief TODO
         *
         * @return TODO.
         */
        KOKKOS_INLINE_FUNCTION std::size_t size() const noexcept
        {
            return degree() + ncells();
        }

        /** @brief Returns the discrete domain including ghost bsplines
         *
         * @return The discrete domain including ghost bsplines.
         */
        KOKKOS_INLINE_FUNCTION discrete_domain_type full_domain() const
        {
            return discrete_domain_type(discrete_element_type(0), discrete_vector_type(size()));
        }

        /** @brief TODO
         *
         * @return TODO
         */
        KOKKOS_INLINE_FUNCTION std::size_t npoints() const noexcept
        {
            return m_nknots - 2 * degree();
        }

        /** @brief TODO
         *
         * @return TODO
         */
        KOKKOS_INLINE_FUNCTION std::size_t nbasis() const noexcept
        {
            return ncells() + !is_periodic() * degree();
        }

        /** @brief TODO
         *
         * @return TODO
         */
        KOKKOS_INLINE_FUNCTION std::size_t ncells() const noexcept
        {
            return npoints() - 1;
        }

    private:
        KOKKOS_INLINE_FUNCTION int find_cell(ddc::Coordinate<Tag> const& x) const;
    };
};

template <class DDim>
struct is_non_uniform_bsplines : public std::is_base_of<detail::NonUniformBSplinesBase, DDim>
{
};

/**
 * @brief Indicates if a tag corresponds to uniform BSplines of not.
 *
 * @tparam The presumed non-uniform BSplines.
 */
template <class DDim>
constexpr bool is_non_uniform_bsplines_v = is_non_uniform_bsplines<DDim>::value;

template <class Tag, std::size_t D>
template <class DDim, class MemorySpace>
template <class RandomIt>
NonUniformBSplines<Tag, D>::Impl<DDim, MemorySpace>::Impl(
        RandomIt const break_begin,
        RandomIt const break_end)
    : m_domain(
            ddc::DiscreteElement<mesh_type>(0),
            ddc::DiscreteVector<mesh_type>(
                    (break_end - break_begin)
                    + 2 * degree())) // Create a mesh including the eventual periodic point
    , m_nknots((break_end - break_begin) + 2 * degree())
{
    std::vector<ddc::Coordinate<Tag>> knots((break_end - break_begin) + 2 * degree());
    // Fill the provided knots
    int ii = 0;
    for (RandomIt it = break_begin; it < break_end; ++it) {
        knots[degree() + ii] = *it;
        ++ii;
    }
    ddc::Coordinate<Tag> const rmin = knots[degree()];
    ddc::Coordinate<Tag> const rmax = knots[(break_end - break_begin) + degree() - 1];
    assert(rmin < rmax);

    // Fill out the extra knots
    if constexpr (is_periodic()) {
        double const period = rmax - rmin;
        for (std::size_t i = 1; i < degree() + 1; ++i) {
            knots[degree() + -i] = knots[degree() + ncells() - i] - period;
            knots[degree() + ncells() + i] = knots[degree() + i] + period;
        }
    } else // open
    {
        for (std::size_t i = 1; i < degree() + 1; ++i) {
            knots[degree() + -i] = rmin;
            knots[degree() + npoints() - 1 + i] = rmax;
        }
    }
    ddc::init_discrete_space<mesh_type>(knots);
}

template <class Tag, std::size_t D>
template <class DDim, class MemorySpace>
KOKKOS_INLINE_FUNCTION ddc::DiscreteElement<DDim> NonUniformBSplines<Tag, D>::
        Impl<DDim, MemorySpace>::eval_basis(DSpan1D values, ddc::Coordinate<Tag> const& x) const
{
    assert(values.size() == D + 1);

    std::array<double, degree()> left;
    std::array<double, degree()> right;

    assert(x >= rmin());
    assert(x <= rmax());
    assert(values.size() == degree() + 1);

    // 1. Compute cell index 'icell'
    int const icell = find_cell(x);

    assert(icell >= 0);
    assert(icell <= int(ncells() - 1));
    assert(get_knot(icell) <= x);
    assert(get_knot(icell + 1) >= x);

    // 2. Compute values of B-splines with support over cell 'icell'
    double temp;
    values[0] = 1.0;
    for (std::size_t j = 0; j < degree(); ++j) {
        left[j] = x - get_knot(icell - j);
        right[j] = get_knot(icell + j + 1) - x;
        double saved = 0.0;
        for (std::size_t r = 0; r < j + 1; ++r) {
            temp = values[r] / (right[r] + left[j - r]);
            values[r] = saved + right[r] * temp;
            saved = left[j - r] * temp;
        }
        values[j + 1] = saved;
    }

    return discrete_element_type(icell);
}

template <class Tag, std::size_t D>
template <class DDim, class MemorySpace>
KOKKOS_INLINE_FUNCTION ddc::DiscreteElement<DDim> NonUniformBSplines<Tag, D>::
        Impl<DDim, MemorySpace>::eval_deriv(DSpan1D derivs, ddc::Coordinate<Tag> const& x) const
{
    std::array<double, degree()> left;
    std::array<double, degree()> right;

    assert(x >= rmin());
    assert(x <= rmax());
    assert(derivs.size() == degree() + 1);

    // 1. Compute cell index 'icell'
    int const icell = find_cell(x);

    assert(icell >= 0);
    assert(icell <= int(ncells() - 1));
    assert(get_knot(icell) <= x);
    assert(get_knot(icell + 1) >= x);

    // 2. Compute values of derivatives of B-splines with support over cell 'icell'

    /*
     * Compute nonzero basis functions and knot differences
     * for splines up to degree degree-1 which are needed to compute derivative
     * First part of Algorithm  A3.2 of NURBS book
     */
    double saved, temp;
    derivs[0] = 1.0;
    for (std::size_t j = 0; j < degree() - 1; ++j) {
        left[j] = x - get_knot(icell - j);
        right[j] = get_knot(icell + j + 1) - x;
        saved = 0.0;
        for (std::size_t r = 0; r < j + 1; ++r) {
            temp = derivs[r] / (right[r] + left[j - r]);
            derivs[r] = saved + right[r] * temp;
            saved = left[j - r] * temp;
        }
        derivs[j + 1] = saved;
    }

    /*
     * Compute derivatives at x using values stored in bsdx and formula
     * for spline derivative based on difference of splines of degree degree-1
     */
    saved = degree() * derivs[0] / (get_knot(icell + 1) - get_knot(icell + 1 - degree()));
    derivs[0] = -saved;
    for (std::size_t j = 1; j < degree(); ++j) {
        temp = saved;
        saved = degree() * derivs[j]
                / (get_knot(icell + j + 1) - get_knot(icell + j + 1 - degree()));
        derivs[j] = temp - saved;
    }
    derivs[degree()] = saved;

    return discrete_element_type(icell);
}

template <class Tag, std::size_t D>
template <class DDim, class MemorySpace>
KOKKOS_INLINE_FUNCTION ddc::DiscreteElement<DDim> NonUniformBSplines<Tag, D>::
        Impl<DDim, MemorySpace>::eval_basis_and_n_derivs(
                ddc::DSpan2D const derivs,
                ddc::Coordinate<Tag> const& x,
                std::size_t const n) const
{
    std::array<double, degree()> left;
    std::array<double, degree()> right;

    std::array<double, 2 * (degree() + 1)> a_ptr;
    std::experimental::
            mdspan<double, std::experimental::extents<std::size_t, degree() + 1, 2>> const a(
                    a_ptr.data());

    std::array<double, (degree() + 1) * (degree() + 1)> ndu_ptr;
    std::experimental::mdspan<
            double,
            std::experimental::extents<std::size_t, degree() + 1, degree() + 1>> const
            ndu(ndu_ptr.data());

    assert(x >= rmin());
    assert(x <= rmax());
    // assert(n >= 0); as long as n is unsigned
    assert(n <= degree());
    assert(derivs.extent(0) == 1 + degree());
    assert(derivs.extent(1) == 1 + n);

    // 1. Compute cell index 'icell' and x_offset
    int const icell = find_cell(x);

    assert(icell >= 0);
    assert(icell <= int(ncells() - 1));
    assert(get_knot(icell) <= x);
    assert(get_knot(icell + 1) >= x);

    // 2. Compute nonzero basis functions and knot differences for splines
    //    up to degree (degree-1) which are needed to compute derivative
    //    Algorithm  A2.3 of NURBS book
    //
    //    21.08.2017: save inverse of knot differences to avoid unnecessary
    //    divisions
    //                [Yaman Güçlü, Edoardo Zoni]

    double saved, temp;
    ndu(0, 0) = 1.0;
    for (std::size_t j = 0; j < degree(); ++j) {
        left[j] = x - get_knot(icell - j);
        right[j] = get_knot(icell + j + 1) - x;
        saved = 0.0;
        for (std::size_t r = 0; r < j + 1; ++r) {
            // compute inverse of knot differences and save them into lower
            // triangular part of ndu
            ndu(r, j + 1) = 1.0 / (right[r] + left[j - r]);
            // compute basis functions and save them into upper triangular part
            // of ndu
            temp = ndu(j, r) * ndu(r, j + 1);
            ndu(j + 1, r) = saved + right[r] * temp;
            saved = left[j - r] * temp;
        }
        ndu(j + 1, j + 1) = saved;
    }
    // Save 0-th derivative
    for (std::size_t j = 0; j < degree() + 1; ++j) {
        derivs(j, 0) = ndu(degree(), j);
    }

    for (int r = 0; r < int(degree() + 1); ++r) {
        int s1 = 0;
        int s2 = 1;
        a(0, 0) = 1.0;
        for (int k = 1; k < int(n + 1); ++k) {
            double d = 0.0;
            int const rk = r - k;
            int const pk = degree() - k;
            if (r >= k) {
                a(0, s2) = a(0, s1) * ndu(rk, pk + 1);
                d = a(0, s2) * ndu(pk, rk);
            }
            int const j1 = rk > -1 ? 1 : (-rk);
            int const j2 = (r - 1) <= pk ? k : (degree() - r + 1);
            for (int j = j1; j < j2; ++j) {
                a(j, s2) = (a(j, s1) - a(j - 1, s1)) * ndu(rk + j, pk + 1);
                d += a(j, s2) * ndu(pk, rk + j);
            }
            if (r <= pk) {
                a(k, s2) = -a(k - 1, s1) * ndu(r, pk + 1);
                d += a(k, s2) * ndu(pk, r);
            }
            derivs(r, k) = d;
            Kokkos::kokkos_swap(s1, s2);
        }
    }

    int r = degree();
    for (int k = 1; k < int(n + 1); ++k) {
        for (std::size_t i = 0; i < derivs.extent(0); i++) {
            derivs(i, k) *= r;
        }
        r *= degree() - k;
    }

    return discrete_element_type(icell);
}

template <class Tag, std::size_t D>
template <class DDim, class MemorySpace>
KOKKOS_INLINE_FUNCTION int NonUniformBSplines<Tag, D>::Impl<DDim, MemorySpace>::find_cell(
        ddc::Coordinate<Tag> const& x) const
{
    if (x > rmax())
        return -1;
    if (x < rmin())
        return -1;

    if (x == rmin())
        return 0;
    if (x == rmax())
        return ncells() - 1;

    // Binary search
    int low = 0, high = ncells();
    int icell = (low + high) / 2;
    while (x < get_knot(icell) || x >= get_knot(icell + 1)) {
        if (x < get_knot(icell)) {
            high = icell;
        } else {
            low = icell;
        }
        icell = (low + high) / 2;
    }
    return icell;
}

template <class Tag, std::size_t D>
template <class DDim, class MemorySpace>
template <class Layout, class MemorySpace2>
KOKKOS_INLINE_FUNCTION ddc::ChunkSpan<double, typename NonUniformBSplines<Tag, D>::Impl<DDim, MemorySpace>::discrete_domain_type, Layout, MemorySpace2>
NonUniformBSplines<Tag, D>::Impl<DDim, MemorySpace>::integrals(
        ddc::ChunkSpan<double, typename NonUniformBSplines<Tag, D>::Impl<DDim, MemorySpace>::discrete_domain_type, Layout, MemorySpace2> int_vals) const
{
    if constexpr (is_periodic()) {
        assert(int_vals.size() == nbasis() || int_vals.size() == size());
    } else {
        assert(int_vals.size() == nbasis());
    }

    double const inv_deg = 1.0 / (degree() + 1);

    discrete_domain_type const dom_bsplines(
            full_domain().take_first(discrete_vector_type {nbasis()}));
    for (auto ix : dom_bsplines) {
        int_vals(ix) = (get_last_support_knot(ix) - get_first_support_knot(ix)) * inv_deg;
    }

    if constexpr (is_periodic()) {
        if (int_vals.size() == size()) {
            discrete_domain_type const dom_bsplines_wrap(
                    full_domain().take_last(discrete_vector_type {degree()}));
            for (auto ix : dom_bsplines_wrap) {
                int_vals(ix) = 0;
            }
        }
    }
    return int_vals;
}
} // namespace ddc
