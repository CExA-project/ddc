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

struct NonUniformBSplinesBase
{
};

} // namespace detail

template <class T>
struct NonUniformBsplinesKnots : NonUniformPointSampling<typename T::continuous_dimension_type>
{
};

/**
 * The type of a non-uniform 1D spline basis (B-spline).
 *
 * Knots for non-uniform B-splines are non-uniformly distributed (no assumption is made on the uniformity of their distribution,
 * the associated discrete dimension is a NonUniformPointSampling).
 *
 * @tparam CDim The tag identifying the continuous dimension on which the support of the B-spline functions are defined.
 * @tparam D The degree of the B-splines.
 */
template <class CDim, std::size_t D>
class NonUniformBSplines : detail::NonUniformBSplinesBase
{
    static_assert(D > 0, "Parameter `D` must be positive");

public:
    /// @brief The tag identifying the continuous dimension on which the support of the B-splines are defined.
    using continuous_dimension_type = CDim;

    /// @brief The discrete dimension identifying B-splines.
    using discrete_dimension_type = NonUniformBSplines;

    /** @brief The degree of B-splines.
     *
     * @return The degree.
     */
    static constexpr std::size_t degree() noexcept
    {
        return D;
    }

    /** @brief Indicates if the B-splines are periodic or not.
     *
     * @return A boolean indicating if the B-splines are periodic or not.
     */
    static constexpr bool is_periodic() noexcept
    {
        return CDim::PERIODIC;
    }

    /** @brief Indicates if the B-splines are uniform or not (this is not the case here).
     *
     * @return A boolean indicating if the B-splines are uniform or not.
     */
    static constexpr bool is_uniform() noexcept
    {
        return false;
    }

    /** @brief Storage class of the static attributes of the discrete dimension.
     *
     * @tparam DDim The name of the discrete dimension.
     * @tparam MemorySpace The Kokkos memory space where the attributes are being stored.
     */
    template <class DDim, class MemorySpace>
    class Impl
    {
        template <class ODDim, class OMemorySpace>
        friend class Impl;

    public:
        /// @brief The type of the knots defining the B-splines.
        using knot_discrete_dimension_type = NonUniformBsplinesKnots<DDim>;

        /// @brief The type of the discrete dimension representing the B-splines.
        using discrete_dimension_type = NonUniformBSplines;

        /// @brief The type of a DiscreteDomain whose elements identify the B-splines.
        using discrete_domain_type = DiscreteDomain<DDim>;

        /// @brief The type of a DiscreteElement identifying a B-spline.
        using discrete_element_type = DiscreteElement<DDim>;

        /// @brief The type of a DiscreteVector representing an "index displacement" between two B-splines.
        using discrete_vector_type = DiscreteVector<DDim>;

    private:
        ddc::DiscreteDomain<knot_discrete_dimension_type> m_knot_domain;
        ddc::DiscreteDomain<knot_discrete_dimension_type> m_break_point_domain;

        ddc::DiscreteElement<DDim> m_reference;

    public:
        Impl() = default;

        /** @brief Constructs an Impl using a brace-list, i.e. `Impl bsplines({0., 1.})`
         *
         * Constructs an Impl by iterating over a list of break points. Internally this constructor calls the constructor
         * Impl(RandomIt breaks_begin, RandomIt breaks_end).
         *
         * @param breaks The std::initializer_list of the coordinates of break points.
         */
        Impl(std::initializer_list<ddc::Coordinate<CDim>> breaks)
            : Impl(breaks.begin(), breaks.end())
        {
        }

        /** @brief Constructs an Impl using a std::vector.
         *
         * Constructs an Impl by iterating over a list of break points. Internally this constructor calls the constructor
         * Impl(RandomIt breaks_begin, RandomIt breaks_end).
         *
         * @param breaks The std::vector of the coordinates of break points.
         */
        explicit Impl(std::vector<ddc::Coordinate<CDim>> const& breaks)
            : Impl(breaks.begin(), breaks.end())
        {
        }

        /** @brief Constructs an Impl by iterating over a range of break points from begin to end.
         *
         * The provided break points describe the separation between the cells on which the polynomials
         * comprising a spline are defined. They are used to build a set of knots. There are 2*degree more
         * knots than break points. In the non-periodic case the knots are defined as follows:
         * \f$ k_i = b_0 \forall 0 \leq i < d \f$
         * \f$ k_{i+d} = b_i \forall 0 \leq i < n_b \f$
         * \f$ k_{i+d+n_b} = b_{n_b-1} \forall 0 \leq i < d \f$
         * where \f$d\f$ is the degree of the polynomials, and \f$n_b\f$ is the number of break points in the input pair of iterators. And in the periodic case:
         * \f$ k_i = b_{n_b-1-d+i} \forall 0 \leq i < d \f$
         * \f$ k_{i+d} = b_i \forall 0 \leq i \leq n_b \f$
         * \f$ k_{i+d+n_b} = b_{i+1} \forall 0 \leq i < d \f$
         *
         * This constructor makes the knots accessible via a DiscreteSpace.
         *
         * @param breaks_begin The iterator which points at the beginning of the break points.
         * @param breaks_end The iterator which points at the end of the break points.
         */
        template <class RandomIt>
        Impl(RandomIt breaks_begin, RandomIt breaks_end);

        /** @brief Copy-constructs from another Impl with a different Kokkos memory space.
         *
         * @param impl A reference to the other Impl.
         */
        template <class OriginMemorySpace>
        explicit Impl(Impl<DDim, OriginMemorySpace> const& impl)
            : m_knot_domain(impl.m_knot_domain)
            , m_break_point_domain(impl.m_break_point_domain)
            , m_reference(impl.m_reference)
        {
        }

        /** @brief Copy-constructs.
         *
         * @param x A reference to another Impl.
         */
        Impl(Impl const& x) = default;

        /** @brief Move-constructs.
         *
         * @param x An rvalue to another Impl.
         */
        Impl(Impl&& x) = default;

        /// @brief Destructs.
        ~Impl() = default;

        /** @brief Copy-assigns.
         *
         * @param x A reference to another Impl.
         * @return A reference to the copied Impl.
         */
        Impl& operator=(Impl const& x) = default;

        /** @brief Move-assigns.
         *
         * @param x An rvalue to another Impl.
         * @return A reference to this object.
         */
        Impl& operator=(Impl&& x) = default;

        /** @brief Evaluates non-zero B-splines at a given coordinate.
         *
         * The values are computed for every B-spline with support at the given coordinate x. There are only (degree+1)
         * B-splines which are non-zero at any given point. It is these B-splines which are evaluated.
         * This can be useful to calculate a spline approximation of a function. A spline approximation at coordinate x
         * is a linear combination of these B-spline evaluations weighted with the spline coefficients of the spline-transformed
         * initial discrete function.
         *
         * @param[out] values The values of the B-splines evaluated at coordinate x. It has to be a 1D mdspan with (degree+1) elements.
         * @param[in] x The coordinate where B-splines are evaluated. It has to be in the range of break points coordinates.
         * @return The index of the first B-spline which is evaluated.
         */
        KOKKOS_INLINE_FUNCTION discrete_element_type
        eval_basis(DSpan1D values, ddc::Coordinate<CDim> const& x) const;

        /** @brief Evaluates non-zero B-spline derivatives at a given coordinate
         *
         * The derivatives are computed for every B-spline with support at the given coordinate x. There are only (degree+1)
         * B-splines which are non-zero at any given point. It is these B-splines which are differentiated.
         * A spline approximation of a derivative at coordinate x is a linear
         * combination of those B-spline derivatives weighted with the spline coefficients of the spline-transformed
         * initial discrete function.
         *
         * @param[out] derivs The derivatives of the B-splines evaluated at coordinate x. It has to be a 1D mdspan with (degree+1) elements.
         * @param[in] x The coordinate where B-spline derivatives are evaluated. It has to be in the range of break points coordinates.
         * @return The index of the first B-spline which is differentiated.
         */
        KOKKOS_INLINE_FUNCTION discrete_element_type
        eval_deriv(DSpan1D derivs, ddc::Coordinate<CDim> const& x) const;

        /** @brief Evaluates non-zero B-spline values and \f$n\f$ derivatives at a given coordinate
         *
         * The values and derivatives are computed for every B-spline with support at the given coordinate x. There are only (degree+1)
         * B-splines which are non-zero at any given point. It is these B-splines which are evaluated and differentiated.
         * A spline approximation of a derivative at coordinate x is a linear
         * combination of those B-spline derivatives weighted with spline coefficients of the spline-transformed
         * initial discrete function.
         *
         * @param[out] derivs The values and \f$n\f$ derivatives of the B-splines evaluated at coordinate x. It has to be a 2D mdspan of sizes (degree+1, n+1).
         * @param[in] x The coordinate where B-spline derivatives are evaluated. It has to be in the range of break points coordinates.
         * @param[in] n The number of derivatives to evaluate (in addition to the B-spline values themselves).
         * @return The index of the first B-spline which is evaluated/derivated.
         */
        KOKKOS_INLINE_FUNCTION discrete_element_type eval_basis_and_n_derivs(
                ddc::DSpan2D derivs,
                ddc::Coordinate<CDim> const& x,
                std::size_t n) const;

        /** @brief Returns the coordinate of the first support knot associated to a DiscreteElement identifying a B-spline.
         *
         * Each B-spline has a support defined over (degree+2) knots. For a B-spline identified by the
         * provided DiscreteElement, this function returns the first knot in the support of the B-spline.
         * In other words it returns the lower bound of the support.
         *
         * @param[in] ix DiscreteElement identifying the B-spline.
         * @return DiscreteElement of the lower bound of the support of the B-spline.
         */
        KOKKOS_INLINE_FUNCTION ddc::DiscreteElement<knot_discrete_dimension_type>
        get_first_support_knot(discrete_element_type const& ix) const
        {
            return m_knot_domain.front() + (ix - m_reference).value();
        }

        /** @brief Returns the coordinate of the last support knot associated to a DiscreteElement identifying a B-spline.
         *
         * Each B-spline has a support defined over (degree+2) knots. For a B-spline identified by the
         * provided DiscreteElement, this function returns the last knot in the support of the B-spline.
         * In other words it returns the upper bound of the support.
         *
         * @param[in] ix DiscreteElement identifying the B-spline.
         * @return DiscreteElement of the upper bound of the support of the B-spline.
         */
        KOKKOS_INLINE_FUNCTION ddc::DiscreteElement<knot_discrete_dimension_type>
        get_last_support_knot(discrete_element_type const& ix) const
        {
            return get_first_support_knot(ix)
                   + ddc::DiscreteVector<knot_discrete_dimension_type>(degree() + 1);
        }

        /** @brief Returns the coordinate of the first break point of the domain on which the B-splines are defined.
         *
         * @return Coordinate of the lower bound of the domain.
         */
        KOKKOS_INLINE_FUNCTION ddc::Coordinate<CDim> rmin() const noexcept
        {
            return ddc::coordinate(m_break_point_domain.front());
        }

        /** @brief Returns the coordinate of the last break point of the domain on which the B-splines are defined.
         *
         * @return Coordinate of the upper bound of the domain.
         */
        KOKKOS_INLINE_FUNCTION ddc::Coordinate<CDim> rmax() const noexcept
        {
            return ddc::coordinate(m_break_point_domain.back());
        }

        /** @brief Returns the length of the domain.
         *
         * @return The length of the domain.
         */
        KOKKOS_INLINE_FUNCTION double length() const noexcept
        {
            return rmax() - rmin();
        }

        /** @brief Returns the number of elements necessary to construct a spline representation of a function.
         *
         * For a non-periodic domain the number of elements necessary to construct a spline representation of a function
         * is equal to the number of basis functions. However in the periodic case it additionally includes degree additional elements
         * which allow the first B-splines to be evaluated close to rmax (where they also appear due to the periodicity).
         *
         * @return The number of elements necessary to construct a spline representation of a function.
         */
        KOKKOS_INLINE_FUNCTION std::size_t size() const noexcept
        {
            return degree() + ncells();
        }

        /** @brief Returns the discrete domain including eventual additional B-splines in the periodic case. See size().
         *
         * @return The discrete domain including eventual additional B-splines.
         */
        KOKKOS_INLINE_FUNCTION discrete_domain_type full_domain() const
        {
            return discrete_domain_type(m_reference, discrete_vector_type(size()));
        }

        /** @brief Returns the discrete domain which describes the break points.
         *
         * @return The discrete domain describing the break points.
         */
        KOKKOS_INLINE_FUNCTION ddc::DiscreteDomain<knot_discrete_dimension_type>
        break_point_domain() const
        {
            return m_break_point_domain;
        }

        /** @brief The number of break points
         *
         * The number of break points or cell boundaries.
         *
         * @return The number of break points
         */
        KOKKOS_INLINE_FUNCTION std::size_t npoints() const noexcept
        {
            return m_knot_domain.size() - 2 * degree();
        }

        /** @brief Returns the number of basis functions.
         *
         * The number of functions in the spline basis.
         *
         * @return The number of basis functions.
         */
        KOKKOS_INLINE_FUNCTION std::size_t nbasis() const noexcept
        {
            return ncells() + !is_periodic() * degree();
        }

        /** @brief Returns the number of cells over which the B-splines are defined.
         *
         * The number of cells over which the B-splines and any spline representation are defined.
         * In other words the number of polynomials that comprise a spline representation on the domain where the basis is defined.
         *
         * @return The number of cells over which the B-splines are defined.
         */
        KOKKOS_INLINE_FUNCTION std::size_t ncells() const noexcept
        {
            return npoints() - 1;
        }

    private:
        KOKKOS_INLINE_FUNCTION discrete_element_type get_first_bspline_in_cell(
                ddc::DiscreteElement<knot_discrete_dimension_type> const& ic) const
        {
            return m_reference + (ic - m_break_point_domain.front()).value();
        }

        /**
         * @brief Get the DiscreteElement describing the knot at the start of the cell where x is found.
         * @param x The point whose location must be determined.
         * @returns The DiscreteElement describing the knot at the lower bound of the cell of interest.
         */
        KOKKOS_INLINE_FUNCTION ddc::DiscreteElement<knot_discrete_dimension_type> find_cell_start(
                ddc::Coordinate<CDim> const& x) const;
    };
};

template <class DDim>
struct is_non_uniform_bsplines : public std::is_base_of<detail::NonUniformBSplinesBase, DDim>::type
{
};

/**
 * @brief Indicates if a tag corresponds to non-uniform B-splines or not.
 *
 * @tparam The presumed non-uniform B-splines.
 */
template <class DDim>
constexpr bool is_non_uniform_bsplines_v = is_non_uniform_bsplines<DDim>::value;

template <class CDim, std::size_t D>
template <class DDim, class MemorySpace>
template <class RandomIt>
NonUniformBSplines<CDim, D>::Impl<DDim, MemorySpace>::Impl(
        RandomIt const breaks_begin,
        RandomIt const breaks_end)
    : m_knot_domain(
              ddc::DiscreteElement<knot_discrete_dimension_type>(0),
              ddc::DiscreteVector<knot_discrete_dimension_type>(
                      (breaks_end - breaks_begin)
                      + 2 * degree())) // Create a mesh of knots including the eventual periodic point
    , m_break_point_domain(
              ddc::DiscreteElement<knot_discrete_dimension_type>(degree()),
              ddc::DiscreteVector<knot_discrete_dimension_type>(
                      (breaks_end - breaks_begin))) // Create a mesh of break points
    , m_reference(ddc::create_reference_discrete_element<DDim>())
{
    std::vector<ddc::Coordinate<CDim>> knots((breaks_end - breaks_begin) + 2 * degree());
    // Fill the provided knots
    int ii = 0;
    for (RandomIt it = breaks_begin; it < breaks_end; ++it) {
        knots[degree() + ii] = *it;
        ++ii;
    }
    ddc::Coordinate<CDim> const rmin = knots[degree()];
    ddc::Coordinate<CDim> const rmax = knots[(breaks_end - breaks_begin) + degree() - 1];
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
    ddc::init_discrete_space<knot_discrete_dimension_type>(knots);
}

template <class CDim, std::size_t D>
template <class DDim, class MemorySpace>
KOKKOS_INLINE_FUNCTION ddc::DiscreteElement<DDim> NonUniformBSplines<CDim, D>::
        Impl<DDim, MemorySpace>::eval_basis(DSpan1D values, ddc::Coordinate<CDim> const& x) const
{
    assert(values.size() == D + 1);

    std::array<double, degree()> left;
    std::array<double, degree()> right;

    assert(x - rmin() >= -length() * 1e-14);
    assert(rmax() - x >= -length() * 1e-14);
    assert(values.size() == degree() + 1);

    // 1. Compute cell index 'icell'
    ddc::DiscreteElement<knot_discrete_dimension_type> const icell = find_cell_start(x);

    assert(icell >= m_break_point_domain.front());
    assert(icell <= m_break_point_domain.back());
    assert(ddc::coordinate(icell) - x <= length() * 1e-14);
    assert(x - ddc::coordinate(icell + 1) <= length() * 1e-14);

    // 2. Compute values of B-splines with support over cell 'icell'
    double temp;
    values[0] = 1.0;
    for (std::size_t j = 0; j < degree(); ++j) {
        left[j] = x - ddc::coordinate(icell - j);
        right[j] = ddc::coordinate(icell + j + 1) - x;
        double saved = 0.0;
        for (std::size_t r = 0; r < j + 1; ++r) {
            temp = values[r] / (right[r] + left[j - r]);
            values[r] = saved + right[r] * temp;
            saved = left[j - r] * temp;
        }
        values[j + 1] = saved;
    }

    return get_first_bspline_in_cell(icell);
}

template <class CDim, std::size_t D>
template <class DDim, class MemorySpace>
KOKKOS_INLINE_FUNCTION ddc::DiscreteElement<DDim> NonUniformBSplines<CDim, D>::
        Impl<DDim, MemorySpace>::eval_deriv(DSpan1D derivs, ddc::Coordinate<CDim> const& x) const
{
    std::array<double, degree()> left;
    std::array<double, degree()> right;

    assert(x - rmin() >= -length() * 1e-14);
    assert(rmax() - x >= -length() * 1e-14);
    assert(derivs.size() == degree() + 1);

    // 1. Compute cell index 'icell'
    ddc::DiscreteElement<knot_discrete_dimension_type> const icell = find_cell_start(x);

    assert(icell >= m_break_point_domain.front());
    assert(icell <= m_break_point_domain.back());
    assert(ddc::coordinate(icell) <= x);
    assert(ddc::coordinate(icell + 1) >= x);

    // 2. Compute values of derivatives of B-splines with support over cell 'icell'

    /*
     * Compute nonzero basis functions and knot differences
     * for splines up to degree degree-1 which are needed to compute derivative
     * First part of Algorithm  A3.2 of NURBS book
     */
    double saved;
    double temp;
    derivs[0] = 1.0;
    for (std::size_t j = 0; j < degree() - 1; ++j) {
        left[j] = x - ddc::coordinate(icell - j);
        right[j] = ddc::coordinate(icell + j + 1) - x;
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
    saved = degree() * derivs[0]
            / (ddc::coordinate(icell + 1) - ddc::coordinate(icell + 1 - degree()));
    derivs[0] = -saved;
    for (std::size_t j = 1; j < degree(); ++j) {
        temp = saved;
        saved = degree() * derivs[j]
                / (ddc::coordinate(icell + j + 1) - ddc::coordinate(icell + j + 1 - degree()));
        derivs[j] = temp - saved;
    }
    derivs[degree()] = saved;

    return get_first_bspline_in_cell(icell);
}

template <class CDim, std::size_t D>
template <class DDim, class MemorySpace>
KOKKOS_INLINE_FUNCTION ddc::DiscreteElement<DDim> NonUniformBSplines<CDim, D>::
        Impl<DDim, MemorySpace>::eval_basis_and_n_derivs(
                ddc::DSpan2D const derivs,
                ddc::Coordinate<CDim> const& x,
                std::size_t const n) const
{
    std::array<double, degree()> left;
    std::array<double, degree()> right;

    std::array<double, 2 * (degree() + 1)> a_ptr;
    Kokkos::mdspan<double, Kokkos::extents<std::size_t, degree() + 1, 2>> const a(a_ptr.data());

    std::array<double, (degree() + 1) * (degree() + 1)> ndu_ptr;
    Kokkos::mdspan<double, Kokkos::extents<std::size_t, degree() + 1, degree() + 1>> const ndu(
            ndu_ptr.data());

    assert(x - rmin() >= -length() * 1e-14);
    assert(rmax() - x >= -length() * 1e-14);
    // assert(n >= 0); as long as n is unsigned
    assert(n <= degree());
    assert(derivs.extent(0) == 1 + degree());
    assert(derivs.extent(1) == 1 + n);

    // 1. Compute cell index 'icell' and x_offset
    ddc::DiscreteElement<knot_discrete_dimension_type> const icell = find_cell_start(x);

    assert(icell >= m_break_point_domain.front());
    assert(icell <= m_break_point_domain.back());
    assert(ddc::coordinate(icell) <= x);
    assert(ddc::coordinate(icell + 1) >= x);

    // 2. Compute nonzero basis functions and knot differences for splines
    //    up to degree (degree-1) which are needed to compute derivative
    //    Algorithm  A2.3 of NURBS book
    //
    //    21.08.2017: save inverse of knot differences to avoid unnecessary
    //    divisions
    //                [Yaman Güçlü, Edoardo Zoni]

    double saved;
    double temp;
    DDC_MDSPAN_ACCESS_OP(ndu, 0, 0) = 1.0;
    for (std::size_t j = 0; j < degree(); ++j) {
        left[j] = x - ddc::coordinate(icell - j);
        right[j] = ddc::coordinate(icell + j + 1) - x;
        saved = 0.0;
        for (std::size_t r = 0; r < j + 1; ++r) {
            // compute inverse of knot differences and save them into lower
            // triangular part of ndu
            DDC_MDSPAN_ACCESS_OP(ndu, r, j + 1) = 1.0 / (right[r] + left[j - r]);
            // compute basis functions and save them into upper triangular part
            // of ndu
            temp = DDC_MDSPAN_ACCESS_OP(ndu, j, r) * DDC_MDSPAN_ACCESS_OP(ndu, r, j + 1);
            DDC_MDSPAN_ACCESS_OP(ndu, j + 1, r) = saved + right[r] * temp;
            saved = left[j - r] * temp;
        }
        DDC_MDSPAN_ACCESS_OP(ndu, j + 1, j + 1) = saved;
    }
    // Save 0-th derivative
    for (std::size_t j = 0; j < degree() + 1; ++j) {
        DDC_MDSPAN_ACCESS_OP(derivs, j, 0) = DDC_MDSPAN_ACCESS_OP(ndu, degree(), j);
    }

    for (int r = 0; r < int(degree() + 1); ++r) {
        int s1 = 0;
        int s2 = 1;
        DDC_MDSPAN_ACCESS_OP(a, 0, 0) = 1.0;
        for (int k = 1; k < int(n + 1); ++k) {
            double d = 0.0;
            int const rk = r - k;
            int const pk = degree() - k;
            if (r >= k) {
                DDC_MDSPAN_ACCESS_OP(a, 0, s2)
                        = DDC_MDSPAN_ACCESS_OP(a, 0, s1) * DDC_MDSPAN_ACCESS_OP(ndu, rk, pk + 1);
                d = DDC_MDSPAN_ACCESS_OP(a, 0, s2) * DDC_MDSPAN_ACCESS_OP(ndu, pk, rk);
            }
            int const j1 = rk > -1 ? 1 : (-rk);
            int const j2 = (r - 1) <= pk ? k : (degree() - r + 1);
            for (int j = j1; j < j2; ++j) {
                DDC_MDSPAN_ACCESS_OP(a, j, s2)
                        = (DDC_MDSPAN_ACCESS_OP(a, j, s1) - DDC_MDSPAN_ACCESS_OP(a, j - 1, s1))
                          * DDC_MDSPAN_ACCESS_OP(ndu, rk + j, pk + 1);
                d += DDC_MDSPAN_ACCESS_OP(a, j, s2) * DDC_MDSPAN_ACCESS_OP(ndu, pk, rk + j);
            }
            if (r <= pk) {
                DDC_MDSPAN_ACCESS_OP(a, k, s2) = -DDC_MDSPAN_ACCESS_OP(a, k - 1, s1)
                                                 * DDC_MDSPAN_ACCESS_OP(ndu, r, pk + 1);
                d += DDC_MDSPAN_ACCESS_OP(a, k, s2) * DDC_MDSPAN_ACCESS_OP(ndu, pk, r);
            }
            DDC_MDSPAN_ACCESS_OP(derivs, r, k) = d;
            Kokkos::kokkos_swap(s1, s2);
        }
    }

    int r = degree();
    for (int k = 1; k < int(n + 1); ++k) {
        for (std::size_t i = 0; i < derivs.extent(0); ++i) {
            DDC_MDSPAN_ACCESS_OP(derivs, i, k) *= r;
        }
        r *= degree() - k;
    }

    return get_first_bspline_in_cell(icell);
}

template <class CDim, std::size_t D>
template <class DDim, class MemorySpace>
KOKKOS_INLINE_FUNCTION ddc::DiscreteElement<NonUniformBsplinesKnots<DDim>> NonUniformBSplines<
        CDim,
        D>::Impl<DDim, MemorySpace>::find_cell_start(ddc::Coordinate<CDim> const& x) const
{
    assert(x - rmin() >= -length() * 1e-14);
    assert(rmax() - x >= -length() * 1e-14);

    if (x <= rmin()) {
        return m_break_point_domain.front();
    }
    if (x >= rmax()) {
        return m_break_point_domain.back() - 1;
    }

    // Binary search
    ddc::DiscreteElement<knot_discrete_dimension_type> low = m_break_point_domain.front();
    ddc::DiscreteElement<knot_discrete_dimension_type> high = m_break_point_domain.back();
    ddc::DiscreteElement<knot_discrete_dimension_type> icell = low + (high - low) / 2;
    while (x < ddc::coordinate(icell) || x >= ddc::coordinate(icell + 1)) {
        if (x < ddc::coordinate(icell)) {
            high = icell;
        } else {
            low = icell;
        }
        icell = low + (high - low) / 2;
    }
    return icell;
}

} // namespace ddc
