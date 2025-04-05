// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cassert>
#include <memory>
#include <tuple>

#include <ddc/ddc.hpp>

#include "math_tools.hpp"
#include "view.hpp"

namespace ddc {

namespace detail {

struct UniformBSplinesBase
{
};

template <class ExecSpace, class ODDim, class Layout, class OMemorySpace>
void uniform_bsplines_integrals(
        ExecSpace const& execution_space,
        ddc::ChunkSpan<double, ddc::DiscreteDomain<ODDim>, Layout, OMemorySpace> int_vals);

} // namespace detail

template <class T>
struct UniformBsplinesKnots : UniformPointSampling<typename T::continuous_dimension_type>
{
};

/**
 * The type of a uniform 1D spline basis (B-spline).
 *
 * Knots for uniform B-splines are uniformly distributed (the associated discrete dimension
 * is a UniformPointSampling).
 *
 * @tparam CDim The tag identifying the continuous dimension on which the support of the B-spline functions are defined.
 * @tparam D The degree of the B-splines.
 */
template <class CDim, std::size_t D>
class UniformBSplines : detail::UniformBSplinesBase
{
    static_assert(D > 0, "Parameter `D` must be positive");

public:
    /// @brief The tag identifying the continuous dimension on which the support of the B-splines are defined.
    using continuous_dimension_type = CDim;

    /// @brief The discrete dimension representing B-splines.
    using discrete_dimension_type = UniformBSplines;

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

    /** @brief Indicates if the B-splines are uniform or not (this is the case here).
     *
     * @return A boolean indicating if the B-splines are uniform or not.
     */
    static constexpr bool is_uniform() noexcept
    {
        return true;
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

        template <class ExecSpace, class ODDim, class Layout, class OMemorySpace>
        friend void detail::uniform_bsplines_integrals(
                ExecSpace const& execution_space,
                ddc::ChunkSpan<double, ddc::DiscreteDomain<ODDim>, Layout, OMemorySpace> int_vals);

    public:
        /// @brief The type of the knots defining the B-splines.
        using knot_discrete_dimension_type = UniformBsplinesKnots<DDim>;

        /// @brief The type of the discrete dimension representing the B-splines.
        using discrete_dimension_type = UniformBSplines;

        /// @brief The type of a DiscreteDomain whose elements identify the B-splines.
        using discrete_domain_type = DiscreteDomain<DDim>;

        /// @brief The type of a DiscreteElement identifying a B-spline.
        using discrete_element_type = DiscreteElement<DDim>;

        /// @brief The type of a DiscreteVector representing an "index displacement" between two B-splines.
        using discrete_vector_type = DiscreteVector<DDim>;

    private:
        // In the periodic case, they contain the periodic point twice!!!
        ddc::DiscreteDomain<knot_discrete_dimension_type> m_knot_domain;
        ddc::DiscreteDomain<knot_discrete_dimension_type> m_break_point_domain;

        ddc::DiscreteElement<DDim> m_reference;

    public:
        Impl() = default;

        /** Constructs a spline basis (B-splines) with n equidistant knots over \f$[a, b]\f$.
         *
         * @param rmin    The real ddc::coordinate of the first knot.
         * @param rmax    The real ddc::coordinate of the last knot.
         * @param ncells The number of cells in the range [rmin, rmax].
         */
        explicit Impl(ddc::Coordinate<CDim> rmin, ddc::Coordinate<CDim> rmax, std::size_t ncells)
            : m_reference(ddc::create_reference_discrete_element<DDim>())
        {
            assert(ncells > 0);
            std::tie(m_break_point_domain, m_knot_domain, std::ignore, std::ignore)
                    = ddc::init_discrete_space<knot_discrete_dimension_type>(
                            knot_discrete_dimension_type::template init_ghosted<
                                    knot_discrete_dimension_type>(
                                    rmin,
                                    rmax,
                                    ddc::DiscreteVector<knot_discrete_dimension_type>(ncells + 1),
                                    ddc::DiscreteVector<knot_discrete_dimension_type>(degree()),
                                    ddc::DiscreteVector<knot_discrete_dimension_type>(degree())));
        }

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
        eval_basis(DSpan1D values, ddc::Coordinate<CDim> const& x) const
        {
            assert(values.size() == degree() + 1);
            return eval_basis(values, x, degree());
        }

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
         * @return The index of the first B-spline which is evaluated.
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
         * @return The index of the first B-spline which is evaluated.
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

        /** @brief Returns the coordinate of the lower bound of the domain on which the B-splines are defined.
         *
         * @return Coordinate of the lower bound of the domain.
         */
        KOKKOS_INLINE_FUNCTION ddc::Coordinate<CDim> rmin() const noexcept
        {
            return ddc::coordinate(m_break_point_domain.front());
        }

        /** @brief Returns the coordinate of the upper bound of the domain on which the B-splines are defined.
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
            return m_break_point_domain.size() - 1;
        }

    private:
        KOKKOS_INLINE_FUNCTION double inv_step() const noexcept
        {
            return 1.0 / ddc::step<knot_discrete_dimension_type>();
        }

        KOKKOS_INLINE_FUNCTION discrete_element_type
        eval_basis(DSpan1D values, ddc::Coordinate<CDim> const& x, std::size_t degree) const;

        KOKKOS_INLINE_FUNCTION void get_icell_and_offset(
                int& icell,
                double& offset,
                ddc::Coordinate<CDim> const& x) const;
    };
};

template <class DDim>
struct is_uniform_bsplines : public std::is_base_of<detail::UniformBSplinesBase, DDim>::type
{
};

/**
 * @brief Indicates if a tag corresponds to uniform B-splines or not.
 *
 * @tparam The presumed uniform B-splines.
 */
template <class DDim>
constexpr bool is_uniform_bsplines_v = is_uniform_bsplines<DDim>::value;

template <class CDim, std::size_t D>
template <class DDim, class MemorySpace>
KOKKOS_INLINE_FUNCTION ddc::DiscreteElement<DDim> UniformBSplines<CDim, D>::
        Impl<DDim, MemorySpace>::eval_basis(
                DSpan1D values,
                ddc::Coordinate<CDim> const& x,
                [[maybe_unused]] std::size_t const degree) const
{
    assert(values.size() == degree + 1);

    double offset;
    int jmin;
    // 1. Compute cell index 'icell' and x_offset
    // 2. Compute index range of B-splines with support over cell 'icell'
    get_icell_and_offset(jmin, offset, x);

    // 3. Compute values of aforementioned B-splines
    double xx;
    double temp;
    double saved;
    DDC_MDSPAN_ACCESS_OP(values, 0) = 1.0;
    for (std::size_t j = 1; j < values.size(); ++j) {
        xx = -offset;
        saved = 0.0;
        for (std::size_t r = 0; r < j; ++r) {
            xx += 1;
            temp = DDC_MDSPAN_ACCESS_OP(values, r) / j;
            DDC_MDSPAN_ACCESS_OP(values, r) = saved + xx * temp;
            saved = (j - xx) * temp;
        }
        DDC_MDSPAN_ACCESS_OP(values, j) = saved;
    }

    return m_reference + jmin;
}

template <class CDim, std::size_t D>
template <class DDim, class MemorySpace>
KOKKOS_INLINE_FUNCTION ddc::DiscreteElement<DDim> UniformBSplines<CDim, D>::
        Impl<DDim, MemorySpace>::eval_deriv(DSpan1D derivs, ddc::Coordinate<CDim> const& x) const
{
    assert(derivs.size() == degree() + 1);

    double offset;
    int jmin;
    // 1. Compute cell index 'icell' and x_offset
    // 2. Compute index range of B-splines with support over cell 'icell'
    get_icell_and_offset(jmin, offset, x);

    // 3. Compute derivatives of aforementioned B-splines
    //    Derivatives are normalized, hence they should be divided by dx
    double xx;
    double temp;
    double saved;
    DDC_MDSPAN_ACCESS_OP(derivs, 0) = 1.0 / ddc::step<knot_discrete_dimension_type>();
    for (std::size_t j = 1; j < degree(); ++j) {
        xx = -offset;
        saved = 0.0;
        for (std::size_t r = 0; r < j; ++r) {
            xx += 1.0;
            temp = DDC_MDSPAN_ACCESS_OP(derivs, r) / j;
            DDC_MDSPAN_ACCESS_OP(derivs, r) = saved + xx * temp;
            saved = (j - xx) * temp;
        }
        DDC_MDSPAN_ACCESS_OP(derivs, j) = saved;
    }

    // Compute derivatives
    double bjm1 = derivs[0];
    double bj = bjm1;
    DDC_MDSPAN_ACCESS_OP(derivs, 0) = -bjm1;
    for (std::size_t j = 1; j < degree(); ++j) {
        bj = DDC_MDSPAN_ACCESS_OP(derivs, j);
        DDC_MDSPAN_ACCESS_OP(derivs, j) = bjm1 - bj;
        bjm1 = bj;
    }
    DDC_MDSPAN_ACCESS_OP(derivs, degree()) = bj;

    return m_reference + jmin;
}

template <class CDim, std::size_t D>
template <class DDim, class MemorySpace>
KOKKOS_INLINE_FUNCTION ddc::DiscreteElement<DDim> UniformBSplines<CDim, D>::
        Impl<DDim, MemorySpace>::eval_basis_and_n_derivs(
                ddc::DSpan2D const derivs,
                ddc::Coordinate<CDim> const& x,
                std::size_t const n) const
{
    std::array<double, (degree() + 1) * (degree() + 1)> ndu_ptr;
    Kokkos::mdspan<double, Kokkos::extents<std::size_t, degree() + 1, degree() + 1>> const ndu(
            ndu_ptr.data());
    std::array<double, 2 * (degree() + 1)> a_ptr;
    Kokkos::mdspan<double, Kokkos::extents<std::size_t, degree() + 1, 2>> const a(a_ptr.data());
    double offset;
    int jmin;

    assert(x - rmin() >= -length() * 1e-14);
    assert(rmax() - x >= -length() * 1e-14);
    // assert(n >= 0); as long as n is unsigned
    assert(n <= degree());
    assert(derivs.extent(0) == 1 + degree());
    assert(derivs.extent(1) == 1 + n);

    // 1. Compute cell index 'icell' and x_offset
    // 2. Compute index range of B-splines with support over cell 'icell'
    get_icell_and_offset(jmin, offset, x);

    // 3. Recursively evaluate B-splines (eval_basis)
    //    up to self%degree, and store them all in the upper-right triangle of
    //    ndu
    double xx;
    double temp;
    double saved;
    DDC_MDSPAN_ACCESS_OP(ndu, 0, 0) = 1.0;
    for (std::size_t j = 1; j < degree() + 1; ++j) {
        xx = -offset;
        saved = 0.0;
        for (std::size_t r = 0; r < j; ++r) {
            xx += 1.0;
            temp = DDC_MDSPAN_ACCESS_OP(ndu, j - 1, r) / j;
            DDC_MDSPAN_ACCESS_OP(ndu, j, r) = saved + xx * temp;
            saved = (j - xx) * temp;
        }
        DDC_MDSPAN_ACCESS_OP(ndu, j, j) = saved;
    }
    for (std::size_t i = 0; i < ndu.extent(1); ++i) {
        DDC_MDSPAN_ACCESS_OP(derivs, i, 0) = DDC_MDSPAN_ACCESS_OP(ndu, degree(), i);
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
                DDC_MDSPAN_ACCESS_OP(a, 0, s2) = DDC_MDSPAN_ACCESS_OP(a, 0, s1) / (pk + 1);
                d = DDC_MDSPAN_ACCESS_OP(a, 0, s2) * DDC_MDSPAN_ACCESS_OP(ndu, pk, rk);
            }
            int const j1 = rk > -1 ? 1 : (-rk);
            int const j2 = (r - 1) <= pk ? k : (degree() - r + 1);
            for (int j = j1; j < j2; ++j) {
                DDC_MDSPAN_ACCESS_OP(a, j, s2)
                        = (DDC_MDSPAN_ACCESS_OP(a, j, s1) - DDC_MDSPAN_ACCESS_OP(a, j - 1, s1))
                          / (pk + 1);
                d += DDC_MDSPAN_ACCESS_OP(a, j, s2) * DDC_MDSPAN_ACCESS_OP(ndu, pk, rk + j);
            }
            if (r <= pk) {
                DDC_MDSPAN_ACCESS_OP(a, k, s2) = -DDC_MDSPAN_ACCESS_OP(a, k - 1, s1) / (pk + 1);
                d += DDC_MDSPAN_ACCESS_OP(a, k, s2) * DDC_MDSPAN_ACCESS_OP(ndu, pk, r);
            }
            DDC_MDSPAN_ACCESS_OP(derivs, r, k) = d;
            Kokkos::kokkos_swap(s1, s2);
        }
    }

    // Multiply result by correct factors:
    // degree!/(degree-n)! = degree*(degree-1)*...*(degree-n+1)
    // k-th derivatives are normalized, hence they should be divided by dx^k
    double const inv_dx = inv_step();
    double d = degree() * inv_dx;
    for (int k = 1; k < int(n + 1); ++k) {
        for (std::size_t i = 0; i < derivs.extent(0); ++i) {
            DDC_MDSPAN_ACCESS_OP(derivs, i, k) *= d;
        }
        d *= (degree() - k) * inv_dx;
    }

    return m_reference + jmin;
}

template <class CDim, std::size_t D>
template <class DDim, class MemorySpace>
KOKKOS_INLINE_FUNCTION void UniformBSplines<CDim, D>::Impl<DDim, MemorySpace>::get_icell_and_offset(
        int& icell,
        double& offset,
        ddc::Coordinate<CDim> const& x) const
{
    assert(x - rmin() >= -length() * 1e-14);
    assert(rmax() - x >= -length() * 1e-14);

    double const inv_dx = inv_step();
    if (x <= rmin()) {
        icell = 0;
        offset = 0.0;
    } else if (x >= rmax()) {
        icell = ncells() - 1;
        offset = 1.0;
    } else {
        offset = (x - rmin()) * inv_dx;
        icell = static_cast<int>(offset);
        offset = offset - icell;

        // When x is very close to xmax, round-off may cause the wrong answer
        // icell=ncells and x_offset=0, which we convert to the case x=xmax:
        if (icell == int(ncells()) && offset == 0.0) {
            icell = ncells() - 1;
            offset = 1.0;
        }
    }
}

} // namespace ddc
