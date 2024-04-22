// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cassert>
#include <memory>

#include <ddc/ddc.hpp>

#include "math_tools.hpp"
#include "view.hpp"

namespace ddc {

namespace detail {

template <class T>
struct UniformBsplinesKnots : UniformPointSampling<typename T::tag_type>
{
};

struct UniformBSplinesBase
{
};

} // namespace detail

/**
 * The type of a uniform B-splines 1D basis.
 *
 * @tparam Tag The tag identifying the continuous dimension on which the support of the B-spline functions are defined.
 * @tparam D The degree of the B-splines.
 */
template <class Tag, std::size_t D>
class UniformBSplines : detail::UniformBSplinesBase
{
    static_assert(D > 0, "Parameter `D` must be positive");

public:
    /// @brief The tag identifying the continuous dimension on which the support of the B-splines are defined.
    using tag_type = Tag;

    /// @brief The discrete dimension representing B-splines.
    using discrete_dimension_type = UniformBSplines;

    /// @brief The degree of B-splines.
    static constexpr std::size_t degree() noexcept
    {
        return D;
    }

    /// @brief Indicates if the B-splines are periodic or not.
    static constexpr bool is_periodic() noexcept
    {
        return Tag::PERIODIC;
    }

    /// @brief Indicates if the B-splines are radial or not (should be deprecated soon because this concept is not in the scope of DDC).
    [[deprecated]] static constexpr bool is_radial() noexcept
    {
        return false;
    }

    /// @brief Indicates if the B-splines are uniform or not (this is the case here).
    static constexpr bool is_uniform() noexcept
    {
        return true;
    }

    /** @brief Impl Storage class of the static attributes of the discrete dimension.
     *
     * @tparam DDim The name of the discrete dimension.
     * @tparam MemorySpace The Kokkos memory space where the attributes are being stored.
     */
    template <class DDim, class MemorySpace>
    class Impl
    {
        template <class ODDim, class OMemorySpace>
        friend class Impl;

    private:
        using mesh_type = detail::UniformBsplinesKnots<DDim>;

        // In the periodic case, it contains twice the periodic point!!!
        ddc::DiscreteDomain<mesh_type> m_domain;

    public:
        /// @brief The type of the discrete dimension representing the B-splines.
        using discrete_dimension_type = UniformBSplines;

        /// @brief The type of a DiscreteDomain whose elements identify the B-splines.
        using discrete_domain_type = DiscreteDomain<DDim>;

        /// @brief The type of a DiscreteElement identifying a B-spline.
        using discrete_element_type = DiscreteElement<DDim>;

        /// @brief The type of a DiscreteVector representing an "index displacement" between two B-splines.
        using discrete_vector_type = DiscreteVector<DDim>;

        Impl() = default;

        /** Constructs a B-splines basis with n equidistant knots over \f$[a, b]\f$
         *
         * @param rmin    the real ddc::coordinate of the first knot
         * @param rmax    the real ddc::coordinate of the last knot
         * @param ncells the number of cells in the range [rmin, rmax]
         */
        explicit Impl(ddc::Coordinate<Tag> rmin, ddc::Coordinate<Tag> rmax, std::size_t ncells)
            : m_domain(
                    ddc::DiscreteElement<mesh_type>(0),
                    ddc::DiscreteVector<mesh_type>(
                            ncells + 1)) // Create a mesh including the eventual periodic point
        {
            assert(ncells > 0);
            ddc::init_discrete_space<mesh_type>(
                    mesh_type::template init<
                            mesh_type>(rmin, rmax, ddc::DiscreteVector<mesh_type>(ncells + 1)));
        }

        /// @brief Copy-constructs from another Impl with different Kokkos memory space
        template <class OriginMemorySpace>
        explicit Impl(Impl<DDim, OriginMemorySpace> const& impl) : m_domain(impl.m_domain)
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

        /** @brief Evaluates non-zero B-splines at a given coordinate.
         *
         * The values are computed for every B-spline with support at the given coordinate x. There are only (degree+1)
         * B-splines which are non-zero at any given point. It is these B-splines which are evaluated.
         * This can be useful to calculate a spline approximation of a function. A spline approximation at coordinate x
         * is a linear combination of these B-spline evaluations weighted with spline coefficients of the spline-transformed
         * initial discrete function.
         *
         * @param[out] values The values of the B-splines evaluated at coordinate x. It has to be a 1D mdspan with (degree+1) elements.
         * @param[in] x The coordinate where B-splines are evaluated.
         * @return The index of the first B-spline which is evaluated.
         */
        KOKKOS_INLINE_FUNCTION discrete_element_type
        eval_basis(DSpan1D values, ddc::Coordinate<Tag> const& x) const
        {
            assert(values.size() == degree() + 1);
            return eval_basis(values, x, degree());
        }

        /** @brief Evaluates non-zero B-splines derivatives at a given coordinate
         *
         * The derivatives are computed for every B-splines with support at the given coordinate x. There are only (degree+1)
         * B-splines which are non-zero at any given point. It is these B-splines which are derivated.
         * A spline approximation of a derivative at coordinate x is a linear
         * combination of those B-splines derivatives weigthed with splines coefficients of the spline-transformed
         * initial discrete function.
         *
         * @param[out] derivs The derivatives of the B-splines evaluated at coordinate x. It has to be a 1D mdspan with (degree+1) elements.
         * @param[in] x The coordinate where B-splines derivatives are evaluated.
         * @return The index of the first B-spline which is evaluated.
         */
        KOKKOS_INLINE_FUNCTION discrete_element_type
        eval_deriv(DSpan1D derivs, ddc::Coordinate<Tag> const& x) const;

        /** @brief Evaluates non-zero B-splines values and \f$n\f$ derivatives at a given coordinate
         *
         * The values and derivatives are computed for every B-splines with support at the given coordinate x. There are only (degree+1)
         * B-splines which are non-zero at any given point. It is these B-splines which are derivated.
         * A spline approximation of a derivative at coordinate x is a linear
         * combination of those B-splines derivatives weigthed with splines coefficients of the spline-transformed
         * initial discrete function.
         *
         * @param[out] derivs The values and \f$n\f$ derivatives of the B-splines evaluated at coordinate x. It has to be a 2D mdspan with (degree+1)*(n+1) elements.
         * @param[in] x The coordinate where B-splines derivatives are evaluated.
         * @param[in] n The number of derivatives to evaluate (in addition to the B-splines values themselves).
         * @return The index of the first B-spline which is evaluated.
         */
        KOKKOS_INLINE_FUNCTION discrete_element_type eval_basis_and_n_derivs(
                ddc::DSpan2D derivs,
                ddc::Coordinate<Tag> const& x,
                std::size_t n) const;

        /** @brief Compute the integrals of the B-splines.
         *
         * The integral of each of the B-splines over their support within the domain on which this basis was defined.
         *
         * @param[out] int_vals The values of the integrals. It has to be a 1D mdspan of size (nbasis).
         * @return The values of the integrals.
         */
        template <class Layout, class MemorySpace2>
        KOKKOS_INLINE_FUNCTION ddc::ChunkSpan<double, discrete_domain_type, Layout, MemorySpace2>
        integrals(
                ddc::ChunkSpan<double, discrete_domain_type, Layout, MemorySpace2> int_vals) const;

        /** @brief Returns the coordinate of the knot corresponding to the given index.
         *
         * Returns the coordinate of the knot corresponding to the given index. The domain
         * over which the B-splines are defined is comprised of ncells+1 knots however there are a total of
         * ncells+1+2*degree knots. The additional knots which control the shape of the B-splines near the
         * boundary are added before and after the break points. The knot index is therefore in the interval [-degree, ncells+degree]
         *
         * @param[in] knot_idx Integer identifying index of the knot.
         * @return Coordinate of the knot.
         */
        KOKKOS_INLINE_FUNCTION ddc::Coordinate<Tag> get_knot(int knot_idx) const noexcept
        {
            return ddc::Coordinate<Tag>(rmin() + idx * ddc::step<mesh_type>());
        }

        /** @brief Returns the coordinate of the first support knot associated to a DiscreteElement identifying a B-spline.
         *
         * Each B-spline has a support defined over (degree+2) knots. For a B-spline identified by the
         * provided DiscreteElement, this function returns the first knot in the support of the B-spline.
         * In other words it returns the lower bound of the support.
         *
         * @param[in] ix DiscreteElement identifying the B-spline.
         * @return Coordinate of the knot.
         */
        KOKKOS_INLINE_FUNCTION double get_first_support_knot(discrete_element_type const& ix) const
        {
            return get_knot(ix.uid() - degree());
        }

        /** @brief Returns the coordinate of the last support knot associated to a DiscreteElement identifying a B-spline.
         *
         * Each B-spline has a support defined over (degree+2) knots. For a B-spline identified by the
         * provided DiscreteElement, this function returns the last knot in the support of the B-spline.
         * In other words it returns the upper bound of the support.
         *
         * @param[in] ix DiscreteElement identifying the B-spline.
         * @return Coordinate of the knot.
         */
        KOKKOS_INLINE_FUNCTION double get_last_support_knot(discrete_element_type const& ix) const
        {
            return get_knot(ix.uid() + 1);
        }

        /** @brief Returns the coordinate of the (n+1)-th knot in the support of the identified B-spline.
         *
         * Each B-spline has a support defined over (degree+2) knots. For a B-spline identified by the
         * provided DiscreteElement, this function returns the (n+1)-th knot in the support of the B-spline.
         *
         * @param[in] ix DiscreteElement identifying the B-spline.
         * @param[in] n Integer indexing a knot in the support of the B-spline.
         * @return Coordinate of the knot.
         */
        KOKKOS_INLINE_FUNCTION double get_support_knot_n(discrete_element_type const& ix, int n)
                const
        {
            return get_knot(ix.uid() + n - degree());
        }

        /** @brief Returns the coordinate of the lower bound of the domain on which the B-splines are defined.
         *
         * @return Coordinate of the lower bound of the domain.
         */
        KOKKOS_INLINE_FUNCTION ddc::Coordinate<Tag> rmin() const noexcept
        {
            return ddc::coordinate(m_domain.front());
        }

        /** @brief Returns the coordinate of the upper bound of the domain on which the B-splines are defined.
         *
         * @return Coordinate of the upper bound of the domain.
         */
        KOKKOS_INLINE_FUNCTION ddc::Coordinate<Tag> rmax() const noexcept
        {
            return ddc::coordinate(m_domain.back());
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

        /** @brief Returns the discrete domain including eventual additionnal bsplines in the periodic case. See size().
         *
         * @return The discrete domain including eventual additionnal bsplines.
         */
        KOKKOS_INLINE_FUNCTION discrete_domain_type full_domain() const
        {
            return discrete_domain_type(discrete_element_type(0), discrete_vector_type(size()));
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
            return m_domain.size() - 1;
        }

    private:
        KOKKOS_INLINE_FUNCTION double inv_step() const noexcept
        {
            return 1.0 / ddc::step<mesh_type>();
        }

        KOKKOS_INLINE_FUNCTION discrete_element_type
        eval_basis(DSpan1D values, ddc::Coordinate<Tag> const& x, std::size_t degree) const;

        KOKKOS_INLINE_FUNCTION void get_icell_and_offset(
                int& icell,
                double& offset,
                ddc::Coordinate<Tag> const& x) const;
    };
};

template <class DDim>
struct is_uniform_bsplines : public std::is_base_of<detail::UniformBSplinesBase, DDim>
{
};

/**
 * @brief Indicates if a tag corresponds to uniform B-splines or not.
 *
 * @tparam The presumed uniform B-splines. 
 */
template <class DDim>
constexpr bool is_uniform_bsplines_v = is_uniform_bsplines<DDim>::value;

template <class Tag, std::size_t D>
template <class DDim, class MemorySpace>
KOKKOS_INLINE_FUNCTION ddc::DiscreteElement<DDim> UniformBSplines<Tag, D>::Impl<DDim, MemorySpace>::
        eval_basis(
                DSpan1D values,
                ddc::Coordinate<Tag> const& x,
                [[maybe_unused]] std::size_t const deg) const
{
    assert(values.size() == deg + 1);

    double offset;
    int jmin;
    // 1. Compute cell index 'icell' and x_offset
    // 2. Compute index range of B-splines with support over cell 'icell'
    get_icell_and_offset(jmin, offset, x);

    // 3. Compute values of aforementioned B-splines
    double xx, temp, saved;
    values(0) = 1.0;
    for (std::size_t j = 1; j < values.size(); ++j) {
        xx = -offset;
        saved = 0.0;
        for (std::size_t r = 0; r < j; ++r) {
            xx += 1;
            temp = values(r) / j;
            values(r) = saved + xx * temp;
            saved = (j - xx) * temp;
        }
        values(j) = saved;
    }

    return discrete_element_type(jmin);
}

template <class Tag, std::size_t D>
template <class DDim, class MemorySpace>
KOKKOS_INLINE_FUNCTION ddc::DiscreteElement<DDim> UniformBSplines<Tag, D>::Impl<DDim, MemorySpace>::
        eval_deriv(DSpan1D derivs, ddc::Coordinate<Tag> const& x) const
{
    assert(derivs.size() == degree() + 1);

    double offset;
    int jmin;
    // 1. Compute cell index 'icell' and x_offset
    // 2. Compute index range of B-splines with support over cell 'icell'
    get_icell_and_offset(jmin, offset, x);

    // 3. Compute derivatives of aforementioned B-splines
    //    Derivatives are normalized, hence they should be divided by dx
    double xx, temp, saved;
    derivs(0) = 1.0 / ddc::step<mesh_type>();
    for (std::size_t j = 1; j < degree(); ++j) {
        xx = -offset;
        saved = 0.0;
        for (std::size_t r = 0; r < j; ++r) {
            xx += 1.0;
            temp = derivs(r) / j;
            derivs(r) = saved + xx * temp;
            saved = (j - xx) * temp;
        }
        derivs(j) = saved;
    }

    // Compute derivatives
    double bjm1 = derivs[0];
    double bj = bjm1;
    derivs(0) = -bjm1;
    for (std::size_t j = 1; j < degree(); ++j) {
        bj = derivs(j);
        derivs(j) = bjm1 - bj;
        bjm1 = bj;
    }
    derivs(degree()) = bj;

    return discrete_element_type(jmin);
}

template <class Tag, std::size_t D>
template <class DDim, class MemorySpace>
KOKKOS_INLINE_FUNCTION ddc::DiscreteElement<DDim> UniformBSplines<Tag, D>::Impl<DDim, MemorySpace>::
        eval_basis_and_n_derivs(
                ddc::DSpan2D const derivs,
                ddc::Coordinate<Tag> const& x,
                std::size_t const n) const
{
    std::array<double, (degree() + 1) * (degree() + 1)> ndu_ptr;
    std::experimental::mdspan<
            double,
            std::experimental::extents<std::size_t, degree() + 1, degree() + 1>> const
            ndu(ndu_ptr.data());
    std::array<double, 2 * (degree() + 1)> a_ptr;
    std::experimental::
            mdspan<double, std::experimental::extents<std::size_t, degree() + 1, 2>> const a(
                    a_ptr.data());
    double offset;
    int jmin;

    assert(x >= rmin());
    assert(x <= rmax());
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
    double xx, temp, saved;
    ndu(0, 0) = 1.0;
    for (std::size_t j = 1; j < degree() + 1; ++j) {
        xx = -offset;
        saved = 0.0;
        for (std::size_t r = 0; r < j; ++r) {
            xx += 1.0;
            temp = ndu(j - 1, r) / j;
            ndu(j, r) = saved + xx * temp;
            saved = (j - xx) * temp;
        }
        ndu(j, j) = saved;
    }
    for (std::size_t i = 0; i < ndu.extent(1); ++i) {
        derivs(i, 0) = ndu(degree(), i);
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
                a(0, s2) = a(0, s1) / (pk + 1);
                d = a(0, s2) * ndu(pk, rk);
            }
            int const j1 = rk > -1 ? 1 : (-rk);
            int const j2 = (r - 1) <= pk ? k : (degree() - r + 1);
            for (int j = j1; j < j2; ++j) {
                a(j, s2) = (a(j, s1) - a(j - 1, s1)) / (pk + 1);
                d += a(j, s2) * ndu(pk, rk + j);
            }
            if (r <= pk) {
                a(k, s2) = -a(k - 1, s1) / (pk + 1);
                d += a(k, s2) * ndu(pk, r);
            }
            derivs(r, k) = d;
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
            derivs(i, k) *= d;
        }
        d *= (degree() - k) * inv_dx;
    }

    return discrete_element_type(jmin);
}

template <class Tag, std::size_t D>
template <class DDim, class MemorySpace>
KOKKOS_INLINE_FUNCTION void UniformBSplines<Tag, D>::Impl<DDim, MemorySpace>::get_icell_and_offset(
        int& icell,
        double& offset,
        ddc::Coordinate<Tag> const& x) const
{
    assert(x >= rmin());
    assert(x <= rmax());

    double const inv_dx = inv_step();
    if (x == rmin()) {
        icell = 0;
        offset = 0.0;
    } else if (x == rmax()) {
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

template <class Tag, std::size_t D>
template <class DDim, class MemorySpace>
template <class Layout, class MemorySpace2>
KOKKOS_INLINE_FUNCTION ddc::ChunkSpan<double, ddc::DiscreteDomain<DDim>, Layout, MemorySpace2>
UniformBSplines<Tag, D>::Impl<DDim, MemorySpace>::integrals(
        ddc::ChunkSpan<double, ddc::DiscreteDomain<DDim>, Layout, MemorySpace2> int_vals) const
{
    if constexpr (is_periodic()) {
        assert(int_vals.size() == nbasis() || int_vals.size() == size());
    } else {
        assert(int_vals.size() == nbasis());
    }
    discrete_domain_type const full_dom_splines(full_domain());

    if constexpr (is_periodic()) {
        discrete_domain_type const dom_bsplines(
                full_dom_splines.take_first(discrete_vector_type {nbasis()}));
        for (auto ix : dom_bsplines) {
            int_vals(ix) = ddc::step<mesh_type>();
        }
        if (int_vals.size() == size()) {
            discrete_domain_type const dom_bsplines_repeated(
                    full_dom_splines.take_last(discrete_vector_type {degree()}));
            for (auto ix : dom_bsplines_repeated) {
                int_vals(ix) = 0;
            }
        }
    } else {
        discrete_domain_type const dom_bspline_entirely_in_domain
                = full_dom_splines
                          .remove(discrete_vector_type(degree()), discrete_vector_type(degree()));
        for (auto ix : dom_bspline_entirely_in_domain) {
            int_vals(ix) = ddc::step<mesh_type>();
        }

        std::array<double, degree() + 2> edge_vals_ptr;
        std::experimental::
                mdspan<double, std::experimental::extents<std::size_t, degree() + 2>> const
                        edge_vals(edge_vals_ptr.data());

        eval_basis(edge_vals, rmin(), degree() + 1);

        double const d_eval = ddc::detail::sum(edge_vals);

        for (std::size_t i = 0; i < degree(); ++i) {
            double const c_eval = ddc::detail::sum(edge_vals, 0, degree() - i);

            double const edge_value = ddc::step<mesh_type>() * (d_eval - c_eval);

            int_vals(discrete_element_type(i)) = edge_value;
            int_vals(discrete_element_type(nbasis() - 1 - i)) = edge_value;
        }
    }
    return int_vals;
}
} // namespace ddc
