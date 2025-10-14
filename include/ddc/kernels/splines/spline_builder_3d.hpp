// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cassert>
#include <cstddef>
#include <optional>
#include <type_traits>

#include <ddc/ddc.hpp>

#include "spline_builder.hpp"
#include "spline_builder_2d.hpp"

namespace ddc {

/**
 * @brief A class for creating a 3D spline approximation of a function.
 *
 * A class which contains an operator () which can be used to build a 3D spline approximation
 * of a function. A 3D spline approximation uses a cross-product between three 1D SplineBuilder.
 *
 * @see SplineBuilder
 */
template <
        class ExecSpace,
        class MemorySpace,
        class BSpline1,
        class BSpline2,
        class BSpline3,
        class DDimI1,
        class DDimI2,
        class DDimI3,
        ddc::BoundCond BcLower1,
        ddc::BoundCond BcUpper1,
        ddc::BoundCond BcLower2,
        ddc::BoundCond BcUpper2,
        ddc::BoundCond BcLower3,
        ddc::BoundCond BcUpper3,
        ddc::SplineSolver Solver>
class SplineBuilder3D
{
public:
    /// @brief The type of the Kokkos execution space used by this class.
    using exec_space = ExecSpace;

    /// @brief The type of the Kokkos memory space used by this class.
    using memory_space = MemorySpace;

    /// @brief The type of the SplineBuilder used by this class to spline-approximate along first dimension.
    using builder_type1 = ddc::
            SplineBuilder<ExecSpace, MemorySpace, BSpline1, DDimI1, BcLower1, BcUpper1, Solver>;

    /// @brief The type of SplineBuilder used by this class to spline-approximate along the second and third dimensions.
    using builder_type_2_3 = ddc::SplineBuilder2D<
            ExecSpace,
            MemorySpace,
            BSpline2,
            BSpline3,
            DDimI2,
            DDimI3,
            BcLower2,
            BcUpper2,
            BcLower3,
            BcUpper3,
            Solver>;

    /// @brief The type of the first interpolation continuous dimension.
    using continuous_dimension_type1 = typename builder_type1::continuous_dimension_type;

    /// @brief The type of the second interpolation continuous dimension.
    using continuous_dimension_type2 = typename builder_type_2_3::continuous_dimension_type1;

    /// @brief The type of the third interpolation continuous dimension.
    using continuous_dimension_type3 = typename builder_type_2_3::continuous_dimension_type2;

    /// @brief The type of the first interpolation discrete dimension.
    using interpolation_discrete_dimension_type1 =
            typename builder_type1::interpolation_discrete_dimension_type;

    /// @brief The type of the second interpolation discrete dimension.
    using interpolation_discrete_dimension_type2 =
            typename builder_type_2_3::interpolation_discrete_dimension_type1;

    /// @brief The type of the third interpolation discrete dimension.
    using interpolation_discrete_dimension_type3 =
            typename builder_type_2_3::interpolation_discrete_dimension_type2;

    /// @brief The type of the B-splines in the first dimension.
    using bsplines_type1 = typename builder_type1::bsplines_type;

    /// @brief The type of the B-splines in the second dimension.
    using bsplines_type2 = typename builder_type_2_3::bsplines_type1;

    /// @brief The type of the B-splines in the third dimension.
    using bsplines_type3 = typename builder_type_2_3::bsplines_type2;

    /// @brief The type of the Deriv domain on boundaries in the first dimension.
    using deriv_type1 = typename builder_type1::deriv_type;

    /// @brief The type of the Deriv domain on boundaries in the second dimension.
    using deriv_type2 = typename builder_type_2_3::deriv_type1;

    /// @brief The type of the Deriv domain on boundaries in the third dimension.
    using deriv_type3 = typename builder_type_2_3::deriv_type2;

    /// @brief The type of the domain for the interpolation mesh in the first dimension.
    using interpolation_domain_type1 =
            typename builder_type1::interpolation_discrete_dimension_type;

    /// @brief The type of the domain for the interpolation mesh in the second dimension.
    using interpolation_domain_type2 =
            typename builder_type_2_3::interpolation_discrete_dimension_type1;

    /// @brief The type of the domain for the interpolation mesh in the third dimension.
    using interpolation_domain_type3 =
            typename builder_type_2_3::interpolation_discrete_dimension_type2;

    /// @brief The type of the domain for the interpolation mesh in the 3D dimension.
    using interpolation_domain_type = ddc::DiscreteDomain<
            interpolation_discrete_dimension_type1,
            interpolation_discrete_dimension_type2,
            interpolation_discrete_dimension_type3>;

    /**
     * @brief The type of the whole domain representing interpolation points.
     *
     * @tparam The batched discrete domain on which the interpolation points are defined.
     */
    template <
            class BatchedInterpolationDDom,
            class = std::enable_if_t<ddc::is_discrete_domain_v<BatchedInterpolationDDom>>>
    using batched_interpolation_domain_type = BatchedInterpolationDDom;

    /**
     * @brief The type of the batch domain (obtained by removing the dimensions of interest
     * from the whole domain).
     *
     * @tparam The batched discrete domain on which the interpolation points are defined.
     *
     * Example: For batched_interpolation_domain_type = DiscreteDomain<X,Y,Z,T> and dimensions of interest X, Y and Z,
     * this is DiscreteDomain<T>.
     */
    template <
            class BatchedInterpolationDDom,
            class = std::enable_if_t<ddc::is_discrete_domain_v<BatchedInterpolationDDom>>>
    using batch_domain_type = ddc::remove_dims_of_t<
            BatchedInterpolationDDom,
            interpolation_discrete_dimension_type1,
            interpolation_discrete_dimension_type2,
            interpolation_discrete_dimension_type3>;

    /**
     * @brief The type of the whole spline domain (cartesian product of 3D spline domain
     * and batch domain) preserving the order of dimensions.
     *
     * @tparam The batched discrete domain on which the interpolation points are defined.
     *
     * Example: For batched_interpolation_domain_type = DiscreteDomain<X,Y,Z,T> and dimensions of interest X, Y and Z
     * (associated to B-splines tags BSplinesX, BSplinesY and BSplinesZ), this is DiscreteDomain<BSplinesX, BSplinesY, BSplinesZ, T>
     */
    template <
            class BatchedInterpolationDDom,
            class = std::enable_if_t<ddc::is_discrete_domain_v<BatchedInterpolationDDom>>>
    using batched_spline_domain_type
            = ddc::detail::convert_type_seq_to_discrete_domain_t<ddc::type_seq_replace_t<
                    ddc::to_type_seq_t<BatchedInterpolationDDom>,
                    ddc::detail::TypeSeq<
                            interpolation_discrete_dimension_type1,
                            interpolation_discrete_dimension_type2,
                            interpolation_discrete_dimension_type3>,
                    ddc::detail::TypeSeq<bsplines_type1, bsplines_type2, bsplines_type3>>>;

    /**
     * @brief The type of the whole Derivs domain (cartesian product of the 1D Deriv domain
     * and the associated batch domain) in the first dimension, preserving the order of dimensions.
     *
     * @tparam The batched discrete domain on which the interpolation points are defined.
     *
     * Example: For batched_interpolation_domain_type = DiscreteDomain<X,Y,Z,T> and dimensions of interest X, Y and Z,
     * this is DiscreteDomain<Deriv<X>, Y, Z, T>.
     */
    template <
            class BatchedInterpolationDDom,
            class = std::enable_if_t<ddc::is_discrete_domain_v<BatchedInterpolationDDom>>>
    using batched_derivs_domain_type1 = ddc::replace_dim_of_t<
            BatchedInterpolationDDom,
            interpolation_discrete_dimension_type1,
            deriv_type1>;

    /**
     * @brief The type of the whole Derivs domain (1D dimension of interest and cartesian
     * product of 1D Deriv domain and batch domain) in the first dimension, to be passed as
     * argument to the builder, preserving the underlying memory layout (order of dimensions).
     *
     * @tparam The batched discrete domain on which the interpolation points are defined.
     *
     * Example: For batched_interpolation_domain_type = DiscreteDomain<X,Y,Z,T> and dimensions of interest X, Y and Z,
     * this is StridedDiscreteDomain<X, Deriv<X>, Y, Z, T>
     */
    template <
            class BatchedInterpolationDDom,
            class = std::enable_if_t<ddc::is_discrete_domain_v<BatchedInterpolationDDom>>>
    using whole_derivs_domain_type1 = detail::to_whole_derivs_domain_t<
            ddc::detail::TypeSeq<interpolation_discrete_dimension_type1>,
            ddc::detail::TypeSeq<deriv_type1>,
            ddc::to_type_seq_t<BatchedInterpolationDDom>>;

    /**
     * @brief The type of the whole Derivs domain (cartesian product of the 1D Deriv domain
     * and the associated batch domain) in the second dimension, preserving the order of dimensions.
     *
     * @tparam The batched discrete domain on which the interpolation points are defined.
     *
     * Example: For batched_interpolation_domain_type = DiscreteDomain<X,Y,Z,T> and dimensions of interest X, Y, and Z
     * this is DiscreteDomain<X, Deriv<Y>, Z, T>.
     */
    template <
            class BatchedInterpolationDDom,
            class = std::enable_if_t<ddc::is_discrete_domain_v<BatchedInterpolationDDom>>>
    using batched_derivs_domain_type2 = ddc::replace_dim_of_t<
            BatchedInterpolationDDom,
            interpolation_discrete_dimension_type2,
            deriv_type2>;

    /**
     * @brief The type of the whole Derivs domain (1D dimension of interest and cartesian
     * product of 1D Deriv domain and batch domain) in the second dimension, to be passed as
     * argument to the builder, preserving the underlying memory layout (order of dimensions).
     *
     * @tparam The batched discrete domain on which the interpolation points are defined.
     *
     * Example: For batched_interpolation_domain_type = DiscreteDomain<X,Y,Z,T> and dimensions of interest X, Y and Z,
     * this is StridedDiscreteDomain<Y, X, Deriv<Y>, Z, T>
     */
    template <
            class BatchedInterpolationDDom,
            class = std::enable_if_t<ddc::is_discrete_domain_v<BatchedInterpolationDDom>>>
    using whole_derivs_domain_type2 = detail::to_whole_derivs_domain_t<
            ddc::detail::TypeSeq<interpolation_discrete_dimension_type2>,
            ddc::detail::TypeSeq<deriv_type2>,
            ddc::to_type_seq_t<BatchedInterpolationDDom>>;

    /**
     * @brief The type of the whole Derivs domain (cartesian product of the 1D Deriv domain
     * and the associated batch domain) in the third dimension, preserving the order of dimensions.
     *
     * @tparam The batched discrete domain on which the interpolation points are defined.
     *
     * Example: For batched_interpolation_domain_type = DiscreteDomain<X,Y,Z,T> and dimensions of interest X, Y, and Z
     * this is DiscreteDomain<X, Y, Deriv<Z>, T>.
     */
    template <
            class BatchedInterpolationDDom,
            class = std::enable_if_t<ddc::is_discrete_domain_v<BatchedInterpolationDDom>>>
    using batched_derivs_domain_type3 = ddc::replace_dim_of_t<
            BatchedInterpolationDDom,
            interpolation_discrete_dimension_type3,
            deriv_type3>;

    /**
     * @brief The type of the whole Derivs domain (1D dimension of interest and cartesian
     * product of 1D Deriv domain and batch domain) in the third dimension, to be passed as
     * argument to the builder, preserving the underlying memory layout (order of dimensions).
     *
     * @tparam The batched discrete domain on which the interpolation points are defined.
     *
     * Example: For batched_interpolation_domain_type = DiscreteDomain<X,Y,Z,T> and dimensions of interest X, Y and Z,
     * this is StridedDiscreteDomain<Z, X, Y, Deriv<Z>, T>
     */
    template <
            class BatchedInterpolationDDom,
            class = std::enable_if_t<ddc::is_discrete_domain_v<BatchedInterpolationDDom>>>
    using whole_derivs_domain_type3 = detail::to_whole_derivs_domain_t<
            ddc::detail::TypeSeq<interpolation_discrete_dimension_type3>,
            ddc::detail::TypeSeq<deriv_type3>,
            ddc::to_type_seq_t<BatchedInterpolationDDom>>;

    /**
     * @brief The type of the whole Derivs domain (cartesian product of the 2D Deriv domain
     * and the associated batch domain) in the first and second dimensions, preserving the order
     * of dimensions.
     *
     * @tparam The batched discrete domain on which the interpolation points are defined.
     *
     * Example: For batched_interpolation_domain_type = DiscreteDomain<X,Y,Z,T> and dimensions of interest X, Y and Z,
     * this is DiscreteDomain<Deriv<X>, Deriv<Y>, Z, T>.
     */
    template <
            class BatchedInterpolationDDom,
            class = std::enable_if_t<ddc::is_discrete_domain_v<BatchedInterpolationDDom>>>
    using batched_derivs_domain_type1_2 = ddc::replace_dim_of_t<
            ddc::replace_dim_of_t<
                    BatchedInterpolationDDom,
                    interpolation_discrete_dimension_type1,
                    deriv_type1>,
            interpolation_domain_type2,
            deriv_type2>;

    /**
     * @brief The type of the whole Derivs domain (cartesian product of the 2D Deriv domain
     * and the associated batch domain) in the first and second dimensions, to be passed
     * as argument to the builder, preserving the order of dimensions.
     *
     * @tparam The batched discrete domain on which the interpolation points are defined.
     *
     * Example: For batched_interpolation_domain_type = DiscreteDomain<X,Y,Z,T> and dimensions of interest X, Y and Z,
     * this is StridedDiscreteDomain<X, Y, Deriv<X>, Deriv<Y>, Z, T>
     */
    template <
            class BatchedInterpolationDDom,
            class = std::enable_if_t<ddc::is_discrete_domain_v<BatchedInterpolationDDom>>>
    using whole_derivs_domain_type1_2 = detail::to_whole_derivs_domain_t<
            ddc::detail::TypeSeq<
                    interpolation_discrete_dimension_type1,
                    interpolation_discrete_dimension_type2>,
            ddc::detail::TypeSeq<deriv_type1, deriv_type2>,
            ddc::to_type_seq_t<BatchedInterpolationDDom>>;

    /**
     * @brief The type of the whole Derivs domain (cartesian product of the 2D Deriv domain
     * and the associated batch domain) in the second and third dimensions, preserving the order
     * of dimensions.
     *
     * @tparam The batched discrete domain on which the interpolation points are defined.
     *
     * Example: For batched_interpolation_domain_type = DiscreteDomain<X,Y,Z,T> and dimensions of interest X, Y, and Z
     * this is DiscreteDomain<X, Deriv<Y>, Deriv<Z>, T>.
     */
    template <
            class BatchedInterpolationDDom,
            class = std::enable_if_t<ddc::is_discrete_domain_v<BatchedInterpolationDDom>>>
    using batched_derivs_domain_type2_3 = ddc::replace_dim_of_t<
            ddc::replace_dim_of_t<
                    BatchedInterpolationDDom,
                    interpolation_discrete_dimension_type2,
                    deriv_type2>,
            interpolation_domain_type3,
            deriv_type3>;

    /**
     * @brief The type of the whole Derivs domain (cartesian product of the 2D Deriv domain
     * and the associated batch domain) in the second and third dimensions, to be passed
     * as argument to the builder, preserving the order of dimensions.
     *
     * @tparam The batched discrete domain on which the interpolation points are defined.
     *
     * Example: For batched_interpolation_domain_type = DiscreteDomain<X,Y,Z,T> and dimensions of interest X, Y and Z,
     * this is StridedDiscreteDomain<Y, Z, X, Deriv<Y>, Deriv<Z>, T>
     */
    template <
            class BatchedInterpolationDDom,
            class = std::enable_if_t<ddc::is_discrete_domain_v<BatchedInterpolationDDom>>>
    using whole_derivs_domain_type2_3 = detail::to_whole_derivs_domain_t<
            ddc::detail::TypeSeq<
                    interpolation_discrete_dimension_type2,
                    interpolation_discrete_dimension_type3>,
            ddc::detail::TypeSeq<deriv_type2, deriv_type3>,
            ddc::to_type_seq_t<BatchedInterpolationDDom>>;

    /**
     * @brief The type of the whole Derivs domain (cartesian product of the 2D Deriv domain
     * and the associated batch domain) in the first and third dimensions, preserving the order
     * of dimensions.
     *
     * @tparam The batched discrete domain on which the interpolation points are defined.
     *
     * Example: For batched_interpolation_domain_type = DiscreteDomain<X,Y,Z,T> and dimensions of interest X, Y, and Z
     * this is DiscreteDomain<Deriv<X>, Y, Deriv<Z>, T>.
     */
    template <
            class BatchedInterpolationDDom,
            class = std::enable_if_t<ddc::is_discrete_domain_v<BatchedInterpolationDDom>>>
    using batched_derivs_domain_type1_3 = ddc::replace_dim_of_t<
            ddc::replace_dim_of_t<
                    BatchedInterpolationDDom,
                    interpolation_discrete_dimension_type1,
                    deriv_type1>,
            interpolation_domain_type3,
            deriv_type3>;

    /**
     * @brief The type of the whole Derivs domain (cartesian product of the 2D Deriv domain
     * and the associated batch domain) in the first and third dimensions, to be passed
     * as argument to the builder, preserving the order of dimensions.
     *
     * @tparam The batched discrete domain on which the interpolation points are defined.
     *
     * Example: For batched_interpolation_domain_type = DiscreteDomain<X,Y,Z,T> and dimensions of interest X, Y and Z,
     * this is StridedDiscreteDomain<X, Z, Deriv<X>, Y, Deriv<Z>, T>
     */
    template <
            class BatchedInterpolationDDom,
            class = std::enable_if_t<ddc::is_discrete_domain_v<BatchedInterpolationDDom>>>
    using whole_derivs_domain_type1_3 = detail::to_whole_derivs_domain_t<
            ddc::detail::TypeSeq<
                    interpolation_discrete_dimension_type1,
                    interpolation_discrete_dimension_type3>,
            ddc::detail::TypeSeq<deriv_type1, deriv_type3>,
            ddc::to_type_seq_t<BatchedInterpolationDDom>>;

    /**
     * @brief The type of the whole Derivs domain (cartesian product of the 3D Deriv domain
     * and the batch domain), preserving the order of dimensions.
     *
     * @tparam The batched discrete domain on which the interpolation points are defined.
     *
     * Example: For batched_interpolation_domain_type = DiscreteDomain<X,Y,Z,T> and dimensions of interest X, Y and Z,
     * this is DiscreteDomain<Deriv<X>, Deriv<Y>, Deriv<Z>, T>.
     */
    template <
            class BatchedInterpolationDDom,
            class = std::enable_if_t<ddc::is_discrete_domain_v<BatchedInterpolationDDom>>>
    using batched_derivs_domain_type
            = ddc::detail::convert_type_seq_to_discrete_domain_t<ddc::type_seq_replace_t<
                    ddc::to_type_seq_t<BatchedInterpolationDDom>,
                    ddc::detail::TypeSeq<
                            interpolation_discrete_dimension_type1,
                            interpolation_discrete_dimension_type2,
                            interpolation_discrete_dimension_type3>,
                    ddc::detail::TypeSeq<deriv_type1, deriv_type2, deriv_type3>>>;

    /**
     * @brief The type of the whole Derivs domain (cartesian product of the 3D Deriv domain
     * and the associated batch domain) to be passed as argument to the builder, preserving
     * the order of dimensions.
     *
     * @tparam The batched discrete domain on which the interpolation points are defined.
     *
     * Example: For batched_interpolation_domain_type = DiscreteDomain<X,Y,Z,T> and dimensions of interest X, Y and Z,
     * this is StridedDiscreteDomain<X, Y, Z, Deriv<X>, Deriv<Y>, Deriv<Z>, T>
     */
    template <
            class BatchedInterpolationDDom,
            class = std::enable_if_t<ddc::is_discrete_domain_v<BatchedInterpolationDDom>>>
    using whole_derivs_domain_type = detail::to_whole_derivs_domain_t<
            ddc::detail::TypeSeq<
                    interpolation_discrete_dimension_type1,
                    interpolation_discrete_dimension_type2,
                    interpolation_discrete_dimension_type3>,
            ddc::detail::TypeSeq<deriv_type1, deriv_type2, deriv_type3>,
            ddc::to_type_seq_t<BatchedInterpolationDDom>>;

private:
    builder_type1 m_spline_builder1;
    builder_type_2_3 m_spline_builder_2_3;

public:
    /**
     * @brief Build a SplineBuilder3D acting on interpolation_domain.
     *
     * @param interpolation_domain The domain on which the interpolation points are defined, without the batch dimensions.
     *
     * @param cols_per_chunk A parameter used by the slicer (internal to the solver) to define the size
     * of a chunk of right-hand-sides of the linear problem to be computed in parallel (chunks are treated
     * by the linear solver one-after-the-other).
     * This value is optional. If no value is provided then the default value is chosen by the requested solver.
     *
     * @param preconditioner_max_block_size A parameter used by the slicer (internal to the solver) to
     * define the size of a block used by the Block-Jacobi preconditioner.
     * This value is optional. If no value is provided then the default value is chosen by the requested solver.
     *
     * @see SplinesLinearProblemSparse
     */
    explicit SplineBuilder3D(
            interpolation_domain_type const& interpolation_domain,
            std::optional<std::size_t> cols_per_chunk = std::nullopt,
            std::optional<unsigned int> preconditioner_max_block_size = std::nullopt)
        : m_spline_builder1(interpolation_domain, cols_per_chunk, preconditioner_max_block_size)
        , m_spline_builder_2_3(interpolation_domain, cols_per_chunk, preconditioner_max_block_size)
    {
    }

    /**
     * @brief Build a SplineBuilder3D acting on the interpolation domain contained in batched_interpolation_domain.
     *
     * @param batched_interpolation_domain The domain on which the interpolation points are defined.
     *
     * @param cols_per_chunk A parameter used by the slicer (internal to the solver) to define the size
     * of a chunk of right-hand-sides of the linear problem to be computed in parallel (chunks are treated
     * by the linear solver one-after-the-other).
     * This value is optional. If no value is provided then the default value is chosen by the requested solver.
     *
     * @param preconditioner_max_block_size A parameter used by the slicer (internal to the solver) to
     * define the size of a block used by the Block-Jacobi preconditioner.
     * This value is optional. If no value is provided then the default value is chosen by the requested solver.
     *
     * @see SplinesLinearProblemSparse
     */
    template <
            class BatchedInterpolationDDom,
            class = std::enable_if_t<ddc::is_discrete_domain_v<BatchedInterpolationDDom>>>
    explicit SplineBuilder3D(
            BatchedInterpolationDDom const& batched_interpolation_domain,
            std::optional<std::size_t> cols_per_chunk = std::nullopt,
            std::optional<unsigned int> preconditioner_max_block_size = std::nullopt)
        : SplineBuilder3D(
                  interpolation_domain_type(batched_interpolation_domain),
                  cols_per_chunk,
                  preconditioner_max_block_size)
    {
    }

    /// @brief Copy-constructor is deleted.
    SplineBuilder3D(SplineBuilder3D const& x) = delete;

    /**
     * @brief Move-constructs.
     *
     * @param x An rvalue to another SplineBuilder3D.
     */
    SplineBuilder3D(SplineBuilder3D&& x) = default;

    /// @brief Destructs.
    ~SplineBuilder3D() = default;

    /// @brief Copy-assignment is deleted.
    SplineBuilder3D& operator=(SplineBuilder3D const& x) = delete;

    /** @brief Move-assigns.
     *
     * @param x An rvalue to another SplineBuilder.
     * @return A reference to this object.
     */
    SplineBuilder3D& operator=(SplineBuilder3D&& x) = default;

    /**
     * @brief Get the domain for the 3D interpolation mesh used by this class.
     *
     * This is 3D because it is defined along the dimensions of interest.
     *
     * @return The 3D domain for the interpolation mesh.
     */
    interpolation_domain_type interpolation_domain() const noexcept
    {
        return ddc::DiscreteDomain<
                interpolation_domain_type1,
                interpolation_domain_type2,
                interpolation_domain_type3>(
                m_spline_builder1.interpolation_domain(),
                m_spline_builder_2_3.interpolation_domain());
    }

    /**
     * @brief Get the whole domain representing interpolation points.
     *
     * Values of the function must be provided on this domain in order
     * to build a spline representation of the function (cartesian product of 3D interpolation_domain and batch_domain).
     *
     * @param batched_interpolation_domain The whole domain on which the interpolation points are defined.
     *
     * @return The domain for the interpolation mesh.
     */
    template <
            class BatchedInterpolationDDom,
            class = std::enable_if_t<ddc::is_discrete_domain_v<BatchedInterpolationDDom>>>
    BatchedInterpolationDDom batched_interpolation_domain(
            BatchedInterpolationDDom const& batched_interpolation_domain) const noexcept
    {
        assert(interpolation_domain() == interpolation_domain_type(batched_interpolation_domain));
        return batched_interpolation_domain;
    }

    /**
     * @brief Get the batch domain.
     *
     * Obtained by removing the dimensions of interest from the whole interpolation domain.
     *
     * @param batched_interpolation_domain The whole domain on which the interpolation points are defined.
     *
     * @return The batch domain.
     */
    template <
            class BatchedInterpolationDDom,
            class = std::enable_if_t<ddc::is_discrete_domain_v<BatchedInterpolationDDom>>>
    batch_domain_type<BatchedInterpolationDDom> batch_domain(
            BatchedInterpolationDDom const& batched_interpolation_domain) const noexcept
    {
        assert(interpolation_domain() == interpolation_domain_type(batched_interpolation_domain));
        return ddc::remove_dims_of(batched_interpolation_domain, interpolation_domain());
    }

    /**
     * @brief Get the 3D domain on which spline coefficients are defined.
     *
     * The 3D spline domain corresponding to the dimensions of interest.
     *
     * @return The 3D domain for the spline coefficients.
     */
    ddc::DiscreteDomain<bsplines_type1, bsplines_type2, bsplines_type3> spline_domain()
            const noexcept
    {
        return ddc::DiscreteDomain<bsplines_type1, bsplines_type2, bsplines_type3>(
                ddc::discrete_space<bsplines_type1>().full_domain(),
                ddc::discrete_space<bsplines_type2>().full_domain(),
                ddc::discrete_space<bsplines_type3>().full_domain());
    }

    /**
     * @brief Get the whole domain on which spline coefficients are defined.
     *
     * Spline approximations (spline-transformed functions) are computed on this domain.
     *
     * @param batched_interpolation_domain The whole domain on which the interpolation points are defined.
     *
     * @return The domain for the spline coefficients.
     */
    template <
            class BatchedInterpolationDDom,
            class = std::enable_if_t<ddc::is_discrete_domain_v<BatchedInterpolationDDom>>>
    batched_spline_domain_type<BatchedInterpolationDDom> batched_spline_domain(
            BatchedInterpolationDDom const& batched_interpolation_domain) const noexcept
    {
        assert(interpolation_domain() == interpolation_domain_type(batched_interpolation_domain));
        return ddc::replace_dim_of<interpolation_discrete_dimension_type1, bsplines_type1>(
                ddc::replace_dim_of<interpolation_discrete_dimension_type2, bsplines_type2>(
                        ddc::replace_dim_of<
                                interpolation_discrete_dimension_type3,
                                bsplines_type3>(batched_interpolation_domain, spline_domain()),
                        spline_domain()),
                spline_domain());
    }

    /**
     * @brief Compute a 3D spline approximation of a function.
     *
     * Use the values of a function (defined on
     * SplineBuilder3D::batched_interpolation_domain) and the derivatives of the
     * function at the boundaries (in the case of BoundCond::HERMITE only)
     * to calculate a 3D spline approximation of this function.
     *
     * The spline approximation is stored as a ChunkSpan of coefficients
     * associated with B-splines.
     *
     * @param[out] spline
     *      The coefficients of the spline computed by this SplineBuilder.
     * @param[in] vals
     *      The values of the function at the interpolation mesh.
     * @param[in] derivs1
     *      The values of the derivatives in the first dimension.
     * @param[in] derivs2
     *      The values of the derivatives in the second dimension.
     * @param[in] derivs3
     *      The values of the derivatives in the third dimension.
     * @param[in] mixed_derivs1_2
     *      The values of the the cross-derivatives in the first dimension and in the second dimension.
     * @param[in] mixed_derivs2_3
     *      The values of the the cross-derivatives in the second dimension and in the third dimension.
     * @param[in] mixed_derivs1_3
     *      The values of the the cross-derivatives in the first dimension and in the third dimension.
     * @param[in] mixed_derivs1_2_3
     *      The values of the the cross-derivatives in the first dimension, in the second dimension and in the third dimension.
     */
    template <class Layout, class DerivsLayout, class BatchedInterpolationDDom>
    void operator()(
            ddc::ChunkSpan<
                    double,
                    batched_spline_domain_type<BatchedInterpolationDDom>,
                    Layout,
                    memory_space> spline,
            ddc::ChunkSpan<double const, BatchedInterpolationDDom, Layout, memory_space> vals,
            ddc::ChunkSpan<
                    double const,
                    whole_derivs_domain_type1<BatchedInterpolationDDom>,
                    DerivsLayout,
                    memory_space> derivs1,
            ddc::ChunkSpan<
                    double const,
                    whole_derivs_domain_type2<BatchedInterpolationDDom>,
                    DerivsLayout,
                    memory_space> derivs2,
            ddc::ChunkSpan<
                    double const,
                    whole_derivs_domain_type3<BatchedInterpolationDDom>,
                    DerivsLayout,
                    memory_space> derivs3,
            ddc::ChunkSpan<
                    double const,
                    whole_derivs_domain_type1_2<BatchedInterpolationDDom>,
                    DerivsLayout,
                    memory_space> mixed_derivs1_2,
            ddc::ChunkSpan<
                    double const,
                    whole_derivs_domain_type2_3<BatchedInterpolationDDom>,
                    DerivsLayout,
                    memory_space> mixed_derivs2_3,
            ddc::ChunkSpan<
                    double const,
                    whole_derivs_domain_type1_3<BatchedInterpolationDDom>,
                    DerivsLayout,
                    memory_space> mixed_derivs1_3,
            ddc::ChunkSpan<
                    double const,
                    whole_derivs_domain_type<BatchedInterpolationDDom>,
                    DerivsLayout,
                    memory_space> mixed_derivs1_2_3) const;
};


template <
        class ExecSpace,
        class MemorySpace,
        class BSpline1,
        class BSpline2,
        class BSpline3,
        class DDimI1,
        class DDimI2,
        class DDimI3,
        ddc::BoundCond BcLower1,
        ddc::BoundCond BcUpper1,
        ddc::BoundCond BcLower2,
        ddc::BoundCond BcUpper2,
        ddc::BoundCond BcLower3,
        ddc::BoundCond BcUpper3,
        ddc::SplineSolver Solver>
template <class Layout, class DerivsLayout, class BatchedInterpolationDDom>
void SplineBuilder3D<
        ExecSpace,
        MemorySpace,
        BSpline1,
        BSpline2,
        BSpline3,
        DDimI1,
        DDimI2,
        DDimI3,
        BcLower1,
        BcUpper1,
        BcLower2,
        BcUpper2,
        BcLower3,
        BcUpper3,
        Solver>::
operator()(
        ddc::ChunkSpan<
                double,
                batched_spline_domain_type<BatchedInterpolationDDom>,
                Layout,
                memory_space> spline,
        ddc::ChunkSpan<double const, BatchedInterpolationDDom, Layout, memory_space> vals,
        ddc::ChunkSpan<
                double const,
                whole_derivs_domain_type1<BatchedInterpolationDDom>,
                DerivsLayout,
                memory_space> derivs1,
        ddc::ChunkSpan<
                double const,
                whole_derivs_domain_type2<BatchedInterpolationDDom>,
                DerivsLayout,
                memory_space> derivs2,
        ddc::ChunkSpan<
                double const,
                whole_derivs_domain_type3<BatchedInterpolationDDom>,
                DerivsLayout,
                memory_space> derivs3,
        ddc::ChunkSpan<
                double const,
                whole_derivs_domain_type1_2<BatchedInterpolationDDom>,
                DerivsLayout,
                memory_space> mixed_derivs1_2,
        ddc::ChunkSpan<
                double const,
                whole_derivs_domain_type2_3<BatchedInterpolationDDom>,
                DerivsLayout,
                memory_space> mixed_derivs2_3,
        ddc::ChunkSpan<
                double const,
                whole_derivs_domain_type1_3<BatchedInterpolationDDom>,
                DerivsLayout,
                memory_space> mixed_derivs1_3,
        ddc::ChunkSpan<
                double const,
                whole_derivs_domain_type<BatchedInterpolationDDom>,
                DerivsLayout,
                memory_space> mixed_derivs1_2_3) const
{
    auto const batched_interpolation_domain = vals.domain();

    using ddim2 = interpolation_discrete_dimension_type2;
    using ddim3 = interpolation_discrete_dimension_type3;
    using detail::dmax;
    using detail::dmin;

    assert(interpolation_domain() == interpolation_domain_type(batched_interpolation_domain));

    // Build the derivatives along the second dimension
    auto const spline_batched_derivs2_domain = ddc::detail::get_whole_derivs_domain<deriv_type2>(
            ddc::select<ddim2>(batched_interpolation_domain),
            m_spline_builder1.batched_spline_domain(batched_interpolation_domain),
            bsplines_type2::degree() / 2);

    ddc::Chunk spline_derivs2_alloc(
            spline_batched_derivs2_domain,
            ddc::KokkosAllocator<double, MemorySpace>());
    auto spline_derivs2 = spline_derivs2_alloc.span_view();

    if constexpr (BcLower2 == ddc::BoundCond::HERMITE) {
        auto const spline_derivs_min2_strided = detail::derivs(spline_derivs2, dmin<ddim2>);
        auto const spline_derivs_min2
                = detail::strided_to_discrete_domain_chunkspan(spline_derivs_min2_strided);

        auto const derivs_min2_strided = detail::derivs(derivs2, dmin<ddim2>);
        auto const derivs_min2 = detail::strided_to_discrete_domain_chunkspan(derivs_min2_strided);
        m_spline_builder1(
                spline_derivs_min2,
                derivs_min2,
                detail::derivs(mixed_derivs1_2, dmin<ddim2>));
    }

    if constexpr (BcUpper2 == ddc::BoundCond::HERMITE) {
        auto const spline_derivs_max2_strided = detail::derivs(spline_derivs2, dmax<ddim2>);
        auto const spline_derivs_max2
                = detail::strided_to_discrete_domain_chunkspan(spline_derivs_max2_strided);

        auto const derivs_max2_strided = detail::derivs(derivs2, dmax<ddim2>);
        auto const derivs_max2 = detail::strided_to_discrete_domain_chunkspan(derivs_max2_strided);
        m_spline_builder1(
                spline_derivs_max2,
                derivs_max2,
                detail::derivs(mixed_derivs1_2, dmax<ddim2>));
    }

    // Build the derivatives along the third dimension
    auto const spline_batched_derivs3_domain = ddc::detail::get_whole_derivs_domain<deriv_type3>(
            ddc::select<ddim3>(batched_interpolation_domain),
            m_spline_builder1.batched_spline_domain(batched_interpolation_domain),
            bsplines_type3::degree() / 2);

    ddc::Chunk spline_derivs3_alloc(
            spline_batched_derivs3_domain,
            ddc::KokkosAllocator<double, MemorySpace>());
    auto spline_derivs3 = spline_derivs3_alloc.span_view();

    if constexpr (BcLower3 == ddc::BoundCond::HERMITE) {
        auto const spline_derivs_min3_strided = detail::derivs(spline_derivs3, dmin<ddim3>);
        auto const spline_derivs_min3
                = detail::strided_to_discrete_domain_chunkspan(spline_derivs_min3_strided);

        auto const derivs_min3_strided = detail::derivs(derivs3, dmin<ddim3>);
        auto const derivs_min3 = detail::strided_to_discrete_domain_chunkspan(derivs_min3_strided);
        m_spline_builder1(
                spline_derivs_min3,
                derivs_min3,
                detail::derivs(mixed_derivs1_3, dmin<ddim3>));
    }

    if constexpr (BcUpper3 == ddc::BoundCond::HERMITE) {
        auto const spline_derivs_max3_strided = detail::derivs(spline_derivs3, dmax<ddim3>);
        auto const spline_derivs_max3
                = detail::strided_to_discrete_domain_chunkspan(spline_derivs_max3_strided);

        auto const derivs_max3_strided = detail::derivs(derivs3, dmax<ddim3>);
        auto const derivs_max3 = detail::strided_to_discrete_domain_chunkspan(derivs_max3_strided);
        m_spline_builder1(
                spline_derivs_max3,
                derivs_max3,
                detail::derivs(mixed_derivs1_3, dmax<ddim3>));
    }

    // Build the cross derivatives along the second and third dimensions
    auto const spline_batched_derivs2_3_domain
            = ddc::detail::get_whole_derivs_domain<deriv_type2, deriv_type3>(
                    ddc::select<ddim2, ddim3>(batched_interpolation_domain),
                    m_spline_builder1.batched_spline_domain(batched_interpolation_domain),
                    bsplines_type2::degree() / 2,
                    bsplines_type3::degree() / 2);

    ddc::Chunk spline_derivs2_3_alloc(
            spline_batched_derivs2_3_domain,
            ddc::KokkosAllocator<double, MemorySpace>());
    auto spline_derivs2_3 = spline_derivs2_3_alloc.span_view();

    if constexpr (BcLower2 == ddc::BoundCond::HERMITE || BcLower3 == ddc::BoundCond::HERMITE) {
        auto const spline_derivs_min2_min3_strided
                = detail::derivs(spline_derivs2_3, dmin<ddim2>, dmin<ddim3>);
        auto const spline_derivs_min2_min3
                = detail::strided_to_discrete_domain_chunkspan(spline_derivs_min2_min3_strided);

        auto const derivs_min2_min3_strided
                = detail::derivs(mixed_derivs2_3, dmin<ddim2>, dmin<ddim3>);
        auto const derivs_min2_min3
                = detail::strided_to_discrete_domain_chunkspan(derivs_min2_min3_strided);
        m_spline_builder1(
                spline_derivs_min2_min3,
                derivs_min2_min3,
                detail::derivs(mixed_derivs1_2_3, dmin<ddim2>, dmin<ddim3>));
    }

    if constexpr (BcLower2 == ddc::BoundCond::HERMITE || BcUpper3 == ddc::BoundCond::HERMITE) {
        auto const spline_derivs_min2_max3_strided
                = detail::derivs(spline_derivs2_3, dmin<ddim2>, dmax<ddim3>);
        auto const spline_derivs_min2_max3
                = detail::strided_to_discrete_domain_chunkspan(spline_derivs_min2_max3_strided);

        auto const derivs_min2_max3_strided
                = detail::derivs(mixed_derivs2_3, dmin<ddim2>, dmax<ddim3>);
        auto const derivs_min2_max3
                = detail::strided_to_discrete_domain_chunkspan(derivs_min2_max3_strided);
        m_spline_builder1(
                spline_derivs_min2_max3,
                derivs_min2_max3,
                detail::derivs(mixed_derivs1_2_3, dmin<ddim2>, dmax<ddim3>));
    }

    if constexpr (BcUpper2 == ddc::BoundCond::HERMITE || BcLower3 == ddc::BoundCond::HERMITE) {
        auto const spline_derivs_max2_min3_strided
                = detail::derivs(spline_derivs2_3, dmax<ddim2>, dmin<ddim3>);
        auto const spline_derivs_max2_min3
                = detail::strided_to_discrete_domain_chunkspan(spline_derivs_max2_min3_strided);

        auto const derivs_max2_min3_strided
                = detail::derivs(mixed_derivs2_3, dmax<ddim2>, dmin<ddim3>);
        auto const derivs_max2_min3
                = detail::strided_to_discrete_domain_chunkspan(derivs_max2_min3_strided);
        m_spline_builder1(
                spline_derivs_max2_min3,
                derivs_max2_min3,
                detail::derivs(mixed_derivs1_2_3, dmax<ddim2>, dmin<ddim3>));
    }

    if constexpr (BcUpper2 == ddc::BoundCond::HERMITE || BcUpper3 == ddc::BoundCond::HERMITE) {
        auto const spline_derivs_max2_max3_strided
                = detail::derivs(spline_derivs2_3, dmax<ddim2>, dmax<ddim3>);
        auto const spline_derivs_max2_max3
                = detail::strided_to_discrete_domain_chunkspan(spline_derivs_max2_max3_strided);

        auto const derivs_max2_max3_strided
                = detail::derivs(mixed_derivs2_3, dmax<ddim2>, dmax<ddim3>);
        auto const derivs_max2_max3
                = detail::strided_to_discrete_domain_chunkspan(derivs_max2_max3_strided);
        m_spline_builder1(
                spline_derivs_max2_max3,
                derivs_max2_max3,
                detail::derivs(mixed_derivs1_2_3, dmax<ddim2>, dmax<ddim3>));
    }

    // Spline1-approximate vals (to spline1)
    ddc::Chunk spline1_alloc(
            m_spline_builder1.batched_spline_domain(batched_interpolation_domain),
            ddc::KokkosAllocator<double, MemorySpace>());
    ddc::ChunkSpan const spline1 = spline1_alloc.span_view();

    m_spline_builder1(spline1, vals, derivs1);

    m_spline_builder_2_3(
            spline,
            spline1.span_cview(),
            spline_derivs2.span_cview(),
            spline_derivs3.span_cview(),
            spline_derivs2_3.span_cview());
}

} // namespace ddc
