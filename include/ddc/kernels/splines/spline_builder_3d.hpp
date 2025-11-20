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
 * @brief A struct that contain info about one dimension that will be used
 * inside a spline builder
 *
 * @see SplineBuilder
 */
template<
   class _BSpline,
   class _DDimI,
   ddc::BoundCond _BcLower,
   ddc::BoundCond _BcUpper>
struct DimInfo {
  using BSpline = _BSpline;
  using DDimI = _DDimI;
  static const ddc::BoundCond BcLower = _BcLower;
  static const ddc::BoundCond BcUpper = _BcUpper;
};

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
        ddc::SplineSolver Solver,
        class DimInfo1,
        class DimInfo2,
        class DimInfo3>
class SplineBuilder3D
{
  // TODO: static_assert that DimInfoI is of type DimInfo
public:
    /// @brief The type of the Kokkos execution space used by this class.
    using exec_space = ExecSpace;

    /// @brief The type of the Kokkos memory space used by this class.
    using memory_space = MemorySpace;

    // Helper function to create tuple types from other types
    template <class... DimInfos>
      static std::tuple<typename ddc::SplineBuilder<
      ExecSpace,
      MemorySpace,
      typename DimInfos::BSpline,
      typename DimInfos::DDimI,
      DimInfos::BcLower,
      DimInfos::BcUpper,
      Solver>...>
        builderTypes(DimInfos...);

    /// @brief The type of the SplineBuilder used by this class to spline-approximate along each dimension.
    using builder_types = decltype(
        builderTypes(
          std::declval<DimInfo1>(),
          std::declval<DimInfo2>(),
          std::declval<DimInfo3>()));

    // TODO: delete
    /// @brief The type of SplineBuilder used by this class to spline-approximate along the second and third dimensions.
    using builder_type_2_3 = ddc::SplineBuilder2D<
            ExecSpace,
            MemorySpace,
            typename DimInfo2::BSpline,
            typename DimInfo3::BSpline,
            typename DimInfo2::DDimI,
            typename DimInfo3::DDimI,
            DimInfo2::BcLower,
            DimInfo2::BcUpper,
            DimInfo3::BcLower,
            DimInfo3::BcUpper,
            Solver>;

    // TODO unused, delete?
    ///// @brief The type of the first interpolation continuous dimension.
    //using continuous_dimension_type1 = typename std::tuple_element_t<0, builder_types>::continuous_dimension_type;

    ///// @brief The type of the second interpolation continuous dimension.
    //using continuous_dimension_type2 = typename std::tuple_element_t<1, builder_types>::continuous_dimension_type;

    ///// @brief The type of the third interpolation continuous dimension.
    //using continuous_dimension_type3 = typename std::tuple_element_t<2, builder_types>::continuous_dimension_type;

    // Extract interpolation_discrete_dimension_type in its own tuple
    template <class... BuilderTypes>
      static std::tuple<typename BuilderTypes::interpolation_discrete_dimension_type...>
      extract_interpolation_discrete_dimension(std::tuple<BuilderTypes...>);

    /// @brief The types of the interpolation discrete dimension for each dimensions.
    using interpolation_discrete_dimension_types =
      decltype(extract_interpolation_discrete_dimension(std::declval<builder_types>()));

    // Extract bsplines_type in its own tuple
    template <class... BuilderTypes>
      static std::tuple<typename BuilderTypes::bsplines_type...>
      extract_bsplines(std::tuple<BuilderTypes...>);

    /// @brief The types of the B-splines for each dimensions.
    using bsplines_types =
      decltype(extract_bsplines(std::declval<builder_types>()));

    // Extract deriv_type in its own tuple
    template <class... BuilderTypes>
      static std::tuple<typename BuilderTypes::deriv_type...>
      extract_deriv(std::tuple<BuilderTypes...>);

    /// @brief The types of the Deriv domain on boundaries for each dimensions.
    using deriv_types =
      decltype(extract_deriv(std::declval<builder_types>()));

    // TODO: same as interpolation_discrete_dimension_types, delete?
    /// @brief The types of the domain for the interpolation mesh for each dimensions.
    using interpolation_domain_types =
      decltype(extract_interpolation_discrete_dimension(std::declval<builder_types>()));

    // Create a ddc::DiscreteDomain type from a tuple of dimensions
    template <class... Dims>
     static ddc::DiscreteDomain<Dims...>
      discreteDomainFromTuple(std::tuple<Dims...>);

    /// @brief The type of the domain for the interpolation mesh in the 3D dimension.
    using interpolation_domain_type =
      decltype(discreteDomainFromTuple(std::declval<interpolation_discrete_dimension_types>()));

    /**
     * @brief The type of the whole domain representing interpolation points.
     *
     * @tparam The batched discrete domain on which the interpolation points are defined.
     */
    template <
            class BatchedInterpolationDDom,
            class = std::enable_if_t<ddc::is_discrete_domain_v<BatchedInterpolationDDom>>>
    using batched_interpolation_domain_type = BatchedInterpolationDDom;

    // Create a batch domain type from a tuple of dimensions
    template <class BatchedInterpolationDDom, class... Dims>
      static ddc::remove_dims_of_t<BatchedInterpolationDDom, Dims...>
      batchDomainFromTuple(std::tuple<Dims...>);

    /**
     * @brief The type of the batch domain (obtained by removing the dimensions of interest
     * from the whole domain).
     *
     * @tparam The batched discrete domain on which the interpolation points are defined.
     *
     * Example: For batched_interpolation_domain_type = DiscreteDomain<X,Y,Z,T> and dimensions of interest X, Y and Z,
     * this is DiscreteDomain<T>.
     */
    template <class BatchedInterpolationDDom,
              class = std::enable_if_t<ddc::is_discrete_domain_v<BatchedInterpolationDDom>>>
    using batch_domain_type =
      decltype(
          batchDomainFromTuple<BatchedInterpolationDDom>(
            std::declval<interpolation_discrete_dimension_types>()));

    // Create a batched spline domain type from a tuple of spline builders
    template <class BatchedInterpolationDDom, class... BuilderTypes>
      static ddc::detail::convert_type_seq_to_discrete_domain_t<
      ddc::type_seq_replace_t<
      ddc::to_type_seq_t<BatchedInterpolationDDom>,
      ddc::detail::TypeSeq<typename BuilderTypes::interpolation_discrete_dimension_type...>,
      ddc::detail::TypeSeq<typename BuilderTypes::bsplines_type...>>>
        batchedSplineDomainFromTuple(std::tuple<BuilderTypes...>);

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
    using batched_spline_domain_type =
      decltype(batchedSplineDomainFromTuple<BatchedInterpolationDDom> (
            std::declval<builder_types>()));

    //template <class BatchedInterpolationDDom, class... BuilderTypes>
    //  static std::tuple<
    //  ddc::replace_dim_of_t<
    //  BatchedInterpolationDDom,
    //  typename BuilderTypes::interpolation_discrete_dimension_type,
    //  typename BuilderTypes::deriv_type>...>
    //    batchedDerivsDomainsFromTuple(std::tuple<BuilderTypes...>);

    // TODO: Add comment
    template <class... BuilderTypes>
      static ddc::detail::TypeSeq<typename BuilderTypes::interpolation_discrete_dimension_type...>
      interpolation_TupleToTypeSeq(std::tuple<BuilderTypes...>);

    template <class... BuilderTypes>
      static ddc::detail::TypeSeq<typename BuilderTypes::deriv_type...>
      deriv_TupleToTypeSeq(std::tuple<BuilderTypes...>);

    // Create a tuple type containing the batched derivative domains for the first derivative
    template <class BatchedInterpolationDDom, class... BuilderTypesTuples>
      static std::tuple<
      ddc::replace_dim_of_t<
        BatchedInterpolationDDom,
        decltype(interpolation_TupleToTypeSeq(std::declval<BuilderTypesTuples>())),
        decltype(deriv_TupleToTypeSeq(std::declval<BuilderTypesTuples>())),
    //  typename BuilderTypes::interpolation_discrete_dimension_type,
    //  typename BuilderTypes::deriv_type>...>
        >...>
        batchedDerivsDomainsFromTuple(std::tuple<BuilderTypesTuples...>);

    /**
     * @brief The type of the whole Derivs domain (cartesian product of the 1D Deriv domain
     * and the associated batch domain) in the first dimension, preserving the order of dimensions.
     *
     * @tparam The batched discrete domain on which the interpolation points are defined.
     *
     * Example:
     *  - For batched_interpolation_domain_type = DiscreteDomain<X,Y,Z,T> and dimensions of interest X, Y and Z,
     * this is DiscreteDomain<Deriv<X>, Y, Z, T>.
     *  - For batched_interpolation_domain_type = DiscreteDomain<X,Y,Z,T> and dimensions of interest X, Y, and Z
     * this is DiscreteDomain<X, Deriv<Y>, Z, T>.
     *  - For batched_interpolation_domain_type = DiscreteDomain<X,Y,Z,T> and dimensions of interest X, Y, and Z
     * this is DiscreteDomain<X, Y, Deriv<Z>, T>.
     */
    template <
            class BatchedInterpolationDDom,
            class = std::enable_if_t<ddc::is_discrete_domain_v<BatchedInterpolationDDom>>>
    //using batched_derivs_domain_types =
    //  decltype(batchedDerivsDomainsFromTuple<BatchedInterpolationDDom>(
    //        std::declval<builder_types>()));
    using batched_derivs_domain_types =
      decltype(batchedDerivsDomainsFromTuple<BatchedInterpolationDDom>(
            std::declval<
              std::tuple<
                std::tuple<
                  std::tuple_element_t<0, builder_types>>,
                std::tuple<
                  std::tuple_element_t<1, builder_types>>,
                std::tuple<
                  std::tuple_element_t<2, builder_types>>>>()));

    template <
            class BatchedInterpolationDDom,
            class = std::enable_if_t<ddc::is_discrete_domain_v<BatchedInterpolationDDom>>>
    using batched_derivs_domain_type1 =
      std::tuple_element_t<0, batched_derivs_domain_types<BatchedInterpolationDDom>>;

    template <
            class BatchedInterpolationDDom,
            class = std::enable_if_t<ddc::is_discrete_domain_v<BatchedInterpolationDDom>>>
    using batched_derivs_domain_type2 =
      std::tuple_element_t<1, batched_derivs_domain_types<BatchedInterpolationDDom>>;

    template <
            class BatchedInterpolationDDom,
            class = std::enable_if_t<ddc::is_discrete_domain_v<BatchedInterpolationDDom>>>
    using batched_derivs_domain_type3 =
      std::tuple_element_t<2, batched_derivs_domain_types<BatchedInterpolationDDom>>;

    // Create a tuple type containing all the derivative domains for a level of derivative
    template <class BatchedInterpolationDDom, class... BuilderTypesTuples>
      static std::tuple<
      ddc::detail::to_whole_derivs_domain_t<
        decltype(interpolation_TupleToTypeSeq(std::declval<BuilderTypesTuples>())),
        decltype(deriv_TupleToTypeSeq(std::declval<BuilderTypesTuples>())),
        ddc::to_type_seq_t<BatchedInterpolationDDom>>...>
        wholeDerivsDomainsFromTuple(std::tuple<BuilderTypesTuples...>);

    /**
     * @brief The type of the whole Derivs domain (1D dimension of interest and cartesian
     * product of 1D Deriv domain and batch domain) in the first dimension, to be passed as
     * argument to the builder, preserving the underlying memory layout (order of dimensions).
     *
     * @tparam The batched discrete domain on which the interpolation points are defined.
     *
     * Example:
     *  - For batched_interpolation_domain_type = DiscreteDomain<X,Y,Z,T> and dimensions of interest X, Y and Z,
     * this is StridedDiscreteDomain<X, Deriv<X>, Y, Z, T>
     *  - For batched_interpolation_domain_type = DiscreteDomain<X,Y,Z,T> and dimensions of interest X, Y and Z,
     * this is StridedDiscreteDomain<Y, X, Deriv<Y>, Z, T>
     *  - For batched_interpolation_domain_type = DiscreteDomain<X,Y,Z,T> and dimensions of interest X, Y and Z,
     * this is StridedDiscreteDomain<Z, X, Y, Deriv<Z>, T>
     */

    template <
            class BatchedInterpolationDDom,
            class = std::enable_if_t<ddc::is_discrete_domain_v<BatchedInterpolationDDom>>>
    using whole_derivs_domain_types =
      decltype(wholeDerivsDomainsFromTuple<BatchedInterpolationDDom>(
            std::declval<
              std::tuple<
                std::tuple<
                  std::tuple_element_t<0, builder_types>>,
                std::tuple<
                  std::tuple_element_t<1, builder_types>>,
                std::tuple<
                  std::tuple_element_t<2, builder_types>>>>()));

    template <
            class BatchedInterpolationDDom,
            class = std::enable_if_t<ddc::is_discrete_domain_v<BatchedInterpolationDDom>>>
    using whole_derivs_domain_type1 = std::tuple_element_t<0, whole_derivs_domain_types<BatchedInterpolationDDom>>;
    template <
            class BatchedInterpolationDDom,
            class = std::enable_if_t<ddc::is_discrete_domain_v<BatchedInterpolationDDom>>>
    using whole_derivs_domain_type2 = std::tuple_element_t<1, whole_derivs_domain_types<BatchedInterpolationDDom>>;
    template <
            class BatchedInterpolationDDom,
            class = std::enable_if_t<ddc::is_discrete_domain_v<BatchedInterpolationDDom>>>
    using whole_derivs_domain_type3 = std::tuple_element_t<2, whole_derivs_domain_types<BatchedInterpolationDDom>>;

    template <class DerivsLayout, class BatchedInterpolationDDom>
    struct DerivLvl1 {
        template <int I>
        using chunk_types = ddc::ChunkSpan<
                double const,
                std::tuple_element_t<I, whole_derivs_domain_types<BatchedInterpolationDDom>>,
                DerivsLayout,
                memory_space>;

        std::tuple<chunk_types<0>, chunk_types<1>, chunk_types<2>> chunks;

        DerivLvl1(chunk_types<0> c1, chunk_types<1> c2, chunk_types<2> c3)
          : chunks(c1, c2, c3)
        {}
    };

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
    using whole_derivs_domain_types2 =
      decltype(wholeDerivsDomainsFromTuple<BatchedInterpolationDDom>(
            std::declval<
              std::tuple<
                std::tuple<
                  std::tuple_element_t<0, builder_types>,
                  std::tuple_element_t<1, builder_types>
                  >,
                std::tuple<
                  std::tuple_element_t<0, builder_types>,
                  std::tuple_element_t<2, builder_types>
                  >,
                std::tuple<
                  std::tuple_element_t<1, builder_types>,
                  std::tuple_element_t<2, builder_types>
                  >>>()));

    template <
            class BatchedInterpolationDDom,
            class = std::enable_if_t<ddc::is_discrete_domain_v<BatchedInterpolationDDom>>>
    using whole_derivs_domain_type1_2 =
    std::tuple_element_t<0, whole_derivs_domain_types2<BatchedInterpolationDDom>>;

    template <
            class BatchedInterpolationDDom,
            class = std::enable_if_t<ddc::is_discrete_domain_v<BatchedInterpolationDDom>>>
    using whole_derivs_domain_type1_3 =
    std::tuple_element_t<1, whole_derivs_domain_types2<BatchedInterpolationDDom>>;

    template <
            class BatchedInterpolationDDom,
            class = std::enable_if_t<ddc::is_discrete_domain_v<BatchedInterpolationDDom>>>
    using whole_derivs_domain_type2_3 =
    std::tuple_element_t<2, whole_derivs_domain_types2<BatchedInterpolationDDom>>;

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
                    std::tuple_element_t<0, interpolation_discrete_dimension_types>,
                    std::tuple_element_t<0, deriv_types>>,
            std::tuple_element_t<1, interpolation_domain_types>,
            std::tuple_element_t<1, deriv_types>>;

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
                    std::tuple_element_t<1, interpolation_discrete_dimension_types>,
                    std::tuple_element_t<1, deriv_types>>,
            std::tuple_element_t<2, interpolation_domain_types>,
            std::tuple_element_t<2, deriv_types>>;

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
                    std::tuple_element_t<0, interpolation_discrete_dimension_types>,
                    std::tuple_element_t<0, deriv_types>>,
            std::tuple_element_t<2, interpolation_domain_types>,
            std::tuple_element_t<2, deriv_types>>;

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
                            std::tuple_element_t<0, interpolation_discrete_dimension_types>,
                            std::tuple_element_t<1, interpolation_discrete_dimension_types>,
                            std::tuple_element_t<2, interpolation_discrete_dimension_types>>,
                    ddc::detail::TypeSeq<std::tuple_element_t<0, deriv_types>,
                                         std::tuple_element_t<1, deriv_types>,
                                         std::tuple_element_t<2, deriv_types>>>>;

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
                    std::tuple_element_t<0, interpolation_discrete_dimension_types>,
                    std::tuple_element_t<1, interpolation_discrete_dimension_types>,
                    std::tuple_element_t<2, interpolation_discrete_dimension_types>>,
            ddc::detail::TypeSeq<std::tuple_element_t<0, deriv_types>, std::tuple_element_t<1, deriv_types>, std::tuple_element_t<2, deriv_types>>,
            ddc::to_type_seq_t<BatchedInterpolationDDom>>;

private:
    std::tuple_element_t<0, builder_types> m_spline_builder1;
    //std::tuple_element_t<1, builder_types> m_spline_builder2;
    //std::tuple_element_t<2, builder_types> m_spline_builder3;
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
                std::tuple_element_t<0, interpolation_domain_types>,
                std::tuple_element_t<1, interpolation_domain_types>,
                std::tuple_element_t<2, interpolation_domain_types>>(
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
    ddc::DiscreteDomain<
      std::tuple_element_t<0, bsplines_types>,
      std::tuple_element_t<1, bsplines_types>,
      std::tuple_element_t<2, bsplines_types>>
        spline_domain()
            const noexcept
    {
        return ddc::DiscreteDomain<
         std::tuple_element_t<0, bsplines_types>, std::tuple_element_t<1, bsplines_types>, std::tuple_element_t<2, bsplines_types>>(
                ddc::discrete_space<std::tuple_element_t<0, bsplines_types>>().full_domain(),
                ddc::discrete_space<std::tuple_element_t<1, bsplines_types>>().full_domain(),
                ddc::discrete_space<std::tuple_element_t<2, bsplines_types>>().full_domain());
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
        return ddc::replace_dim_of<std::tuple_element_t<0, interpolation_discrete_dimension_types>, std::tuple_element_t<0, bsplines_types>>(
                ddc::replace_dim_of<std::tuple_element_t<1, interpolation_discrete_dimension_types>, std::tuple_element_t<1, bsplines_types>>(
                        ddc::replace_dim_of<
                                std::tuple_element_t<2, interpolation_discrete_dimension_types>,
                                std::tuple_element_t<2, bsplines_types>>(batched_interpolation_domain, spline_domain()),
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
            //ddc::ChunkSpan<
            //        double const,
            //        whole_derivs_domain_type1<BatchedInterpolationDDom>,
            //        DerivsLayout,
            //        memory_space> derivs1,
            //ddc::ChunkSpan<
            //        double const,
            //        whole_derivs_domain_type2<BatchedInterpolationDDom>,
            //        DerivsLayout,
            //        memory_space> derivs2,
            //ddc::ChunkSpan<
            //        double const,
            //        whole_derivs_domain_type3<BatchedInterpolationDDom>,
            //        DerivsLayout,
            //        memory_space> derivs3,
         ddc::SplineBuilder3D<ExecSpace, MemorySpace, Solver, DimInfo1, DimInfo2, DimInfo3>::DerivLvl1<DerivsLayout, BatchedInterpolationDDom> derivs1,
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
        ddc::SplineSolver Solver,
        class DimInfo1,
        class DimInfo2,
        class DimInfo3>
template <class Layout, class DerivsLayout, class BatchedInterpolationDDom>
void SplineBuilder3D<
        ExecSpace,
        MemorySpace,
        Solver,
        DimInfo1,
        DimInfo2,
        DimInfo3>::
operator()(
        ddc::ChunkSpan<
                double,
                batched_spline_domain_type<BatchedInterpolationDDom>,
                Layout,
                memory_space> spline,
        ddc::ChunkSpan<double const, BatchedInterpolationDDom, Layout, memory_space> vals,
        ddc::SplineBuilder3D<ExecSpace, MemorySpace, Solver, DimInfo1, DimInfo2, DimInfo3>::DerivLvl1<DerivsLayout, BatchedInterpolationDDom> derivs,
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
  //TODO
  auto derivs1 = std::get<0>(derivs.chunks);
  auto derivs2 = std::get<1>(derivs.chunks);
  auto derivs3 = std::get<2>(derivs.chunks);


    auto const batched_interpolation_domain = vals.domain();

    using ddim2 = std::tuple_element_t<1, interpolation_discrete_dimension_types>;
    using ddim3 = std::tuple_element_t<2, interpolation_discrete_dimension_types>;
    using detail::dmax;
    using detail::dmin;

    assert(interpolation_domain() == interpolation_domain_type(batched_interpolation_domain));

    // Build the derivatives along the second dimension
    auto const spline_batched_derivs2_domain = ddc::detail::get_whole_derivs_domain<std::tuple_element_t<1, deriv_types>>(
            ddc::select<ddim2>(batched_interpolation_domain),
            m_spline_builder1.batched_spline_domain(batched_interpolation_domain),
            std::tuple_element_t<1, bsplines_types>::degree() / 2);

    ddc::Chunk spline_derivs2_alloc(
            spline_batched_derivs2_domain,
            ddc::KokkosAllocator<double, MemorySpace>());
    auto spline_derivs2 = spline_derivs2_alloc.span_view();

    if constexpr (DimInfo2::BcLower == ddc::BoundCond::HERMITE) {
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

    if constexpr (DimInfo2::BcUpper == ddc::BoundCond::HERMITE) {
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
    auto const spline_batched_derivs3_domain = ddc::detail::get_whole_derivs_domain<std::tuple_element_t<2, deriv_types>>(
            ddc::select<ddim3>(batched_interpolation_domain),
            m_spline_builder1.batched_spline_domain(batched_interpolation_domain),
            std::tuple_element_t<2, bsplines_types>::degree() / 2);

    ddc::Chunk spline_derivs3_alloc(
            spline_batched_derivs3_domain,
            ddc::KokkosAllocator<double, MemorySpace>());
    auto spline_derivs3 = spline_derivs3_alloc.span_view();

    if constexpr (DimInfo3::BcLower == ddc::BoundCond::HERMITE) {
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

    if constexpr (DimInfo3::BcUpper == ddc::BoundCond::HERMITE) {
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
            = ddc::detail::get_whole_derivs_domain<std::tuple_element_t<1, deriv_types>, std::tuple_element_t<2, deriv_types>>(
                    ddc::select<ddim2, ddim3>(batched_interpolation_domain),
                    m_spline_builder1.batched_spline_domain(batched_interpolation_domain),
                    std::tuple_element_t<1, bsplines_types>::degree() / 2,
                    std::tuple_element_t<2, bsplines_types>::degree() / 2);

    ddc::Chunk spline_derivs2_3_alloc(
            spline_batched_derivs2_3_domain,
            ddc::KokkosAllocator<double, MemorySpace>());
    auto spline_derivs2_3 = spline_derivs2_3_alloc.span_view();

    if constexpr (DimInfo2::BcLower == ddc::BoundCond::HERMITE || DimInfo3::BcLower == ddc::BoundCond::HERMITE) {
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

    if constexpr (DimInfo2::BcLower == ddc::BoundCond::HERMITE || DimInfo3::BcUpper == ddc::BoundCond::HERMITE) {
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

    if constexpr (DimInfo2::BcUpper == ddc::BoundCond::HERMITE || DimInfo3::BcLower == ddc::BoundCond::HERMITE) {
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

    if constexpr (DimInfo2::BcUpper == ddc::BoundCond::HERMITE || DimInfo3::BcUpper == ddc::BoundCond::HERMITE) {
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
