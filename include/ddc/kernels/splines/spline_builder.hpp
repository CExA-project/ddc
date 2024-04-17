// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once
#include "ddc/chunk_span.hpp"
#include "ddc/discrete_domain.hpp"
#include "ddc/kokkos_allocator.hpp"

#include "deriv.hpp"

namespace ddc {

/**
 * @brief An enum determining the backend solver of a SplineBuilder or SplineBuilder2d.
 *
 * An enum determining the backend solver of a SplineBuilder or SplineBuilder2d. Only GINKGO available at the moment,
 * other solvers will be implemented in the futur.
 */
enum class SplineSolver {
    GINKGO
};

/**
 * @brief An helper giving the uniform/non_uniform status of a spline interpolation mesh according to its attributes.
 *
 * An helper giving the uniform/non_uniform status of a spline interpolation mesh according to its attributes.
 * @param is_uniform A boolean giving the presumed status before considering boundary conditions.
 * @param BcXmin The lower boundary condition.
 * @param BcXmax The upper boundary condition.
 * @param int The degree of the spline.
 */
constexpr bool is_spline_interpolation_mesh_uniform(
        bool const is_uniform,
        ddc::BoundCond const BcXmin,
        ddc::BoundCond const BcXmax,
        int degree)
{
    int N_BE_MIN = n_boundary_equations(BcXmin, degree);
    int N_BE_MAX = n_boundary_equations(BcXmax, degree);
    bool is_periodic = (BcXmin == ddc::BoundCond::PERIODIC) && (BcXmax == ddc::BoundCond::PERIODIC);
    return is_uniform && ((N_BE_MIN != 0 && N_BE_MAX != 0) || is_periodic);
}

/**
 * @brief A class for creating a spline approximation of a function.
 *
 * A class which contains an operator () which can be used to build a spline approximation
 * of a function. A spline approximation is represented by coefficients stored in a Chunk
 * of BSplines. The spline is constructed such that it respects the boundary conditions
 * BcXmin and BcXmax, and it interpolates the function at the points on the interpolation_mesh
 * associated with interpolation_mesh_type.
 * @tparam ExecSpace The Kokkos execution space on which the spline transform is performed.
 * @tparam MemorySpace The Kokkos memory space on which the data (interpolation function and splines coefficients) are stored.
 * @tparam BSplines The discrete dimension representing the BSplines.
 * @tparam InterpolationMesh The discrete dimension supporting the interpolation points.
 * @tparam BcXmin The lower boundary condition.
 * @tparam BcXmax The upper boundary condition.
 * @tparam Solver The SplineSolver giving the backend used to perform the spline transform.
 * @tparam IDimX A variadic template of all the discrete dimensions forming the full space (InterpolationMesh + batched dimensions).
 */
template <
        class ExecSpace,
        class MemorySpace,
        class BSplines,
        class InterpolationMesh,
        ddc::BoundCond BcXmin,
        ddc::BoundCond BcXmax,
        SplineSolver Solver,
        class... IDimX>
class SplineBuilder
{
    static_assert(
            (BSplines::is_periodic() && (BcXmin == ddc::BoundCond::PERIODIC)
             && (BcXmax == ddc::BoundCond::PERIODIC))
            || (!BSplines::is_periodic() && (BcXmin != ddc::BoundCond::PERIODIC)
                && (BcXmax != ddc::BoundCond::PERIODIC)));
    static_assert(!BSplines::is_radial());

private:
    using tag_type = typename InterpolationMesh::continuous_dimension_type;

public:
    /**
     * @brief The type of the Kokkos execution space used by this class.
     */
    using exec_space = ExecSpace;

    /**
     * @brief The type of the Kokkos memory space used by this class.
     */
    using memory_space = MemorySpace;

    /**
     * @brief The type of the interpolation discrete dimension (discrete dimension of interest) used by this class.
     */
    using interpolation_mesh_type = InterpolationMesh;

    /**
     * @brief The discrete dimension representing the BSplines.
     */
    using bsplines_type = BSplines;

    /**
     * @brief The Deriv dimension at the boundaries.
     */
    using deriv_type = ddc::Deriv<tag_type>;

    /**
     * @brief The type of the domain for the 1D interpolation mesh used by this class.
     */
    using interpolation_domain_type = ddc::DiscreteDomain<interpolation_mesh_type>;

    /**
     * @brief The type of the whole domain representing interpolation points.
     */
    using batched_interpolation_domain_type = ddc::DiscreteDomain<IDimX...>;

    /**
     * @brief The type of the batch domain (obtained by removing dimension of interest from whole space).
     */
    using batch_domain_type =
            typename ddc::detail::convert_type_seq_to_discrete_domain<ddc::type_seq_remove_t<
                    ddc::detail::TypeSeq<IDimX...>,
                    ddc::detail::TypeSeq<interpolation_mesh_type>>>;

    /**
     * @brief Helper to get the dimension of batched_spline_domain_type associated to a dimension of batched_interpolation_domain_type.
     */
    template <typename Tag>
    using spline_dim_type
            = std::conditional_t<std::is_same_v<Tag, interpolation_mesh_type>, bsplines_type, Tag>;

    /**
     * @brief The type of the whole spline domain (cartesian product of 1D spline domain and batch domain) preserving the underlying memory layout (order of dimensions).
     */
    using batched_spline_domain_type =
            typename ddc::detail::convert_type_seq_to_discrete_domain<ddc::type_seq_replace_t<
                    ddc::detail::TypeSeq<IDimX...>,
                    ddc::detail::TypeSeq<interpolation_mesh_type>,
                    ddc::detail::TypeSeq<bsplines_type>>>;

    /**
     * @brief The type of the whole spline domain (cartesian product of 1D spline domain and batch domain) with 1D spline domain being contiguous .
     */
    using batched_spline_tr_domain_type =
            typename ddc::detail::convert_type_seq_to_discrete_domain<ddc::type_seq_merge_t<
                    ddc::detail::TypeSeq<bsplines_type>,
                    ddc::type_seq_remove_t<
                            ddc::detail::TypeSeq<IDimX...>,
                            ddc::detail::TypeSeq<interpolation_mesh_type>>>>;

    /**
     * @brief The type of the whole Derivs domain (cartesian product of 1D Deriv domain and batch domain) preserving the underlying memory layout (order of dimensions).
     */
    using batched_derivs_domain_type =
            typename ddc::detail::convert_type_seq_to_discrete_domain<ddc::type_seq_replace_t<
                    ddc::detail::TypeSeq<IDimX...>,
                    ddc::detail::TypeSeq<interpolation_mesh_type>,
                    ddc::detail::TypeSeq<deriv_type>>>;

    /**
     * @brief Indicates if the degree of the splines is odd or even.
     */
    static constexpr bool s_odd = BSplines::degree() % 2;

    /**
     * @brief The number of equations defining the boundary conditions at the lower bound.
     */
    static constexpr int s_nbe_xmin = n_boundary_equations(BcXmin, BSplines::degree());

    /**
     * @brief The number of equations defining the boundary conditions at the upper bound.
     */
    static constexpr int s_nbe_xmax = n_boundary_equations(BcXmax, BSplines::degree());

    /**
     * @brief The number of boundary conditions which must be provided by the user at the lower bound.
     *
     * This value is usually equal to s_nbe_xmin, but it may be difference if the chosen boundary
     * conditions impose a specific value (e.g. no values need to be provided for Dirichlet boundary
     * conditions).
     */
    static constexpr int s_nbc_xmin = n_user_input(BcXmin, BSplines::degree());

    /**
     * @brief The number of boundary conditions which must be provided by the user at the upper bound.
     *
     * This value is usually equal to s_nbe_xmin, but it may be difference if the chosen boundary
     * conditions impose a specific value (e.g. no values need to be provided for Dirichlet boundary
     * conditions).
     */
    static constexpr int s_nbc_xmax = n_user_input(BcXmax, BSplines::degree());

    /**
     * @brief The boundary condition implemented at the lower bound.
     */
    static constexpr ddc::BoundCond s_bc_xmin = BcXmin;

    /**
     * @brief The boundary condition implemented at the upper bound.
     */
    static constexpr ddc::BoundCond s_bc_xmax = BcXmax;

private:
    batched_interpolation_domain_type m_batched_interpolation_domain;

    int m_offset;

    double m_dx; // average cell size for normalization of derivatives

    // interpolator specific
    std::unique_ptr<ddc::detail::Matrix> matrix;

public:
    /**
     * @brief An helper to compute the offset.
     */
    int compute_offset(interpolation_domain_type const& interpolation_domain);

    /**
     * @brief Build a SplineBuilder acting on batched_interpolation_domain.
     * 
     * @param batched_interpolation_domain The domain on which are defined the interpolation points.
     * @param cols_per_chunk An hyperparameter used by the slicer (internal to the solver) to define the size of a chunk of right-and-sides of the linear problem to be computed in parallel.
     */
    explicit SplineBuilder(
            batched_interpolation_domain_type const& batched_interpolation_domain,
            std::optional<int> cols_per_chunk = std::nullopt,
            std::optional<unsigned int> preconditionner_max_block_size = std::nullopt)
        : m_batched_interpolation_domain(batched_interpolation_domain)
        , m_offset(compute_offset(interpolation_domain()))
        , m_dx((ddc::discrete_space<BSplines>().rmax() - ddc::discrete_space<BSplines>().rmin())
               / ddc::discrete_space<BSplines>().ncells())
    {
        static_assert(
                ((BcXmin == BoundCond::PERIODIC) == (BcXmax == BoundCond::PERIODIC)),
                "Incompatible boundary conditions");

        // Calculate block sizes
        int lower_block_size, upper_block_size;
        if constexpr (bsplines_type::is_uniform()) {
            compute_block_sizes_uniform(lower_block_size, upper_block_size);
        } else {
            compute_block_sizes_non_uniform(lower_block_size, upper_block_size);
        }
        allocate_matrix(
                lower_block_size,
                upper_block_size,
                cols_per_chunk,
                preconditionner_max_block_size);
    }

    SplineBuilder(SplineBuilder const& x) = delete;

    /**
     * @brief Create a new SplineBuilder by copy
     *
     * @param x The SplineBuilder being copied.
     */
    SplineBuilder(SplineBuilder&& x) = default;

    ~SplineBuilder() = default;

    SplineBuilder& operator=(SplineBuilder const& x) = delete;

    /**
     * @brief Copy a SplineBuilder.
     *
     * @param x The SplineBuilder being copied.
     * @returns A reference to this object.
     */
    SplineBuilder& operator=(SplineBuilder&& x) = default;

    /**
     * @brief Get the domain for the 1D interpolation mesh used by this class.
     *
     * Get the 1D interpolation domain associated to dimension of interest.
     *
     * @return The 1D domain for the grid points.
     */
    interpolation_domain_type interpolation_domain() const noexcept
    {
        return interpolation_domain_type(m_batched_interpolation_domain);
    }

    /**
     * @brief Get the whole domain representing interpolation points.
     *
     * Get the domain on which values of the function must be provided in order
     * to build a spline transform of the function.
     *
     * @return The domain for the grid points.
     */
    batched_interpolation_domain_type batched_interpolation_domain() const noexcept
    {
        return m_batched_interpolation_domain;
    }

    /**
     * @brief Get the batch domain.
     *
     * Get the batch domain (obtained by removing dimension of interest from whole interpolation domain).
     *
     * @return The batch domain.
     */
    batch_domain_type batch_domain() const noexcept
    {
        return ddc::remove_dims_of(batched_interpolation_domain(), interpolation_domain());
    }

    /**
     * @brief Get the 1D domain on which spline coefficients are defined.
     *
     * Get the 1D spline domain corresponding to dimension of interest.
     *
     * @return The 1D domain for the spline coefficients.
     */
    ddc::DiscreteDomain<bsplines_type> spline_domain() const noexcept
    {
        return ddc::discrete_space<bsplines_type>().full_domain();
    }

    /**
     * @brief Get the whole domain on which spline coefficients are defined, preserving memory layout.
     *
     * Get the whole domain on which spline coefficients will be computed, preserving memory layout (order of dimensions).
     *
     * @return The domain for the spline coefficients.
     */
    batched_spline_domain_type batched_spline_domain() const noexcept
    {
        return ddc::replace_dim_of<
                interpolation_mesh_type,
                bsplines_type>(batched_interpolation_domain(), spline_domain());
    }

    /**
     * @brief Get the whole domain on which spline coefficients are defined, with dimension of interest contiguous.
     *
     * Get the (transposed) whole domain on which spline coefficients will be computed, with dimension of interest contiguous.
     *
     * @return The (transposed) domain for the spline coefficients.
     */
    batched_spline_tr_domain_type batched_spline_tr_domain() const noexcept
    {
        return batched_spline_tr_domain_type(spline_domain(), batch_domain());
    }

    /**
     * @brief Get the whole domain on which derivatives on lower boundary are defined.
     *
     * Get the whole domain on which derivatives on lower boundary are defined. This is used only with HERMITE boundary conditions.
     *
     * @return The domain for the Derivs values.
     */
    batched_derivs_domain_type batched_derivs_xmin_domain() const noexcept
    {
        return ddc::replace_dim_of<interpolation_mesh_type, deriv_type>(
                batched_interpolation_domain(),
                ddc::DiscreteDomain<deriv_type>(
                        ddc::DiscreteElement<deriv_type>(1),
                        ddc::DiscreteVector<deriv_type>(s_nbc_xmin)));
    }

    /**
     * @brief Get the whole domain on which derivatives on upper boundary are defined.
     *
     * Get the whole domain on which derivatives on upper boundary are defined. This is used only with HERMITE boundary conditions.
     *
     * @return The domain for the Derivs values.
     */
    batched_derivs_domain_type batched_derivs_xmax_domain() const noexcept
    {
        return ddc::replace_dim_of<interpolation_mesh_type, deriv_type>(
                batched_interpolation_domain(),
                ddc::DiscreteDomain<deriv_type>(
                        ddc::DiscreteElement<deriv_type>(1),
                        ddc::DiscreteVector<deriv_type>(s_nbc_xmax)));
    }

    /**
     * @brief Get the interpolation matrix.
     *
     * Get the interpolation matrix. This can be useful for debugging (as it allows
     * one to print the matrix) or for more complex quadrature schemes.
	 *
	 * Warning: the returned ddc::detail::Matrix class is not supposed to be exposed
	 * to user, which means its usage is not tested out of the scope of DDC splines transforms.
	 * Use at your own risk.
     *
     * @return A reference to the interpolation matrix.
     */
    const ddc::detail::Matrix& get_interpolation_matrix() const noexcept
    {
        return *matrix;
    }

    /**
     * @brief Build a spline approximation of a function.
     *
     * Use the values of a function at known grid points (as specified by
     * SplineBuilder::interpolation_domain) and the derivatives of the
     * function at the boundaries (if necessary for the chosen boundary
     * conditions) to calculate a spline approximation of a function.
     *
     * The spline approximation is stored as a ChunkSpan of coefficients
     * associated with basis-splines.
     *
     * @param[out] spline The coefficients of the spline calculated by the function.
     * @param[in] vals The values of the function at the grid points.
     * @param[in] derivs_xmin The values of the derivatives at the lower boundary
	 * (used only with HERMITE lower boundary condition).
     * @param[in] derivs_xmax The values of the derivatives at the upper boundary
	 * (used only with HERMITE upper boundary condition).
     */
    template <class Layout>
    void operator()(
            ddc::ChunkSpan<double, batched_spline_domain_type, Layout, memory_space> spline,
            ddc::ChunkSpan<double const, batched_interpolation_domain_type, Layout, memory_space>
                    vals,
            std::optional<ddc::ChunkSpan<
                    double const,
                    batched_derivs_domain_type,
                    Layout,
                    memory_space>> const derivs_xmin
            = std::nullopt,
            std::optional<ddc::ChunkSpan<
                    double const,
                    batched_derivs_domain_type,
                    Layout,
                    memory_space>> const derivs_xmax
            = std::nullopt) const;

private:
    void compute_block_sizes_uniform(int& lower_block_size, int& upper_block_size) const;

    void compute_block_sizes_non_uniform(int& lower_block_size, int& upper_block_size) const;

    void allocate_matrix(
            int lower_block_size,
            int upper_block_size,
            std::optional<int> cols_per_chunk = std::nullopt,
            std::optional<unsigned int> preconditionner_max_block_size = std::nullopt);

    void build_matrix_system();
};

template <
        class ExecSpace,
        class MemorySpace,
        class BSplines,
        class InterpolationMesh,
        ddc::BoundCond BcXmin,
        ddc::BoundCond BcXmax,
        SplineSolver Solver,
        class... IDimX>
int SplineBuilder<
        ExecSpace,
        MemorySpace,
        BSplines,
        InterpolationMesh,
        BcXmin,
        BcXmax,
        Solver,
        IDimX...>::compute_offset(interpolation_domain_type const& interpolation_domain)
{
    int offset;
    if constexpr (bsplines_type::is_periodic()) {
        // Calculate offset so that the matrix is diagonally dominant
        std::array<double, bsplines_type::degree() + 1> values_ptr;
        std::experimental::mdspan<
                double,
                std::experimental::extents<std::size_t, bsplines_type::degree() + 1>> const
                values(values_ptr.data());
        ddc::DiscreteElement<interpolation_mesh_type> start(interpolation_domain.front());
        auto jmin = ddc::discrete_space<BSplines>()
                            .eval_basis(values, ddc::coordinate(start + BSplines::degree()));
        if constexpr (bsplines_type::degree() % 2 == 0) {
            offset = jmin.uid() - start.uid() + bsplines_type::degree() / 2 - BSplines::degree();
        } else {
            int const mid = bsplines_type::degree() / 2;
            offset = jmin.uid() - start.uid() + (values(mid) > values(mid + 1) ? mid : mid + 1)
                     - BSplines::degree();
        }
    } else {
        offset = 0;
    }
    return offset;
}

template <
        class ExecSpace,
        class MemorySpace,
        class BSplines,
        class InterpolationMesh,
        ddc::BoundCond BcXmin,
        ddc::BoundCond BcXmax,
        SplineSolver Solver,
        class... IDimX>
void SplineBuilder<
        ExecSpace,
        MemorySpace,
        BSplines,
        InterpolationMesh,
        BcXmin,
        BcXmax,
        Solver,
        IDimX...>::compute_block_sizes_uniform(int& lower_block_size, int& upper_block_size) const
{
    switch (BcXmin) {
    case ddc::BoundCond::PERIODIC:
        upper_block_size = (bsplines_type::degree()) / 2;
        break;
    case ddc::BoundCond::NATURAL:
    case ddc::BoundCond::HERMITE:
        upper_block_size = s_nbc_xmin;
        break;
    case ddc::BoundCond::GREVILLE:
        upper_block_size = bsplines_type::degree() - 1;
        break;
    default:
        throw std::runtime_error("ddc::BoundCond not handled");
    }
    switch (BcXmax) {
    case ddc::BoundCond::PERIODIC:
        lower_block_size = (bsplines_type::degree()) / 2;
        break;
    case ddc::BoundCond::NATURAL:
    case ddc::BoundCond::HERMITE:
        lower_block_size = s_nbc_xmax;
        break;
    case ddc::BoundCond::GREVILLE:
        lower_block_size = bsplines_type::degree() - 1;
        break;
    default:
        throw std::runtime_error("ddc::BoundCond not handled");
    }
}

template <
        class ExecSpace,
        class MemorySpace,
        class BSplines,
        class InterpolationMesh,
        ddc::BoundCond BcXmin,
        ddc::BoundCond BcXmax,
        SplineSolver Solver,
        class... IDimX>
void SplineBuilder<
        ExecSpace,
        MemorySpace,
        BSplines,
        InterpolationMesh,
        BcXmin,
        BcXmax,
        Solver,
        IDimX...>::compute_block_sizes_non_uniform(int& lower_block_size, int& upper_block_size)
        const
{
    switch (BcXmin) {
    case ddc::BoundCond::PERIODIC:
        upper_block_size = bsplines_type::degree() - 1;
        break;
    case ddc::BoundCond::HERMITE:
        upper_block_size = s_nbc_xmin + 1;
        break;
    case ddc::BoundCond::GREVILLE:
        upper_block_size = bsplines_type::degree() - 1;
        break;
    default:
        throw std::runtime_error("ddc::BoundCond not handled");
    }
    switch (BcXmax) {
    case ddc::BoundCond::PERIODIC:
        lower_block_size = bsplines_type::degree() - 1;
        break;
    case ddc::BoundCond::HERMITE:
        lower_block_size = s_nbc_xmax + 1;
        break;
    case ddc::BoundCond::GREVILLE:
        lower_block_size = bsplines_type::degree() - 1;
        break;
    default:
        throw std::runtime_error("ddc::BoundCond not handled");
    }
}

template <
        class ExecSpace,
        class MemorySpace,
        class BSplines,
        class InterpolationMesh,
        ddc::BoundCond BcXmin,
        ddc::BoundCond BcXmax,
        SplineSolver Solver,
        class... IDimX>
void SplineBuilder<
        ExecSpace,
        MemorySpace,
        BSplines,
        InterpolationMesh,
        BcXmin,
        BcXmax,
        Solver,
        IDimX...>::
        allocate_matrix(
                [[maybe_unused]] int lower_block_size,
                [[maybe_unused]] int upper_block_size,
                std::optional<int> cols_per_chunk,
                std::optional<unsigned int> preconditionner_max_block_size)
{
    // Special case: linear spline
    // No need for matrix assembly
    // (desactivated)
    /*
    if constexpr (bsplines_type::degree() == 1)
        return;
	*/

    matrix = ddc::detail::MatrixMaker::make_new_sparse<ExecSpace>(
            ddc::discrete_space<BSplines>().nbasis(),
            cols_per_chunk,
            preconditionner_max_block_size);

    build_matrix_system();

    matrix->factorize();
}

template <
        class ExecSpace,
        class MemorySpace,
        class BSplines,
        class InterpolationMesh,
        ddc::BoundCond BcXmin,
        ddc::BoundCond BcXmax,
        SplineSolver Solver,
        class... IDimX>
void SplineBuilder<
        ExecSpace,
        MemorySpace,
        BSplines,
        InterpolationMesh,
        BcXmin,
        BcXmax,
        Solver,
        IDimX...>::build_matrix_system()
{
    // Hermite boundary conditions at xmin, if any
    if constexpr (BcXmin == ddc::BoundCond::HERMITE) {
        std::array<double, (bsplines_type::degree() / 2 + 1) * (bsplines_type::degree() + 1)>
                derivs_ptr;
        ddc::DSpan2D
                derivs(derivs_ptr.data(),
                       bsplines_type::degree() + 1,
                       bsplines_type::degree() / 2 + 1);
        ddc::discrete_space<BSplines>().eval_basis_and_n_derivs(
                derivs,
                ddc::discrete_space<BSplines>().rmin(),
                s_nbc_xmin);

        // In order to improve the condition number of the matrix, we normalize
        // all derivatives by multiplying the i-th derivative by dx^i
        for (std::size_t i = 0; i < bsplines_type::degree() + 1; ++i) {
            for (std::size_t j = 1; j < bsplines_type::degree() / 2 + 1; ++j) {
                derivs(i, j) *= ddc::detail::ipow(m_dx, j);
            }
        }

        // iterate only to deg as last bspline is 0
        for (std::size_t i = 0; i < s_nbc_xmin; ++i) {
            for (std::size_t j = 0; j < bsplines_type::degree(); ++j) {
                matrix->set_element(i, j, derivs(j, s_nbc_xmin - i - 1 + s_odd));
            }
        }
    }

    // Interpolation points
    std::array<double, bsplines_type::degree() + 1> values_ptr;
    std::experimental::mdspan<
            double,
            std::experimental::extents<std::size_t, bsplines_type::degree() + 1>> const
            values(values_ptr.data());

    int start = interpolation_domain().front().uid();
    ddc::for_each(interpolation_domain(), [&](auto ix) {
        auto jmin = ddc::discrete_space<BSplines>().eval_basis(
                values,
                ddc::coordinate(ddc::DiscreteElement<interpolation_mesh_type>(ix)));
        for (std::size_t s = 0; s < bsplines_type::degree() + 1; ++s) {
            int const j = ddc::detail::
                    modulo(int(jmin.uid() - m_offset + s),
                           (int)ddc::discrete_space<BSplines>().nbasis());
            matrix->set_element(ix.uid() - start + s_nbc_xmin, j, values(s));
        }
    });

    // Hermite boundary conditions at xmax, if any
    if constexpr (BcXmax == ddc::BoundCond::HERMITE) {
        std::array<double, (bsplines_type::degree() / 2 + 1) * (bsplines_type::degree() + 1)>
                derivs_ptr;
        std::experimental::mdspan<
                double,
                std::experimental::extents<
                        std::size_t,
                        bsplines_type::degree() + 1,
                        bsplines_type::degree() / 2 + 1>> const derivs(derivs_ptr.data());

        ddc::discrete_space<BSplines>().eval_basis_and_n_derivs(
                derivs,
                ddc::discrete_space<BSplines>().rmax(),
                s_nbc_xmax);

        // In order to improve the condition number of the matrix, we normalize
        // all derivatives by multiplying the i-th derivative by dx^i
        for (std::size_t i = 0; i < bsplines_type::degree() + 1; ++i) {
            for (std::size_t j = 1; j < bsplines_type::degree() / 2 + 1; ++j) {
                derivs(i, j) *= ddc::detail::ipow(m_dx, j);
            }
        }

        int const i0 = ddc::discrete_space<BSplines>().nbasis() - s_nbc_xmax;
        int const j0 = ddc::discrete_space<BSplines>().nbasis() - bsplines_type::degree();
        for (std::size_t j = 0; j < bsplines_type::degree(); ++j) {
            for (std::size_t i = 0; i < s_nbc_xmax; ++i) {
                matrix->set_element(i0 + i, j0 + j, derivs(j + 1, i + s_odd));
            }
        }
    }
}

template <
        class ExecSpace,
        class MemorySpace,
        class BSplines,
        class InterpolationMesh,
        ddc::BoundCond BcXmin,
        ddc::BoundCond BcXmax,
        SplineSolver Solver,
        class... IDimX>
template <class Layout>
void SplineBuilder<
        ExecSpace,
        MemorySpace,
        BSplines,
        InterpolationMesh,
        BcXmin,
        BcXmax,
        Solver,
        IDimX...>::
operator()(
        ddc::ChunkSpan<double, batched_spline_domain_type, Layout, memory_space> spline,
        ddc::ChunkSpan<double const, batched_interpolation_domain_type, Layout, memory_space> vals,
        std::optional<ddc::ChunkSpan<
                double const,
                batched_derivs_domain_type,
                Layout,
                memory_space>> const derivs_xmin,
        std::optional<ddc::ChunkSpan<
                double const,
                batched_derivs_domain_type,
                Layout,
                memory_space>> const derivs_xmax) const
{
    assert(vals.template extent<interpolation_mesh_type>()
           == ddc::discrete_space<bsplines_type>().nbasis() - s_nbe_xmin - s_nbe_xmax);

    assert((BcXmin == ddc::BoundCond::HERMITE)
           != (!derivs_xmin.has_value() || derivs_xmin->template extent<deriv_type>() == 0));
    assert((BcXmax == ddc::BoundCond::HERMITE)
           != (!derivs_xmax.has_value() || derivs_xmax->template extent<deriv_type>() == 0));
    if constexpr (BcXmin == BoundCond::HERMITE) {
        assert(ddc::DiscreteElement<deriv_type>(derivs_xmin->domain().front()).uid() == 1);
    }
    if constexpr (BcXmax == BoundCond::HERMITE) {
        assert(ddc::DiscreteElement<deriv_type>(derivs_xmax->domain().front()).uid() == 1);
    }

    // Hermite boundary conditions at xmin, if any
    // NOTE: For consistency with the linear system, the i-th derivative
    //       provided by the user must be multiplied by dx^i
    if constexpr (BcXmin == BoundCond::HERMITE) {
        assert(derivs_xmin->template extent<deriv_type>() == s_nbc_xmin);
        auto derivs_xmin_values = *derivs_xmin;
        auto const dx_proxy = m_dx;
        ddc::parallel_for_each(
                exec_space(),
                batch_domain(),
                KOKKOS_LAMBDA(typename batch_domain_type::discrete_element_type j) {
                    for (int i = s_nbc_xmin; i > 0; --i) {
                        spline(ddc::DiscreteElement<bsplines_type>(s_nbc_xmin - i), j)
                                = derivs_xmin_values(ddc::DiscreteElement<deriv_type>(i), j)
                                  * ddc::detail::ipow(dx_proxy, i + s_odd - 1);
                    }
                });
    }

    // TODO : Consider optimizing
    // Fill spline with vals (to work in spline afterward and preserve vals)
    auto const& offset_proxy = m_offset;
    auto const& interp_size_proxy = interpolation_domain().extents();
    auto const& nbasis_proxy = ddc::discrete_space<bsplines_type>().nbasis();
    ddc::parallel_for_each(
            exec_space(),
            batch_domain(),
            KOKKOS_LAMBDA(typename batch_domain_type::discrete_element_type j) {
                for (int i = s_nbc_xmin; i < s_nbc_xmin + offset_proxy; ++i) {
                    spline(ddc::DiscreteElement<bsplines_type>(i), j) = 0.0;
                }
                for (int i = 0; i < interp_size_proxy; ++i) {
                    spline(ddc::DiscreteElement<bsplines_type>(s_nbc_xmin + i + offset_proxy), j)
                            = vals(ddc::DiscreteElement<interpolation_mesh_type>(i), j);
                }
            });

    // Hermite boundary conditions at xmax, if any
    // NOTE: For consistency with the linear system, the i-th derivative
    //       provided by the user must be multiplied by dx^i
    if constexpr (BcXmax == BoundCond::HERMITE) {
        assert(derivs_xmax->template extent<deriv_type>() == s_nbc_xmax);
        auto derivs_xmax_values = *derivs_xmax;
        auto const dx_proxy = m_dx;
        ddc::parallel_for_each(
                exec_space(),
                batch_domain(),
                KOKKOS_LAMBDA(typename batch_domain_type::discrete_element_type j) {
                    for (int i = 0; i < s_nbc_xmax; ++i) {
                        spline(ddc::DiscreteElement<bsplines_type>(nbasis_proxy - s_nbc_xmax - i),
                               j)
                                = derivs_xmax_values(ddc::DiscreteElement<deriv_type>(i + 1), j)
                                  * ddc::detail::ipow(dx_proxy, i + s_odd);
                    }
                });
    }

    // TODO : Consider optimizing
    // Allocate and fill a transposed version of spline in order to get dimension of interest as last dimension (optimal for GPU, necessary for Ginkgo)
    ddc::Chunk spline_tr_alloc(
            batched_spline_tr_domain(),
            ddc::KokkosAllocator<double, memory_space>());
    ddc::ChunkSpan spline_tr = spline_tr_alloc.span_view();
    ddc::parallel_for_each(
            exec_space(),
            batch_domain(),
            KOKKOS_LAMBDA(typename batch_domain_type::discrete_element_type const j) {
                for (std::size_t i = 0; i < nbasis_proxy; i++) {
                    spline_tr(ddc::DiscreteElement<bsplines_type>(i), j)
                            = spline(ddc::DiscreteElement<bsplines_type>(i + offset_proxy), j);
                }
            });
    // Create a 2D Kokkos::View to manage spline_tr as a matrix
    Kokkos::View<double**, Kokkos::LayoutRight, exec_space> bcoef_section(
            spline_tr.data_handle(),
            ddc::discrete_space<bsplines_type>().nbasis(),
            batch_domain().size());
    // Compute spline coef
    matrix->solve_batch_inplace(bcoef_section);
    // Transpose back spline_tr in spline
    ddc::parallel_for_each(
            exec_space(),
            batch_domain(),
            KOKKOS_LAMBDA(typename batch_domain_type::discrete_element_type const j) {
                for (std::size_t i = 0; i < nbasis_proxy; i++) {
                    spline(ddc::DiscreteElement<bsplines_type>(i + offset_proxy), j)
                            = spline_tr(ddc::DiscreteElement<bsplines_type>(i), j);
                }
            });

    // Not sure yet of what this part do
    if (bsplines_type::is_periodic()) {
        ddc::parallel_for_each(
                exec_space(),
                batch_domain(),
                KOKKOS_LAMBDA(typename batch_domain_type::discrete_element_type const j) {
                    if (offset_proxy != 0) {
                        for (int i = 0; i < offset_proxy; ++i) {
                            spline(ddc::DiscreteElement<bsplines_type>(i), j) = spline(
                                    ddc::DiscreteElement<bsplines_type>(nbasis_proxy + i),
                                    j);
                        }
                        for (std::size_t i = offset_proxy; i < bsplines_type::degree(); ++i) {
                            spline(ddc::DiscreteElement<bsplines_type>(nbasis_proxy + i), j)
                                    = spline(ddc::DiscreteElement<bsplines_type>(i), j);
                        }
                    }
                    for (std::size_t i(0); i < bsplines_type::degree(); ++i) {
                        const ddc::DiscreteElement<bsplines_type> i_start(i);
                        const ddc::DiscreteElement<bsplines_type> i_end(nbasis_proxy + i);

                        spline(i_end, j) = spline(i_start, j);
                    }
                });
    }
}
} // namespace ddc
