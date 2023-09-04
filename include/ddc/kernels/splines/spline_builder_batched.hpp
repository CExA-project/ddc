#pragma once
#include "spline_builder.hpp"


template <class SplineBuilder, class... BatchTags> // TODO : Remove BatchedTags... dependency (compute it automatically)
class SplineBuilderBatched
{
private:
    using tag_type = typename SplineBuilder::bsplines_type::tag_type;
public:
    using bsplines_type = typename SplineBuilder::bsplines_type;

    using builder_type = SplineBuilder;

    using interpolation_mesh_type = typename SplineBuilder::mesh_type;
    using batch_mesh_type = typename ddc::DiscreteElement<BatchTags...>;

    using interpolation_domain_type = ddc::DiscreteDomain<interpolation_mesh_type>;
    
    using batch_domain_type = ddc::DiscreteDomain<BatchTags...>; // TODO : Compute

    static constexpr BoundCond BcXmin = SplineBuilder::s_bc_xmin;
    static constexpr BoundCond BcXmax = SplineBuilder::s_bc_xmax;

private:
    builder_type spline_builder;
    interpolation_domain_type m_interpolation_domain;
	batch_domain_type m_batch_domain;

public:
    SplineBuilderBatched(interpolation_domain_type const& interpolation_domain, batch_domain_type const& batch_domain)
        : spline_builder(ddc::select<interpolation_mesh_type>(interpolation_domain))
        , m_interpolation_domain(interpolation_domain)
        , m_batch_domain(batch_domain)
    {
		// auto m_batch_domain = ddc::remove_dims_of<interpolation_mesh_type(dom, ddc::DiscreteDomain<DDimSp, DDimX, DDimV>(sp_dom, x_dom, v_dom)); // TODO
    }

    SplineBuilderBatched(SplineBuilderBatched const& x) = delete;

    SplineBuilderBatched(SplineBuilderBatched&& x) = default;

    ~SplineBuilderBatched() = default;

    SplineBuilderBatched& operator=(SplineBuilderBatched const& x) = delete;

    SplineBuilderBatched& operator=(SplineBuilderBatched&& x) = default;

    void operator()(
            ddc::ChunkSpan<double, ddc::DiscreteDomain<bsplines_type, BatchTags...>> spline,
            ddc::ChunkSpan<double, ddc::DiscreteDomain<interpolation_mesh_type,BatchTags...>> vals,
            std::optional<CDSpan2D> const derivs_xmin = std::nullopt,
            std::optional<CDSpan2D> const derivs_xmax = std::nullopt) const;
            // std::optional<CDSpan2D> const derivs_ymin = std::nullopt,
            // std::optional<CDSpan2D> const derivs_ymax = std::nullopt,
            // std::optional<CDSpan2D> const mixed_derivs_xmin_ymin = std::nullopt,
            // std::optional<CDSpan2D> const mixed_derivs_xmax_ymin = std::nullopt,
            // std::optional<CDSpan2D> const mixed_derivs_xmin_ymax = std::nullopt,
            // std::optional<CDSpan2D> const mixed_derivs_xmax_ymax = std::nullopt) const;

	# if 1
    interpolation_domain_type const& interpolation_domain() const noexcept
    {
        return m_interpolation_domain;
    }
	# endif

    ddc::DiscreteDomain<bsplines_type, BatchTags...> spline_domain() const noexcept
    {
        return ddc::DiscreteDomain<bsplines_type, BatchTags...>(
                ddc::DiscreteElement<bsplines_type, BatchTags...>(0), // TODO : (0,0...)
                ddc::DiscreteVector<bsplines_type>(
                        ddc::discrete_space<bsplines_type>().size(),
                        ddc::discrete_space<BatchTags>().size()...));
    }
};


template <class SplineBuilder, class... BatchTags>
void SplineBuilderBatched<SplineBuilder, BatchTags...>::operator()(
        ddc::ChunkSpan<double, ddc::DiscreteDomain<bsplines_type, BatchTags...>> spline, // TODO: batch_dims_type
        ddc::ChunkSpan<double, ddc::DiscreteDomain<interpolation_mesh_type,BatchTags...>> vals,
        std::optional<CDSpan2D> const derivs_xmin,
        std::optional<CDSpan2D> const derivs_xmax) const
        // std::optional<CDSpan2D> const derivs_ymin,
        // std::optional<CDSpan2D> const derivs_ymax,
        // std::optional<CDSpan2D> const mixed_derivs_xmin_ymin,
        // std::optional<CDSpan2D> const mixed_derivs_xmax_ymin,
        // std::optional<CDSpan2D> const mixed_derivs_xmin_ymax,
        // std::optional<CDSpan2D> const mixed_derivs_xmax_ymax) const
{
    const std::size_t nbc_xmin = spline_builder.s_nbc_xmin;
    const std::size_t nbc_xmax = spline_builder.s_nbc_xmax;

    assert((BcXmin == BoundCond::HERMITE)
           != (!derivs_xmin.has_value() || derivs_xmin->extent(0) == 0));
    assert((BcXmax == BoundCond::HERMITE)
           != (!derivs_xmax.has_value() || derivs_xmax->extent(0) == 0));

    ddc::Chunk<double, ddc::DiscreteDomain<bsplines_type>> spline1(
            spline_builder.spline_domain());

    using IMesh = ddc::DiscreteElement<interpolation_mesh_type>;
    using BatchMesh = ddc::DiscreteElement<batch_mesh_type>;

    /******************************************************************
    *  Cycle over x1 position (or order of x1-derivative at boundary)
    *  and interpolate f along x2 direction.
    *******************************************************************/
	# if 0
    if constexpr (BcXmin2 == BoundCond::HERMITE) {
        assert((long int)(derivs_ymin->extent(0))
                       == spline_builder1.interpolation_domain().extents()
               && derivs_ymin->extent(1) == nbc_ymin);
        if constexpr (BcXmin1 == BoundCond::HERMITE) {
            assert(mixed_derivs_xmin_ymin->extent(0) == nbc_xmin
                   && mixed_derivs_xmin_ymin->extent(1) == nbc_ymin);
        }
        if constexpr (BcXmax1 == BoundCond::HERMITE) {
            assert(mixed_derivs_xmax_ymin->extent(0) == nbc_xmax
                   && mixed_derivs_xmax_ymin->extent(1) == nbc_ymin);
        }
        // In the boundary region we interpolate the derivatives
        for (int i = nbc_ymin; i > 0; --i) {
            const ddc::DiscreteElement<bsplines_type2> spl_idx(i - 1);

            // Get interpolated values
            ddc::Chunk<double, interpolation_domain_type1> vals1(
                    spline_builder1.interpolation_domain());
            ddc::for_each(spline_builder1.interpolation_domain(), [&](IMesh1 const j) {
                vals1(j) = (*derivs_ymin)(j.uid(), i - 1);
            });

            // Get interpolated derivatives
            std::vector<double> l_derivs(nbc_xmin);
            if constexpr (BcXmin1 == BoundCond::HERMITE) {
                for (std::size_t j(0); j < nbc_xmin; ++j)
                    l_derivs[j] = (*mixed_derivs_xmin_ymin)(j, i - 1);
            }
            const std::optional<CDSpan1D> deriv_l(
                    BcXmin1 == BoundCond::HERMITE
                            ? std::optional(CDSpan1D(l_derivs.data(), nbc_xmin))
                            : std::nullopt);

            std::vector<double> r_derivs(nbc_xmax);
            if constexpr (BcXmax1 == BoundCond::HERMITE) {
                for (std::size_t j(0); j < nbc_xmax; ++j)
                    r_derivs[j] = (*mixed_derivs_xmax_ymin)(j, i - 1);
            }
            const std::optional<CDSpan1D> deriv_r(
                    BcXmax1 == BoundCond::HERMITE
                            ? std::optional(CDSpan1D(r_derivs.data(), nbc_xmax))
                            : std::nullopt);

            // Interpolate derivatives
            spline_builder1(spline1, vals1, deriv_l, deriv_r);

            // Save result into 2d spline structure
            ddc::for_each(
                    ddc::get_domain<bsplines_type1>(spline),
                    [&](ddc::DiscreteElement<bsplines_type1> const j) {
                        spline(spl_idx, j) = spline1(j);
                    });
        }
    }
	# endif

    if (BcXmin == BoundCond::HERMITE) {
        // assert((long int)(derivs_xmin->extent(0))
        //               == spline_builder2.interpolation_domain().extents()
        //       && derivs_xmin->extent(1) == nbc_xmin);
    }
    if (BcXmax == BoundCond::HERMITE) {
        // assert((long int)(derivs_xmax->extent(0))
        //                == spline_builder2.interpolation_domain().extents()
        //        && derivs_xmax->extent(1) == nbc_xmax);
    }
	Kokkos::View<double**, Kokkos::DefaultHostExecutionSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> vals_flatten(vals.data_handle(), interpolation_domain().size(), m_batch_domain.size());
	Kokkos::View<double**, Kokkos::DefaultHostExecutionSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> spline_flatten(spline.data_handle(), ddc::discrete_space<bsplines_type>().nbasis(), m_batch_domain.size());
	Kokkos::deep_copy(spline_flatten, vals_flatten);
	# if 0
    
	ddc::for_each(m_batch_domain, [&](BatchMesh const i) {
        const std::size_t ii = i.uid();
        const ddc::DiscreteElement<bsplines_type> spl_idx(nbc_ymin + ii);
		
        // Get interpolated values
        ddc::Chunk<double, interpolation_domain_type1> vals1(
                spline_builder1.interpolation_domain());
        ddc::deepcopy(vals1, vals[i]);

        // Get interpolated derivatives
        const std::optional<CDSpan1D> deriv_l(
                BcXmin1 == BoundCond::HERMITE ? std::optional(
                        CDSpan1D(derivs_xmin->data_handle() + ii * nbc_xmin, nbc_xmin))
                                              : std::nullopt);
        const std::optional<CDSpan1D> deriv_r(
                BcXmax1 == BoundCond::HERMITE ? std::optional(
                        CDSpan1D(derivs_xmax->data_handle() + ii * nbc_xmax, nbc_xmax))
                                              : std::nullopt);

        // Interpolate values
        spline_builder1(spline1, vals1, deriv_l, deriv_r);

        // Save result into 2d spline structure
        ddc::for_each(
                ddc::get_domain<bsplines_type1>(spline),
                [&](ddc::DiscreteElement<bsplines_type1> const j) {
                    spline(spl_idx, j) = spline1(j);
                });
    });
	# endif

	spline_builder.matrix->solve_batch_inplace(spline_flatten);
    
	# if 0
    if constexpr (BcXmax2 == BoundCond::HERMITE) {
        assert((long int)(derivs_ymax->extent(0))
                       == spline_builder1.interpolation_domain().extents()
               && derivs_ymax->extent(1) == nbc_ymax);
        if constexpr (BcXmin2 == BoundCond::HERMITE) {
            assert(mixed_derivs_xmin_ymax->extent(0) == nbc_xmin
                   && mixed_derivs_xmin_ymax->extent(1) == nbc_ymax);
        }
        if constexpr (BcXmax2 == BoundCond::HERMITE) {
            assert(mixed_derivs_xmax_ymax->extent(0) == nbc_xmax
                   && mixed_derivs_xmax_ymax->extent(1) == nbc_ymax);
        }
        for (int i = nbc_ymax; i > 0; --i) {
            // In the boundary region we interpolate the derivatives
            const ddc::DiscreteElement<bsplines_type2> spl_idx(
                    i + ddc::discrete_space<bsplines_type2>().nbasis() - nbc_ymax - 1);

            // Get interpolated values
            ddc::Chunk<double, interpolation_domain_type1> vals1(
                    spline_builder1.interpolation_domain());
            ddc::for_each(spline_builder1.interpolation_domain(), [&](IMesh1 const j) {
                vals1(j) = (*derivs_ymax)(j.uid(), i - 1);
            });

            // Get interpolated derivatives
            std::vector<double> l_derivs(nbc_xmin);
            if constexpr (BcXmin1 == BoundCond::HERMITE) {
                for (std::size_t j(0); j < nbc_xmin; ++j)
                    l_derivs[j] = (*mixed_derivs_xmin_ymax)(j, i - 1);
            }
            const std::optional<CDSpan1D> deriv_l(
                    BcXmin1 == BoundCond::HERMITE
                            ? std::optional(CDSpan1D(l_derivs.data(), nbc_xmin))
                            : std::nullopt);

            std::vector<double> r_derivs(nbc_xmax);
            if constexpr (BcXmax1 == BoundCond::HERMITE) {
                for (std::size_t j(0); j < nbc_xmax; ++j)
                    r_derivs[j] = (*mixed_derivs_xmax_ymax)(j, i - 1);
            }
            const std::optional<CDSpan1D> deriv_r(
                    BcXmax1 == BoundCond::HERMITE
                            ? std::optional(CDSpan1D(r_derivs.data(), nbc_xmax))
                            : std::nullopt);

            // Interpolate derivatives
            spline_builder1(spline1, vals1, deriv_l, deriv_r);

            // Save result into 2d spline structure
            ddc::for_each(
                    ddc::get_domain<bsplines_type1>(spline),
                    [&](ddc::DiscreteElement<bsplines_type1> const j) {
                        spline(spl_idx, j) = spline1(j);
                    });
        }
    }
	# endif

	# if 0
    using IMeshV2 = ddc::DiscreteVector<bsplines_type2>;

    /******************************************************************
    *  Cycle over x1 position (or order of x1-derivative at boundary)
    *  and interpolate x2 cofficients along x2 direction.
    *******************************************************************/
    const ddc::DiscreteDomain<bsplines_type1> spline_basis_domain
            = ddc::DiscreteDomain<bsplines_type1>(
                    ddc::DiscreteElement<bsplines_type1>(0),
                    ddc::DiscreteVector<bsplines_type1>(
                            ddc::discrete_space<bsplines_type1>().nbasis()));

    ddc::for_each(spline_basis_domain, [&](ddc::DiscreteElement<bsplines_type1> const i) {
        const ddc::ChunkSpan<double, ddc::DiscreteDomain<bsplines_type2>> line_2 = spline[i];
        const ddc::DiscreteDomain<bsplines_type2> whole_line_dom
                = ddc::get_domain<bsplines_type2>(spline);
        // Get interpolated values
        ddc::ChunkSpan<double, ddc::DiscreteDomain<bsplines_type2>> const vals2(
                line_2[whole_line_dom.remove(IMeshV2(nbc_ymin), IMeshV2(nbc_ymax))]);
        // Get interpolated values acting as derivatives
        ddc::ChunkSpan<double, ddc::DiscreteDomain<bsplines_type2>> const l_derivs(
                line_2[whole_line_dom.take_first(IMeshV2(nbc_ymin))]);
        ddc::ChunkSpan<double, ddc::DiscreteDomain<bsplines_type2>> const r_derivs(
                line_2[whole_line_dom.take_last(IMeshV2(nbc_ymax))]);
        const std::optional<CDSpan1D> deriv_l(
                BcXmin2 == BoundCond::HERMITE ? std::optional(l_derivs.allocation_mdspan())
                                              : std::nullopt);
        const std::optional<CDSpan1D> deriv_r(
                BcXmax2 == BoundCond::HERMITE ? std::optional(r_derivs.allocation_mdspan())
                                              : std::nullopt);

        ddc::ChunkSpan<double const, interpolation_domain_type2>
                vals2_i(vals2.data_handle(), spline_builder2.interpolation_domain());

        // Interpolate coefficients
        spline_builder2(spline2, vals2_i, deriv_l, deriv_r);

        // Re-write result into 2d spline structure
        ddc::for_each(
                ddc::get_domain<bsplines_type2>(spline),
                [&](ddc::DiscreteElement<bsplines_type2> const j) { spline(i, j) = spline2(j); });
    });

    if (bsplines_type1::is_periodic()) {
        for (std::size_t i(0); i < bsplines_type1::degree(); ++i) {
            const ddc::DiscreteElement<bsplines_type1> i_start(i);
            const ddc::DiscreteElement<bsplines_type1> i_end(
                    ddc::discrete_space<bsplines_type1>().nbasis() + i);
            ddc::for_each(
                    ddc::get_domain<bsplines_type2>(spline),
                    [&](ddc::DiscreteElement<bsplines_type2> const j) {
                        spline(i_end, j) = spline(i_start, j);
                    });
        }
    }
	# endif
}
