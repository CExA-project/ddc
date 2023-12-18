#pragma once
#include "spline_builder.hpp"
#include "spline_builder_batched.hpp"


namespace ddc {
template <class ExecSpace, class MemorySpace, class BSpline1, class BSpline2, class IDimI1, class IDimI2, ddc::BoundCond BcMin1, ddc::BoundCond BcMax1, ddc::BoundCond BcMin2, ddc::BoundCond BcMax2, class... IDimX>
class SplineBuilder2DBatched
{
public:
    using exec_space = ExecSpace;
    
	using memory_space = MemorySpace; //TODO: assert same for 1 and 2

	using builder_type1 = ddc::SplineBuilderBatched<typename ddc::SplineBuilder<ExecSpace,MemorySpace,BSpline1,IDimI1,BcMin1,BcMax1>,IDimX...>;
	using builder_type2 = typename ddc::SplineBuilderBatched<typename ddc::SplineBuilder<ExecSpace,MemorySpace,BSpline2,IDimI2,BcMin2,BcMax2>, typename std::conditional_t<std::is_same_v<IDimX,IDimI1>,BSpline1,IDimX>...>;

private:
    using tag_type1 = typename builder_type1::bsplines_type::tag_type;
    using tag_type2 = typename builder_type2::bsplines_type::tag_type;

public:
    using bsplines_type1 = typename builder_type1::bsplines_type;
    using bsplines_type2 = typename builder_type2::bsplines_type;

    using deriv_type1 = typename builder_type1::deriv_type;
    using deriv_type2 = typename builder_type2::deriv_type;
 
    using interpolation_mesh_type1 = typename builder_type1::interpolation_mesh_type;
    using interpolation_mesh_type2 = typename builder_type2::interpolation_mesh_type;

    using interpolation_domain_type1 = typename builder_type1::interpolation_mesh_type;
    using interpolation_domain_type2 = typename builder_type2::interpolation_mesh_type;
    using interpolation_domain_type
            = ddc::DiscreteDomain<interpolation_mesh_type1, interpolation_mesh_type2>;

    using vals_domain_type = 
            ddc::DiscreteDomain<IDimX...>; // TODO: assert same for 1 and 2

	using batch_domain_type =
            typename ddc::detail::convert_type_seq_to_discrete_domain<ddc::type_seq_remove_t<
                    ddc::detail::TypeSeq<IDimX...>,
                    ddc::detail::TypeSeq<interpolation_mesh_type1, interpolation_mesh_type2>>>;

	using spline_domain_type =
            typename ddc::detail::convert_type_seq_to_discrete_domain<ddc::type_seq_replace_t<
                    ddc::detail::TypeSeq<IDimX...>,
                    ddc::detail::TypeSeq<interpolation_mesh_type1, interpolation_mesh_type2>,
                    ddc::detail::TypeSeq<bsplines_type1, bsplines_type2>>>;

	using derivs_domain_type1 = typename builder_type1::derivs_domain_type;
	using derivs_domain_type2 = typename builder_type2::derivs_domain_type;
	using mixed_derivs_domain_type = typename builder_type2::derivs_domain_type;
	using derivs_domain_type =
            typename ddc::detail::convert_type_seq_to_discrete_domain<ddc::type_seq_replace_t<
                    ddc::detail::TypeSeq<IDimX...>,
                    ddc::detail::TypeSeq<interpolation_mesh_type1, interpolation_mesh_type2>,
                    ddc::detail::TypeSeq<deriv_type1, deriv_type2>>>;

private:
    builder_type1 m_spline_builder1;
    builder_type2 m_spline_builder2;

public:
    SplineBuilder2DBatched(
			vals_domain_type const& vals_domain,
            std::optional<int> cols_per_par_chunk = std::nullopt,
            std::optional<int> par_chunks_per_seq_chunk = std::nullopt,
            std::optional<unsigned int> preconditionner_max_block_size = std::nullopt
			)
	  : m_spline_builder1(vals_domain, cols_per_par_chunk, par_chunks_per_seq_chunk, preconditionner_max_block_size),
	    m_spline_builder2(m_spline_builder1.spline_domain(), cols_per_par_chunk, par_chunks_per_seq_chunk, preconditionner_max_block_size)
    {
	  //TODO: static_asserts
    }

    SplineBuilder2DBatched(SplineBuilder2DBatched const& x) = delete;

    SplineBuilder2DBatched(SplineBuilder2DBatched&& x) = default;

    ~SplineBuilder2DBatched() = default;

    SplineBuilder2DBatched& operator=(SplineBuilder2DBatched const& x) = delete;

    SplineBuilder2DBatched& operator=(SplineBuilder2DBatched&& x) = default;

	vals_domain_type const vals_domain() const noexcept
    {
        return m_spline_builder1.vals_domain();
    }

    interpolation_domain_type const interpolation_domain() const noexcept
    {
        return ddc::DiscreteDomain<interpolation_domain_type1,interpolation_domain_type2>(m_spline_builder1.interpolation_domain(), m_spline_builder2.interpolation_domain());
    }

    batch_domain_type const batch_domain() const noexcept
    {
        return ddc::remove_dims_of(vals_domain(), interpolation_domain());
    }

	ddc::DiscreteDomain<bsplines_type1, bsplines_type2> const bsplines_domain() const noexcept // TODO : clarify name
    {
        return ddc::DiscreteDomain<bsplines_type1,bsplines_type2>(ddc::discrete_space<bsplines_type1>().full_domain(), ddc::discrete_space<bsplines_type2>().full_domain());
    }

	spline_domain_type const spline_domain() const noexcept
    {
        return ddc::replace_dim_of<
                interpolation_mesh_type1,
                bsplines_type1>(
				ddc::replace_dim_of<
                interpolation_mesh_type2,
                bsplines_type2>(vals_domain(), bsplines_domain()), bsplines_domain());
    }

    template <class Layout>
    void operator()(
            ddc::ChunkSpan<
                    double,
					spline_domain_type,
                    Layout,
                    memory_space> spline,
            ddc::ChunkSpan<double, vals_domain_type, Layout, memory_space> vals,
            std::optional<ddc::ChunkSpan<double, derivs_domain_type1, Layout, memory_space>> const derivs_min1 = std::nullopt,
            std::optional<ddc::ChunkSpan<double, derivs_domain_type1, Layout, memory_space>> const derivs_max1 = std::nullopt,
            std::optional<ddc::ChunkSpan<double, derivs_domain_type2, Layout, memory_space>> const derivs_min2 = std::nullopt,
            std::optional<ddc::ChunkSpan<double, derivs_domain_type2, Layout, memory_space>> const derivs_max2 = std::nullopt,
            std::optional<ddc::ChunkSpan<double, mixed_derivs_domain_type, Layout, memory_space>> const mixed_derivs_min1_min2 = std::nullopt,
            std::optional<ddc::ChunkSpan<double, mixed_derivs_domain_type, Layout, memory_space>> const mixed_derivs_max1_min2 = std::nullopt,
            std::optional<ddc::ChunkSpan<double, mixed_derivs_domain_type, Layout, memory_space>> const mixed_derivs_min1_max2 = std::nullopt,
            std::optional<ddc::ChunkSpan<double, mixed_derivs_domain_type, Layout, memory_space>> const mixed_derivs_max1_max2 = std::nullopt) const;
};


template <class ExecSpace, class MemorySpace, class BSpline1, class BSpline2, class IDimI1, class IDimI2, ddc::BoundCond BcMin1, ddc::BoundCond BcMax1, ddc::BoundCond BcMin2, ddc::BoundCond BcMax2, class... IDimX>
template <class Layout>
void SplineBuilder2DBatched<ExecSpace, MemorySpace, BSpline1, BSpline2, IDimI1, IDimI2, BcMin1, BcMax1, BcMin2, BcMax2, IDimX...>::operator()(
            ddc::ChunkSpan<
                    double,
					spline_domain_type,
                    Layout,
                    memory_space> spline,
            ddc::ChunkSpan<double, vals_domain_type, Layout, memory_space> vals,
            std::optional<ddc::ChunkSpan<double, derivs_domain_type1, Layout, memory_space>> const derivs_min1,
            std::optional<ddc::ChunkSpan<double, derivs_domain_type1, Layout, memory_space>> const derivs_max1,
            std::optional<ddc::ChunkSpan<double, derivs_domain_type2, Layout, memory_space>> const derivs_min2,
            std::optional<ddc::ChunkSpan<double, derivs_domain_type2, Layout, memory_space>> const derivs_max2,
            std::optional<ddc::ChunkSpan<double, mixed_derivs_domain_type, Layout, memory_space>> const mixed_derivs_min1_min2,
            std::optional<ddc::ChunkSpan<double, mixed_derivs_domain_type, Layout, memory_space>> const mixed_derivs_max1_min2,
            std::optional<ddc::ChunkSpan<double, mixed_derivs_domain_type, Layout, memory_space>> const mixed_derivs_min1_max2,
            std::optional<ddc::ChunkSpan<double, mixed_derivs_domain_type, Layout, memory_space>> const mixed_derivs_max1_max2) const
{
#if 0
    const std::size_t nbc_xmin = spline_builder1.s_nbc_xmin;
    const std::size_t nbc_xmax = spline_builder1.s_nbc_xmax;
    const std::size_t nbc_ymin = spline_builder2.s_nbc_xmin;
    const std::size_t nbc_ymax = spline_builder2.s_nbc_xmax;

    assert((BcXmin1 == ddc::BoundCond::HERMITE)
           != (!derivs_xmin.has_value() || derivs_xmin->extent(0) == 0));
    assert((BcXmax1 == ddc::BoundCond::HERMITE)
           != (!derivs_xmax.has_value() || derivs_xmax->extent(0) == 0));
    assert((BcXmin2 == ddc::BoundCond::HERMITE)
           != (!derivs_ymin.has_value() || derivs_ymin->extent(0) == 0));
    assert((BcXmax2 == ddc::BoundCond::HERMITE)
           != (!derivs_ymax.has_value() || derivs_ymax->extent(0) == 0));
    assert((BcXmin1 == ddc::BoundCond::HERMITE && BcXmin2 == ddc::BoundCond::HERMITE)
           != (!mixed_derivs_xmin_ymin.has_value()
               || mixed_derivs_xmin_ymin->extent(0) != nbc_xmin));
    assert((BcXmax1 == ddc::BoundCond::HERMITE && BcXmin2 == ddc::BoundCond::HERMITE)
           != (!mixed_derivs_xmax_ymin.has_value()
               || mixed_derivs_xmax_ymin->extent(0) != nbc_xmax));
    assert((BcXmin2 == ddc::BoundCond::HERMITE && BcXmax2 == ddc::BoundCond::HERMITE)
           != (!mixed_derivs_xmin_ymax.has_value()
               || mixed_derivs_xmin_ymax->extent(0) != nbc_xmin));
    assert((BcXmax2 == ddc::BoundCond::HERMITE && BcXmax2 == ddc::BoundCond::HERMITE)
           != (!mixed_derivs_xmax_ymax.has_value()
               || mixed_derivs_xmax_ymax->extent(0) != nbc_xmax));

    using IMesh1 = ddc::DiscreteElement<interpolation_mesh_type1>;
    using IMesh2 = ddc::DiscreteElement<interpolation_mesh_type2>;
    ddc::Chunk<double, ddc::DiscreteDomain<bsplines_type1>> spline1_alloc(
            spline_builder1.spline_domain());
    ddc::ChunkSpan spline1 = spline1_alloc.span_view();
    ddc::Chunk<double, ddc::DiscreteDomain<bsplines_type2>> spline2_alloc(
            spline_builder2.spline_domain());
    ddc::ChunkSpan spline2 = spline2_alloc.span_view();
    ddc::Chunk<double, interpolation_domain_type1> vals1_alloc(
            spline_builder1.interpolation_domain());
    ddc::ChunkSpan vals1 = vals1_alloc.span_view();

    /******************************************************************
    *  Cycle over x1 position (or order of x1-derivative at boundary)
    *  and interpolate f along x2 direction.
    *******************************************************************/
    if constexpr (BcXmin2 == ddc::BoundCond::HERMITE) {
        assert((long int)(derivs_ymin->extent(0))
                       == spline_builder1.interpolation_domain().extents()
               && derivs_ymin->extent(1) == nbc_ymin);
        if constexpr (BcXmin1 == ddc::BoundCond::HERMITE) {
            assert(mixed_derivs_xmin_ymin->extent(0) == nbc_xmin
                   && mixed_derivs_xmin_ymin->extent(1) == nbc_ymin);
        }
        if constexpr (BcXmax1 == ddc::BoundCond::HERMITE) {
            assert(mixed_derivs_xmax_ymin->extent(0) == nbc_xmax
                   && mixed_derivs_xmax_ymin->extent(1) == nbc_ymin);
        }
        // In the boundary region we interpolate the derivatives
        for (int i = nbc_ymin; i > 0; --i) {
            const ddc::DiscreteElement<bsplines_type2> spl_idx(i - 1);

            // Get interpolated values
            ddc::for_each(spline_builder1.interpolation_domain(), [&](IMesh1 const j) {
                vals1(j) = (*derivs_ymin)(j.uid(), i - 1);
            });

            // Get interpolated derivatives
            std::vector<double> l_derivs(nbc_xmin);
            if constexpr (BcXmin1 == ddc::BoundCond::HERMITE) {
                for (std::size_t j(0); j < nbc_xmin; ++j)
                    l_derivs[j] = (*mixed_derivs_xmin_ymin)(j, i - 1);
            }
            const std::optional<ddc::CDSpan1D> deriv_l(
                    BcXmin1 == ddc::BoundCond::HERMITE
                            ? std::optional(ddc::CDSpan1D(l_derivs.data(), nbc_xmin))
                            : std::nullopt);

            std::vector<double> r_derivs(nbc_xmax);
            if constexpr (BcXmax1 == ddc::BoundCond::HERMITE) {
                for (std::size_t j(0); j < nbc_xmax; ++j)
                    r_derivs[j] = (*mixed_derivs_xmax_ymin)(j, i - 1);
            }
            const std::optional<ddc::CDSpan1D> deriv_r(
                    BcXmax1 == ddc::BoundCond::HERMITE
                            ? std::optional(ddc::CDSpan1D(r_derivs.data(), nbc_xmax))
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

    if (BcXmin1 == ddc::BoundCond::HERMITE) {
        assert((long int)(derivs_xmin->extent(0))
                       == spline_builder2.interpolation_domain().extents()
               && derivs_xmin->extent(1) == nbc_xmin);
    }
    if (BcXmax1 == ddc::BoundCond::HERMITE) {
        assert((long int)(derivs_xmax->extent(0))
                       == spline_builder2.interpolation_domain().extents()
               && derivs_xmax->extent(1) == nbc_xmax);
    }
    ddc::for_each(
            // ddc::policies::policy(typename SplineBuilder1::exec_space()),
            spline_builder2.interpolation_domain(),
            [&](IMesh2 const i) {
                const std::size_t ii = i.uid();
                const ddc::DiscreteElement<bsplines_type2> spl_idx(nbc_ymin + ii);

                // Get interpolated values
                // ddc::deepcopy(vals1, vals[i]);
                for (auto j : spline_builder1.interpolation_domain()) {
                    vals1(j) = vals(i, j);
                }

                // Get interpolated derivatives
                const std::optional<ddc::CDSpan1D> deriv_l(
                        BcXmin1 == ddc::BoundCond::HERMITE ? std::optional(
                                ddc::CDSpan1D(derivs_xmin->data_handle() + ii * nbc_xmin, nbc_xmin))
                                                           : std::nullopt);
                const std::optional<ddc::CDSpan1D> deriv_r(
                        BcXmax1 == ddc::BoundCond::HERMITE ? std::optional(
                                ddc::CDSpan1D(derivs_xmax->data_handle() + ii * nbc_xmax, nbc_xmax))
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

    if constexpr (BcXmax2 == ddc::BoundCond::HERMITE) {
        assert((long int)(derivs_ymax->extent(0))
                       == spline_builder1.interpolation_domain().extents()
               && derivs_ymax->extent(1) == nbc_ymax);
        if constexpr (BcXmin2 == ddc::BoundCond::HERMITE) {
            assert(mixed_derivs_xmin_ymax->extent(0) == nbc_xmin
                   && mixed_derivs_xmin_ymax->extent(1) == nbc_ymax);
        }
        if constexpr (BcXmax2 == ddc::BoundCond::HERMITE) {
            assert(mixed_derivs_xmax_ymax->extent(0) == nbc_xmax
                   && mixed_derivs_xmax_ymax->extent(1) == nbc_ymax);
        }
        for (int i = nbc_ymax; i > 0; --i) {
            // In the boundary region we interpolate the derivatives
            const ddc::DiscreteElement<bsplines_type2> spl_idx(
                    i + ddc::discrete_space<bsplines_type2>().nbasis() - nbc_ymax - 1);

            // Get interpolated values
            ddc::for_each(spline_builder1.interpolation_domain(), [&](IMesh1 const j) {
                vals1(j) = (*derivs_ymax)(j.uid(), i - 1);
            });

            // Get interpolated derivatives
            std::vector<double> l_derivs(nbc_xmin);
            if constexpr (BcXmin1 == ddc::BoundCond::HERMITE) {
                for (std::size_t j(0); j < nbc_xmin; ++j)
                    l_derivs[j] = (*mixed_derivs_xmin_ymax)(j, i - 1);
            }
            const std::optional<ddc::CDSpan1D> deriv_l(
                    BcXmin1 == ddc::BoundCond::HERMITE
                            ? std::optional(ddc::CDSpan1D(l_derivs.data(), nbc_xmin))
                            : std::nullopt);

            std::vector<double> r_derivs(nbc_xmax);
            if constexpr (BcXmax1 == ddc::BoundCond::HERMITE) {
                for (std::size_t j(0); j < nbc_xmax; ++j)
                    r_derivs[j] = (*mixed_derivs_xmax_ymax)(j, i - 1);
            }
            const std::optional<ddc::CDSpan1D> deriv_r(
                    BcXmax1 == ddc::BoundCond::HERMITE
                            ? std::optional(ddc::CDSpan1D(r_derivs.data(), nbc_xmax))
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

    ddc::for_each(
            // ddc::policies::policy(typename SplineBuilder2::exec_space()),
            spline_basis_domain,
            [&](ddc::DiscreteElement<bsplines_type1> const i) {
                const ddc::ChunkSpan<double, ddc::DiscreteDomain<bsplines_type2>> line_2
                        = spline[i];
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
                const std::optional<ddc::CDSpan1D> deriv_l(
                        BcXmin2 == ddc::BoundCond::HERMITE
                                ? std::optional(l_derivs.allocation_mdspan())
                                : std::nullopt);
                const std::optional<ddc::CDSpan1D> deriv_r(
                        BcXmax2 == ddc::BoundCond::HERMITE
                                ? std::optional(r_derivs.allocation_mdspan())
                                : std::nullopt);
                // TODO: const double ?
                ddc::ChunkSpan<double, interpolation_domain_type2>
                        vals2_i(vals2.data_handle(), spline_builder2.interpolation_domain());

                // Interpolate coefficients
                spline_builder2(spline2, vals2_i, deriv_l, deriv_r);

                // Re-write result into 2d spline structure
                ddc::for_each(
                        ddc::get_domain<bsplines_type2>(spline),
                        [&](ddc::DiscreteElement<bsplines_type2> const j) {
                            spline(i, j) = spline2(j);
                        });
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
#endif
}
} // namespace ddc
