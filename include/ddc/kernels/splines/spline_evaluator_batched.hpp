#pragma once

#include <array>

#include <ddc/ddc.hpp>

#include "ddc/for_each.hpp"

#include "Kokkos_Macros.hpp"
#include "spline_boundary_value.hpp"
#include "view.hpp"

template <class SplineEvaluator, class... IDimX>
class SplineEvaluatorBatched
{
private:
    // Tags to determine what to evaluate
    struct eval_type
    {
    };

    struct eval_deriv_type
    {
    };

    using tag_type = typename SplineEvaluator::tag_type;

public:
    using exec_space = typename SplineEvaluator::exec_space;

    using memory_space = typename SplineEvaluator::memory_space;

    using bsplines_type = typename SplineEvaluator::bsplines_type;

    using evaluator_type = SplineEvaluator;

    using interpolation_mesh_type = typename SplineEvaluator::mesh_type;

    using interpolation_domain_type = ddc::DiscreteDomain<interpolation_mesh_type>;

    using vals_domain_type = ddc::DiscreteDomain<IDimX...>;

    using bsplines_domain_type = ddc::DiscreteDomain<bsplines_type>;

    using batch_domain_type =
            typename ddc::detail::convert_type_seq_to_discrete_domain<ddc::type_seq_remove_t<
                    ddc::detail::TypeSeq<IDimX...>,
                    ddc::detail::TypeSeq<interpolation_mesh_type>>>;

    template <typename Tag>
    using spline_dim_type
            = std::conditional_t<std::is_same_v<Tag, interpolation_mesh_type>, bsplines_type, Tag>;

    using spline_domain_type =
            typename ddc::detail::convert_type_seq_to_discrete_domain<ddc::type_seq_replace_t<
                    ddc::detail::TypeSeq<IDimX...>,
                    ddc::detail::TypeSeq<interpolation_mesh_type>,
                    ddc::detail::TypeSeq<bsplines_type>>>;


private:
    SplineEvaluator spline_evaluator;
    const spline_domain_type m_spline_domain; // Necessary ?


public:
    SplineEvaluatorBatched() = delete;

    explicit SplineEvaluatorBatched(
            spline_domain_type const& spline_domain,
            SplineBoundaryValue<bsplines_type> const& left_bc,
            SplineBoundaryValue<bsplines_type> const& right_bc)
        : spline_evaluator(left_bc, right_bc)
        , m_spline_domain(spline_domain) // Necessary ?
    {
    }

    SplineEvaluatorBatched(SplineEvaluatorBatched const& x) = default;

    SplineEvaluatorBatched(SplineEvaluatorBatched&& x) = default;

    ~SplineEvaluatorBatched() = default;

    SplineEvaluatorBatched& operator=(SplineEvaluatorBatched const& x) = default;

    SplineEvaluatorBatched& operator=(SplineEvaluatorBatched&& x) = default;



    KOKKOS_INLINE_FUNCTION spline_domain_type const spline_domain() const noexcept
    {
        return m_spline_domain;
    }

    KOKKOS_INLINE_FUNCTION bsplines_domain_type const bsplines_domain()
            const noexcept // TODO : clarify name
    {
        return ddc::discrete_space<bsplines_type>().full_domain();
    }

    KOKKOS_INLINE_FUNCTION batch_domain_type const batch_domain() const noexcept
    {
        return ddc::remove_dims_of(spline_domain(), bsplines_domain());
    }

    /*
      vals_domain_type const vals_domain(interpolation_domain_type interpolation_domain) const noexcept
      {
          return ddc::replace_dim_of<
                  bsplines_type,
				  interpolation_mesh_type>(spline_domain(), interpolation_domain);
      }

      spline_tr_domain_type const spline_tr_domain() const noexcept
      {
          return spline_tr_domain_type(bsplines_domain(), batch_domain());
      }
      */
    template <class Layout>
    double operator()(
            ddc::Coordinate<IDimX...> const& coord_eval,
            ddc::ChunkSpan<double const, spline_domain_type, Layout, memory_space> const
                    spline_coef) const
    {
        // std::array<double, bsplines_type::degree() + 1> values;
        // DSpan1D const vals = as_span(values);

        return eval(coord_eval, spline_coef);
    }

    template <class Layout1, class Layout2, class Layout3, class... CoordsDims>
    void operator()(
            ddc::ChunkSpan<double, vals_domain_type, Layout1, memory_space> const spline_eval,
            ddc::ChunkSpan<
                    ddc::Coordinate<CoordsDims...> const,
                    vals_domain_type,
                    Layout2,
                    memory_space> const coords_eval,
            ddc::ChunkSpan<double const, spline_domain_type, Layout3, memory_space> const
                    spline_coef) const
    {
        // std::array<double, bsplines_type::degree() + 1> values;
        // DSpan1D const vals1 = as_span(values);

		interpolation_domain_type interpolation_domain = ddc::select<interpolation_mesh_type>(spline_eval.domain());
		std::size_t interpolation_size = interpolation_domain.size();
        // TODO: Consider optimizing
		auto spline_coef_0 = spline_coef[batch_domain().front()];
        ddc::for_each(
                ddc::policies::policy(exec_space()),
                batch_domain(),
                KOKKOS_CLASS_LAMBDA (typename batch_domain_type::discrete_element_type const j) {
                    // const auto spline_eval_1D = spline_eval[j];
                    // const auto coords_eval_1D = coords_eval[j];
                    // const auto spline_coef_1D = spline_coef[j];
                    for (int i=0; i<interpolation_size; i++
                         ) { // replace with Kokkos::Team loop ? And chunk if overload of scratch memory ?
					// printf("interpolation_domain_size%i", interpolation_size);
                    spline_eval(typename interpolation_domain_type::discrete_element_type(i),j) = eval(coords_eval(typename interpolation_domain_type::discrete_element_type(i),j), spline_coef_0);
                    // spline_eval_1D(interpolation_domain(typename interpolation_domain_type::discrete_element_type(i))) = eval(coords_eval_1D(interpolation_domain(typename interpolation_domain_type::discrete_element_type(i))), spline_coef_1D);
                    }
                });
    }

#if 0
    double deriv_dim_1(
            ddc::Coordinate<IDimX...> const& coord_eval,
            ddc::ChunkSpan<double const, spline_domain_type> const
                    spline_coef) const
    {
		/*
        std::array<double, bsplines_type1::degree() + 1> values1;
        DSpan1D const vals1 = as_span(values1);
        std::array<double, bsplines_type2::degree() + 1> values2;
        DSpan1D const vals2 = as_span(values2);
		*/

        return eval_no_bc(
                ddc::select<Dim1>(coord_eval),
                ddc::select<Dim2>(coord_eval),
                spline_coef,
        //        vals1,
        //        vals2,
                eval_deriv_type(),
                eval_type());
    }

    double deriv_dim_2(
            ddc::Coordinate<IDimX...> const& coord_eval,
            ddc::ChunkSpan<double const, spline_domain_type> const
                    spline_coef) const
    {
		/*
        std::array<double, bsplines_type1::degree() + 1> values1;
        DSpan1D const vals1 = as_span(values1);
        std::array<double, bsplines_type2::degree() + 1> values2;
        DSpan1D const vals2 = as_span(values2);
		*/

        return eval_no_bc(
                ddc::select<IDimX>(coord_eval)...,
                spline_coef,
        //        vals1,
        //        vals2,
                eval_type(),
                eval_deriv_type());
    }

    double deriv_1_and_2(
            ddc::Coordinate<IDimX...> const& coord_eval,
            ddc::ChunkSpan<double const, ddc::DiscreteDomain<BSplinesType1, BSplinesType2>> const
                    spline_coef) const
    {
		/*
        std::array<double, bsplines_type1::degree() + 1> values1;
        DSpan1D const vals1 = as_span(values1);
        std::array<double, bsplines_type2::degree() + 1> values2;
        DSpan1D const vals2 = as_span(values2);
		*/

        return eval_no_bc(
                ddc::select<IDimX>(coord_eval)...,
                spline_coef,
                vals1,
                vals2,
                eval_deriv_type(),
                eval_deriv_type());
    }

    template <class Domain>
    void deriv_dim_1(
            ddc::ChunkSpan<double, Domain> const spline_eval,
            ddc::ChunkSpan<ddc::Coordinate<IDimX...> const, Domain> const coords_eval,
            ddc::ChunkSpan<double const, spline_domain_type> const
                    spline_coef) const
    {
		/*
        std::array<double, bsplines_type1::degree() + 1> values1;
        DSpan1D const vals1 = as_span(values1);
        std::array<double, bsplines_type2::degree() + 1> values2;
        DSpan1D const vals2 = as_span(values2);
		*/

        ddc::for_each(coords_eval.domain(), [=](auto i) {
            spline_eval(i) = eval_no_bc(
                    ddc::select<IDimX>(coords_eval(i))...,
                    spline_coef,
                    vals1,
                    vals2,
                    eval_deriv_type(),
                    eval_type());
        });
    }

    template <class Domain>
    void deriv_dim_2(
            ddc::ChunkSpan<double, Domain> const spline_eval,
            ddc::ChunkSpan<ddc::Coordinate<IDimX...> const, Domain> const coords_eval,
            ddc::ChunkSpan<double const, spline_domain_type> const
                    spline_coef) const
    {
		/*
        std::array<double, bsplines_type1::degree() + 1> values1;
        DSpan1D const vals1 = as_span(values1);
        std::array<double, bsplines_type2::degree() + 1> values2;
        DSpan1D const vals2 = as_span(values2);
		*/

        ddc::for_each(coords_eval.domain(), [=](auto i) {
            spline_eval(i) = eval_no_bc(
                    ddc::select<DimX>(coords_eval(i))...,
                    spline_coef,
         //           vals1,
         //           vals2,
                    eval_type(),
                    eval_deriv_type());
        });
    }

    template <class Domain>
    void deriv_dim_1_and_2(
            ddc::ChunkSpan<double, Domain> const spline_eval,
            ddc::ChunkSpan<ddc::Coordinate<IDimX...> const, Domain> const coords_eval,
            ddc::ChunkSpan<double const, spline_domain_type> const
                    spline_coef) const
    {
		/*
        std::array<double, bsplines_type1::degree() + 1> values1;
        DSpan1D const vals1 = as_span(values1);
        std::array<double, bsplines_type2::degree() + 1> values2;
        DSpan1D const vals2 = as_span(values2);
		*/

        ddc::for_each(coords_eval.domain(), [=](auto i) {
            spline_eval(i) = eval_no_bc(
                    ddc::select<IDimX>(coords_eval(i))...,
                    spline_coef,
                    //vals1,
                    //vals2,
                    eval_deriv_type(),
                    eval_deriv_type());
        });
    }

    double integrate(
            ddc::ChunkSpan<double const, ddc::DiscreteDomain<BSplinesType1, BSplinesType2>> const
                    spline_coef) const
    {
        ddc::Chunk<double, ddc::DiscreteDomain<BSplinesType1>> values1(
                ddc::DiscreteDomain<BSplinesType1>(spline_coef.domain()));
        DSpan1D vals1 = values1.allocation_mdspan();
        ddc::Chunk<double, ddc::DiscreteDomain<BSplinesType2>> values2(
                ddc::DiscreteDomain<BSplinesType2>(spline_coef.domain()));
        DSpan1D vals2 = values2.allocation_mdspan();

        ddc::discrete_space<bsplines_type1>().integrals(values1.span_view());
        ddc::discrete_space<bsplines_type2>().integrals(values2.span_view());

        return ddc::transform_reduce(
                spline_coef.domain(),
                0.0,
                ddc::reducer::sum<double>(),
                [&](ddc::DiscreteElement<BSplinesType1, BSplinesType2> const i) {
                    return spline_coef(i) * values1(ddc::select<BSplinesType1>(i))
                           * values2(ddc::select<BSplinesType2>(i));
                });
    }
#endif

#if 1
private:
    template <class Layout, class... CoordsDims>
    KOKKOS_INLINE_FUNCTION double eval(
            ddc::Coordinate<CoordsDims...> const& coord_eval,
            ddc::ChunkSpan<double const, bsplines_domain_type, Layout, memory_space> const
                    spline_coef) const
    {
			// printf("\n OLEEEEE");
        ddc::Coordinate<typename interpolation_mesh_type::continuous_dimension_type>
                coord_eval_interpolation
                = ddc::select<typename interpolation_mesh_type::continuous_dimension_type>(
                        coord_eval);
        if constexpr (bsplines_type::is_periodic()) {
            if (coord_eval_interpolation < ddc::discrete_space<bsplines_type>().rmin()
                || coord_eval_interpolation > ddc::discrete_space<bsplines_type>().rmax()) {
                coord_eval_interpolation -= Kokkos::floor(
                                                    (coord_eval_interpolation
                                                     - ddc::discrete_space<bsplines_type>().rmin())
                                                    / ddc::discrete_space<bsplines_type>().length())
                                            * ddc::discrete_space<bsplines_type>().length();
            }
        }
		// printf("\n iYAAASS");
        /*
		 else {
            if (coord_eval_interpolation1 < ddc::discrete_space<bsplines_type1>().rmin()) {
                return m_left_bc_1(coord_eval_interpolation1, coord_eval_interpolation2, spline_coef);
            }
            if (coord_eval_interpolation1 > ddc::discrete_space<bsplines_type1>().rmax()) {
                return m_right_bc_1(coord_eval_interpolation1, coord_eval_interpolation2, spline_coef);
            }
        }
		*/
        /*
        if constexpr (bsplines_type2::is_periodic()) {
            if (coord_eval_interpolation2 < ddc::discrete_space<bsplines_type2>().rmin()
                || coord_eval_interpolation2 > ddc::discrete_space<bsplines_type2>().rmax()) {
                coord_eval_interpolation2 -= std::floor(
                                       (coord_eval_interpolation2 - ddc::discrete_space<bsplines_type2>().rmin())
                                       / ddc::discrete_space<bsplines_type2>().length())
                               * ddc::discrete_space<bsplines_type2>().length();
            }
        } else {
            if (coord_eval_interpolation2 < ddc::discrete_space<bsplines_type2>().rmin()) {
                return m_left_bc_2(coord_eval_interpolation1, coord_eval_interpolation2, spline_coef);
            }
            if (coord_eval_interpolation2 > ddc::discrete_space<bsplines_type2>().rmax()) {
                return m_right_bc_2(coord_eval_interpolation1, coord_eval_interpolation2, spline_coef);
            }
        }
		*/
        return eval_no_bc<eval_type>(
                coord_eval,
                // coord_eval2,
                spline_coef
                //    eval_type(),
                // eval_type()
				);
    }

    template <class EvalType, class Layout, class... CoordsDims>
    KOKKOS_INLINE_FUNCTION double eval_no_bc(
            ddc::Coordinate<CoordsDims...> const& coord_eval,
            // ddc::Coordinate<Dim2> const& coord_eval2,
            ddc::ChunkSpan<double const, bsplines_domain_type, Layout, memory_space> const
                    spline_coef) const
//            EvalType const) const
    //EvalType2 const) const
    {
        static_assert(
                std::is_same_v<EvalType, eval_type> || std::is_same_v<EvalType, eval_deriv_type>);
        ddc::DiscreteElement<bsplines_type> jmin;
        std::array<double, bsplines_type::degree() + 1> vals;
        ddc::Coordinate<typename interpolation_mesh_type::continuous_dimension_type>
                coord_eval_interpolation
                = ddc::select<typename interpolation_mesh_type::continuous_dimension_type>(
                        coord_eval);
		// printf("%f", coord_eval_interpolation);
        if constexpr (std::is_same_v<EvalType, eval_type>) {
            jmin = ddc::discrete_space<bsplines_type>().eval_basis(vals, coord_eval_interpolation);
			// printf(" %f\n", vals[0]);
		// printf("lululululul");
        } else if constexpr (std::is_same_v<EvalType, eval_deriv_type>) {
            jmin = ddc::discrete_space<bsplines_type>().eval_deriv(vals, coord_eval_interpolation);
        }
        double y = 0.0;
        for (std::size_t i = 0; i < bsplines_type::degree() + 1; ++i) {
			// printf("%i",i);
			// printf("%f\n",spline_coef(ddc::DiscreteElement<bsplines_type>(jmin + i)));
            y += spline_coef(ddc::DiscreteElement<bsplines_type>(jmin + i)) * vals[i];
        }
        return y;
    }
#endif
};
