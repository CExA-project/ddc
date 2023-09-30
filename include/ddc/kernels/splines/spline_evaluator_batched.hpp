#pragma once

#include <array>

#include <ddc/ddc.hpp>
#include "ddc/for_each.hpp"

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

      // using vals_domain_type = ddc::DiscreteDomain<IDimX...>;

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
		: spline_evaluator(left_bc,right_bc)
        , m_spline_domain(spline_domain) // Necessary ?
    {
    }

    SplineEvaluatorBatched(SplineEvaluatorBatched const& x) = default;

    SplineEvaluatorBatched(SplineEvaluatorBatched&& x) = default;

    ~SplineEvaluatorBatched() = default;

    SplineEvaluatorBatched& operator=(SplineEvaluatorBatched const& x) = default;

    SplineEvaluatorBatched& operator=(SplineEvaluatorBatched&& x) = default;



	  spline_domain_type const spline_domain() const noexcept
      {
          return m_spline_domain;
      }

	  ddc::DiscreteDomain<bsplines_type> const bsplines_domain() const noexcept // TODO : clarify name
      {
          return ddc::discrete_space<bsplines_type>().full_domain();
      }

      batch_domain_type const batch_domain() const noexcept
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

    double operator()(
            ddc::Coordinate<IDimX...> const& coord_eval,
            ddc::ChunkSpan<double const, spline_domain_type> const
                    spline_coef) const
    {
        // std::array<double, bsplines_type::degree() + 1> values;
        // DSpan1D const vals = as_span(values);

        return spline_evaluator.eval(coord_eval, spline_coef);
    }

    template <class Domain>
    void operator()(
            ddc::ChunkSpan<double, Domain> const spline_eval,
            ddc::ChunkSpan<ddc::Coordinate<IDimX...> const, Domain> const coords_eval,
            ddc::ChunkSpan<double const, spline_domain_type> const
                    spline_coef) const
    {
        // std::array<double, bsplines_type::degree() + 1> values;
        // DSpan1D const vals1 = as_span(values);

		// TODO: Consider optimizing
        ddc::for_each(batch_domain(), DDC_LAMBDA (typename batch_domain_type::discrete_element_type const j) {
		const auto spline_eval_1D = spline_eval[j];
		const auto coords_eval_1D = coords_eval[j];
		const auto spline_coef_1D = spline_coef[j];
		for (typename interpolation_domain_type::discrete_element_type i : coords_eval_1D.domain()) { // replace with Kokkos::Team loop ? And chunk if overload of scratch memory ?
            spline_eval_1D(i) = spline_evaluator.eval(coords_eval_1D(i), spline_coef_1D);
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

	#if 0
private:
    double eval(
            ddc::Coordinate<IDimX...> const& coord_eval,
            ddc::ChunkSpan<double const, spline_domain_type> const
                    spline_coef) const
     //       DSpan1D const vals1,
     //       DSpan1D const vals2) const
    {
        ddc::Coordinate<bsplines_type> coord_eval = ddc::select<bsplines_type>(coord_eval);
        if constexpr (bsplines_type::is_periodic()) {
            if (coord_eval < ddc::discrete_space<bsplines_type>().rmin()
                || coord_eval > ddc::discrete_space<bsplines_type>().rmax()) {
                coord_eval -= std::floor(
                                       (coord_eval - ddc::discrete_space<bsplines_type>().rmin())
                                       / ddc::discrete_space<bsplines_type>().length())
                               * ddc::discrete_space<bsplines_type>().length();
            }
        }
		/*
		 else {
            if (coord_eval1 < ddc::discrete_space<bsplines_type1>().rmin()) {
                return m_left_bc_1(coord_eval1, coord_eval2, spline_coef);
            }
            if (coord_eval1 > ddc::discrete_space<bsplines_type1>().rmax()) {
                return m_right_bc_1(coord_eval1, coord_eval2, spline_coef);
            }
        }
		*/
		/*
        if constexpr (bsplines_type2::is_periodic()) {
            if (coord_eval2 < ddc::discrete_space<bsplines_type2>().rmin()
                || coord_eval2 > ddc::discrete_space<bsplines_type2>().rmax()) {
                coord_eval2 -= std::floor(
                                       (coord_eval2 - ddc::discrete_space<bsplines_type2>().rmin())
                                       / ddc::discrete_space<bsplines_type2>().length())
                               * ddc::discrete_space<bsplines_type2>().length();
            }
        } else {
            if (coord_eval2 < ddc::discrete_space<bsplines_type2>().rmin()) {
                return m_left_bc_2(coord_eval1, coord_eval2, spline_coef);
            }
            if (coord_eval2 > ddc::discrete_space<bsplines_type2>().rmax()) {
                return m_right_bc_2(coord_eval1, coord_eval2, spline_coef);
            }
        }
		*/
        return eval_no_bc(
                coord_eval,
               // coord_eval2,
                spline_coef,
               // vals1,
               // vals2,
                eval_type(),
                eval_type());
    }

    template <class EvalType>
    double eval_no_bc(
            ddc::Coordinate<IDimX> const&... coord_eval,
            // ddc::Coordinate<Dim2> const& coord_eval2,
            ddc::ChunkSpan<double const, spline_domain_type>> const
                    spline_coef,
    //        DSpan1D const vals1,
    //        DSpan1D const vals2,
            EvalType const) const
            //EvalType2 const) const
    {
        static_assert(
                std::is_same_v<EvalType, eval_type> || std::is_same_v<EvalType, eval_deriv_type>);
        ddc::DiscreteElement<BSplinesType> jmin;

        std::array<double, bsplines_type::degree() + 1> values;
        DSpan1D const vals = as_span(values);


        if constexpr (std::is_same_v<EvalType, eval_type>) {
            jmin = ddc::discrete_space<bsplines_type>().eval_basis(vals, coord_eval);
        } else if constexpr (std::is_same_v<EvalType, eval_deriv_type>) {
            jmin = ddc::discrete_space<bsplines_type>().eval_deriv(vals, coord_eval);
        }

        double y = 0.0;
        for (std::size_t i = 0; i < bsplines_type::degree() + 1; ++i) {
            // for (std::size_t j = 0; j < bsplines_type2::degree() + 1; ++j) {
                y += spline_coef(jmin1 + i, jmin2 + j) * vals(i);
            //}
        }
        return y;
    }
	# endif
};
