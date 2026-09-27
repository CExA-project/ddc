// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

module;

#include <ddc/kernels/splines.hpp>

export module ddc.splines;

export namespace ddc {

using ::ddc::ConstantExtrapolationRule;
using ::ddc::Deriv;
using ::ddc::GrevilleInterpolationPoints;
using ::ddc::integrals;
using ::ddc::is_evaluator_admissible;
using ::ddc::is_evaluator_admissible_v;
using ::ddc::is_non_uniform_bsplines;
using ::ddc::is_non_uniform_bsplines_v;
using ::ddc::is_spline_builder;
using ::ddc::is_spline_builder2d;
using ::ddc::is_spline_builder2d_v;
using ::ddc::is_spline_builder_v;
using ::ddc::is_spline_evaluator;
using ::ddc::is_spline_evaluator2d;
using ::ddc::is_spline_evaluator2d_v;
using ::ddc::is_spline_evaluator_v;
using ::ddc::is_uniform_bsplines;
using ::ddc::is_uniform_bsplines_v;
using ::ddc::knot_discrete_dimension_t;
using ::ddc::KnotDiscreteDimension;
using ::ddc::KnotsAsInterpolationPoints;
using ::ddc::NonUniformBSplines;
using ::ddc::NonUniformBsplinesKnots;
using ::ddc::NullExtrapolationRule;
using ::ddc::PeriodicExtrapolationRule;
using ::ddc::SplineBuilder;
using ::ddc::SplineBuilder2D;
using ::ddc::SplineBuilder3D;
using ::ddc::SplineBuilderClosure;
using ::ddc::SplineEvaluator;
using ::ddc::SplineEvaluator2D;
using ::ddc::SplineEvaluator3D;
using ::ddc::SplineEvaluatorND;
using ::ddc::SplineSolver;
using ::ddc::UniformBSplines;
using ::ddc::UniformBsplinesKnots;
using ::ddc::operator<<;

namespace concepts {

using ::ddc::concepts::non_uniform_bsplines;
using ::ddc::concepts::uniform_bsplines;

} // namespace concepts

namespace detail {

using ::ddc::detail::SplinesLinearProblem;
using ::ddc::detail::SplinesLinearProblem2x2Blocks;
using ::ddc::detail::SplinesLinearProblem3x3Blocks;
using ::ddc::detail::SplinesLinearProblemBand;
using ::ddc::detail::SplinesLinearProblemDense;
using ::ddc::detail::SplinesLinearProblemMaker;
using ::ddc::detail::SplinesLinearProblemPDSBand;
using ::ddc::detail::SplinesLinearProblemPDSTridiag;
using ::ddc::detail::SplinesLinearProblemSparse;
using ::ddc::detail::operator<<;
using ::ddc::detail::ipow;
using ::ddc::detail::modulo;

} // namespace detail

} // namespace ddc
