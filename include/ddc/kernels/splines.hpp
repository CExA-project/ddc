// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include "ddc/ddc.hpp"
#include "ddc/kernels/splines/bsplines_non_uniform.hpp"
#include "ddc/kernels/splines/bsplines_uniform.hpp"
#include "ddc/kernels/splines/constant_extrapolation_rule.hpp"
#include "ddc/kernels/splines/deriv.hpp"
#include "ddc/kernels/splines/greville_interpolation_points.hpp"
#include "ddc/kernels/splines/integrals.hpp"
#include "ddc/kernels/splines/knot_discrete_dimension_type.hpp"
#include "ddc/kernels/splines/knots_as_interpolation_points.hpp"
#include "ddc/kernels/splines/math_tools.hpp"
#include "ddc/kernels/splines/null_extrapolation_rule.hpp"
#include "ddc/kernels/splines/periodic_extrapolation_rule.hpp"
#include "ddc/kernels/splines/spline_boundary_conditions.hpp"
#include "ddc/kernels/splines/spline_builder.hpp"
#include "ddc/kernels/splines/spline_builder_2d.hpp"
#include "ddc/kernels/splines/spline_evaluator.hpp"
#include "ddc/kernels/splines/spline_evaluator_2d.hpp"
#include "ddc/kernels/splines/splines_linear_problem.hpp"
#include "ddc/kernels/splines/splines_linear_problem_2x2_blocks.hpp"
#include "ddc/kernels/splines/splines_linear_problem_3x3_blocks.hpp"
#include "ddc/kernels/splines/splines_linear_problem_band.hpp"
#include "ddc/kernels/splines/splines_linear_problem_dense.hpp"
#include "ddc/kernels/splines/splines_linear_problem_maker.hpp"
#include "ddc/kernels/splines/splines_linear_problem_pds_band.hpp"
#include "ddc/kernels/splines/splines_linear_problem_pds_tridiag.hpp"
#include "ddc/kernels/splines/splines_linear_problem_sparse.hpp"
#include "ddc/kernels/splines/view.hpp"
