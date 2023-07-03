#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <iosfwd>
#include <memory>
#include <tuple>
#include <vector>

#include <experimental/mdspan>

#include <sll/deprecated/boundary_conditions.hpp>
#include <sll/deprecated/bsplines.hpp>
#include <sll/deprecated/bsplines_uniform.hpp>
#include <sll/deprecated/spline_1d.hpp>
#include <sll/deprecated/spline_builder_1d.hpp>
#include <sll/view.hpp>

#include <gtest/gtest.h>

#include <stdlib.h>

using namespace std;
using namespace std::experimental;
using namespace deprecated;

constexpr double TWO_PI = 2. * M_PI;

static inline double eval_cos(
        double const x,
        Span1D<double> const& coeffs,
        int const derivative = 0);

Span1D<double>& eval_cos(
        Span1D<double>& y,
        Span1D<double> const& x,
        Span1D<double> const& coeffs,
        int const derivative = 0);

void cos_splines_test(
        double& max_norm_error,
        double& max_norm_error_diff,
        double& max_norm_error_int,
        int const degree,
        int const N,
        double const x0,
        double const xN,
        BoundCond const bc_xmin,
        BoundCond const bc_xmax,
        Span1D<double> const& coeffs,
        DSpan1D const& eval_pts_input = {});

static inline double eval_cos(double const x, Span1D<double> const& coeffs, int const derivative)
{
    return pow(TWO_PI * coeffs[0], derivative)
           * cos(M_PI_2 * derivative + TWO_PI * (coeffs[0] * x + coeffs[1]));
}

Span1D<double>& eval_cos(
        Span1D<double>& y,
        Span1D<double> const& x,
        Span1D<double> const& coeffs,
        int const derivative)
{
    assert(y.extent(0) == x.extent(0));
    for (int ii = 0; ii < y.extent(0); ++ii) {
        y[ii] = eval_cos(x[ii], coeffs, derivative);
    }
    return y;
}


void cos_splines_test(
        double& max_norm_error,
        double& max_norm_error_diff,
        double& max_norm_error_int,
        int const degree,
        int const N,
        double const x0,
        double const xN,
        BoundCond const bc_xmin,
        BoundCond const bc_xmax,
        Span1D<double> const& coeffs,
        DSpan1D const& eval_pts_input)
{
    // Create B-splines (uniform or non-uniform depending on input)
    std::shared_ptr<BSplines> bspline = std::make_shared<UniformBSplines>(
            degree,
            (bc_xmin == BoundCond::PERIODIC),
            x0,
            xN,
            SplineBuilder1D::compute_num_cells(degree, bc_xmin, bc_xmax, N));

    // Initialize 1D spline
    Spline1D spline(*bspline);

    // Initialize 1D spline interpolator
    SplineBuilder1D spline_interpolator(*bspline, bc_xmin, bc_xmax);

    DSpan1D const xgrid = spline_interpolator.get_interp_points();
    DSpan1D eval_pts;
    if (0 != eval_pts_input.extent(0)) {
        eval_pts = eval_pts_input;
    } else {
        eval_pts = xgrid;
    }

    vector<double> yvals_data(N);
    Span1D<double> yvals(yvals_data.data(), yvals_data.size());
    eval_cos(yvals, xgrid, coeffs);

    // computation of rhs'(0) and rhs'(n)
    // -> deriv_rhs(0) = rhs'(n) and deriv_rhs(1) = rhs'(0)
    int const shift = (degree % 2); // shift = 0 for even order, -1 for odd order
    vector<double> Sderiv_lhs_data(degree / 2);
    Span1D<double> Sderiv_lhs(Sderiv_lhs_data.data(), Sderiv_lhs_data.size());
    vector<double> Sderiv_rhs_data(degree / 2);
    Span1D<double> Sderiv_rhs(Sderiv_rhs_data.data(), Sderiv_rhs_data.size());
    Span1D<double>* deriv_l(nullptr);
    Span1D<double>* deriv_r(nullptr);
    if (bc_xmin == BoundCond::HERMITE) {
        for (int ii = 0; ii < degree / 2; ++ii) {
            Sderiv_lhs(ii) = eval_cos(x0, coeffs, ii - shift);
        }
        deriv_l = &Sderiv_lhs;
    }
    if (bc_xmax == BoundCond::HERMITE) {
        for (int ii = 0; ii < degree / 2; ++ii) {
            Sderiv_rhs(ii) = eval_cos(xN, coeffs, ii - shift);
        }
        deriv_r = &Sderiv_rhs;
    }
    spline_interpolator.compute_interpolant(spline, yvals, deriv_l, deriv_r);

    max_norm_error = 0.;
    max_norm_error_diff = 0.;

    for (int ii = 0; ii < eval_pts.extent(0); ++ii) {
        // Check eval function
        double const spline_value = spline.eval(eval_pts[ii]);

        // Compute error
        double const error = spline_value - eval_cos(eval_pts[ii], coeffs);
        max_norm_error = std::max(max_norm_error, abs(error));

        // Check eval_deriv function
        double const spline_deriv_value = spline.eval_deriv(eval_pts[ii]);

        // Compute error
        double const error_deriv = spline_deriv_value - eval_cos(eval_pts[ii], coeffs, 1);
        max_norm_error_diff = std::max(max_norm_error_diff, abs(error_deriv));
    }

    max_norm_error_int = spline.integrate();
}



constexpr array<BoundCond, 3> available_bc
        = {BoundCond::PERIODIC, BoundCond::HERMITE, BoundCond::GREVILLE};
//TODO : Add missing test conditions NEUMANN and HERMITE_LAGRANGE

class SplinesTest : public testing::TestWithParam<std::tuple<int, BoundCond, BoundCond>>
{
    // You can implement all the usual fixture class members here.
    // To access the test parameter, call GetParam() from class
    // TestWithParam<T>.
};

INSTANTIATE_TEST_SUITE_P(
        SplinesAtPoints,
        SplinesTest,
        testing::Combine(
                testing::Range(4, 7),
                testing::ValuesIn(available_bc),
                testing::ValuesIn(available_bc)));

//--------------------------------------
// TESTS
//--------------------------------------
TEST_P(SplinesTest, Test)
{
    //int const ncells;
    //int const bc_xmin, bc_xmax;


    constexpr double x0 = 0.0;
    constexpr double xN = 1.0;
    constexpr int N = 22;
    constexpr double tol = 1e-12;
    constexpr double max_norm_profile = 1.0;


    array<double, 2> coeffs_data = {1., 0.};
    Span1D<double> coeffs(coeffs_data.data(), coeffs_data.size());

    auto const [degree, bc_xmin, bc_xmax] = GetParam();

    if (bc_xmin != bc_xmax and (bc_xmin == BoundCond::PERIODIC or bc_xmax == BoundCond::PERIODIC)) {
        return;
    }

    //if (degree != 3
    //    and (bc_xmin == BoundCond::HERMITE_LAGRANGE
    //         or bc_xmax == BoundCond::HERMITE_LAGRANGE)) {
    //    continue;
    //}

    //if ((degree > 3 or degree == 2)
    //    and (bc_xmin == BoundCond::NEUMANN or bc_xmax == BoundCond::NEUMANN)) {
    //    continue;
    //}

    double max_norm_error;
    double max_norm_error_diff;
    double max_norm_error_int;
    cos_splines_test(
            max_norm_error,
            max_norm_error_diff,
            max_norm_error_int,
            degree,
            N,
            x0,
            xN,
            bc_xmin,
            bc_xmax,
            coeffs);

    // Calculate relative error norms from absolute ones
    max_norm_error = max_norm_error / max_norm_profile;
    EXPECT_LE(max_norm_error, tol)
            << "While evaluating spline at interpolation points (error should be zero) \n"
            << "With \n"
            << " * Number of points in grid: N = " << N << "\n"
            << " * Number of evaluation points : " << N << "\n";
}
