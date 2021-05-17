#ifndef SPLINE_1D_H
#define SPLINE_1D_H
#include <memory>

#include "boundary_value.h"
#include "bsplines.h"
#include "null_boundary_value.h"

class Spline_1D
{
public:
    Spline_1D(
            const BSplines& bspl,
            const BoundaryValue& left_bc = NullBoundaryValue::value,
            const BoundaryValue& right_bc = NullBoundaryValue::value);
    ~Spline_1D() = default;
    bool belongs_to_space(const BSplines& bspline) const;
    double eval(double x) const;
    double eval_deriv(double x) const;
    void eval_array(DSpan1D const x, DSpan1D y) const;
    void eval_array_deriv(DSpan1D const x, DSpan1D y) const;
    double integrate() const;

private:
    std::unique_ptr<double[]> bcoef_ptr;
    DSpan1D bcoef;
    const BSplines& bspl;
    const BoundaryValue& left_bc;
    const BoundaryValue& right_bc;

    // Internal templated functions
    template <
            class T,
            typename std::enable_if<std::is_base_of<BSplines, T>::value>::type* = nullptr>
    double eval_intern_no_bcs(double x, const T& bspl, DSpan1D& vals) const;
    template <
            class T,
            bool periodic,
            typename std::enable_if<std::is_base_of<BSplines, T>::value>::type* = nullptr>
    double eval_intern(double x, const T& bspl, DSpan1D& vals) const;
    template <
            class T,
            typename std::enable_if<std::is_base_of<BSplines, T>::value>::type* = nullptr>
    double eval_deriv_intern(double x, const T& bspl, DSpan1D& vals) const;
    template <
            class T,
            bool periodic,
            typename std::enable_if<std::is_base_of<BSplines, T>::value>::type* = nullptr>
    void eval_array_loop(DSpan1D const& x, DSpan1D& y) const;
    template <
            class T,
            typename std::enable_if<std::is_base_of<BSplines, T>::value>::type* = nullptr>
    void eval_array_deriv_loop(DSpan1D const& x, DSpan1D& y) const;

    // Friends
    friend class Spline_interpolator_1D;
    friend class Spline_interpolator_2D;
    friend int spline_1d_get_ncoeffs(Spline_1D* spl);
    friend double* spline_1d_get_coeffs(Spline_1D* spl);
};

#endif // SPLINE_1D_H
