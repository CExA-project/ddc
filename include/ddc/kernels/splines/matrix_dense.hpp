#ifndef MATRIX_DENSE_H
#define MATRIX_DENSE_H
#include <memory>

#include "sll/matrix.hpp"

class Matrix_Dense : public Matrix
{
public:
    Matrix_Dense(int);
    virtual double get_element(int i, int j) const override;
    virtual void set_element(int i, int j, double aij) override;

private:
    virtual int factorize_method() override;
    virtual int solve_inplace_method(double* b, char transpose, int n_equations) const override;
    std::unique_ptr<int[]> ipiv;
    std::unique_ptr<double[]> a;
};

#endif // MATRIX_DENSE_H
