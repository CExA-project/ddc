#ifndef MATRIX_DENSE_H
#define MATRIX_DENSE_H
#include <cassert>
#include <memory>

#include "matrix.hpp"

namespace ddc::detail {
extern "C" int dgetrf_(int const* m, int const* n, double* a, int const* lda, int* ipiv, int* info);
extern "C" int dgetrs_(
        char const* trans,
        int const* n,
        int const* nrhs,
        double* a,
        int const* lda,
        int* ipiv,
        double* b,
        int const* ldb,
        int* info);

class Matrix_Dense : public Matrix
{
public:
    Matrix_Dense(int const n) : Matrix(n)
    {
        assert(get_size() > 0);
        ipiv = std::make_unique<int[]>(get_size());
        a = std::make_unique<double[]>(get_size() * get_size());
        for (int i = 0; i < get_size() * get_size(); ++i) {
            a[i] = 0;
        }
    }
    double get_element(int const i, int const j) const override
    {
        assert(i < get_size());
        assert(j < get_size());
        return a[j * get_size() + i];
    }
    void set_element(int const i, int const j, double const aij) override
    {
        a[j * get_size() + i] = aij;
    }

private:
    int factorize_method() override
    {
        int info;
        int const n = get_size();
        dgetrf_(&n, &n, a.get(), &n, ipiv.get(), &info);
        return info;
    }
    int solve_inplace_method(double* b, char const transpose, int const n_equations) const override
    {
        int info;
        int const n = get_size();
        dgetrs_(&transpose, &n, &n_equations, a.get(), &n, ipiv.get(), b, &n, &info);
        return info;
    }
    std::unique_ptr<int[]> ipiv;
    std::unique_ptr<double[]> a;
};

} // namespace ddc::detail
#endif // MATRIX_DENSE_H
