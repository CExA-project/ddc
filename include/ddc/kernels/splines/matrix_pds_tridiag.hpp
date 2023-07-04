#ifndef MATRIX_PDS_BANDED_H
#define MATRIX_PDS_BANDED_H

#include <cassert>
#include <cmath>
#include <memory>

#include <string.h>

#include "matrix.hpp"

extern "C" int dpttrf_(int const* n, double* d, double* e, int* info);
extern "C" int dpttrs_(
        int const* n,
        int const* nrhs,
        double* d,
        double* e,
        double* b,
        int const* ldb,
        int* info);

class Matrix_PDS_Tridiag : public Matrix
{
    /*
     * Represents a real symmetric positive definite matrix
     * stored in a block format
     * */
public:
    Matrix_PDS_Tridiag(int const n)
        : Matrix(n)
        , d(std::make_unique<double[]>(n))
        , l(std::make_unique<double[]>(n - 1))
    {
        memset(d.get(), 0, n * sizeof(double));
        memset(l.get(), 0, (n - 1) * sizeof(double));
    }
    double get_element(int i, int j) const override
    {
        if (i == j) {
            return d[i];
        }
        if (i > j) {
            std::swap(i, j);
        }
        if (i + 1 == j) {
            return l[i];
        }
        return 0.0;
    }
    void set_element(int i, int j, double const a_ij) override
    {
        if (i == j) {
            d[i] = a_ij;
            return;
        }
        if (i > j) {
            std::swap(i, j);
        }
        if (i + 1 != j) {
            assert(std::fabs(a_ij) < 1e-20);
        } else {
            l[i] = a_ij;
        }
    }

protected:
    int factorize_method() override
    {
        int info;
        dpttrf_(&n, d.get(), l.get(), &info);
        return info;
    }
    int solve_inplace_method(double* b, char, int const n_equations) const override
    {
        int info;
        dpttrs_(&n, &n_equations, d.get(), l.get(), b, &n, &info);
        return info;
    }
    std::unique_ptr<double[]> d; // diagonal
    std::unique_ptr<double[]> l; // lower diagonal
};

#endif // MATRIX_SYMMETRIC_BANDED_H
