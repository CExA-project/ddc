#pragma once
#include <algorithm>
#include <cassert>
#include <iomanip>
#include <iostream>
#include <memory>
#include <utility>

#include <petscksp.h>
#include <petscvec.h>

#include "view.hpp"

class Matrix
{
public:
    Matrix(int mat_size) : n(mat_size)
    {
        data = std::make_unique<double[]>(n * n);
        for (int i = 0; i < n * n; i++) {
            // data.get()[i] = std::rand()%10==0 ? std::rand()%1000 : 0; // Fills randomly a sparse matrix
            data.get()[i] = 1 + std::rand() % 1000; // Fills randomly a dense matrix
        }
    }
    virtual ~Matrix() = default;
    std::unique_ptr<double[]> data;
    virtual Vec to_petsc_vec(double* vec_ptr, size_t n) const
    {
        Vec v;
        VecCreate(PETSC_COMM_SELF, &v);
        VecSetSizes(v, PETSC_DECIDE, n);
        VecSetFromOptions(v);
        VecPlaceArray(v, vec_ptr);
        return v;
    }
    virtual Mat to_petsc_mat(double* mat_ptr, size_t n, size_t m) const
    {
        PetscInt* rows = (PetscInt*)malloc((n + 1) * sizeof(PetscInt));
        PetscInt* cols = (PetscInt*)malloc(n * m * sizeof(PetscInt));

        // Generate rows indices
        for (PetscInt i = 0; i < n + 1; i++) {
            rows[i] = i * m;
        }

        // Generate cols indices
        for (PetscInt k = 0; k < n * m; k++) {
            cols[k] = k % n;
        }

        Mat M;
        MatCreateSeqAIJWithArrays(PETSC_COMM_SELF, n, m, rows, cols, mat_ptr, &M);
        return M;
    }
    virtual double get_element(int i, int j) const = 0;
    virtual void set_element(int i, int j, double aij) = 0;
    virtual void factorize()
    {
        int const info = factorize_method();

        if (info < 0) {
            std::cerr << -info << "-th argument had an illegal value" << std::endl;
            // TODO: Add LOG_FATAL_ERROR
        } else if (info > 0) {
            std::cerr << "U(" << info << "," << info << ") is exactly zero.";
            std::cerr << " The factorization has been completed, but the factor";
            std::cerr << " U is exactly singular, and division by zero will occur "
                         "if "
                         "it is used to";
            std::cerr << " solve a system of equations.";
            // TODO: Add LOG_FATAL_ERROR
        }
    }
    virtual DSpan1D solve_inplace(DSpan1D const b) const
    {
        assert(int(b.extent(0)) == n);
        int const info = solve_inplace_method(b.data_handle(), 'N', 1);

        if (info < 0) {
            std::cerr << -info << "-th argument had an illegal value" << std::endl;
            // TODO: Add LOG_FATAL_ERROR
        }
        return b;
    }
    virtual DSpan1D solve_inplace_krylov(DSpan1D const b) const
    {
        Vec b_vec = to_petsc_vec(b.data_handle(), b.size());
        Mat data_mat = to_petsc_mat(data.get(), n, n);
        Vec x_vec;
        VecCreate(PETSC_COMM_SELF, &x_vec);
        VecSetSizes(x_vec, PETSC_DECIDE, n);
        VecSetFromOptions(x_vec);
        KSP ksp;
        KSPCreate(PETSC_COMM_SELF, &ksp);
        KSPSetFromOptions(ksp);
        KSPSetOperators(ksp, data_mat, data_mat);
        KSPSolve(ksp, b_vec, x_vec);
        PetscInt its;
        KSPGetIterationNumber(ksp, &its);
        PetscPrintf(PETSC_COMM_SELF, "Iterations %" PetscInt_FMT "\n", its);
        double* x;
        VecGetArray(x_vec, &x);
        return DSpan1D(x, n);
    }
    virtual DSpan1D solve_transpose_inplace(DSpan1D const b) const
    {
        assert(int(b.extent(0)) == n);
        int const info = solve_inplace_method(b.data_handle(), 'T', 1);

        if (info < 0) {
            std::cerr << -info << "-th argument had an illegal value" << std::endl;
            // TODO: Add LOG_FATAL_ERROR
        }
        return b;
    }
    virtual DSpan2D solve_multiple_inplace(DSpan2D const bx) const
    {
        assert(int(bx.extent(1)) == n);
        int const info = solve_inplace_method(bx.data_handle(), 'N', bx.extent(0));

        if (info < 0) {
            std::cerr << -info << "-th argument had an illegal value" << std::endl;
            // TODO: Add LOG_FATAL_ERROR
        }
        return bx;
    }
    int get_size() const
    {
        return n;
    }

    std::ostream& operator<<(std::ostream& os)
    {
        int const n = get_size();
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                os << std::fixed << std::setprecision(3) << std::setw(10) << get_element(i, j);
            }
            os << std::endl;
        }
        return os;
    }

protected:
    virtual int factorize_method() = 0;
    virtual int solve_inplace_method(double* b, char transpose, int n_equations) const = 0;
    int const n; // matrix size
};
