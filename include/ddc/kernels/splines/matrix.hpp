#pragma once
#include <algorithm>
#include <cassert>
#include <iomanip>
#include <iostream>
#include <memory>
#include <utility>

#include <petscksp.h>
#include <petscmat.h>
#include <petscvec.h>

#include "view.hpp"
#include <Kokkos_Core.hpp>

class Matrix
{
public:
    Matrix(const int mat_size) : n(mat_size)
    {
        data = (double*)malloc((n * n)*sizeof(double));
        // Kokkos::View<double*, Kokkos::DefaultExecutionSpace> data_view("data",n*n);
		// data = data_view.data();
        for (int i = 0; i < n * n; i++) {
            // data[i] = std::rand()%10==0 ? std::rand()%1000 : 0; // Fills randomly a sparse matrix
            data[i] = 1 + std::rand() % 10; // Fills randomly a dense matrix
			// std::cout << data_view[i];
			// std::cout << data[i];
			// std::cout << "\n";
        }
    }
    virtual ~Matrix() = default;
    double* data;
    virtual Vec to_petsc_vec(double* vec_ptr, size_t n) const
    {
        Vec v;
        VecCreate(PETSC_COMM_SELF, &v);
        VecSetSizes(v, PETSC_DECIDE, n);
		// VecSetType(v, VECKOKKOS);
		VecSetFromOptions(v);
        VecPlaceArray(v, vec_ptr);
        return v;
    }
    virtual Mat to_petsc_mat(double* mat_ptr, size_t m, size_t n) const
    {
        PetscInt* rows = (PetscInt*)malloc((m + 1) * sizeof(PetscInt));
        PetscInt* cols = (PetscInt*)malloc(m * n * sizeof(PetscInt));

        // Generate rows indices
        for (PetscInt i = 0; i < m + 1; i++) {
            rows[i] = i * n;
        }

        // Generate cols indices
        for (PetscInt k = 0; k < m * n; k++) {
            cols[k] = k % n;
        }

        Mat M;
		double* data_copy = data;
        MatCreateSeqAIJWithArrays(PETSC_COMM_SELF, m, n, rows, cols, data_copy, &M);
		MatSetFromOptions(M);
        // MatCreateAIJ(PETSC_COMM_SELF, n, m, n, m, 0, NULL, 0, NULL, &M);
		// MatSetType(M, MATAIJKOKKOS);
		// MatSetValues(M, n, rows, m, cols, mat_ptr, INSERT_VALUES);
		// MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
		// MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);

		PetscScalar va;
		MatGetValue(M,0,0,&va);
		std::cout <<va;
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
        Kokkos::View<double*, Kokkos::HostSpace> b_cpu(b.data_handle(), b.size());
        Kokkos::View<double*, Kokkos::DefaultExecutionSpace> b_gpu("b_gpu", b.size());
		Kokkos::deep_copy(b_gpu, b_cpu);
        Vec b_vec = to_petsc_vec(b_gpu.data(), b.size());
        Mat data_mat = to_petsc_mat(data, n, n);
        Kokkos::View<double*, Kokkos::HostSpace> x_cpu("x_cpu", b.size());
        Kokkos::View<double*, Kokkos::DefaultExecutionSpace> x_gpu("x_gpu", b.size());
		Kokkos::deep_copy(x_gpu, x_cpu);
        Vec x_vec = to_petsc_vec(x_gpu.data(), b.size());
        KSP ksp;
        KSPCreate(PETSC_COMM_SELF, &ksp);
        KSPSetFromOptions(ksp);
        KSPSetOperators(ksp, data_mat, data_mat);
        KSPSolve(ksp, b_vec, x_vec);
        PetscInt its;
        KSPGetIterationNumber(ksp, &its);

		Vec err;
		VecCreate(PETSC_COMM_SELF, &err);
        VecSetSizes(err, PETSC_DECIDE, b.size());
        // VecSetType(err, VECKOKKOS);
        VecSetFromOptions(err);
		MatMult(data_mat,x_vec,err);
		VecAXPY(err, -1, b_vec);
		PetscReal norm;
		VecNorm(err, NORM_2, &norm);
		PetscPrintf(PETSC_COMM_WORLD, "Norm of error %g iterations %" PetscInt_FMT "\n", (double)norm, its);
        // PetscPrintf(PETSC_COMM_SELF, "Iterations %" PetscInt_FMT "\n", its);
		Kokkos::deep_copy(x_cpu, x_gpu);
        return DSpan1D(x_cpu.data(), n);
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
