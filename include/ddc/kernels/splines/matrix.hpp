#pragma once
#include <algorithm>
#include <cassert>
#include <iomanip>
#include <iostream>
#include <memory>
#include <utility>

#include "Kokkos_Core_fwd.hpp"
#include "view.hpp"
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include <ginkgo/core/base/device_matrix_data.hpp>
#include <ginkgo/ginkgo.hpp>

class Matrix
{
public:
	struct FillMatrixFunctor
	{
		int m;
		int n;
		int* rows;
		int* cols;
		double* data;
		Kokkos::Random_XorShift64_Pool<> random_pool;

		FillMatrixFunctor(int m, int n, int* rows_ptr, int* cols_ptr, double* data_ptr) : m(m), n(n), rows(rows_ptr), cols(cols_ptr), data(data_ptr) {
			random_pool = Kokkos::Random_XorShift64_Pool<>(/*seed=*/12345);
		}

		__host__ __device__
		void operator()(const int i) const
		{
			if (i<m+1) {
           		rows[i] = i * n; //CSR
			}
			// rows[i] = i * n; //COO
           	cols[i] = i % n;
			auto generator = random_pool.get_state();
			data[i] = 1 + generator.drand(0.,9.); // Fills randomly a dense matrix
			random_pool.free_state(generator);
			// data[i] = 5+0.1*(i%5); // Fills randomly a dense matrix
		}
	};
	Matrix(const int mat_size) : m(mat_size), n(mat_size)
    {
		rows = (int*)Kokkos::kokkos_malloc<Kokkos::DefaultExecutionSpace>((n +1) * sizeof(int));
        cols = (int*)Kokkos::kokkos_malloc<Kokkos::DefaultExecutionSpace>(n * n * sizeof(int));
        data = (double*)Kokkos::kokkos_malloc<Kokkos::DefaultExecutionSpace>(n * n *sizeof(double));

		# if 1 
        // Kokkos::View<double*, Kokkos::DefaultExecutionSpace> data_view("data",n*n);
		// data = data_view.data();
        // for (int i = 0; i < n * n; i++) {
        Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0,n*n), FillMatrixFunctor(m,n,rows,cols,data));
		# endif
    }
    virtual ~Matrix() {
		Kokkos::kokkos_free(rows);
		Kokkos::kokkos_free(cols);
		Kokkos::kokkos_free(data);
	};
	int n_batch = 10;
	int m;
	int n;
	int* rows;
	int* cols;
    double* data; // TODO : make a struct for COO
	virtual std::unique_ptr<gko::matrix::Dense<>, std::default_delete<gko::matrix::Dense<>>> to_gko_vec(double* vec_ptr, size_t n) const
    {
		# if 0
        int* indices = (int*)Kokkos::kokkos_malloc<Kokkos::DefaultExecutionSpace::memory_space>(n * sizeof(int));
        int* zero = (int*)Kokkos::kokkos_malloc<Kokkos::DefaultExecutionSpace::memory_space>(sizeof(int));
        // Kokkos::View<PetscInt*, Kokkos::DefaultExecutionSpace> indices_view("indices",n);
        // PetscInt* indices = indices_view.data();
		
		// Generate cols indices
        Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0,n), KOKKOS_LAMBDA (int i)  {
            indices[i] = i;
        });
		zero[0] = 0;
		# endif

		// auto v = gko::matrix::Dense<>::create(gko_device_exec, gko::dim<2>{n,1});
		// v->get_values() = vec_ptr;
		auto v = gko::matrix::Dense<>::create(gko_device_exec, gko::dim<2>{n,1}, gko::array<double>::view(gko_device_exec, n, vec_ptr), 1);
		// auto v = gko::matrix::Dense<>::create(gko_device_exec, gko::dim<2>{n,1});
		// auto v = gko::matrix::Dense<>(gko_device_exec, gko::dim<2>{n,1}, gko::array<double>(gko_device_exec, n, vec_ptr), n);
		// v->read(gko::device_matrix_data<double,int>(gko_device_exec, gko::dim<2>{n,1}, &indices, &zero, &vec_ptr));
        return v;
    }
	#if 1
    virtual std::unique_ptr<gko::matrix::Csr<>, std::default_delete<gko::matrix::Csr<>>> to_gko_mat(double* mat_ptr, size_t m, size_t n) const
    {
                // Kokkos::View<PetscInt*, Kokkos::DefaultExecutionSpace> rows_view("rows",m*n);
        // PetscInt* rows = rows_view.data();
        // Kokkos::View<PetscInt*, Kokkos::DefaultExecutionSpace> cols_view("cols",m*n);
        // PetscInt* cols = cols_view.data();
		auto M = gko::matrix::Csr<>::create(gko_device_exec, gko::dim<2>{m,n}, gko::array<double>::view(gko_device_exec, m*n, mat_ptr), gko::array<int>::view(gko_device_exec, m*n, cols), gko::array<int>::view(gko_device_exec, m+1, rows));
		// M->read(gko::device_matrix_data<double,int>(gko_device_exec, gko::dim<2>{n,1}, &rows, &cols, &data));

        return M;
    }
	# endif
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
	# if 1 
    virtual DSpan1D solve_inplace_krylov(DSpan1D const b) const
    {
        Kokkos::View<double*, Kokkos::HostSpace> b_cpu(b.data_handle(), b.size());
        Kokkos::View<double*, Kokkos::DefaultExecutionSpace> b_gpu("b_gpu", b.size());
		Kokkos::deep_copy(b_gpu, b_cpu);
        // double* b_gpu = (double*)Kokkos::kokkos_malloc<Kokkos::DefaultExecutionSpace>((b.size())*sizeof(double));
        // double* b_gpu = gko_device_exec->alloc<double>(b.size());
        auto b_vec = to_gko_vec(b_gpu.data(), b.size());
        auto data_mat = gko::share(to_gko_mat(data, n, n));
        Kokkos::View<double*, Kokkos::HostSpace> x_cpu("x_cpu", b.size());
        Kokkos::View<double*, Kokkos::DefaultExecutionSpace> x_gpu("x_gpu", b.size());
		Kokkos::deep_copy(x_gpu, x_cpu);
        auto x_vec = to_gko_vec(x_gpu.data(), b.size());

		// Create the solver
		auto solver =
			gko::solver::Cg<>::build()
				.with_preconditioner(gko::preconditioner::Jacobi<>::build().on(gko_device_exec))
				.with_criteria(
					gko::stop::Iteration::build().with_max_iters(20u).on(gko_device_exec),
					gko::stop::ResidualNorm<>::build()
						.with_reduction_factor(1e-15)
						.on(gko_device_exec))
				.on(gko_device_exec);
		// Solve system
		solver->generate(data_mat)->apply(b_vec, x_vec);

		// Write result
		std::cout << "-----------------------";
		write(std::cout, data_mat);
		std::cout << "-----------------------";
		write(std::cout, x_vec);

		// Calculate residual
		auto err = gko::clone(gko_device_exec, b_vec);
		auto one = gko::initialize<gko::matrix::Dense<>>({1.0}, gko_device_exec);
		auto neg_one = gko::initialize<gko::matrix::Dense<>>({-1.0}, gko_device_exec);
		auto res = gko::initialize<gko::matrix::Dense<>>({0.0}, gko_device_exec);
		std::cout << "-----------------------";
		data_mat->apply(one, x_vec, neg_one, err);
		err->compute_norm2(res);

		std::cout << "Residual norm sqrt(r^T r):\n";
		write(std::cout, res);

		# if 0 
        KSP ksp;
        KSPCreate(PETSC_COMM_SELF, &ksp);
		KSPSetType(ksp, KSPBCGS);
        KSPSetFromOptions(ksp);
        KSPSetOperators(ksp, data_mat, data_mat);
        KSPSolve(ksp, b_vec, x_vec);
        PetscInt its;
        KSPGetIterationNumber(ksp, &its);

		Vec err;
		VecCreate(PETSC_COMM_SELF, &err);
        VecSetSizes(err, PETSC_DECIDE, b.size());
        VecSetType(err, VECCUDA);
        VecSetFromOptions(err);
		MatMult(data_mat,x_vec,err);
		VecAXPY(err, -1, b_vec);
		PetscReal norm;
		VecNorm(err, NORM_2, &norm);
		PetscPrintf(PETSC_COMM_WORLD, "Norm of error %g iterations %" PetscInt_FMT "\n", (double)norm, its);
		KSPDestroy(&ksp);
		MatDestroy(&data_mat);
		VecDestroy(&x_vec);
		VecDestroy(&b_vec);
		VecDestroy(&err);
		# endif
		Kokkos::deep_copy(x_cpu, x_gpu);
        return DSpan1D(x_cpu.data(), n);
    }
	# endif
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
    // int const n; // matrix size
};
