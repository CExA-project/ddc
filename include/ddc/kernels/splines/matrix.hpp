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
#include "ginkgo/core/matrix/dense.hpp"

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
	// int n_batch = 400000000;
	int n_batch = 1e2;
	int m;
	int n;
	int* rows;
	int* cols;
    double* data; // TODO : make a struct for CSR
	virtual std::unique_ptr<gko::matrix::BatchDense<>, std::default_delete<gko::matrix::BatchDense<>>> to_gko_vec(double* vec_ptr, size_t n) const
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
		auto v = gko::matrix::BatchDense<>::create(gko_device_exec, gko::batch_dim<>(1,gko::dim<2>{n,1}), gko::array<double>::view(gko_device_exec, n, vec_ptr), gko::batch_stride(1, 1));
		// auto v = gko::matrix::Dense<>::create(gko_device_exec, gko::dim<2>{n,1});
		// auto v = gko::matrix::Dense<>(gko_device_exec, gko::dim<2>{n,1}, gko::array<double>(gko_device_exec, n, vec_ptr), n);
		// v->read(gko::device_matrix_data<double,int>(gko_device_exec, gko::dim<2>{n,1}, &indices, &zero, &vec_ptr));
        return v;
    }
	#if 1
    virtual std::unique_ptr<gko::matrix::BatchCsr<>, std::default_delete<gko::matrix::BatchCsr<>>> to_gko_mat(double* mat_ptr, size_t m, size_t n) const
    {
       //  Kokkos::View<PetscInt*, Kokkos::DefaultExecutionSpace> rows_view("rows",m*n);
        // PetscInt* rows = rows_view.data();
        // Kokkos::View<PetscInt*, Kokkos::DefaultExecutionSpace> cols_view("cols",m*n);
        // PetscInt* cols = cols_view.data();
		auto M = gko::matrix::BatchCsr<>::create(gko_device_exec, gko::batch_dim<>(1, gko::dim<2>{m,n}), gko::array<double>::view(gko_device_exec, m*n, mat_ptr), gko::array<int>::view(gko_device_exec, m*n, cols), gko::array<int>::view(gko_device_exec, m+1, rows));
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
        auto b_vec_batch = gko::matrix::BatchDense<>::create(gko_device_exec, n_batch, b_vec.get());
        auto data_mat = gko::share(to_gko_mat(data, n, n));
        auto data_mat_batch = gko::share(gko::matrix::BatchCsr<>::create(gko_device_exec, n_batch, data_mat.get()));
        Kokkos::View<double*, Kokkos::HostSpace> x_cpu("x_cpu", b.size());
        Kokkos::View<double*, Kokkos::DefaultExecutionSpace> x_gpu("x_gpu", b.size());
		Kokkos::deep_copy(x_gpu, x_cpu);
        auto x_vec = to_gko_vec(x_gpu.data(), b.size());
        auto x_vec_batch = gko::matrix::BatchDense<>::create(gko_device_exec, n_batch, x_vec.get());

		// Create the solver
		# if 1 // matrix-matrix linear system
		auto solver =
			gko::solver::Cg<>::build()
				.with_preconditioner(gko::preconditioner::Jacobi<>::build().on(gko_device_exec))
				.with_criteria(
					gko::stop::Iteration::build().with_max_iters(20u).on(gko_device_exec),
					gko::stop::ResidualNorm<>::build()
						.with_reduction_factor(1e-15)
						.on(gko_device_exec))
				.on(gko_device_exec);
		solver->generate(std::move(data_mat->unbatch().at(0)))
			  ->apply(gko::matrix::Dense<>::create_with_type_of(gko::concatenate_dense_matrices(gko_device_exec, b_vec_batch->unbatch()), gko_device_exec, gko::dim<2>{n,n_batch})
					, gko::matrix::Dense<>::create_with_type_of(gko::concatenate_dense_matrices(gko_device_exec, x_vec_batch->unbatch()), gko_device_exec, gko::dim<2>{n,n_batch})); // NOTE : There is an implicit copy here dur to gko::copy_with_type_of, need to avoid that 
		# else // full batched
		auto solver =
		gko::solver::BatchBicgstab<>::build()
			.with_default_max_iterations(500)
            .with_default_residual_tol(1e-15)
	        .with_tolerance_type(gko::stop::batch::ToleranceType::relative)
			.on(gko_device_exec);
		solver->generate(data_mat_batch)->apply(b_vec_batch.get(), x_vec_batch.get());
		#endif

		# if 0
		// Write result
		std::cout << "-----------------------";
		write(std::cout, data_mat);
		std::cout << "-----------------------";
		write(std::cout, x_vec);

		#endif
		# if 1
		// Calculate residual
		auto err = gko::clone(gko_device_exec, b_vec_batch);
		auto one = gko::batch_initialize<gko::matrix::BatchDense<>>(n_batch, {1.0}, gko_device_exec);
		auto neg_one = gko::batch_initialize<gko::matrix::BatchDense<>>(n_batch, {-1.0}, gko_device_exec);
		auto err_norms = gko::matrix::BatchDense<>::create(gko_device_exec->get_master(), gko::batch_dim<>(n_batch,gko::dim<2>{1,1}));
		data_mat_batch->apply(one.get(), x_vec_batch.get(), neg_one.get(), err.get());
		err->compute_norm2(err_norms.get());
		auto unb_err_norms = err_norms->unbatch();

		std::cout << "-----------------------";
		std::cout << "Residual norms sqrt(r^T r):\n";
		for (int i = 0; i < n_batch; ++i) {
			std::cout << unb_err_norms[i]->at(0,0) << "\n";
		}

		#endif
		
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
