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

#include "matrix.hpp"

#include "ginkgo/core/matrix/dense.hpp"
#include <ginkgo/core/base/device_matrix_data.hpp>
#include <ginkgo/ginkgo.hpp>

template <typename ValueType>
ValueType get_first_element(const gko::matrix::Dense<ValueType>* norm)
{
    return norm->get_executor()->copy_val_to_host(norm->get_const_values());
}

template <typename ValueType>
ValueType compute_norm(const gko::matrix::Dense<ValueType>* b)
{
    auto exec = b->get_executor();
    auto b_norm = gko::initialize<gko::matrix::Dense<ValueType>>({0.0}, exec);
    b->compute_norm2(b_norm);
    return get_first_element(b_norm.get());
}
 
 
template <typename ValueType>
ValueType compute_residual_norm(
    const gko::LinOp* system_matrix, const gko::matrix::Dense<ValueType>* b,
    const gko::matrix::Dense<ValueType>* x)
{
    auto exec = system_matrix->get_executor();
    auto one = gko::initialize<gko::matrix::Dense<ValueType>>({1.0}, exec);
    auto neg_one = gko::initialize<gko::matrix::Dense<ValueType>>({-1.0}, exec);
    auto res = gko::clone(b);
    system_matrix->apply(one, x, neg_one, res);
    return compute_norm(res.get());
}

template <typename ValueType>
struct ResidualLogger : gko::log::Logger {
    void on_iteration_complete(const gko::LinOp*, const gko::size_type&,
                               const gko::LinOp* residual,
                               const gko::LinOp* solution,
                               const gko::LinOp* residual_norm) const override
    {
        if (residual_norm) {
            rec_res_norms.push_back(get_first_element(
                gko::as<gko::matrix::Dense<ValueType>>(residual_norm)));
        } else {
            rec_res_norms.push_back(
                compute_norm(gko::as<gko::matrix::Dense<ValueType>>(residual)));
        }
        if (solution) {
            true_res_norms.push_back(compute_residual_norm(
                matrix, b, gko::as<gko::matrix::Dense<ValueType>>(solution)));
        } else {
            true_res_norms.push_back(-1.0);
        }
    }
 
    ResidualLogger(const gko::LinOp* matrix, const gko::matrix::Dense<ValueType>* b)
        : gko::log::Logger(gko::log::Logger::iteration_complete_mask),
          matrix{matrix},
          b{b}
    {}
 
    void write_data(std::ostream& ostream)
    {
        ostream << "Recurrent Residual Norms: " << std::endl;
        ostream << "[" << std::endl;
        for (const auto& entry : rec_res_norms) {
            ostream << " " << entry;
        }
        ostream << "];" << std::endl;
 
        ostream << "True Residual Norms: " << std::endl;
        ostream << "[" << std::endl;
        for (const auto& entry : true_res_norms) {
            ostream << " " << entry;
        }
        ostream << "];" << std::endl;
    }
 
private:
    const gko::LinOp* matrix;
    const gko::matrix::Dense<ValueType>* b;
    mutable std::vector<ValueType> rec_res_norms;
    mutable std::vector<ValueType> true_res_norms;
};
 
class Matrix_Sparse : public Matrix
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
			random_pool = Kokkos::Random_XorShift64_Pool<>(/*seed=*/73);
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
			data[i] = 0;
			// data[i] = generator.drand(0.,10.); // Fills randomly a dense matrix
			// data[i] = Kokkos::max(0.,generator.drand(-200.,10.)); // Fills randomly a dense matrix
			// data[i] = i%n>=i/n-10 && i%n<=i/n+10 ? Kokkos::max(0.,generator.drand(-10,10.)) : 0; // Fills randomly a dense matrix
			// data[i] = i%n>=i/n-10 && i%n<=i/n+10 ? 10-Kokkos::abs(i%n-i/n) : 0; // Create proper band
			// data[i] = i%n==i/n  ? 1 : 0; // Fills randomly a dense matrix
			// data[i] = i%n==i/n || i%n==i/n-1  ? 1 : 0; // Fills randomly a dense matrix
			random_pool.free_state(generator);
			// data[i] = 5+0.1*(i%5); // Fills randomly a dense matrix
		}
	};
	Matrix_Sparse(const int mat_size) : Matrix(mat_size), m(mat_size), n(mat_size)
    {
		rows = (int*)Kokkos::kokkos_malloc<Kokkos::DefaultHostExecutionSpace>((n +1) * sizeof(int));
        cols = (int*)Kokkos::kokkos_malloc<Kokkos::DefaultHostExecutionSpace>(m * n * sizeof(int));
        data = (double*)Kokkos::kokkos_malloc<Kokkos::DefaultHostExecutionSpace>(m * n *sizeof(double));

		# if 1 
        // Kokkos::View<double*, Kokkos::DefaultExecutionSpace> data_view("data",n*n);
		// data = data_view.data();
        for (int i = 0; i < m * n; i++) {
		  if (i<m+1) {
			rows[i] = i * n; //CSR
		  }
		  // rows[i] = i * n; //COO
          cols[i] = i % n;
		  data[i] = 0;
		}
        //Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,n*n), FillMatrixFunctor(m,n,rows,cols,data));
		# endif
    }
    virtual ~Matrix_Sparse() {
		Kokkos::kokkos_free(rows);
		Kokkos::kokkos_free(cols);
		Kokkos::kokkos_free(data);
	};
	// int n_batch = 400000000;
	const int n_batch = 1;
	int m;
	int n;
	int* rows;
	int* cols;
    double* data; // TODO : make a struct for CSR

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
		// auto v = gko::matrix::BatchDense<>::create(gko_device_exec, gko::batch_dim<>(1,gko::dim<2>{n,1}), gko::array<double>::view(gko_device_exec, n, vec_ptr), gko::batch_stride(1, 1));
		// auto v = gko::matrix::Dense<>::create(gko_device_exec, gko::dim<2>{n,1});
		auto v = gko::matrix::Dense<>::create(gko_device_exec, gko::dim<2>{n,n_batch}, gko::array<double>::view(gko_device_exec, n, vec_ptr), 1);
		// auto v = gko::matrix::Dense<>::create(gko_device_exec, gko::dim<2>{n,n_batch});
	// v->read(gko::device_matrix_data<double,int>(gko_device_exec, gko::dim<2>{n,1}, &indices, &zero, &vec_ptr));
        return v;
    }
	#if 1
    virtual std::unique_ptr<gko::matrix::Csr<>, std::default_delete<gko::matrix::Csr<>>> to_gko_mat(double* mat_ptr, size_t m, size_t n) const
    {
       //  Kokkos::View<PetscInt*, Kokkos::DefaultExecutionSpace> rows_view("rows",m*n);
        // PetscInt* rows = rows_view.data();
        // Kokkos::View<PetscInt*, Kokkos::DefaultExecutionSpace> cols_view("cols",m*n);
        // PetscInt* cols = cols_view.data();
		auto M = gko::matrix::Csr<>::create(gko_host_par_exec, gko::dim<2>{m,n}, gko::array<double>::view(gko_host_par_exec, m*n, mat_ptr), gko::array<int>::view(gko_host_par_exec, m*n, cols), gko::array<int>::view(gko_host_par_exec, m+1, rows));
		// auto M = gko::matrix::BatchCsr<>::create(gko_device_exec, gko::batch_dim<>(1, gko::dim<2>{m,n}), gko::array<double>::view(gko_device_exec, m*n, mat_ptr), gko::array<int>::view(gko_device_exec, m*n, cols), gko::array<int>::view(gko_device_exec, m+1, rows));
		// M->read(gko::device_matrix_data<double,int>(gko_device_exec, gko::dim<2>{n,1}, &rows, &cols, &data));

        return M;
    }
	# endif
    virtual double get_element(int i, int j) const override {
	  return 0;
	}
    virtual void set_element(int i, int j, double aij) override {
	  data[i*n+j] = aij;
	}
  
	virtual int factorize_method() override {
	  return 0;
	}
    virtual int solve_inplace_method(double* b, char transpose, int n_equations) const override
    {
        Kokkos::View<double*, Kokkos::HostSpace> b_cpu(b, n);
        Kokkos::View<double*, Kokkos::DefaultExecutionSpace> b_gpu("b_gpu", n);
		Kokkos::deep_copy(b_gpu, b_cpu);
        // double* b_gpu = (double*)Kokkos::kokkos_malloc<Kokkos::DefaultExecutionSpace>((b.size())*sizeof(double));
        // double* b_gpu = gko_device_exec->alloc<double>(b.size());
        auto b_vec_batch = to_gko_vec(b_gpu.data(), n);
        // auto b_vec_batch = gko::matrix::BatchDense<>::create(gko_device_exec, n_batch, b_vec.get());
		// auto b_vec_batch = gko::matrix::Dense<>::create(gko_device_exec, gko::dim<2>{n,n_batch});
		// b_vec_batch->fill(1);
        auto data_mat = gko::share(to_gko_mat(data, n, n));

		// Remove zeros
		# if 1
		auto data_mat_ = gko::matrix_data<>(gko::dim<2>{n,n});
		data_mat->write(data_mat_);
		data_mat_.remove_zeros();
		data_mat->read(data_mat_); // TODO : restore remove_zeros
		# endif
	
		auto data_mat_gpu = gko::share(gko::clone(gko_device_exec, data_mat));
		// auto data_mat_gpu = gko::share(gko::matrix::Csr<>::create(gko_device_exec, gko::dim<2>{n,n})); 
		// data_mat_gpu->copy_from(data_mat);
        // auto data_mat_batch = gko::share(gko::matrix::BatchCsr<>::create(gko_device_exec, n_batch, data_mat.get()));
        Kokkos::View<double*, Kokkos::HostSpace> x_cpu("x_cpu", n);
        Kokkos::View<double*, Kokkos::DefaultExecutionSpace> x_gpu("x_gpu", n);
		Kokkos::deep_copy(x_gpu, x_cpu);
        auto x_vec_batch = to_gko_vec(x_gpu.data(), n);
        // auto x_vec_batch = gko::matrix::BatchDense<>::create(gko_device_exec, n_batch, x_vec.get());
		// auto x_vec_batch = gko::matrix::Dense<>::create(gko_device_exec, gko::dim<2>{n,n_batch});
		// x_vec_batch->fill(1e3);
		
		// Create the solver
		# if 1 // matrix-matrix linear system
		std::shared_ptr<gko::log::Stream<>> stream_logger =
		  gko::log::Stream<>::create(
			  gko::log::Logger::all_events_mask ^
				  gko::log::Logger::linop_factory_events_mask ^
				  gko::log::Logger::polymorphic_object_events_mask,
			  std::cout);
		// gko_device_exec->add_logger(stream_logger);
		std::shared_ptr<gko::stop::ResidualNorm<>::Factory> residual_criterion =
			gko::stop::ResidualNorm<>::build()
				.with_reduction_factor(1e-20)
				.on(gko_device_exec);
		std::shared_ptr<const gko::log::Convergence<>> convergence_logger = gko::log::Convergence<>::create(gko_device_exec);
		residual_criterion->add_logger(convergence_logger);
		auto solver =
			gko::solver::Bicgstab<>::build()
				.with_preconditioner(gko::preconditioner::Jacobi<>::build()
					.with_max_block_size(1u)
					.on(gko_device_exec))
				.with_criteria(
					residual_criterion,
					gko::stop::Iteration::build().with_max_iters(1000u).on(gko_device_exec))
				.on(gko_device_exec);
		auto solver_ = solver->generate(data_mat_gpu);
	    // solver_->add_logger(stream_logger);
		auto res_logger = std::make_shared<ResidualLogger<double>>(data_mat_gpu.get(), b_vec_batch.get());
		solver_->add_logger(res_logger);
	    solver_->apply(b_vec_batch, x_vec_batch);
		res_logger->write_data(std::cout);
		# else // full batched
		auto solver =
		gko::solver::BatchBicgstab<>::build()
			.with_default_max_iterations(500)
            .with_default_residual_tol(1e-15)
	        .with_tolerance_type(gko::stop::batch::ToleranceType::relative)
			.on(gko_device_exec);
		solver->generate(data_mat_batch)->apply(b_vec_batch.get(), x_vec_batch.get());
		#endif

		# if 1 
		// Write result
		std::cout << "-----------------------";
		write(std::cout, data_mat);
		write(std::cout, data_mat_gpu);
		std::cout << "-----------------------";
		write(std::cout, b_vec_batch);
		std::cout << "-----------------------";
		write(std::cout, x_vec_batch);


		#endif
		# if 0
		// Calculate residual
		auto err = gko::clone(gko_device_exec, b_vec_batch);
		// auto one = gko::batch_initialize<gko::matrix::BatchDense<>>(n_batch, {1.0}, gko_device_exec);
		auto one = gko::initialize<gko::matrix::Dense<>>({1.0}, gko_device_exec);
		// auto neg_one = gko::batch_initialize<gko::matrix::BatchDense<>>(n_batch, {-1.0}, gko_device_exec);
		auto neg_one = gko::initialize<gko::matrix::Dense<>>({-1.0}, gko_device_exec);
		// auto err_norms = gko::matrix::BatchDense<>::create(gko_device_exec->get_master(), gko::batch_dim<>(n_batch,gko::dim<2>{1,1}));
		auto err_norms = gko::matrix::Dense<>::create(gko_device_exec->get_master(), gko::dim<2>{1,n_batch});
		// data_mat_batch->apply(one.get(), x_vec_batch.get(), neg_one.get(), err.get());
		data_mat_gpu->apply(one, x_vec_batch, neg_one, err);
		err->compute_norm2(err_norms);
		// auto unb_err_norms = err_norms->unbatch();

		std::cout << "-----------------------";
		std::cout << "Residual norms sqrt(r^T r):\n";
		for (int i = 0; i < n_batch; ++i) {
			// std::cout << unb_err_norms[i]->at(0,0) << "\n";
			std::cout << err_norms->at(0,0) << "\n";
		}

		#endif
		
		Kokkos::deep_copy(x_cpu, x_gpu);
		Kokkos::deep_copy(b_cpu, x_cpu); //inplace temporary trick
		return 1;
    }
};
