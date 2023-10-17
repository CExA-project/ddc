// SPDX-License-Identifier: MIT

//! [includes]
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>

#include <ddc/ddc.hpp>
#include <ddc/kernels/splines/bsplines_uniform.hpp>
#include <ddc/kernels/splines/greville_interpolation_points.hpp>
#include <ddc/kernels/splines/spline_builder_batched.hpp>
#include <ddc/kernels/splines/spline_evaluator_batched.hpp>
#include <ddc/kernels/splines/null_boundary_value.hpp>

#include <Kokkos_Core.hpp>
//! [includes]
static constexpr std::size_t s_degree_x = 3;

//! [X-dimension]
/// Our first continuous dimension
struct X
{
      static constexpr bool PERIODIC = true;
};
//! [X-dimension]

//! [X-discretization]
/// A uniform discretization of X
using BSplinesX = ddc::UniformBSplines<X, s_degree_x>;

using DDimX = ddc::GrevilleInterpolationPoints<BSplinesX, ddc::BoundCond::PERIODIC, ddc::BoundCond::PERIODIC>::interpolation_mesh_type;
//! [X-discretization]

//! [Y-space]
// Our second continuous dimension
struct Y;
// Its uniform discretization
using DDimY = ddc::UniformPointSampling<Y>;
//! [Y-space]

//! [time-space]
// Our simulated time dimension
struct T;
// Its uniform discretization
using DDimT = ddc::UniformPointSampling<T>;
//! [time-space]

//! [display]
/** A function to pretty print the temperature
 * @param time the time at which the output is made
 * @param temp the temperature at this time-step
 */
template <class ChunkType>
void display(double time, ChunkType temp)
{
    double const mean_temp = ddc::transform_reduce(
                                     temp.domain(),
                                     0.,
                                     ddc::reducer::sum<double>(),
                                     temp)
                             / temp.domain().size();
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "At t = " << time << ",\n";
    std::cout << "  * mean temperature  = " << mean_temp << "\n";
    // take a slice in the middle of the box
    ddc::ChunkSpan temp_slice
            = temp[ddc::get_domain<DDimY>(temp).front()
                   + ddc::get_domain<DDimY>(temp).size() / 2];
    std::cout << "  * temperature[y:"
              << ddc::get_domain<DDimY>(temp).size() / 2 << "] = {";
    ddc::for_each(
            ddc::policies::serial_host,
            ddc::get_domain<DDimX>(temp),
            [=](ddc::DiscreteElement<DDimX> const ix) {
                std::cout << std::setw(6) << temp_slice(ix) << " ";
            });
    std::cout << " }" << std::endl;
}
//! [display]

//! [main-start]
int main(int argc, char** argv)
{
    ddc::ScopeGuard scope(argc, argv);

    // some parameters that would typically be read from some form of
    // configuration file in a more realistic code

    //! [parameters]
    // Start of the domain of interest in the X dimension
    double const x_start = -1.;
    // End of the domain of interest in the X dimension
    double const x_end = 1.;
    // Number of discretization points in the X dimension
    size_t const nb_x_points = 100;
    // Velocity along x dimension
    double const vx = .2;
    // Start of the domain of interest in the Y dimension
    double const y_start = -1.;
    // End of the domain of interest in the Y dimension
    double const y_end = 1.;
    // Number of discretization points in the Y dimension
    size_t const nb_y_points = 100;
    // Simulated time at which to start simulation
    double const start_time = 0.;
    // Simulated time to reach as target of the simulation
    double const end_time = 10.;
    // Number of time-steps between outputs
    ptrdiff_t const t_output_period = 10;
    //! [parameters]

    //! [main-start]
    std::cout << "Using spectral method \n";

    //! [X-global-domain]
    // Initialization of the global domain in X with gwx ghost points on
    // each side
	/*
    auto const x_domain = ddc::init_discrete_space(
            DDimX::
                    init(ddc::Coordinate<X>(x_start),
                         ddc::Coordinate<X>(x_end),
                         ddc::DiscreteVector<DDimX>(nb_x_points)));
						 */
	ddc::init_discrete_space<BSplinesX>(ddc::Coordinate<X>(x_start),
                         ddc::Coordinate<X>(x_end),nb_x_points);
	ddc::init_discrete_space<DDimX>(ddc::GrevilleInterpolationPoints<BSplinesX, ddc::BoundCond::PERIODIC, ddc::BoundCond::PERIODIC>::get_sampling());

    //! [X-global-domain]
	auto const x_domain = ddc::GrevilleInterpolationPoints<BSplinesX, ddc::BoundCond::PERIODIC, ddc::BoundCond::PERIODIC>::get_domain(); 
    // Initialization of the global domain in Y with gwy ghost points on
    // each side
    auto const y_domain = ddc::init_discrete_space(
            DDimY::
                    init(ddc::Coordinate<Y>(y_start),
                         ddc::Coordinate<Y>(y_end),
                         ddc::DiscreteVector<DDimY>(nb_y_points)));

    //! [time-domains]
    // max(1/dx^2)
    double const invdx2_max = ddc::transform_reduce(
            x_domain,
            0.,
            ddc::reducer::max<double>(),
            [](ddc::DiscreteElement<DDimX> ix) {
                return 1.
                       / (ddc::distance_at_left(ix)
                          * ddc::distance_at_right(ix));
            });
    // max(1/dy^2)
    double const invdy2_max = ddc::transform_reduce(
            y_domain,
            0.,
            ddc::reducer::max<double>(),
            [](ddc::DiscreteElement<DDimY> iy) {
                return 1.
                       / (ddc::distance_at_left(iy)
                          * ddc::distance_at_right(iy));
            });
    ddc::Coordinate<T> const max_dt {
            0.1}; // Classical stability theory gives .5 but empirically we see that for FFT method we need .2

    // number of time intervals required to reach the end time
    ddc::DiscreteVector<DDimT> const nb_time_steps {
            std::ceil((end_time - start_time) / max_dt) + .2};
    // Initialization of the global domain in time:
    // - the number of discrete time-points is equal to the number of
    //   steps + 1
    ddc::DiscreteDomain<DDimT> const time_domain
            = ddc::init_discrete_space(
                    DDimT::
                            init(ddc::Coordinate<T>(start_time),
                                 ddc::Coordinate<T>(end_time),
                                 nb_time_steps + 1));
    //! [time-domains]

    //! [data allocation]
    // Maps temperature into the full domain (including ghosts) twice:
    // - once for the last fully computed time-step
    ddc::Chunk _last_temp(
            ddc::DiscreteDomain<DDimX, DDimY>(x_domain, y_domain),
            ddc::DeviceAllocator<double>());

    // - once for time-step being computed
    ddc::Chunk _next_temp(
            ddc::DiscreteDomain<DDimX, DDimY>(x_domain, y_domain),
            ddc::DeviceAllocator<double>());
    //! [data allocation]

    //! [initial-conditions]
    ddc::ChunkSpan const initial_temp = _last_temp.span_view();
    // Initialize the temperature on the main domain
    ddc::DiscreteDomain<DDimX, DDimY> x_mesh
            = ddc::DiscreteDomain<DDimX, DDimY>(x_domain, y_domain);
    ddc::for_each(
            ddc::policies::parallel_device,
            x_mesh,
            DDC_LAMBDA(ddc::DiscreteElement<DDimX, DDimY> const ixy) {
                double const x
                        = ddc::coordinate(ddc::select<DDimX>(ixy));
                double const y
                        = ddc::coordinate(ddc::select<DDimY>(ixy));
                // initial_temp(ixy) = 9.999 * Kokkos::exp(-(x * x + y * y) / 0.1 / 2);
                initial_temp(ixy) = 9.999 * ((x * x + y * y) < 0.25);
            });
    //! [initial-conditions]

    ddc::Chunk _host_temp(
            ddc::DiscreteDomain<DDimX, DDimY>(x_domain, y_domain),
            ddc::HostAllocator<double>());


    //! [initial output]
    // display the initial data
    ddc::deepcopy(_host_temp, _last_temp);
    display(ddc::coordinate(time_domain.front()),
            _host_temp[x_domain][y_domain]);
    // time of the iteration where the last output happened
    ddc::DiscreteElement<DDimT> last_output = time_domain.front();
    //! [initial output]

	// instantiate solver
	ddc::SplineBuilderBatched<
            ddc::SplineBuilder<
                    Kokkos::DefaultExecutionSpace,
                    Kokkos::DefaultExecutionSpace::memory_space,
                    BSplinesX,
                    DDimX,
                    ddc::BoundCond::PERIODIC,
                    ddc::BoundCond::PERIODIC>,
            DDimX,DDimY>
            spline_builder(x_mesh);	
	ddc::SplineEvaluatorBatched<
            ddc::SplineEvaluator<Kokkos::DefaultExecutionSpace, Kokkos::DefaultExecutionSpace::memory_space, BSplinesX, DDimX>,
            DDimX, DDimY>
            spline_evaluator(
                    spline_builder.spline_domain(),
                    ddc::g_null_boundary<BSplinesX>,
                    ddc::g_null_boundary<BSplinesX>);

	// Instantiate chunk of spline coefs to receive output of spline_builder
    ddc::Chunk coef_alloc(spline_builder.spline_domain(), ddc::KokkosAllocator<double, Kokkos::DefaultExecutionSpace::memory_space>());
    ddc::ChunkSpan coef = coef_alloc.span_view();

	// Find the coordinates of the characteristics feet
    ddc::Chunk feet_coords_alloc(spline_builder.vals_domain(), ddc::KokkosAllocator<ddc::Coordinate<X,Y>, Kokkos::DefaultExecutionSpace::memory_space>());
    ddc::ChunkSpan feet_coords = feet_coords_alloc.span_view();


    //! [time iteration]
    for (auto const iter :
        time_domain.remove_first(ddc::DiscreteVector<DDimT>(1))) {
        //! [time iteration]

        //! [boundary conditions]
        //! [boundary conditions]

        //! [manipulated views]
        // a span excluding ghosts of the temperature at the time-step we
        // will build
        ddc::ChunkSpan const next_temp {_next_temp.span_view()};
        // a read-only view of the temperature at the previous time-step
        ddc::ChunkSpan const last_temp {_last_temp.span_view()};
        //! [manipulated views]

        //! [numerical scheme]
        // Stencil computation on the main domain
		ddc::for_each(
		ddc::policies::parallel_device,
		feet_coords.domain(),
		DDC_LAMBDA (ddc::DiscreteElement<DDimX,DDimY> const e) {
		  feet_coords(e) = ddc::Coordinate<X,Y>(ddc::coordinate(ddc::select<DDimX>(e)) - ddc::Coordinate<X>(vx*ddc::step<DDimT>()),ddc::coordinate(ddc::select<DDimY>(e)));
		});
		spline_builder(coef,last_temp);
		spline_evaluator(next_temp,feet_coords.span_cview(),coef.span_cview());
        //! [numerical scheme]

        //! [output]
        if (iter - last_output >= t_output_period) {
            last_output = iter;
            ddc::deepcopy(_host_temp, _last_temp);
            display(ddc::coordinate(iter),
                    _host_temp[x_domain][y_domain]);
        }
        //! [output]

        //! [swap]
        // Swap our two buffers
        std::swap(_last_temp, _next_temp);
        //! [swap]
    }

    //! [final output]
    if (last_output < time_domain.back()) {
        ddc::deepcopy(_host_temp, _last_temp);
        display(ddc::coordinate(time_domain.back()),
                _host_temp[x_domain][y_domain]);
    }
    //! [final output]
}
