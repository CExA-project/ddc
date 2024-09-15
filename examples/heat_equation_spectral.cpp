// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <string>
#include <utility>

#include <ddc/ddc.hpp>
#include <ddc/kernels/fft.hpp>

#include <Kokkos_Core.hpp>

/// Our first continuous dimension
struct X;

/// A uniform discretization of X
struct DDimX : ddc::UniformPointSampling<X>
{
};

struct DDimFx : ddc::PeriodicSampling<ddc::Fourier<X>>
{
};

// Our second continuous dimension
struct Y;
// Its uniform discretization
struct DDimY : ddc::UniformPointSampling<Y>
{
};

struct DDimFy : ddc::PeriodicSampling<ddc::Fourier<Y>>
{
};

// Our simulated time dimension
struct T;
// Its uniform discretization
struct DDimT : ddc::UniformPointSampling<T>
{
};

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
    ddc::ChunkSpan const temp_slice
            = temp[ddc::get_domain<DDimY>(temp).front()
                   + ddc::get_domain<DDimY>(temp).size() / 2];
    std::cout << "  * temperature[y:"
              << ddc::get_domain<DDimY>(temp).size() / 2 << "] = {";
    ddc::for_each(
            ddc::get_domain<DDimX>(temp),
            [=](ddc::DiscreteElement<DDimX> const ix) {
                std::cout << std::setw(6) << temp_slice(ix);
            });
    std::cout << " }" << std::endl;
}

int main(int argc, char** argv)
{
    Kokkos::ScopeGuard const kokkos_scope(argc, argv);
    ddc::ScopeGuard const ddc_scope(argc, argv);

    // some parameters that would typically be read from some form of
    // configuration file in a more realistic code

    // Start of the domain of interest in the X dimension
    double const x_start = -1.;
    // End of the domain of interest in the X dimension
    double const x_end = 1.;
    // Number of discretization points in the X dimension
    std::size_t const nb_x_points = 10;
    // Thermal diffusion coefficient
    double const kx = .01;
    // Start of the domain of interest in the Y dimension
    double const y_start = -1.;
    // End of the domain of interest in the Y dimension
    double const y_end = 1.;
    // Number of discretization points in the Y dimension
    std::size_t const nb_y_points = 100;
    // Thermal diffusion coefficient
    double const ky = .002;
    // Simulated time at which to start simulation
    double const start_time = 0.;
    // Simulated time to reach as target of the simulation
    double const end_time = 10.;
    // Number of time-steps between outputs
    std::ptrdiff_t const t_output_period = 10;

    // Initialization of the global domain in X including periodic point to have correct step
    auto const x_domain_with_periodic_point
            = ddc::init_discrete_space<DDimX>(DDimX::init<DDimX>(
                    ddc::Coordinate<X>(x_start),
                    ddc::Coordinate<X>(x_end),
                    ddc::DiscreteVector<DDimX>(nb_x_points)));
    ddc::DiscreteDomain<DDimX> const x_domain
            = x_domain_with_periodic_point.remove_last(
                    ddc::DiscreteVector<DDimX>(1));

    // Initialization of the global domain in Y including periodic point to have correct step
    auto const y_domain_with_periodic_point
            = ddc::init_discrete_space<DDimY>(DDimY::init<DDimY>(
                    ddc::Coordinate<Y>(y_start),
                    ddc::Coordinate<Y>(y_end),
                    ddc::DiscreteVector<DDimY>(nb_y_points)));
    ddc::DiscreteDomain<DDimY> const y_domain
            = y_domain_with_periodic_point.remove_last(
                    ddc::DiscreteVector<DDimY>(1));

    double const max_rkx = Kokkos::numbers::pi / ddc::step<DDimX>();
    double const max_rky = Kokkos::numbers::pi / ddc::step<DDimY>();
    ddc::Coordinate<T> const dt(
            2. / (kx * max_rkx * max_rkx + ky * max_rky * max_rky));

    // number of time intervals required to reach the end time
    ddc::DiscreteVector<DDimT> const nb_time_steps(
            std::ceil((end_time - start_time) / dt) + .2);
    // Initialization of the global domain in time:
    // - the number of discrete time-points is equal to the number of
    //   steps + 1
    ddc::DiscreteDomain<DDimT> const time_domain
            = ddc::init_discrete_space<DDimT>(DDimT::init<DDimT>(
                    ddc::Coordinate<T>(start_time),
                    ddc::Coordinate<T>(end_time),
                    nb_time_steps + 1));

    ddc::DiscreteDomain<DDimX, DDimY> const
            xy_domain(x_domain, y_domain);

    // Maps temperature into the full domain (including ghosts) twice:
    // - once for the last fully computed time-step
    ddc::Chunk _last_temp(
            "_last_temp",
            xy_domain,
            ddc::DeviceAllocator<double>());

    // - once for time-step being computed
    ddc::Chunk _next_temp(
            "_next_temp",
            xy_domain,
            ddc::DeviceAllocator<double>());

    ddc::ChunkSpan const initial_temp = _last_temp.span_view();
    // Initialize the temperature on the main domain
    ddc::parallel_for_each(
            xy_domain,
            KOKKOS_LAMBDA(ddc::DiscreteElement<DDimX, DDimY> const ixy) {
                double const x = ddc::coordinate(
                        ddc::DiscreteElement<DDimX>(ixy));
                double const y = ddc::coordinate(
                        ddc::DiscreteElement<DDimY>(ixy));
                initial_temp(ixy) = 9.999 * ((x * x + y * y) < 0.25);
            });

    ddc::Chunk _host_temp = ddc::create_mirror(_last_temp.span_cview());

    // display the initial data
    ddc::parallel_deepcopy(_host_temp, _last_temp);
    display(ddc::coordinate(time_domain.front()),
            _host_temp.span_cview());
    // time of the iteration where the last output happened
    ddc::DiscreteElement<DDimT> last_output = time_domain.front();

    ddc::init_discrete_space<DDimFx>(
            ddc::init_fourier_space<DDimFx>(x_domain));
    ddc::init_discrete_space<DDimFy>(
            ddc::init_fourier_space<DDimFy>(y_domain));

    ddc::DiscreteDomain<DDimFx, DDimFy> const k_mesh
            = ddc::FourierMesh<DDimFx, DDimFy>(xy_domain, false);
    ddc::Chunk Ff_allocation(
            "Ff_allocation",
            k_mesh,
            ddc::DeviceAllocator<Kokkos::complex<double>>());
    ddc::ChunkSpan const Ff = Ff_allocation.span_view();

    for (ddc::DiscreteElement<DDimT> const iter :
         time_domain.remove_first(ddc::DiscreteVector<DDimT>(1))) {
        // a span excluding ghosts of the temperature at the time-step we
        // will build
        ddc::ChunkSpan const next_temp = _next_temp.span_view();
        // a read-only view of the temperature at the previous time-step
        ddc::ChunkSpan const last_temp = _last_temp.span_view();

        // Stencil computation on the main domain
        ddc::kwArgs_fft const kwargs {ddc::FFT_Normalization::BACKWARD};
        Kokkos::DefaultExecutionSpace const execution_space;
        ddc::fft(execution_space, Ff, last_temp, kwargs);
        ddc::parallel_for_each(
                execution_space,
                k_mesh,
                KOKKOS_LAMBDA(ddc::DiscreteElement<DDimFx, DDimFy> const
                                      ikxky) {
                    ddc::DiscreteElement<DDimFx> const ikx(ikxky);
                    ddc::DiscreteElement<DDimFy> const iky(ikxky);
                    double const rkx = ddc::coordinate(ikx);
                    double const rky = ddc::coordinate(iky);
                    Ff(ikxky)
                            *= 1
                               - (kx * rkx * rkx + ky * rky * rky) * dt;
                });
        ddc::ifft(execution_space, next_temp, Ff, kwargs);

        if (iter - last_output >= t_output_period) {
            last_output = iter;
            ddc::parallel_deepcopy(_host_temp, _next_temp);
            display(ddc::coordinate(iter), _host_temp.span_cview());
        }

        // Swap our two buffers
        std::swap(_last_temp, _next_temp);
    }

    if (last_output < time_domain.back()) {
        ddc::parallel_deepcopy(_host_temp, _last_temp);
        display(ddc::coordinate(time_domain.back()),
                _host_temp.span_cview());
    }
}
