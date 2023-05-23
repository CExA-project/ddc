// SPDX-License-Identifier: MIT

//! [includes]
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>

#include <ddc/ddc.hpp>

#include <kernels/fft.hpp>

#include <Kokkos_Core.hpp>
//! [includes]

#define FINITE_DIFF 0
#define SPECTRAL 1

#define METHOD SPECTRAL // FINITE_DIFF or SPECTRAL

//! [X-dimension]
/// Our first continuous dimension
struct X;
//! [X-dimension]

//! [X-discretization]
/// A uniform discretization of X
using DDimX = ddc::UniformPointSampling<X>;
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
                std::cout << std::setw(6) << temp_slice(ix);
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
    size_t const nb_x_points = 10;
    // Thermal diffusion coefficient
    double const kx = .01;
    // Start of the domain of interest in the Y dimension
    double const y_start = -1.;
    // End of the domain of interest in the Y dimension
    double const y_end = 1.;
    // Number of discretization points in the Y dimension
    size_t const nb_y_points = 100;
    // Thermal diffusion coefficient
    double const ky = .002;
    // Simulated time at which to start simulation
    double const start_time = 0.;
    // Simulated time to reach as target of the simulation
    double const end_time = 10.;
    // Number of time-steps between outputs
    ptrdiff_t const t_output_period = 10;
//! [parameters]

//! [main-start]
#if (METHOD == FINITE_DIFF)
    std::cout << "Using finite differences method \n";
#elif (METHOD == SPECTRAL)
    std::cout << "Using spectral method \n";
#endif

//! [X-parameters]
// Number of ghost points to use on each side in X
#if (METHOD == FINITE_DIFF)
    ddc::DiscreteVector<DDimX> static constexpr gwx {1};
#elif (METHOD == SPECTRAL)
    ddc::DiscreteVector<DDimX> static constexpr gwx {0};
#endif
    //! [X-parameters]

    //! [X-global-domain]
    // Initialization of the global domain in X with gwx ghost points on
    // each side
    auto const [x_domain, ghosted_x_domain, x_pre_ghost, x_post_ghost]
            = ddc::init_discrete_space(DDimX::init_ghosted(
                    ddc::Coordinate<X>(x_start),
                    ddc::Coordinate<X>(x_end),
                    ddc::DiscreteVector<DDimX>(nb_x_points),
                    gwx));
    //! [X-global-domain]

    //! [X-domains]
    // our zone at the start of the domain that will be mirrored to the
    // ghost
    ddc::DiscreteDomain const
            x_domain_begin(x_domain.front(), x_post_ghost.extents());
    // our zone at the end of the domain that will be mirrored to the
    // ghost
    ddc::DiscreteDomain const x_domain_end(
            x_domain.back() - x_pre_ghost.extents() + 1,
            x_pre_ghost.extents());
//! [X-domains]

//! [Y-domains]
// Number of ghost points to use on each side in Y
#if (METHOD == FINITE_DIFF)
    ddc::DiscreteVector<DDimY> static constexpr gwy {1};
#elif (METHOD == SPECTRAL)
    ddc::DiscreteVector<DDimY> static constexpr gwy {0};
#endif

    // Initialization of the global domain in Y with gwy ghost points on
    // each side
    auto const [y_domain, ghosted_y_domain, y_pre_ghost, y_post_ghost]
            = ddc::init_discrete_space(DDimY::init_ghosted(
                    ddc::Coordinate<Y>(y_start),
                    ddc::Coordinate<Y>(y_end),
                    ddc::DiscreteVector<DDimY>(nb_y_points),
                    gwy));

    // our zone at the start of the domain that will be mirrored to the
    // ghost
    ddc::DiscreteDomain const
            y_domain_begin(y_domain.front(), y_post_ghost.extents());
    // our zone at the end of the domain that will be mirrored to the
    // ghost
    ddc::DiscreteDomain const y_domain_end(
            y_domain.back() - y_pre_ghost.extents() + 1,
            y_pre_ghost.extents());
    //! [Y-domains]

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
	  2./Kokkos::pow(Kokkos::numbers::pi,2)
            / (kx * invdx2_max
               + ky * invdy2_max)}; // Classical stability theory gives .5 but empirically we see that for FFT method we need .2

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
    ddc::Chunk ghosted_last_temp(
            ddc::DiscreteDomain<
                    DDimX,
                    DDimY>(ghosted_x_domain, ghosted_y_domain),
            ddc::DeviceAllocator<double>());

    // - once for time-step being computed
    ddc::Chunk ghosted_next_temp(
            ddc::DiscreteDomain<
                    DDimX,
                    DDimY>(ghosted_x_domain, ghosted_y_domain),
            ddc::DeviceAllocator<double>());
    //! [data allocation]

    //! [initial-conditions]
    ddc::ChunkSpan const ghosted_initial_temp
            = ghosted_last_temp.span_view();
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
                ghosted_initial_temp(ixy)
                        = 9.999 * ((x * x + y * y) < 0.25);
            });
    //! [initial-conditions]

    ddc::Chunk ghosted_temp(
            ddc::DiscreteDomain<
                    DDimX,
                    DDimY>(ghosted_x_domain, ghosted_y_domain),
            ddc::HostAllocator<double>());


    //! [initial output]
    // display the initial data
    ddc::deepcopy(ghosted_temp, ghosted_last_temp);
    display(ddc::coordinate(time_domain.front()),
            ghosted_temp[x_domain][y_domain]);
    // time of the iteration where the last output happened
    ddc::DiscreteElement<DDimT> last_output = time_domain.front();
    //! [initial output]

#if (METHOD == SPECTRAL)
    ddc::DiscreteDomain<
            ddc::PeriodicSampling<Fourier<X>>,
            ddc::PeriodicSampling<Fourier<Y>>> const k_mesh
            = ddc::FourierMesh(ghosted_initial_temp.domain(), false);
    ddc::Chunk _Ff = ddc::
            Chunk(k_mesh, ddc::DeviceAllocator<Kokkos::complex<double>>());
    ddc::ChunkSpan Ff = _Ff.span_view();
#endif

    //! [time iteration]
    for (auto const iter :
         time_domain.remove_first(ddc::DiscreteVector<DDimT>(1))) {
        //! [time iteration]

        //! [boundary conditions]
        // Periodic boundary conditions
        ddc::deepcopy(
                ghosted_last_temp[x_pre_ghost][y_domain],
                ghosted_last_temp[y_domain][x_domain_end]);
        ddc::deepcopy(
                ghosted_last_temp[y_domain][x_post_ghost],
                ghosted_last_temp[y_domain][x_domain_begin]);
        ddc::deepcopy(
                ghosted_last_temp[x_domain][y_pre_ghost],
                ghosted_last_temp[x_domain][y_domain_end]);
        ddc::deepcopy(
                ghosted_last_temp[x_domain][y_post_ghost],
                ghosted_last_temp[x_domain][y_domain_begin]);
//! [boundary conditions]

//! [manipulated views]
// a span excluding ghosts of the temperature at the time-step we
// will build
#if (METHOD == FINITE_DIFF)
        ddc::ChunkSpan const next_temp {
                ghosted_next_temp[x_domain][y_domain]};
#elif (METHOD == SPECTRAL)
        ddc::ChunkSpan const next_temp {ghosted_next_temp.span_view()};
#endif
        // a read-only view of the temperature at the previous time-step
        ddc::ChunkSpan const last_temp {ghosted_last_temp.span_view()};
//! [manipulated views]

//! [numerical scheme]
// Stencil computation on the main domain
#if (METHOD == FINITE_DIFF)
        ddc::for_each(
                ddc::policies::parallel_device,
                next_temp.domain(),
                DDC_LAMBDA(
                        ddc::DiscreteElement<DDimX, DDimY> const ixy) {
                    ddc::DiscreteElement<DDimX> const ix
                            = ddc::select<DDimX>(ixy);
                    ddc::DiscreteElement<DDimY> const iy
                            = ddc::select<DDimY>(ixy);
                    double const dx_l = ddc::distance_at_left(ix);
                    double const dx_r = ddc::distance_at_right(ix);
                    double const dx_m = 0.5 * (dx_l + dx_r);
                    double const dy_l = ddc::distance_at_left(iy);
                    double const dy_r = ddc::distance_at_right(iy);
                    double const dy_m = 0.5 * (dy_l + dy_r);
                    next_temp(ix, iy) = last_temp(ix, iy);
                    next_temp(ix, iy)
                            += kx * max_dt
                               * (dx_l * last_temp(ix + 1, iy)
                                  - 2.0 * dx_m * last_temp(ix, iy)
                                  + dx_r * last_temp(ix - 1, iy))
                               / (dx_l * dx_m * dx_r);
                    next_temp(ix, iy)
                            += ky * max_dt
                               * (dy_l * last_temp(ix, iy + 1)
                                  - 2.0 * dy_m * last_temp(ix, iy)
                                  + dy_r * last_temp(ix, iy - 1))
                               / (dy_l * dy_m * dy_r);
                });
#elif (METHOD == SPECTRAL)
		ddc::FFT_Normalization norm = ddc::FFT_Normalization::BACKWARD;
        ddc::
                fft(Kokkos::DefaultExecutionSpace(),
                    Ff,
                    last_temp,
                    {norm});
        ddc::for_each(
                ddc::policies::parallel_device,
                k_mesh,
                DDC_LAMBDA(ddc::DiscreteElement<
                           ddc::PeriodicSampling<Fourier<X>>,
                           ddc::PeriodicSampling<Fourier<Y>>> const ikxky) {
                    ddc::DiscreteElement<
                            ddc::PeriodicSampling<Fourier<X>>> const ikx
                            = ddc::select<ddc::PeriodicSampling<Fourier<X>>>(
                                    ikxky);
                    ddc::DiscreteElement<
                            ddc::PeriodicSampling<Fourier<Y>>> const iky
                            = ddc::select<ddc::PeriodicSampling<Fourier<Y>>>(
                                    ikxky);
                    Ff(ikx, iky) = Ff(ikx, iky)*(1
                                    - (coordinate(ikx) * coordinate(ikx)
                                               * kx
                                       + coordinate(iky)
                                                 * coordinate(iky) * ky)
                                              * max_dt); // Ff(t+dt) = (1-D*k^2*dt)*Ff(t)
                });
        ddc::
                ifft(Kokkos::DefaultExecutionSpace(),
                    next_temp,
                    Ff,
                    {norm});
#endif
        //! [numerical scheme]

        //! [output]
        if (iter - last_output >= t_output_period) {
            last_output = iter;
            ddc::deepcopy(ghosted_temp, ghosted_last_temp);
            display(ddc::coordinate(iter),
                    ghosted_temp[x_domain][y_domain]);
        }
        //! [output]

        //! [swap]
        // Swap our two buffers
        std::swap(ghosted_last_temp, ghosted_next_temp);
        //! [swap]
    }

    //! [final output]
    if (last_output < time_domain.back()) {
        ddc::deepcopy(ghosted_temp, ghosted_last_temp);
        display(ddc::coordinate(time_domain.back()),
                ghosted_temp[x_domain][y_domain]);
    }
    //! [final output]
}
