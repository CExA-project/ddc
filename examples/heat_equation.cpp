// SPDX-License-Identifier: MIT

//! [includes]
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>

#include <ddc/ddc.hpp>

#include <Kokkos_Core.hpp>
//! [includes]


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
    double const mean_temp
            = ddc::transform_reduce(temp.domain(), 0., ddc::reducer::sum<double>(), temp)
              / temp.domain().size();
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "At t = " << time << ",\n";
    std::cout << "  * mean temperature  = " << mean_temp << "\n";
    // take a slice in the middle of the box
    ddc::ChunkSpan temp_slice
            = temp[ddc::get_domain<DDimY>(temp).front() + ddc::get_domain<DDimY>(temp).size() / 2];
    std::cout << "  * temperature[y:" << ddc::get_domain<DDimY>(temp).size() / 2 << "] = {";
    ddc::for_each(ddc::get_domain<DDimX>(temp), [=](ddc::DiscreteElement<DDimX> const ix) {
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
    // Number of subdomains in the X dimension
    size_t const nb_x_subdomains = 2;
    // Thermal diffusion coefficient
    double const kx = .01;
    // Start of the domain of interest in the Y dimension
    double const y_start = -1.;
    // End of the domain of interest in the Y dimension
    double const y_end = 1.;
    // Number of discretization points in the Y dimension
    size_t const nb_y_points = 100;
    // Number of subdomains in the Y dimension
    size_t const nb_y_subdomains = 2;
    // Thermal diffusion coefficient
    double const ky = .002;
    // Simulated time at which to start simulation
    double const start_time = 0.;
    // Simulated time to reach as target of the simulation
    double const end_time = 10.;
    // Number of time-steps between outputs
    ptrdiff_t const t_output_period = 10;
    //! [parameters]

    //! [units]
    // our unit vector in X dimension
    static constexpr ddc::Coordinate<X> x(1);
    // our unit vector in Y dimension
    static constexpr ddc::Coordinate<Y> y(1);
    // our unit vector in time dimension
    static constexpr ddc::Coordinate<T> t(1);
    //! [units]

    //! [main-start]
    //! [X-parameters]
    // Number of halo points to use on each side in X for periodicity
    static constexpr ddc::DiscreteVector<DDimX> x_halo_width(1);
    //! [X-parameters]

    //! [X-global-domain]
    // Initialization of the global domain in X with x_halo_width halo points on each side
    auto const x_domain = ddc::init_discrete_space(
            DDimX::
                    init(x_start * x,
                         x_end * x,
                         ddc::DiscreteVector<DDimX>(nb_x_points) + 2 * x_halo_width));

    //! [X-global-domain]

    //! [Y-domains]
    // Number of halo points to use on each side in Y for periodicity
    static constexpr ddc::DiscreteVector<DDimY> y_halo_width(1);

    // Initialization of the global domain in Y with y_halo_width halo points on each side
    auto const y_domain = ddc::init_discrete_space(
            DDimY::
                    init(y_start * y,
                         y_end * y,
                         ddc::DiscreteVector<DDimY>(nb_y_points) + 2 * y_halo_width));

    //! [Y-domains]

    //! [time-domains]
    // max(1/dx^2)
    double const invdx2_max
            = ddc::transform_reduce(x_domain, 0., ddc::reducer::max<double>(), [](auto ix) {
                  return 1. / (ddc::distance_at_left(ix) * ddc::distance_at_right(ix));
              });
    // max(1/dy^2)
    double const invdy2_max
            = ddc::transform_reduce(y_domain, 0., ddc::reducer::max<double>(), [](auto iy) {
                  return 1. / (ddc::distance_at_left(iy) * ddc::distance_at_right(iy));
              });
    ddc::Coordinate<T> const max_dt = .5 * t / (kx * invdx2_max + ky * invdy2_max);

    // number of time intervals required to reach the end time
    ddc::DiscreteVector<DDimT> const nb_time_steps(
            std::ceil((end_time - start_time) / max_dt) + .2);
    // Initialization of the global domain in time:
    // - the number of discrete time-points is equal to the number of
    //   steps + 1
    auto const time_domain = ddc::init_discrete_space(DDimT::
                                                              init(ddc::Coordinate<T>(start_time),
                                                                   ddc::Coordinate<T>(end_time),
                                                                   nb_time_steps + 1));
    //! [time-domains]

    // we make the Cartesian product of our x and y domains and chunk it with halos
    auto haloed_domain = ddc::BlockChunking<2>(
            {nb_x_subdomains, nb_y_subdomains},
            {x_halo_width, y_halo_width})(x_domain, y_domain);

    // the domain widthout its halos
    // auto core_domain = haloed_domain.chunks.remove_first(x_halo_width, y_halo_width)
    //                            .remove_last(x_halo_width, y_halo_width);

    // the halos and boundary domains
    // auto left_halos = haloed_domain.chunks.keep_first(x_halo_width);
    // auto domain_left = core_domain.keep_last(x_halo_width);
    // auto right_halos = haloed_domain.chunks.keep_last(x_halo_width);
    // auto domain_right = core_domain.keep_first(x_halo_width);
    // auto top_halos = haloed_domain.chunks.keep_first(y_halo_width);
    // auto domain_top = core_domain.keep_last(y_halo_width);
    // auto bottom_halos = haloed_domain.chunks.keep_last(y_halo_width);
    // auto domain_bottom = core_domain.keep_first(y_halo_width);

    ddc::DirectDistribution distribution_policy;

    //! [data allocation]
    // Allocate two temperature buffers:
    // - once for the last fully computed time-step
    auto prev_temperature = ddc::distributed_field<double>(haloed_domain, distribution_policy);
    // - once for time-step being computed
    auto next_temperature = ddc::distributed_field<double>(haloed_domain, distribution_policy);
    //! [data allocation]

    //! [initial-conditions]
    // Initialize the temperature on the core domain to a circle centered on 0
    // auto const& initial_temperature = prev_temperature[core_domain]; //<=TODO
    auto const& initial_temperature = prev_temperature.span_view();
    ddc::for_each(
            initial_temperature.domain(),
            DDC_LAMBDA(auto const& pnt) {
                auto const& x = ddc::get<X>(ddc::coordinate(pnt));
                auto const& y = ddc::get<Y>(ddc::coordinate(pnt));
                initial_temperature(pnt) = 9.999 * ((x * x + y * y) < 0.25);
            });
    //! [initial-conditions]

    //! [time iteration]
    // the iteration in time, our first element in time is given by the initial conditions, no need
    // to compute it
    for (auto const iter : time_domain.remove_first(ddc::DiscreteVector<DDimT>(1))) {
        //! [time iteration]

        //! [boundary conditions]
        // option #1: sync the haloed domain from the core domain, auto-detect self assignment &
        //            don't copy those //<=TODO
        // prev_temperature = prev_temperature[core_domain]

        // option #2: manually sync each halo //<=TODO
        // ddc::deepcopy(prev_temperature[left_halos], prev_temperature[domain_right]);
        // ddc::deepcopy(prev_temperature[right_halos], prev_temperature[domain_left]);
        // ddc::deepcopy(prev_temperature[top_halos], prev_temperature[domain_bottom]);
        // ddc::deepcopy(prev_temperature[bottom_halos], prev_temperature[domain_top]);
        //! [boundary conditions]

        //! [manipulated views]
        // The core domain where we compute
        // auto const next_temperature_core = next_temperature[core_domain]; //<=TODO
        auto const next_temperature_core = next_temperature.span_view();
        // a read-only view of the temperature at the previous time-step
        auto const prev_temperature_ro = prev_temperature.span_cview();
        //! [manipulated views]

        //! [numerical scheme]
        // Stencil computation on the main domain
        ddc::for_each(
                next_temperature_core.domain(),
                DDC_LAMBDA(auto const& pnt) {
                    static constexpr auto const right = ddc::DiscreteVector<DDimX>(1);
                    static constexpr auto const left = ddc::DiscreteVector<DDimX>(-1);
                    static constexpr auto const up = ddc::DiscreteVector<DDimY>(1);
                    static constexpr auto const down = ddc::DiscreteVector<DDimY>(-1);
                    double const dx_l = ddc::distance_before<X>(pnt);
                    double const dx_r = ddc::distance_after<X>(pnt);
                    double const dx_m = 0.5 * (dx_l + dx_r);
                    double const dy_l = ddc::distance_before<Y>(pnt);
                    double const dy_r = ddc::distance_before<Y>(pnt);
                    double const dy_m = 0.5 * (dy_l + dy_r);
                    next_temperature_core(pnt) = prev_temperature_ro(pnt)
                                                 + (kx * ddc::step<DDimT>()
                                                    * (dx_l * prev_temperature_ro(pnt + right)
                                                       - 2.0 * dx_m * prev_temperature_ro(pnt)
                                                       + dx_r * prev_temperature_ro(pnt + left))
                                                    / (dx_l * dx_m * dx_r))
                                                 + (ky * ddc::step<DDimT>()
                                                    * (dy_l * prev_temperature_ro(pnt + up)
                                                       - 2.0 * dy_m * prev_temperature_ro(pnt)
                                                       + dy_r * prev_temperature_ro(pnt + down))
                                                    / (dy_l * dy_m * dy_r));
                });
        //! [numerical scheme]

        //! [output]
        // if (iter - last_output >= t_output_period) {
        //     last_output = iter;
        //     ddc::deepcopy(ghosted_temp, ghosted_last_temp);
        //     display(ddc::coordinate(iter),
        //             ghosted_temp[x_domain][y_domain]);
        // }
        //! [output]

        //! [swap]
        // Swap our two buffers
        // std::swap(prev_temperature, next_temperature); //<=TODO
        //! [swap]
    }

    //! [final output]
    // if (last_output < time_domain.back()) {
    //     ddc::deepcopy(temperature, next_temperature);
    //     display(ddc::coordinate(time_domain.back()),
    //             temperature[x_domain][y_domain]);
    // }
    //! [final output]
}
