// SPDX-License-Identifier: MIT

//! [includes]
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>

#include <ddc/ddc.hpp>
#include <ddc/experimental/discrete_set.hpp>
#include <ddc/experimental/uniform_mesh.hpp>

#include <Kokkos_Core.hpp>
//! [includes]


namespace ddce = ddc::experimental;

//! [X-dimension]
/// Our first continuous dimension
struct X
{
};
//! [X-dimension]

//! [X-discretization]
/// A uniform discretization of X
struct MeshX : ddce::UniformMesh<X>
{
};
//! [X-discretization]

//! [Y-space]
// Our second continuous dimension
struct Y
{
};
// Its uniform discretization
struct MeshY : ddce::UniformMesh<Y>
{
};
//! [Y-space]

//! [time-space]
// Our simulated time dimension
struct T
{
};
// Its uniform discretization
struct MeshT : ddce::UniformMesh<T>
{
};
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
            = temp[ddc::get_domain<ddce::Node<MeshY>>(temp).front()
                   + ddc::get_domain<ddce::Node<MeshY>>(temp).size()
                             / 2];
    std::cout << "  * temperature[y:"
              << ddc::get_domain<ddce::Node<MeshY>>(temp).size() / 2
              << "] = {";
    ddc::for_each(
            ddc::policies::serial_host,
            ddc::get_domain<ddce::Node<MeshX>>(temp),
            [=](ddc::DiscreteElement<ddce::Node<MeshX>> const ix) {
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
    //! [X-parameters]
    // Number of ghost points to use on each side in X
    ddc::DiscreteVector<ddce::Node<MeshX>> static constexpr gwx {1};
    //! [X-parameters]

    //! [X-global-domain]
    // Initialization of the global domain in X with gwx ghost points on
    // each side
    auto const [x_domain, ghosted_x_domain, x_pre_ghost, x_post_ghost]
            = ddce::init_discrete_set<MeshX>(
                    ddce::uniform_mesh_init_ghosted<MeshX>(
                            ddc::Coordinate<X>(x_start),
                            ddc::Coordinate<X>(x_end),
                            ddc::DiscreteVector<ddce::Node<MeshX>>(
                                    nb_x_points),
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
    ddc::DiscreteVector<ddce::Node<MeshY>> static constexpr gwy {1};

    // Initialization of the global domain in Y with gwy ghost points on
    // each side
    auto const [y_domain, ghosted_y_domain, y_pre_ghost, y_post_ghost]
            = ddce::init_discrete_set<MeshY>(
                    ddce::uniform_mesh_init_ghosted<MeshY>(
                            ddc::Coordinate<Y>(y_start),
                            ddc::Coordinate<Y>(y_end),
                            ddc::DiscreteVector<ddce::Node<MeshY>>(
                                    nb_y_points),
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
            [](ddc::DiscreteElement<ddce::Node<MeshX>> ix) {
                return 1.
                       / (ddce::distance_at_left(ix)
                          * ddce::distance_at_right(ix));
            });
    // max(1/dy^2)
    double const invdy2_max = ddc::transform_reduce(
            y_domain,
            0.,
            ddc::reducer::max<double>(),
            [](ddc::DiscreteElement<ddce::Node<MeshY>> iy) {
                return 1.
                       / (ddce::distance_at_left(iy)
                          * ddce::distance_at_right(iy));
            });
    ddc::Coordinate<T> const max_dt {
            .5 / (kx * invdx2_max + ky * invdy2_max)};

    // number of time intervals required to reach the end time
    ddc::DiscreteVector<ddce::Node<MeshT>> const nb_time_steps {
            std::ceil((end_time - start_time) / max_dt) + .2};
    // Initialization of the global domain in time:
    // - the number of discrete time-points is equal to the number of
    //   steps + 1
    ddc::DiscreteDomain<ddce::Node<MeshT>> const time_domain
            = ddce::init_discrete_set<MeshT>(
                    ddce::uniform_mesh_init<MeshT>(
                            ddc::Coordinate<T>(start_time),
                            ddc::Coordinate<T>(end_time),
                            nb_time_steps + 1));
    //! [time-domains]

    //! [data allocation]
    // Maps temperature into the full domain (including ghosts) twice:
    // - once for the last fully computed time-step
    ddc::Chunk ghosted_last_temp(
            ddc::DiscreteDomain<
                    ddce::Node<MeshX>,
                    ddce::Node<
                            MeshY>>(ghosted_x_domain, ghosted_y_domain),
            ddc::DeviceAllocator<double>());

    // - once for time-step being computed
    ddc::Chunk ghosted_next_temp(
            ddc::DiscreteDomain<
                    ddce::Node<MeshX>,
                    ddce::Node<
                            MeshY>>(ghosted_x_domain, ghosted_y_domain),
            ddc::DeviceAllocator<double>());
    //! [data allocation]

    //! [initial-conditions]
    ddc::ChunkSpan const ghosted_initial_temp
            = ghosted_last_temp.span_view();
    // Initialize the temperature on the main domain
    ddc::for_each(
            ddc::policies::parallel_device,
            ddc::DiscreteDomain<
                    ddce::Node<MeshX>,
                    ddce::Node<MeshY>>(x_domain, y_domain),
            DDC_LAMBDA(ddc::DiscreteElement<
                       ddce::Node<MeshX>,
                       ddce::Node<MeshY>> const ixy) {
                double const x = ddce::coordinate(
                        ddc::select<ddce::Node<MeshX>>(ixy));
                double const y = ddce::coordinate(
                        ddc::select<ddce::Node<MeshY>>(ixy));
                ghosted_initial_temp(ixy)
                        = 9.999 * ((x * x + y * y) < 0.25);
            });
    //! [initial-conditions]

    ddc::Chunk ghosted_temp(
            ddc::DiscreteDomain<
                    ddce::Node<MeshX>,
                    ddce::Node<
                            MeshY>>(ghosted_x_domain, ghosted_y_domain),
            ddc::HostAllocator<double>());


    //! [initial output]
    // display the initial data
    ddc::deepcopy(ghosted_temp, ghosted_last_temp);
    display(ddce::coordinate(time_domain.front()),
            ghosted_temp[x_domain][y_domain]);
    // time of the iteration where the last output happened
    ddc::DiscreteElement<ddce::Node<MeshT>> last_output
            = time_domain.front();
    //! [initial output]

    //! [time iteration]
    for (auto const iter : time_domain.remove_first(
                 ddc::DiscreteVector<ddce::Node<MeshT>>(1))) {
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
        ddc::ChunkSpan const next_temp {
                ghosted_next_temp[x_domain][y_domain]};
        // a read-only view of the temperature at the previous time-step
        ddc::ChunkSpan const last_temp {ghosted_last_temp.span_view()};
        //! [manipulated views]

        //! [numerical scheme]
        // Stencil computation on the main domain
        ddc::for_each(
                ddc::policies::parallel_device,
                next_temp.domain(),
                DDC_LAMBDA(ddc::DiscreteElement<
                           ddce::Node<MeshX>,
                           ddce::Node<MeshY>> const ixy) {
                    ddc::DiscreteElement<ddce::Node<MeshX>> const ix
                            = ddc::select<ddce::Node<MeshX>>(ixy);
                    ddc::DiscreteElement<ddce::Node<MeshY>> const iy
                            = ddc::select<ddce::Node<MeshY>>(ixy);
                    double const dx_l
                            = ddce::rlength(ddce::cell_left(ix));
                    double const dx_r
                            = ddce::rlength(ddce::cell_right(ix));
                    double const dx_m = 0.5 * (dx_l + dx_r);
                    double const dy_l
                            = ddce::rlength(ddce::cell_left(iy));
                    double const dy_r
                            = ddce::rlength(ddce::cell_right(iy));
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
        //! [numerical scheme]

        //! [output]
        if (iter - last_output >= t_output_period) {
            last_output = iter;
            ddc::deepcopy(ghosted_temp, ghosted_last_temp);
            display(ddce::coordinate(iter),
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
        display(ddce::coordinate(time_domain.back()),
                ghosted_temp[x_domain][y_domain]);
    }
    //! [final output]
}
