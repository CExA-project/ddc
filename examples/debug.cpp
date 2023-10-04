// SPDX-License-Identifier: MIT

//! [includes]
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>

#include <ddc/ddc.hpp>

#include <Kokkos_Core.hpp>
#include "ddc/for_each.hpp"
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
    // Start of the domain of interest in the Y dimension
    double const y_start = -1.;
    // End of the domain of interest in the Y dimension
    double const y_end = 1.;
    // Number of discretization points in the Y dimension
    size_t const nb_y_points = 100;
     //! [parameters]

	 //! [X-global-domain]
    // Initialization of the global domain in X with gwx ghost points on
    // each side
    auto const x_domain
            = ddc::init_discrete_space(DDimX::init(
                    ddc::Coordinate<X>(x_start),
                    ddc::Coordinate<X>(x_end),
                    ddc::DiscreteVector<DDimX>(nb_x_points)
                    ));
    //! [X-global-domain]
	//
    //! [Y-domains]
        auto const y_domain
            = ddc::init_discrete_space(DDimY::init(
                    ddc::Coordinate<Y>(y_start),
                    ddc::Coordinate<Y>(y_end),
                    ddc::DiscreteVector<DDimY>(nb_y_points)
                    ));
    //! [data allocation]
    // Maps temperature into the full domain (including ghosts) twice:
    // - once for the last fully computed time-step
    ddc::Chunk chunk(
            ddc::DiscreteDomain<
                    DDimX,
                    DDimY>(x_domain, y_domain),
            ddc::DeviceAllocator<double>());
	ddc::ChunkSpan chunkspan = chunk.span_view();
    // Initialize the temperature on the main domain
    ddc::for_each(
            ddc::policies::parallel_host,
            x_domain,
            DDC_LAMBDA(ddc::DiscreteElement<DDimX> const ix) {
				printf("----- DEBUG LOG -----");
				auto slice = chunkspan[ix];
				printf("size = %i", slice.size());
            });
    //! [initial-conditions]
}
