// SPDX-License-Identifier: MIT

#include <cmath>
#include <iostream>

#include <ddc/Chunk>
#include <ddc/DiscreteCoordinate>
#include <ddc/DiscreteDomain>
#include <ddc/PdiEvent>
#include <ddc/UniformDiscretization>
#include <ddc/for_each>

using cell = bool;

// Name of the axis
struct X;
struct Y;

using DDimX = UniformDiscretization<X>;
using DDimY = UniformDiscretization<Y>;

static unsigned nt = 10;
static unsigned length = 5;
static unsigned height = 5;

void blinker_init(DiscreteDomain<DDimX, DDimY> const& domain, Chunk<cell, DiscreteDomain<DDimX, DDimY>>& cells) {
    for_each(domain, [&](DiscreteCoordinate<DDimX, DDimY> const ixy) {
        double const x = to_real(select<DDimX>(ixy));
        double const y = to_real(select<DDimY>(ixy));
        if (y == 2 && ((x-3) * (x-1)<=0)) cells(ixy) = true;
        else cells(ixy) = false;
    });
}

template <class ElementType, class DDimX, class DDimY>
std::ostream& print_2DChunk(std::ostream& os, DiscreteDomain<DDimX, DDimY> const& domain, Chunk<ElementType, DiscreteDomain<DDimX, DDimY>>& chunk, int length) {
    for_each(domain, [&](DiscreteCoordinate<DDimX, DDimY> const ixy) {
        DiscreteCoordinate<DDimX> const ix = select<DDimX>(ixy);
        DiscreteCoordinate<DDimY> const iy = select<DDimY>(ixy);
        os << chunk(iy, ix);
        if (iy >= length - 1) os << "\n";
    });
    return os;
}

int main()
{
    // Origin on X
    Coordinate<X> const min_x(0.);

    // Sampling step on X
    Coordinate<X> const dx(1);

    // Actual mesh on X
    init_discretization<DDimX>(min_x, dx);

    // Origin on Y
    Coordinate<Y> const min_y(0.);

    // Sampling step on Y
    Coordinate<Y> const dy(1);

    // Actual mesh on Y
    init_discretization<DDimY>(min_y, dy);

    // Two-dimensional mesh on X,Y

    // Take (nx+2gw) x (ny+2gw) points of `mesh_xy` starting from (0,0)
    DiscreteDomain<DDimX, DDimY> const domain_xy(
            DiscreteVector<DDimX, DDimY>(length, height));

    Chunk<cell, DiscreteDomain<DDimX, DDimY>> cells_in(domain_xy);
    Chunk<cell, DiscreteDomain<DDimX, DDimY>> cells_out(domain_xy);

    // Initialize the whole domain
    blinker_init(domain_xy, cells_in);
    blinker_init(domain_xy, cells_out);

    for_each(domain_xy, [&](DiscreteCoordinate<DDimX, DDimY> const ixy) {
        DiscreteCoordinate<DDimX> const ix = select<DDimX>(ixy);
        DiscreteCoordinate<DDimY> const iy = select<DDimY>(ixy);
        //std::cout << ix << ", " << iy << ": " << cells_in(ix, iy) << "\n";
    }); std::cout << "\n\n";

    std::size_t iter = 0;
    for (; iter < nt; ++iter) {
        print_2DChunk(std::cout, domain_xy, cells_in, length) << "\n";
        for_each(domain_xy, [&](DiscreteCoordinate<DDimX, DDimY> const ixy) {
            DiscreteCoordinate<DDimX> const ix = select<DDimX>(ixy);
            DiscreteCoordinate<DDimY> const iy = select<DDimY>(ixy);
            int alive_neighbors = 0;
            for (int i = -1; i < 2; ++i) {
                for (int j = -1; j < 2; j++) {
                    if (cells_in(ix + i, iy + j) == true) alive_neighbors++;
                }
            }
            if (cells_in(ix, iy) == true) {
                alive_neighbors--;
                if (alive_neighbors < 2 || alive_neighbors > 3) cells_out(ix, iy) = false;
            }
            else {
                if (alive_neighbors == 3) cells_out(ix, iy) = true;
            }
        });
        deepcopy(cells_in, cells_out);
    }
    print_2DChunk(std::cout, domain_xy, cells_in, length) << "\n";

    return 0;
}
