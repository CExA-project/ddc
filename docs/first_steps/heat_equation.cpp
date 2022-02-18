#include <ddc/Chunk>
#include <ddc/DiscreteCoordinate>
#include <ddc/DiscreteDomain>
#include <ddc/PdiEvent>
#include <ddc/UniformDiscretization>
#include <ddc/NonUniformDiscretization>
#include <ddc/for_each>

#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>

// Name of the axis
struct X;
struct Y;

using DDimX = UniformDiscretization<X>;
using DDimY = NonUniformDiscretization<Y>;

static unsigned nt = 10;
static unsigned nx = 101;
static unsigned ny = 201;
static unsigned gw = 1;
static double kx = 100.;
static double ky = 1.;

constexpr char const* const PDI_CFG = R"PDI_CFG(
metadata:
  ghostwidth: int
  iter : int

data:
  temperature_extents: { type: array, subtype: int64, size: 2 }
  temperature:
    type: array
    subtype: double
    size: [ '$temperature_extents[0]', '$temperature_extents[1]' ]
    start: [ '$ghostwidth', '$ghostwidth' ]
    subsize: [ '$temperature_extents[0]-2*$ghostwidth', '$temperature_extents[1]-2*$ghostwidth' ]

plugins:
  decl_hdf5:
    - file: 'temperature_${iter:04}.h5'
      on_event: temperature
      collision_policy: replace_and_warn
      write: [temperature]
  trace: ~
)PDI_CFG";

using dual_view = Kokkos::DualView<double**>;
using device_view = dual_view::t_dev;
using host_view = dual_view::t_host;

int main()
{
    Kokkos::ScopeGuard guard;

    //! [mesh]
    // Origin on X
    Coordinate<X> const min_x(-1.);

    // Sampling step on X
    Coordinate<X> const dx(0.02);

    // Actual mesh on X
    DDimX discretization_x(min_x, dx);

    // Origin on Y
    Coordinate<Y> const min_y(-1.);

    // Sampling step on Y
    Coordinate<Y> const dy(0.01);

    std::vector<Coordinate<Y>> points;
    points.reserve(ny);
    for (std::size_t i = 0; i < ny; ++i) {
        points.push_back(min_y + i * dy);
    }

    // Actual mesh on Y
    DDimY discretization_y(points);

    // Two-dimensional mesh on X,Y
    //! [mesh]

    //! [domain]
    // Take (nx+2gw) x (ny+2gw) points of `mesh_xy` starting from (0,0)
    DiscreteDomain<DDimX, DDimY> const domain_xy(
            DiscreteVector<DDimX, DDimY>(nx + 2 * gw, ny + 2 * gw));

    // Take only the inner domain (i.e. without ghost zone)
    DiscreteDomain<DDimX, DDimY> const inner_xy(
            DiscreteCoordinate<DDimX, DDimY>(gw, gw),
            DiscreteVector<DDimX, DDimY>(nx, ny));
    //! [domain]

    // Allocate data located at each point of `domain_xy` (including ghost region)
    //! [memory allocation]
    // Using Kokkos to do the allocation
    device_view T_in_kokkos(
            "T_in",
            domain_xy.extent<DDimX>().value(),
            domain_xy.extent<DDimY>().value());
    device_view T_out_kokkos(
            "T_out",
            domain_xy.extent<DDimX>().value(),
            domain_xy.extent<DDimY>().value());
    host_view T_in_kokkos_host = Kokkos::create_mirror_view(T_in_kokkos);
    ChunkSpan T_in(T_in_kokkos, domain_xy);
    ChunkSpan T_out(T_out_kokkos, domain_xy);
    ChunkSpan T_in_host(T_in_kokkos_host, domain_xy);
    //! [memory allocation]

    //! [subdomains]
    // Ghost borders
    ChunkSpan const temperature_g_x_left = T_in[DiscreteCoordinate<DDimX>(gw - 1)];
    ChunkSpan const temperature_g_x_right = T_in[DiscreteCoordinate<DDimX>(nx + 2 * gw - 1)];
    ChunkSpan const temperature_g_y_left = T_in[DiscreteCoordinate<DDimY>(gw - 1)];
    ChunkSpan const temperature_g_y_right = T_in[DiscreteCoordinate<DDimY>(ny + 2 * gw - 1)];

    // Inner borders
    ChunkSpan const temperature_i_x_left = std::as_const(T_in)[DiscreteCoordinate<DDimX>(gw)];
    ChunkSpan const temperature_i_x_right = std::as_const(T_in)[DiscreteCoordinate<DDimX>(nx + gw)];
    ChunkSpan const temperature_i_y_left = std::as_const(T_in)[DiscreteCoordinate<DDimY>(gw)];
    ChunkSpan const temperature_i_y_right = std::as_const(T_in)[DiscreteCoordinate<DDimY>(ny + gw)];
    //! [subdomains]

    // Initialize the whole domain
    for_each(
            domain_xy,
            KOKKOS_LAMBDA(DiscreteCoordinate<DDimX, DDimY> const ixy) {
                double const x = discretization_x.to_real(select<DDimX>(ixy));
                double const y = discretization_y.to_real(select<DDimY>(ixy));
                T_in(ixy) = 0.75 * ((x * x + y * y) < 0.25);
            });

    PDI_init(PC_parse_string(PDI_CFG));
    PDI_expose("ghostwidth", &gw, PDI_OUT);

    double const cfl = 0.99;
    double const dt = 0.5 * cfl / (kx / (dx * dx) + ky / (dy * dy));
    double const Cx = kx * dt / (dx * dx);
    double const Cy = ky * dt / (dy * dy);
    std::size_t iter = 0;
    for (; iter < nt; ++iter) {
        // How to deal with PDI ?
        // Contrary to the pure CPU case, data may be ready but not in an accessible memory space
        //! [io/pdi]
        deepcopy(T_in_host, T_in);
        PdiEvent("temperature").with("iter", iter).and_with("temperature", T_in_host);
        //! [io/pdi]

        //! [numerical scheme]
        // Periodic boundary conditions
        deepcopy(temperature_g_x_left, temperature_i_x_right);
        deepcopy(temperature_g_x_right, temperature_i_x_left);
        deepcopy(temperature_g_y_left, temperature_i_y_right);
        deepcopy(temperature_g_y_right, temperature_i_y_left);

        // Stencil computation on inner domain `inner_xy`
        for_each(
                inner_xy,
                KOKKOS_LAMBDA(DiscreteCoordinate<DDimX, DDimY> const ixy) {
                    DiscreteCoordinate<DDimX> const ix = select<DDimX>(ixy);
                    DiscreteCoordinate<DDimY> const iy = select<DDimY>(ixy);
                    T_out(ix, iy) = T_in(ix, iy);
                    T_out(ix, iy)
                            += Cx * (T_in(ix + 1, iy) - 2.0 * T_in(ix, iy) + T_in(ix - 1, iy));
                    T_out(ix, iy)
                            += Cy * (T_in(ix, iy + 1) - 2.0 * T_in(ix, iy) + T_in(ix, iy - 1));
                });
        //! [numerical scheme]

        // Copy buf2 into buf1, a swap could also do the job
        deepcopy(T_in, T_out);
    }

    PdiEvent("temperature").with("iter", iter).and_with("temperature", T_in_host);

    PDI_finalize();

    return 0;
}
