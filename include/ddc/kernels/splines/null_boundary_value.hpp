#pragma once

#include "spline_boundary_value.hpp"

namespace ddc {
template <class BSplines>
class NullBoundaryValue : public SplineBoundaryValue<BSplines>
{
public:
    NullBoundaryValue() = default;

    ~NullBoundaryValue() override = default;

    template <class Layout, class MemorySpace>
    KOKKOS_FUNCTION double operator()(
            ddc::Coordinate<typename BSplines::tag_type>,
            ddc::ChunkSpan<const double, ddc::DiscreteDomain<BSplines>, Layout, MemorySpace>)
            const final
    {
        return 0.0;
    }
};

template <class BSplines>
inline NullBoundaryValue<BSplines> const g_null_boundary;
} // namespace ddc
