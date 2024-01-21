#pragma once

namespace ddc {
struct NullExtrapolationRule
{
    explicit NullExtrapolationRule() {}

    template <class CoordType, class... BSplines, class Layout, class MemorySpace>
    KOKKOS_FUNCTION double operator()(
            CoordType,
            ddc::ChunkSpan<const double, ddc::DiscreteDomain<BSplines...>, Layout, MemorySpace>)
            const
    {
        return 0.0;
    }
};
} // namespace ddc
