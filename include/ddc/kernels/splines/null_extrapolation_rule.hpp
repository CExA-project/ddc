#pragma once

namespace ddc {
struct NullExtrapolationRule
{
    template <class BSplines, class Layout, class MemorySpace>
    KOKKOS_INLINE_FUNCTION double operator()(
            ddc::Coordinate<typename BSplines::tag_type>,
            ddc::ChunkSpan<const double, ddc::DiscreteDomain<BSplines>, Layout, MemorySpace>) const
    {
        return 0.0;
    }
};
} // namespace ddc
