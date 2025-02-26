// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <ddc/ddc.hpp>

struct DDimX
{
};

template <class T>
struct reducer_sum
{
    using value_type = T;

    KOKKOS_FUNCTION constexpr value_type operator()(value_type const& lhs, value_type const& rhs)
            const noexcept
    {
        return lhs + rhs;
    }
};

inline int sum(Kokkos::View<const int*, Kokkos::LayoutRight> const& view)
{
    auto lam = KOKKOS_LAMBDA(int const ix)
    {
        return view(ix);
    };
    reducer_sum<int> reducer;
    int res;
    Kokkos::parallel_reduce(
            view.extent(0),
            KOKKOS_LAMBDA(int const ix, int& lsum) { lsum = reducer(lsum, lam(ix)); },
            Kokkos::Sum<int>(res));
    return res;
}
