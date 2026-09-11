// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>

#include <ddc/ddc.hpp>

#include <Kokkos_Core.hpp>

namespace ddc::detail {

template <std::size_t N, class ElementType, bool CONTIGUOUS = true>
struct ViewNDMaker
{
};

template <std::size_t N, class ElementType>
struct ViewNDMaker<N, ElementType, true>
{
    using type
            = Kokkos::mdspan<ElementType, Kokkos::dextents<std::size_t, N>, Kokkos::layout_right>;
};

template <std::size_t N, class ElementType>
struct ViewNDMaker<N, ElementType, false>
{
    using type
            = Kokkos::mdspan<ElementType, Kokkos::dextents<std::size_t, N>, Kokkos::layout_stride>;
};

} // namespace ddc::detail

namespace ddc {

#if DDC_BUILD_DEPRECATED_CODE()
template <std::size_t N, class ElementType>
using SpanND [[deprecated]] = Kokkos::mdspan<ElementType, Kokkos::dextents<std::size_t, N>>;

template <std::size_t N, class ElementType>
using ViewND [[deprecated]] = Kokkos::mdspan<ElementType const, Kokkos::dextents<std::size_t, N>>;

template <class ElementType>
using Span1D [[deprecated]] = Kokkos::mdspan<ElementType, Kokkos::dextents<std::size_t, 1>>;

template <class ElementType>
using Span2D [[deprecated]] = Kokkos::mdspan<ElementType, Kokkos::dextents<std::size_t, 2>>;

template <class ElementType>
using View1D [[deprecated]] = Kokkos::mdspan<ElementType const, Kokkos::dextents<std::size_t, 1>>;

template <class ElementType>
using View2D [[deprecated]] = Kokkos::mdspan<ElementType const, Kokkos::dextents<std::size_t, 2>>;

using DSpan1D [[deprecated]] = Kokkos::mdspan<double, Kokkos::dextents<std::size_t, 1>>;

using DSpan2D [[deprecated]] = Kokkos::mdspan<double, Kokkos::dextents<std::size_t, 2>>;

using CDSpan1D [[deprecated]] = Kokkos::mdspan<double const, Kokkos::dextents<std::size_t, 1>>;

using CDSpan2D [[deprecated]] = Kokkos::mdspan<double const, Kokkos::dextents<std::size_t, 2>>;

using DView1D [[deprecated]] = Kokkos::mdspan<double const, Kokkos::dextents<std::size_t, 1>>;

using DView2D [[deprecated]] = Kokkos::mdspan<double const, Kokkos::dextents<std::size_t, 2>>;
#endif

} // namespace ddc
