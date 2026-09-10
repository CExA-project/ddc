// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>

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

template <std::size_t N, class ElementType>
using SpanND [[deprecated]] = Kokkos::mdspan<ElementType, Kokkos::dextents<std::size_t, N>>;

template <std::size_t N, class ElementType>
using ViewND [[deprecated]] = SpanND<N, ElementType const>;

template <class ElementType>
using Span1D [[deprecated]] = SpanND<1, ElementType>;

template <class ElementType>
using Span2D [[deprecated]] = SpanND<2, ElementType>;

template <class ElementType>
using View1D [[deprecated]] = ViewND<1, ElementType>;

template <class ElementType>
using View2D [[deprecated]] = ViewND<2, ElementType>;

using DSpan1D [[deprecated]] = ddc::Span1D<double>;

using DSpan2D [[deprecated]] = ddc::Span2D<double>;

using CDSpan1D [[deprecated]] = ddc::Span1D<double const>;

using CDSpan2D [[deprecated]] = ddc::Span2D<double const>;

using DView1D [[deprecated]] = View1D<double>;

using DView2D [[deprecated]] = View2D<double>;

} // namespace ddc
