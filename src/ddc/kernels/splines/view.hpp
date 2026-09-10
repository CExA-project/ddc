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
using [[deprecated]] SpanND = Kokkos::mdspan<ElementType, Kokkos::dextents<std::size_t, N>>;

template <std::size_t N, class ElementType>
using [[deprecated]] ViewND = SpanND<N, ElementType const>;

template <class ElementType>
using [[deprecated]] Span1D = SpanND<1, ElementType>;

template <class ElementType>
using [[deprecated]] Span2D = SpanND<2, ElementType>;

template <class ElementType>
using [[deprecated]] View1D = ViewND<1, ElementType>;

template <class ElementType>
using [[deprecated]] View2D = ViewND<2, ElementType>;

using [[deprecated]] DSpan1D = ddc::Span1D<double>;

using [[deprecated]] DSpan2D = ddc::Span2D<double>;

using [[deprecated]] CDSpan1D = ddc::Span1D<double const>;

using [[deprecated]] CDSpan2D = ddc::Span2D<double const>;

using [[deprecated]] DView1D = View1D<double>;

using [[deprecated]] DView2D = View2D<double>;

} // namespace ddc
