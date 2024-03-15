// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <ostream>
#include <utility>

#include <experimental/mdspan>

namespace ddc {

template <std::size_t N, class ElementType>
using SpanND = std::experimental::mdspan<ElementType, std::experimental::dextents<std::size_t, N>>;

template <std::size_t N, class ElementType>
using SpanND_left = std::experimental::mdspan<
        ElementType,
        std::experimental::dextents<std::size_t, N>,
        std::experimental::layout_left>;

template <std::size_t N, class ElementType>
using SpanND_stride = std::experimental::mdspan<
        ElementType,
        std::experimental::dextents<std::size_t, N>,
        std::experimental::layout_stride>;

template <std::size_t N, class ElementType>
using ViewND = SpanND<N, ElementType const>;

template <std::size_t N, class ElementType>
using ViewND_left = SpanND_left<N, ElementType const>;

template <std::size_t N, class ElementType>
using ViewND_stride = SpanND_stride<N, ElementType const>;

template <class ElementType>
using Span1D = SpanND<1, ElementType>;

template <class ElementType>
using Span2D = SpanND<2, ElementType>;

template <class ElementType>
using Span2D_left = SpanND_left<2, ElementType>;

template <class ElementType>
using Span2D_stride = SpanND_stride<2, ElementType>;

template <class ElementType>
using View1D = ViewND<1, ElementType>;

template <class ElementType>
using View2D = ViewND<2, ElementType>;

template <class ElementType>
using View2D_left = ViewND_left<2, ElementType>;

template <class ElementType>
using View2D_stride = ViewND_stride<2, ElementType>;

using DSpan1D = ddc::Span1D<double>;

using DSpan2D = ddc::Span2D<double>;

using DSpan2D_left = ddc::Span2D_left<double>;

using DSpan2D_stride = ddc::Span2D_stride<double>;

using DView1D = View1D<double>;

using DView2D = View2D<double>;

using DView2D_left = View2D_left<double>;

using DView2D_stride = View2D_stride<double>;

} // namespace ddc
