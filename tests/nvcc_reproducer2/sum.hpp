// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <ddc/ddc.hpp>

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

template <class Reducer>
struct ddc_to_kokkos_reducer;

template <class T>
struct ddc_to_kokkos_reducer<reducer_sum<T>>
{
    using type = Kokkos::Sum<T>;
};

template <class Reducer>
using ddc_to_kokkos_reducer_t = typename ddc_to_kokkos_reducer<Reducer>::type;

template <class Reducer, class Functor, class IndexSequence>
class TransformReducerKokkosLambdaAdapter;

template <class Reducer, class Functor, std::size_t... Idx>
class TransformReducerKokkosLambdaAdapter<Reducer, Functor, std::index_sequence<Idx...>>
{
    template <std::size_t I>
    using index_type = std::size_t;

    Reducer reducer;

    Functor functor;

public:
    TransformReducerKokkosLambdaAdapter(Reducer const& r, Functor const& f) : reducer(r), functor(f)
    {
    }


    template <std::size_t N = sizeof...(Idx), std::enable_if_t<(N == 0), bool> = true>
    KOKKOS_FUNCTION void operator()(
            [[maybe_unused]] index_type<0> unused_id,
            typename Reducer::value_type& a) const
    {
        a = reducer(a, functor());
    }

    template <std::size_t N = sizeof...(Idx), std::enable_if_t<(N > 0), bool> = true>
    KOKKOS_FUNCTION void operator()(index_type<Idx>... ids, typename Reducer::value_type& a) const
    {
        a = reducer(a, functor(ids...));
    }
};

template <class ExecSpace, class T, class BinaryReductionOp, class UnaryTransformOp>
T transform_reduce_kokkos(
        std::string const& label,
        ExecSpace const& execution_space,
        int n,
        T neutral,
        BinaryReductionOp const& reduce,
        UnaryTransformOp const& transform) noexcept
{
    T result = neutral;
    Kokkos::parallel_reduce(
            label,
            Kokkos::RangePolicy<ExecSpace, void, std::size_t>(execution_space, 0, n),
            TransformReducerKokkosLambdaAdapter<
                    BinaryReductionOp,
                    UnaryTransformOp,
                    std::make_index_sequence<1>>(reduce, transform),
            ddc_to_kokkos_reducer_t<BinaryReductionOp>(result));
    return result;
}

inline int sum(Kokkos::View<const int*, Kokkos::LayoutRight> const& view)
{
    return transform_reduce_kokkos(
            "test_parallel_transform_reduce_default",
            Kokkos::DefaultExecutionSpace(),
            view.extent(0),
            0,
            reducer_sum<int>(),
            KOKKOS_LAMBDA(int const ix) { return view(ix); });
}
