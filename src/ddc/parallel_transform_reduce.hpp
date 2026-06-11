// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <string>
#include <type_traits>
#include <utility>

#include <Kokkos_Core.hpp>

#include "chunk_traits.hpp"
#include "ddc_to_kokkos_execution_policy.hpp"
#include "discrete_vector.hpp"
#include "reducer.hpp"

namespace ddc {

namespace detail {

template <class Reducer, class MemorySpace>
struct DdcToKokkosReducer
{
};

template <class T, class MemorySpace>
struct DdcToKokkosReducer<reducer::sum<T>, MemorySpace>
{
    using type = Kokkos::Sum<T, MemorySpace>;
};

template <class T, class MemorySpace>
struct DdcToKokkosReducer<reducer::prod<T>, MemorySpace>
{
    using type = Kokkos::Prod<T, MemorySpace>;
};

template <class T, class MemorySpace>
struct DdcToKokkosReducer<reducer::land<T>, MemorySpace>
{
    using type = Kokkos::LAnd<T, MemorySpace>;
};

template <class T, class MemorySpace>
struct DdcToKokkosReducer<reducer::lor<T>, MemorySpace>
{
    using type = Kokkos::LOr<T, MemorySpace>;
};

template <class T, class MemorySpace>
struct DdcToKokkosReducer<reducer::band<T>, MemorySpace>
{
    using type = Kokkos::BAnd<T, MemorySpace>;
};

template <class T, class MemorySpace>
struct DdcToKokkosReducer<reducer::bor<T>, MemorySpace>
{
    using type = Kokkos::BOr<T, MemorySpace>;
};

template <class T, class MemorySpace>
struct DdcToKokkosReducer<reducer::bxor<T>, MemorySpace>
{
    static_assert(std::is_same_v<T, T>, "This reducer is not yet implemented");
};

template <class T, class MemorySpace>
struct DdcToKokkosReducer<reducer::min<T>, MemorySpace>
{
    using type = Kokkos::Min<T, MemorySpace>;
};

template <class T, class MemorySpace>
struct DdcToKokkosReducer<reducer::max<T>, MemorySpace>
{
    using type = Kokkos::Max<T, MemorySpace>;
};

template <class T, class MemorySpace>
struct DdcToKokkosReducer<reducer::minmax<T>, MemorySpace>
{
    using type = Kokkos::MinMax<T, MemorySpace>;
};

/// Alias template to transform a DDC reducer type to a Kokkos reducer type
template <class Reducer, class MemorySpace = Kokkos::HostSpace>
using ddc_to_kokkos_reducer_t = DdcToKokkosReducer<Reducer, MemorySpace>::type;

template <class Reducer, class Functor, class Support, class IndexSequence>
class TransformReducerKokkosLambdaAdapter
{
};

template <class Reducer, class Functor, class Support, std::size_t... Idx>
class TransformReducerKokkosLambdaAdapter<Reducer, Functor, Support, std::index_sequence<Idx...>>
{
    template <std::size_t I>
    using index_type = DiscreteVectorElement;

    Reducer m_reducer;

    Functor m_functor;

    Support m_support;

public:
    TransformReducerKokkosLambdaAdapter(Reducer const& r, Functor const& f, Support const& support)
        : m_reducer(r)
        , m_functor(f)
        , m_support(support)
    {
    }

    KOKKOS_FUNCTION void operator()(index_type<0> /*id*/, Reducer::value_type& a) const
        requires(sizeof...(Idx) == 0)
    {
        a = m_reducer(a, m_functor(m_support(typename Support::discrete_vector_type())));
    }

    KOKKOS_FUNCTION void operator()(index_type<Idx>... ids, Reducer::value_type& a) const
        requires(sizeof...(Idx) > 0)
    {
        a = m_reducer(a, m_functor(m_support(typename Support::discrete_vector_type(ids...))));
    }
};

template <class Reducer, class Functor, class Support>
TransformReducerKokkosLambdaAdapter(Reducer const& r, Functor const& f, Support const& support)
        -> TransformReducerKokkosLambdaAdapter<
                Reducer,
                Functor,
                Support,
                std::make_index_sequence<Support::rank()>>;

/** A parallel reduction over a nD domain using the default Kokkos execution space
 * @param[in] label  name for easy identification of the parallel_for_each algorithm
 * @param[in] execution_space a Kokkos execution space where the loop will be executed on
 * @param[in] domain the range over which to apply the algorithm
 * @param[in] neutral the neutral element of the reduction operation
 * @param[in] reduce a binary FunctionObject that will be applied in unspecified order to the
 *            results of transform, the results of other reduce and neutral.
 * @param[in] transform a unary FunctionObject that will be applied to each element of the input
 *            range. The return type must be acceptable as input to reduce
 */
template <class ExecSpace, class Support, class T, class BinaryReductionOp, class UnaryTransformOp>
T transform_reduce_kokkos(
        std::string const& label,
        ExecSpace const& execution_space,
        Support const& domain,
        T neutral,
        BinaryReductionOp const& reduce,
        UnaryTransformOp const& transform) noexcept
{
    T result = neutral;
    Kokkos::parallel_reduce(
            label,
            ddc_to_kokkos_execution_policy(execution_space, detail::array(domain.extents())),
            TransformReducerKokkosLambdaAdapter(reduce, transform, domain),
            ddc_to_kokkos_reducer_t<BinaryReductionOp>(result));
    return result;
}

} // namespace detail

/** A reduction over a nD domain using a given `Kokkos` execution space
 * @param[in] label  name for easy identification of the parallel_for_each algorithm
 * @param[in] execution_space a Kokkos execution space where the loop will be executed on
 * @param[in] domain the range over which to apply the algorithm
 * @param[in] neutral the neutral element of the reduction operation
 * @param[in] reduce a binary FunctionObject that will be applied in unspecified order to the
 *            results of transform, the results of other reduce and neutral.
 * @param[in] transform a unary FunctionObject that will be applied to each element of the input
 *            range. The return type must be acceptable as input to reduce
 */
template <class ExecSpace, class Support, class T, class BinaryReductionOp, class UnaryTransformOp>
T parallel_transform_reduce(
        std::string const& label,
        ExecSpace const& execution_space,
        Support const& domain,
        T neutral,
        BinaryReductionOp&& reduce,
        UnaryTransformOp&& transform) noexcept
{
    return detail::transform_reduce_kokkos(
            label,
            execution_space,
            domain,
            neutral,
            std::forward<BinaryReductionOp>(reduce),
            std::forward<UnaryTransformOp>(transform));
}

/** A reduction over a nD domain using a given `Kokkos` execution space
 * @param[in] execution_space a Kokkos execution space where the loop will be executed on
 * @param[in] domain the range over which to apply the algorithm
 * @param[in] neutral the neutral element of the reduction operation
 * @param[in] reduce a binary FunctionObject that will be applied in unspecified order to the
 *            results of transform, the results of other reduce and neutral.
 * @param[in] transform a unary FunctionObject that will be applied to each element of the input
 *            range. The return type must be acceptable as input to reduce
 */
template <class ExecSpace, class Support, class T, class BinaryReductionOp, class UnaryTransformOp>
T parallel_transform_reduce(
        ExecSpace const& execution_space,
        Support const& domain,
        T neutral,
        BinaryReductionOp&& reduce,
        UnaryTransformOp&& transform) noexcept
    requires(Kokkos::is_execution_space_v<ExecSpace>)
{
    return detail::transform_reduce_kokkos(
            "ddc_parallel_transform_reduce_default",
            execution_space,
            domain,
            neutral,
            std::forward<BinaryReductionOp>(reduce),
            std::forward<UnaryTransformOp>(transform));
}

/** A reduction over a nD domain using the `Kokkos` default execution space
 * @param[in] label  name for easy identification of the parallel_for_each algorithm
 * @param[in] domain the range over which to apply the algorithm
 * @param[in] neutral the neutral element of the reduction operation
 * @param[in] reduce a binary FunctionObject that will be applied in unspecified order to the
 *            results of transform, the results of other reduce and neutral.
 * @param[in] transform a unary FunctionObject that will be applied to each element of the input
 *            range. The return type must be acceptable as input to reduce
 */
template <class Support, class T, class BinaryReductionOp, class UnaryTransformOp>
T parallel_transform_reduce(
        std::string const& label,
        Support const& domain,
        T neutral,
        BinaryReductionOp&& reduce,
        UnaryTransformOp&& transform) noexcept
{
    return parallel_transform_reduce(
            label,
            Kokkos::DefaultExecutionSpace(),
            domain,
            neutral,
            std::forward<BinaryReductionOp>(reduce),
            std::forward<UnaryTransformOp>(transform));
}

/** A reduction over a nD domain using the `Kokkos` default execution space
 * @param[in] domain the range over which to apply the algorithm
 * @param[in] neutral the neutral element of the reduction operation
 * @param[in] reduce a binary FunctionObject that will be applied in unspecified order to the
 *            results of transform, the results of other reduce and neutral.
 * @param[in] transform a unary FunctionObject that will be applied to each element of the input
 *            range. The return type must be acceptable as input to reduce
 */
template <class Support, class T, class BinaryReductionOp, class UnaryTransformOp>
T parallel_transform_reduce(
        Support const& domain,
        T neutral,
        BinaryReductionOp&& reduce,
        UnaryTransformOp&& transform) noexcept
{
    return parallel_transform_reduce(
            "ddc_parallel_transform_reduce_default",
            Kokkos::DefaultExecutionSpace(),
            domain,
            neutral,
            std::forward<BinaryReductionOp>(reduce),
            std::forward<UnaryTransformOp>(transform));
}

namespace experimental {

namespace detail {

template <class Reducer, class Functor, class Support, class DElem, class IndexSequence>
class TransformReducerChunkKokkosLambdaAdapter
{
};

template <class Reducer, class Functor, class Support, class DElem, std::size_t... Idx>
class TransformReducerChunkKokkosLambdaAdapter<
        Reducer,
        Functor,
        Support,
        DElem,
        std::index_sequence<Idx...>>
{
    template <std::size_t I>
    using index_type = DiscreteVectorElement;

    Reducer m_reducer;

    Functor m_functor;

    Support m_support;

    DElem m_delem;

public:
    TransformReducerChunkKokkosLambdaAdapter(
            Reducer const& r,
            Functor const& f,
            Support const& support,
            DElem const& delem)
        : m_reducer(r)
        , m_functor(f)
        , m_support(support)
        , m_delem(delem)
    {
    }

    KOKKOS_FUNCTION void operator()(
            [[maybe_unused]] index_type<0> unused_id,
            Reducer::value_type& a) const
        requires(sizeof...(Idx) == 0)
    {
        a = m_reducer(a, m_functor(m_delem, m_support(typename Support::discrete_vector_type())));
    }

    KOKKOS_FUNCTION void operator()(index_type<Idx>... ids, Reducer::value_type& a) const
        requires(sizeof...(Idx) > 0)
    {
        a = m_reducer(
                a,
                m_functor(m_delem, m_support(typename Support::discrete_vector_type(ids...))));
    }
};

template <class Reducer, class Functor, class Support, class DElem, std::size_t... Idx>
TransformReducerChunkKokkosLambdaAdapter(
        Reducer const& r,
        Functor const& f,
        Support const& support,
        DElem const& delem)
        -> TransformReducerChunkKokkosLambdaAdapter<
                Reducer,
                Functor,
                Support,
                DElem,
                std::make_index_sequence<Support::rank()>>;

} // namespace detail

/** Performs a parallel transform-reduce over an nD domain using a Kokkos execution space.
 *
 * For each element of `out`, a reduction is performed over the dimensions of
 * `domain` that are not present in `out.domain()`. The reduction combines the
 * values obtained by applying `transform` to each element of the corresponding
 * subdomain.
 *
 * @param[in] label name used to identify the Kokkos kernel.
 * @param[in] execution_space Kokkos execution space on which the reductions are executed.
 * @param[in] domain full domain over which the transform-reduce is defined.
 * @param[out] out chunk receiving the reduction result for each point of
 *                 `out.domain()`. Its domain must be a subdomain of `domain`.
 * @param[in] reduce binary reduction operator used to combine transformed values.
 *                   It must be compatible with the value type stored in `out`.
 * @param[in] transform unary function applied to each element of the reduction
 *                      subdomain. Its return type must be accepted by `reduce`.
 */
template <
        class ExecSpace,
        class Support,
        concepts::borrowed_chunk ChunkDst,
        class BinaryReductionOp,
        class UnaryTransformOp>
void parallel_transform_reduce(
        std::string const& label,
        ExecSpace const& execution_space,
        Support const& domain,
        ChunkDst&& out,
        BinaryReductionOp const& reduce,
        UnaryTransformOp const& transform) noexcept
{
    using DDomOut = std::remove_cvref_t<ChunkDst>::discrete_domain_type;
    using DElemOut = DDomOut::discrete_element_type;
    using MemorySpaceOut = std::remove_cvref_t<ChunkDst>::memory_space;
    assert(out.domain() == DDomOut(domain));

    auto ddom_interest = remove_dims_of(domain, out.domain());
    host_for_each(out.domain(), [&](DElemOut iout) {
        Kokkos::parallel_reduce(
                label,
                ddc::detail::ddc_to_kokkos_execution_policy(
                        execution_space,
                        ddc::detail::array(ddom_interest.extents())),
                detail::TransformReducerChunkKokkosLambdaAdapter(
                        reduce,
                        transform,
                        ddom_interest,
                        iout),
                ddc::detail::ddc_to_kokkos_reducer_t<BinaryReductionOp, MemorySpaceOut>(
                        out[iout].allocation_kokkos_view()));
    });
}

} // namespace experimental

} // namespace ddc
