// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <string>
#include <type_traits>
#include <utility>

#include <Kokkos_Core.hpp>

#include "ddc/ddc_to_kokkos_execution_policy.hpp"
#include "ddc/detail/kokkos.hpp"
#include "ddc/discrete_domain.hpp"
#include "ddc/discrete_element.hpp"
#include "ddc/reducer.hpp"

namespace ddc {

namespace detail {

template <class Reducer>
struct ddc_to_kokkos_reducer;

template <class T>
struct ddc_to_kokkos_reducer<reducer::sum<T>>
{
    using type = Kokkos::Sum<T>;
};

template <class T>
struct ddc_to_kokkos_reducer<reducer::prod<T>>
{
    using type = Kokkos::Prod<T>;
};

template <class T>
struct ddc_to_kokkos_reducer<reducer::land<T>>
{
    using type = Kokkos::LAnd<T>;
};

template <class T>
struct ddc_to_kokkos_reducer<reducer::lor<T>>
{
    using type = Kokkos::LOr<T>;
};

template <class T>
struct ddc_to_kokkos_reducer<reducer::band<T>>
{
    using type = Kokkos::BAnd<T>;
};

template <class T>
struct ddc_to_kokkos_reducer<reducer::bor<T>>
{
    using type = Kokkos::BOr<T>;
};

template <class T>
struct ddc_to_kokkos_reducer<reducer::bxor<T>>
{
    static_assert(std::is_same_v<T, T>, "This reducer is not yet implemented");
};

template <class T>
struct ddc_to_kokkos_reducer<reducer::min<T>>
{
    using type = Kokkos::Min<T>;
};

template <class T>
struct ddc_to_kokkos_reducer<reducer::max<T>>
{
    using type = Kokkos::Max<T>;
};

template <class T>
struct ddc_to_kokkos_reducer<reducer::minmax<T>>
{
    using type = Kokkos::MinMax<T>;
};

/// Alias template to transform a DDC reducer type to a Kokkos reducer type
template <class Reducer>
using ddc_to_kokkos_reducer_t = typename ddc_to_kokkos_reducer<Reducer>::type;

template <class Reducer, class Functor, class... DDims>
class TransformReducerKokkosLambdaAdapter
{
    template <class T>
    using index_type = DiscreteElementType;

    Reducer reducer;

    Functor functor;

public:
    TransformReducerKokkosLambdaAdapter(Reducer const& r, Functor const& f) : reducer(r), functor(f)
    {
    }

    template <std::size_t N = sizeof...(DDims), std::enable_if_t<(N == 0), bool> = true>
    KOKKOS_FUNCTION void operator()(
            [[maybe_unused]] index_type<void> unused_id,
            typename Reducer::value_type& a) const

    {
        a = reducer(a, functor(DiscreteElement<>()));
    }

    template <std::size_t N = sizeof...(DDims), std::enable_if_t<(N > 0), bool> = true>
    KOKKOS_FUNCTION void operator()(index_type<DDims>... ids, typename Reducer::value_type& a) const
    {
        a = reducer(a, functor(DiscreteElement<DDims...>(ids...)));
    }
};

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
template <class ExecSpace, class... DDims, class T, class BinaryReductionOp, class UnaryTransformOp>
T transform_reduce_kokkos(
        std::string const& label,
        ExecSpace const& execution_space,
        DiscreteDomain<DDims...> const& domain,
        T neutral,
        BinaryReductionOp const& reduce,
        UnaryTransformOp const& transform) noexcept
{
    T result = neutral;
    Kokkos::parallel_reduce(
            label,
            ddc_to_kokkos_execution_policy(execution_space, domain),
            TransformReducerKokkosLambdaAdapter<
                    BinaryReductionOp,
                    UnaryTransformOp,
                    DDims...>(reduce, transform),
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
template <class ExecSpace, class... DDims, class T, class BinaryReductionOp, class UnaryTransformOp>
T parallel_transform_reduce(
        std::string const& label,
        ExecSpace const& execution_space,
        DiscreteDomain<DDims...> const& domain,
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
template <class ExecSpace, class... DDims, class T, class BinaryReductionOp, class UnaryTransformOp>
std::enable_if_t<Kokkos::is_execution_space_v<ExecSpace>, T> parallel_transform_reduce(
        ExecSpace const& execution_space,
        DiscreteDomain<DDims...> const& domain,
        T neutral,
        BinaryReductionOp&& reduce,
        UnaryTransformOp&& transform) noexcept
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
template <class... DDims, class T, class BinaryReductionOp, class UnaryTransformOp>
T parallel_transform_reduce(
        std::string const& label,
        DiscreteDomain<DDims...> const& domain,
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
template <class... DDims, class T, class BinaryReductionOp, class UnaryTransformOp>
T parallel_transform_reduce(
        DiscreteDomain<DDims...> const& domain,
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

} // namespace ddc
