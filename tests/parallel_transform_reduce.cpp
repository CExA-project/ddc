// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <ddc/ddc.hpp>

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

inline namespace anonymous_namespace_workaround_parallel_transform_reduce_cpp {

using DElem0D = ddc::DiscreteElement<>;
using DVect0D = ddc::DiscreteVector<>;
using DDom0D = ddc::DiscreteDomain<>;

struct DDimX
{
};
using DElemX = ddc::DiscreteElement<DDimX>;
using DVectX = ddc::DiscreteVector<DDimX>;
using DDomX = ddc::DiscreteDomain<DDimX>;

struct DDimY
{
};
using DElemY = ddc::DiscreteElement<DDimY>;
using DVectY = ddc::DiscreteVector<DDimY>;
using DDomY = ddc::DiscreteDomain<DDimY>;

using DElemXY = ddc::DiscreteElement<DDimX, DDimY>;
using DVectXY = ddc::DiscreteVector<DDimX, DDimY>;
using DDomXY = ddc::DiscreteDomain<DDimX, DDimY>;

DElemX constexpr lbound_x = ddc::init_trivial_half_bounded_space<DDimX>();
DVectX constexpr nelems_x(10);

DElemY constexpr lbound_y = ddc::init_trivial_half_bounded_space<DDimY>();
DVectY constexpr nelems_y(12);

DElemXY constexpr lbound_x_y(lbound_x, lbound_y);
DVectXY constexpr nelems_x_y(nelems_x, nelems_y);

} // namespace anonymous_namespace_workaround_parallel_transform_reduce_cpp

TEST(ParallelTransformReduceHost, ZeroDimension)
{
    DDom0D const dom;
    Kokkos::View<int, Kokkos::HostSpace> const storage("storage");
    ddc::ChunkSpan const view(storage, dom);

    ddc::parallel_fill(view, 1);
    int const sum = ddc::parallel_transform_reduce(
            Kokkos::DefaultHostExecutionSpace(),
            dom,
            0,
            ddc::reducer::sum<int>(),
            view.span_cview());
    EXPECT_EQ(sum, dom.size());
}

TEST(ParallelTransformReduceHost, OneDimension)
{
    DDomX const dom(lbound_x, nelems_x);
    Kokkos::View<int*, Kokkos::HostSpace> const storage("storage", dom.size());
    ddc::ChunkSpan const view(storage, dom);

    ddc::parallel_fill(view, 1);
    int const sum = ddc::parallel_transform_reduce(
            Kokkos::DefaultHostExecutionSpace(),
            dom,
            0,
            ddc::reducer::sum<int>(),
            view.span_cview());
    EXPECT_EQ(sum, dom.size());
}

TEST(ParallelTransformReduceHost, TwoDimensions)
{
    DDomXY const dom(lbound_x_y, nelems_x_y);
    Kokkos::View<int*, Kokkos::HostSpace> const storage("storage", dom.size());
    ddc::ChunkSpan const
            view(Kokkos::View<int**, Kokkos::HostSpace>(storage.data(), nelems_x, nelems_y), dom);

    ddc::parallel_fill(view, 1);
    int const sum = ddc::parallel_transform_reduce(
            Kokkos::DefaultHostExecutionSpace(),
            dom,
            0,
            ddc::reducer::sum<int>(),
            view.span_cview());
    EXPECT_EQ(sum, dom.size());
}

TEST(ParallelTransformReduceDevice, ZeroDimension)
{
    DDom0D const dom;
    Kokkos::View<int> const storage("storage");
    ddc::ChunkSpan const view(storage, dom);

    ddc::parallel_fill(view, 1);
    int const sum
            = ddc::parallel_transform_reduce(dom, 0, ddc::reducer::sum<int>(), view.span_cview());
    EXPECT_EQ(sum, dom.size());
}

TEST(ParallelTransformReduceDevice, OneDimension)
{
    DDomX const dom(lbound_x, nelems_x);
    Kokkos::View<int*> const storage("storage", dom.size());
    ddc::ChunkSpan const view(storage, dom);

    ddc::parallel_fill(view, 1);
    int const sum
            = ddc::parallel_transform_reduce(dom, 0, ddc::reducer::sum<int>(), view.span_cview());
    EXPECT_EQ(sum, dom.size());
}

TEST(ParallelTransformReduceDevice, TwoDimensions)
{
    DDomXY const dom(lbound_x_y, nelems_x_y);
    Kokkos::View<int*> const storage("storage", dom.size());
    ddc::ChunkSpan const view(Kokkos::View<int**>(storage.data(), nelems_x, nelems_y), dom);

    ddc::parallel_fill(view, 1);
    int const sum
            = ddc::parallel_transform_reduce(dom, 0, ddc::reducer::sum<int>(), view.span_cview());
    EXPECT_EQ(sum, dom.size());
}

namespace ddc::experimental {

template <class Reducer, class Functor, class Support, class DElem, class IndexSequence>
class TransformReducerKokkosLambdaAdapter
{
};

template <class Reducer, class Functor, class Support, class DElem, std::size_t... Idx>
class TransformReducerKokkosLambdaAdapter<
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
    TransformReducerKokkosLambdaAdapter(
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
            typename Reducer::value_type& a) const
        requires(sizeof...(Idx) == 0)
    {
        a = m_reducer(a, m_functor(m_delem, m_support(typename Support::discrete_vector_type())));
    }

    KOKKOS_FUNCTION void operator()(index_type<Idx>... ids, typename Reducer::value_type& a) const
        requires(sizeof...(Idx) > 0)
    {
        a = m_reducer(
                a,
                m_functor(m_delem, m_support(typename Support::discrete_vector_type(ids...))));
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
template <
        class ExecSpace,
        class Support,
        class Tout,
        class DDomOut,
        class LayoutOut,
        class MemorySpace,
        class BinaryReductionOp,
        class UnaryTransformOp>
void parallel_transform_reduce(
        std::string const& label,
        ExecSpace const& execution_space,
        Support const& domain,
        ChunkSpan<Tout, DDomOut, LayoutOut, MemorySpace> const& out,
        BinaryReductionOp const& reduce,
        UnaryTransformOp const& transform) noexcept
{
    assert(out.domain() == DDomOut(domain));

    auto ddom_interest = remove_dims_of(domain, out.domain());
    host_for_each(out.domain(), [&](typename DDomOut::discrete_element_type iout) {
        Kokkos::parallel_reduce(
                label,
                detail::ddc_to_kokkos_execution_policy(
                        execution_space,
                        detail::array(ddom_interest.extents())),
                TransformReducerKokkosLambdaAdapter<
                        BinaryReductionOp,
                        UnaryTransformOp,
                        decltype(ddom_interest),
                        typename DDomOut::discrete_element_type,
                        std::make_index_sequence<
                                decltype(ddom_interest)::
                                        rank()>>(reduce, transform, ddom_interest, iout),
                detail::ddc_to_kokkos_reducer_t<BinaryReductionOp>(
                        out[iout].allocation_kokkos_view()));
    });
}

template <
        typename Tin,
        typename Tout,
        typename DDomIn,
        typename DDomOut,
        typename ExecSpace,
        typename MemorySpace,
        typename LayoutIn,
        typename LayoutOut>
void sum(
        ExecSpace const& exec_space,
        ddc::ChunkSpan<Tout, DDomOut, LayoutOut, MemorySpace> const& out,
        ddc::ChunkSpan<Tin const, DDomIn, LayoutIn, MemorySpace> const& in)
{
    ddc::experimental::parallel_transform_reduce(
            "",
            exec_space,
            in.domain(),
            out.span_view(),
            ddc::reducer::sum<Tin>(),
            in.span_cview());
}

template <
        typename Tin,
        typename Tout,
        typename DDomIn,
        typename DDomOut,
        typename ExecSpace,
        typename MemorySpace,
        typename LayoutIn,
        typename LayoutOut>
void prod(
        ExecSpace const& exec_space,
        ddc::ChunkSpan<Tout, DDomOut, LayoutOut, MemorySpace> const& out,
        ddc::ChunkSpan<Tin const, DDomIn, LayoutIn, MemorySpace> const& in)
{
    ddc::experimental::parallel_transform_reduce(
            "",
            exec_space,
            in.domain(),
            out.span_view(),
            ddc::reducer::prod<Tin>(),
            in.span_cview());
}

} // namespace ddc::experimental

TEST(Algorithms, Sum)
{
    DDomX const dom_x(lbound_x, nelems_x);
    DDomXY const dom_x_y(lbound_x_y, nelems_x_y);

    Kokkos::DefaultExecutionSpace const exec_space;

    ddc::Chunk chunk(dom_x_y, ddc::DeviceAllocator<int>());
    ddc::parallel_fill(chunk, 1);

    ddc::Chunk chunk2(dom_x, ddc::DeviceAllocator<int>());
    ddc::parallel_fill(chunk2, 0);

    ddc::experimental::sum(exec_space, chunk2.span_view(), chunk.span_cview());

    EXPECT_EQ(chunk2.domain(), dom_x);
    for (DElemX const ix : chunk2.domain()) {
        EXPECT_EQ(chunk2(ix), nelems_y.value());
    }
    ddc::print(std::cout, chunk2.span_cview()) << '\n';
}

TEST(ParallelTransformReduce, Broadcast)
{
    DElem0D constexpr lbound_0d {};
    DVect0D constexpr nelems_0d {};
    DDom0D constexpr dom_0d(lbound_0d, nelems_0d);
    DDomX const dom_x(lbound_x, nelems_x);
    DDomXY const dom_x_y(lbound_x_y, nelems_x_y);

    Kokkos::DefaultExecutionSpace const exec_space;

    ddc::Chunk chunk_x_y(dom_x_y, ddc::DeviceAllocator<int>());
    ddc::parallel_fill(chunk_x_y, 0);

    ddc::Chunk chunk_x(dom_x, ddc::DeviceAllocator<int>());
    ddc::parallel_fill(chunk_x, 1);

    ddc::parallel_copy(
            Kokkos::DefaultExecutionSpace(),
            chunk_x_y.span_view(),
            chunk_x.span_cview());

    ddc::print(std::cout, chunk_x_y.span_cview()) << '\n';

    ddc::experimental::sum(exec_space, chunk_x.span_view(), chunk_x_y.span_cview());

    ddc::print(std::cout, chunk_x.span_cview()) << '\n';

    ddc::Chunk chunk0d(dom_0d, ddc::DeviceAllocator<int>());
    ddc::experimental::prod(exec_space, chunk0d.span_view(), chunk_x.span_cview());

    ddc::print(std::cout, chunk0d.span_cview()) << '\n';
}
