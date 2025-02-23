#pragma once

#include <ddc/ddc.hpp>

#include <gtest/gtest.h>

struct Vx
{
};

struct GridVx : ddc::UniformPointSampling<Vx>
{
};

template <class DDomQuadrature>
class Quadrature
{
private:
    using MemorySpace = Kokkos::DefaultExecutionSpace::memory_space;

    using DElemQuadrature = typename DDomQuadrature::discrete_element_type;

    using QuadConstChunkSpan
            = ddc::ChunkSpan<const double, DDomQuadrature, Kokkos::layout_right, MemorySpace>;

    QuadConstChunkSpan m_coefficients;

public:
    explicit Quadrature(QuadConstChunkSpan coeffs) : m_coefficients(coeffs) {}

    template <class ExecutionSpace, class IntegratorFunction>
    double operator()(ExecutionSpace exec_space, IntegratorFunction integrated_function) const
    {
        QuadConstChunkSpan const coeff_proxy = m_coefficients;

        return ddc::parallel_transform_reduce(
                exec_space,
                coeff_proxy.domain(),
                0.0,
                ddc::reducer::sum<double>(),
                KOKKOS_LAMBDA(DElemQuadrature const ix) {
                    return coeff_proxy(ix) * integrated_function(ix);
                });
    }
};
