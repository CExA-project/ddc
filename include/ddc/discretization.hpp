// SPDX-License-Identifier: MIT

#pragma once

#include <optional>
#include <stdexcept>

#include <Kokkos_Core.hpp>
#if defined(__CUDACC__)
#include <cuda.h>
#endif

#include "ddc/discrete_coordinate.hpp"
#include "ddc/discrete_domain.hpp"

namespace detail {

// For now, in the future, this should be specialized by tag
template <class IDimImpl>
inline IDimImpl* discretization_host = nullptr;

#if defined(__CUDACC__)
template <class IDimImpl>
inline __device__ IDimImpl* discretization_device = nullptr;
#endif

template <class Tuple, std::size_t... Ids>
auto extract_after(Tuple&& t, std::index_sequence<Ids...>)
{
    return std::make_tuple(std::move(std::get<Ids + 1>(t))...);
}

} // namespace detail

template <class IDimImpl, class Arg>
Arg init_discretization(std::tuple<IDimImpl, Arg>&& a)
{
    using IDim = typename IDimImpl::ddim_type;
    using IDimImplHost = typename IDim::template Impl<Kokkos::HostSpace>;
    if (detail::discretization_host<IDimImplHost>) {
        throw std::runtime_error("Discretization function already initialized.");
    }
    IDimImplHost* const tmp_host = new IDimImplHost(std::move(std::get<0>(a)));
    detail::discretization_host<IDimImplHost> = tmp_host;
#if defined(__CUDACC__)
    using IDimImplDevice = typename IDim::template Impl<Kokkos::CudaSpace>;
    IDimImplDevice tmp_device(*tmp_host);
    IDimImplDevice* ptr_device;
    cudaMalloc(&ptr_device, sizeof(IDimImplDevice));
    cudaMemcpy((void*)ptr_device, &tmp_device, sizeof(IDimImplDevice), cudaMemcpyDefault);
    cudaMemcpyToSymbol(
            detail::discretization_device<IDimImplDevice>,
            &ptr_device,
            sizeof(IDimImplDevice*),
            0,
            cudaMemcpyDefault);
#endif
    return std::get<1>(a);
}

template <class IDimImpl, class... Args>
std::enable_if_t<2 <= sizeof...(Args), std::tuple<Args...>> init_discretization(
        std::tuple<IDimImpl, Args...>&& a)
{
    using IDim = typename IDimImpl::ddim_type;
    using IDimImplHost = typename IDim::template Impl<Kokkos::HostSpace>;
    if (detail::discretization_host<IDimImplHost>) {
        throw std::runtime_error("Discretization function already initialized.");
    }
    IDimImplHost* const tmp_host = new IDimImplHost(std::move(std::get<0>(a)));
    detail::discretization_host<IDimImplHost> = tmp_host;
#if defined(__CUDACC__)
    using IDimImplDevice = typename IDim::template Impl<Kokkos::CudaSpace>;
    IDimImplDevice tmp_device(*tmp_host);
    IDimImplDevice* ptr_device;
    cudaMalloc(&ptr_device, sizeof(IDimImplDevice));
    cudaMemcpy((void*)ptr_device, &tmp_device, sizeof(IDimImplDevice), cudaMemcpyDefault);
    cudaMemcpyToSymbol(
            detail::discretization_device<IDimImplDevice>,
            &ptr_device,
            sizeof(IDimImplDevice*),
            0,
            cudaMemcpyDefault);
#endif
    return detail::extract_after(std::move(a), std::index_sequence_for<Args...>());
}

template <class D, class... Args>
void init_discretization(Args&&... a)
{
    if (detail::discretization_host<std::remove_cv_t<std::remove_reference_t<D>>>) {
        throw std::runtime_error("Discretization function already initialized.");
    }
    detail::discretization_host<std::remove_cv_t<std::remove_reference_t<D>>> = new D(
            std::forward<Args>(a)...);
}

template <class IDim>
inline typename IDim::template Impl<Kokkos::HostSpace> const& discretization_host()
{
    return *detail::discretization_host<typename IDim::template Impl<Kokkos::HostSpace>>;
}

#if defined(__CUDACC__)
template <class IDim>
__device__ inline typename IDim::template Impl<Kokkos::CudaSpace> const& discretization_device()
{
    return *detail::discretization_device<typename IDim::template Impl<Kokkos::CudaSpace>>;
}
#endif
