// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <cassert>
#include <cstddef>
#include <stdexcept>
#include <string>

#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>

#if __has_include(<mkl_lapacke.h>)
#    include <mkl_lapacke.h>
#else
#    include <lapacke.h>
#endif

#include <KokkosBatched_Util.hpp>

#include "kokkos-kernels-ext/KokkosBatched_Getrs.hpp"

#include "splines_linear_problem.hpp"
#include "splines_linear_problem_dense.hpp"

namespace ddc::detail {

template <class ExecSpace>
SplinesLinearProblemDense<ExecSpace>::SplinesLinearProblemDense(std::size_t const mat_size)
    : SplinesLinearProblem<ExecSpace>(mat_size)
    , m_a("a", mat_size, mat_size)
    , m_ipiv("ipiv", mat_size)
{
    Kokkos::deep_copy(m_a.view_host(), 0.);
}

template <class ExecSpace>
SplinesLinearProblemDense<ExecSpace>::~SplinesLinearProblemDense() = default;

template <class ExecSpace>
double SplinesLinearProblemDense<ExecSpace>::get_element(std::size_t const i, std::size_t const j)
        const
{
    assert(i < size());
    assert(j < size());
    return m_a.view_host()(i, j);
}

template <class ExecSpace>
void SplinesLinearProblemDense<ExecSpace>::set_element(
        std::size_t const i,
        std::size_t const j,
        double const aij)
{
    assert(i < size());
    assert(j < size());
    m_a.view_host()(i, j) = aij;
}

template <class ExecSpace>
void SplinesLinearProblemDense<ExecSpace>::setup_solver()
{
    int const info = LAPACKE_dgetrf(
            LAPACK_ROW_MAJOR,
            size(),
            size(),
            m_a.view_host().data(),
            size(),
            m_ipiv.view_host().data());
    if (info != 0) {
        throw std::runtime_error("LAPACKE_dgetrf failed with error code " + std::to_string(info));
    }

    // Convert 1-based index to 0-based index
    for (std::size_t i = 0; i < size(); ++i) {
        m_ipiv.view_host()(i) -= 1;
    }

    // Push on device
    m_a.modify_host();
    m_a.sync_device();
    m_ipiv.modify_host();
    m_ipiv.sync_device();
}

template <class ExecSpace>
void SplinesLinearProblemDense<ExecSpace>::solve(MultiRHS const b, bool const transpose) const
{
    assert(b.extent(0) == size());

    // For order 1 splines, size() can be 0 then we bypass the solver call.
    if (size() == 0) {
        return;
    }

    auto a_device = m_a.view_device();
    auto ipiv_device = m_ipiv.view_device();

    Kokkos::RangePolicy<ExecSpace> const policy(0, b.extent(1));

    if (transpose) {
        Kokkos::parallel_for(
                "gerts",
                policy,
                KOKKOS_LAMBDA(int const i) {
                    auto sub_b = Kokkos::subview(b, Kokkos::ALL, i);
                    KokkosBatched::SerialGetrs<
                            KokkosBatched::Trans::Transpose,
                            KokkosBatched::Algo::Level3::Unblocked>::
                            invoke(a_device, ipiv_device, sub_b);
                });
    } else {
        Kokkos::parallel_for(
                "gerts",
                policy,
                KOKKOS_LAMBDA(int const i) {
                    auto sub_b = Kokkos::subview(b, Kokkos::ALL, i);
                    KokkosBatched::SerialGetrs<
                            KokkosBatched::Trans::NoTranspose,
                            KokkosBatched::Algo::Level3::Unblocked>::
                            invoke(a_device, ipiv_device, sub_b);
                });
    }
}

#if defined(KOKKOS_ENABLE_SERIAL)
template class SplinesLinearProblemDense<Kokkos::Serial>;
#endif
#if defined(KOKKOS_ENABLE_OPENMP)
template class SplinesLinearProblemDense<Kokkos::OpenMP>;
#endif
#if defined(KOKKOS_ENABLE_CUDA)
template class SplinesLinearProblemDense<Kokkos::Cuda>;
#endif
#if defined(KOKKOS_ENABLE_HIP)
template class SplinesLinearProblemDense<Kokkos::HIP>;
#endif
#if defined(KOKKOS_ENABLE_SYCL)
template class SplinesLinearProblemDense<Kokkos::SYCL>;
#endif

} // namespace ddc::detail
