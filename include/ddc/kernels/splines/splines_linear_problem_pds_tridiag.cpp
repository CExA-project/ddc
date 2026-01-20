// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <cassert>
#if !defined(NDEBUG)
#    include <cmath>
#endif
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>

#if __has_include(<mkl_lapacke.h>)
#    include <mkl_lapacke.h>
#else
#    include <lapacke.h>
#endif

#include <KokkosBatched_Pttrs.hpp>
#include <KokkosBatched_Util.hpp>

#include "splines_linear_problem.hpp"
#include "splines_linear_problem_pds_tridiag.hpp"

namespace ddc::detail {

template <class ExecSpace>
SplinesLinearProblemPDSTridiag<ExecSpace>::SplinesLinearProblemPDSTridiag(
        std::size_t const mat_size)
    : SplinesLinearProblem<ExecSpace>(mat_size)
    , m_q("q", 2, mat_size)
{
    Kokkos::deep_copy(m_q.view_host(), 0.);
}

template <class ExecSpace>
SplinesLinearProblemPDSTridiag<ExecSpace>::~SplinesLinearProblemPDSTridiag() = default;

template <class ExecSpace>
double SplinesLinearProblemPDSTridiag<ExecSpace>::get_element(std::size_t i, std::size_t j) const
{
    assert(i < size());
    assert(j < size());

    // Indices are swapped for an element on subdiagonal
    if (i > j) {
        std::swap(i, j);
    }

    if (j - i < 2) {
        return m_q.view_host()(j - i, i);
    }

    return 0.0;
}

template <class ExecSpace>
void SplinesLinearProblemPDSTridiag<ExecSpace>::set_element(
        std::size_t i,
        std::size_t j,
        double const aij)
{
    assert(i < size());
    assert(j < size());

    // Indices are swapped for an element on subdiagonal
    if (i > j) {
        std::swap(i, j);
    }
    if (j - i < 2) {
        m_q.view_host()(j - i, i) = aij;
    } else {
        assert(std::fabs(aij) < 1e-15);
    }
}

template <class ExecSpace>
void SplinesLinearProblemPDSTridiag<ExecSpace>::setup_solver()
{
    int const info = LAPACKE_dpttrf(
            size(),
            m_q.view_host().data(),
            m_q.view_host().data() + m_q.view_host().stride(0));
    if (info != 0) {
        throw std::runtime_error("LAPACKE_dpttrf failed with error code " + std::to_string(info));
    }

    // Push on device
    m_q.modify_host();
    m_q.sync_device();
}

template <class ExecSpace>
void SplinesLinearProblemPDSTridiag<ExecSpace>::solve(MultiRHS const b, bool const) const
{
    assert(b.extent(0) == size());
    auto q_device = m_q.view_device();
    auto d = Kokkos::subview(q_device, 0, Kokkos::ALL);
    auto e = Kokkos::subview(q_device, 1, Kokkos::pair<int, int>(0, q_device.extent_int(1) - 1));
    Kokkos::RangePolicy<ExecSpace> const policy(0, b.extent(1));
    Kokkos::parallel_for(
            "pttrs",
            policy,
            KOKKOS_LAMBDA(int const i) {
                auto sub_b = Kokkos::subview(b, Kokkos::ALL, i);
                KokkosBatched::SerialPttrs<
                        KokkosBatched::Uplo::Lower,
                        KokkosBatched::Algo::Pttrs::Unblocked>::invoke(d, e, sub_b);
            });
}

#if defined(KOKKOS_ENABLE_SERIAL)
template class SplinesLinearProblemPDSTridiag<Kokkos::Serial>;
#endif
#if defined(KOKKOS_ENABLE_OPENMP)
template class SplinesLinearProblemPDSTridiag<Kokkos::OpenMP>;
#endif
#if defined(KOKKOS_ENABLE_CUDA)
template class SplinesLinearProblemPDSTridiag<Kokkos::Cuda>;
#endif
#if defined(KOKKOS_ENABLE_HIP)
template class SplinesLinearProblemPDSTridiag<Kokkos::HIP>;
#endif
#if defined(KOKKOS_ENABLE_SYCL)
template class SplinesLinearProblemPDSTridiag<Kokkos::SYCL>;
#endif

} // namespace ddc::detail
