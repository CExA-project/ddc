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

#include <KokkosBatched_Pbtrs.hpp>
#include <KokkosBatched_Util.hpp>

#include "splines_linear_problem.hpp"
#include "splines_linear_problem_pds_band.hpp"

namespace ddc::detail {

template <class ExecSpace>
SplinesLinearProblemPDSBand<ExecSpace>::SplinesLinearProblemPDSBand(
        std::size_t const mat_size,
        std::size_t const kd)
    : SplinesLinearProblem<ExecSpace>(mat_size)
    , m_q("q", kd + 1, mat_size)
{
    assert(m_q.extent(0) <= mat_size);

    Kokkos::deep_copy(m_q.view_host(), 0.);
}

template <class ExecSpace>
SplinesLinearProblemPDSBand<ExecSpace>::~SplinesLinearProblemPDSBand() = default;

template <class ExecSpace>
double SplinesLinearProblemPDSBand<ExecSpace>::get_element(std::size_t i, std::size_t j) const
{
    assert(i < size());
    assert(j < size());

    // Indices are swapped for an element on subdiagonal
    if (i > j) {
        std::swap(i, j);
    }

    if (j - i < m_q.extent(0)) {
        return m_q.view_host()(j - i, i);
    }

    return 0.0;
}

template <class ExecSpace>
void SplinesLinearProblemPDSBand<ExecSpace>::set_element(
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
    if (j - i < m_q.extent(0)) {
        m_q.view_host()(j - i, i) = aij;
    } else {
        assert(std::fabs(aij) < 1e-15);
    }
}

template <class ExecSpace>
void SplinesLinearProblemPDSBand<ExecSpace>::setup_solver()
{
    int const info = LAPACKE_dpbtrf(
            LAPACK_ROW_MAJOR,
            'L',
            size(),
            m_q.extent(0) - 1,
            m_q.view_host().data(),
            m_q.view_host().stride(
                    0) // m_q.view_host().stride(0) if LAPACK_ROW_MAJOR, m_q.view_host().stride(1) if LAPACK_COL_MAJOR
    );
    if (info != 0) {
        throw std::runtime_error("LAPACKE_dpbtrf failed with error code " + std::to_string(info));
    }

    // Push on device
    m_q.modify_host();
    m_q.sync_device();
}

template <class ExecSpace>
void SplinesLinearProblemPDSBand<ExecSpace>::solve(MultiRHS const b, bool const) const
{
    assert(b.extent(0) == size());

    auto q_device = m_q.view_device();
    Kokkos::RangePolicy<ExecSpace> const policy(0, b.extent(1));
    Kokkos::parallel_for(
            "pbtrs",
            policy,
            KOKKOS_LAMBDA(int const i) {
                auto sub_b = Kokkos::subview(b, Kokkos::ALL, i);
                KokkosBatched::SerialPbtrs<
                        KokkosBatched::Uplo::Lower,
                        KokkosBatched::Algo::Pbtrs::Unblocked>::invoke(q_device, sub_b);
            });
}

#if defined(KOKKOS_ENABLE_SERIAL)
template class SplinesLinearProblemPDSBand<Kokkos::Serial>;
#endif
#if defined(KOKKOS_ENABLE_OPENMP)
template class SplinesLinearProblemPDSBand<Kokkos::OpenMP>;
#endif
#if defined(KOKKOS_ENABLE_CUDA)
template class SplinesLinearProblemPDSBand<Kokkos::Cuda>;
#endif
#if defined(KOKKOS_ENABLE_HIP)
template class SplinesLinearProblemPDSBand<Kokkos::HIP>;
#endif
#if defined(KOKKOS_ENABLE_SYCL)
template class SplinesLinearProblemPDSBand<Kokkos::SYCL>;
#endif

} // namespace ddc::detail
