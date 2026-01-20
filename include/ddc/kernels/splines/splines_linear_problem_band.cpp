// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <cassert>
#if !defined(NDEBUG)
#    include <cmath>
#endif
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

#include "kokkos-kernels-ext/KokkosBatched_Gbtrs.hpp"

#include "splines_linear_problem.hpp"
#include "splines_linear_problem_band.hpp"

namespace ddc::detail {

SplinesLinearProblemBand::SplinesLinearProblemBand(
        std::size_t const mat_size,
        std::size_t const kl,
        std::size_t const ku)
    : SplinesLinearProblem(mat_size)
    , m_kl(kl)
    , m_ku(ku)
    /*
         * The matrix itself stored in band format requires a (kl + ku + 1)*mat_size
         * allocation, but the LU-factorization requires an additional kl*mat_size block
         */
    , m_q("q", 2 * kl + ku + 1, mat_size)
    , m_ipiv("ipiv", mat_size)
{
    assert(m_kl <= mat_size);
    assert(m_ku <= mat_size);

    Kokkos::deep_copy(m_q.view_host(), 0.);
}

SplinesLinearProblemBand::~SplinesLinearProblemBand() = default;

std::size_t SplinesLinearProblemBand::band_storage_row_index(
        std::size_t const i,
        std::size_t const j) const
{
    return m_kl + m_ku + i - j;
}

double SplinesLinearProblemBand::get_element(std::size_t const i, std::size_t const j) const
{
    assert(i < size());
    assert(j < size());
    /*
         * The "row index" of the band format storage identify the (sub/super)-diagonal
         * while the column index is actually the column index of the matrix. Two layouts
         * are supported by LAPACKE. The m_kl first lines are irrelevant for the storage of
         * the matrix itself but required for the storage of its LU factorization.
         */
    if (i >= static_cast<std::size_t>(
                std::
                        max(static_cast<std::ptrdiff_t>(0),
                            static_cast<std::ptrdiff_t>(j) - static_cast<std::ptrdiff_t>(m_ku)))
        && i < std::min(size(), j + m_kl + 1)) {
        return m_q.view_host()(band_storage_row_index(i, j), j);
    }

    return 0.0;
}

void SplinesLinearProblemBand::set_element(
        std::size_t const i,
        std::size_t const j,
        double const aij)
{
    assert(i < size());
    assert(j < size());
    /*
         * The "row index" of the band format storage identify the (sub/super)-diagonal
         * while the column index is actually the column index of the matrix. Two layouts
         * are supported by LAPACKE. The m_kl first lines are irrelevant for the storage of
         * the matrix itself but required for the storage of its LU factorization.
         */
    if (i >= static_cast<std::size_t>(
                std::
                        max(static_cast<std::ptrdiff_t>(0),
                            static_cast<std::ptrdiff_t>(j) - static_cast<std::ptrdiff_t>(m_ku)))
        && i < std::min(size(), j + m_kl + 1)) {
        m_q.view_host()(band_storage_row_index(i, j), j) = aij;
    } else {
        assert(std::fabs(aij) < 1e-15);
    }
}

void SplinesLinearProblemBand::setup_solver()
{
    int const info = LAPACKE_dgbtrf(
            LAPACK_ROW_MAJOR,
            size(),
            size(),
            m_kl,
            m_ku,
            m_q.view_host().data(),
            m_q.view_host().stride(
                    0), // m_q.view_host().stride(0) if LAPACK_ROW_MAJOR, m_q.view_host().stride(1) if LAPACK_COL_MAJOR
            m_ipiv.view_host().data());
    if (info != 0) {
        throw std::runtime_error("LAPACKE_dgbtrf failed with error code " + std::to_string(info));
    }

    // Convert 1-based index to 0-based index
    for (std::size_t i = 0; i < size(); ++i) {
        m_ipiv.view_host()(i) -= 1;
    }

    // Push on device
    m_q.modify_host();
    m_q.sync_device();
    m_ipiv.modify_host();
    m_ipiv.sync_device();
}

void SplinesLinearProblemBand::solve(MultiRHS const b, bool const transpose) const
{
    assert(b.extent(0) == size());

    std::size_t const kl_proxy = m_kl;
    std::size_t const ku_proxy = m_ku;
    auto q_device = m_q.view_device();
    auto ipiv_device = m_ipiv.view_device();
    Kokkos::RangePolicy<Kokkos::Serial> const policy(0, b.extent(1));
    if (transpose) {
        Kokkos::parallel_for(
                "gbtrs",
                policy,
                KOKKOS_LAMBDA(int const i) {
                    auto sub_b = Kokkos::subview(b, Kokkos::ALL, i);
                    KokkosBatched::SerialGbtrs<
                            KokkosBatched::Trans::Transpose,
                            KokkosBatched::Algo::Level3::Unblocked>::
                            invoke(q_device, sub_b, ipiv_device, kl_proxy, ku_proxy);
                });
    } else {
        Kokkos::parallel_for(
                "gbtrs",
                policy,
                KOKKOS_LAMBDA(int const i) {
                    auto sub_b = Kokkos::subview(b, Kokkos::ALL, i);
                    KokkosBatched::SerialGbtrs<
                            KokkosBatched::Trans::NoTranspose,
                            KokkosBatched::Algo::Level3::Unblocked>::
                            invoke(q_device, sub_b, ipiv_device, kl_proxy, ku_proxy);
                });
    }
}

} // namespace ddc::detail
