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

SplinesLinearProblemDense::SplinesLinearProblemDense(std::size_t const mat_size)
    : SplinesLinearProblem(mat_size)
    , m_a("a", mat_size, mat_size)
    , m_ipiv("ipiv", mat_size)
{
    Kokkos::deep_copy(m_a.view_host(), 0.);
}

SplinesLinearProblemDense::~SplinesLinearProblemDense() = default;

double SplinesLinearProblemDense::get_element(std::size_t const i, std::size_t const j) const
{
    assert(i < size());
    assert(j < size());
    return m_a.view_host()(i, j);
}

void SplinesLinearProblemDense::set_element(
        std::size_t const i,
        std::size_t const j,
        double const aij)
{
    assert(i < size());
    assert(j < size());
    m_a.view_host()(i, j) = aij;
}

void SplinesLinearProblemDense::setup_solver()
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

void SplinesLinearProblemDense::solve(MultiRHS const b, bool const transpose) const
{
    assert(b.extent(0) == size());

    // For order 1 splines, size() can be 0 then we bypass the solver call.
    if (size() == 0) {
        return;
    }

    auto a_device = m_a.view_device();
    auto ipiv_device = m_ipiv.view_device();

    Kokkos::RangePolicy<Kokkos::Serial> const policy(0, b.extent(1));

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

} // namespace ddc::detail
