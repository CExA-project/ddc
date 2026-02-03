// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <cassert>
#include <cstddef>
#include <iomanip>
#include <ostream>

#include <Kokkos_Core.hpp>

#include "splines_linear_problem.hpp"

namespace ddc::detail {

template <class ExecSpace>
SplinesLinearProblem<ExecSpace>::SplinesLinearProblem(std::size_t const size) : m_size(size)
{
}

template <class ExecSpace>
SplinesLinearProblem<ExecSpace>::~SplinesLinearProblem() = default;

template <class ExecSpace>
std::size_t SplinesLinearProblem<ExecSpace>::size() const
{
    return m_size;
}

template <class ExecSpace>
std::size_t SplinesLinearProblem<ExecSpace>::required_number_of_rhs_rows() const
{
    std::size_t const nrows = impl_required_number_of_rhs_rows();
    assert(nrows >= size());
    return nrows;
}

template <class ExecSpace>
std::size_t SplinesLinearProblem<ExecSpace>::impl_required_number_of_rhs_rows() const
{
    return m_size;
}

template <class ExecSpace>
std::ostream& operator<<(std::ostream& os, SplinesLinearProblem<ExecSpace> const& linear_problem)
{
    std::size_t const n = linear_problem.size();
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            os << std::fixed << std::setprecision(3) << std::setw(10)
               << linear_problem.get_element(i, j);
        }
        os << "\n";
    }
    return os;
}

#if defined(KOKKOS_ENABLE_SERIAL)
template class SplinesLinearProblem<Kokkos::Serial>;
template std::ostream& operator<<(
        std::ostream& os,
        SplinesLinearProblem<Kokkos::Serial> const& linear_problem);
#endif
#if defined(KOKKOS_ENABLE_OPENMP)
template class SplinesLinearProblem<Kokkos::OpenMP>;
template std::ostream& operator<<(
        std::ostream& os,
        SplinesLinearProblem<Kokkos::OpenMP> const& linear_problem);
#endif
#if defined(KOKKOS_ENABLE_CUDA)
template class SplinesLinearProblem<Kokkos::Cuda>;
template std::ostream& operator<<(
        std::ostream& os,
        SplinesLinearProblem<Kokkos::Cuda> const& linear_problem);
#endif
#if defined(KOKKOS_ENABLE_HIP)
template class SplinesLinearProblem<Kokkos::HIP>;
template std::ostream& operator<<(
        std::ostream& os,
        SplinesLinearProblem<Kokkos::HIP> const& linear_problem);
#endif
#if defined(KOKKOS_ENABLE_SYCL)
template class SplinesLinearProblem<Kokkos::SYCL>;
template std::ostream& operator<<(
        std::ostream& os,
        SplinesLinearProblem<Kokkos::SYCL> const& linear_problem);
#endif

} // namespace ddc::detail
