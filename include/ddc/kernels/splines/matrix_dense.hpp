// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cassert>
#include <memory>

#include <KokkosBatched_ApplyPivot_Decl.hpp>
#include <KokkosBatched_Gesv.hpp>

#include "matrix.hpp"

namespace ddc::detail {
extern "C" int dgetrf_(int const* m, int const* n, double* a, int const* lda, int* ipiv, int* info);

template <class ExecSpace>
class Matrix_Dense : public Matrix
{
protected:
    Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace> m_a;
    Kokkos::View<int*, Kokkos::HostSpace> m_ipiv;

public:
    explicit Matrix_Dense(int const mat_size)
        : Matrix(mat_size)
        , m_a("a", mat_size, mat_size)
        , m_ipiv("ipiv", mat_size)
    {
        assert(mat_size > 0);

        Kokkos::deep_copy(m_a, 0.);
    }

    double get_element(int const i, int const j) const override
    {
        assert(i < get_size());
        assert(j < get_size());
        return m_a(i, j);
    }

    void set_element(int const i, int const j, double const aij) override
    {
        m_a(i, j) = aij;
    }

    int factorize_method() override
    {
        int info;
        int const n = get_size();
        dgetrf_(&n, &n, m_a.data(), &n, m_ipiv.data(), &info);
        return info;
        /*
        Kokkos::parallel_for(
                "gertf",
                Kokkos::RangePolicy<ExecSpace>(0, 1),
                KOKKOS_CLASS_LAMBDA(const int i) {
                    int info = KokkosBatched::SerialLU<
                            KokkosBatched::Algo::Level3::Unblocked>::invoke(m_a);
                });
*/
        //return 0;
    }

public:
    int solve_inplace_method(ddc::DSpan2D_stride b, char const transpose) const override
    {
        assert(b.stride(0) == 1);
        int const n_equations = b.extent(1);
        int const stride = b.stride(1);

        Kokkos::View<double**, Kokkos::LayoutStride, typename ExecSpace::memory_space>
                b_view(b.data_handle(), Kokkos::LayoutStride(get_size(), 1, n_equations, stride));

        auto a_device = create_mirror_view_and_copy(ExecSpace(), m_a);
        auto ipiv_device = create_mirror_view_and_copy(ExecSpace(), m_ipiv);
        Kokkos::parallel_for(
                "gerts",
                Kokkos::TeamPolicy<ExecSpace>(n_equations, Kokkos::AUTO),
                KOKKOS_LAMBDA(
                        const typename Kokkos::TeamPolicy<ExecSpace>::member_type& teamMember) {
                    const int i = teamMember.league_rank();

                    int info;
                    auto b_slice = Kokkos::subview(b_view, Kokkos::ALL, i);

                    if (transpose == 'N') {
                        teamMember.team_barrier();
                        KokkosBatched::TeamVectorApplyPivot<
                                typename Kokkos::TeamPolicy<ExecSpace>::member_type,
                                KokkosBatched::Side::Left,
                                KokkosBatched::Direct::Forward>::
                                invoke(teamMember, ipiv_device, b_slice);
                        teamMember.team_barrier();
                        KokkosBatched::TeamVectorTrsm<
                                typename Kokkos::TeamPolicy<ExecSpace>::member_type,
                                KokkosBatched::Side::Left,
                                KokkosBatched::Uplo::Lower,
                                KokkosBatched::Trans::NoTranspose,
                                KokkosBatched::Diag::Unit,
                                KokkosBatched::Algo::Level3::Unblocked>::
                                invoke(teamMember, 1.0, a_device, b_slice);
                        teamMember.team_barrier();
                        KokkosBatched::TeamVectorTrsm<
                                typename Kokkos::TeamPolicy<ExecSpace>::member_type,
                                KokkosBatched::Side::Left,
                                KokkosBatched::Uplo::Upper,
                                KokkosBatched::Trans::NoTranspose,
                                KokkosBatched::Diag::NonUnit,
                                KokkosBatched::Algo::Level3::Unblocked>::
                                invoke(teamMember, 1.0, a_device, b_slice);
                        teamMember.team_barrier();
                    } else if (transpose == 'T') {
                        teamMember.team_barrier();
                        KokkosBatched::TeamVectorTrsm<
                                typename Kokkos::TeamPolicy<ExecSpace>::member_type,
                                KokkosBatched::Side::Left,
                                KokkosBatched::Uplo::Upper,
                                KokkosBatched::Trans::Transpose,
                                KokkosBatched::Diag::NonUnit,
                                KokkosBatched::Algo::Level3::Unblocked>::
                                invoke(teamMember, 1.0, a_device, b_slice);
                        teamMember.team_barrier();
                        KokkosBatched::TeamVectorTrsm<
                                typename Kokkos::TeamPolicy<ExecSpace>::member_type,
                                KokkosBatched::Side::Left,
                                KokkosBatched::Uplo::Lower,
                                KokkosBatched::Trans::Transpose,
                                KokkosBatched::Diag::Unit,
                                KokkosBatched::Algo::Level3::Unblocked>::
                                invoke(teamMember, 1.0, a_device, b_slice);
                        teamMember.team_barrier();
                        KokkosBatched::TeamVectorApplyPivot<
                                typename Kokkos::TeamPolicy<ExecSpace>::member_type,
                                KokkosBatched::Side::Left,
                                KokkosBatched::Direct::Backward>::
                                invoke(teamMember, ipiv_device, b_slice);
                        teamMember.team_barrier();
                    } else {
                        info = -1;
                    }
                });
        return 0;
    }
};

} // namespace ddc::detail
