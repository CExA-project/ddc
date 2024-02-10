#ifndef MATRIX_DENSE_H
#define MATRIX_DENSE_H
#include <cassert>
#include <memory>

#include <KokkosBatched_Gesv.hpp>

#include "matrix.hpp"

namespace ddc::detail {

template <class ExecSpace>
class Matrix_Dense : public Matrix
{
protected:
    Kokkos::View<double**, Kokkos::LayoutLeft, typename ExecSpace::memory_space> m_a;

public:
    explicit Matrix_Dense(int const mat_size) : Matrix(mat_size), m_a("a", mat_size, mat_size)
    {
        assert(mat_size > 0);
    }

    void reset() const override
    {
        Kokkos::parallel_for(
                "fill_a",
                Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {get_size(), get_size()}),
                KOKKOS_CLASS_LAMBDA(const int i, const int j) { m_a(i, j) = 0; });
    }

    double KOKKOS_FUNCTION get_element(int const i, int const j) const override
    {
        assert(i < get_size());
        assert(j < get_size());
        KOKKOS_IF_ON_HOST(
                if constexpr (Kokkos::SpaceAccessibility<
                                      Kokkos::DefaultHostExecutionSpace,
                                      typename ExecSpace::memory_space>::accessible) {
                    return m_a(i, j);
                } else {
                    // Inefficient, usage is strongly discouraged
                    double aij;
                    Kokkos::deep_copy(
                            Kokkos::View<double, Kokkos::HostSpace>(&aij),
                            Kokkos::subview(m_a, i, j));
                    return aij;
                })
        KOKKOS_IF_ON_DEVICE(return m_a(i, j);)
    }

    void KOKKOS_FUNCTION set_element(int const i, int const j, double const aij) const override
    {
        KOKKOS_IF_ON_HOST(
                if constexpr (Kokkos::SpaceAccessibility<
                                      Kokkos::DefaultHostExecutionSpace,
                                      typename ExecSpace::memory_space>::accessible) {
                    m_a(i, j) = aij;
                } else {
                    // Inefficient, usage is strongly discouraged
                    Kokkos::deep_copy(
                            Kokkos::subview(m_a, i, j),
                            Kokkos::View<const double, Kokkos::HostSpace>(&aij));
                })
        KOKKOS_IF_ON_DEVICE(m_a(i, j) = aij;)
    }

    int factorize_method() override
    {
        /*
        Kokkos::parallel_for(
                "gertf",
                Kokkos::RangePolicy<ExecSpace>(0, 1),
                KOKKOS_CLASS_LAMBDA(const int i) {
                    int info = KokkosBatched::SerialLU<
                            KokkosBatched::Algo::Level3::Unblocked>::invoke(m_a);
                });
*/
        return 0;
    }

    int solve_inplace_method(
            double* const b,
            char const transpose,
            int const n_equations,
            int const stride) const override
    {
        Kokkos::View<double**, Kokkos::LayoutStride, typename ExecSpace::memory_space>
                b_view(b, Kokkos::LayoutStride(get_size(), 1, n_equations, stride));

        Kokkos::parallel_for(
                "gerts",
                Kokkos::TeamPolicy<ExecSpace>(n_equations, Kokkos::AUTO),
                KOKKOS_CLASS_LAMBDA(
                        const typename Kokkos::TeamPolicy<ExecSpace>::member_type& teamMember) {
                    const int i = teamMember.league_rank();

                    int info;
                    auto b_slice = Kokkos::subview(b_view, Kokkos::ALL, i);
                    Kokkos::View<double**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>
                            a_buffer = create_mirror(ExecSpace(), m_a);
                    if (transpose == 'N') {
                        Kokkos::deep_copy(a_buffer, m_a);
                    } else if (transpose == 'T') {
                        Kokkos::View<
                                double**,
                                Kokkos::LayoutRight,
                                typename ExecSpace::memory_space>
                                a_buffer2(m_a.data(), get_size(), get_size());
                        Kokkos::deep_copy(a_buffer, a_buffer2);
                    } else {
                        info = -1;
                    }
                    auto buffer = create_mirror(ExecSpace(), b_slice);
                    Kokkos::deep_copy(buffer, b_slice);
                    Kokkos::View<double**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>
                            tmp("tmp", get_size(), get_size() + 4);
                    teamMember.team_barrier();
                    info = KokkosBatched::TeamGesv<
                            typename Kokkos::TeamPolicy<ExecSpace>::member_type,
                            KokkosBatched::Gesv::StaticPivoting>::
                            invoke(teamMember,
                                   a_buffer,
                                   (Kokkos::View<
                                           double*,
                                           Kokkos::LayoutLeft,
                                           typename ExecSpace::memory_space>)b_slice,
                                   (Kokkos::View<
                                           double*,
                                           Kokkos::LayoutLeft,
                                           typename ExecSpace::memory_space>)buffer);
                    teamMember.team_barrier();
                    /*
                    int info;
                    if (transpose == 'N') {
                        info = KokkosBatched::SerialTrsm<
                                KokkosBatched::Side::Left,
                                KokkosBatched::Uplo::Lower,
                                KokkosBatched::Trans::NoTranspose,
                                KokkosBatched::Diag::Unit,
                                KokkosBatched::Algo::Level3::Unblocked>::invoke(1.0, m_a, b_slice);
                        info = KokkosBatched::SerialTrsm<
                                KokkosBatched::Side::Left,
                                KokkosBatched::Uplo::Upper,
                                KokkosBatched::Trans::NoTranspose,
                                KokkosBatched::Diag::NonUnit,
                                KokkosBatched::Algo::Level3::Unblocked>::invoke(1.0, m_a, b_slice);
                    } else if (transpose == 'T') {
                        info = KokkosBatched::SerialTrsm<
                                KokkosBatched::Side::Left,
                                KokkosBatched::Uplo::Upper,
                                KokkosBatched::Trans::Transpose,
                                KokkosBatched::Diag::NonUnit,
                                KokkosBatched::Algo::Level3::Unblocked>::invoke(1.0, m_a, b_slice);
                        info = KokkosBatched::SerialTrsm<
                                KokkosBatched::Side::Left,
                                KokkosBatched::Uplo::Lower,
                                KokkosBatched::Trans::Transpose,
                                KokkosBatched::Diag::Unit,
                                KokkosBatched::Algo::Level3::Unblocked>::invoke(1.0, m_a, b_slice);
                    } else {
                        info = -1;
                    }
					*/
                });
        return 0;
    }
};

} // namespace ddc::detail
#endif // MATRIX_DENSE_H
