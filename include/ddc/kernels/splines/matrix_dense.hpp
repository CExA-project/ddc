#ifndef MATRIX_DENSE_H
#define MATRIX_DENSE_H
#include <cassert>
#include <memory>

#include "matrix.hpp"

namespace ddc::detail {
extern "C" int dgetrf_(int const* m, int const* n, double* a, int const* lda, int* ipiv, int* info);
extern "C" int dgetrs_(
        char const* trans,
        int const* n,
        int const* nrhs,
        double* a,
        int const* lda,
        int* ipiv,
        double* b,
        int const* ldb,
        int* info);

template <class ExecSpace>
class Matrix_Dense : public Matrix
{
private:
    Kokkos::View<double**, Kokkos::LayoutRight, typename ExecSpace::memory_space> m_a;
    Kokkos::View<int*, Kokkos::LayoutRight, typename ExecSpace::memory_space> m_ipiv;

public:
    explicit Matrix_Dense(const int mat_size)
        : Matrix(mat_size)
        , m_a("a", mat_size, mat_size)
        , m_ipiv("ipiv", mat_size)
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

    double get_element(int const i, int const j) const override
    {
        assert(i < get_size());
        assert(j < get_size());
        return Kokkos::create_mirror_view_and_copy(Kokkos::DefaultHostExecutionSpace(), Kokkos::subview(m_a, i, j))();
    }

    void set_element(int const i, int const j, double const aij) override
    {
        Kokkos::parallel_for(
                "set_element",
                Kokkos::RangePolicy<ExecSpace>(0, 1),
                KOKKOS_CLASS_LAMBDA(const int) { m_a(i, j) = aij; });
    }

private:
    int factorize_method() override
    {
	    auto a_host = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultHostExecutionSpace(), m_a);
        auto ipiv_host = Kokkos::create_mirror_view(Kokkos::DefaultHostExecutionSpace(), m_ipiv);
        int info;
        int const n = get_size();
        dgetrf_(&n, &n, a_host.data(), &n, ipiv_host.data(), &info);
        Kokkos::deep_copy(m_a, a_host);
        Kokkos::deep_copy(m_ipiv, ipiv_host);
		std::cout << info;
        return info;
    }

    int solve_inplace_method(double* b, char const transpose, int const n_equations) const override
    {
	    auto a_host = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultHostExecutionSpace(), m_a);
        auto ipiv_host = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultHostExecutionSpace(), m_ipiv);
        Kokkos::View<double**, Kokkos::LayoutRight, typename ExecSpace::memory_space> b_view(b, get_size(), n_equations);
        auto b_host = Kokkos::create_mirror_view(Kokkos::DefaultHostExecutionSpace(), b_view);
        int info;
        int const n = get_size();
        dgetrs_(&transpose,
                &n,
                &n_equations,
                a_host.data(),
                &n,
                ipiv_host.data(),
                b_host.data(),
                &n,
                &info);
        Kokkos::deep_copy(b_view, b_host);
        return info;
    }
};

} // namespace ddc::detail
#endif // MATRIX_DENSE_H
