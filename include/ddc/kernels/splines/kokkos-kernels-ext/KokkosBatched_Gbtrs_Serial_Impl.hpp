// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

// clang-format off
// NOLINTBEGIN(*)

#ifndef KOKKOSBATCHED_GBTRS_SERIAL_IMPL_HPP_
#define KOKKOSBATCHED_GBTRS_SERIAL_IMPL_HPP_

#include <Kokkos_Swap.hpp>
#include <KokkosBatched_Util.hpp>
#include <KokkosBlas2_gemv.hpp>
#include <KokkosBatched_Tbsv.hpp>

namespace KokkosBatched {

template <typename AViewType, typename BViewType>
KOKKOS_INLINE_FUNCTION static int checkGbtrsInput(
    [[maybe_unused]] const AViewType &A, [[maybe_unused]] const BViewType &b,
    [[maybe_unused]] const int kl, [[maybe_unused]] const int ku) {
  static_assert(Kokkos::is_view_v<AViewType>,
                "KokkosBatched::gbtrs: AViewType is not a Kokkos::View.");
  static_assert(Kokkos::is_view_v<BViewType>,
                "KokkosBatched::gbtrs: BViewType is not a Kokkos::View.");
  static_assert(AViewType::rank == 2,
                "KokkosBatched::gbtrs: AViewType must have rank 2.");
  static_assert(BViewType::rank == 1,
                "KokkosBatched::gbtrs: BViewType must have rank 1.");
#if (KOKKOSKERNELS_DEBUG_LEVEL > 0)
  if (kl < 0) {
    Kokkos::printf(
        "KokkosBatched::gbtrs: input parameter kl must not be less than 0: kl "
        "= "
        "%d\n",
        kl);
    return 1;
  }

  if (ku < 0) {
    Kokkos::printf(
        "KokkosBatched::gbtrs: input parameter ku must not be less than 0: ku "
        "= "
        "%d\n",
        ku);
    return 1;
  }

  const int lda = A.extent(0), n = A.extent(1);
  if (lda < (2 * kl + ku + 1)) {
    Kokkos::printf(
        "KokkosBatched::gbtrs: leading dimension of A must be smaller than 2 * "
        "kl + ku + 1: "
        "lda = %d, kl = %d, ku = %d\n",
        lda, kl, ku);
    return 1;
  }

  const int ldb = b.extent(0);
  if (ldb < Kokkos::max(1, n)) {
    Kokkos::printf(
        "KokkosBatched::gbtrs: leading dimension of b must be smaller than "
        "max(1, n): "
        "ldb = %d, n = %d\n",
        ldb, n);
    return 1;
  }

#endif
  return 0;
}

//// Non-transpose ////
template <>
struct SerialGbtrs<Trans::NoTranspose, Algo::Level3::Unblocked> {
  template <typename AViewType, typename BViewType, typename PivViewType>
  KOKKOS_INLINE_FUNCTION static int invoke(const AViewType &A,
                                           const BViewType &b,
                                           const PivViewType &piv, const int kl,
                                           const int ku) {
    // Quick return if possible
    const int n = A.extent(1);
    if (n == 0) return 0;

    auto info = checkGbtrsInput(A, b, kl, ku);
    if (info) return info;

    bool lonti = kl > 0;
    const int kd = ku + kl + 1;
    if (lonti) {
      for (int j = 0; j < n - 1; ++j) {
        const int lm = Kokkos::min(kl, n - j - 1);
        auto l       = piv(j);
        // If pivot index is not j, swap rows l and j in b
        if (l != j) {
          Kokkos::kokkos_swap(b(l), b(j));
        }

        // Perform a rank-1 update of the remaining part of the current column
        // (ger)
        for (int i = 0; i < lm; ++i) {
          b(j + 1 + i) = b(j + 1 + i) - A(kd + i, j) * b(j);
        }
      }
    }

    // Solve U*X = b for each right hand side, overwriting B with X.
    [[maybe_unused]] auto info_tbsv =
        KokkosBatched::SerialTbsv<Uplo::Upper, Trans::NoTranspose,
                                  Diag::NonUnit,
                                  Algo::Trsv::Unblocked>::invoke(A, b, kl + ku);

    return 0;
  }
};

//// Transpose ////
template <>
struct SerialGbtrs<Trans::Transpose, Algo::Level3::Unblocked> {
  template <typename AViewType, typename BViewType, typename PivViewType>
  KOKKOS_INLINE_FUNCTION static int invoke(const AViewType &A,
                                           const BViewType &b,
                                           const PivViewType &piv, const int kl,
                                           const int ku) {
    // Quick return if possible
    const int n = A.extent(1);
    if (n == 0) return 0;

    auto info = checkGbtrsInput(A, b, kl, ku);
    if (info) return info;

    bool lonti   = kl > 0;
    const int kd = ku + kl + 1;

    // Solve U*X = b for each right hand side, overwriting B with X.
    [[maybe_unused]] auto info_tbsv =
        KokkosBatched::SerialTbsv<Uplo::Upper, Trans::Transpose, Diag::NonUnit,
                                  Algo::Tbsv::Unblocked>::invoke(A, b, kl + ku);

    if (lonti) {
      for (int j = n - 2; j >= 0; --j) {
        const int lm = Kokkos::min(kl, n - j - 1);

        // Gemv transposed
        auto a = Kokkos::subview(b, Kokkos::pair(j + 1, j + 1 + lm));
        auto x = Kokkos::subview(A, Kokkos::pair(kd, kd + lm), j);
        auto y = Kokkos::subview(b, Kokkos::pair(j, j + lm));

        [[maybe_unused]] auto info_gemv =
            KokkosBlas::Impl::SerialGemvInternal<Algo::Gemv::Unblocked>::invoke(
                1, a.extent(0), -1.0, a.data(), a.stride_0(), a.stride_0(),
                x.data(), x.stride_0(), 1.0, y.data(), y.stride_0());

        // If pivot index is not j, swap rows l and j in b
        auto l = piv(j);
        if (l != j) {
          Kokkos::kokkos_swap(b(l), b(j));
        }
      }
    }

    return 0;
  }
};
}  // namespace KokkosBatched

#endif  // KOKKOSBATCHED_GBTRS_SERIAL_IMPL_HPP_

// NOLINTEND(*)
// clang-format on
