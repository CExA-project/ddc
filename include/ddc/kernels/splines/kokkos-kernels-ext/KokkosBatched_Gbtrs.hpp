// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

// clang-format off
// NOLINTBEGIN(*)

#pragma once

#include <KokkosBatched_Util.hpp>

/// \author Yuuichi Asahi (yuuichi.asahi@cea.fr)

namespace KokkosBatched {

/// \brief Serial Batched Gbtrs:
///
/// Solve A_l x_l = b_l for all l = 0, ..., N
///   with a general band matrix A using the LU factorization computed
///   by gbtrf.
///
/// \tparam AViewType: Input type for the matrix, needs to be a 2D view
/// \tparam BViewType: Input type for the right-hand side and the solution,
/// needs to be a 1D view
/// \tparam PivViewType: Integer type for pivot indices, needs to be a 1D view
///
/// \param A [in]: A is a ldab by n banded matrix.
/// Details of the LU factorization of the band matrix A, as computed by
/// gbtrf. U is stored as an upper triangular band matrix with KL+KU
/// superdiagonals in rows 1 to KL+KU+1, and the multipliers used during
/// the factorization are stored in rows KL+KU+2 to 2*KL+KU+1.
/// \param b [inout]: right-hand side and the solution
/// \param piv [in]: The pivot indices; for 1 <= i <= N, row i of the matrix
/// was interchanged with row piv(i).
/// \param kl [in]: kl specifies the number of subdiagonals within the band
/// of A. kl >= 0
/// \param ku [in]: ku specifies the number of superdiagonals within the band
/// of A. ku >= 0
///
/// No nested parallel_for is used inside of the function.
///

template <typename ArgTrans, typename ArgAlgo>
struct SerialGbtrs {
  template <typename AViewType, typename BViewType, typename PivViewType>
  KOKKOS_INLINE_FUNCTION static int invoke(const AViewType &A,
                                           const BViewType &b,
                                           const PivViewType &piv, const int kl,
                                           const int ku);
};
}  // namespace KokkosBatched

#include "KokkosBatched_Gbtrs_Serial_Impl.hpp"

// NOLINTEND(*)
// clang-format on
