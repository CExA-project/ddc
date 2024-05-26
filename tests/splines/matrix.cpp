// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <cmath>
#include <memory>
#include <sstream>

#include <ddc/ddc.hpp>
#include <ddc/kernels/splines.hpp>

#include <gtest/gtest.h>

#include <Kokkos_DualView.hpp>

#include "test_utils.hpp"

namespace {

void fill_identity(
        ddc::detail::SplinesLinearProblem<Kokkos::DefaultHostExecutionSpace>::MultiRHS mat)
{
    assert(mat.extent(0) == mat.extent(1));
    for (std::size_t i(0); i < mat.extent(0); ++i) {
        for (std::size_t j(0); j < mat.extent(1); ++j) {
            mat(i, j) = int(i == j);
        }
    }
}

void copy_matrix(
        ddc::detail::SplinesLinearProblem<Kokkos::DefaultHostExecutionSpace>::MultiRHS copy,
        std::unique_ptr<ddc::detail::SplinesLinearProblem<Kokkos::DefaultExecutionSpace>>& mat)
{
    assert(mat->size() == copy.extent(0));
    assert(mat->size() == copy.extent(1));

    for (std::size_t i(0); i < copy.extent(0); ++i) {
        for (std::size_t j(0); j < copy.extent(1); ++j) {
            copy(i, j) = mat->get_element(i, j);
        }
    }
}

void check_inverse(
        ddc::detail::SplinesLinearProblem<Kokkos::DefaultHostExecutionSpace>::MultiRHS matrix,
        ddc::detail::SplinesLinearProblem<Kokkos::DefaultHostExecutionSpace>::MultiRHS inv)
{
    double TOL = 1e-10;
    std::size_t N = matrix.extent(0);

    for (std::size_t i(0); i < N; ++i) {
        for (std::size_t j(0); j < N; ++j) {
            double id_val = 0.0;
            for (std::size_t k(0); k < N; ++k) {
                id_val += matrix(i, k) * inv(k, j);
            }
            EXPECT_NEAR(id_val, static_cast<double>(i == j), TOL);
        }
    }
}

void check_inverse_transpose(
        ddc::detail::SplinesLinearProblem<Kokkos::DefaultHostExecutionSpace>::MultiRHS matrix,
        ddc::detail::SplinesLinearProblem<Kokkos::DefaultHostExecutionSpace>::MultiRHS inv)
{
    double TOL = 1e-10;
    std::size_t N = matrix.extent(0);

    for (std::size_t i(0); i < N; ++i) {
        for (std::size_t j(0); j < N; ++j) {
            double id_val = 0.0;
            for (std::size_t k(0); k < N; ++k) {
                id_val += matrix(i, k) * inv(j, k);
            }
            EXPECT_NEAR(id_val, static_cast<double>(i == j), TOL);
        }
    }
}
} // namespace

TEST(MatrixSparse, Formatting)
{
    ddc::detail::SplinesLinearProblemSparse<Kokkos::DefaultExecutionSpace> matrix(2);
    matrix.set_element(0, 0, 1);
    matrix.set_element(0, 1, 2);
    matrix.set_element(1, 0, 3);
    matrix.set_element(1, 1, 4);
    std::stringstream ss;
    ss << matrix;
    EXPECT_EQ(ss.str(), "     1.000     2.000\n     3.000     4.000\n");
}

class MatrixSizesFixture : public testing::TestWithParam<std::tuple<std::size_t, std::size_t>>
{
};

TEST_P(MatrixSizesFixture, Sparse)
{
    auto const [N, k] = GetParam();


    std::unique_ptr<ddc::detail::SplinesLinearProblem<Kokkos::DefaultExecutionSpace>> matrix
            = ddc::detail::SplinesLinearProblemMaker::make_new_sparse<
                    Kokkos::DefaultExecutionSpace>(N);

    std::vector<double> val_ptr(N * N);
    ddc::detail::SplinesLinearProblem<Kokkos::DefaultHostExecutionSpace>::MultiRHS
            val(val_ptr.data(), N, N);
    for (std::size_t i(0); i < N; ++i) {
        for (std::size_t j(0); j < N; ++j) {
            if (i == j) {
                matrix->set_element(i, j, 3. / 4);
            } else if (std::abs((std::ptrdiff_t)(j - i)) <= (std::ptrdiff_t)k) {
                matrix->set_element(i, j, -(1. / 4) / k);
            }
        }
    }
    copy_matrix(val, matrix);

    matrix->setup_solver();

    Kokkos::DualView<double*> inv_ptr("inv_ptr", N * N);
    ddc::detail::SplinesLinearProblem<Kokkos::DefaultHostExecutionSpace>::MultiRHS
            inv(inv_ptr.h_view.data(), N, N);
    fill_identity(inv);
    inv_ptr.modify_host();
    inv_ptr.sync_device();
    matrix->solve(ddc::detail::SplinesLinearProblem<
                  Kokkos::DefaultExecutionSpace>::MultiRHS(inv_ptr.d_view.data(), N, N));
    inv_ptr.modify_device();
    inv_ptr.sync_host();

    Kokkos::DualView<double*> inv_tr_ptr("inv_tr_ptr", N * N);
    ddc::detail::SplinesLinearProblem<Kokkos::DefaultHostExecutionSpace>::MultiRHS
            inv_tr(inv_tr_ptr.h_view.data(), N, N);
    fill_identity(inv_tr);
    inv_tr_ptr.modify_host();
    inv_tr_ptr.sync_device();
    matrix
            ->solve(ddc::detail::SplinesLinearProblem<Kokkos::DefaultExecutionSpace>::
                            MultiRHS(inv_tr_ptr.d_view.data(), N, N),
                    true);
    inv_tr_ptr.modify_device();
    inv_tr_ptr.sync_host();

    check_inverse(val, inv);
    check_inverse_transpose(val, inv_tr);
}

INSTANTIATE_TEST_SUITE_P(
        MyGroup,
        MatrixSizesFixture,
        testing::Combine(testing::Values<std::size_t>(10, 20), testing::Range<std::size_t>(1, 7)));
