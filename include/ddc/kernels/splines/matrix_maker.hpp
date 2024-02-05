#pragma once

#include <memory>
#include <optional>

#include "matrix_banded.hpp"
#include "matrix_center_block.hpp"
#include "matrix_corner_block.hpp"
#include "matrix_dense.hpp"
#include "matrix_pds_tridiag.hpp"
#include "matrix_periodic_banded.hpp"
#include "matrix_sparse.hpp"

namespace ddc::detail {

class MatrixMaker
{
public:
    template <typename ExecSpace>
    static std::unique_ptr<Matrix> make_new_dense(int const n)
    {
        return std::make_unique<Matrix_Dense<ExecSpace>>(n);
    }

    template <typename ExecSpace>
    static std::unique_ptr<Matrix> make_new_banded(
            int const n,
            int const kl,
            int const ku,
            bool const pds)
    {
        if (kl == ku && kl == 1 && pds) {
            return std::make_unique<Matrix_PDS_Tridiag>(n);
        } else if (2 * kl + 1 + ku >= n) {
            return std::make_unique<Matrix_Dense<ExecSpace>>(n);
        } else {
            return std::make_unique<Matrix_Banded<ExecSpace>>(n, kl, ku);
        }
    }

    template <typename ExecSpace>
    static std::unique_ptr<Matrix> make_new_periodic_banded(
            int const n,
            int const kl,
            int const ku,
            bool const pds)
    {
        int const border_size = std::max(kl, ku);
        int const banded_size = n - border_size;
        std::unique_ptr<Matrix> block_mat;
        if (pds && kl == ku && kl == 1) {
            // block_mat = std::make_unique<Matrix_PDS_Tridiag>(banded_size);
            block_mat = std::make_unique<Matrix_Banded<ExecSpace>>(banded_size, 1, 1);
        } else if (
                border_size * n + border_size * (border_size + 1) + (2 * kl + 1 + ku) * banded_size
                >= n * n) {
            return std::make_unique<Matrix_Dense<ExecSpace>>(n);
        } else {
            block_mat = std::make_unique<Matrix_Banded<ExecSpace>>(banded_size, kl, ku);
        }
        return std::make_unique<Matrix_Periodic_Banded<ExecSpace>>(n, kl, ku, std::move(block_mat));
    }

    template <typename ExecSpace>
    static std::unique_ptr<Matrix> make_new_block_with_banded_region(
            int const n,
            int const kl,
            int const ku,
            bool const pds,
            int const block1_size,
            int const block2_size = 0)
    {
        int const banded_size = n - block1_size - block2_size;
        std::unique_ptr<Matrix> block_mat;
        if (pds && kl == ku && kl == 1) {
            //block_mat = std::make_unique<Matrix_PDS_Tridiag>(banded_size);
            block_mat = std::make_unique<Matrix_Banded<ExecSpace>>(banded_size,1,1);
        } else if (2 * kl + 1 + ku >= banded_size) {
            return std::make_unique<Matrix_Dense<ExecSpace>>(n);
        } else {
            block_mat = std::make_unique<Matrix_Banded<ExecSpace>>(banded_size, kl, ku);
        }
        if (block2_size == 0) {
            return std::make_unique<
                    Matrix_Corner_Block<ExecSpace>>(n, block1_size, std::move(block_mat));
        } else {
            return std::make_unique<Matrix_Center_Block<
                    ExecSpace>>(n, block1_size, block2_size, std::move(block_mat));
        }
    }

    template <typename ExecSpace>
    static std::unique_ptr<Matrix> make_new_sparse(
            int const n,
            std::optional<int> cols_per_chunk = std::nullopt,
            std::optional<unsigned int> preconditionner_max_block_size = std::nullopt)
    {
        return std::make_unique<
                Matrix_Sparse<ExecSpace>>(n, cols_per_chunk, preconditionner_max_block_size);
    }
};

} // namespace ddc::detail
