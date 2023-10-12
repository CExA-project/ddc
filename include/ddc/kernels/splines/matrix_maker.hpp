#pragma once
#include <memory>

#include "Kokkos_Core_fwd.hpp"
#include "matrix_sparse.hpp"


namespace ddc::detail {
class MatrixMaker
{
public:
    template <typename ExecSpace>
    static std::unique_ptr<Matrix> make_new_sparse(int const n)
    {
        return std::make_unique<Matrix_Sparse<ExecSpace>>(n);
    }
};
} // namespace ddc::detail
