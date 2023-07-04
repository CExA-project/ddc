#ifndef MATRIX_PERIODIC_BANDED_H
#define MATRIX_PERIODIC_BANDED_H
#include <algorithm>
#include <cassert>
#include <cmath>
#include <memory>
#include <utility>

#include <experimental/mdspan>

#include "matrix.hpp"
#include "matrix_corner_block.hpp"
#include "matrix_dense.hpp"
#include "view.hpp"



class Matrix;

class Matrix_Periodic_Banded : public Matrix_Corner_Block
{
public:
    Matrix_Periodic_Banded(int const n, int const kl, int const ku, std::unique_ptr<Matrix> q)
        : Matrix_Corner_Block(
                n,
                std::max(kl, ku),
                std::move(q),
                std::max(kl, ku) + 1,
                std::max(kl, ku))
        , kl(kl)
        , ku(ku)
    {
    }
    double get_element(int const i, int j) const override
    {
        assert(i >= 0);
        assert(i < n);
        assert(j >= 0);
        assert(i < n);
        if (i >= nb && j < nb) {
            int d = j - i;
            if (d > n / 2)
                d -= n;
            if (d < -n / 2)
                d += n;

            if (d < -kl || d > ku)
                return 0.0;
            if (d > 0)
                return lambda(j, i - nb);
            else
                return lambda(j - nb + k + 1, i - nb);
        } else {
            return Matrix_Corner_Block::get_element(i, j);
        }
    }
    void set_element(int const i, int j, double const a_ij) override
    {
        assert(i >= 0);
        assert(i < n);
        assert(j >= 0);
        assert(i < n);
        if (i >= nb && j < nb) {
            int d = j - i;
            if (d > n / 2)
                d -= n;
            if (d < -n / 2)
                d += n;

            if (d < -kl || d > ku) {
                assert(std::fabs(a_ij) < 1e-20);
                return;
            }

            if (d > 0)
                lambda(j, i - nb) = a_ij;
            else
                lambda(j - nb + k + 1, i - nb) = a_ij;
        } else {
            Matrix_Corner_Block::set_element(i, j, a_ij);
        }
    }

protected:
    void calculate_delta_to_factorize() override
    {
        for (int i = 0; i < k; ++i) {
            for (int j = 0; j < k; ++j) {
                double val = 0.0;
                // Upper diagonals in lambda, lower diagonals in Abm_1_gamma
                for (int l = 0; l < i + 1; ++l) {
                    val += lambda(l, i) * Abm_1_gamma(j, l);
                }
                // Lower diagonals in lambda, upper diagonals in Abm_1_gamma
                for (int l = i + 1; l < k + 1; ++l) {
                    int l_full = nb - 1 - k + l;
                    val += lambda(l, i) * Abm_1_gamma(j, l_full);
                }
                delta.set_element(i, j, delta.get_element(i, j) - val);
            }
        }
    }
    DSpan1D solve_lambda_section(DSpan1D const v, DView1D const u) const override
    {
        for (int i = 0; i < k; ++i) {
            // Upper diagonals in lambda
            for (int j = 0; j <= i; ++j) {
                v(i) -= lambda(j, i) * u(j);
            }
            // Lower diagonals in lambda
            for (int j = i + 1; j < k + 1; ++j) {
                v(i) -= lambda(j, i) * u(nb - 1 - k + j);
            }
        }
        return v;
    }
    DSpan1D solve_lambda_section_transpose(DSpan1D const u, DView1D const v) const override
    {
        for (int i = 0; i < k; ++i) {
            // Upper diagonals in lambda
            for (int j = 0; j <= i; ++j) {
                u(j) -= lambda(j, i) * v(i);
            }
            // Lower diagonals in lambda
            for (int j = i + 1; j < k + 1; ++j) {
                u(nb - 1 - k + j) -= lambda(j, i) * v(i);
            }
        }
        return u;
    }
    int const kl; // no. of subdiagonals
    int const ku; // no. of superdiagonals
};

#endif // MATRIX_PERIODIC_BANDED_H
