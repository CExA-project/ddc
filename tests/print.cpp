// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <iomanip>
#include <sstream>
#include <string>

#include <ddc/ddc.hpp>

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

inline namespace anonymous_namespace_workaround_print_cpp {

struct DDim0
{
};
struct DDim1
{
};
struct DDim2
{
};
struct DDim3
{
};
struct DDim4
{
};
struct DDim5
{
};

} // namespace anonymous_namespace_workaround_print_cpp

TEST(Print, Simple2DChunk)
{
    using cell = double;

    unsigned const dim0 = 2;
    unsigned const dim1 = 2;

    ddc::DiscreteDomain<DDim0> const domain_0
            = ddc::init_trivial_bounded_space(ddc::DiscreteVector<DDim0>(dim0));
    ddc::DiscreteDomain<DDim1> const domain_1
            = ddc::init_trivial_bounded_space(ddc::DiscreteVector<DDim1>(dim1));

    ddc::DiscreteDomain<DDim0, DDim1> const domain_2d(domain_0, domain_1);

    ddc::Chunk cells_in_dev_alloc("cells_in_dev", domain_2d, ddc::DeviceAllocator<cell>());
    ddc::ChunkSpan const cells_in = cells_in_dev_alloc.span_view();

    Kokkos::Random_XorShift64_Pool<> random_pool(/*seed=*/12345);
    ddc::parallel_for_each(
            domain_2d,
            KOKKOS_LAMBDA(ddc::DiscreteElement<DDim0, DDim1> const i) {
                auto generator = random_pool.get_state();
                cells_in(i) = generator.drand(-1., 1.);
                random_pool.free_state(generator);
            });

    {
        std::stringstream ss;
        ss << cells_in;
        EXPECT_EQ(
                "[[ 0.470186 -0.837013]\n"
                " [ 0.439832  0.347536]]",
                ss.str());
    }
    {
        std::stringstream ss;
        ss << std::setprecision(2) << cells_in;
        EXPECT_EQ(
                "[[ 0.47 -0.84]\n"
                " [ 0.44  0.35]]",
                ss.str());
    }
    {
        std::stringstream ss;
        ss << std::hexfloat << cells_in;
        EXPECT_EQ(
                "[[ 0x1.e178596d8678cp-2 -0x1.ac8cf5563aa03p-1]\n"
                " [ 0x1.c263537950d98p-2   0x1.63e08ca1dfd3p-2]]",
                ss.str());
    }
    {
        std::stringstream ss;
        ss << std::scientific << cells_in;
        EXPECT_EQ(
                "[[ 4.701857e-01 -8.370129e-01]\n"
                " [ 4.398320e-01  3.475363e-01]]",
                ss.str());
    }
}

struct OutputChecker
{
    void error(const std::string& input, size_t line, std::string msg)
    {
        std::cerr << "Error at line " << line << ":\n";
        std::cerr << msg;
    }

    struct ParserState
    {
        std::size_t line;
        std::size_t pos;
        std::size_t alignement;
        std::size_t largest_element_seen;
        std::size_t total_dims;
    };

    bool element(const std::string& input, ParserState& state)
    {
        bool leading_spaces = false;

        if (state.alignement == 0) {
            // First element seen
            std::size_t elem_size = 0;
            std::size_t n_spaces = 0;
            while (state.pos < input.size() && input[state.pos] == ' ') {
                ++state.pos;
                ++elem_size;
                ++n_spaces;
            }
            while (state.pos < input.size()
                   && !(input[state.pos] == ' ' || input[state.pos] == ']')) {
                ++state.pos;
                ++elem_size;
            }

            if (elem_size == 0 || elem_size == n_spaces) {
                error(input, state.line, "Empty element\n");
                return false;
            }

            state.alignement = elem_size;
            state.largest_element_seen = (n_spaces == 0);
        } else {
            std::size_t elem_size = 0;
            std::size_t n_spaces = 0;
            std::size_t elem_start = state.pos;

            // Elision
            if (input[state.pos] == '.' && state.pos < input.size() + 2
                && input[state.pos + 1] == '.' && input[state.pos + 2] == '.') {
                state.pos += 3;
                return true;
            }

            // Leading ' '
            while (state.pos < input.size() && input[state.pos] == ' ') {
                ++state.pos;
                ++elem_size;
                ++n_spaces;
            }
            // Actual element
            while (state.pos < input.size()
                   && !(input[state.pos] == ' ' || input[state.pos] == ']')) {
                ++state.pos;
                ++elem_size;
            }
            if (state.alignement != elem_size) {
                error(input,
                      state.line,
                      "Element '" + input.substr(elem_start, elem_size)
                              + "' badly aligned. Expected " + std::to_string(state.alignement)
                              + " chars but got " + std::to_string(elem_size) + " instead.\n");
                return false;
            }
            state.largest_element_seen = state.largest_element_seen || (n_spaces == 0);
        }

        return true;
    }

    bool dimension(const std::string& input, ParserState& state, std::size_t num_dims)
    {
        if (state.pos < input.size()) {
            // Elision
            if (input[state.pos] == '.' && state.pos < input.size() + 2
                && input[state.pos + 1] == '.' && input[state.pos + 2] == '.') {
                state.pos += 3;
                return true;
            }

            if (input[state.pos] != '[') {
                error(input,
                      state.line,
                      "Expected '[' but got '" + std::string(1, input[state.pos]) + "' instead\n");
                return false;
            }
        }

        ++state.pos;

        if (num_dims > 1) {
            bool ok;
            while ((ok = dimension(input, state, num_dims - 1)) && state.pos < input.size()
                   && input[state.pos] != ']') {
                for (int i = 0; i < num_dims - 1; ++i) {
                    if (input[state.pos] != '\n') {
                        error(input,
                              state.line,
                              "Expected Newline but got '" + std::string(1, input[state.pos])
                                      + "' instead.\n");
                        return false;
                    }
                    ++state.line;
                    ++state.pos;
                }

                for (int i = 0; i < state.total_dims - num_dims + 1; ++i) {
                    if (input[state.pos] != ' ') {
                        error(input,
                              state.line,
                              "Expected ' ' but got '" + std::string(1, input[state.pos])
                                      + "' instead.\n");
                        return false;
                    }
                    ++state.pos;
                }
            }
            if (!ok) {
                return false;
            }
            ++state.pos;
        } else if (num_dims == 0) {
            if (input.size() != 2 || input[1] != ']') {
                error(input, state.line, "Expected ']'.\n");
                return false;
            }
            ++state.pos;
            state.largest_element_seen = true;
            return true;
        } else { // num_dims == 1
            bool ok;
            while ((ok = element(input, state)) && state.pos < input.size()
                   && input[state.pos] != ']') {
                if (input[state.pos] == ' ') {
                    ++state.pos;
                }
            }
            if (!ok) {
                return false;
            }
            if (state.pos >= input.size()) {
                error(input, state.line, "Expected ']' but got EoF.\n");
                return false;
            }
            ++state.pos;
        }

        return true;
    }

    bool check_output(const std::string& input, std::size_t num_dims)
    {
        ParserState state;
        state.line = 1;
        state.pos = 0;
        state.alignement = 0;
        state.largest_element_seen = false;
        state.total_dims = num_dims;

        if (!dimension(input, state, num_dims)) {
            return false;
        }
        if (!state.largest_element_seen) {
            std::cerr << "Error, elements are aligned on a wrong value.\n";
            return false;
        }
        if (state.pos != input.size()) {
            std::cerr << "Error, trailing characters.\n";
            return false;
        }
        return true;
    }
};


TEST(Print, CheckOutput)
{
    using cell = double;

    unsigned const dim0 = 10;
    unsigned const dim1 = 10;

    ddc::DiscreteDomain<DDim0> const domain_0
            = ddc::init_trivial_bounded_space(ddc::DiscreteVector<DDim0>(dim0));
    ddc::DiscreteDomain<DDim1> const domain_1
            = ddc::init_trivial_bounded_space(ddc::DiscreteVector<DDim1>(dim1));

    ddc::DiscreteDomain<DDim0, DDim1> const domain_2d(domain_0, domain_1);

    ddc::Chunk cells_in_dev_alloc("cells_in_dev", domain_2d, ddc::DeviceAllocator<cell>());
    ddc::ChunkSpan const cells_in = cells_in_dev_alloc.span_view();

    Kokkos::Random_XorShift64_Pool<> random_pool(/*seed=*/12345);
    ddc::parallel_for_each(
            domain_2d,
            KOKKOS_LAMBDA(ddc::DiscreteElement<DDim0, DDim1> const i) {
                auto generator = random_pool.get_state();
                cells_in(i) = generator.drand(-1., 1.);
                random_pool.free_state(generator);
            });

    OutputChecker checker;
    {
        std::stringstream ss;
        ss << cells_in;
        std::cout << "\n" << cells_in << "\n";
        EXPECT_TRUE(checker.check_output(ss.str(), 2));
    }
    {
        EXPECT_TRUE(checker.check_output(
                "[[[ 0.470186 -0.837013]\n"
                "  [ 0.439832  0.347536]]\n"
                "\n"
                " [[ 0.311911 -0.601701]\n"
                "  [-0.734212 -0.948613]]]",
                3));
    }
    {
        EXPECT_TRUE(checker.check_output(
                "[[[[1. 1.]\n"
                "   [1. 1.]]\n"
                "\n"
                "  [[1. 1.]\n"
                "   [1. 1.]]]\n"
                "\n"
                "\n"
                " [[[1. 1.]\n"
                "   [1. 1.]]\n"
                "\n"
                "  [[1. 1.]\n"
                "   [1. 1.]]]]",
                4));
    }
    {
        EXPECT_TRUE(checker.check_output("[]", 0));
    }
    {
        EXPECT_FALSE(checker.check_output("[] ", 0));
    }
    {
        EXPECT_FALSE(checker.check_output("[ ] ", 0));
    }
    {
        EXPECT_FALSE(checker.check_output(" [ ] ", 0));
    }
}

TEST(Print, CheckOutput6d)
{
  using cell = double;

  unsigned const dim0 = 10;
  unsigned const dim1 = 10;
  unsigned const dim2 = 10;
  unsigned const dim3 = 10;
  unsigned const dim4 = 10;
  unsigned const dim5 = 10;

  struct DDim0 {};
  struct DDim1 {};
  struct DDim2 {};
  struct DDim3 {};
  struct DDim4 {};
  struct DDim5 {};

  ddc::DiscreteDomain<DDim0> const domain_0
    = ddc::init_trivial_bounded_space(ddc::DiscreteVector<DDim0>(dim0));
  ddc::DiscreteDomain<DDim1> const domain_1
    = ddc::init_trivial_bounded_space(ddc::DiscreteVector<DDim1>(dim1));
  ddc::DiscreteDomain<DDim2> const domain_2
    = ddc::init_trivial_bounded_space(ddc::DiscreteVector<DDim2>(dim2));
  ddc::DiscreteDomain<DDim3> const domain_3
    = ddc::init_trivial_bounded_space(ddc::DiscreteVector<DDim3>(dim3));
  ddc::DiscreteDomain<DDim4> const domain_4
    = ddc::init_trivial_bounded_space(ddc::DiscreteVector<DDim4>(dim4));
  ddc::DiscreteDomain<DDim5> const domain_5
    = ddc::init_trivial_bounded_space(ddc::DiscreteVector<DDim5>(dim5));

  ddc::DiscreteDomain<DDim0, DDim1, DDim2, DDim3, DDim4, DDim5> 
    const domain_full(domain_0, domain_1, domain_2, domain_3, domain_4, domain_5);

  ddc::Chunk cells_in_dev_alloc("cells_in_dev", domain_full, ddc::DeviceAllocator<cell>());
  ddc::ChunkSpan const cells_in = cells_in_dev_alloc.span_view();

  Kokkos::Random_XorShift64_Pool<> random_pool(/*seed=*/12345);
  ddc::parallel_for_each(
      domain_full,
      KOKKOS_LAMBDA(ddc::DiscreteElement<DDim0, DDim1, DDim2, DDim3, DDim4, DDim5> const i) {
      auto generator = random_pool.get_state();
      cells_in(i) = generator.drand(-1.,1.);
      random_pool.free_state(generator);
      });

  OutputChecker checker;
  {
    std::stringstream ss;
    ss << cells_in;
    std::cout << "\n" << cells_in << "\n";
    EXPECT_TRUE(checker.check_output(ss.str(), 6));
  }
}
