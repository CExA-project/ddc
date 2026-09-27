// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <cstddef>
#include <filesystem>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

import ddc.core;
import ddc.core.detail;

TEST(SaveNpy, LargeHeader)
{
    ddc::detail::NpyArrayView const view {
            .data = nullptr,
            .dtype = ddc::detail::convert_to_npy_dtype<char>(),
            .shape = std::vector<std::size_t>(25'000, 0),
            .fortran_order = true,
    };
    EXPECT_THROW(ddc::detail::save_npy("test.npy", view), std::runtime_error);
}
