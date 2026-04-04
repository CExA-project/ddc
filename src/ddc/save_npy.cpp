// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <bit>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "save_npy.hpp"

namespace ddc::detail {

NpyByteOrder get_byte_order(std::size_t itemsize) noexcept
{
    if (itemsize == 1) {
        return NpyByteOrder::not_applicable;
    }

    if (std::endian::native == std::endian::little) {
        return NpyByteOrder::little_endian;
    }

    return NpyByteOrder::big_endian;
}

std::string NpyDtype::str() const
{
    return std::string(1, static_cast<char>(byte_order)) + static_cast<char>(kind)
           + std::to_string(itemsize);
}

// See specification at https://numpy.org/neps/nep-0001-npy-format.html#format-specification-version-1-0
void save_npy(std::ostream& os, NpyArrayView const& view)
{
    // Build shape string: (d0, d1, ..., dN,)
    std::string shape_str = "(";
    for (std::size_t ext : view.shape) {
        shape_str += std::to_string(ext);
        shape_str += ", ";
    }
    shape_str += ")";

    std::string const header_dict
            = std::string("{'descr': '") + view.dtype.str() + "', 'fortran_order': "
              + (view.fortran_order ? "True" : "False") + ", 'shape': " + shape_str + ", }";

    // Pad header to a multiple of 16
    std::size_t const non_padded_header_len = header_dict.size() + 1;
    // magic(6) + major(1) + minor(1) + header_len(2) + header
    std::size_t const padding = 16 - (6 + 1 + 1 + 2 + non_padded_header_len) % 16;
    if (non_padded_header_len + padding > std::numeric_limits<std::uint16_t>::max()) {
        throw std::runtime_error("save_npy: header too large for npy v1.0.");
    }
    auto const header_len = static_cast<std::uint16_t>(non_padded_header_len + padding);

    // magic string
    os.write("\x93NUMPY", 6);
    // major version
    os.put(1);
    // minor version
    os.put(0);

    // header length
    os.write(reinterpret_cast<char const*>(&header_len), sizeof(header_len));
    // header + padding + newline
    os.write(header_dict.data(), header_dict.size());
    os.write("               ", padding);
    os.put('\n');

    // Raw data
    std::size_t const n_elems
            = std::accumulate(view.shape.begin(), view.shape.end(), 1ULL, std::multiplies<> {});
    os.write(reinterpret_cast<char const*>(view.data), n_elems * view.dtype.itemsize);
}

void save_npy(std::filesystem::path const& filename, NpyArrayView const& view)
{
    std::ofstream file(filename, std::ios::binary);
    file.exceptions(std::ios::failbit | std::ios::badbit);

    save_npy(file, view);
}

} // namespace ddc::detail
