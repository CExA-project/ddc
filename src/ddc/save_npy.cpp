#include <bit>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>

#include "save_npy.hpp"

namespace ddc::detail {

NpyByteOrder get_byte_order() noexcept
{
    return (std::endian::native == std::endian::little) ? NpyByteOrder::little_endian
                                                        : NpyByteOrder::big_endian;
}

std::string NpyDtype::str() const
{
    return std::string(1, static_cast<char>(byte_order)) + static_cast<char>(kind)
           + std::to_string(itemsize);
}

void save_npy(std::ostream& os, NpyArrayView const& view)
{
    // Build shape string: (d0, d1, ..., dN,)
    std::string shape_str = "(";
    for (std::size_t ext : view.shape) {
        shape_str += std::to_string(ext);
        shape_str += ", ";
    }
    shape_str += ")";

    std::string header_dict = std::string("{'descr': '") + view.dtype.str()
                              + "', 'fortran_order': " + (view.fortran_order ? "True" : "False")
                              + ", 'shape': " + shape_str + ", }";

    // Pad header to a multiple of 64 bytes
    constexpr std::size_t prefix_size = 6 + 1 + 1 + 2; // magic + major + minor + hlen
    std::size_t const total_header = prefix_size + header_dict.size() + 1; // +1 for '\n'
    std::size_t const padded = ((total_header + 63) / 64) * 64;
    header_dict += std::string(padded - total_header, ' ');
    header_dict += '\n';

    if (header_dict.size() > std::numeric_limits<std::uint16_t>::max()) {
        throw std::runtime_error("save_npy: header too large for npy v1.0.");
    }
    auto const hlen = static_cast<std::uint16_t>(header_dict.size());

    // Magic + version
    os.write("\x93NUMPY", 6);
    std::uint8_t const major = 1;
    std::uint8_t const minor = 0;
    os.write(reinterpret_cast<char const*>(&major), 1);
    os.write(reinterpret_cast<char const*>(&minor), 1);

    // Header length + content
    os.write(reinterpret_cast<char const*>(&hlen), sizeof(hlen));
    os.write(header_dict.data(), header_dict.size());

    // Raw data
    std::size_t const n_elems
            = std::accumulate(view.shape.begin(), view.shape.end(), 1ULL, std::multiplies<> {});
    os.write(reinterpret_cast<char const*>(view.data), n_elems * view.dtype.itemsize);
}

} // namespace ddc::detail
