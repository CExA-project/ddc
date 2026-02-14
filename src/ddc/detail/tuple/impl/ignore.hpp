#pragma once

#include "macros.hpp"

namespace cexa {
namespace impl {
struct ignore_t {
  KOKKOS_DEFAULTED_FUNCTION constexpr ignore_t() = default;
  KOKKOS_DEFAULTED_FUNCTION
#if defined(CEXA_HAS_CXX20)
  constexpr
#endif
      ~ignore_t() = default;

  template <typename T>
  KOKKOS_INLINE_FUNCTION constexpr const ignore_t& operator=(
      const T&) const noexcept {
    return *this;
  }
};
}  // namespace impl

// TODO: check if we need a host and a device version of this
inline constexpr impl::ignore_t ignore;
}  // namespace cexa
