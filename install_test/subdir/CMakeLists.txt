# Copyright (C) The DDC development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT

find_package(DDC 0.12 REQUIRED)

assert(DEFINED DDC_FOUND)
assert(TARGET DDC::core)
assert(TARGET DDC::fft)
