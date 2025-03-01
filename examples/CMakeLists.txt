# Copyright (C) The DDC development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.22)

add_executable(game_of_life game_of_life.cpp)
target_link_libraries(game_of_life PUBLIC DDC::core)
