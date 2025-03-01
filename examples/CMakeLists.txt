# Copyright (C) The DDC development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.22)

add_executable(game_of_life game_of_life.cpp)
target_link_libraries(game_of_life PUBLIC DDC::core)

add_executable(heat_equation heat_equation.cpp)
target_link_libraries(heat_equation PUBLIC DDC::core)

add_executable(uniform_heat_equation uniform_heat_equation.cpp)
target_link_libraries(uniform_heat_equation PUBLIC DDC::core)

add_executable(non_uniform_heat_equation non_uniform_heat_equation.cpp)
target_link_libraries(non_uniform_heat_equation PUBLIC DDC::core)
