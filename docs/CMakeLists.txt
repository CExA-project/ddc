# Copyright (C) The DDC development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.25)

if(NOT ("${DOXYGEN_VERSION}" VERSION_EQUAL "1.9.8"))
    message(WARNING "The documentation works best with Doxygen 1.9.8")
endif()

file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/html")
set(DOXYGEN_AUTOLINK_SUPPORT YES)
set(DOXYGEN_BUILTIN_STL_SUPPORT YES)
if(
    "${DOXYGEN_VERSION}" VERSION_GREATER_EQUAL 1.9.4
    AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang"
    AND "${CMAKE_EXPORT_COMPILE_COMMANDS}"
)
    # Doxygen <= 1.9.4 suffers from https://github.com/doxygen/doxygen/issues/8784
    set(DOXYGEN_CLANG_ASSISTED_PARSING YES)
    set(DOXYGEN_CLANG_DATABASE_PATH "${CMAKE_BINARY_DIR}")
endif()
set(DOXYGEN_ENUM_VALUES_PER_LINE 1)
set(DOXYGEN_EXAMPLE_PATH "${DDC_SOURCE_DIR}/examples")
set(DOXYGEN_EXPAND_ONLY_PREDEF YES)
set(DOXYGEN_EXTRACT_ALL YES CACHE STRING "")
set(DOXYGEN_EXCLUDE_SYMBOLS "detail")
set(DOXYGEN_EXCLUDE_PATTERNS "*/detail/*;*/experimental/*;*/kokkos-kernels-ext/*")
set(DOXYGEN_EXTRACT_LOCAL_CLASSES YES)
set(DOXYGEN_FULL_PATH_NAMES NO)
set(DOXYGEN_HTML_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/html")
set(DOXYGEN_HTML_TIMESTAMP NO)
set(DOXYGEN_INCLUDE_GRAPH NO)
set(DOXYGEN_INCLUDE_PATH "${DDC_SOURCE_DIR}/include/")
set(DOXYGEN_INCLUDED_BY_GRAPH NO)
set(DOXYGEN_INLINE_SIMPLE_STRUCTS YES)
set(DOXYGEN_JAVADOC_AUTOBRIEF YES)
set(DOXYGEN_MACRO_EXPANSION YES)
set(DOXYGEN_PREDEFINED "DOXYGEN_IGNORE")
set(DOXYGEN_PROJECT_LOGO "${CMAKE_CURRENT_SOURCE_DIR}/_template/logo.png")
set(DOXYGEN_PROJECT_NUMBER "${DDC_VERSION}")
set(DOXYGEN_QT_AUTOBRIEF YES)
set(DOXYGEN_RECURSIVE YES)
set(DOXYGEN_SEARCHENGINE YES)
set(DOXYGEN_SHOW_FILES NO)
set(DOXYGEN_SHOW_INCLUDE_FILES NO)
set(DOXYGEN_SORT_MEMBER_DOCS NO)
set(DOXYGEN_SORT_MEMBERS_CTORS_1ST YES)
set(DOXYGEN_SOURCE_BROWSER YES)
set(DOXYGEN_SOURCE_TOOLTIPS YES)
set(DOXYGEN_TOC_INCLUDE_HEADINGS 4)
set(DOXYGEN_TYPEDEF_HIDES_STRUCT YES)
set(DOXYGEN_UML_LIMIT_NUM_FIELDS 20)
set(DOXYGEN_UML_LOOK YES)
set(DOXYGEN_USE_MATHJAX YES)
set(DOXYGEN_WARN_IF_UNDOCUMENTED NO CACHE STRING "")
set(DOXYGEN_WARN_IF_DOC_ERROR YES CACHE STRING "")
set(DOXYGEN_WARN_IF_INCOMPLETE_DOC YES CACHE STRING "")
set(DOXYGEN_WARN_NO_PARAMDOC YES CACHE STRING "")
set(DOXYGEN_WARN_IF_UNDOC_ENUM_VAL YES CACHE STRING "")
set(DOXYGEN_WARN_AS_ERROR YES CACHE STRING "")
set(DOXYGEN_WARN_LOGFILE "doxygen.log")
set(DOXYGEN_IMAGE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/images")

set(DOXYGEN_GENERATE_TREEVIEW YES) # required!
set(DOXYGEN_DISABLE_INDEX NO)
set(DOXYGEN_FULL_SIDEBAR NO)
set(DOXYGEN_HTML_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/_template/header.html")
set(DOXYGEN_HTML_FOOTER "${CMAKE_CURRENT_SOURCE_DIR}/_template/footer.html")
set(DOXYGEN_HTML_EXTRA_STYLESHEET
    "${CMAKE_CURRENT_SOURCE_DIR}/_template/custom.css;${PROJECT_SOURCE_DIR}/vendor/doxygen-awesome-css/doxygen-awesome.css;${PROJECT_SOURCE_DIR}/vendor/doxygen-awesome-css/doxygen-awesome-sidebar-only.css;${PROJECT_SOURCE_DIR}/vendor/doxygen-awesome-css/doxygen-awesome-sidebar-only-darkmode-toggle.css"
)
set(DOXYGEN_HTML_COLORSTYLE LIGHT) # required with Doxygen >= 1.9.5
set(DOXYGEN_HTML_EXTRA_FILES
    "${PROJECT_SOURCE_DIR}/vendor/doxygen-awesome-css/doxygen-awesome-darkmode-toggle.js;${PROJECT_SOURCE_DIR}/vendor/doxygen-awesome-css/doxygen-awesome-fragment-copy-button.js;${PROJECT_SOURCE_DIR}/vendor/doxygen-awesome-css/doxygen-awesome-paragraph-link.js;${PROJECT_SOURCE_DIR}/vendor/doxygen-awesome-css/doxygen-awesome-interactive-toc.js"
)

doxygen_add_docs(
    doc
    "${CMAKE_CURRENT_SOURCE_DIR}/getting_started.md"
    "${CMAKE_CURRENT_SOURCE_DIR}/about.md"
    "${CMAKE_CURRENT_SOURCE_DIR}/first_steps.md"
    "${CMAKE_CURRENT_SOURCE_DIR}/going_further.md"
    "${CMAKE_CURRENT_SOURCE_DIR}/uniform_heat_equation.md"
    "${CMAKE_CURRENT_SOURCE_DIR}/non_uniform_heat_equation.md"
    "${CMAKE_CURRENT_SOURCE_DIR}/concepts.md"
    "${CMAKE_CURRENT_SOURCE_DIR}/heat_equation_spectral.md"
    "${CMAKE_CURRENT_SOURCE_DIR}/examples.md"
    "${CMAKE_CURRENT_SOURCE_DIR}/using_ddc.md"
    "${CMAKE_CURRENT_SOURCE_DIR}/installation.md"
    "${DDC_SOURCE_DIR}/include/ddc/"
    ALL
)
