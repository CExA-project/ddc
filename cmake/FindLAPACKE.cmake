# FindLAPACKE.cmake
# ----------------
# Finds the LAPACKE library (C interface to LAPACK).
#
# LAPACKE may be bundled inside an existing LAPACK/BLAS provider (e.g.
# OpenBLAS) or shipped as an independent liblapacke. This module first
# delegates to CMake's own FindLAPACK to locate the LAPACK libraries, then
# checks whether LAPACKE symbols are already present in those libraries
# before falling back to searching for a standalone liblapacke.
#
# The header lapacke.h is always resolved from the shared object that actually
# provides the LAPACKE symbols, so that the include path and the library stay
# in sync.
#
# Imported target
# ^^^^^^^^^^^^^^^
#   LAPACKE::LAPACKE
#     Compile-and-link target. Carries the correct include directory and links
#     to whichever library (bundled or standalone) provides LAPACKE.
#
# Result variables
# ^^^^^^^^^^^^^^^^
#   LAPACKE_FOUND          - TRUE when both the library and the header are found
#   LAPACKE_LIBRARIES      - Full list of libraries needed to use LAPACKE
#                            (may re-use LAPACK_LIBRARIES when bundled)
#   LAPACKE_INCLUDE_DIRS   - Include directory containing lapacke.h
#
# Hints
# ^^^^^
#   LAPACKE_ROOT / ENV{LAPACKE_ROOT}
#     Prefix searched first for the standalone library and header.
#   LAPACK_ROOT / ENV{LAPACK_ROOT}  (also honoured by FindLAPACK)
#     Prefix for the underlying LAPACK provider.
#
# Notes
# ^^^^^
# * BLA_VENDOR (see FindLAPACK docs) can be set before find_package(LAPACKE)
#   to steer which BLAS/LAPACK flavour is considered, e.g.
#     set(BLA_VENDOR OpenBLAS)
#     find_package(LAPACKE REQUIRED)
# * The check_function_exists() probes require try_compile, so the C language
#   must be enabled in the calling project.

cmake_minimum_required(VERSION 3.14)

include(CheckFunctionExists)
include(FindPackageHandleStandardArgs)

# ---------------------------------------------------------------------------
# Internal helper: look for lapacke.h relative to a given library path.
# Sets _lapacke_include_dir in the parent scope if found.
# ---------------------------------------------------------------------------
function(_lapacke_find_header_near_lib lib_path)
    if(NOT lib_path)
        return()
    endif()

    # Resolve symlinks so we always work with the real filesystem layout.
    get_filename_component(_lib_real "${lib_path}" REALPATH)
    get_filename_component(_lib_dir  "${_lib_real}" DIRECTORY)

    # Typical install layouts relative to the lib directory:
    #   <prefix>/lib/         -> <prefix>/include/
    #   <prefix>/lib64/       -> <prefix>/include/
    #   <prefix>/lib/openblas -> <prefix>/include/openblas  (OpenBLAS multiarch)
    set(_candidate_roots
        "${_lib_dir}/.."          # standard: lib -> ..
        "${_lib_dir}/../.."       # multiarch: lib/x86_64-linux-gnu -> ../..
        "${_lib_dir}"             # header shipped alongside the .so (rare)
    )

    set(_candidate_subdirs
        "include"
        "include/openblas"
        "include/lapacke"
        "include/lapack"
        ""                        # root itself (e.g. when _lib_dir IS include)
    )

    foreach(_root IN LISTS _candidate_roots)
        foreach(_sub IN LISTS _candidate_subdirs)
            if(_sub STREQUAL "")
                set(_dir "${_root}")
            else()
                set(_dir "${_root}/${_sub}")
            endif()
            get_filename_component(_dir "${_dir}" ABSOLUTE)
            if(EXISTS "${_dir}/lapacke.h")
                set(_lapacke_include_dir "${_dir}" PARENT_SCOPE)
                return()
            endif()
        endforeach()
    endforeach()
endfunction()

# ---------------------------------------------------------------------------
# Step 1 – Locate LAPACK (and its transitive BLAS dependency).
# ---------------------------------------------------------------------------
if(NOT LAPACK_FOUND)
    find_package(LAPACK QUIET)
endif()

# ---------------------------------------------------------------------------
# Step 2 – Check whether LAPACKE is already bundled in the LAPACK libraries.
#
# OpenBLAS and MKL both ship LAPACKE symbols inside their main library, so
# we avoid adding a redundant -llapacke in those cases.
# ---------------------------------------------------------------------------
set(_lapacke_bundled FALSE)

if(LAPACK_FOUND)
    # Save/restore CMAKE_REQUIRED_* to avoid polluting the caller's state.
    set(_saved_req_libs "${CMAKE_REQUIRED_LIBRARIES}")
    set(CMAKE_REQUIRED_LIBRARIES ${LAPACK_LIBRARIES})

    # LAPACK_LIBRARIES may already contain the BLAS libraries; if not, add them.
    if(BLAS_LIBRARIES)
        list(APPEND CMAKE_REQUIRED_LIBRARIES ${BLAS_LIBRARIES})
    endif()

    # Use the Fortran-mangled name most C callers end up resolving at link time.
    # LAPACKE_dgetrf is a representative function present in every full LAPACKE
    # implementation (double precision LU factorization of a general matrix).
    unset(_lapacke_bundled_check CACHE)
    check_function_exists(LAPACKE_dgetrf _lapacke_bundled_check)

    if(_lapacke_bundled_check)
        set(_lapacke_bundled TRUE)
        set(_lapacke_lib_providing "")  # symbols live inside LAPACK_LIBRARIES
        message(STATUS "FindLAPACKE: LAPACKE symbols found inside LAPACK libraries (bundled)")
    endif()

    set(CMAKE_REQUIRED_LIBRARIES "${_saved_req_libs}")
endif()

# ---------------------------------------------------------------------------
# Step 3 – If not bundled, search for a standalone liblapacke.
# ---------------------------------------------------------------------------
if(NOT _lapacke_bundled)
    find_library(LAPACKE_LIBRARY
        NAMES lapacke
        HINTS
            ${LAPACKE_ROOT}
            ENV LAPACKE_ROOT
            ${LAPACK_ROOT}
            ENV LAPACK_ROOT
        PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu lib/aarch64-linux-gnu
    )

    if(LAPACKE_LIBRARY)
        # Verify the symbols are really there (guards against empty stub libs).
        set(_saved_req_libs "${CMAKE_REQUIRED_LIBRARIES}")
        set(CMAKE_REQUIRED_LIBRARIES "${LAPACKE_LIBRARY}" ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})
        unset(_lapacke_standalone_check CACHE)
        check_function_exists(LAPACKE_dgetrf _lapacke_standalone_check)
        set(CMAKE_REQUIRED_LIBRARIES "${_saved_req_libs}")

        if(_lapacke_standalone_check)
            set(_lapacke_lib_providing "${LAPACKE_LIBRARY}")
            message(STATUS "FindLAPACKE: found standalone LAPACKE library: ${LAPACKE_LIBRARY}")
        else()
            message(WARNING "FindLAPACKE: liblapacke found at ${LAPACKE_LIBRARY} "
                            "but LAPACKE_dgetrf is not resolvable — ignoring.")
            unset(LAPACKE_LIBRARY CACHE)
        endif()
    endif()
endif()

# ---------------------------------------------------------------------------
# Step 4 – Locate lapacke.h from the library that provides the symbols.
# ---------------------------------------------------------------------------
# Prefer an explicit hint, then probe near the providing library.
find_path(LAPACKE_INCLUDE_DIR
    NAMES lapacke.h
    HINTS
        ${LAPACKE_ROOT}
        ENV LAPACKE_ROOT
        ${LAPACK_ROOT}
        ENV LAPACK_ROOT
    PATH_SUFFIXES include include/openblas include/lapacke include/lapack
)

if(NOT LAPACKE_INCLUDE_DIR)
    # Derive the header location from the library path when the standard search
    # above came up empty (common with non-system OpenBLAS installs).
    if(_lapacke_bundled)
        # Bundled case: probe relative to each library in LAPACK_LIBRARIES.
        foreach(_lib IN LISTS LAPACK_LIBRARIES)
            if(EXISTS "${_lib}")          # skip flags like "-lpthread"
                _lapacke_find_header_near_lib("${_lib}")
                if(_lapacke_include_dir)
                    set(LAPACKE_INCLUDE_DIR "${_lapacke_include_dir}" CACHE PATH
                        "Directory containing lapacke.h" FORCE)
                    break()
                endif()
            endif()
        endforeach()
    elseif(LAPACKE_LIBRARY)
        _lapacke_find_header_near_lib("${LAPACKE_LIBRARY}")
        if(_lapacke_include_dir)
            set(LAPACKE_INCLUDE_DIR "${_lapacke_include_dir}" CACHE PATH
                "Directory containing lapacke.h" FORCE)
        endif()
    endif()
endif()

# ---------------------------------------------------------------------------
# Step 5 – Assemble result variables.
# ---------------------------------------------------------------------------
if(_lapacke_bundled)
    # All required symbols live in LAPACK_LIBRARIES; no extra -l needed.
    set(LAPACKE_LIBRARIES ${LAPACK_LIBRARIES})
elseif(LAPACKE_LIBRARY)
    # Standalone: prepend liblapacke, keep LAPACK/BLAS for transitive deps.
    set(LAPACKE_LIBRARIES ${LAPACKE_LIBRARY} ${LAPACK_LIBRARIES})
else()
    set(LAPACKE_LIBRARIES "")
endif()

set(LAPACKE_INCLUDE_DIRS "${LAPACKE_INCLUDE_DIR}")

# ---------------------------------------------------------------------------
# Step 6 – Standard "found" handling.
# ---------------------------------------------------------------------------
find_package_handle_standard_args(LAPACKE
    REQUIRED_VARS
        LAPACKE_LIBRARIES
        LAPACKE_INCLUDE_DIRS
    REASON_FAILURE_MESSAGE
        "Could not find LAPACKE. Set LAPACKE_ROOT or BLA_VENDOR to guide the search."
)

# ---------------------------------------------------------------------------
# Step 7 – Create the imported target.
# ---------------------------------------------------------------------------
if(LAPACKE_FOUND AND NOT TARGET LAPACKE::LAPACKE)
    add_library(LAPACKE::LAPACKE INTERFACE IMPORTED)
    set_target_properties(LAPACKE::LAPACKE PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${LAPACKE_INCLUDE_DIRS}"
        INTERFACE_LINK_LIBRARIES      "${LAPACKE_LIBRARIES}"
    )
endif()

# Clean up internal variables from the caller's scope.
unset(_lapacke_bundled)
unset(_lapacke_lib_providing)
unset(_lapacke_include_dir)
unset(_lapacke_bundled_check CACHE)
unset(_lapacke_standalone_check CACHE)
