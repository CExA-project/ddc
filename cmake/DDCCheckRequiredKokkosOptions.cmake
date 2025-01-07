# Copyright (C) The DDC development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT

function(ddc_check_required_kokkos_options)
    kokkos_check(DEVICES CUDA RETURN_VALUE is_cuda_enabled)
    if("${is_cuda_enabled}")
        kokkos_check(OPTIONS CUDA_CONSTEXPR CUDA_RELOCATABLE_DEVICE_CODE)
    endif()
    kokkos_check(DEVICES HIP RETURN_VALUE is_hip_enabled)
    if("${is_hip_enabled}")
        kokkos_check(OPTIONS HIP_RELOCATABLE_DEVICE_CODE)
    endif()
endfunction()
