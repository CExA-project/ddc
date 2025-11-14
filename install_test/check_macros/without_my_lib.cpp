// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <iostream>

#include <ginkgo/extensions/kokkos.hpp>
#include <ginkgo/ginkgo.hpp>

#include <KokkosBatched_Laswp.hpp>
#include <KokkosBatched_Pbtrs.hpp>
#include <KokkosBatched_Pttrs.hpp>
#include <KokkosBatched_Tbsv.hpp>
#include <KokkosBatched_Trsm_Decl.hpp>
#include <KokkosBatched_Util.hpp>
#include <KokkosBlas2_gemv.hpp>
#include <KokkosFFT.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>

#if __has_include(<mkl_lapacke.h>)
#    include <mkl_lapacke.h>
#else
#    include <lapacke.h>
#endif
#include <any>
#include <list>

#include <Kokkos_DualView.hpp>
#include <paraconf.h>
#include <pdi.h>
