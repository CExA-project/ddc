// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

module;

#include <ddc/ddc.hpp>

export module ddc.core;

export namespace ddc {

using ::ddc::AlignedAllocator;
using ::ddc::Chunk;
using ::ddc::chunk_value_t;
using ::ddc::ChunkSpan;
using ::ddc::combine_t;
using ::ddc::Coordinate;
using ::ddc::coordinate;
using ::ddc::create_mirror;
using ::ddc::create_mirror_and_copy;
using ::ddc::create_mirror_view;
using ::ddc::create_mirror_view_and_copy;
using ::ddc::device_for_each;
using ::ddc::device_transform_reduce;
using ::ddc::DeviceAllocator;
using ::ddc::discrete_space;
using ::ddc::DiscreteDomain;
using ::ddc::DiscreteElement;
using ::ddc::DiscreteElementType;
using ::ddc::DiscreteVector;
using ::ddc::DiscreteVectorElement;
using ::ddc::distance_at_left;
using ::ddc::distance_at_right;
using ::ddc::get;
using ::ddc::get_domain;
using ::ddc::get_print_options;
using ::ddc::host_discrete_space;
using ::ddc::host_for_each;
using ::ddc::host_for_each_block;
using ::ddc::host_transform_reduce;
using ::ddc::HostAllocator;
using ::ddc::init_discrete_space;
using ::ddc::init_trivial_bounded_space;
using ::ddc::init_trivial_half_bounded_space;
using ::ddc::is_discrete_space_initialized;
using ::ddc::is_writable_chunk_v;
using ::ddc::KokkosAllocator;
using ::ddc::origin;
using ::ddc::parallel_copy;
using ::ddc::parallel_deepcopy;
using ::ddc::parallel_fill;
using ::ddc::parallel_for_each;
using ::ddc::parallel_transform;
using ::ddc::parallel_transform_reduce;
using ::ddc::print;
using ::ddc::print_content;
using ::ddc::print_full;
using ::ddc::print_type_info;
using ::ddc::PrinterOptions;
using ::ddc::prod;
using ::ddc::Real;
using ::ddc::remove_dims_of;
using ::ddc::remove_dims_of_t;
using ::ddc::replace_dim_of;
using ::ddc::replace_dim_of_t;
using ::ddc::rlength;
using ::ddc::rmax;
using ::ddc::rmin;
using ::ddc::ScopeGuard;
using ::ddc::select;
using ::ddc::set_print_options;
using ::ddc::SparseDiscreteDomain;
using ::ddc::step;
using ::ddc::StridedDiscreteDomain;
using ::ddc::TypeSeq;

using ::ddc::type_seq_cat_t;
using ::ddc::type_seq_element_t;
using ::ddc::type_seq_is_unique_v;
using ::ddc::type_seq_merge_t;
using ::ddc::type_seq_rank_v;
using ::ddc::type_seq_remove_t;
using ::ddc::type_seq_replace_t;
using ::ddc::type_seq_same_v;
using ::ddc::type_seq_size_v;

using ::ddc::operator<<;
using ::ddc::operator-;
using ::ddc::operator+;
using ::ddc::operator*;
using ::ddc::operator==;
using ::ddc::operator<=;
using ::ddc::operator<;
using ::ddc::operator>=;
using ::ddc::operator>;

using ::ddc::NonUniformPointSampling;
using ::ddc::PeriodicSampling;
using ::ddc::UniformPointSampling;

namespace concepts {

using ::ddc::concepts::borrowed_chunk;
using ::ddc::concepts::discrete_domain;
using ::ddc::concepts::discrete_element;
using ::ddc::concepts::discrete_vector;
using ::ddc::concepts::non_uniform_point_sampling;
using ::ddc::concepts::periodic_sampling;
using ::ddc::concepts::sparse_discrete_domain;
using ::ddc::concepts::strided_discrete_domain;
using ::ddc::concepts::uniform_point_sampling;

} // namespace concepts

namespace reducer {

using ::ddc::reducer::band;
using ::ddc::reducer::bor;
using ::ddc::reducer::bxor;
using ::ddc::reducer::land;
using ::ddc::reducer::lor;
using ::ddc::reducer::max;
using ::ddc::reducer::min;
using ::ddc::reducer::minmax;
using ::ddc::reducer::prod;
using ::ddc::reducer::sum;

} // namespace reducer

} // namespace ddc
