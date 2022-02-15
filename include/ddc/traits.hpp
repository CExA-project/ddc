#pragma once

#include <type_traits>

#include "non_uniform_discretization.hpp"
#include "uniform_discretization.hpp"

template <class T>
struct is_uniform_domain : std::false_type
{
};

template <class... DDims>
struct is_uniform_domain<DiscreteDomain<DDims...>>
    : std::conditional_t<
              (is_uniform_disretization_v<DDims> && ...),
              std::true_type,
              std::false_type>
{
};

template <class T>
constexpr bool is_uniform_domain_v = is_uniform_domain<T>::value;

template <class T>
struct is_rectilinear_domain : std::false_type
{
};

template <class... DDims>
struct is_rectilinear_domain<DiscreteDomain<DDims...>>
    : std::conditional_t<
              ((is_uniform_disretization_v<DDims> || is_non_uniform_disretization_v<DDims>)&&...),
              std::true_type,
              std::false_type>
{
};

template <class T>
constexpr bool is_rectilinear_domain_v = is_uniform_domain<T>::value;
