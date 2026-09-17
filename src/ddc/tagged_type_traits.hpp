// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

namespace ddc {

namespace detail {

template <class... TaggedTypes>
struct Combine;

template <class TaggedType>
struct Combine<TaggedType>
{
    using type = TaggedType;
};

template <
        template <typename...> class TaggedType,
        class... Tags,
        class... OTags,
        class... TaggedTypes>
struct Combine<TaggedType<Tags...>, TaggedType<OTags...>, TaggedTypes...>
{
    using type = Combine<TaggedType<Tags..., OTags...>, TaggedTypes...>::type;
};

} // namespace detail

/**
 * @brief A helper structure to determine the type when combining tags across containers.
 *
 * E.g. combine_t<DiscreteElement<Tag1>, DiscreteElement<Tag2>> == DiscreteElement<Tag1, Tag2>
 *
 * @tparam Containers The Types that should be combined. They should differ only in the tags.
 */
template <class... TaggedTypes>
using combine_t = detail::Combine<TaggedTypes...>::type;

} // namespace ddc
