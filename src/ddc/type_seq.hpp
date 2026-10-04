// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <limits>
#include <tuple>
#include <type_traits>
#include <utility>

#include <ddc/config.hpp>

namespace ddc {

/// @brief A compile-time sequence of types.
///
/// `TypeSeq` is used to represent an ordered collection of types at compile time.
/// It is primarily intended for manipulating lists of tags or types in generic
/// programming code.
///
/// The order of the types is significant for operations such as
/// @ref type_seq_rank_v, @ref type_seq_element_t, and the sequence manipulation
/// utilities.
///
/// @tparam Tags The types contained in the sequence.
template <class... Tags>
struct TypeSeq
{
};

namespace detail {

#if DDC_BUILD_DEPRECATED_CODE()
template <class... Tags>
using TypeSeq [[deprecated("Use `ddc::TypeSeq' instead")]] = ::ddc::TypeSeq<Tags...>;
#endif

template <class Tag>
struct SingleType
{
};

template <class QueryTagSeq, class TagSeq>
struct TypeSeqRank
{
};

template <class QueryTag>
struct TypeSeqRank<SingleType<QueryTag>, ddc::TypeSeq<>>
{
    static constexpr bool present = false;
    static constexpr std::size_t val = std::numeric_limits<std::size_t>::max();
};

template <class QueryTag, class... TagsTail>
struct TypeSeqRank<SingleType<QueryTag>, ddc::TypeSeq<QueryTag, TagsTail...>>
{
    static constexpr bool present = true;
    static constexpr std::size_t val = 0;
};

template <class QueryTag, class TagsHead, class... TagsTail>
struct TypeSeqRank<SingleType<QueryTag>, ddc::TypeSeq<TagsHead, TagsTail...>>
{
    static constexpr bool present
            = TypeSeqRank<SingleType<QueryTag>, ddc::TypeSeq<TagsTail...>>::present;
    static constexpr std::size_t val
            = present ? 1 + TypeSeqRank<SingleType<QueryTag>, ddc::TypeSeq<TagsTail...>>::val
                      : std::numeric_limits<std::size_t>::max();
};

template <class... QueryTags, class... Tags>
struct TypeSeqRank<ddc::TypeSeq<QueryTags...>, ddc::TypeSeq<Tags...>>
{
    using ValSeq = std::index_sequence<TypeSeqRank<QueryTags, ddc::TypeSeq<Tags...>>::val...>;
};

template <std::size_t I, class TagSeq>
struct TypeSeqElement
{
};

template <std::size_t I, class... Tags>
struct TypeSeqElement<I, ddc::TypeSeq<Tags...>>
{
    using type = std::tuple_element_t<I, std::tuple<Tags...>>;
};

/// R contains all elements in A that are not in B.
/// Remark 1: This operation preserves the order from A.
/// Remark 2: It is similar to the set difference in the set theory (R = A\\B).
/// Example: A = [a, b, c], B = [z, c, y], R = [a, b]
template <class TagSeqA, class TagSeqB, class TagSeqR>
struct TypeSeqRemove
{
};

template <class... TagsB, class... TagsR>
struct TypeSeqRemove<ddc::TypeSeq<>, ddc::TypeSeq<TagsB...>, ddc::TypeSeq<TagsR...>>
{
    using type = ddc::TypeSeq<TagsR...>;
};

template <class HeadTagsA, class... TailTagsA, class... TagsB, class... TagsR>
struct TypeSeqRemove<
        ddc::TypeSeq<HeadTagsA, TailTagsA...>,
        ddc::TypeSeq<TagsB...>,
        ddc::TypeSeq<TagsR...>>
    : std::conditional_t<
              TypeSeqRank<detail::SingleType<HeadTagsA>, ddc::TypeSeq<TagsB...>>::present,
              TypeSeqRemove<
                      ddc::TypeSeq<TailTagsA...>,
                      ddc::TypeSeq<TagsB...>,
                      ddc::TypeSeq<TagsR...>>,
              TypeSeqRemove<
                      ddc::TypeSeq<TailTagsA...>,
                      ddc::TypeSeq<TagsB...>,
                      ddc::TypeSeq<TagsR..., HeadTagsA>>>
{
};

/// R contains all elements in A and elements in B that are not in A.
/// Remark 1: This operation preserves the order from A.
/// Remark 2: It is similar to the set union in the set theory (R = AUB).
/// Example: A = [a, b, c], B = [z, c, y], R = [a, b, c, z, y]
template <class TagSeqA, class TagSeqB, class TagSeqR>
struct TypeSeqMerge
{
};

template <class... TagsA, class... TagsR>
struct TypeSeqMerge<ddc::TypeSeq<TagsA...>, ddc::TypeSeq<>, ddc::TypeSeq<TagsR...>>
{
    using type = ddc::TypeSeq<TagsR...>;
};

template <class... TagsA, class HeadTagsB, class... TailTagsB, class... TagsR>
struct TypeSeqMerge<
        ddc::TypeSeq<TagsA...>,
        ddc::TypeSeq<HeadTagsB, TailTagsB...>,
        ddc::TypeSeq<TagsR...>>
    : std::conditional_t<
              TypeSeqRank<detail::SingleType<HeadTagsB>, ddc::TypeSeq<TagsA...>>::present,
              TypeSeqMerge<
                      ddc::TypeSeq<TagsA...>,
                      ddc::TypeSeq<TailTagsB...>,
                      ddc::TypeSeq<TagsR...>>,
              TypeSeqMerge<
                      ddc::TypeSeq<TagsA...>,
                      ddc::TypeSeq<TailTagsB...>,
                      ddc::TypeSeq<TagsR..., HeadTagsB>>>
{
};

/// `type` contains all elements in A then all elements in B.
/// Example: A = [a, b, c], B = [z, c, y], returned type [a, b, c, z, c, y]
template <class TagSeqA, class TagSeqB>
struct TypeSeqCat
{
};

template <class... TagsA, class... TagsB>
struct TypeSeqCat<ddc::TypeSeq<TagsA...>, ddc::TypeSeq<TagsB...>>
{
    using type = ddc::TypeSeq<TagsA..., TagsB...>;
};

/// A is replaced by element of C at same position than the first element of B equal to A.
/// Remark : It may not be useful in its own, it is an helper for TypeSeqReplace
template <class TagA, class TagSeqB, class TagSeqC>
struct TypeSeqReplaceSingle
{
};

template <class TagA>
struct TypeSeqReplaceSingle<TagA, ddc::TypeSeq<>, ddc::TypeSeq<>>
{
    using type = TagA;
};

template <class TagA, class HeadTagsB, class... TailTagsB, class HeadTagsC, class... TailTagsC>
struct TypeSeqReplaceSingle<
        TagA,
        ddc::TypeSeq<HeadTagsB, TailTagsB...>,
        ddc::TypeSeq<HeadTagsC, TailTagsC...>>
    : std::conditional_t<
              std::is_same_v<TagA, HeadTagsB>,
              TypeSeqReplaceSingle<HeadTagsC, ddc::TypeSeq<>, ddc::TypeSeq<>>,
              TypeSeqReplaceSingle<TagA, ddc::TypeSeq<TailTagsB...>, ddc::TypeSeq<TailTagsC...>>>
{
};

/// R contains all elements of A except those of B which are replaced by those of C.
/// Remark : This operation preserves the orders.
template <class TagSeqA, class TagSeqB, class TagSeqC, class TagSeqR>
struct TypeSeqReplace
{
};

template <class... TagsB, class... TagsC, class... TagsR>
struct TypeSeqReplace<
        ddc::TypeSeq<>,
        ddc::TypeSeq<TagsB...>,
        ddc::TypeSeq<TagsC...>,
        ddc::TypeSeq<TagsR...>>
{
    using type = ddc::TypeSeq<TagsR...>;
};

template <class HeadTagsA, class... TailTagsA, class... TagsB, class... TagsC, class... TagsR>
struct TypeSeqReplace<
        ddc::TypeSeq<HeadTagsA, TailTagsA...>,
        ddc::TypeSeq<TagsB...>,
        ddc::TypeSeq<TagsC...>,
        ddc::TypeSeq<TagsR...>>
    : TypeSeqReplace<
              ddc::TypeSeq<TailTagsA...>,
              ddc::TypeSeq<TagsB...>,
              ddc::TypeSeq<TagsC...>,
              ddc::TypeSeq<
                      TagsR...,
                      typename TypeSeqReplaceSingle<
                              HeadTagsA,
                              ddc::TypeSeq<TagsB...>,
                              ddc::TypeSeq<TagsC...>>::type>>
{
};

template <class T>
struct ToTypeSeq
{
};

template <class T, class TagSeq>
struct Rebind
{
};

template <class T, class TagSeq>
using rebind_t = Rebind<T, TagSeq>::type;

} // namespace detail

template <class TypeSeq>
constexpr std::size_t type_seq_size_v = std::numeric_limits<std::size_t>::max();

/// Returns the number of types in a @ref TypeSeq.
template <class... Tags>
constexpr std::size_t type_seq_size_v<TypeSeq<Tags...>> = sizeof...(Tags);

template <class QueryTag, class TypeSeq>
constexpr std::size_t type_seq_rank_v = std::numeric_limits<std::size_t>::max();

template <class QueryTag, class OTypeSeq>
constexpr bool in_tags_v = false;

template <class TypeSeq, class OTypeSeq>
constexpr bool type_seq_contains_v = false;

template <class TypeSeq>
constexpr bool type_seq_is_unique_v = false;

template <class TypeSeq, class B>
constexpr bool type_seq_same_v = type_seq_contains_v<TypeSeq, B> && type_seq_contains_v<B, TypeSeq>;

template <class QueryTag, class... Tags>
constexpr bool in_tags_v<QueryTag, TypeSeq<Tags...>>
        = detail::TypeSeqRank<detail::SingleType<QueryTag>, TypeSeq<Tags...>>::present;

template <class... Tags, class OTypeSeq>
constexpr bool type_seq_contains_v<TypeSeq<Tags...>, OTypeSeq>
        = (detail::TypeSeqRank<detail::SingleType<Tags>, OTypeSeq>::present && ...);

template <class QueryTag, class... Tags>
constexpr std::size_t type_seq_rank_v<QueryTag, TypeSeq<Tags...>>
        = detail::TypeSeqRank<detail::SingleType<QueryTag>, TypeSeq<Tags...>>::val;

template <std::size_t I, class TagSeq>
using type_seq_element_t = detail::TypeSeqElement<I, TagSeq>::type;

template <class TagSeqA, class TagSeqB>
using type_seq_remove_t = detail::TypeSeqRemove<TagSeqA, TagSeqB, TypeSeq<>>::type;

template <class TagSeqA, class TagSeqB>
using type_seq_merge_t = detail::TypeSeqMerge<TagSeqA, TagSeqB, TagSeqA>::type;

template <class TagSeqA, class TagSeqB>
using type_seq_cat_t = detail::TypeSeqCat<TagSeqA, TagSeqB>::type;

template <class TagSeqA, class TagSeqB, class TagSeqC>
using type_seq_replace_t = detail::TypeSeqReplace<TagSeqA, TagSeqB, TagSeqC, TypeSeq<>>::type;

template <class... Tags>
constexpr bool type_seq_is_unique_v<TypeSeq<Tags...>>
        = ((type_seq_size_v<type_seq_remove_t<TypeSeq<Tags...>, TypeSeq<Tags>>>
            == sizeof...(Tags) - 1)
           && ...);

template <class T>
using to_type_seq_t = detail::ToTypeSeq<T>::type;

} // namespace ddc
