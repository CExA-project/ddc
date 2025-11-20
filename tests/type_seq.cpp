// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <type_traits>

#include <ddc/ddc.hpp>

#include <gtest/gtest.h>

inline namespace anonymous_namespace_workaround_type_seq_cpp {

struct a;
struct b;
struct c;
struct d;
struct e;
struct y;
struct z;

} // namespace anonymous_namespace_workaround_type_seq_cpp

TEST(TypeSeqTest, Size)
{
    using Empty = ddc::detail::TypeSeq<>;
    using A = ddc::detail::TypeSeq<a, b, c>;
    EXPECT_EQ((ddc::type_seq_size_v<Empty>), 0);
    EXPECT_EQ((ddc::type_seq_size_v<A>), 3);
}

TEST(TypeSeqTest, Rank)
{
    using A = ddc::detail::TypeSeq<a, b, c>;
    EXPECT_EQ((ddc::type_seq_rank_v<a, A>), 0);
    EXPECT_EQ((ddc::type_seq_rank_v<b, A>), 1);
    EXPECT_EQ((ddc::type_seq_rank_v<c, A>), 2);
}

TEST(TypeSeqTest, Element)
{
    using A = ddc::detail::TypeSeq<a, b, c>;
    EXPECT_TRUE((std::is_same_v<a, ddc::type_seq_element_t<0, A>>));
    EXPECT_TRUE((std::is_same_v<b, ddc::type_seq_element_t<1, A>>));
    EXPECT_TRUE((std::is_same_v<c, ddc::type_seq_element_t<2, A>>));
}

TEST(TypeSeqTest, SameTags)
{
    using A = ddc::detail::TypeSeq<a, b, c>;
    using B = ddc::detail::TypeSeq<z, c, y>;
    using C = ddc::detail::TypeSeq<c, b, a>;
    EXPECT_FALSE((ddc::type_seq_same_v<A, B>));
    EXPECT_FALSE((ddc::type_seq_same_v<B, A>));
    EXPECT_TRUE((ddc::type_seq_same_v<A, C>));
    EXPECT_TRUE((ddc::type_seq_same_v<C, A>));
    EXPECT_FALSE((ddc::type_seq_same_v<B, C>));
    EXPECT_FALSE((ddc::type_seq_same_v<C, B>));
}

TEST(TypeSeqTest, Remove)
{
    using A = ddc::detail::TypeSeq<a, b, c>;
    using B = ddc::detail::TypeSeq<z, c, y>;
    using R = ddc::type_seq_remove_t<A, B>;
    using ExpectedR = ddc::detail::TypeSeq<a, b>;
    EXPECT_TRUE((ddc::type_seq_same_v<R, ExpectedR>));
}

TEST(TypeSeqTest, Merge)
{
    using A = ddc::detail::TypeSeq<a, b, c>;
    using B = ddc::detail::TypeSeq<z, c, y>;
    using R = ddc::type_seq_merge_t<A, B>;
    using ExpectedR = ddc::detail::TypeSeq<a, b, c, z, y>;
    EXPECT_TRUE((ddc::type_seq_same_v<R, ExpectedR>));
}

TEST(TypeSeqTest, Cat)
{
    using A = ddc::detail::TypeSeq<a, b, c>;
    using B = ddc::detail::TypeSeq<z, c, y>;
    using R = ddc::type_seq_cat_t<A, B>;
    using ExpectedR = ddc::detail::TypeSeq<a, b, c, z, c, y>;
    EXPECT_TRUE((ddc::type_seq_same_v<R, ExpectedR>));
}

TEST(TypeSeqTest, Replace)
{
    using A = ddc::detail::TypeSeq<a, b, c, d, e>;
    using B = ddc::detail::TypeSeq<b, d>;
    using C = ddc::detail::TypeSeq<y, z>;
    using R = ddc::type_seq_replace_t<A, B, C>;
    using ExpectedR = ddc::detail::TypeSeq<a, y, c, z, e>;
    EXPECT_TRUE((ddc::type_seq_same_v<R, ExpectedR>));
}

TEST(TypeSeqTest, IsUnique)
{
    using A = ddc::detail::TypeSeq<a, b, c, d, e>;
    using B = ddc::detail::TypeSeq<>;
    using C = ddc::detail::TypeSeq<y>;
    using D = ddc::detail::TypeSeq<y, y>;
    using E = ddc::detail::TypeSeq<a, b, c, d, b, e>;
    using F = ddc::detail::TypeSeq<a, b, a, c, b, a, d, b, e, a>;
    EXPECT_TRUE((ddc::type_seq_is_unique_v<A>));
    EXPECT_TRUE((ddc::type_seq_is_unique_v<B>));
    EXPECT_TRUE((ddc::type_seq_is_unique_v<C>));
    EXPECT_FALSE((ddc::type_seq_is_unique_v<D>));
    EXPECT_FALSE((ddc::type_seq_is_unique_v<E>));
    EXPECT_FALSE((ddc::type_seq_is_unique_v<F>));
}
