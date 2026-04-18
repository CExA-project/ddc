// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <type_traits>

#include <ddc/ddc.hpp>

#include <gtest/gtest.h>

inline namespace anonymous_namespace_workaround_type_seq_cpp {

struct T0
{
};
struct T1
{
};
struct T2
{
};
struct T3
{
};
struct T4
{
};
struct T5
{
};
struct T6
{
};

} // namespace anonymous_namespace_workaround_type_seq_cpp

TEST(TypeSeqTest, Size)
{
    using Empty = ddc::detail::TypeSeq<>;
    using A = ddc::detail::TypeSeq<T0, T1, T2>;
    EXPECT_EQ((ddc::type_seq_size_v<Empty>), 0);
    EXPECT_EQ((ddc::type_seq_size_v<A>), 3);
}

TEST(TypeSeqTest, Rank)
{
    using A = ddc::detail::TypeSeq<T0, T1, T2>;
    EXPECT_EQ((ddc::type_seq_rank_v<T0, A>), 0);
    EXPECT_EQ((ddc::type_seq_rank_v<T1, A>), 1);
    EXPECT_EQ((ddc::type_seq_rank_v<T2, A>), 2);
}

TEST(TypeSeqTest, Element)
{
    using A = ddc::detail::TypeSeq<T0, T1, T2>;
    EXPECT_TRUE((std::is_same_v<T0, ddc::type_seq_element_t<0, A>>));
    EXPECT_TRUE((std::is_same_v<T1, ddc::type_seq_element_t<1, A>>));
    EXPECT_TRUE((std::is_same_v<T2, ddc::type_seq_element_t<2, A>>));
}

TEST(TypeSeqTest, SameTags)
{
    using A = ddc::detail::TypeSeq<T0, T1, T2>;
    using B = ddc::detail::TypeSeq<T6, T2, T5>;
    using C = ddc::detail::TypeSeq<T2, T1, T0>;
    EXPECT_FALSE((ddc::type_seq_same_v<A, B>));
    EXPECT_FALSE((ddc::type_seq_same_v<B, A>));
    EXPECT_TRUE((ddc::type_seq_same_v<A, C>));
    EXPECT_TRUE((ddc::type_seq_same_v<C, A>));
    EXPECT_FALSE((ddc::type_seq_same_v<B, C>));
    EXPECT_FALSE((ddc::type_seq_same_v<C, B>));
}

TEST(TypeSeqTest, Remove)
{
    using A = ddc::detail::TypeSeq<T0, T1, T2>;
    using B = ddc::detail::TypeSeq<T6, T2, T5>;
    using R = ddc::type_seq_remove_t<A, B>;
    using ExpectedR = ddc::detail::TypeSeq<T0, T1>;
    EXPECT_TRUE((ddc::type_seq_same_v<R, ExpectedR>));
}

TEST(TypeSeqTest, Merge)
{
    using A = ddc::detail::TypeSeq<T0, T1, T2>;
    using B = ddc::detail::TypeSeq<T6, T2, T5>;
    using R = ddc::type_seq_merge_t<A, B>;
    using ExpectedR = ddc::detail::TypeSeq<T0, T1, T2, T6, T5>;
    EXPECT_TRUE((ddc::type_seq_same_v<R, ExpectedR>));
}

TEST(TypeSeqTest, Cat)
{
    using A = ddc::detail::TypeSeq<T0, T1, T2>;
    using B = ddc::detail::TypeSeq<T6, T2, T5>;
    using R = ddc::type_seq_cat_t<A, B>;
    using ExpectedR = ddc::detail::TypeSeq<T0, T1, T2, T6, T2, T5>;
    EXPECT_TRUE((ddc::type_seq_same_v<R, ExpectedR>));
}

TEST(TypeSeqTest, Replace)
{
    using A = ddc::detail::TypeSeq<T0, T1, T2, T3, T4>;
    using B = ddc::detail::TypeSeq<T1, T3>;
    using C = ddc::detail::TypeSeq<T5, T6>;
    using R = ddc::type_seq_replace_t<A, B, C>;
    using ExpectedR = ddc::detail::TypeSeq<T0, T5, T2, T6, T4>;
    EXPECT_TRUE((ddc::type_seq_same_v<R, ExpectedR>));
}

TEST(TypeSeqTest, IsUnique)
{
    using A = ddc::detail::TypeSeq<T0, T1, T2, T3, T4>;
    using B = ddc::detail::TypeSeq<>;
    using C = ddc::detail::TypeSeq<T5>;
    using D = ddc::detail::TypeSeq<T5, T5>;
    using E = ddc::detail::TypeSeq<T0, T1, T2, T3, T1, T4>;
    using F = ddc::detail::TypeSeq<T0, T1, T0, T2, T1, T0, T3, T1, T4, T0>;
    EXPECT_TRUE((ddc::type_seq_is_unique_v<A>));
    EXPECT_TRUE((ddc::type_seq_is_unique_v<B>));
    EXPECT_TRUE((ddc::type_seq_is_unique_v<C>));
    EXPECT_FALSE((ddc::type_seq_is_unique_v<D>));
    EXPECT_FALSE((ddc::type_seq_is_unique_v<E>));
    EXPECT_FALSE((ddc::type_seq_is_unique_v<F>));
}
