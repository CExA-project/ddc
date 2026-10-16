// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstddef>
#include <span>
#include <string>
#include <utility>

#include <Kokkos_Core.hpp>

#include "discrete_vector.hpp"

namespace ddc {

/** The type of the handle given to the functor of `ddc::parallel_for_each_team`.
 *
 * It is the `member_type` of the `Kokkos::TeamPolicy` used for the given execution space, and it
 * can be used with all the hierarchical parallelism facilities of Kokkos (`Kokkos::TeamThreadRange`,
 * `Kokkos::ThreadVectorRange`, `Kokkos::single`, `team.team_barrier()`, scratch memory, ...).
 *
 * @tparam ExecSpace a Kokkos execution space
 */
template <class ExecSpace>
using TeamHandle = Kokkos::TeamPolicy<ExecSpace>::member_type;

namespace detail {

int team_league_size(std::span<DiscreteVectorElement const> extents);

/// Converts the rank of a team into the index of the element of the support it is assigned to.
/// The last dimension is the fastest varying one, which is consistent with the iteration pattern
/// used by `ddc::parallel_for_each`.
template <class Support, std::size_t... Idx>
KOKKOS_FUNCTION Support::discrete_vector_type delinearize_team_rank(
        Support const& support,
        DiscreteVectorElement league_rank,
        std::index_sequence<Idx...> /*unused*/) noexcept
{
    std::array<DiscreteVectorElement, sizeof...(Idx)> ids {};
    if constexpr (sizeof...(Idx) > 0) {
        std::array const extents = array(support.extents());
        for (std::size_t i = sizeof...(Idx); i > 0; --i) {
            ids[i - 1] = league_rank % extents[i - 1];
            league_rank /= extents[i - 1];
        }
    }
    return typename Support::discrete_vector_type(ids[Idx]...);
}

template <class ExecSpace, class F, class Support>
class ForEachTeamKokkosLambdaAdapter
{
    F m_f;

    Support m_support;

public:
    using team_handle_type = TeamHandle<ExecSpace>;

    explicit ForEachTeamKokkosLambdaAdapter(F const& f, Support const& support)
        : m_f(f)
        , m_support(support)
    {
    }

    KOKKOS_FUNCTION void operator()(team_handle_type const& team) const
    {
        m_f(team,
            m_support(delinearize_team_rank(
                    m_support,
                    static_cast<DiscreteVectorElement>(team.league_rank()),
                    std::make_index_sequence<Support::rank()>())));
    }
};

template <class ExecSpace, class Support, class Functor, class... TeamSizes>
void for_each_team_kokkos(
        std::string const& label,
        ExecSpace const& execution_space,
        Support const& domain,
        Functor const& f,
        TeamSizes... team_sizes) noexcept
{
    Kokkos::TeamPolicy<ExecSpace> const
    policy(execution_space, team_league_size(array(domain.extents())), team_sizes...);
    Kokkos::parallel_for(
            label,
            policy,
            ForEachTeamKokkosLambdaAdapter<ExecSpace, Functor, Support>(f, domain));
}

} // namespace detail

/** iterates over a nD domain using a given `Kokkos` execution space and a `Kokkos` team policy.
 *
 * A team is created for each element of the domain. The functor is called by every thread of the
 * team (as with a `Kokkos::TeamPolicy`) with the handle of the team and the element of the domain
 * assigned to this team.
 *
 * The team size is chosen by Kokkos (`Kokkos::AUTO`).
 *
 * @param[in] label  name for easy identification of the parallel_for_each_team algorithm
 * @param[in] execution_space a Kokkos execution space where the loop will be executed on
 * @param[in] domain the domain over which to iterate
 * @param[in] f      a functor taking as parameters a `ddc::TeamHandle<ExecSpace>` and an index
 */
template <class ExecSpace, class Support, class Functor>
void parallel_for_each_team(
        std::string const& label,
        ExecSpace const& execution_space,
        Support const& domain,
        Functor&& f) noexcept
{
    detail::for_each_team_kokkos(
            label,
            execution_space,
            domain,
            std::forward<Functor>(f),
            Kokkos::AUTO);
}

/** iterates over a nD domain using a given `Kokkos` execution space and a `Kokkos` team policy.
 *
 * A team is created for each element of the domain. The functor is called by every thread of the
 * team (as with a `Kokkos::TeamPolicy`) with the handle of the team and the element of the domain
 * assigned to this team.
 *
 * @param[in] label  name for easy identification of the parallel_for_each_team algorithm
 * @param[in] execution_space a Kokkos execution space where the loop will be executed on
 * @param[in] domain the domain over which to iterate
 * @param[in] team_size the number of threads per team
 * @param[in] vector_length the vector length of each thread of the team
 * @param[in] f      a functor taking as parameters a `ddc::TeamHandle<ExecSpace>` and an index
 */
template <class ExecSpace, class Support, class Functor>
void parallel_for_each_team(
        std::string const& label,
        ExecSpace const& execution_space,
        Support const& domain,
        int team_size,
        int vector_length,
        Functor&& f) noexcept
{
    detail::for_each_team_kokkos(
            label,
            execution_space,
            domain,
            std::forward<Functor>(f),
            team_size,
            vector_length);
}

/** iterates over a nD domain using a given `Kokkos` execution space and a `Kokkos` team policy.
 *
 * A team is created for each element of the domain. The functor is called by every thread of the
 * team (as with a `Kokkos::TeamPolicy`) with the handle of the team and the element of the domain
 * assigned to this team.
 *
 * The team size is chosen by Kokkos (`Kokkos::AUTO`).
 *
 * @param[in] execution_space a Kokkos execution space where the loop will be executed on
 * @param[in] domain the domain over which to iterate
 * @param[in] f      a functor taking as parameters a `ddc::TeamHandle<ExecSpace>` and an index
 */
template <class ExecSpace, class Support, class Functor>
void parallel_for_each_team(
        ExecSpace const& execution_space,
        Support const& domain,
        Functor&& f) noexcept
    requires(Kokkos::is_execution_space_v<ExecSpace>)
{
    detail::for_each_team_kokkos(
            "ddc_for_each_team_default",
            execution_space,
            domain,
            std::forward<Functor>(f),
            Kokkos::AUTO);
}

/** iterates over a nD domain using the `Kokkos` default execution space and a `Kokkos` team policy.
 *
 * A team is created for each element of the domain. The functor is called by every thread of the
 * team (as with a `Kokkos::TeamPolicy`) with the handle of the team and the element of the domain
 * assigned to this team.
 *
 * The team size is chosen by Kokkos (`Kokkos::AUTO`).
 *
 * @param[in] label  name for easy identification of the parallel_for_each_team algorithm
 * @param[in] domain the domain over which to iterate
 * @param[in] f      a functor taking as parameters a `ddc::TeamHandle<Kokkos::DefaultExecutionSpace>`
 *                   and an index
 */
template <class Support, class Functor>
void parallel_for_each_team(std::string const& label, Support const& domain, Functor&& f) noexcept
{
    parallel_for_each_team(
            label,
            Kokkos::DefaultExecutionSpace(),
            domain,
            std::forward<Functor>(f));
}

/** iterates over a nD domain using the `Kokkos` default execution space and a `Kokkos` team policy.
 *
 * A team is created for each element of the domain. The functor is called by every thread of the
 * team (as with a `Kokkos::TeamPolicy`) with the handle of the team and the element of the domain
 * assigned to this team.
 *
 * The team size is chosen by Kokkos (`Kokkos::AUTO`).
 *
 * @param[in] domain the domain over which to iterate
 * @param[in] f      a functor taking as parameters a `ddc::TeamHandle<Kokkos::DefaultExecutionSpace>`
 *                   and an index
 */
template <class Support, class Functor>
void parallel_for_each_team(Support const& domain, Functor&& f) noexcept
{
    parallel_for_each_team(
            "ddc_for_each_team_default",
            Kokkos::DefaultExecutionSpace(),
            domain,
            std::forward<Functor>(f));
}

} // namespace ddc
