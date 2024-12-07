#!/usr/bin/env python3

# From here onwards, a modification of a homework assignment about implementing CFR.

import sys
import os
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from tqdm import tqdm


###############################################################################
# The next functions are already implemented for your convenience
#
# In all the functions in this stub file, `game` is the parsed input game json
# file, whereas `tfsdp` is either `game["decision_problem_pl1"]` or
# `game["decision_problem_pl2"]`.
#
# See the homework handout for a description of each field.


def get_sequence_set(tfsdp):
    """Returns a set of all sequences in the given tree-form sequential decision
    process (TFSDP)"""

    sequences = set()
    for node in tfsdp:
        if node["type"] == "decision":
            for action in node["actions"]:
                sequences.add((node["id"], action))
    return sequences


def is_valid_RSigma_vector(tfsdp, obj):
    """Checks that the given object is a dictionary keyed on the set of sequences
    of the given tree-form sequential decision process (TFSDP)"""

    sequence_set = get_sequence_set(tfsdp)
    return isinstance(obj, dict) and obj.keys() == sequence_set


def assert_is_valid_sf_strategy(tfsdp, obj):
    """Checks whether the given object `obj` represents a valid sequence-form
    strategy vector for the given tree-form sequential decision process
    (TFSDP)"""

    if not is_valid_RSigma_vector(tfsdp, obj):
        print(
            "The sequence-form strategy should be a dictionary with key set equal to the set of sequences in the game")
        sys.exit(1)
    for node in tfsdp:
        if node["type"] == "decision":
            parent_reach = 1.0
            if node["parent_sequence"] is not None:
                parent_reach = obj[node["parent_sequence"]]
            if abs(sum([obj[(node["id"], action)] for action in node["actions"]]) - parent_reach) > 1e-3:
                print("At node ID %s the sum of the child sequences is not equal to the parent sequence", node["id"])
                sys.exit(1)


def best_response_value(tfsdp, utility):
    """Computes the value of max_{x in Q} x^T utility, where Q is the
    sequence-form polytope for the given tree-form sequential decision
    process (TFSDP)"""

    assert is_valid_RSigma_vector(tfsdp, utility)

    utility_ = utility.copy()
    utility_[None] = 0.0
    for node in tfsdp[::-1]:
        if node["type"] == "decision":
            max_ev = max([utility_[(node["id"], action)]
                          for action in node["actions"]])
            utility_[node["parent_sequence"]] += max_ev
    return utility_[None]


def compute_utility_vector_pl1(game, sf_strategy_pl2):
    """Returns A * y, where A is the payoff matrix of the game and y is
    the given strategy for Player 2"""

    assert_is_valid_sf_strategy(game["decision_problem_pl2"], sf_strategy_pl2)

    sequence_set = get_sequence_set(game["decision_problem_pl1"])
    utility = {sequence: 0.0 for sequence in sequence_set}
    for entry in game["utility_pl1"]:
        utility[entry["sequence_pl1"]] += entry["value"] * sf_strategy_pl2[entry["sequence_pl2"]]

    assert is_valid_RSigma_vector(game["decision_problem_pl1"], utility)
    return utility


def compute_utility_vector_pl2(game, sf_strategy_pl1):
    """Returns A^transpose * x, where A is the payoff matrix of the
    game and x is the given strategy for Player 1. *Note that this is different
    from the original version of this file, which uses -A^transpose * x.*"""

    assert_is_valid_sf_strategy(game["decision_problem_pl1"], sf_strategy_pl1)

    sequence_set = get_sequence_set(game["decision_problem_pl2"])
    utility = {sequence: 0.0 for sequence in sequence_set}
    for entry in game["utility_pl1"]:
        utility[entry["sequence_pl2"]] += entry["value"] * sf_strategy_pl1[entry["sequence_pl1"]]

    assert is_valid_RSigma_vector(game["decision_problem_pl2"], utility)
    return utility


def gap(game, sf_strategy_pl1, sf_strategy_pl2):
    """Computes the saddle point gap of the given sequence-form strategies
    for the players. Perhaps this isn't accurate since it's a symmetric game, though."""

    assert_is_valid_sf_strategy(game["decision_problem_pl1"], sf_strategy_pl1)
    assert_is_valid_sf_strategy(game["decision_problem_pl2"], sf_strategy_pl2)

    utility_pl1 = compute_utility_vector_pl1(game, sf_strategy_pl2)
    utility_pl2 = compute_utility_vector_pl2(game, sf_strategy_pl1)

    # In the original, this was the sum, since they have opposite utilities.
    # In this version with equal utilities, take the difference.
    return (best_response_value(game["decision_problem_pl1"], utility_pl1)
            - best_response_value(game["decision_problem_pl2"], utility_pl2))


###########################################################################
# Starting from here, you should fill in the implementation of the
# different functions


def expected_utility_pl1(game, sf_strategy_pl1, sf_strategy_pl2):
    """Returns the expected utility for Player 1 in the game, when the two
    players play according to the given strategies"""

    assert_is_valid_sf_strategy(
        game["decision_problem_pl1"], sf_strategy_pl1)
    assert_is_valid_sf_strategy(
        game["decision_problem_pl2"], sf_strategy_pl2)

    return sum(entry["value"] * sf_strategy_pl1[entry["sequence_pl1"]] * sf_strategy_pl2[entry["sequence_pl2"]]
               for entry in game["utility_pl1"])


def uniform_sf_strategy(tfsdp):
    """Returns the uniform sequence-form strategy for the given tree-form
    sequential decision process"""

    # Start with the empty sequence. Really it should be returned with the empty sequence too,
    # but we'll remove it at the end to make it cooperate with the provided stub code.
    sf_strategy = {None: 1.0}

    # Top-down order
    for node in tfsdp:
        if node["type"] == "decision":
            for action in node["actions"]:
                # divide by len(node["actions"]) to make it uniform
                sf_strategy[(node["id"], action)] = sf_strategy[node["parent_sequence"]] / len(node["actions"])

    del sf_strategy[None]

    assert_is_valid_sf_strategy(tfsdp, sf_strategy)
    return sf_strategy


class RegretMatching(object):
    def __init__(self, action_set):
        # Something's probably wrong if we have an action set with no actions
        assert len(action_set) > 0
        self.action_set = set(action_set)
        self.r = {action: 0.0 for action in self.action_set}
        # starting with a uniform distribution is fine
        self.x = {action: 1.0 / len(self.action_set) for action in self.action_set}

    def next_strategy(self):
        # I think I'm supposed to copy here. Curse you, pass by reference!
        return self.x.copy()

    def observe_utility(self, utility):
        assert isinstance(utility, dict) and utility.keys() == self.action_set
        dot = sum(utility[action] * self.x[action] for action in self.action_set)
        for action in self.action_set:
            self.r[action] += utility[action] - dot
        r = self.r.copy()
        for action in self.action_set:
            r[action] = max(0.0, r[action])
        s = sum(r.values())
        # Make sure the sum is positive, otherwise fall back to not changing x.
        # I believe the equality check is appropriate here.
        if s != 0.0:
            self.x = {action: r[action] / s for action in self.action_set}


class RegretMatchingPlus(object):
    def __init__(self, action_set):
        # Something's probably wrong if we have an action set with no actions
        assert len(action_set) > 0
        self.action_set = set(action_set)
        self.r = {action: 0.0 for action in self.action_set}
        # starting with a uniform distribution is fine
        self.x = {action: 1.0 / len(self.action_set) for action in self.action_set}

    def next_strategy(self):
        # I think I'm supposed to copy here. Curse you, pass by reference!
        return self.x.copy()

    def observe_utility(self, utility):
        assert isinstance(utility, dict) and utility.keys() == self.action_set
        dot = sum(utility[action] * self.x[action] for action in self.action_set)
        for action in self.action_set:
            self.r[action] += utility[action] - dot
        # same as normal regret matching, but actually modify the stored regret
        for action in self.action_set:
            self.r[action] = max(0.0, self.r[action])
        s = sum(self.r.values())
        # Make sure the sum is positive, otherwise fall back to not changing x.
        # I believe the equality check is appropriate here.
        if s != 0.0:
            self.x = {action: self.r[action] / s for action in self.action_set}


class Cfr(object):
    def __init__(self, tfsdp, rm_class=RegretMatching):
        self.tfsdp = tfsdp
        self.local_regret_minimizers = {}

        # For each decision point, we instantiate a local regret minimizer
        for node in tfsdp:
            if node["type"] == "decision":
                self.local_regret_minimizers[node["id"]] = rm_class(node["actions"])

    def next_strategy(self):
        # Similar to `uniform_sf_strategy` above, but using the regret minimizer behavioral strategies rather than
        # uniform behavioral strategies.

        # This takes non-negligible time to calculate, but it's probably not called often enough to require caching.

        # Start with the empty sequence. Really it should be returned with the empty sequence too,
        # but we'll remove it at the end to make it cooperate with the provided stub code.
        sf_strategy = {None: 1.0}

        # Top-down order
        for node in self.tfsdp:
            if node["type"] == "decision":
                for action in node["actions"]:
                    sf_strategy[(node["id"], action)] = sf_strategy[node["parent_sequence"]] * \
                                                        self.local_regret_minimizers[node["id"]].next_strategy()[action]

        del sf_strategy[None]

        assert_is_valid_sf_strategy(self.tfsdp, sf_strategy)
        return sf_strategy

    def observe_utility(self, utility):
        assert is_valid_RSigma_vector(self.tfsdp, utility)

        # Calculate the counterfactual utilities...
        cfu = {sequence: 0.0 for sequence in get_sequence_set(self.tfsdp)}
        for node in self.tfsdp[::-1]:
            if node["type"] == "decision":
                for action in node["actions"]:
                    # If it is a terminal node, this will be nonzero
                    cfu[(node["id"], action)] += utility[(node["id"], action)]
                    if node["parent_sequence"] is not None:
                        # Passing the value to the parent sequence like this exactly follows the formula.
                        cfu[node["parent_sequence"]] += cfu[(node["id"], action)] * \
                                                        self.local_regret_minimizers[node["id"]].next_strategy()[action]

        # ...and pass them to the regret minimizers.
        for node in self.tfsdp:
            if node["type"] == "decision":
                u = {action: cfu[(node["id"], action)] for action in node["actions"]}
                self.local_regret_minimizers[node["id"]].observe_utility(u)


# Problem 3.3 was the best-performing, so it is the only one I will include.
#
# def solve_problem_3_1(game, plot_dir):
#     cfr = Cfr(game["decision_problem_pl1"])
#     uniform_sf = uniform_sf_strategy(game["decision_problem_pl2"])
#     utility_vector_pl1 = compute_utility_vector_pl1(game, uniform_sf)
#
#     strategy_sum = {sequence: 0.0 for sequence in get_sequence_set(game["decision_problem_pl1"])}
#     vT = [0.0 for _ in range(1000)]
#     # for T in range(1, 1001):
#     for T in range(1, 1001):
#         cfr.observe_utility(utility_vector_pl1)
#         strategy = cfr.next_strategy()
#         for sequence in get_sequence_set(game["decision_problem_pl1"]):
#             strategy_sum[sequence] += strategy[sequence]
#         average_strategy = {sequence: strategy_sum[sequence] / T
#                             for sequence in get_sequence_set(game["decision_problem_pl1"])}
#         vT[T - 1] = sum(utility_vector_pl1[sequence] * average_strategy[sequence]
#                         for sequence in get_sequence_set(game["decision_problem_pl1"]))
#
#     try:
#         os.mkdir(plot_dir)
#     except FileExistsError:
#         pass
#
#     plt.plot(vT)
#     plt.savefig(plot_dir + '/plot.png')
#
#     print(f"Last iterate of vT: {vT[-1]}")
#
#
# def solve_problem_3_2(game, plot_dir):
#     cfr1, cfr2 = Cfr(game["decision_problem_pl1"]), Cfr(game["decision_problem_pl2"])
#
#     strategy_sum1 = {sequence: 0.0 for sequence in get_sequence_set(game["decision_problem_pl1"])}
#     strategy_sum2 = {sequence: 0.0 for sequence in get_sequence_set(game["decision_problem_pl2"])}
#     gapT = [0.0 for _ in range(1000)]
#     uT = [0.0 for _ in range(1000)]
#     for T in range(1, 1001):
#         x = cfr1.next_strategy()
#         y = cfr2.next_strategy()
#         lx = compute_utility_vector_pl1(game, y)
#         ly = compute_utility_vector_pl2(game, x)
#         cfr1.observe_utility(lx)
#         cfr2.observe_utility(ly)
#
#         for sequence in get_sequence_set(game["decision_problem_pl1"]):
#             strategy_sum1[sequence] += x[sequence]
#         for sequence in get_sequence_set(game["decision_problem_pl2"]):
#             strategy_sum2[sequence] += y[sequence]
#         average_strategy1 = {sequence: strategy_sum1[sequence] / T
#                              for sequence in get_sequence_set(game["decision_problem_pl1"])}
#         average_strategy2 = {sequence: strategy_sum2[sequence] / T
#                              for sequence in get_sequence_set(game["decision_problem_pl2"])}
#         gapT[T - 1] = gap(game, average_strategy1, average_strategy2)
#         uT[T - 1] = expected_utility_pl1(game, average_strategy1, average_strategy2)
#
#     try:
#         os.mkdir(plot_dir)
#     except FileExistsError:
#         pass
#
#     plt.yscale('log')
#     plt.plot(gapT)
#     plt.savefig(plot_dir + '/gap.png')
#
#     plt.clf()
#     plt.plot(uT)
#     plt.savefig(plot_dir + '/utility.png')


def solve_problem_3_3(game, plot_dir):
    N = 100
    cfr1 = Cfr(game["decision_problem_pl1"], rm_class=RegretMatchingPlus)
    cfr2 = Cfr(game["decision_problem_pl2"], rm_class=RegretMatchingPlus)

    strategy_sum1 = {sequence: 0.0 for sequence in get_sequence_set(game["decision_problem_pl1"])}
    strategy_sum2 = {sequence: 0.0 for sequence in get_sequence_set(game["decision_problem_pl2"])}
    gapT = [0.0 for _ in range(N)]
    uT = [0.0 for _ in range(N)]
    for T in tqdm(range(1, N + 1)):
        y = cfr2.next_strategy()
        lx = compute_utility_vector_pl1(game, y)
        cfr1.observe_utility(lx)
        x = cfr1.next_strategy()
        ly = compute_utility_vector_pl2(game, x)
        cfr2.observe_utility(ly)
        # needs to be updated again
        y = cfr2.next_strategy()

        for sequence in get_sequence_set(game["decision_problem_pl1"]):
            strategy_sum1[sequence] += x[sequence]
        for sequence in get_sequence_set(game["decision_problem_pl2"]):
            strategy_sum2[sequence] += y[sequence]
        average_strategy1 = {sequence: strategy_sum1[sequence] / T
                             for sequence in get_sequence_set(game["decision_problem_pl1"])}
        average_strategy2 = {sequence: strategy_sum2[sequence] / T
                             for sequence in get_sequence_set(game["decision_problem_pl2"])}
        gapT[T - 1] = gap(game, average_strategy1, average_strategy2)
        uT[T - 1] = expected_utility_pl1(game, average_strategy1, average_strategy2)

    try:
        os.mkdir(plot_dir)
    except FileExistsError:
        pass

    plt.yscale('log')
    plt.plot(gapT)
    plt.savefig(plot_dir + '/gap.png')

    plt.clf()
    plt.plot(uT)
    plt.savefig(plot_dir + '/utility.png')

    average_strategy1 = {str(sequence): strategy_sum1[sequence] / N
                         for sequence in get_sequence_set(game["decision_problem_pl1"])}
    average_strategy2 = {str(sequence): strategy_sum2[sequence] / N
                         for sequence in get_sequence_set(game["decision_problem_pl2"])}

    with open(plot_dir + '/average_strategy.json', 'w') as f:
        json.dump({
            "average_strategy1": average_strategy1,
            "average_strategy2": average_strategy2,
        }, f)


def solve(game_file: Path, plot_dir: Path):
    game = json.load(open(game_file))

    # Convert all sequences from lists to tuples
    for tfsdp in [game["decision_problem_pl1"], game["decision_problem_pl2"]]:
        for node in tfsdp:
            if isinstance(node["parent_edge"], list):
                node["parent_edge"] = tuple(node["parent_edge"])
            if "parent_sequence" in node and isinstance(node["parent_sequence"], list):
                node["parent_sequence"] = tuple(node["parent_sequence"])
    for entry in game["utility_pl1"]:
        assert isinstance(entry["sequence_pl1"], list)
        assert isinstance(entry["sequence_pl2"], list)
        entry["sequence_pl1"] = tuple(entry["sequence_pl1"])
        entry["sequence_pl2"] = tuple(entry["sequence_pl2"])

    solve_problem_3_3(game, str(plot_dir))
