#!/usr/bin/python3
from cgt_bandits.nodes import ChanceNode, TerminalNode, PersonalNode
from cgt_bandits import import_efg, export_dot

import contextlib
import pygambit
import sys

import gurobipy as gp
from gurobipy import GRB

import copy


class Infoset:
    def __init__(self):
        self.p0_isets = {}
        self.p1_isets = {}

    def update_infoset(self, pl_num, iset_num, chance):
        if pl_num == 0:
            if iset_num not in self.p0_isets:
                self.p0_isets[iset_num] = chance
            else:
                self.p0_isets[iset_num] += chance
        elif pl_num == 1:
            if iset_num not in self.p1_isets:
                self.p1_isets[iset_num] = chance
            else:
                self.p1_isets[iset_num] += chance
        assert pl_num in [0, 1], "Wrong player number: {}".format(pl_num)

    def get_isets(self, pl_num):
        if pl_num == 0:
            return self.p0_isets
        elif pl_num == 1:
            return self.p1_isets

    def get_iset(self, pl_num, idx):
        if pl_num == 0:
            return self.p0_isets[idx]
        elif pl_num == 1:
            return self.p1_isets[idx]


class Actions:
    def __init__(self):
        self.actions = {}

    def add_actions(self, iset_num, act_arr):
        self.actions[iset_num] = [act + "_" + str(iset_num) for act in act_arr]

    def get_actions(self, iset_num):
        return self.actions[iset_num]

    def get_action(self, iset_num, idx):
        return self.actions[iset_num][idx]


class Paths:
    def __init__(self):
        self.p0_path = ""
        self.p1_path = ""

    def update_path(self, pl_num, path_ext):
        if pl_num == 0:
            self.p0_path += path_ext
        elif pl_num == 1:
            self.p1_path += path_ext
        assert pl_num in [0, 1], "Wrong player number: {}".format(pl_num)

    def get_path(self, pl_num):
        if pl_num == 0:
            return self.p0_path
        elif pl_num == 1:
            return self.p1_path
        assert pl_num in [0, 1], "Wrong player number: {}".format(pl_num)


class Seq:
    def __init__(self):
        self.p0_seq = {}
        self.p1_seq = {}

    def add_seq(self, pl_num, iset_num, paths):
        path = paths.get_path(pl_num)
        if pl_num == 0:
            self.p0_seq[iset_num] = path
        elif pl_num == 1:
            self.p1_seq[iset_num] = path
        assert pl_num in [0, 1], "Wrong player number: {}".format(pl_num)

    def get_seq(self, pl_num, iset_num):
        if pl_num == 0:
            return self.p0_seq[iset_num]
        elif pl_num == 1:
            return self.p1_seq[iset_num]
        assert pl_num in [0, 1], "Wrong player number: {}".format(pl_num)


class ExtUtil:
    def __init__(self):
        self.utils = {}

    def add_util(self, path1, path2, payoff):
        if path1 not in self.utils:
            self.utils[path1] = {path2: payoff}
        elif path2 not in self.utils[path1]:
            self.utils[path1][path2] = payoff
        else:
            self.utils[path1][path2] += payoff

    def get_util(self, path1, path2):
        utility = 0
        if path1 in self.utils:
            if path2 in self.utils[path1]:
                utility = self.utils[path1][path2]
        return utility


def sqf(I, Sigma, A, seq, g):
    """The Sequence form linear program defined in terms of the parameters
        I: {p_0 : [], p_1: []}
    Sigma: [seq1, seq2, ...]
        A: {(p_0, infoset_0) : [action1, action2, ...], ... }
      seq: {(p_0, infoset_0) : seq_0, ... }:
        g: {(seq_0, seq_1) : utility, ... }
    """
    # Implement this first. The function should be completely general. Do not
    # traverse the graph at this point! Simply use the gurobi modeling interface
    # and formulate the SQF. If you do this correctly, you should be able to
    # compute the payoff of the miners by switching the parameters.

    m = gp.Model("SQF")

    pl_0_isets = I.get_isets(0)  # values for the infosets of first player pl_0
    pl_1_isets = I.get_isets(1)  # values for the infosets of second player pl_1

    values = {}
    for iset1 in pl_1_isets:
        v_name = "v_" + str(iset1)
        v = m.addVar(name=v_name, lb=-GRB.INFINITY, ub=GRB.INFINITY)
        values[v_name] = v

    realisations = {}  # realisations for the actions of pl_0
    for s0 in Sigma:  # sequences of the first player pl_0
        r_s0_name = "r_" + str(s0)
        r_s0 = m.addVar(name=r_s0_name, lb=0, ub=1)
        realisations[r_s0_name] = r_s0

    # create the expression to be maximised
    v_root = gp.LinExpr()
    for s0 in Sigma:
        r_s0 = realisations["r_" + str(s0)]
        v_root += r_s0 * g.get_util(s0, "")
    for iset1 in pl_1_isets:
        v_s1 = values["v_" + str(iset1)]
        seq1 = seq.get_seq(1, iset1)
        if seq1 == "":
            v_root += v_s1

    m.setObjective(v_root, GRB.MAXIMIZE)

    # create constraints for node values
    for iset1 in pl_1_isets:
        v_s1 = values["v_" + str(iset1)]
        seq1 = seq.get_seq(1, iset1)
        for a in A.get_actions(iset1):
            seq1_a = seq1 + a  # string concatenation

            var = gp.LinExpr()
            for s0 in Sigma:
                r_s0 = realisations["r_" + str(s0)]
                var += r_s0 * g.get_util(s0, seq1_a)
            for iset11 in pl_1_isets:
                if iset11 == iset1:
                    continue
                v_s11 = values["v_" + str(iset11)]
                iset11_seq = seq.get_seq(1, iset11)
                if iset11_seq == seq1_a:
                    var += v_s11

            m.addConstr(v_s1 <= var, name="c_v_" + str(iset1))

    # create constrains for realisations
    for iset0 in pl_0_isets:
        var = gp.LinExpr()
        seq0 = seq.get_seq(0, iset0)
        for a in A.get_actions(iset0):
            seq0_a = seq0 + a
            var += realisations["r_" + str(seq0_a)]

        r_seq0 = realisations["r_" + str(seq0)]
        m.addConstr(r_seq0 == var, name="r_" + str(iset0))
    m.addConstr(realisations["r_"] == 1, name="r_")

    m.optimize()

    return m.ObjVal


def extract_node_params(I, S, A, seq, g, cur_node, path, chance):

    if isinstance(cur_node, TerminalNode):
        pl0_path = path.get_path(0)
        pl1_path = path.get_path(1)
        pl0_payoff = chance * cur_node.payoffs[0]
        g.add_util(pl0_path, pl1_path, pl0_payoff)
        if pl0_path not in S:
            S.append(pl0_path)

    elif isinstance(cur_node, PersonalNode):
        pl_num = cur_node.player
        iset_num = cur_node.infoset
        cur_actions = cur_node.action_names

        I.update_infoset(pl_num, iset_num, chance)
        A.add_actions(iset_num, cur_actions)
        seq.add_seq(pl_num, iset_num, path)

        pl0_path = path.get_path(0)
        if pl0_path not in S:
            S.append(pl0_path)

        for i, next_node in enumerate(cur_node.children):
            act = A.get_action(iset_num, i)  # act = cur_node.action_names[i]
            new_path = copy.deepcopy(path)
            new_path.update_path(pl_num, act)
            I, S, A, seq, g = extract_node_params(I, S, A, seq, g, next_node, new_path, chance)

    elif isinstance(cur_node, ChanceNode):
        for i, next_node in enumerate(cur_node.children):
            new_chance = chance * float(cur_node.action_probs[i])
            I, S, A, seq, g = extract_node_params(I, S, A, seq, g, next_node, path, new_chance)
    else:
        assert False

    return I, S, A, seq, g


def extract_parameters(root_node):
    """Converts an extensive form game into the SQF parameters:
    I, Sigma, A, seq, g."""

    # Implement this second. It does not matter how you implement the
    # parameters -- functions, classes, or dictionaries, anything will work.

    I = Infoset()
    S = [""]  # player 0 sequences
    A = Actions()
    seq = Seq()
    g = ExtUtil()
    path = Paths()
    chance = 1.0
    I, S, A, seq, g = extract_node_params(I, S, A, seq, g, root_node, path, chance)
    return I, S, A, seq, g


def payoff(root_node):
    """Computes the value of the extensive form game"""

    I, S, A, seq, g = extract_parameters(root_node)
    with contextlib.redirect_stdout(sys.stderr):
        p = sqf(I, S, A, seq, g)

    return p


def draw_game(root, filename="out.pdf"):
    "You can use this to visualize your games."
    # File IO not allowed on BRUTE! GraphViz is required to render graphs.

    dot = export_dot.nodes_to_dot(root)
    dot.write_pdf(filename)

def game_value(efg):
    "You can use this to check the solution for small games."
    # Standard output on BRUTE must not contain debugging messages!

    equilibrium = pygambit.nash.enummixed_solve(efg, rational=False)
    print(equilibrium[0].payoff(efg.players[0]), file=sys.stderr)

if __name__ == "__main__":
    # with open('examples/example_from_slides.efg', 'r') as file:
    #     efg = file.read()

    efg = sys.stdin.read()
    game = pygambit.Game.parse_game(efg)
    # game_value(game)
    root = import_efg.efg_to_nodes(game)
    # draw_game(root)

    print(payoff(root))
