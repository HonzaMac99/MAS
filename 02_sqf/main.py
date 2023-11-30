import contextlib
import pygambit
import sys

import gurobipy as gp
from gurobipy import GRB


def sqf(I, Sigma, A, seq, g):
    """The Sequence form linear program defined in terms of the parameters
        I: {p_0 : [], p_1: []}
    Sigma: {p_0 : [seq1, seq2, ...]}
        A: {(p_0, infoset_0) : [action1, action2, ...], ... }
      seq: {(p_0, infoset_0) : seq_0, ... }:
        g: {(seq_0, seq_1) : utility, ... }
    """
    # todo: does seq return strings?

    # Implement this first. The function should be completely general. Do not
    # traverse the graph at this point! Simply use the gurobi modeling interface
    # and formulate the SQF. If you do this correctly, you should be able to
    # compute the payoff of the miners by switching the parameters.

    m = gp.Model("SQF")

    values = {}
    for i in range(len(I[1])):
        iset1 = I[1][i]  # values for the infosets of second player p_1
        v_name = "v_" + str(iset1)
        v = m.addVars(iset1, name=v_name, lb=-GRB.INFINITY, ub=GRB.INFINITY)
        values[v_name] = v

    realisations = {}
    for i in range(len(Sigma[0])):
        s0 = Sigma[0][i]  # realisations for the actions of p_0
        r_s0_name = "r_" + str(s0)
        r_s0 = m.addVars(s0, name=r_s0_name, lb=0, ub=1)
        realisations[r_s0_name] = r_s0

    # create the expression to be maximised
    v_root = gp.LinExpr()
    for i in range(len(Sigma[0])):
        s0 = Sigma[0][i]
        r_s0 = realisations["r_" + str(s0)]
        v_root += r_s0 * g(s0, None)
    for i in range(len(I[1])):
        iset1 = I[1][i]
        v_s1 = values["v_" + str(iset1)]
        if not seq(1, iset1):
            v_root += v_s1

    m.setObjective(v_root, GRB.MAXIMIZE)

    # create constraints for node values
    for i in range(len(I[1])):
        iset1 = I[1][i]
        v_s1 = values["v_" + str(iset1)]
        seq1 = seq(1, iset1)
        for j in range(len(A[1][iset1])):
            a = A[1][iset1][j]
            seq1_a = seq1 + a  # TODO: contcatenating strings?

            var = gp.LinExpr()
            for k in range(len(Sigma[0])):
                s0 = Sigma[0][k]
                r_s0 = realisations["r_" + str(s0)]
                var += r_s0 * g(s0, seq1_a)
            for k in range(len(I[1])):
                iset11 = I[1][k]
                if iset11 == iset1:
                    continue
                v_s11 = values["v_" + str(iset11)]
                if seq(1, iset11) == seq1_a:
                    var += v_s11

            m.addConstr(v_s1 <= var, name="c_v_" + str(iset1))

    # create constrains for realisations
    for i in range(len(I[0])):
        iset0 = I[0][i]
        seq0 = seq(0, iset0)
        var = gp.LinExpr()
        for j in range(len(A[0][iset0])):
            a = A[0][iset0][j]
            seq0_a = seq0 + a
            var += realisations[seq0_a]

        r_seq0 = realisations[seq0]
        m.addConstr(r_seq0 == var, name="r_" + str(iset0))

    m.optimize()

    return m.ObjVal


def extract_parameters(root_node):
    """Converts an extensive form game into the SQF parameters:
    I, Sigma, A, seq, g."""

    # Implement this second. It does not matter how you implement the
    # parameters -- functions, classes, or dictionaries, anything will work.

    print("root_node")
    print(root_node)
    pass


def payoff(root_node):
    """Computes the value of the extensive form game"""

    parameters = extract_parameters(root_node)

    with contextlib.redirect_stdout(sys.stderr):
        # p = sqf(None, None, None, None, None)
        p = None

    return p


if __name__ == "__main__":
    efg = sys.stdin.read()
    game = pygambit.Game.parse_game(efg)
    root = import_efg.efg_to_nodes(game)
    print("hello")

    print(payoff(root))
