import contextlib
import pygambit
import sys

import gurobipy as gp
from gurobipy import GRB


def sqf(I, Sigma, A, seq, g):
    """The Sequence form linear program defined in terms of the parameters
    I, Sigma, A, seq, g."""

    # Implement this first. The function should be completely general. Do not
    # traverse the graph at this point! Simply use the gurobi modeling interface
    # and formulate the SQF. If you do this correctly, you should be able to
    # compute the payoff of the miners by switching the parameters.

    m = gp.Model("SQF")

    # v = m.addVars(I[1], name="v", lb=-GRB.INFINITY, ub=GRB.INFINITY)
    # r = m.addVars(...)

    # m.setObjective(..., GRB.MAXIMIZE)

    # m.addConstr(..., name="c1")
    # m.addConstrs(..., name="c2")
    # m.addConstrs(..., name="c3")

    m.optimize()

    return m.ObjVal


def extract_parameters(efg):
    """Converts an extensive form game into the SQF parameters:
    I, Sigma, A, seq, g."""

    # Implement this second. It does not matter how you implement the
    # parameters -- functions, classes, or dictionaries, anything will work.

    pass


def payoff(efg):
    """Computes the value of the extensive form game"""

    parameters = extract_parameters(efg)
    with contextlib.redirect_stdout(sys.stderr):
        p = sqf(None, None, None, None, None)

    return p


if __name__ == "__main__":
    efg = sys.stdin.read()
    game = pygambit.Game.parse_game(efg)
    root = import_efg.efg_to_nodes(game)

    print(payoff(root))
