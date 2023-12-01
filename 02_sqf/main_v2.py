import contextlib
import pygambit
import sys
from cgt_bandits import import_efg
import gurobipy as gp
from gurobipy import GRB
from cgt_bandits import export_dot
import copy


empty_path = '/'

def select_infosets_seq(I, seq, condition_seq, player="miners"):
    infosets = []
    for i in I["miners"]:
        if seq[player][i] == condition_seq:
            infosets.append(i)
    return infosets

def e_list_based_on_g(g,r,condition):
    e1_p1 = []
    for i in g:
        if condition in g[i]:
            e1_p1.append(r[i] * g[i][condition])
    return e1_p1

def get_v(v,I,seq,seq_condtion,player):
    e1_p1 = []
    for idx in v:
        if idx in seq[player] and seq[player][idx] == seq_condtion:
            p = I[player][idx]
            e1_p1.append(v[idx])
    return e1_p1

def sqf(I, A, seq, g,S):
    """The Sequence form linear program defined in terms of the parameters
    I, Sigma, A, seq, g."""

    # Implement this first. The function should be completely general. Do not
    # traverse the graph at this point! Simply use the gurobi modeling interface
    # and formulate the SQF. If you do this correctly, you should be able to
    # compute the payoff of the miners by switching the parameters.

    m = gp.Model("SQF")

    I_tmp = [idx for idx in I[1]]
    v = m.addVars(I_tmp, name='v', lb=-GRB.INFINITY, ub=GRB.INFINITY)

    #create r
    R = [empty_path]
    for i in seq[0]:
        sigma = seq[0][i]
        for action in A[i]:
            sigma_a = sigma + action
            if sigma_a not in R:
                R.append(sigma_a)
    r = m.addVars(R, name='r', lb=0, ub=1);
    # v = m.addVars(I[1], name="v", lb=-GRB.INFINITY, ub=GRB.INFINITY)
    # r = m.addVars(...)

    # equation 1
    e1_p1 = e_list_based_on_g(g,r,empty_path)
    e1_p1 = gp.quicksum(e1_p1)
    e1_p2 = get_v(v,I,seq,empty_path,1)
    e1_p2 = gp.quicksum(e1_p2)
    m.setObjective(e1_p1 + e1_p2, GRB.MAXIMIZE)

    # equation 2
    for i in I[1]:
        v_i = v[i]
        for a in A[i]:
            sigma = seq[1][i]
            sigma_a = sigma + a
            e_p1 = e_list_based_on_g(g,r,sigma_a)
            e_p1 = gp.quicksum(e_p1)

            v_2 = get_v(v, I, seq, sigma_a, 1)
            if not v_2:
                m.addConstr(e_p1  >= v_i, name=f"Infoset {i} action {a}")
                continue
            e_p2 = gp.quicksum(v_2)
            m.addConstr(e_p1 + e_p2 >= v_i, name=f"Infoset {i} action {a}")

    # equation 3
    for i in I[0]:
        sigma = seq[0][i]
        r_i = r[sigma]
        e_p1 = []
        for a in A[i]:
            sigma_a = sigma + a
            e_p1.append(r[sigma_a])
        e_p1 = gp.quicksum(e_p1)
        m.addConstr(e_p1 == r_i, name=f"c2_{i}")
    # equation 4
    m.addConstr(r[empty_path] == 1, name=f"c2_{i}_{a}")

    m.optimize()
    # print(m.display())
    return m.ObjVal


def extract_param_utils(node, I, A, seq, g, p, travel,S):
    # add actions to infoset
    if hasattr(node, 'payoffs'):
        p = float(p)
        np = node.payoffs[0]
        node_payoff = p * node.payoffs[0]  # player one
        idx1 = ''.join(travel[0])
        idx2 = ''.join(travel[1])
        if idx1 in g:
            if idx2 not in g[idx1]:
                g[idx1][idx2] = node_payoff
            else:
                g[idx1][idx2] = node_payoff + g[idx1][idx2]
        else:
            g[idx1] = {idx2:  node_payoff }
        if travel[0] not in S:
            S.append(travel[0])

    elif hasattr(node, 'infoset'):
        infoset = node.infoset
        player = node.player
        if infoset not in I[player]:
            I[player][infoset] = p
        else:
            I[player][infoset] += p
        actions = [a + "_" +str(infoset) for a in node.action_names]
        A[infoset] = actions
        node_seq = travel[player]
        seq[player][infoset] = node_seq
        for idx, child_node in enumerate(node.children):
            action = actions[idx]
            next_travel = copy.deepcopy(travel)
            next_travel[player] = next_travel[player] + action
            I, A, seq, g, S = extract_param_utils(child_node, I, A, seq, g, p, next_travel, S)
    else:
        for idx, child_node in enumerate(node.children):
            I, A, seq, g, S = extract_param_utils(child_node, I, A, seq, g, p * node.action_probs[idx],
                                                  travel, S)
    return I, A, seq, g, S


def extract_parameters(efg):
    """Converts an extensive form game into the SQF parameters:
    I, Sigma, A, seq, g."""

    # Implement this second. It does not matter how you implement the
    # parameters -- functions, classes, or dictionaries, anything will work.
    I = {
        1: {},
        0: {}
    }
    seq = {
        1: {},
        0: {}
    }
    travel = {
        1: "/",
        0: "/",
    }
    A = {}
    g = {}
    S = []
    I, A, seq, g, S = extract_param_utils(efg, I, A, seq, g, 1, travel, S)
    return I, A, seq, g,S


def payoff(efg):
    """Computes the value of the extensive form game"""

    I, A, seq, g,S = extract_parameters(efg)
    with contextlib.redirect_stdout(sys.stderr):
        p = sqf(I, A, seq, g,S)

    return p

def game_value(efg):
    "You can use this to check the solution for small games."
    # Standard output on BRUTE must not contain debugging messages!

    equilibrium = pygambit.nash.enummixed_solve(efg, rational=False)
    print(equilibrium[0].payoff(efg.players[0]), file=sys.stderr)

def draw_game(root, filename="out.pdf"):
    "You can use this to visualize your games."
    # File IO not allowed on BRUTE! GraphViz is required to render graphs.

    dot = export_dot.nodes_to_dot(root)
    dot.write_pdf(filename)


if __name__ == "__main__":
    efg = sys.stdin.read()
    #with open('examples/example_from_slides.efg', 'r') as file:
    #efg = file.read()


    game = pygambit.Game.parse_game(efg)
    #game_value(game)
    root = import_efg.efg_to_nodes(game)
    a = type(root)


    print(payoff(root))
