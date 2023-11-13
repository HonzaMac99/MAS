from cgt_bandits.nodes import ChanceNode, TerminalNode, PersonalNode
from cgt_bandits import export_efg
from cgt_bandits import export_dot
from enum import IntEnum
import pygambit
import itertools
import numpy
import sys


class Player(IntEnum):
    PLAYER = 0
    BANDIT = 1


def define_game(maze, bandit_cnt, prob):
    # Describe the tree of possible playthroughs of the game based on the 
    # maze and the rules. Start with each node in a unique infoset, then 
    # consider the information the players have at each point and encode it.
    # node1 = PersonalNode("Player1 node", ["up", "down", "left", "right"])
    # node2 = PersonalNode("Player2 node", ["up", "down", "left", "right"])
    end1 = TerminalNode("p1 wins", [3])
    end2 = TerminalNode("p2 wins", [-3])
    chance1 = ChanceNode("chance", [end1, end2], ["left", "right"], [0.8, 0.2])
    chance2 = ChanceNode("chance", [end2, end1], ["left", "right"], [0.8, 0.2])
    pl1 = PersonalNode("p1 choice", 1, 1, [chance1, chance2], ["option a", "option b"])

    return pl1


def read_maze(io):
    bandit_cnt, prob = numpy.fromstring(io.readline(), sep=' ')
    lines = [list(lin) for lin in map(str.strip, io.readlines()) if lin]
    maze = numpy.array(lines)
    print("Maze interpretation:")
    print(maze)

    return maze, int(bandit_cnt), prob


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


if __name__ == '__main__':
    print("hello")
    maze, bandit_cnt, prob = read_maze(sys.stdin)
    root = define_game(maze, bandit_cnt, prob)    

    # NOT on BRUTE!
    draw_game(root)

    efg = export_efg.nodes_to_efg(root)

    # NOT on BRUTE!
    game_value(efg)

    # Print the efg representation.
    # Yes on BRUTE
    print(repr(efg)) 
