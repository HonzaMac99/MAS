from cgt_bandits.nodes import ChanceNode, TerminalNode, PersonalNode
from cgt_bandits import export_efg
from cgt_bandits import export_dot
from enum import IntEnum
import pygambit
import itertools
import numpy as np
import copy
import sys


class Player(IntEnum):
    PLAYER = 0
    BANDIT = 1


START_INFOSET = 1
infosets = [2, 2]  # [I_player, i_bandit]


def find_start(maze):
    "input: [np.array] maze"
    for i in range(maze.shape[0]):
        for j in range(maze.shape[1]):
            if maze[i, j] == 'S':
                return (i, j)


def find_hidings(maze):
    "input: [np.array] maze"
    hidings = []
    for i in range(maze.shape[0]):
        for j in range(maze.shape[1]):
            if maze[i, j] == 'E':
                hidings.append((i, j))
    return hidings


def get_enemy_opts(hidings, n_enemies):
    def backtrack(start, subset):
        if len(subset) == n_enemies:
            enemy_opts.append(subset.copy())
            return
        for i in range(start, len(hidings)):
            subset.append(hidings[i])
            backtrack(i + 1, subset)
            subset.pop()

    enemy_opts = []
    backtrack(0, [])
    return enemy_opts


def in_maze(maze, coord):
    inside = (0 <= coord[0] < maze.shape[0]) and (0 <= coord[1] <= maze.shape[1])
    is_wall = (maze[coord] == '#')
    return inside and not is_wall


def expand_player(maze, visited, enemy_pos, pos, capture_prob, points):
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    pl_actions = ["up", "down", "left", "right"]
    cur_actions = []

    childs = []
    for i in range(len(directions)):
        drct = directions[i]
        new_pos = (pos[0] + drct[0], pos[1] + drct[1])
        print(" ", in_maze(maze, new_pos) and not visited[new_pos], new_pos)
        if in_maze(maze, new_pos) and not visited[new_pos]:
            new_node = search_maze(maze, copy.deepcopy(visited), enemy_pos, new_pos, capture_prob, points)
            childs.append(copy.deepcopy(new_node))
            cur_actions.append(pl_actions[i])
    return childs, cur_actions


def search_maze(maze, visited, enemy_pos, pos, capture_prob, points):
    e_actions = ["take gold", "fight"]
    f_actions = ["lost fight", "won fight"]
    visited[pos] = 1
    cur_node = None

    if maze[pos] == 'D':
        print("Goal reached:")
        points += 2
        return TerminalNode("Exit: " + str(pos), [points])
    elif maze[pos] == 'G':
        points += 1
    # if maze[pos] == 'E' then we use the expand_player values for the win fight case

    print("Branching:", pos)
    print(visited)
    childs, cur_actions = expand_player(maze, copy.deepcopy(visited), enemy_pos, pos, capture_prob, points)

    if len(childs) == 0:
        print("Got stuck:")
        cur_node = TerminalNode("Got stuck: " + str(pos), [0])
    elif maze[pos] in ['S', '-', 'G'] or (maze[pos] == 'E' and pos not in enemy_pos):
        print("Returning from:", pos)
        cur_node = PersonalNode("Player: " + str(pos), infosets[0], Player.PLAYER, childs, cur_actions)
        infosets[0] += 1
    elif maze[pos] == 'E' and pos in enemy_pos:
        lost_fight = TerminalNode("Got captured: " + str(pos), [0])
        won_fight = PersonalNode("Player: " + str(pos), infosets[0], Player.PLAYER, childs, cur_actions)
        infosets[0] += 1

        fight_childs = [lost_fight, won_fight]
        probs = [capture_prob, 1 - capture_prob]
        fight_choice = ChanceNode("Fight: " + str(pos), fight_childs, f_actions, probs)
        childs2, cur_actions2 = expand_player(maze, copy.deepcopy(visited), enemy_pos, pos, capture_prob, 0)
        take_gold_choice = PersonalNode("Player: " + str(pos), infosets[0], Player.PLAYER, childs2, cur_actions2)
        infosets[0] += 1

        enemy_childs = [take_gold_choice, fight_choice]
        cur_node = PersonalNode("Bandit: " + str(pos), infosets[1], Player.BANDIT, enemy_childs, e_actions)
        infosets[1] += 1

    return cur_node



def define_game(maze, bandit_cnt, prob):
    # Describe the tree of possible playthroughs of the game based on the 
    # maze and the rules. Start with each node in a unique infoset, then 
    # consider the information the players have at each point and encode it.

    visited = np.zeros(maze.shape)
    start_pose = find_start(maze)
    hiding_poses = find_hidings(maze)
    enemy_opts = get_enemy_opts(hiding_poses, bandit_cnt)
    points = 0

    print("The maze:")
    print(maze)
    print("Start: ", start_pose)
    print("Hidings: ", hiding_poses)

    visited[start_pose] = 1

    if len(hiding_poses) == 0:
        root_node = search_maze(maze, copy.deepcopy(visited), [], start_pose, prob, points)
    else:
        actions = []
        childs = []
        for enemy_poses in enemy_opts:
            new_node = search_maze(maze, copy.deepcopy(visited), enemy_poses, start_pose, prob, points)
            childs.append(copy.deepcopy(new_node))
            actions.append(str(enemy_poses))
        root_node = PersonalNode("Bandit choice", infosets[1], Player.BANDIT, childs, actions)

    return root_node


def read_maze(io):
    bandit_cnt, prob = np.fromstring(io.readline(), sep=' ')
    lines = [list(lin) for lin in map(str.strip, io.readlines()) if lin]
    maze = np.array(lines)
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
    maze, bandit_cnt, prob = read_maze(sys.stdin)
    root = define_game(maze, bandit_cnt, prob)    

    # NOT on BRUTE!
    draw_game(root)

    # efg = export_efg.nodes_to_efg(root)

    # NOT on BRUTE!
    # game_value(efg)

    # Print the efg representation.
    # Yes on BRUTE
    # print(repr(efg))
