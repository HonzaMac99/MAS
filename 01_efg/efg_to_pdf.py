import pygambit as pg
from cgt_bandits import import_efg
from cgt_bandits import export_dot
import sys

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("usage: python {sys.argv[0]} in.efg out.pdf", file=sys.stderr)
        exit(1)

    graph = pg.Game.read_game(sys.argv[1])

    root = import_efg.efg_to_nodes(graph)
    dot = export_dot.nodes_to_dot(root)

    dot.write_pdf(sys.argv[2])
