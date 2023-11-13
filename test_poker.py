from cgt_bandits.nodes import ChanceNode, PersonalNode, TerminalNode
from cgt_bandits import export_efg


# Cards are separated from the rest of history simply for convenience.

# In fact, this game is so simple we do not even need to keep track of
# the actions played. We show how to do it only because it might be
# useful for larger games.


def build_terminal(cards, history=[]):
    # If we are here, a bet must have been followed by call.
    if cards in ['jj', 'qq']:
        return TerminalNode('tie', [0, 0])
    elif cards == 'qj':
        return TerminalNode('p1 wins', [3, -3])
    elif cards == 'jq':
        return TerminalNode('p2 wins', [-3, 3])


def build_second_player(cards, history=[]):
    # The second player also knows the first players action,
    # however if it is his turn, it must have been a bet.

    # Leaving p1's action out changes nothing.
    infoset = ord(cards[1])  # + hash(history[0])
    actions = ['fold2', 'call']
    children = [
        TerminalNode('p2 folded', [1, -1]),
        build_terminal(cards, history + [actions[1]])
    ]

    return PersonalNode("", infoset, 1, children, actions)


def build_first_player(cards, history=[]):
    # The only information player 1 knows is the card he is holding.
    infoset = ord(cards[0])
    actions = ['fold1', 'bet']
    children = [
        TerminalNode('p1 folded', [-1, 1]),
        build_second_player(cards, history + [actions[1]])
    ]

    return PersonalNode("", infoset, 0, children, actions)


def build_deal():
    deals = ['jj', 'jq', 'qj', 'qq']
    probs = [1/6, 1/3, 1/3, 1/6]
    children = map(build_first_player, deals)

    return ChanceNode("Deal", children, deals, probs)


def build_game():
    return build_deal()


if __name__ == '__main__':
    efg = export_efg.nodes_to_efg(build_game())
    print(repr(efg))