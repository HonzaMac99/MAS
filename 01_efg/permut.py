import numpy as np


def generate_subsets(hidings, n_enemies):
    def backtrack(start, path):
        if len(path) == n_enemies:
            subsets.append(path.copy())
            return
        for i in range(start, len(hidings)):
            path.append(hidings[i])
            backtrack(i + 1, path)
            path.pop()

    subsets = []
    backtrack(0, [])
    return subsets

# Example usage:
hidings = [(1, 2), (3, 4), (5, 6), (7, 8)]
result = generate_subsets(hidings, 3)
print(result)