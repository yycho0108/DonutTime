#!/usr/bin/env python3

# from ortools.sat.python import cp_model
# from ortools.constraint_solver import ;q

import itertools
import numpy as np
import cvxpy as cp


def minimize_overlap(G, T: int):
    """Just count duplicates."""
    H = [Gi @ Gi.T for Gi in G]
    dup = []
    for dt in range(1, T):
        weight = 2.0 ** (T - dt - 1)
        for i in range(dt, T):
            dup.append(weight * sum(sum(cp.multiply(H[i], H[i - dt]))))
    return sum(dup)


def main():
    # sparsity?
    T: int = 3
    N: int = 17
    K: int = 3

    A = np.random.choice(2, size=N)
    G = [cp.Variable(shape=(N, K), name=F'G{i}',
                     integer=True) for i in range(T)]
    objective = cp.Minimize(minimize_overlap(G, T))

    constraints = []

    def _rule_one_group(t, i):
        if t == 0:
            # [This week]
            # Everyone who has attended must be assigned to one group.
            if A[i]:
                return sum(G[t][i, j] for j in range(K)) == 1
            else:
                return sum(G[t][i, j] for j in range(K)) == 0
        # In future weeks, we assume everyone attended.
        # each person may be assigned to only one group.
        return sum(G[t][i, j] for j in range(K)) == 1
    constraints.extend(
        [_rule_one_group(t, i) for (t, i) in itertools.product(
            range(T), range(N))])

    # The size of each group should be roughly equal.
    # This is achieved by enforcing |g_i| <= |G|//K.
    def _rule_group_size(t, j):
        if t == 0:
            Nt = A.sum()
        else:
            Nt = N
        vmax: int = (Nt + K - 1) // K
        return sum(G[t][i, j] for i in range(N)) <= vmax
    constraints.extend(
        [_rule_group_size(t, j) for (t, j) in itertools.product(
            range(T), range(K))])

    prob = cp.Problem(objective, constraints)
    result = prob.solve()
    print(result)

    #for t in range(T):
    #    print(G[t].value)


if __name__ == '__main__':
    main()
